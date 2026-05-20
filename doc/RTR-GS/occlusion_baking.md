# RTR-GS 遮挡体烘焙（Occlusion Baking）说明文档

## 1. 背景与目的

在逆渲染（Inverse Rendering）任务中，从多视角图像分解几何、材质和光照是一个欠约束问题。RTR-GS 在 **Stage 2（PBR 材质-光照分解阶段）** 需要区分直接光照和间接光照，而遮挡（Occlusion/Visibility）是正确分离这两者的关键。

**遮挡体烘焙的核心目的**：防止阴影、光照和反照率（albedo）分解中产生混叠伪影（aliasing artifacts）。

> 论文原文（Section 3.4）：
> *"To prevent aliasing artifacts in shadows, lighting, and albedo, we leverage the recovered geometric structure to bake occlusion information into a voxel grid, following the approach in GS-IR [32]."*

### 1.1 问题：为什么需要遮挡体？

在 PBR 渲染方程中，漫反射分量为：

$$L_d(\mathbf{x}) \approx \frac{c}{\pi} \big[ V(\mathbf{x}) \cdot L_d^{dir}(\mathbf{x}) + (1 - V(\mathbf{x})) \cdot L_d^{ind}(\mathbf{x}) \big]$$

其中：
- $V(\mathbf{x})$：可见性（visibility），值为 1 表示直接可见光源，0 表示完全被遮挡
- $L_d^{dir}(\mathbf{x})$：直接环境光照（仅依赖法线方向 $\mathbf{n}$）
- $L_d^{ind}(\mathbf{x})$：间接光照（通过每个高斯点上的参数 $L_{ind}$ 混合获得）

**如果没有遮挡体**，所有表面点都会被当作完全可见，导致：
1. 阴影区域的颜色被错误解释为材质（albedo）偏暗
2. 光照分解时无法区分"暗是因为材质黑"还是"暗是因为有阴影"
3. 间接光照建模失效

### 1.2 遮挡体的整体思路

核心思想：**在 3D 空间中预计算每个位置的可见性分布**，以法线方向为查询键，在渲染时快速查找该位置沿该法线方向的遮挡程度。

具体流程：
1. **烘焙阶段**（Stage 1 完成后）：在每个 3D 体素位置渲染 6 面 cubemap，判断每个方向是否被几何体遮挡，将结果投影到球谐（SH）系数并存于体素网格
2. **渲染阶段**（Stage 2）：对每个像素的 3D 表面点，从体素网格中三线性插值获取 SH 系数，通过重要性采样重建该法线方向的遮挡值

---

## 2. 烘焙流程详解（`baking.py`）

### 2.1 整体流程概览

```
Stage 1 训练完成
     ↓
加载高斯模型 checkpoint
     ↓
创建 3D 体素网格 (occlu_res × occlu_res × occlu_res)
     ↓
标记有效体素（被高斯覆盖的区域）
     ↓
对每个有效体素：
  ① 筛选附近高斯
  ② 渲染 6 面 cubemap（FOV=90, 白色背景）
  ③ 转换为 equirectangular 环境贴图
  ④ 生成遮挡掩码 → 球面积分投影到 SH 系数
     ↓
膨胀（Dilate）：将有效体素的 ID 传播到所有体素
     ↓
保存 occlusion_volumes.pth
```

### 2.2 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bound` | 1.5 | 遮挡体空间范围的半边长，AABB 为 `[-bound, bound]^3` |
| `--valid` | 1.5 | 烘焙每个体素时，筛选附近高斯的裁剪半径 |
| `--occlu_res` | 160 | 体素网格分辨率（每维 160 个体素） |
| `--cubemap_res` | 256 | cubemap 每面的分辨率 |
| `--occlusion` | 0.4 | 遮挡判定阈值（越小，环境光遮蔽越轻） |
| `--checkpoint` | None | Stage 1 的 checkpoint 路径 |

### 2.3 核心步骤详解

#### 步骤 1：创建体素网格与有效掩码

```python
# [baking.py:L210-L254]
grid = (aabb_max - aabb_min) / (args.occlu_res - 1)
positions = torch.tensor(prods).cuda() * grid + aabb_min  # [bs, 3]
```

- 在世界空间 `[-bound, bound]^3` 内均匀采样体素中心位置
- 将每个高斯点量化到体素网格，标记其所在的 8 个角点为 `valid`
- 为有效体素分配连续的 `occlusion_ids`（从 0 开始），无效体素保持 `-1`

#### 步骤 2：逐体素渲染与 SH 投影

对每个有效体素（`grid_id`），执行以下操作：

**2a. 筛选局部高斯**

```python
# [baking.py:L293-L300]
diff = means3D - position
valid = (diff.abs() < args.valid).all(dim=1)
```

仅选取该体素周围半径为 `valid`（默认 1.5）范围内的高斯点参与渲染，大幅加速烘焙过程。

**2b. 6 面 Cubemap 渲染**

```python
# [baking.py:L301-L341]
for r_idx, rotation in enumerate(rotations):
    # 6 个方向：+X, -X, +Y, -Y, +Z, -Z
    # 每个面 FOV=90°，分辨率 cubemap_res × cubemap_res
    # 白色背景（bg_color = torch.ones）
```

从该体素位置出发，使用 `_C.lite_rasterize_gaussians` 渲染 6 个方向的 RGB 图和深度图。背景设为白色（表示无遮挡），高斯覆盖区域为黑色或深色。

**2c. Cubemap → Equirectangular 转换**

```python
# [baking.py:L344-L362]
depth_envmap = dr.texture(
    torch.stack(depth_cubemap)[None, ...],
    envmap_dirs[None, ...].contiguous(),
    filter_mode="nearest", boundary_mode="cube",
)[0]
```

使用 `nvdiffrast` 的 cubemap 纹理查找功能，将 6 面 cubemap 转换为 equirectangular 格式的环境贴图。

**2d. 遮挡掩码与 SH 投影**

```python
# [baking.py:L370-L374]
occlu_mask = torch.where(rgb_envmap > 0.5, 1.0, 0.0)  # RGB > 0.5 视为可见
weighted_color = occlu_mask * solid_angles             # 球面积分权重
temp_coefficients = (weighted_color * components).sum(0).sum(0)  # SH投影
occlusion_coefficients[grid_id] = temp_coefficients[:, None]
```

核心原理：
- `rgb_envmap > 0.5`：在白色背景上，被几何体遮挡的方向 RGB 值接近 0，未被遮挡的方向接近 1
- `occlu_mask * solid_angles`：对可见区域进行球面加权
- 与预计算的 SH 基函数 `components` 逐元素相乘并求和，得到 9 个 SH 系数（degree=3 → d²=9）

#### 步骤 3：膨胀（Dilation）传播

```python
# [baking.py:L376-L378]
while (occlusion_ids == -1).sum() > 0:
    gs_ir_ext.dialate_occlusion_ids(occlusion_ids)
```

核心 CUDA kernel 逻辑（[occlusion_kernel.cu:L247-L294](file://d:/localSpace/relighting/RTR-GS/submodules/gs-ir/src/occlusion_kernel.cu#L247-L294)）：

- 对每个 `occlusion_ids == -1` 的体素
- 从 6 个面邻居（±X, ±Y, ±Z）中查找有效 ID
- 复制最近的有效 ID
- 循环迭代直到所有 `-1` 被填充

这样做的原因：只有被高斯覆盖的体素才进行了实际烘焙，远处的空体素通过膨胀继承最近有效体素的遮挡系数。

#### 步骤 4：保存

```python
# [baking.py:L380-L390]
torch.save({
    "occlusion_ids": occlusion_ids,           # [R, R, R] int32
    "occlusion_coefficients": occlusion_coefficients,  # [num_grid, d², 1] float32
    "bound": args.bound,                      # float
    "degree": occlu_sh_degree,                # int (3)
    "occlusion_threshold": occlusion_threshold,  # float (0.4)
}, save_file)
```

保存为 `occlusion_volumes.pth`，默认路径为 Stage 1 checkpoint 同目录。

---

## 3. 遮挡体数据结构

| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `occlusion_ids` | `[R, R, R]` | int32 | 体素网格索引表。值为 `-1` 时无效，`≥0` 时为 `occlusion_coefficients` 的行索引 |
| `occlusion_coefficients` | `[num_valid, d², 1]` | float32 | 有效体素的球谐系数，`d² = degree²`（默认 9） |
| `bound` | scalar | float | 遮挡体的空间半边长 |
| `degree` | scalar | int | 球谐度数（默认 3） |
| `occlusion_threshold` | scalar | float | 烘焙时的遮挡判定阈值 |

---

## 4. 渲染时的使用流程

### 4.1 加载与传递

在 Stage 2 训练或评估脚本中加载：

```python
# [train.py:L86-L92]
if is_pbr:
    if args.occlusion_path is not None:
        occlusion_volumes = torch.load(args.occlusion_path)
        bound = occlusion_volumes["bound"]
        aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
        pbr_kwargs["occlusion_volumes"] = occlusion_volumes
        pbr_kwargs["aabb"] = aabb
```

**重要**：遮挡体仅在 PBR 分支（`render_ref_pbr`）中使用，混合渲染分支（`render_ref`）不需要。

### 4.2 渲染时查询

在每个视点的渲染中（[render.py:L334-L366](file://d:/localSpace/relighting/RTR-GS/gaussian_renderer/render.py#L334-L366)）：

```python
# 1. 从深度图计算每个像素的 3D 世界坐标
points = (-view_dirs.reshape(-1, 3) * rendered_depth.reshape(-1, 1) + c2w[:3, 3])

# 2. 调用遮挡重建函数
occlusion_map = recon_occlusion(
    H=H, W=W,
    bound=occlusion_volumes["bound"],
    points=points,                    # [HW, 3]
    normals=normal_map.reshape(-1, 3),  # [HW, 3]
    roughness=roughness_map.reshape(-1, 1),
    occlusion_coefficients=occlusion_volumes["occlusion_coefficients"],
    occlusion_ids=occlusion_volumes["occlusion_ids"],
    aabb=aabb,
    degree=occlusion_volumes["degree"],
).reshape(H, W, 1)
```

### 4.3 `recon_occlusion` 内部流程

定义在 [gs_ir/\_\_init\_\_.py:L6-L42](file:///d:/localSpace/relighting/RTR-GS/submodules/gs-ir/gs_ir/__init__.py#L6-L42)：

```python
@torch.no_grad()
def recon_occlusion(H, W, bound, points, normals, roughness,
                    occlusion_coefficients, occlusion_ids, aabb, sample_rays=256, degree=4):
    occlu_res = occlusion_ids.shape[0]
    half_grid = bound / float(occlu_res)
    shift_points = points + normals * half_grid  # 沿法线偏移避免自遮挡

    # 步骤 1：三线性插值获取 SH 系数
    (coefficients, coeff_ids) = _C.sparse_interpolate_coefficients(
        occlusion_coefficients, occlusion_ids, aabb, shift_points, normals, degree,
    )
    coefficients = coefficients.permute(0, 2, 1)  # [HW, 1, d²]

    # 步骤 2：GGX 重要性采样 + SH 重建
    roughness = torch.ones([H * W, 1]).cuda()
    occlusion = _C.SH_reconstruction(
        coefficients, normals, roughness, sample_rays, degree
    )  # [HW, 1]
    return occlusion
```

#### 步骤 1：`sparse_interpolate_coefficients`（CUDA Kernel）

实现在 [occlusion_kernel.cu:L22-L144](file:///d:/localSpace/relighting/RTR-GS/submodules/gs-ir/src/occlusion_kernel.cu#L22-L144)：

1. **量化坐标**：将 3D 世界坐标映射到体素网格索引
2. **8 角点法线感知掩码**：对 8 个相邻体素角点，仅当方向点积 `dot(dir, normal) > 0`（即在表面上方）时计入贡献
3. **三线性插值权重**：基于小数部分计算权重
4. **加权融合 SH 系数**：从 `occlusion_ids` 查找 8 个角点对应的 SH 系数，加权求和

#### 步骤 2：`SH_reconstruction`（CUDA Kernel）

实现在 [occlusion_kernel.cu:L146-L245](file:///d:/localSpace/relighting/RTR-GS/submodules/gs-ir/src/occlusion_kernel.cu#L146-L245)：

1. 以法线为中心方向，`roughness=1`（固定），使用 GGX 重要性采样生成 256 个方向
2. 对每个采样方向，用 SH 系数重建可见性值
3. 累加取平均得到最终的遮挡值（范围 `[0, 1]`）

### 4.4 PBR 着色应用

在 [pbr/shade.py:L255-L308](file:///d:/localSpace/relighting/RTR-GS/pbr/shade.py#L255-L308) 中：

```python
def pbr_shading(light, normals, view_dirs, albedo, roughness,
                occlusion=None, irradiance=None, metallic=None, brdf_lut=None):
    # 从环境光 cubemap 采样直接漫反射光
    diffuse_light = dr.texture(light.diffuse[None, ...],
                               sampling_normals.contiguous(),
                               filter_mode="linear", boundary_mode="cube")

    # 核心公式：遮挡区域用间接光照替代
    diffuse_light = diffuse_light * occlusion
    diffuse_light = diffuse_light + (1.0 - occlusion) * irradiance

    results["incidents_light"] = ((1.0 - occlusion) * irradiance).squeeze(0)
    ...
```

**物理含义**：
- `occlusion → 1`（完全可见）：使用直接环境光照 `diffuse_light`
- `occlusion → 0`（完全遮挡）：使用间接光照 `irradiance`（来自每个高斯的 $L_{ind}$ 参数 splatting）
- 中间值平滑混合两种光照

---

## 5. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Stage 1 完成后                                  │
│                                                                   │
│  baking.py                                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ 1. 创建 R³ 体素网格 (occlu_res=160, bound=1.5)             │    │
│  │ 2. 标记有效体素 (被高斯覆盖的 8 角)                          │    │
│  │ 3. 每体素: 6面渲染 → equirect → SH投影 (degree=3, d²=9)    │    │
│  │ 4. 膨胀: -1 体素 ← 继承邻居 ID                               │    │
│  │ 5. 保存: occlusion_volumes.pth                              │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│                    occlusion_volumes.pth                           │
│                    { ids, coefficients,                            │
│                      bound, degree, threshold }                    │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Stage 2 / 评估                                  │
│                                                                   │
│  train.py / render_and_eval.py                                    │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ torch.load() → pbr_kwargs["occlusion_volumes"]             │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  gaussian_renderer/render.py (或 render_fast.py)                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ 每帧渲染:                                                   │    │
│  │   depth_map → 3D points → recon_occlusion():               │    │
│  │   ┌─────────────────────────────────────────────────┐     │    │
│  │   │ a) shift = points + normal * half_grid          │     │    │
│  │   │ b) sparse_interpolate: 8角三线性+法线掩码       │     │    │
│  │   │    → [HW, 1, 9] SH coefficients                 │     │    │
│  │   │ c) SH_reconstruction: GGX采样(256 rays) + SH重建│     │    │
│  │   │    → [HW, 1] occlusion map                      │     │    │
│  │   └─────────────────────────────────────────────────┘     │    │
│  └──────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  pbr/shade.py → pbr_shading()                                     │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ diffuse = occlusion × direct_diffuse                      │    │
│  │         + (1-occlusion) × indirect_irradiance             │    │
│  │ → 最终 PBR 渲染结果                                       │    │
│  └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 关键设计决策

### 6.1 为什么在 Stage 1 和 Stage 2 之间烘焙？

- Stage 1 完成后，几何结构已基本重建完毕，遮挡体的基础（3D 高斯位置）已相对准确
- Stage 2 需要 PBR 分解，遮挡体是不可或缺的先验条件
- Stage 2 中几何仍会微调（双分支协同），但过大的几何突变会使遮挡体失效，因此论文消融实验验证了冻结几何或单 PBR 分支会导致质量下降

### 6.2 为什么用 SH 存储遮挡？

- **紧凑性**：每个体素仅需 9 个浮点数（degree=3），体素网格 160³ = 4,096,000 个有效体素，总计约 140MB
- **平滑性**：SH 表示自动提供角度域的平滑重建，适合表示低频可见性变化
- **高效查询**：渲染时通过矩阵乘法即可快速重建

### 6.3 沿法线偏移 `half_grid` 的作用

```python
shift_points = points + normals * half_grid
```

避免采样点恰好在表面（体素边界）上导致不稳定。沿法线偏移半个网格单元确保采样在"表面上方"进行。

### 6.4 法线感知的 8 角掩码

在 `sparse_interpolate_coefficients` 中，仅当体素角点方向与表面法线夹角 < 90° 时才参与插值。这避免了使用"表面下方"体素的遮挡信息，防止错误的遮挡估计。

### 6.5 膨胀（Dilation）的必要性

直接烘焙只在被高斯覆盖的体素处有值。但渲染时像素的 3D 表面点可能落在没有直接烘焙的体素上。通过膨胀，所有体素都获得最近有效体素的 SH 系数，保证查询不会失败。

---

## 7. 运行命令

```bash
# Stage 1 训练完成后，运行遮挡体烘焙
python baking.py \
    --checkpoint <output_path>/stage1/checkpoint/chkpnt30000.pth \
    --bound 1.5 \
    --occlu_res 128
```

输出文件：`<output_path>/stage1/checkpoint/occlusion_volumes.pth`

---

## 8. 调试建议

如果遮挡体效果不理想，可以从以下几个方面排查：

1. **遮挡值全为 1（无阴影）**：渲染时会打印警告。可能原因：
   - Stage 1 几何重建不充分，高斯未能覆盖表面
   - `--occlusion` 阈值设置过低
   - `--bound` 设置不合理

2. **遮挡值全为 0（全黑）**：可能原因：
   - 几何膨胀过度（floater 太多）
   - `--valid` 裁剪半径过大

3. **遮挡体不匹配**：如果 Stage 2 中几何发生较大变化，烘焙的遮挡体会失效。确保双分支训练稳定。
