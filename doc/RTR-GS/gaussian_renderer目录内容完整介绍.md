   
## gaussian_renderer/ 目录文件列表

以下是 `/home/huangpengyue/projects/RTR-GS/gaussian_renderer/` 目录中的文件：

| 文件 | 说明 |
|------|------|
| `__init__.py` | 渲染函数分发表，将不同渲染类型映射到对应的 render 函数 |
| `render.py` | **主渲染管线**（混合渲染模式）：PRT + 反射 + PBR，包含完整的训练损失计算 |
| `render_fast.py` | **快速渲染管线**：仅 PBR 渲染，无反射分支，无训练损失 |
| `render_equirect.py` | **ODGS 全景渲染**：使用 ODGS 光栅化器处理等距柱状投影图像 |
| `rtr_gs_rasterization.py` | RTR-GS 自定义 CUDA 光栅化器（基于 3DGS 的 `torch.autograd.Function`） |

---

## 文件完整内容

### 1. `__init__.py` — 渲染函数分发表

[__init__.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/__init__.py)

```python
from gaussian_renderer.render import render
from gaussian_renderer.render_fast import render_fast
from gaussian_renderer.render_equirect import render as render_equirect

render_fn_dict = {
    "render_ref": render,
    "render_ref_pbr": render,
    "render_ref_fast": render_fast,
    "neilf_ref": render,
    "neilf_ref_pbr": render,
    "neilf_ref_fast": render_fast,
    "render_ref_equirect": render_equirect,
    "render_ref_pbr_equirect": render_equirect,
}
```

**说明**：`render_fn_dict` 将命令行参数 `-t`（渲染类型）映射到具体的渲染函数。`render_ref` 和 `render_ref_pbr` 共享同一个主渲染器 `render`（通过 `pc.use_pbr` 区分是否启用 PBR），`render_ref_fast` 和 `neilf_ref_fast` 使用快速渲染器。

---

### 2. `render.py` — 主渲染管线（混合渲染 + PBR + 训练损失）

[render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py) （共652行）

**核心架构**（两个主要函数）：

#### 2.1 `render_view()` — 前向渲染函数（行 18-507）

这是混合渲染模式的核心实现，采用**延迟着色**（deferred shading）策略：

1. **环境贴图设置**（行 91-104）：根据训练/评估模式设置 `refmap`（反射贴图）和可选的 `cubemap`（PBR 环境贴图），构建 mipmap。

2. **光栅化器初始化**（行 115-136）：使用 `GaussianRasterizationSettings` 设置相机内参、视角、投影矩阵等，创建 `GaussianRasterizer`。

3. **高斯属性提取**（行 140-154）：
   - `means3D`：3D 位置
   - `opacity`：不透明度
   - `ref_tint`：反射色调（RGB）
   - `ref_roughness`：反射粗糙度
   - `ref_strength`：反射强度
   - `normal`：法线（高斯最短轴，朝向相机）
   - `depths` / `depths2`：深度及其平方

4. **PRT 颜色计算**（行 157-165）：
   - 如果 `diffuse_iteration` 未达到，使用 `PRTutils.cal_diffuse()`（仅漫反射）
   - 否则使用 `PRTutils.cal_color()`（全 PRT，含 MLP 解码）
   - 结果作为 `override_color`

5. **特征向量构建**（行 203-249）：
   - 基础特征：`[depths, depths2, normal, ref_tint, ref_roughness, ref_strength]`（共 10 维）
   - 如果启用 PBR：拼接 `[base_color, roughness, metallic, incidents_light]`（再加 8 维，共 18 维）
   - 光照处理：正常模式使用高斯属性 `incidents`（SH 系数）计算；重光照模式下可通过 `transfer_light` 与 `cubemap.shs` 相乘进行光照传输

6. **前向着色**（`pipe.forward_shading`，行 207-211）：
   - 如果开启，使用 `get_reflectance_color_forward()` 在光栅化前计算反射颜色
   - 最终颜色 = `(1 - ref_strength) * radiance + ref_strength * reflection`

7. **CUDA 光栅化**（行 253-264）：调用 `rasterizer()` 一次性输出所有属性和特征。

8. **特征分解**（行 267-284）：
   - 从光栅化结果中按通道拆分出 `depth`, `depth2`, `normal`, `ref_tint`, `ref_roughness`, `ref_strength`, `base_color`, `roughness`, `metallic`, `incident_lights`

9. **法线规范化**：`normal_map = F.normalize(normal_map, dim=-1)`（行 298）

10. **反射着色（非前向模式）**（行 307-310）：
    - 使用 `get_reflectance_color()` 计算延迟反射颜色（split-sum 近似）
    - 最终混合：`(1 - ref_strength) * radiance + ref_strength * reflection`
    - 背景合成：`ref_rgb * opacity + (1 - opacity) * bg`

11. **PBR 着色**（行 326-404）：
    - 计算 3D 表面点位置（通过深度反投影）
    - 可选的遮挡映射：`recon_occlusion()` 读取遮挡体积
    - `pbr_shading()`：包含漫反射（Lambertian）和镜面反射（GGX）
    - 输出：`diffuse_pbr`, `specular_pbr`, `incidents_light`

12. **可视化字典（非训练模式）**（行 411-458）：
    - 收集所有中间结果用于保存/显示
    - 对 `base_color` 应用 gamma 校正
    - 导出环境贴图基础版本

13. **结果组装**（行 464-507）：整合渲染图像、深度、法线等，可选附加 `pbr_env` 和 `env_only`

#### 2.2 `calculate_loss()` — 损失函数计算（行 511-633）

综合损失函数，包含多个组成部分：

| 损失项 | 参数 | 说明 |
|--------|------|------|
| L1 + SSIM | `lambda_rgb`, `lambda_dssim` | RGB 图像重建损失 |
| Depth | `lambda_depth` | 深度图 L1 损失 |
| Mask entropy | `lambda_mask_entropy` | 不透明度正则化 |
| Normal render-depth | `lambda_normal_render_depth` | 渲染法线与伪法线（从深度梯度导出的）MSE |
| Normal smooth | `lambda_normal_smooth` | 法线全变差（TV）平滑 |
| Ref roughness smooth | `lambda_ref_roughness_smooth` | 反射粗糙度边缘感知平滑 |
| Ref strength smooth | `lambda_ref_strength_smooth` | 反射强度边缘感知平滑 |
| PBR L1 + SSIM | `lambda_pbr` | PBR 分支重建损失 |
| Base color smooth | `lambda_base_color_smooth` | 基础颜色边缘感知平滑 |
| Roughness smooth | `lambda_roughness_smooth` | 粗糙度边缘感知平滑 |
| Metallic smooth | `lambda_metallic_smooth` | 金属度边缘感知平滑 |
| Env smooth | `lambda_env_smooth` | 环境贴图 TV 平滑 |
| White light | `lambda_white_light` | 白光照假设（各通道一致性） |
| Reflect = metallic | `lambda_reflect_strength_equal_metallic` | 反射强度接近金属度的先验 |

#### 2.3 `render()` — 入口函数（行 636-652）

```python
def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, 
           override_color=None, opt=False, is_training=False, dict_params=None):
    results = render_view(...)
    if is_training:
        loss, tb_dict = calculate_loss(...)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    return results
```

训练模式下附加损失计算，输出 `tb_dict`（用于 TensorBoard 日志）和 `loss`（用于反向传播）。

---

### 3. `render_fast.py` — 快速渲染管线（仅 PBR）

[render_fast.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render_fast.py) （共280行）

**与 `render.py` 的关键差异**：

1. **无反射分支**：不处理 `refmap`、`ref_tint`、`ref_roughness`、`ref_strength`，也没有 `forward_shading` 逻辑
2. **输入特征更简单**：`[depths, depths2, normal]`（3 维基础 + 可选的 PBR 属性）
3. **输出只包含 PBR 结果**：`render` 键直接就是 PBR 渲染图像
4. **无训练损失**：`render_fast()` 调用 `render_view()` 后直接返回，不计算损失
5. **光照处理**：PBR 模式下正常使用 `incidents` 属性，重光照模式下用 `transfer_light` 或置零

**关键代码结构**（行 15-264）：
- 相机内参、光栅化器初始化与 `render.py` 相同
- 特征向量：`[depths, depths2, normal]`（3 维）或拼接 `[base_color, roughness, metallic, incidents_light]`
- 调用 CUDA 光栅化器
- 特征分解后直接进行 PBR 着色
- 法线来自 `rendered_normal`，归一化后使用
- 不涉及反射颜色混合

---

### 4. `render_equirect.py` — ODGS 全景渲染

[render_equirect.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render_equirect.py) （共442行）

**核心特点**：

1. **使用 ODGS 光栅化器**：`from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer`，而非 RTR-GS 自己的 CUDA 光栅化器
2. **等距柱状投影坐标计算**（行 18-29）：`_compute_equirect_ray_dirs()` 根据像素位置计算经度/纬度方向向量
3. **多 Pass 渲染策略**：由于 ODGS 光栅化器不直接支持多属性输出，需要多次光栅化调用：
   - Pass 1：主颜色（PRT + 反射前向）
   - Pass 2：法线图（编码为 `n*0.5+0.5`）
   - Pass 3：反射属性（强度 + 粗糙度）
   - Pass 4/5：PBR 属性（base_color, roughness, metallic, depth）
4. **Alpha 归一化**：每个 pass 都需要 `rendered / opacity * alpha_mask`
5. **PBR 着色**：使用全景特定的 `view_dirs`（通过 `ray_dirs @ c2w[:3,:3].T` 计算）
6. **训练损失**：`calculate_loss()` 支持可选的 WS-SSIM（加权结构相似性，`use_ws_ssim` 开关）

---

### 5. `rtr_gs_rasterization.py` — RTR-GS CUDA 光栅化器

[rtr_gs_rasterization.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/rtr_gs_rasterization.py) （共243行）

这是一个封装了自定义 CUDA/C++ 光栅化内核的 PyTorch `autograd.Function`：

1. **`GaussianRasterizationSettings`**（行 169-185）：定义光栅化参数（NamedTuple），包含标准参数加上：
   - `backward_geometry`：是否计算几何梯度
   - `computer_pseudo_normal`：是否计算伪法线
2. **`GaussianRasterizer`**（行 188-243）：`nn.Module` 封装，提供 `forward()` 方法和 `markVisible()`
3. **`_RasterizeGaussians`**（行 39-166）：`torch.autograd.Function`，包含：
   - `forward()`：调用 `_C.rasterize_gaussians`，支持 `features` 输入（相较于标准 3DGS 新增）
   - `backward()`：调用 `_C.rasterize_gaussians_backward`
4. **输出**：`(num_rendered, num_contrib, color, opacity, depth, feature, normal, surface_xyz, weights, radii)`
   - `feature` 是 RTR-GS 特有输出，承载了后续延迟着色所需的所有属性
   - `normal` 和 `surface_xyz` 也是相对于原始 3DGS 的新输出

---

### 6. `render_and_eval.py` — 渲染与评估入口

[render_and_eval.py](file:///home/huangpengyue/projects/RTR-GS/render_and_eval.py) （共664行）

这是评估/推理阶段的入口脚本，主要功能：

1. **`evaling()`**（行 265-371）：主设置函数
   - 从 checkpoint 加载高斯模型
   - 加载 PBR 组件：`CubemapLight`、BRDF LUT、遮挡体积、传输网络
   - 加载反射贴图 `refmap`
   - 选择 `render_fn` 并调用 `eval_render()`

2. **`eval_render()`**（行 424-619）：逐帧渲染和评估
   - 遍历 `transforms_test.json` 中的每帧
   - 调用 `render_fn(custom_cam, gaussians, pipe, background, ...)` 渲染
   - 计算 PSNR/SSIM/LPIPS 指标
   - 保存所有可视化输出（深度、法线、反射属性、PBR 属性等）
   - 可选生成视频

3. **`exported_mesh()`**（行 374-422）：TSDF 融合网格提取（实验性功能）

4. **网格提取**（行 39-262）：
   - `extract_mesh_unbounded()`：用于无界场景，包含空间收缩
   - `marching_cubes_with_contraction()`：分块 Marching Cubes，支持收缩空间
   - `estimate_bounding_sphere()`：从相机位姿估计包围球

---

### 渲染管线总结

```
render_and_eval.py
  └─ evaling()
       ├─ 加载 gaussian, cubemap, refmap, occlusion_volumes, transfer_net
       └─ eval_render()
            └─ render_fn (来自 render_fn_dict)
                 ├─ render()         ← render.py（主渲染器，混合反射 + PBR）
                 │    ├─ render_view()
                 │    │    ├─ PRT 颜色计算
                 │    │    ├─ 特征向量构建 [10维基础 + 8维PBR]
                 │    │    ├─ CUDA 光栅化（rtr_gs_rasterization）
                 │    │    ├─ 延迟着色（反射 + PBR）
                 │    │    └─ 结果组装
                 │    └─ calculate_loss()（仅训练模式）
                 │
                 ├─ render_fast()    ← render_fast.py（快速渲染，仅PBR）
                 │    └─ render_view()
                 │         ├─ 特征向量构建 [3维基础 + 8维PBR]
                 │         ├─ CUDA 光栅化
                 │         └─ PBR 着色
                 │
                 └─ render_equirect() ← render_equirect.py（ODGS 全景）
                      └─ render_view()
                           ├─ PRT 颜色计算
                           ├─ 多 Pass 光栅化（ODGS 光栅化器）
                           │    ├─ Pass1: 主颜色
                           │    ├─ Pass2: 法线
                           │    ├─ Pass3: 反射属性
                           │    └─ Pass4/5: PBR 属性（可选）
                           └─ PBR 着色 + 全景视图方向
```
        
          