# compute_with_prt：预计算辐射传输（PRT）

## 一句话总结

`compute_with_prt` 控制渲染时使用 **PRT（Precomputed Radiance Transfer，预计算辐射传输）** 还是标准 **SH（Spherical Harmonics，球谐函数）** 来计算每个 Gaussian 的出射辐射度。模型训练时必须开启，viewer 渲染时必须与训练一致，否则结果异常。

## 为什么需要 PRT

来自论文第 3.3 节：

> Compared to spherical harmonics, radiance transfer ... provides stronger global low-frequency constraints.
>
> In the shading process, all Gaussians share two global components: the spherical harmonics lighting $c_j$ and the MLP $G$. This design enables shading across Gaussians to be connected through shared components, promoting the representation of overall low-frequency variations.

PRT 的核心优势：
- **共享全局光照**：所有 Gaussian 共享同一组 SH 光照系数和同一个 MLP 网络，提供强低频约束
- **防止高频过拟合**：SH 在每个 Gaussian 上独立，容易过拟合高频细节导致 floating artifacts
- **更好的几何平滑性**：低频约束有助于保持几何结构的平滑

> **注意**：论文描述的"共享"是概念上的整体低频约束。在具体实现中，SHS（`_shs_dc`/`_shs_rest`）是 per-Gaussian 存储的（每个 Gaussian 有自己的一份），但训练过程中它们收敛为描述场景光照的辐射场，而非独立的高频颜色。

## PRT 在代码中的实现

### 总入口（透视模式）

[render.py:L197](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L197)

```python
if pipe.compute_with_prt and override_color is None:
    net = dict_params["transfer_net"]
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    if only_diffuse:
        prt_color = PRTutils.cal_diffuse(pc)
    else:
        prt_color = PRTutils.cal_color(pc, net, viewdirs, normal, is_training)
    override_color = prt_color
```

当 `compute_with_prt=True` 时，`override_color` 被设置为 PRT 计算的颜色，后续光栅化会直接使用这个颜色（跳过 SH 系数）：

```python
colors_precomp = override_color  # L252, 使用 PRT 颜色，不走 SH
```

当 `compute_with_prt=False` 时，颜色回退到标准 SH 求值：

```python
shs = pc.get_shs  # L249, 标准 SH 系数，直接用 eval_sh 得到颜色
```

### 总入口（equirect 全景模式）

[render_equirect.py:L294](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render_equirect.py#L294)

```python
if not fast_pbr:
    only_diffuse = dict_params.get("iteration", 0) < pipe.diffuse_iteration
    if pipe.compute_with_prt and override_color is None and transfer_net is not None:
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
        if only_diffuse:
            prt_color = PRTutils.cal_diffuse(pc)
        else:
            prt_color = PRTutils.cal_color(pc, transfer_net, viewdirs, normal, is_training)
        override_color = prt_color
```

逻辑与透视模式一致，额外多了一个 `transfer_net is not None` 的保护条件（全景模式下 transfer_net 可能尚未初始化）。

### PRT 计算的两个分量

[prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py)

#### 漫反射分量（view-independent）

```python
def cal_diffuse(gaussian, mask=None):  # L8
    # C_d = ρ_d · ReLU( Σ(c_j · c_j^t) + 0.5 )

    diffuse_tint = gaussian.get_diffuse_tint          # L16, 漫反射反照率 ρ_d
    shs_direct_light = ...                             # L24, get_shs → 入射光照 c_j
    shs_diffust_transfer = gaussian.get_diffuse_transfer  # L28, 每个 Gaussian 的漫反射传输向量 c_j^t
    transport = relu((transfer · light).sum(-1) + 0.5)    # L31
    cd = diffuse_tint * transport                          # L34
    return cd
```

对应论文公式 5：
$$C_d \approx \rho_d \sum_{j=0}^{n^2} c_j c_j^t$$

其中 `get_shs`（`_shs_dc` + `_shs_rest`）作为入射光照的 SH 系数 $c_j$，`get_diffuse_transfer` 作为每个 Gaussian 各自的传输向量 $c_j^t$。

> **关于 SHS 的角色**：在 PRT 模式下，SHS（`_shs_dc`/`_shs_rest`）不再编码视角相关的颜色，而是编码位置相关的入射光照。直接用 `eval_sh(shs, dir)` 看到的是"光照图"，不是物体外观。

#### 镜面反射分量（view-dependent）

```python
def cal_specular(gaussian, net, dir, normal, mask=None):  # L43
    # 1. 计算反射方向
    reflect_dir = 2.0 * (normal · view_dir) * normal - view_dir  # L62

    # 2. MLP 解码神经辐射传输向量
    LT_coeff = cal_spec_coff(gaussian, net, reflect_dir)  # L65

    # 3. 传输向量 · 光照 = 反射辐射度
    direct_color = relu((LT_coeff × light).sum(-1))  # L71
    cs = specular_tint * direct_color                 # L74
    return cs
```

对应论文公式 6：
$$C_s(o) \approx \rho_s \sum_{j=0}^{n^2} c_j c_j^t(o)$$

其中 $c_j^t(o) = G(f_t, o)$ 由 MLP 解码。

### MLP 架构

[transfer_mlp.py:L6](file:///home/huangpengyue/projects/RTR-GS/scene/transfer_mlp.py#L6)

```python
class TransferMLP:
    # 3 层 MLP，64 隐藏单元
    net = [
        Linear(3 → 64) + ReLU,       # 输入：反射方向
        Concat + Linear(64+feat → 64) + ReLU,  # 拼接 specular feature
        Linear(64 → 16)              # 输出：SH 传输系数
    ]
```

输入是反射方向 + 每个 Gaussian 的 10 维 specular feature，输出 16 个 SH 传输系数（degree 3）。

### 属性清单

| 属性 | 角色 | 形状 | 可训练 | 学习率 |
|------|------|------|--------|--------|
| `_shs_dc` + `_shs_rest` | 入射光照 $c_j$（SH 域） | `[N, (D+1)², 3]` | ✅ | 0.0025 / 0.000125 |
| `_diffuse_tint` | 漫反射反照率 $\rho_d$ | `[N, 3]` | ✅ | 0.01 |
| `_specular_tint` | 高光色调 $\rho_s$ | `[N, 3]` | ✅ | 0.01 |
| `_diffuse_transfer` | 漫反射传输 $c_j^t$（SH 域） | `[N, (D+1)², 1]` | ✅ | 0.01 / 0.0005 |
| `_specular_feature` | 高光材质特征（MLP 输入 $f_t$） | `[N, 10]` | ✅ | 0.01 |
| `TransferMLP`（共享） | 高光传输解码器 $G$ | 3层MLP 64→16 | ✅ | 2.5e-5 |

### 训练流程中的 PRT

两个阶段都用 `--compute_with_prt`：

```
Stage 1:  python train.py ... --ref_map -t render_ref --compute_with_prt
Stage 2:  python train.py ... --ref_map -t render_ref_pbr --compute_with_prt
Equirect: python train.py ... --ref_map -t render_ref_equirect --compute_with_prt
```

- **Stage 1**：PRT 计算完整的混合渲染颜色（`override_color`），联合训练几何 + 低频辐射 + 高频反射
- **Stage 2**：PRT 继续为混合渲染分支提供辐射度，PBR 分支额外使用 `base_color`/`roughness`/`metallic` 进行材质分解。两个分支并行训练
- **Equirect 模式**：逻辑相同，使用 SGS 的 equirect CUDA 光栅化器

### SHS 在训练中的语义变化

这是理解 PRT 的关键：

1. **初始时**（SGS 刚转换完）：SHS 编码的是标准 3DGS 的视角颜色，`eval_sh` 能看到物体的正常外观
2. **Stage 1 训练后**：SHS 被优化为"位置相关的入射光照"——因为 PRT 模式下梯度通过 `cal_diffuse`/`cal_specular` 反向传播到 SHS，让它学习编码光照而非颜色
3. **Stage 2 训练后**：SHS 继续优化以适应 PBR 分解后的光照模型

这意味着：**训练后的点云放到普通 3DGS 查看器中，直接用 SHS + `eval_sh` 显示会得到"光照图"而非正常外观**。

## 为什么 viewer 与训练不一致

如果模型是用 `--compute_with_prt` 训练的，**但 viewer 中关闭了它**：

1. `override_color` 保持 `None`
2. 渲染器回退到 `shs = pc.get_shs`（标准 3DGS SH 系数）
3. 这些 SH 系数在 PRT 训练过程中已经被优化为"光照"而非"颜色"——因为训练时 `override_color = prt_color` 直接覆盖了颜色，SHS 没有收到有效的渲染损失梯度
4. 结果是：SHS 编码的是光照信息，直接 `eval_sh` 渲染出奇怪的结果

**这就是 viewer 渲染和训练结果不一致的根本原因。**

## 相关命令和参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--compute_with_prt` | `False` | 启用 PRT 替代 SH |
| `--diffuse_iteration 3000` | `0` | 前 N 次迭代仅用漫反射 PRT（让几何先收敛） |

在代码中对应 [PipelineParams](file:///home/huangpengyue/projects/RTR-GS/arguments/__init__.py#L69)：
```python
class PipelineParams:
    self.compute_with_prt = False
    self.diffuse_iteration = 0
```

## 关键代码定位

| 文件 | 行号 | 内容 |
|---|---|---|
| [render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py) | 197-211 | `compute_with_prt` 分支入口（透视模式） |
| [render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py) | 248-252 | SH 回退路径（PRT 关闭时） |
| [render_equirect.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render_equirect.py) | 292-300 | `compute_with_prt` 分支入口（equirect 模式） |
| [render_equirect.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render_equirect.py) | 321-323 | SH 回退路径（equirect 模式） |
| [prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py) | 8-38 | `cal_diffuse` 漫反射 PRT 计算 |
| [prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py) | 43-76 | `cal_specular` 高光 PRT 计算 |
| [prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py) | 79-94 | `cal_spec_coff` MLP 高光传输系数解码 |
| [prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py) | 98-112 | `cal_color` 完整 PRT 颜色（diffuse + specular） |
| [transfer_mlp.py](file:///home/huangpengyue/projects/RTR-GS/scene/transfer_mlp.py) | 6-60 | TransferMLP 网络定义 |
| [gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py) | 273-277 | `get_incidents` 入射光 SH 系数 |
| [arguments/\_\_init\_\_.py](file:///home/huangpengyue/projects/RTR-GS/arguments/__init__.py) | 69 | PipelineParams `compute_with_prt` 定义 |
