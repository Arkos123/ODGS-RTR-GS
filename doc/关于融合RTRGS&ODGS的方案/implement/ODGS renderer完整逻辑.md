好的，我已经完整读取了所有四个文件。以下是详细的 API 分析：

---

## 1. `gaussian_renderer/__init__.py` — ODGS 渲染器 API

**文件路径**: `/home/huangpengyue/projects/RTR-GS/submodules/odgs/gaussian_renderer/__init__.py`

### `render()` 函数签名

```python
def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None):
```

### 参数说明

| 参数 | 类型 | 说明 |
|---|---|---|
| `viewpoint_camera` | `Camera` | 来自 `scene/cameras.py` 的相机对象 |
| `pc` | `GaussianModel` | 高斯模型，包含位置、SH系数、不透明度、缩放、旋转 |
| `pipe` | `PipelineParams` | 包含 `compute_cov3D_python`, `convert_SHs_python`, `debug` 标志 |
| `bg_color` | `torch.Tensor` | 背景色，必须在 GPU 上 |
| `scaling_modifier` | `float` | 缩放修饰符，默认 1.0 |
| `override_color` | `torch.Tensor` | 可选，覆盖预设颜色 |

### 栅格化设置 (`GaussianRasterizationSettings`)

从 `odgs_gaussian_rasterization` 导入。关键参数字段：

```python
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    bg=bg_color,
    scale_modifier=scaling_modifier,
    viewmatrix=viewpoint_camera.world_view_transform,    # [4x4] 世界到相机矩阵
    sh_degree=pc.active_sh_degree,
    campos=viewpoint_camera.camera_center,                # [3] 相机位置
    prefiltered=False,
    debug=pipe.debug
)
```

### 栅格化调用 (`rasterizer()`)

```python
rendered_image, depth, acc, radii, psi, lat, lon = rasterizer(
    means3D=means3D,        # [N, 3] 高斯中心位置
    means2D=means2D,        # [N, 2] 屏幕空间坐标（梯度跟踪用）
    shs=shs,                # [N, 3, (max_sh_degree+1)^2] SH 系数
    colors_precomp=colors_precomp,  # [N, 3] 预设颜色（可选）
    opacities=opacity,      # [N, 1] 不透明度
    scales=scales,          # [N, 3] 各向同性缩放（ODGS 特有！3 个值相同）
    rotations=rotations,    # [N, 4] 四元数旋转
    cov3D_precomp=cov3D_precomp  # [N, 6] 预设协方差（可选）
)
```

### 返回值

```python
{
    "render": rendered_image,    # [C, H, W] 最终渲染图
    "depth": depth,              # [1, H, W] 深度图
    "accuracy": acc,             # [1, H, W] 累积不透明度
    "viewspace_points": screenspace_points,  # 用于梯度跟踪
    "visibility_filter": radii > 0,          # 可见高斯掩码
    "psi": psi,    # 自定义 ODGS 输出 — 可能是 equirectangular 投影的中间量
    "lat": lat,    # 自定义 ODGS 输出 — 纬度信息
    "lon": lon,    # 自定义 ODGS 输出 — 经度信息
    "radii": radii # [N] 投影后高斯半径
}
```

**关键差异**：与原始 3DGS 相比，ODGS 的栅格化器返回额外的输出：`psi`、`lat`、`lon`（equirectangular 投影相关）。`scales` 参数是**各向同性**的一维向量（不是三维），这是 ODGS 论文的设计。

---

## 2. `scene/cameras.py` — 相机模型

**文件路径**: `/home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/cameras.py`

### `Camera` 类构造函数

```python
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
```

### 关键属性

| 属性 | 计算方式 | 说明 |
|---|---|---|
| `R` | 直接传入 | 旋转矩阵 `[3, 3]`（世界到相机） |
| `T` | 直接传入 | 平移向量 `[3]`（世界到相机） |
| `FoVx` | 直接传入 | 水平视场角（弧度） — 对 equirectangular 固定为 `2*pi` |
| `FoVy` | 直接传入 | 垂直视场角（弧度） — 对 equirectangular 固定为 `pi` |
| `image_height` / `image_width` | 来自输入图像尺寸 | 像素尺寸 |
| `world_view_transform` | `getWorld2View2(R, T, trans, scale).transpose(0,1)` | `[4, 4]` 视图矩阵 |
| `projection_matrix` | `getProjectionMatrix(znear, zfar, fovX, fovY).transpose(0,1)` | `[4, 4]` 投影矩阵 |
| `full_proj_transform` | `world_view_transform @ projection_matrix` | `[4, 4]` 完整变换 |
| `camera_center` | `world_view_transform.inverse()[3, :3]` | `[3]` 相机世界坐标 |

### 关键计算细节

1. **`world_view_transform`**: 用 `getWorld2View2(R, T, trans, scale)` 构建后**转置**。原始 3DGS 在 OpenGL 列主序和 PyTorch 行主序之间做了转置。CUDA 栅格化器期望的 `viewmatrix` = `world_view_transform`（转置后的 `[4, 4]`）。

2. **`projection_matrix`**: 用 `znear=0.01`, `zfar=100.0` 以及 `FoVx`, `FoVy` 调用 `getProjectionMatrix()`，然后再次**转置**。

3. **`camera_center`**: 通过取 `world_view_transform` 的逆矩阵的第 3 行前 3 个元素得到。这对应世界坐标系中相机的位置。

4. **FoV 约定**: 对于 equirectangular 模式，`FoVx = 2*pi`（360度），`FoVy = pi`（180度）。这些值通过 `utils/graphics_utils.py` 中的 `getProjectionMatrix()` 函数传入投影矩阵。

5. **`MiniCam`**: 一个轻量替代类，直接接受 `world_view_transform` 和 `full_proj_transform`，主要用于推理时的快速相机创建。

---

## 3. `arguments/__init__.py` — ODGS 参数定义

**文件路径**: `/home/huangpengyue/projects/RTR-GS/submodules/odgs/arguments/__init__.py`

### `ModelParams`

```python
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""       # 数据集路径，通过 -s 指定
        self._model_path = ""        # 输出路径，通过 -m 指定
        self._images = "images"      # 图像子目录名
        self._resolution = -1        # 分辨率缩放（-1 为原始尺寸）
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False            # 启用 train/test 分割
        self.use_depth = False
        self.use_dense = False
```

**重要**：`extract()` 方法将 `source_path` 转为绝对路径：
```python
def extract(self, args):
    g = super().extract(args)
    g.source_path = os.path.abspath(g.source_path)
    return g
```

### `PipelineParams`

```python
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False  # 在 Python 中计算 SH → RGB
        self.compute_cov3D_python = False # 在 Python 中计算 3D 协方差
        self.debug = False
```

### `OptimizationParams`

| 参数 | 默认值 | 说明 |
|---|---|---|
| `iterations` | 30,000 | 总迭代数 |
| `position_lr_init` | 0.00016 | 初始位置学习率 |
| `position_lr_final` | 0.0000016 | 最终位置学习率 |
| `position_lr_delay_mult` | 0.01 | 学习率延迟倍数 |
| `position_lr_max_steps` | 30,000 | 学习率调度步数 |
| `feature_lr` | 0.0025 | 特征/SH 学习率 |
| `opacity_lr` | 0.05 | 不透明度学习率 |
| `scaling_lr` | 0.005 | 缩放学习率 |
| `rotation_lr` | 0.001 | 旋转学习率 |
| `percent_dense` | 0.01 | 密集化百分位数 |
| `lambda_dssim` | 0.2 | SSIM 损失权重 |
| `densification_interval` | 100 | 密集化间隔（步数） |
| `opacity_reset_interval` | 3,000 | 不透明度重置间隔 |
| `densify_from_iter` | 500 | 开始密集化 |
| `densify_until_iter` | 15,000 | 停止密集化 |
| `random_background` | False | 随机背景 |
| `densify_grad_threshold_min` | 0.0001 | 最小梯度阈值（纬度自适应） |
| `densify_grad_threshold_max` | 0.0001 | 最大梯度阈值（纬度自适应） |

**ODGS 特有**：与原始 3DGS（单梯度阈值）不同，ODGS 定义了 `densify_grad_threshold_min` 和 `densify_grad_threshold_max` 两个阈值，实现论文中的**纬度自适应密集化策略**。当两者相等时（默认值 0.0001），相当于退化为原始 3DGS 行为。

### `get_combined_args()`

从 `cfg_args` 文件加载保存的参数，并与命令行参数合并。命令行参数优先级更高。

---

## 4. `utils/loss_utils.py` — 纬度加权损失函数

**文件路径**: `/home/huangpengyue/projects/RTR-GS/submodules/odgs/utils/loss_utils.py`

### 损失函数

#### `l1_loss(network_output, gt)`
标准 L1 损失：`|output - gt|.mean()`

#### `l2_loss(network_output, gt)`
标准 MSE 损失：`(output - gt)^2.mean()`

#### `ssim(img1, img2, window_size=11, ws_map=None)`
标准 SSIM。如果提供 `ws_map`，则返回 `(ssim_map.mean(), ws_ssim_map.mean() / ws_map.mean())` 元组，同时计算普通 SSIM 和纬度加权 SSIM（WS-SSIM）。

#### `est_wsmap(img)`
**纬度权重图生成**：
```python
def est_wsmap(img):
    H, W = img.shape[-2:]
    col = torch.arange(H)
    ws_map = torch.cos((col + 0.5 - H/2) * torch.pi / H).reshape((H,1)).expand(H, W)
    ws_map = ws_map.type_as(img)
    return ws_map
```

**数学公式**：
```
ws_map[row] = cos((row + 0.5 - H/2) * pi / H)
```

这对应 equirectangular 图像中**每行的余弦纬度权重**：
- 赤道（中间行，`row ≈ H/2`）：权重 ≈ cos(0) = **1.0**（最大值）
- 极点（顶部/底部行，`row = 0` 或 `row = H-1`）：权重 ≈ cos(pi/2) = **0.0**（最小值）

此权重图用于实现 **WS-PSNR** 和 **WS-SSIM** 指标，补偿 equirectangular 投影中的像素过采样（极点附近的像素比赤道附近的像素覆盖更少的立体角）。

---

## 总结：ODGS CUDA 栅格化器 Python 绑定

| 方面 | 详情 |
|---|---|
| **导入路径** | `from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer` |
| **设置参数** | `image_height`, `image_width`, `bg`, `scale_modifier`, `viewmatrix`, `sh_degree`, `campos`, `prefiltered`, `debug` |
| **前向调用** | `rasterizer(means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp)` |
| **输出** | 7 个张量：`rendered_image [C,H,W]`, `depth [1,H,W]`, `acc [1,H,W]`, `radii [N]`, `psi`, `lat`, `lon` |
| **缩放维度** | **一维**（各向同性）— ODGS 特有，与 3DGS 的三维各向异性缩放不同 |
| **额外 CUDA 输出** | `psi`, `lat`, `lon` — equirectangular 投影的辅助量 |
| **相机约定** | `FoVx=2*pi`, `FoVy=pi` 用于 equirectangular；`world_view_transform` 转置后传入栅格化器 |
| **纬度加权损失** | `est_wsmap()` 生成 `cos(lat)` 权重图；`ssim()` 可选接受 `ws_map` 计算 WS-SSIM |
        
    