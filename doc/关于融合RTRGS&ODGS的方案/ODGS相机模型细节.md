# ODGS 子模块调查结果
以下是关于 ODGS (Omnidirectional Gaussian Splatting) 子模块的四项详细调查结果。

---

## 1. Camera 类定义 (scene/cameras.py)

**文件路径**: [/home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/cameras.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/cameras.py)

ODGS 中的 `Camera` 类与原始 3DGS 的 Camera 类**几乎完全相同**，没有添加任何 equirectangular 专有参数。其定义如下：

```python
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
```

**核心参数**:
- `FoVx`, `FoVy` -- 水平和垂直视场角（弧度）。在 ODGS 的 equirectangular 场景中，这些值被硬编码为 `FoVx = 3.13768641`（约 179.8 度，接近 360 度），`FoVy` 由 `focal2fov` 计算得出
- `R` (3x3 旋转矩阵), `T` (3D 平移向量) -- 标准世界到相机变换
- `image` -- 加载的图像张量 (CxHxW)
- `world_view_transform`, `projection_matrix`, `full_proj_transform`, `camera_center` -- 标准 3DGS 的投影变换

**关键差异**: `projection_matrix` 通过 `getProjectionMatrix(znear, zfar, fovX, fovY)` 计算，使用标准透视投影矩阵公式。**但实际上 ODGS 的 CUDA rasterizer 完全忽略了这个投影矩阵** -- 它不使用 `tanfovx/tanfovy` 或 `projmatrix` 进行透视投影，而是使用自己的球面/equirectangular 投影逻辑。

**没有 equirectangular 专有字段**: Camera 类中 **没有** 以下字段：
- `latitude_weights` (纬度权重)
- `spherical_coords` (球坐标)
- `equirectangular` 标记
- 任何与 equirectangular 投影相关的额外参数

还有一个 `MiniCam` 内部类，同样无 equirectangular 专有字段。

---

## 2. 渲染函数与 Equirectangular 投影设置 (gaussian_renderer/__init__.py)

**文件路径**: [/home/huangpengyue/projects/RTR-GS/submodules/odgs/gaussian_renderer/__init__.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/gaussian_renderer/__init__.py)

### RasterizationSettings 配置

```python
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    bg=bg_color,
    scale_modifier=scaling_modifier,
    viewmatrix=viewpoint_camera.world_view_transform,
    sh_degree=pc.active_sh_degree,
    campos=viewpoint_camera.camera_center,
    prefiltered=False,
    debug=pipe.debug
)
```

**与原始 3DGS 的关键区别**:
1. **没有 `tanfovx` / `tanfovy`** -- 原始 3DGS 传递 `tan(FoV/2)` 用于透视投影，ODGS 完全不需要
2. **没有 `projmatrix`** -- 原始 3DGS 传递完整的投影矩阵，ODGS 不需要
3. **使用的 rasterizer** 是 `odgs_gaussian_rasterization`（自定义 CUDA 扩展），而非 `diff_gaussian_rasterization`

### Equirectangular 投影的实现

ODGS 的 equirectangular 投影完全在 CUDA 内核中实现，通过 [forward.cu](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/submodules/odgs-gaussian-rasterization/cuda_rasterizer/forward.cu) 中的 `computeOmniCov2D` 函数完成：

```cpp
__device__ float3 computeOmniCov2D(float lon, float lat, float dist,
    const float* cov3D, const float* viewmatrix, const int height, const int width)
{
    const float x_scale = width/(2*M_PI);   // 水平像素缩放因子
    const float y_scale = height/(M_PI);    // 垂直像素缩放因子

    // 核心 Jacobian 矩阵 J_omni = S_o * Q_o * J_o * T
    glm::mat3 SQJ = glm::mat3(
        x_scale/((cos(lat) + e) * dist), 0, 0,
        0, y_scale/dist, 0,
        0, 0, 0);

    // 旋转矩阵 T，将坐标对齐到 Gaussian 中心方向
    glm::mat3 T = glm::mat3(
        cos(lon), 0.0f, -sin(lon),
        sin(lat) * sin(lon), cos(lat), sin(lat) * cos(lon),
        cos(lat) * sin(lon), -sin(lat), cos(lat) * cos(lon));

    glm::mat3 W = ...  // 视图矩阵的左上 3x3
    glm::mat3 J_o = W * T * SQJ;
    // 最终 2D 协方差 = J_o^T * Vrk * J_o
}
```

这个函数实现了论文中的公式 (5)-(8)，关键步骤：
1. **球面切线平面投影**: 计算每个 Gaussian 在单位球面上的球坐标 `(lon, lat)` 和距离 `dist`
2. **Equirectangular 畸变补偿**: 通过 `x_scale / (cos(lat) + epsilon)` 实现，`1/cos(lat) = sec(lat)` 因子使极地附近的 Gaussian 被水平拉伸
3. **标准像素空间缩放**: `width/(2*PI)` 和 `height/PI` 将经度/纬度映射到像素坐标

### 渲染输出

```python
rendered_image, depth, acc, radii, psi, lat, lon = rasterizer(...)
```

ODGS rasterizer **额外返回三个 equirectangular 专有张量**:
- `psi` -- 每个像素的某种渲染权重/概率（用于透明度混合）
- `lat` (latitude) -- 每个像素/高斯对应的纬度值（-PI/2 到 PI/2），用于纬度自适应稠密化
- `lon` (longitude) -- 每个像素/高斯对应的经度值

### 与 Pinhole 渲染的对比

[gaussian_renderer/pinhole_renderer.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/gaussian_renderer/pinhole_renderer.py) 使用标准的 `diff_gaussian_rasterization_pinhole`，传递 `tanfovx`、`tanfovy`、`projmatrix` 等标准透视参数，且**不返回** `psi`, `lat`, `lon`。

---

## 3. Equirectangular 数据加载 (scene/dataset_readers.py)

**文件路径**: [/home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/dataset_readers.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/dataset_readers.py)

### 支持的数据格式

ODGS 的 Scene 类（[scene/__init__.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/__init__.py)）**仅支持 OpenMVG 格式**:

```python
if os.path.exists(os.path.join(args.source_path, "data_extrinsics.json")):
    scene_info = sceneLoadTypeCallbacks["OpenMVG"](args.source_path, ...)
else:
    assert False, "Could not recognize scene type!"
```

### OpenMVG 数据格式要求

数据集目录下需要：
- `data_extrinsics.json` -- 相机外参（OpenMVG 格式的 JSON 文件）
- `data_views.json` -- 视图列表，将 camera_key 映射到图像文件名
- `images/` -- 图像目录
- `train.txt` / `test.txt` -- 训练/测试集分割（每行一个图像名）
- `pcd.ply` 或 `colorized.ply` -- 初始点云

### Equirectangular 特有的 FoV 硬编码

```python
def readCamerasFromOpenMVG(path, extrinsicsfile, cam_dict, white_background):
    fovx = 3.13768641  # 硬编码！约 179.8 度
    ...
    fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
    FovY = fovy
    FovX = fovx
```

关键点：
- `FoVx` 被硬编码为 `3.13768641`（接近 180 度），因为 equirectangular 图像理论上覆盖了水平方向整个 360 度视野
- `FoVy` 由图像宽高比自动计算得出
- 没有从数据文件中读取相机内参（无焦距、主点等），全部硬编码

### CameraInfo NamedTuple

```python
class CameraInfo(NamedTuple):
    uid: int
    R: np.array        # 旋转矩阵
    T: np.array        # 平移向量
    FovY: np.array     # 垂直 FoV
    FovX: np.array     # 水平 FoV（硬编码为 ~180度）
    image: np.array    # PIL Image
    image_path: str
    image_name: str
    width: int
    height: int
```

**没有 equirectangular 专有字段**。

### 数据加载流程

```
data_extrinsics.json + data_views.json
    → readOpenMVGInfo()
        → readCamerasFromOpenMVG()   [生成 CameraInfo 列表]
            → cameraList_from_camInfos()  [utils/camera_utils.py]
                → loadCam()
                    → PILtoTorch() 调整图像分辨率
                    → Camera(R, T, FoVx, FoVy, image, ...)
```

---

## 4. Equirectangular 专有字段/组件总结

ODGS 的 equirectangular 支持分布在以下组件中，而非集中在 Camera 类中：

### (a) CUDA Rasterizer 内部 (核心实现)

**[submodules/odgs-gaussian-rasterization/cuda_rasterizer/forward.cu](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/submodules/odgs-gaussian-rasterization/cuda_rasterizer/forward.cu)**:

| 元素 | 位置 | 描述 |
|------|------|------|
| `computeOmniCov2D(lon, lat, dist, ...)` | forward.cu:75 | equirectangular 投影的核心 Jacobian 计算 |
| 水平拉伸因子 `1/(cos(lat) + e)` | forward.cu:86 | sec(lat) 畸变补偿，极地附近高斯被放大 |
| 像素缩放 `width/(2*PI)`, `height/PI` | forward.cu:82-83 | 球坐标到像素的映射 |
| 旋转矩阵 `T` | forward.cu:90-93 | 将坐标系统对齐到高斯中心方向 |
| 输出 `psi, lat, lon` | __init__.py:83-84 | 返回纬度、经度等额外信息 |

### (b) 纬度自适应稠密化 (GaussianModel)

**[scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/gaussian_model.py)**:

| 方法 | 行号 | 描述 |
|------|------|------|
| `densify_and_split(grads, grad_threshold_min, grad_threshold_max, scene_extent, lat, N=2)` | L363 | 使用动态梯度阈值 `cos(lat)` 进行分裂 |
| `densify_and_clone(grads, ..., lat)` | L393 | 使用动态梯度阈值进行克隆 |
| `densify_and_prune(..., lat)` | L410 | 统一入口，传递 `lat` |

**动态梯度阈值公式** (实现于 L369-371, L396):
```python
cos_lat = torch.cos(lat)
dynamic_grad_threshold = (1 - cos_lat) * (grad_threshold_max - grad_threshold_min) + grad_threshold_min
```

在 [train.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/train.py) 的 L126，训练循环将 render 返回的 `lat` 传递给 densify 函数:
```python
gaussians.densify_and_prune(opt.densify_grad_threshold_min, opt.densify_grad_threshold_max,
                             0.005, scene.cameras_extent+100, size_threshold, lat)
```

### (c) 加权损失函数 (utils/loss_utils.py)

**[utils/loss_utils.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/utils/loss_utils.py)**:

```python
def est_wsmap(img):
    H, W = img.shape[-2:]
    col = torch.arange(H)
    ws_map = torch.cos((col + 0.5 - H/2) * torch.pi / H).reshape((H,1)).expand(H, W)
    ws_map = ws_map.type_as(img)
    return ws_map
```

- `est_wsmap` 生成一个 **纬度权重图**，值为 `cos(latitude)`，在高纬度（极地）处权重低，在赤道处权重高
- 在 [train.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/train.py) L186-L194 中用于 **WS-SSIM** 和 **WS-PSNR** 评估:
  ```python
  ws_map = est_wsmap(image)
  test_psnr, test_wspsnr = psnr(image, gt_image, ws_map)
  test_ssim, test_wsssim = ssim(image, gt_image, ws_map=ws_map)
  ```

### (d) 球面剔除 (CUDA 内部)

ODGS 的 rasterizer 使用**球壳剔除**（spherical shell culling）替代了原始 3DGS 的**视锥剔除**（frustum culling），因为 equirectangular 相机没有明确的视锥体边界。

---

## 总结

| 方面 | 关键发现 |
|------|---------|
| **Camera 类** | 与原始 3DGS 一致，**无任何 equirectangular 专有字段**。FoV 硬编码为 ~180度 |
| **Rasterizer** | 使用 `odgs_gaussian_rasterization`（自定义 CUDA），**不传递 tanfovx/tanfovy/projmatrix**，而使用球面投影。额外返回 `psi, lat, lon` |
| **数据格式** | 仅支持 **OpenMVG 格式**（`data_extrinsics.json` + `data_views.json`），FoVx 硬编码为 3.13768641，无内参读取 |
| **纬度权重** | 损失函数中的 `est_wsmap()` 生成 `cos(lat)` 权重图，用于 WS-SSIM/WS-PSNR |
| **稠密化** | GaussianModel 接收 `lat` 参数，使用公式 `tau = tau_min + (1-cos(lat))*(tau_max-tau_min)` 动态调整梯度阈值 |
| **CUDA 核心** | `computeOmniCov2D()` 实现完整 equirectangular 投影管线：切线平面 -> 旋转对齐 -> sec(lat) 拉伸 -> 像素缩放 |
        
          
