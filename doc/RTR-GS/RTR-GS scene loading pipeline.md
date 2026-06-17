# RTR-GS Scene Loading Pipeline

以下是关于 RTR-GS 框架中相机数据加载、构建、迭代和消费的详细分析报告，包含关键代码片段和行号。

---

## 1. Scene 类如何加载相机数据

**文件**: [scene/__init__.py](file:///home/huangpengyue/projects/RTR-GS/scene/__init__.py)

`Scene.__init__` (第 29-109 行) 是加载所有数据的入口。核心逻辑是**根据数据集的目录结构自动检测格式**，然后调用相应的读取函数。

### 1.1 场景类型自动检测 (第 56-78 行)

```python
if os.path.exists(os.path.join(source_path, "sparse")):
    scene_info = sceneLoadTypeCallbacks["Colmap"](...)
elif os.path.exists(os.path.join(source_path, "transforms_train.json")):
    if "stanford_orb" in source_path:
        scene_info = sceneLoadTypeCallbacks["StanfordORB"](...)
    elif "Synthetic4Relight" in source_path:
        scene_info = sceneLoadTypeCallbacks["Synthetic4Relight"](...)
    else:
        scene_info = sceneLoadTypeCallbacks["Blender"](...)
elif os.path.exists(os.path.join(source_path, "inputs/sfm_scene.json")):
    scene_info = sceneLoadTypeCallbacks["NeILF"](...)
```

检测优先级: `sparse/` (Colmap) > `transforms_train.json` (Blender/StanfordORB/Synthetic4Relight) > `inputs/sfm_scene.json` (NeILF)。

### 1.2 从 `CamInfo` 到 `Camera` 对象的转换 (第 101-107 行)

`Scene.__init__` 调用 `cameraList_from_camInfos` 将所有 `CameraInfo` 元组批量转换成 `Camera` 对象：

```python
for resolution_scale in resolution_scales:
    self.train_cameras[resolution_scale] = cameraList_from_camInfos(
        scene_info.train_cameras, resolution_scale, args, read_cam_only=read_cam_only)
    self.test_cameras[resolution_scale] = cameraList_from_camInfos(
        scene_info.test_cameras, resolution_scale, args, read_cam_only=read_cam_only)
```

结果存储在 `self.train_cameras` 和 `self.test_cameras` 两个字典中，key 为分辨率尺度（如 1.0, 2.0 等），value 为 `Camera` 对象列表。

---

## 2. Camera 对象的构造

**文件**: [scene/cameras.py](file:///home/huangpengyue/projects/RTR-GS/scene/cameras.py)

### 2.1 `Camera` 类构造函数 (第 8-81 行)

```python
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, fx, fy, cx, cy,
                 image=None, image_name='', uid=0,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 height=None, width=None, depth=None, normal=None, image_mask=None,
                 render_only=False):
```

**加载的相机参数**：
- `colmap_id`: COLMAP 中的相机 ID
- `R` (np.array): 旋转矩阵（3x3），已转置存储以适配 CUDA GLM 约定
- `T` (np.array): 平移向量（3,）
- `FoVx`, `FoVy`: 水平和垂直视场角（弧度）
- `fx`, `fy`: 焦距（像素单位）
- `cx`, `cy`: 主点坐标（像素单位）
- `image`: 原始图像张量（[3, H, W]），自动 clamp 到 [0,1] 并移到指定设备
- `image_name`: 图像文件名（不含扩展名）
- `depth`: 深度图（可选），默认为全零
- `normal`: 法线图（可选），默认为全零
- `image_mask`: 图像掩码（可选），默认为全 1
- `trans`, `scale`: 额外变换参数（场景归一化用）

### 2.2 派生矩阵的计算 (第 63-81 行)

构造时自动计算以下变换矩阵：

```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

# 投影矩阵（使用 FoV 或者 fx/fy）
self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()

# 完整投影矩阵 = W2V * Proj
self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
    self.projection_matrix.unsqueeze(0))).squeeze(0)

# 相机中心（世界空间）
self.camera_center = self.world_view_transform.inverse()[3, :3]

# 相机到世界矩阵
self.c2w = self.world_view_transform.transpose(0, 1).inverse()

# 内参矩阵 (3x3)
self.intrinsics = self.get_intrinsics()

# 外参矩阵 (4x4)
self.extrinsics = self.get_extrinsics()

# 投影矩阵 K * [R|T]
self.proj_matrix = self.get_proj_matrix()
```

### 2.3 内参计算逻辑 (第 103-114 行)

根据是否提供 `fx`/`fy` 有两种模式：

- **有显式内参**: `K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
- **无显式内参（仅 FoV）**: `fx = W / (2 * tan(FoVx/2))`, `fy = H / (2 * tan(FoVy/2))`, 主点取图像中心

### 2.4 `loadCam` 函数中的图像缩放逻辑

**文件**: [utils/camera_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/camera_utils.py) (第 13-81 行)

当 `args.resolution` 指定时，图像会被缩放，内参也会同步调整：

```python
# 调整内参：cx, cy, fx, fy 都除以 scale
scale_cx = cam_info.cx / scale
scale_cy = cam_info.cy / scale
scale_fx = cam_info.fx / scale
scale_fy = cam_info.fy / scale
```

---

## 3. 支持的 5 种数据集格式及其解析

**文件**: [scene/dataset_readers.py](file:///home/huangpengyue/projects/RTR-GS/scene/dataset_readers.py)

所有解析器将数据转换为 `CameraInfo` NamedTuple（第 19-37 行），然后由 `Scene` 类统一处理。

### 3.1 数据结构定义 (第 19-44 行)

```python
class CameraInfo(NamedTuple):
    uid: int           # 相机唯一 ID
    R: np.array        # 旋转矩阵（3x3，转置存储）
    T: np.array        # 平移向量
    FovY: np.array     # 垂直视场角
    FovX: np.array     # 水平视场角
    fx: np.array       # x 方向焦距
    fy: np.array       # y 方向焦距
    cx: np.array       # 主点 x
    cy: np.array       # 主点 y
    image: np.array    # RGB 图像
    image_path: str    # 图像路径
    image_name: str    # 图像名称（不含扩展名）
    width: int         # 图像宽度
    height: int        # 图像高度
    image_mask: np.array  # 图像掩码
    trans: np.array    # 变换偏移
```

`SceneInfo`（第 39-44 行）包含点云、相机列表、归一化参数等。

### 3.2 注册表 (第 655-661 行)

```python
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Synthetic4Relight": readSynthetic4RelightInfo,
    "NeILF": readNeILFInfo,
    "StanfordORB": readStanfordORBInfo,
}
```

### 3.3 COLMAP 格式 (第 70-220 行)

- **检测**: 目录下有 `sparse/` 文件夹
- **文件**: `images.bin`/`images.txt`, `cameras.bin`/`cameras.txt`, `points3D.bin`/`points3D.txt`
- **相机模型**: 仅支持 `SIMPLE_PINHOLE`（一个焦距）和 `PINHOLE`（两个焦距）
- **内外参**: 从 `cam_extrinsics` 读取 `qvec`（四元数转旋转矩阵）和 `tvec`，从 `cam_intrinsics` 读取焦距和主点
- **训练/测试划分**: 若 `eval=True`，用 LLFF 8 间隔法抽出测试集（第 189-194 行）；DTU 数据集有固定测试索引 `[2, 12, 17, 30, 34]`（第 186 行）

### 3.4 Blender/NeRF-Synthetic 格式 (第 223-331 行)

- **检测**: `transforms_train.json` 文件，且路径中不含 `stanford_orb` 或 `Synthetic4Relight`
- **JSON 结构**: 包含 `camera_angle_x` 和 `frames[]`，每帧有 `file_path` 和 `transform_matrix`
- **坐标系转换**: COLMAP 约定 (Y 向下, Z 向前) 与 Blender (Y 向上, Z 向后) 之间的转换在第 243-244 行：
  ```python
  c2w[:3, 1:3] *= -1   # 将 OpenGL/Blender 坐标转为 COLMAP 坐标
  ```
- **RGBA 处理**: 若有透明通道，对白色/黑色背景做合成（第 265 行）
- **点云**: 无 COLMAP 数据，因此生成 100,000 个随机点（第 309-318 行）

### 3.5 StanfordORB 格式 (第 518-563 行)

- **检测**: 路径中含 `stanford_orb` 关键字
- **特殊处理**: 使用 `readCamerasFromTransforms2`（第 462-515 行），支持 `.png` 和 `.exr` 格式，图像缩放到 `benchmark_size = 512`，有独立的 mask 路径（`_mask` 后缀）
- **点云**: 生成 100,000 个随机点，范围 `[-0.5, 0.5]^3`（第 544 行），与 NeRF 的不同

### 3.6 Synthetic4Relight 格式 (第 614-653 行)

- **检测**: 路径中含 `Synthetic4Relight` 关键字
- **特殊处理**: 使用 `readCamerasFromTransforms3`（第 565-611 行），训练集用 `_rgb.exr`，测试集用 `_rgba.png`，mask 路径自动从 `_rgb.exr` 替换为 `_mask.png`

### 3.7 NeILF 格式 (第 415-460 行)

- **检测**: `inputs/sfm_scene.json` 文件
- **数据结构**: JSON 包含 `camera_track_map.images` 字典，每个相机有 `intrinsic`（焦距和主点）、`extrinsic`（4x4 矩阵）和 `flg`（==2 表示有效）
- **包围盒变换**: 使用 `bbox_transform` 对点云和相机位置做归一化（第 339-366 行）
- **训练/测试划分**: 对 DTU 数据集，测试索引 `[2, 12, 17, 30, 34]`（第 419 行）

---

## 4. 训练循环中的相机迭代方式

**文件**: [train.py](file:///home/huangpengyue/projects/RTR-GS/train.py)

### 4.1 训练循环中的随机采样 (第 133-152 行)

```python
viewpoint_stack = None
for iteration in progress_bar:
    # ...
    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
```

核心机制：
1. 开始时 `viewpoint_stack = None`
2. 当 stack 为空时，重新复制一份完整的训练相机列表
3. 每次迭代从 stack 中**随机 pop 一个相机**
4. 所有相机遍历完一轮后重新填充 stack

这种方式确保每个 epoch 中每个相机被选中恰好一次（无放回抽样），但 epoch 间顺序是随机的。

### 4.2 渲染调用 (第 158-160 行)

```python
render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                       opt=opt, is_training=True, dict_params=pbr_kwargs)
```

`render_fn` 根据 `args.type` 决定：
- `render_ref` / `render_ref_pbr` -> `render` (来自 `gaussian_renderer/render.py`)
- `render_ref_fast` -> `render_fast`

### 4.3 验证阶段的批量迭代 (第 284-298 行)

```python
validation_configs = (
    {'name': 'test', 'cameras': scene.getTestCameras()},
    {'name': 'train', 'cameras': scene.getTrainCameras()})
for config in validation_configs:
    for idx, viewpoint in enumerate(tqdm(config['cameras'], ...)):
        render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color, ...)
```

验证和评估时（`eval_render`，第 367 行），是顺序遍历所有测试相机。

---

## 5. 渲染器中相机的消费方式

**文件**: [gaussian_renderer/__init__.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/__init__.py)

入口注册表（第 5-12 行）：
```python
render_fn_dict = {
    "render_ref": render,
    "render_ref_pbr": render,
    "render_ref_fast": render_fast,
    "neilf_ref": render, "neilf_ref_pbr": render, "neilf_ref_fast": render_fast,
}
```

**文件**: [gaussian_renderer/render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py)

### 5.1 主渲染函数 `render_view` (第 18-507 行)

该函数接收 `viewpoint_camera: Camera` 参数，相机被用于以下方面：

#### a) 构建光栅化配置 (第 115-136 行)

```python
tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
intrinsic = viewpoint_camera.intrinsics

raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx, tanfovy=tanfovy,
    cx=float(intrinsic[0, 2]), cy=float(intrinsic[1, 2]),  # 主点
    viewmatrix=viewpoint_camera.world_view_transform,        # W2V 矩阵
    projmatrix=viewpoint_camera.full_proj_transform,         # Proj * W2V
    campos=viewpoint_camera.camera_center,                   # 相机位置
    # ...
)
```

#### b) 获取法线方向 (第 146 行)

```python
normal = pc.get_min_axis(viewpoint_camera.camera_center)
```

#### c) 计算视线方向 (第 149-150 行)

```python
dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
dir_pp_normalized = F.normalize(dir_pp, dim=-1)
```

#### d) 计算深度 (第 153-155 行)

```python
xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
```

#### e) PRT 颜色计算 (第 161-165 行)

```python
viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
prt_color = PRTutils.cal_color(pc, net, viewdirs, normal, is_training)
```

#### f) 规范光线与 c2w 矩阵 (第 199-200 行)

```python
canonical_rays = dict_params["canonical_rays"]
c2w = viewpoint_camera.c2w   # 用于将视线方向从相机空间转到世界空间
```

#### g) 延迟渲染中的法线/反射计算 (第 297-308 行)

渲染出法线图后，结合 c2w 和 canonical_rays 计算每个像素的世界空间视线方向：

```python
view_dirs = -(
    (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])
    .sum(dim=-1).reshape(H, W, 3)
)
```

#### h) 损失函数中的 GT 图像访问 (第 525 行)

```python
gt_image = viewpoint_camera.original_image.cuda()
```

以及 `viewpoint_camera.depth`, `viewpoint_camera.image_mask`, `viewpoint_camera.normal` 等。

---

## 总结: 完整数据流

```
磁盘文件 (COLMAP/JSON)
    |
    v
sceneLoadTypeCallbacks[...]()  ->  CameraInfo (NamedTuple)
    |
    v
Scene.__init__()
    |-- shuffle (随机打乱)
    |-- cameraList_from_camInfos()
        |-- loadCam()  ->  图像缩放 + 内参缩放
            |-- Camera(R, T, FoVx, FoVy, fx, fy, cx, cy, ...)
                |-- 自动计算: world_view_transform, projection_matrix,
                |   full_proj_transform, camera_center, c2w,
                |   intrinsics (3x3 K), extrinsics (4x4), proj_matrix (3x4)
    |
    v
train.py 训练循环:
    |-- scene.getTrainCameras()  -> Camera 列表
    |-- 随机 pop 一个 Camera
    |-- render_fn(camera, ...)
        |-- 使用: FoVx/FoVy (tanfov), intrinsics (cx/cy),
        |   world_view_transform, full_proj_transform, camera_center,
        |   c2w, image_height, image_width
        |-- 损失使用: original_image, depth, image_mask, normal
```

相机参数的关键设计特点是**双路径内参**：
- 如果提供了 `fx`/`fy`/`cx`/`cy`，则使用精确的像素单位内参（Colmap, NeILF 格式）
- 如果未提供，则从 `FoVx`/`FoVy` 反算焦距，主点取图像中心（Blender/NeRF 格式）
        