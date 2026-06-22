# Perspective 模型渲染 Equirect 全景图

## 需求

在 perspective（非 equirect）模式下训练的 checkpoint，给定一个世界空间位置，渲染 6 个 cubemap face 并拼接为 equirectangular panorama，以便和全景模式（SGS/ODGS）的训练结果直观对比。

## 方案

### 核心思路

1. 在目标位置创建 6 个 Camera，分别朝 nvdiffrast 约定的 6 个方向（+X/-X/+Y/-Y/+Z/-Z），FOV=90°
2. 对每个 Camera 调用 perspective 的 `render_fn` 渲染图像
3. 将 6 张 face 图像合成为 equirect

## 新增文件 / 函数

### `scripts/render_cubemap_equitect.py` (新文件)

| 函数 | 作用 |
|------|------|
| `_c2w_rotation(forward, up)` | GLM 风格 look-at → C2W 旋转矩阵 |
| `_make_canonical_rays(H, W, fov)` | 构造给定分辨率+视场角的 canonical rays |
| `make_cubemap_camera(position, forward, up, face_res, uid)` | 创建朝向指定方向的 Camera |
| `render_equirect_from_position(position, gaussians, pipe, render_fn, dict_params, ...)` | **主入口**：渲染 6 个 face 并拼接为 equirect |

### `utils/graphics_utils.py`

| 函数 | 作用 |
|------|------|
| `latlong_to_cubemap_equirect(latlong_map, res)` | equirect → 6 cubemap faces（nvdiffrast `cube_to_dir`） |
| `cubemap_to_equirect(cubemap_faces, eq_width, eq_height)` | 6 cubemap faces → equirect（`dr.texture(boundary_mode="cube")`） |

### `scripts/render_checkpoint.py`

新增 CLI 参数：
- `--render_equirect` — 透视模式下同时渲染 equirect 全景
- `--render_equirect_only` — 只渲染 equirect 跳过透视视图（省显存）
- `--cubemap_position X Y Z` — 相机位置（COLMAP 世界空间）
- `--face_res` — cubemap face 分辨率（默认 512）
- `--eq_width` / `--eq_height` — 输出 equirect 分辨率

## 坐标系约定与注意的坑

### `make_cubemap_camera` 的构造方式

Camera 的 `R` 被设为 `c2w_rot`（而不像数据集中是 w2c_rot），因为 `getWorld2View(R,T)` 内部做 `.T` 转置。但加上 `world_view_transform.transpose(0,1)` 与 CUDA 列主序读取的叠加，有效旋转矩阵为 `c2w_rot` 而非 `c2w_rot.T`。

当 `forward = ±X 或 ±Z` 时，`_c2w_rotation` 输出的 c2w 恰好对称（`c2w == c2w.T`），所以不会出错。  
当 `forward = ±Y` 时，c2w 不对称，导致 +Y/-Y face 渲染结果相对于 nvdiffrast cubemap 约定有 **180° 旋转**。

**补偿**：在 `render_equirect_from_position` 中，对 index 2（+Y）和 3（-Y）的渲染结果做 `torch.flip(dims=[1, 2])`（180° 旋转）。

### `cubemap_to_equirect` 的方向约定

方向向量的计算使用 **nvdiffrast/reflvec** 空间约定（+Y=up, +Z=forward），与 `dr.texture(boundary_mode="cube")` 的内部约定一致：
```python
y = torch.sin(lat)    # +Y=up（不是 COLMAP 的 +Y=down）
z = torch.cos(lat) * torch.cos(lon)
```

### 相关文件

- `scripts/render_cubemap_equirect.py` — 主实现
- `utils/graphics_utils.py` — `cubemap_to_equirect` / `latlong_to_cubemap_equirect`
- `scripts/render_checkpoint.py` — CLI 入口
