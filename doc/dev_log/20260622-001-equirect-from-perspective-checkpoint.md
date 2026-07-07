# Perspective 模型渲染 Equirect 全景图

## 需求

在 perspective（非 equirect）模式下训练的 checkpoint，给定一个世界空间位置，渲染 6 个 cubemap face 并拼接为 equirectangular panorama，以便和全景模式（SGS/ODGS）的训练结果直观对比。

## 方案

1. 在目标位置创建 6 个 Camera，分别朝 nvdiffrast 约定的 6 个方向（+X/-X/+Y/-Y/+Z/-Z），FOV=90°
2. 对每个 Camera 调用 perspective 的 `render_fn` 渲染图像
3. 将 6 张 face 图像合成为 equirect

## 文件 / 函数

### `scripts/render_cubemap_equirect.py`

| 函数 | 作用 |
|------|------|
| `_c2w_rotation(forward, up)` | GLM 风格 look-at → C2W 旋转矩阵 |
| `_make_canonical_rays(H, W, fov)` | 构造给定分辨率+视场角的 canonical rays |
| `make_cubemap_camera(position, forward, up, face_res, uid)` | 创建朝向指定方向的 Camera |
| `render_equirect_from_position(position, gaussians, pipe, render_fn, dict_params, ...)` | **主入口**：渲染 6 个 face 并拼接为 equirect |

### `utils/graphics_utils.py`

| 函数 | 作用 |
|------|------|
| `cubemap_to_equirect(cubemap_faces, eq_width, eq_height)` | 6 cubemap faces → equirect（`dr.texture(boundary_mode="cube")`）。输入 `[6, H, W, C]`，输出 `[C, eq_h, eq_w]`。`dr.texture` 天然支持任意 channel 数。 |

### `scripts/render_checkpoint.py` — CLI 入口

参数：
- `--render_equirect` — 透视模式下同时渲染 equirect 全景
- `--render_equirect_only` — 只渲染 equirect 跳过透视视图（省显存）
- `--cubemap_position X Y Z` — 相机位置（COLMAP 世界空间）
- `--face_res` — cubemap face 分辨率（默认 512）
- `--eq_width` / `--eq_height` — 输出 equirect 分辨率
- `--eq_yaw` — 偏航角（度），对 equirect 做水平循环移位。`yaw=180` 将 +X 置于画面中央。
- `--viewpoint` — Camera 对象，提取其 `camera_center` 作为渲染位置（与 `--cubemap_position` 二选一）

## 坐标系约定与注意的坑

### `make_cubemap_camera` 的构造方式

Camera 的 `R` 被设为 `c2w_rot`（而不像数据集中是 w2c_rot），因为 `getWorld2View(R,T)` 内部做 `.T` 转置。但加上 `world_view_transform.transpose(0,1)` 与 CUDA 列主序读取的叠加，有效旋转矩阵为 `c2w_rot` 而非 `c2w_rot.T`。

当 `forward = ±X 或 ±Z` 时，`_c2w_rotation` 输出的 c2w 恰好对称（`c2w == c2w.T`），所以不会出错。
当 `forward = ±Y` 时，c2w 不对称，导致 +Y/-Y face 渲染结果相对于 nvdiffrast cubemap 约定有 **180° 旋转**。

**补偿**：在 `render_equirect_from_position` 中，对 index 2（+Y）和 3（-Y）的渲染结果做 `torch.flip(dims=[1, 2])`。

### `cubemap_to_equirect` 的方向约定

方向向量的计算使用 **nvdiffrast/reflvec** 空间约定（+Y=up, +Z=forward），与 `dr.texture(boundary_mode="cube")` 的内部约定一致。

### 通道自动拼接

`render_equirect_from_position` 自动检测所有形状与 `render` 通道空间尺寸匹配的 tensor，统一拼接为 equirect。

检测规则：
1. 取 `face_results[0]["render"].shape[-2:]`（H, W）作为参考
2. **优先从 `vis_dict` 收集通道**（已做 gamma 校正、`*0.5+0.5`、背景混合等后处理，可直接保存）
3. **补充 pkg 层独有通道**（`render`、`pbr` 等不在 `vis_dict` 中的）

拼接时对 +Y/-Y face（index 2, 3）做 `torch.flip(dims=[1,2])` 补偿，与主渲染通道一致。

### `vis_dict` 数据源选择

`vis_dict` 中的 `base_color` 等通道已经做了 `linear2srgb` gamma 校正和背景混合，而 `out_feature_dict`（pkg 层）的 `base_color` 是线性 RGB。自动收集优先使用 `vis_dict` 版本，确保保存的图片与 `checkpoint_vis` 的可视化结果一致。

## 输出目录结构

```
equirect_from_perspective/
    render.png            # equirect 主渲染
    depth.png             # 深度图
    normal.png            # 法线图
    opacity.png
    base_color.png        # PBR 模式
    roughness.png
    metallic.png
    visibility.png
    ...
    cubemap/
        render.png        # 6 face 拼图
        depth.png
        normal.png
        ...
```

## 相关讨论

- `env_export_base` 和 `env_export_diffuse` 因分辨率与 rendered face 不一致，不参与多通道 equirect。
- `vis_dict` 只在 `is_training=False` 时存在。训练时的可视化（`train.py` 的 `save_vis_images`）直接从 pkg 层取数据手动后处理。
