# 点光源重光照 — 设计文档

## 概述

在现有 IBL（CubemapLight）环境光重光照的基础上，为 RTR-GS 增加**点光源重光照**能力。支持单个/多个点光源的 PBR 着色、距离衰减、阴影，同时兼容 perspective 和 equirect 两种渲染模式。

## 架构

采用**独立叠加层**方案（方案 A），不影响现有 IBL 逻辑：

```
pbr_shading(light=cubemap, ...)   →   IBL render_rgb  (不变)
                                            ↓
point_light_shading(lights=[...]) →   point_rgb       (新增)
                                            ↓
final_rgb = render_rgb + point_rgb            (逐像素叠加)
```

## 模块设计

### 1. `PointLight` 数据类

**位置**：`pbr/light.py`（与 `CubemapLight` 同级）

```python
@dataclass
class PointLight:
    position: torch.Tensor  # [3] 世界坐标
    color: torch.Tensor     # [3] RGB 颜色（线性空间）
    intensity: float = 1.0  # 强度倍率
```

三个参数足够 MVP 使用。衰减统一用 `1/d²`（物理正确），不暴露配置项。

### 2. `point_light_shading()` 着色函数

**位置**：`pbr/shade.py`（与 `pbr_shading` 同级）

签名：
```python
def point_light_shading(
    lights: List[PointLight],
    points: torch.Tensor,      # [H, W, 3] 表面世界坐标
    normals: torch.Tensor,     # [H, W, 3] 法线
    view_dirs: torch.Tensor,   # [H, W, 3] 视线方向
    albedo: torch.Tensor,      # [H, W, 3] 漫反射颜色
    roughness: torch.Tensor,   # [H, W, 1]
    metallic: Optional[torch.Tensor],  # [H, W, 1] 或 None
    shadow_func: Optional[Callable] = None,
) -> torch.Tensor:             # [H, W, 3] 点光源总贡献
```

每个光源独立计算：
```
light_dir = normalize(light.pos - surface_pos)
distance  = |light.pos - surface_pos|
attenuation = 1 / max(distance², ε)
radiance = light.color * light.intensity * attenuation

half_dir = normalize(light_dir + view_dir)
NoL = saturate_dot(normal, light_dir)   # clamp to [0, 1]
NoV = saturate_dot(normal, view_dir)
HoV = saturate_dot(half_dir, view_dir)

# Cook-Torrance BRDF
NDF = GGX_NDF(normal, half_dir, roughness²)
G   = Smith_GGX(NoV, NoL, roughness)
F   = Schlick_Fresnel(HoV, F0)  # F0 = 0.04 (非金属) 或 lerp(0.04, albedo, metallic)
specular = NDF * G * F / (4 * NoV * NoL + ε)

kd = (1 - F) * (1 - metallic)
diffuse = kd * albedo / π

color += (diffuse + specular) * radiance * NoL

# 阴影衰减
if shadow_func and shadow_mask:
    color *= 1.0 - shadow_mask * 0.7
```

核心数学直接复用 GS-IR `shadow_map.py:light_pbr_shading()` 的已验证实现。

多个光源时求和即可，全部在一个 kernel 中完成（纯 PyTorch 张量运算，无需 CUDA 修改）。

### 3. 阴影模块

#### 透视模式

**位置**：`pbr/point_light_shadow.py`（新增）

复用 GS-IR `shadow_map.py:get_depth_cubemap()` 的逻辑：

```python
def get_depth_cubemap(gaussians, light_pos, res=2048):
    """
    从光源位置渲染 6 张深度图 (cubemap)
    使用 diff-gaussian-rasterization 的 lite_forward + argmax_depth
    viewer 中可用 `Z` 在 argmax_depth 与原 depth 聚合之间切换
    返回: [6, res, res, 1] 深度 cubemap
    """
    # 定义 6 个方向的相机（+X, -X, +Y, -Y, +Z, -Z）
    # 对每个方向：
    #   c2w = rotation; c2w[:3,3] = light_pos
    #   w2c = inverse(c2w); R = w2c[:3,:3].T; T = w2c[:3,3]
    #   构建 view_matrix / proj_matrix (90° FOV)
    #   调用 _C.lite_rasterize_gaussians(argmax_depth=True)
    #   收集深度图
```

对阴影查询（运行时逐像素）：
```python
def query_shadow_cubemap(depth_cubemap, surface_pos, light_pos, threshold=0.3):
    dir_to_light = normalize(light_pos - surface_pos)
    dist = |light_pos - surface_pos|
    closest = dr.texture(depth_cubemap, dir_to_light, boundary_mode="cube")
    return (dist - threshold > closest).float()
```

#### Equirect 模式

**位置**：`pbr/point_light_shadow.py`

```python
def get_depth_equirect(gaussians, light_pos, H=512, W=1024):
    """
    从光源位置渲染单张全景深度图
    使用 SGS GaussianRasterizer (camera_type=3, depth_mode=1)
    viewer 中可用 `Z` 在 min-depth 与 alpha-weighted depth 之间切换
    返回: [1, H, W] 深度图
    """
    # 创建 equirect 相机：identity 旋转 + light_pos 位置
    # SGS rasterizer 一次渲染覆盖 4π 球面
    # 返回最近贡献高斯的中心深度，避免 alpha-weighted depth 让 shadow map 偏远
```

对阴影查询：
```python
def query_shadow_equirect(depth_erp, surface_pos, light_pos, threshold=0.3):
    dir_to_light = normalize(light_pos - surface_pos)  # COLMAP space
    dist = |light_pos - surface_pos|
    
    # 方向 → equirect UV
    lat = asin(-dir_to_light.y)   # COLMAP: +Y down
    lon = atan2(dir_to_light.x, dir_to_light.z)
    u = (lon/π + 1) / 2
    v = (lat/(π/2) + 1) / 2
    
    closest = F.grid_sample(depth_erp, uv_grid, align_corners=False)
    return (dist - threshold > closest).float()
```

#### 深度图缓存

如果场景中光源位置固定，深度图只需在渲染开始时生成一次，后续复用：
```python
shadow_func = partial(query_shadow_cubemap, depth_cubemap_cache, light_pos)
```

如果光源持续移动（如视频），每帧重新生成。

### 4. 渲染管线集成

#### `render.py`（perspective 模式）

在 PBR 分支中，`pbr_shading()` 之后添加：
```python
# 在 render.py 约第 492 行之后
if point_lights is not None and len(point_lights) > 0:
    point_rgb = point_light_shading(
        lights=point_lights,
        points=surface_points,    # 从 depth map 重建
        normals=normal_map,
        view_dirs=view_dirs,
        albedo=base_color_map,
        roughness=roughness_map,
        metallic=metallic_map if pipe.metallic else None,
        shadow_func=shadow_func,
    )
    rendered_pbr = rendered_pbr + point_rgb
```

#### `render_equirect.py`（equirect 模式）

同样的模式，在 PBR 分支类似位置添加。光照计算逻辑完全一致，仅 `shadow_func` 不同（用 `query_shadow_equirect` 而非 `query_shadow_cubemap`）。

### 5. 重光照 Eval 脚本扩展

#### `eval_relighting_colmap.py`

新增参数：
```python
parser.add_argument("--point_lights_config", type=str, default=None,
    help="JSON 文件：定义点光源列表")
```

JSON 格式示例：
```json
{
    "lights": [
        {
            "position": [1.0, 2.0, 0.5],
            "color": [1.0, 0.8, 0.6],
            "intensity": 100.0
        },
        {
            "position": [-1.0, 0.5, 2.0],
            "color": [0.3, 0.6, 1.0],
            "intensity": 80.0
        }
    ]
}
```

加载后通过 `pbr_kwargs["point_lights"]` 传入 render function。

现有的 HDR 环境贴图重光照和点光源重光照可以同时使用（环境光提供整体照明，点光源提供局部高光/阴影效果）。

### 6. 新增/修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `pbr/light.py` | 修改 | 添加 `PointLight` 类 |
| `pbr/shade.py` | 修改 | 添加 `point_light_shading()` 函数 |
| `pbr/point_light_shadow.py` | **新增** | 阴影模块：`get_depth_cubemap()`, `get_depth_equirect()`, `query_*` |
| `gaussian_renderer/render.py` | 修改 | PBR 分支叠加点光源 |
| `gaussian_renderer/render_equirect.py` | 修改 | PBR 分支叠加点光源 |
| `eval_relighting_colmap.py` | 修改 | 支持 `--point_lights_config` 参数 |

不需要修改的文件：
- `baking.py` — 点光源不需要 occlusion baking（阴影用 shadow map 而非体素）
- 所有 CUDA 代码 — 不需要改，全部是 PyTorch 层面
- `CubemapLight` / `pbr_shading()` — 不需要改动

## 边界情况

1. **无光照像素**：背景区域（alpha=0）不参与点光源着色
2. **光源在场景外部**：距离可能很大，衰减后贡献接近 0，自然无影响
3. **极近距离**：`1/d²` 可能爆炸，加最小距离保护 `max(distance, ε)`
4. **全金属表面**：金属度 = 1 时 diffuse = 0，只有 specular 高光，物理正确
5. **多光源性能**：每个光源需计算一次 BRDF + 一次阴影查询。N 个光源 ≈ 2N 个 tensor ops，建议单次最多 ~8 个点光源有合理性能

## 未包含（未来可能的扩展）

- 光源 gizmo 交互式拖动（Editor 端功能）
- 光源面积/软阴影
- 光源可见性裁剪（通过 voxel occlusion baking 加速）
- 光源动画路径（逐帧光源移动）
