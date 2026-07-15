# Point Light Relighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add point light relighting to RTR-GS — PointLight class, PBR shading, shadow mapping, integration into perspective/equirect renderers and eval scripts.

**Architecture:** Independent overlay approach — `point_light_shading()` runs after existing `pbr_shading()`; their results are summed pixel-wise. Shadow maps use `argmax_depth` via `diff_gaussian_rasterization` (perspective) or equirect depth panorama via SGS `GaussianRasterizer` (equirect).

**Tech Stack:** PyTorch, nvdiffrast, SGS GaussianRasterizer, diff-gaussian-rasterization

## Global Constraints

- All new code goes in `pbr/` — no CUDA modifications needed
- `PointLight` in `pbr/light.py` alongside `CubemapLight`
- `point_light_shading()` in `pbr/shade.py` alongside `pbr_shading()`
- Shadow functions in new file `pbr/point_light_shadow.py`
- Integration in `render.py` / `render_equirect.py` via `dict_params["point_lights"]`
- Eval script adds `--point_lights_config` JSON argument

---

### Task 1: PointLight dataclass + point_light_shading()

**Files:**
- Modify: `pbr/light.py`
- Modify: `pbr/shade.py`

**Interfaces:**
- Consumes: numpy/torch types for light params; albedo/roughness/metallic tensors from rasterizer
- Produces: `PointLight` dataclass, `point_light_shading()` function

- [ ] **Step 1.1: Add PointLight to pbr/light.py**

Append after `CubemapLight` class:

```python
@dataclass
class PointLight:
    """点光源：位置 + 颜色 + 强度"""
    position: torch.Tensor  # [3] 世界坐标
    color: torch.Tensor     # [3] RGB 线性颜色
    intensity: float = 1.0
```

Add import at top:
```python
from dataclasses import dataclass
```

- [ ] **Step 1.2: Add point_light_shading() to pbr/shade.py**

Add after `pbr_shading()` function:

```python
def point_light_shading(
    lights: List[PointLight],
    points: torch.Tensor,       # [H, W, 3] 表面世界坐标
    normals: torch.Tensor,      # [H, W, 3]
    view_dirs: torch.Tensor,    # [H, W, 3]
    albedo: torch.Tensor,       # [H, W, 3]
    roughness: torch.Tensor,    # [H, W, 1]
    metallic: Optional[torch.Tensor] = None,  # [H, W, 1]
    shadow_funcs: Optional[List[Callable]] = None,  # 每个光源对应的阴影函数, None=无阴影
) -> torch.Tensor:              # [H, W, 3] 所有点光源的累计贡献
    """
    点光源 PBR 着色。对每个光源独立计算 Cook-Torrance BRDF + 距离衰减，
    求和后返回。

    Args:
        lights: 点光源列表
        points: 表面 3D 位置 [H, W, 3]
        normals: 表面法线 [H, W, 3]
        view_dirs: 视图方向（表面→相机）[H, W, 3]
        albedo: 基础颜色 [H, W, 3]
        roughness: 粗糙度 [H, W, 1]
        metallic: 金属度 [H, W, 1] 或 None
        shadow_funcs: 每个光源对应的阴影查询函数, callable(points) -> [H, W, 1]
                      阴影函数创建时已绑定对应的 depth_map 和 light_pos。
                      None 或长度不足时对应光源无阴影。
    Returns:
        combined_rgb: [H, W, 3] 累加后的点光源颜色
    """
    H, W, _ = points.shape
    device = points.device
    combined_rgb = torch.zeros(H, W, 3, device=device)

    for light in lights:
        # 光源方向 & 距离衰减
        to_light = light.position[None, None, :] - points  # [H, W, 3]
        light_dir = F.normalize(to_light, dim=-1)
        distance = torch.norm(to_light, dim=-1, keepdim=True).clamp(min=1e-4)
        attenuation = 1.0 / (distance * distance)
        radiance = light.color[None, None, :] * light.intensity * attenuation  # [H, W, 3]

        # NoL
        NoL = saturate_dot(normals, light_dir)  # [H, W, 1]

        # Half vector
        half_dir = F.normalize(light_dir + view_dirs, dim=-1)
        HoV = saturate_dot(half_dir, view_dirs)  # [H, W, 1]
        NoH = saturate_dot(normals, half_dir)    # [H, W, 1]
        NoV = saturate_dot(normals, view_dirs)   # [H, W, 1]

        # F0
        if metallic is None:
            F0 = torch.ones_like(albedo) * 0.04
        else:
            F0 = (1.0 - metallic) * 0.04 + albedo * metallic

        # Cook-Torrance BRDF
        alpha = roughness * roughness
        alpha2 = alpha * alpha

        # NDF (GGX)
        NoH2 = NoH * NoH
        denom = (NoH2 * (alpha2 - 1.0) + 1.0)
        NDF = alpha2 / (torch.pi * denom * denom + 1e-6)

        # Geometry (Smith GGX)
        k = (roughness + 1.0).pow(2) / 8.0
        G_vis = NoV / (NoV * (1.0 - k) + k + 1e-6)
        G_light = NoL / (NoL * (1.0 - k) + k + 1e-6)
        G = G_vis * G_light

        # Fresnel (Schlick)
        F = F0 + (1.0 - F0) * (1.0 - HoV).pow(5)

        # Specular
        specular = NDF * G * F / (4.0 * NoV * NoL + 1e-6)

        # Diffuse
        kd = (1.0 - F) * (1.0 - metallic) if metallic is not None else (1.0 - F)
        diffuse = kd * albedo / torch.pi

        # Combine
        rgb = (diffuse + specular) * radiance * NoL

        # Shadow（每个光源独立查询，阴影函数已绑定对应的 depth_map 和 light_pos）
        if shadow_funcs is not None and i < len(shadow_funcs) and shadow_funcs[i] is not None:
            shadow = shadow_funcs[i](points)
            rgb = rgb * (1.0 - shadow * 0.7)

        combined_rgb = combined_rgb + rgb

    return combined_rgb
```

Add imports at top of `pbr/shade.py`:
```python
from typing import Callable, List, Optional
from .light import PointLight
```

- [ ] **Step 1.3: Update pbr/__init__.py**

```python
from .light import CubemapLight, PointLight
from .shade import get_brdf_lut, pbr_shading, point_light_shading, saturate_dot

__all__ = ["CubemapLight", "PointLight", "get_brdf_lut", "pbr_shading", "point_light_shading", "saturate_dot"]
```

- [ ] **Step 1.4: Verify**

```bash
cd /home/huangpengyue/projects/RTR-GS
python -c "from pbr import PointLight, point_light_shading; print('OK')"
```
Expected: `OK`

- [ ] **Step 1.5: Commit**

```bash
git add pbr/light.py pbr/shade.py pbr/__init__.py
git commit -m "feat: add PointLight and point_light_shading"
```

---

### Task 2: Shadow module — pbr/point_light_shadow.py

**Files:**
- Create: `pbr/point_light_shadow.py`

**Interfaces:**
- Produces: `get_depth_cubemap()`, `get_depth_equirect()`, `make_shadow_func_cubemap()`, `make_shadow_func_equirect()`

- [ ] **Step 2.1: Write shadow module**

```python
"""
点光源阴影模块。

透视模式：从光源位置渲染 6 张深度图（cubemap），用 nvdiffrast 查询。
Equirect 模式：从光源位置渲染单张全景深度图，用方向→UV 映射查询。

参考 GS-IR/shadow_map.py:get_depth_cubemap()
"""
from typing import Callable, List, Tuple

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import _C as diff_C
from scene.gaussian_model import GaussianModel
from spherical_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix


def _getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """GS-IR 中的世界→视图矩阵，与 COLMAP 约定一致"""
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


# 6 个 cubemap 面的旋转矩阵（与 GS-IR 一致）
CUBEMAP_ROTATIONS = [
    # +X: lookAt(eye=[0,0,0], center=[-1,0,0], up=[0,-1,0])
    torch.tensor([[0,0,1,0],[0,-1,0,0],[-1,0,0,0],[0,0,0,1]], dtype=torch.float32),
    # -X: lookAt(eye=[0,0,0], center=[1,0,0], up=[0,-1,0])
    torch.tensor([[0,0,-1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]], dtype=torch.float32),
    # +Y: lookAt(eye=[0,0,0], center=[0,-1,0], up=[0,0,-1])
    torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=torch.float32),
    # -Y: lookAt(eye=[0,0,0], center=[0,1,0], up=[0,0,1])
    torch.tensor([[1,0,0,0],[0,0,-1,0],[0,-1,0,0],[0,0,0,1]], dtype=torch.float32),
    # +Z: lookAt(eye=[0,0,0], center=[0,0,-1], up=[0,1,0])
    torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=torch.float32),
    # -Z: lookAt(eye=[0,0,0], center=[0,0,1], up=[0,-1,0])
    torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=torch.float32),
]


def get_depth_cubemap(
    gaussians: GaussianModel,
    light_pos: torch.Tensor,
    res: int = 512,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> torch.Tensor:
    """
    从光源位置渲染 6 张深度图组成 cubemap。

    使用 diff_gaussian_rasterization._C.lite_rasterize_gaussians
    配合 argmax_depth=True 保证深度来自最近（贡献最大）的高斯。

    Args:
        gaussians: 场景高斯模型
        light_pos: 光源位置 [3]
        res: 每面的分辨率（默认 512）
        znear/zfar: 近/远裁面
    Returns:
        depth_cubemap: [6, res, res, 1] 深度 cubemap
    """
    bg = torch.zeros(3, device="cuda")
    proj_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=torch.pi * 0.5, fovY=torch.pi * 0.5
    ).transpose(0, 1).cuda()

    means3D = gaussians.get_xyz
    colors_precomp = torch.tensor([], device="cuda")  # not used

    depth_faces = []
    for rot in CUBEMAP_ROTATIONS:
        rot = rot.cuda()
        c2w = rot.clone()
        c2w[:3, 3] = light_pos
        w2c = torch.inverse(c2w)
        T = w2c[:3, 3]
        R = w2c[:3, :3].T
        world_view_transform = _getWorld2ViewTorch(R, T).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            proj_matrix.unsqueeze(0))).squeeze(0)
        cam_center = world_view_transform.inverse()[3, :3]

        args = (
            bg,
            means3D,
            colors_precomp,
            gaussians.get_opacity,
            gaussians.get_scaling,
            gaussians.get_rotation,
            colors_precomp,  # cov3D_precomp
            world_view_transform,
            full_proj_transform,
            cam_center,
            1.0,              # tanfovx
            1.0,              # tanfovy
            1.0,              # scale_modifier
            res,              # height
            res,              # width
            gaussians.active_sh_degree,
            False,            # prefiltered
            True,             # argmax_depth ← 关键！
        )
        _, _, _, _, depth_map = diff_C.lite_rasterize_gaussians(*args)
        depth_faces.append(depth_map.permute(1, 2, 0))  # [res, res, 1]

    depth_cubemap = torch.stack(depth_faces, dim=0)  # [6, res, res, 1]
    depth_cubemap[depth_cubemap == 0] = depth_cubemap.max()
    return depth_cubemap


def get_depth_equirect(
    gaussians: GaussianModel,
    light_pos: torch.Tensor,
    H: int = 512,
    W: int = 1024,
) -> torch.Tensor:
    """
    从光源位置渲染单张全景深度图（4π 球面覆盖）。

    使用 SGS GaussianRasterizer (camera_type=3)，适配 SGS 训练的点云。

    Args:
        gaussians: 场景高斯模型
        light_pos: 光源位置 [3]
        H, W: 全景图分辨率
    Returns:
        depth_erp: [1, H, W] 全景深度图
    """
    c2w = torch.eye(4, device="cuda")
    c2w[:3, 3] = light_pos
    w2c = torch.inverse(c2w)
    viewmatrix = w2c.T

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=0.0,
        tanfovy=0.0,
        bg=torch.zeros(3, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=viewmatrix,
        sh_degree=gaussians.active_sh_degree,
        campos=light_pos,
        prefiltered=False,
        debug=False,
        camera_type=3,
        render_depth=True,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    _, _, _, depth_raw, _, _ = rasterizer(
        means3D=gaussians.get_xyz,
        means2D=torch.zeros_like(gaussians.get_xyz[:, :2], requires_grad=True, device="cuda"),
        shs=gaussians.get_shs,
        colors_precomp=None,
        extra_features=None,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
    )
    return depth_raw  # [1, H, W]


def make_shadow_func_cubemap(
    depth_cubemap: torch.Tensor,
    light_pos: torch.Tensor,
    threshold: float = 0.3,
) -> Callable:
    """
    创建 cubemap 模式的阴影查询函数。
    light_pos 和 depth_cubemap 已在闭包中绑定。

    Returns:
        shadow_func(points) -> [H, W, 1], 1=阴影中
    """
    depth_cubemap = depth_cubemap[None, ...]  # [1, 6, res, res, 1]

    def shadow_func(points: torch.Tensor) -> torch.Tensor:
        """逐像素阴影查询"""
        H, W, _ = points.shape
        dir_to_light = F.normalize(light_pos[None, None, :] - points, dim=-1)
        dist = torch.norm(light_pos[None, None, :] - points, dim=-1, keepdim=True)

        closest_depth = dr.texture(
            depth_cubemap,
            dir_to_light.contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[0, ...]  # [H, W, 1]

        return (dist - threshold > closest_depth).float()

    return shadow_func


def make_shadow_func_equirect(
    depth_erp: torch.Tensor,
    light_pos: torch.Tensor,
    threshold: float = 0.3,
) -> Callable:
    """
    创建 equirect 模式的阴影查询函数。
    方向向量 → (lat, lon) → equirect UV → grid_sample 查深度。
    """
    # 预计算 equirect UV 网格（用于 grid_sample）
    H, W = depth_erp.shape[1:]
    theta = torch.linspace(-torch.pi, torch.pi, W, device=depth_erp.device)
    phi = torch.linspace(torch.pi / 2, -torch.pi / 2, H, device=depth_erp.device)

    def shadow_func(points: torch.Tensor) -> torch.Tensor:
        """逐像素阴影查询（equirect）"""
        H, W, _ = points.shape
        dir_to_light = F.normalize(light_pos[None, None, :] - points, dim=-1)
        dist = torch.norm(light_pos[None, None, :] - points, dim=-1, keepdim=True)

        # 方向 → equirect UV（COLMAP space: +Y down, +Z forward）
        # lat = asin(-y), lon = atan2(x, z)
        lat = torch.asin((-dir_to_light[..., 1]).clamp(-1.0, 1.0))
        lon = torch.atan2(dir_to_light[..., 0], dir_to_light[..., 2])

        # 纬度: lat∈[-π/2,π/2] → v∈[0,1]（图像从上到下）
        # 经度: lon∈[-π,π]   → u∈[0,1]（图像从左到右）
        v = (lat / (torch.pi / 2) + 1.0) * 0.5  # [H, W]
        u = (lon / torch.pi + 1.0) * 0.5         # [H, W]

        # grid_sample 需要 [-1, 1] 范围
        grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        closest_depth = F.grid_sample(
            depth_erp.unsqueeze(0),   # [1, 1, H, W]
            grid,
            mode="bilinear",
            align_corners=False,
        )  # [1, 1, H, W]

        return (dist - threshold > closest_depth.squeeze(0).permute(1, 2, 0)).float()

    return shadow_func
```

- [ ] **Step 2.2: Verify imports**

```bash
cd /home/huangpengyue/projects/RTR-GS
python -c "from pbr.point_light_shadow import get_depth_cubemap, get_depth_equirect; print('OK')"
```
Expected: `OK`

- [ ] **Step 2.3: Commit**

```bash
git add pbr/point_light_shadow.py
git commit -m "feat: add point light shadow module (cubemap + equirect)"
```

---

### Task 3: Integrate into perspective renderer

**Files:**
- Modify: `gaussian_renderer/render.py`

**Interfaces:**
- Consumes: `dict_params["point_lights"]` (List[PointLight]), `dict_params["point_light_shadow"]` (optional pre-baked shadow func)
- Produces: Rendered images with point light contribution overlaid

- [ ] **Step 3.1: Add point light pass after pbr_shading in render.py**

Find the section around line 514 (`specular_pbr = pbr_result["specular_rgb"]`) and after it add:

```python
# ── Point light overlay ──────────────────────────────────────────────
point_lights = dict_params.get("point_lights", None) if dict_params else None
if point_lights and len(point_lights) > 0:
    # Surface positions: from rasterizer output [3, H, W] → [H, W, 3]
    surf_points = rendered_surface_xyz.permute(1, 2, 0)  # [H, W, 3]
    # Only shade foreground pixels (where opacity > 0)
    point_rgb = point_light_shading(
        lights=point_lights,
        points=surf_points,
        normals=normal_map,
        view_dirs=view_dirs,
        albedo=base_color_map,
        roughness=roughness_map,
        metallic=metallic_map if pipe.metallic else None,
        shadow_funcs=dict_params.get("point_light_shadow_funcs", None),
    )
    # Blend with opacity (same pattern as PBR above)
    point_rgb = point_rgb * opacity_map
    rendered_pbr = rendered_pbr + point_rgb
```

Add imports at top of `render.py`:
```python
from pbr import point_light_shading
```

- [ ] **Step 3.2: Verify import**

```bash
cd /home/huangpengyue/projects/RTR-GS
python -c "from gaussian_renderer.render import render; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 3.3: Commit**

```bash
git add gaussian_renderer/render.py
git commit -m "feat: integrate point light overlay into perspective renderer"
```

---

### Task 4: Integrate into equirect renderer

**Files:**
- Modify: `gaussian_renderer/render_equirect.py`

**Interfaces:**
- Consumes: Same `dict_params` as perspective; surface points reconstructed from depth + view_dirs

- [ ] **Step 4.1: Add point light pass after pbr_shading in render_equirect.py**

Find the section around line 514 (`specular_pbr = pbr_result["specular_rgb"]`) and add after it:

```python
# ── Point light overlay ──────────────────────────────────────────────
point_lights = dict_params.get("point_lights", None) if dict_params else None
if point_lights and len(point_lights) > 0:
    # Surface positions already computed above for occlusion (variable `points`)
    # points shape: [HW, 3] — need to reshape to [H, W, 3]
    surf_points = points.reshape(H, W, 3)
    point_rgb = point_light_shading(
        lights=point_lights,
        points=surf_points,
        normals=normal_map,
        view_dirs=view_dirs,
        albedo=base_color_map,
        roughness=roughness_map,
        metallic=metallic_map if pipe.metallic else None,
        shadow_funcs=dict_params.get("point_light_shadow_funcs", None),
    )
    # Blend with opacity
    point_rgb = point_rgb * opacity_map
    rendered_pbr = rendered_pbr + point_rgb
```

Add import at top:
```python
from pbr import point_light_shading
```

- [ ] **Step 4.2: Verify import**

```bash
cd /home/huangpengyue/projects/RTR-GS
python -c "from gaussian_renderer.render_equirect import render_equirect; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 4.3: Commit**

```bash
git add gaussian_renderer/render_equirect.py
git commit -m "feat: integrate point light overlay into equirect renderer"
```

---

### Task 5: Extend eval_relighting_colmap.py

**Files:**
- Modify: `eval_relighting_colmap.py`

**Interfaces:**
- Consumes: `--point_lights_config` CLI arg → JSON → List[PointLight]
- Produces: Relit images with point light effects

- [ ] **Step 5.1: Add CLI argument**

Find the ArgumentParser section and add:

```python
parser.add_argument("--point_lights_config", type=str, default=None,
    help="点光源 JSON 配置文件路径")
```

- [ ] **Step 5.2: Add point light loading helper**

After the imports, add:

```python
def load_point_lights(config_path: str) -> List[PointLight]:
    """从 JSON 加载点光源列表。
    JSON 格式: { "lights": [{"position":[x,y,z], "color":[r,g,b], "intensity":f}, ...] }
    """
    import json
    with open(config_path) as f:
        data = json.load(f)
    lights = []
    for item in data["lights"]:
        pos = torch.tensor(item["position"], dtype=torch.float32, device="cuda")
        col = torch.tensor(item["color"], dtype=torch.float32, device="cuda")
        lights.append(PointLight(position=pos, color=col, intensity=item.get("intensity", 1.0)))
    return lights
```

Add import:
```python
from pbr import PointLight, point_light_shading
from pbr.point_light_shadow import get_depth_cubemap, get_depth_equirect, make_shadow_func_cubemap, make_shadow_func_equirect
```

- [ ] **Step 5.3: Wire point lights into pbr_kwargs**

In `training()` function, after building pbr_kwargs (around line 103), add:

```python
# Point lights
if args.point_lights_config:
    point_lights = load_point_lights(args.point_lights_config)
    pbr_kwargs["point_lights"] = point_lights
    print(f"[point light] Loaded {len(point_lights)} point light(s) from {args.point_lights_config}")

    # Pre-bake shadow maps for fixed lights（每个光源独立渲染深度图）
    is_equirect_mode = hasattr(args, 'equirect_width') and args.equirect_width is not None
    shadow_funcs = []
    for i, light in enumerate(point_lights):
        if is_equirect_mode:
            depth_map = get_depth_equirect(gaussians, light.position)
            shadow_fn = make_shadow_func_equirect(depth_map, light.position)
        else:
            depth_map = get_depth_cubemap(gaussians, light.position)
            shadow_fn = make_shadow_func_cubemap(depth_map, light.position)
        shadow_funcs.append(shadow_fn)
        print(f"  [shadow] Light {i}: {'equirect' if is_equirect_mode else 'cubemap'} depth map ready")
    pbr_kwargs["point_light_shadow_funcs"] = shadow_funcs
```

- [ ] **Step 5.4: Verify**

```bash
cd /home/huangpengyue/projects/RTR-GS
python -c "from eval_relighting_colmap import load_point_lights; print('OK')"
```
Expected: `OK`

- [ ] **Step 5.5: Commit**

```bash
git add eval_relighting_colmap.py
git commit -m "feat: add point light support to relighting eval script"
```

---

### Task 6: Integration test

- [ ] **Step 6.1: Create test point light config**

Create test config at `test_data/test_point_light.json`:
```json
{
    "lights": [
        {
            "position": [0.0, 2.0, 0.0],
            "color": [1.0, 1.0, 1.0],
            "intensity": 50.0
        }
    ]
}
```

- [ ] **Step 6.2: Run a quick test with an existing checkpoint**

```bash
cd /home/huangpengyue/projects/RTR-GS

conda activate odgs-rtr

# Quick smoke test: render one frame with point light
python eval_relighting_colmap.py \
    -m lab_output/.../... \
    --checkpoint lab_output/.../chkpnt30000.pth \
    --point_lights_config test_data/test_point_light.json \
    --eval
```

Command will vary based on actual checkpoint path. Expected: runs without error, produces images with visible point light contribution.

- [ ] **Step 6.3: Verify output images**

Check `{model_path}/test_rli/{task_name}/` for point-light-enhanced output.

- [ ] **Step 6.4: Commit test config**

```bash
git add test_data/test_point_light.json
git commit -m "test: add point light test config"
```
