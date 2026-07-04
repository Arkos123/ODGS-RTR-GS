import os
import json
import itertools
import math
from argparse import ArgumentParser
from os import makedirs
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from tqdm import trange
from diff_gaussian_rasterization import _C
from gs_ir import _C as gs_ir_ext

from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import components_from_spherical_harmonics, eval_sh
from spherical_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

"""
TODO:
- 阅读 baking 源码流程。
- 确认哪里涉及坐标系的问题
"""

# =================================================================
# 坐标系约定（baking.py 涉及三个坐标系，混用是常见 bug 来源）
# =================================================================
# ┌─────────────────────────┬──────────┬──────────┬──────────┬──────────────────────────────────┐
# │ 空间                    │ +X       │ +Y       │ +Z       │ 使用场景                         │
# ├─────────────────────────┼──────────┼──────────┼──────────┼──────────────────────────────────┤
# │ COLMAP world space      │ 右       │ **下**  │ **前**  │ gaussians.get_xyz, scene_min/max │
# │ reflvec / cubemap space │ 右       │ 上       │ **-前**  │ envmap_dirs, nvdiffrast sampling │
# │ Equirect view space     │ 右       │ 上       │ 前       │ _equirect_ray_dirs（自身定义）   │
# └─────────────────────────┴──────────┴──────────┴──────────┴──────────────────────────────────┘
#
# reflvec (nvdiffrast cubemap) 空间（+Y 上、-Z 前）：
#    +y
#    |  -z前
#  __|/___ +x
#   /|
#  +z
#
# COLMAP 世界空间（+Y 下、+Z 前）
#    上
#    | +z 前
#  __|/___ +x 右
#   /|
#    +y
#
# Equirect 空间：纬度 θ∈[-π/2,π/2]，经度 φ∈[-π,π]
#    +Y 向上，北极 θ=+π/2，南极 θ=-π/2（与 get_envmap_dirs 的 θ 方向相反）
#    SGS rasterizer 的 point3ToLonlatScreen 使用此约定
#
# 关键转换：
#   COLMAP → reflvec:  diag(1, -1, -1)     （flip Y 和 Z）
#   COLMAP → equirect: diag(1, -1,  1)     （flip Y）
#   reflvec → COLMAP:  diag(1, -1, -1)     （同上，自逆）
#
# 在烘焙中使用场景：
#   - envmap_dirs() 返回的方向在 reflvec 空间
#   - 计算 hit_pos 时需转 COLMAP world space：world_dirs = envmap_dirs * diag(1,-1,-1)
#   - 墙壁检测（skip_walls）的 scene_min/max 在 COLMAP space
#   - 法线可视化 vis_walls 中 equi_dirs 需通过 diag(1,-1,1) 转 COLMAP view space
# =================================================================


# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res: List[int] = [256, 512]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算环境贴图采样方向和立体角

    reflvec (nvdiffrast cubemap) 空间（+Y 上、-Z 前）：
    θ: 纬度，范围 [0, π]，北极(+y)→南极(-y)
    φ: 经度，范围 [-π, π]，-π(+z)→-π/2(-x) → 0（-z前）→π/2(+x)→π(+z)
    +y
    | -z 前
 ___|/___ +x
   /|
 +z

    返回：
      solid_angles: [H, W, 1] 每个像素对应的立体角（用于球面积分加权）
      reflvec:      [H, W, 3] 世界空间采样方向
    """

    # 生成 [res[0], res[1]] （默认 [256, 512] ）大小的二维网格。
    # gy[i,j] 和 gx[i,j] 是位置 (i,j) 的 (θ,φ) 坐标
    # 其中 θ∈[0,π]，φ∈[-π,π]
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )
    d_theta, d_phi = np.pi / res[0], 2 * np.pi / res[1]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    # reflvec = (sinθ·sinφ, cosθ, -sinθ·cosφ)  即 nvdiffrast cubemap 约定
    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    # 计算立体角权重：dΩ = (cosθ - cos(θ+dθ)) * dφ
    solid_angles = ((costheta - torch.cos(gy * np.pi + d_theta)) * d_phi)[..., None]  # [H, W, 1]
    # 检查立体角总和是否为 4π
    print(f"solid_angles_sum error: {solid_angles.sum() - 4 * np.pi}")

    return solid_angles, reflvec


def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> torch.Tensor:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs.cuda()


MIN_DEPTH = 1e-6


def _equirect_ray_dirs(H: int, W: int, device: str = "cuda") -> torch.Tensor:
    """等距柱面投影(ERP)像素 → 世界空间射线方向 [H, W, 3]

    行序：row 0 = lat=+π/2（北极），使用 +sin(lat) 作为 Y（等距空间 +Y 向上），
    与 SGS rasterizer 的 point3ToLonlatScreen 的行序匹配。
    """
    ys = torch.linspace(0.5 * math.pi, -0.5 * math.pi, H, device=device)
    xs = torch.linspace(-math.pi, math.pi, W, device=device)
    lat, lon = torch.meshgrid(ys, xs, indexing="ij")
    # dir = (sin(lon)*cos(lat), sin(lat), cos(lon)*cos(lat))
    # 注意：这里 +Y 向上（等距空间），需通过 diag(1,-1,1) 转到 COLMAP 空间
    return torch.stack([
        torch.sin(lon) * torch.cos(lat),
        torch.sin(lat),
        torch.cos(lon) * torch.cos(lat),
    ], dim=-1)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--bound", default=1.5, type=float, help="The bound of occlusion volumes.")
    parser.add_argument("--valid", default=1.5, type=float, help="Identify valid area (cull invalid 3D Gaussians) to accelerate baking.")
    parser.add_argument("--occlu_res", default=160, type=int, help="The resolution of the baked occlusion volumes.")
    parser.add_argument("--cubemap_res", default=256, type=int, help="The resolution of the cubemap produced during baking.")
    parser.add_argument("--occlusion", default=0.4, type=float, help="The occlusion threshold to control visible area, the smaller the bound, the lighter the ambient occlusion.")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--skip_walls", action="store_true", default=False, help="Skip wall surfaces during occlusion baking: treat surfaces near the scene boundary as unoccluded (useful for sealed indoor scenes).")
    parser.add_argument("--wall_margin", type=str, default="0.3", help="Distance threshold(s) for wall detection. Single float (e.g. 0.3) applies to all 6 faces. 6 comma-separated values (e.g. 0.1,0.1,0.3,0.2,0.1,0.1) for min_x,max_x,min_y,max_y,min_z,max_z.")
    parser.add_argument("--vis_walls", action="store_true", default=False, help="Visualize wall detection: save wall/non-wall Gaussians as PLY files and exit (requires --skip_walls).")
    parser.add_argument("--extent_percentile", type=float, default=0.01, help="Percentile (0~1) used to compute robust scene extent from Gaussian positions. E.g. 0.01 means 1st/99th percentile. Used with --skip_walls and --auto_bound.")
    parser.add_argument("--auto_bound", action="store_true", default=False, help="Automatically compute occlusion volume bound from scene extent (with --bound_padding margin), overriding --bound.")
    parser.add_argument("--bound_padding", type=float, default=1.1, help="Padding factor for --auto_bound. E.g. 1.1 means 10%% margin beyond scene extent.")
    # already defined by PipelineParams class
    # parser.add_argument("--equirect", action="store_true", default=False, help="Use SGS equirect rasterizer (camera_type=3) instead of 6-face cubemap. Requires the equirect GaussianModel checkpoint.")  
    parser.add_argument("--equirect_res", type=int, default=128, help="Equirect render height (width = 2 * height). Only used with --equirect.")
    parser.add_argument("--occlusion_path", type=str, default=None, help="保存occlusion volumes的路径。默认在checkpoint目录下.")
    args = get_combined_args(parser)

    # Parse wall_margin: single float → 6 identical values, or "a,b,c,d,e,f"
    raw = str(args.wall_margin)
    parts = [float(p.strip()) for p in raw.split(",")]
    if len(parts) == 1:
        parts = parts * 6
    elif len(parts) != 6:
        parser.error("--wall_margin must be a single float or 6 comma-separated values (min_x,max_x,min_y,max_y,min_z,max_z)")
    wall_margins = torch.tensor(parts, device="cuda")  # [6]

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Save command-line args to checkpoint directory
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "args_baking.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Saved baking args to {os.path.join(model_path, 'args_baking.json')}")

    # dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    gaussians = GaussianModel(4)

    # checkpoint = torch.load(args.checkpoint)
    # if isinstance(checkpoint, Tuple):
    #     model_params = checkpoint[0]
    # elif isinstance(checkpoint, Dict):
    #     model_params = checkpoint["gaussians"]
    # else:
    #     raise TypeError
    # gaussians.restore(model_params)
    gaussians.create_from_ckpt(args.checkpoint)

    # Set up rasterization configuration
    res = args.cubemap_res
    bg_color = torch.ones([3, res, res], device="cuda")
    # # NOTE: for debuging HDRi
    bg_colors = [
        torch.zeros([3, res, res], device="cuda"),  # black
        torch.zeros([3, res, res], device="cuda"),  # red
        torch.zeros([3, res, res], device="cuda"),  # green
        torch.zeros([3, res, res], device="cuda"),  # blue
        torch.zeros([3, res, res], device="cuda"),  # yellow
        torch.ones([3, res, res], device="cuda"),  # white
    ]
    # 1-red
    bg_colors[1][0, ...] = 1
    # 2-green
    bg_colors[2][1, ...] = 1
    # 3-blue
    bg_colors[3][2, ...] = 1
    # 4-yellow
    bg_colors[4][:2, ...] = 1

    # NOTE: capture 6 views with fov=90

    """
    rotations[6] = nvdiffrast cubemap 六面 (+X,-X,+Y,-Y,+Z,-Z) 的 c2w 旋转矩阵（仅旋转部分）。
    配合 rasterizer 的 +Z forward 约定使用：R^T @ face_forward_direction → [0,0,+Z]。
    第 4 列留空，由调用处填入 position。
    """
    rotations: List[torch.Tensor] = [
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(), 
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(), 
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(), 
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(), 
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(), 
    ]

    zfar = 100.0
    znear = 0.01
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=math.pi * 0.5, fovY=math.pi * 0.5)
        .transpose(0, 1)
        .cuda()
    )

    # compute scene extent from Gaussian positions (used by --auto_bound and --skip_walls)
    if args.auto_bound or args.skip_walls:
        p = args.extent_percentile
        scene_min = torch.quantile(gaussians.get_xyz, p, dim=0)
        scene_max = torch.quantile(gaussians.get_xyz, 1.0 - p, dim=0)
        print(f"Scene extent ({p*100:.1f}th-{(1.0-p)*100:.1f}th percentile): min={scene_min.detach().cpu().numpy()}, max={scene_max.detach().cpu().numpy()}")
        print(f"  (raw min/max: min={gaussians.get_xyz.min(dim=0).values.detach().cpu().numpy()}, max={gaussians.get_xyz.max(dim=0).values.detach().cpu().numpy()})")

    if args.auto_bound:
        scene_extent = max(scene_max.max().item(), (-scene_min).max().item())
        pad = scene_extent * (args.bound_padding - 1.0) / 2.0

        # Non-symmetric AABB with per-axis padding
        auto_aabb_min = scene_min - pad
        auto_aabb_max = scene_max + pad

        # Backward-compat symmetric bound = max half-extent
        sym_bound = max(auto_aabb_max.max().item(), (-auto_aabb_min).max().item())
        print(f"[auto_bound] scene_extent={scene_extent:.3f}, aabb_min={auto_aabb_min.detach().cpu().numpy()}, aabb_max={auto_aabb_max.detach().cpu().numpy()}, bound(sym)={sym_bound:.3f}")
        args.bound = sym_bound
        args.valid = sym_bound

    # Create voxel grid
    if args.auto_bound:
        aabb_min = auto_aabb_min.clone().cuda()
        aabb_max = auto_aabb_max.clone().cuda()
    else:
        aabb_min = torch.tensor([-args.bound] * 3).cuda()
        aabb_max = torch.tensor([args.bound] * 3).cuda()

    prods = list(itertools.product(range(args.occlu_res), range(args.occlu_res), range(args.occlu_res)))
    grid = (aabb_max - aabb_min) / (args.occlu_res - 1)
    positions = torch.tensor(prods).cuda() * grid + aabb_min  # [bs, 3]

    # =================================================================
    # SECTION 体素遮挡网格初始化
    # =================================================================
    # 创建均匀 3D 体素网格，每个体素存储可见性的 SH 系数。
    # 通过从每个体素中心向场景渲染，将哪些方向被遮挡编码为 SH 系数。
    # SH 度数为 3 → (3+1)^2 = 16 个系数（实际只用 d^2=9），支持平滑方向变化。
    occlu_sh_degree = 3
    occlusion_threshold = args.occlusion
    valid_mask = torch.zeros([args.occlu_res, args.occlu_res, args.occlu_res]).bool().cuda()
    points = gaussians.get_xyz
    # 将高斯点分配到体素（含相邻体素标记），确定哪些体素需要烘焙。
    # 思路：对每个高斯点，除了它所在的体素外，还把它在 +x/+y/+z 方向上的
    # 相邻体素一并标记为有效，构成 2×2×2 的 8 个体素块。这样可避免点恰好
    # 落在体素边界处时仅标记单一体素，导致周围体素漏烘焙、出现遮挡空洞。
    # quat: 每个高斯点所在的体素整数索引（ floors(point - aabb_min) / grid ）
    quat = ((points - aabb_min) // grid).long()
    # 每个轴取当前体素索引(q*0)与下一个体素索引(q*1=+1)，并 clamp 到 [0, occlu_res-1]
    # 以处理点落在网格外或最后一层体素的边界情况
    qx0= quat[..., 0].clamp(min=0, max=args.occlu_res - 1)
    qx1 = (quat[..., 0] + 1).clamp(min=0, max=args.occlu_res - 1)
    qy0= quat[..., 1].clamp(min=0, max=args.occlu_res - 1)
    qy1 = (quat[..., 1] + 1).clamp(min=0, max=args.occlu_res - 1)
    qz0= quat[..., 2].clamp(min=0, max=args.occlu_res - 1)
    qz1 = (quat[..., 2] + 1).clamp( min=0, max=args.occlu_res - 1)
    # 标记高斯点周围 2×2×2 共 8 个体素为有效（需要烘焙遮挡 SH 的体素）
    valid_mask[qx0, qy0, qz0] = True
    valid_mask[qx0, qy0, qz1] = True
    valid_mask[qx0, qy1, qz0] = True
    valid_mask[qx0, qy1, qz1] = True
    valid_mask[qx1, qy0, qz0] = True
    valid_mask[qx1, qy0, qz1] = True
    valid_mask[qx1, qy1, qz0] = True
    valid_mask[qx1, qy1, qz1] = True
    # 返回(x_indices, y_indices, z_indices) 对应 valid_mask 中所有 True 元素在三个轴上的坐标索引。
    xyz_ids = torch.where(valid_mask)
    coords_of_id = torch.stack(xyz_ids, dim=-1).cuda()  # [num_grid, 3]
    num_grid = valid_mask.sum()
    # occlusion_ids: 3D 体素网格，-1=空体素
    occlusion_ids = (
        torch.ones(
            [args.occlu_res, args.occlu_res, args.occlu_res],
            dtype=torch.int32,
        )
        * -1
    ).cuda()
    # 填入有效索引
    occlusion_ids[xyz_ids[0], xyz_ids[1], xyz_ids[2]] = torch.arange(
        num_grid, dtype=torch.int32
    ).cuda()
    # occlusion_coefficients: [num_grid, d^2, 1] 有效体素的 SH 可见性系数
    # 训练前初始化为零（全可见），训练中通过体素级渲染更新
    occlusion_coefficients = torch.zeros(
        [num_grid, occlu_sh_degree**2, 1], dtype=torch.float32
    ).cuda()
    # !SECTION

    # SECTION prepare 数据
    render_path = os.path.join(model_path, "temp")

    makedirs(render_path, exist_ok=True)

    # prepare
    screenspace_points = (
        torch.zeros_like(
            gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=False, device="cuda"
        )
        + 0
    )
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians.get_opacity
    shs = gaussians.get_shs
    scales = gaussians.get_scaling
    rots = gaussians.get_rotation

    # Equirect mode: use equirect rasterizer resolution (shorter envmap arrays)
    if args.equirect:
        eq_H, eq_W = args.equirect_res, args.equirect_res * 2
        print(f"[equirect] Using equirect rasterizer at resolution {eq_H}×{eq_W}")
        solid_angles, envmap_dirs = get_envmap_dirs([eq_H, eq_W])
        components = components_from_spherical_harmonics(occlu_sh_degree, envmap_dirs)
    else:
        # [H, W, 1]、[H, W, 3]
        solid_angles, envmap_dirs = get_envmap_dirs()
        components = components_from_spherical_harmonics(occlu_sh_degree, envmap_dirs)  # [H, W, d2]

    # get canonical ray and its norm to normalize depth
    canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

    # scene_min/max already computed above (for --auto_bound and/or --skip_walls)
    if args.skip_walls:
        if args.auto_bound:
            print(f"  Using scene extent as reference for wall detection (grid covers [{aabb_min[0]:.3f}, {aabb_max[0]:.3f}] x [{aabb_min[1]:.3f}, {aabb_max[1]:.3f}] x [{aabb_min[2]:.3f}, {aabb_max[2]:.3f}])")
        else:
            print(f"  Using scene extent as reference for wall detection (bound={args.bound})")
    # !SECTION

    # =================================================================
    # SECTION --vis_walls: 墙壁检测可视化
    # 从场景中心用 SGS 等距光栅化器渲染全景图，将墙壁区域高亮为红色覆盖层，
    # 同时输出法线朝向可视化用于调试。保存后直接退出（不执行烘焙）。
    # =================================================================
    if args.vis_walls:
        center = ((scene_min + scene_max) / 2)
        print(f"\n[vis_walls] Rendering from scene center: {center.detach().cpu().numpy()}")
        print(f"[vis_walls] scene_min={scene_min.detach().cpu().numpy()}, scene_max={scene_max.detach().cpu().numpy()}")
        print(f"[vis_walls] wall_margin={args.wall_margin} → per-face: min_x={parts[0]}, max_x={parts[1]}, min_y={parts[2]}, max_y={parts[3]}, min_z={parts[4]}, max_z={parts[5]}")

        # Build a dummy Camera at scene center for the SGS equirect rasterizer
        H, W = 256, 512  # equirect resolution
        R = np.eye(3, dtype=np.float32)
        T = -center.detach().cpu().numpy().astype(np.float32)
        dummy_cam = Camera(colmap_id=0, R=R, T=T, FoVx=1.0, FoVy=1.0,
                           fx=1.0, fy=1.0, cx=W / 2, cy=H / 2,
                           image=None, image_name="vis", uid=0,
                           height=H, width=W, data_device="cuda")

        # SGS equirect rasterizer settings (camera_type=3)
        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=0.0,
            tanfovy=0.0,
            bg=torch.zeros(3, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=dummy_cam.world_view_transform,
            projmatrix=dummy_cam.full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=dummy_cam.camera_center,
            prefiltered=False,
            debug=False,
            camera_type=3,
            render_depth=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Render with SH
        shs = gaussians.get_shs
        rendered_image, _, radii, depth_raw, acc, normal_raw = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            extra_features=None,
            opacities=opacity,
            scales=scales,
            rotations=rots,
            cov3D_precomp=None,
        )  # rendered_image: [3, H, W], depth_raw: [1, H, W], acc: [1, H, W]

        rendered_image = rendered_image.permute(1, 2, 0)  # [H, W, 3]
        depth = depth_raw.squeeze(0).unsqueeze(-1)  # [1, H, W] → [H, W, 1]; already alpha-normalized radial distance
        alpha = acc.unsqueeze(-1)  # [H, W, 1]

        # Equirect ray directions for hit position computation.
        # _equirect_ray_dirs returns rays in equirect space (+Y up) but the rasterizer's
        # view space uses COLMAP convention (+Y down).  Flip Y to match before computing
        # world-space hit positions and normal-facing dot products.
        equi_dirs = _equirect_ray_dirs(H, W).cuda()  # [H, W, 3] equirect space (+Y up)
        equi_dirs = equi_dirs * equi_dirs.new_tensor([1.0, -1.0, 1.0])  # → COLMAP view space
        hit_pos = dummy_cam.camera_center.view(1, 1, 3) + equi_dirs * depth  # [H, W, 3]

        # ---- Normal-facing debug: check if normals point toward or away from camera ----
        def save_img(tensor, path):
            imageio.imwrite(path, (tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8))

        # normal_raw is [3, H, W] from the SGS rasterizer → permute to [H, W, 3]
        normal_img = normal_raw.permute(1, 2, 0)  # [H, W, 3]
        normal_len = normal_img.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        normal_unit = normal_img / normal_len  # [H, W, 3] unit-length world-space normals

        # view_dir = hit_pos - camera (direction from camera → hit point), equi_dirs is already unit.
        # cos_angle = dot(normal, view_dir):
        #   > 0 → same direction → normal points AWAY from camera (back-facing, "背向视野")
        #   < 0 → opposite → normal points TOWARD camera (front-facing, "朝向视野")
        cos_angle = (normal_unit * equi_dirs).sum(dim=-1)  # [H, W], range ≈ [-1, 1]

        # 1) Binary facing map: red = back-facing (normal away from camera), blue = front-facing
        facing_vis = torch.where(
            cos_angle.unsqueeze(-1) > 0,
            torch.tensor([1.0, 0.1, 0.1], device="cuda"),  # red: back-facing
            torch.tensor([0.1, 0.3, 1.0], device="cuda"),  # blue: front-facing
        )
        facing_vis = torch.where(alpha > 0.5, facing_vis, rendered_image * 0.3 + 0.5)

        # 2) Cos heatmap: white(cos=+1,back) → gray(cos=0) → black(cos=-1,front)
        cos_heat = ((cos_angle + 1) / 2).clamp(0, 1)  # [H, W] mapped to 0~1
        cos_heat_rgb = cos_heat.unsqueeze(-1).expand(-1, -1, 3)
        cos_heat_rgb = torch.where(alpha > 0.5, cos_heat_rgb, rendered_image * 0.3 + 0.5)

        # 3) Raw normal map for reference (world-space normal encoded as RGB)
        normal_map = normal_unit * 0.5 + 0.5  # [-1,1] → [0,1]

        save_img(facing_vis.squeeze(), os.path.join(model_path, "vis_walls_normal_facing.png"))
        save_img(cos_heat_rgb.squeeze(), os.path.join(model_path, "vis_walls_normal_cos.png"))
        save_img(normal_map.squeeze(), os.path.join(model_path, "vis_walls_normal_map.png"))
        print(f"[vis_walls] Saved normal-facing debug: vis_walls_normal_facing.png (red=back, blue=front)")
        print(f"[vis_walls]   Also saved: vis_walls_normal_cos.png (heatmap), vis_walls_normal_map.png")

        # Wall detection with per-face margins
        is_bg = (alpha < 0.5)
        dist_to_min = hit_pos - scene_min.cuda()  # [H, W, 3] — (dist_to_min_x, dist_to_min_y, dist_to_min_z)
        dist_to_max = scene_max.cuda() - hit_pos  # [H, W, 3] — (dist_to_max_x, dist_to_max_y, dist_to_max_z)
        is_wall_faces = torch.cat([
            dist_to_min[..., 0:1] < wall_margins[0],  # near min_x
            dist_to_max[..., 0:1] < wall_margins[1],  # near max_x
            dist_to_min[..., 1:2] < wall_margins[2],  # near min_y
            dist_to_max[..., 1:2] < wall_margins[3],  # near max_y
            dist_to_min[..., 2:3] < wall_margins[4],  # near min_z
            dist_to_max[..., 2:3] < wall_margins[5],  # near max_z
        ], dim=-1)  # [H, W, 6]
        is_wall = (~is_bg) & is_wall_faces.any(dim=-1, keepdim=True)  # [H, W, 1]

        # Build overlay visualization
        overlay = rendered_image.clone()
        if is_wall.any() and wall_margins.max() > 0:
            wmask = is_wall.squeeze().unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]
            overlay = torch.where(wmask, overlay * 0.3 + torch.tensor([0.7, 0.1, 0.0], device="cuda") * 0.7, overlay)

        save_img(rendered_image, os.path.join(model_path, "vis_walls_rgb.png"))
        save_img(overlay, os.path.join(model_path, "vis_walls_overlay.png"))

        n_wall = is_wall.sum().item()
        n_surface = (~is_bg).sum().item()
        print(f"[vis_walls] Stats: wall={n_wall}, non-wall={n_surface - n_wall}, bg={is_bg.sum().item()}")
        print(f"[vis_walls] Saved to {model_path}/vis_walls_rgb.png and vis_walls_overlay.png")
    # !SECTION



    with torch.no_grad():
        # =================================================================
        # SECTION Equirect 烘焙模式（使用 SGS 等距光栅化器，camera_type=3）
        # =================================================================
        # 对每个有效体素，从体素中心渲染一张等距全景图（单次渲染覆盖 4π 球面），
        # 用背景色检测遮挡（白色背景 → 被遮挡的像素不是白色），
        # 将遮挡掩码投影到 SH 基函数上，累加得到体素的可见性 SH 系数。
        if args.equirect:
            bg = torch.ones(3, device="cuda")  # white: unoccluded pixels remain white (>0.5) after alpha blend
            for grid_id in trange(num_grid):
                quat = coords_of_id[grid_id]
                position = positions[quat[0] * args.occlu_res**2 + quat[1] * args.occlu_res + quat[2]]

                # crop by position
                diff = means3D - position
                valid = (diff.abs() < args.valid).all(dim=1)

                # Camera at voxel center, identity rotation (equirect covers full sphere)
                c2w = torch.eye(4, device="cuda")
                c2w[:3, 3] = position
                w2c = torch.inverse(c2w)
                viewmatrix = w2c.T

                raster_settings = GaussianRasterizationSettings(
                    image_height=eq_H,
                    image_width=eq_W,
                    tanfovx=0.0,
                    tanfovy=0.0,
                    bg=bg,
                    scale_modifier=1.0,
                    viewmatrix=viewmatrix,
                    projmatrix=viewmatrix,  # not used by camera_type=3
                    sh_degree=gaussians.active_sh_degree,
                    campos=position,
                    prefiltered=False,
                    debug=False,
                    camera_type=3,
                    render_depth=False,
                )
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                rendered_image, _, radii, depth_raw, acc, normal_raw = rasterizer(
                    means3D=means3D[valid],
                    means2D=means2D[valid],
                    shs=shs[valid],
                    colors_precomp=None,
                    extra_features=None,
                    opacities=opacity[valid],
                    scales=scales[valid],
                    rotations=rots[valid],
                    cov3D_precomp=None,
                )

                rendered_image = rendered_image.permute(1, 2, 0)  # [eq_H, eq_W, 3]
                # 背景色白色 (1,1,1)，被高斯遮挡的像素 < 0.5 → masked out
                is_bg = rendered_image[..., 0:1] > 0.5  # [eq_H, eq_W, 1]

                if args.skip_walls:
                    depth = depth_raw * (acc > 0.5).float()  # filter floaters
                    depth = depth.squeeze(0).unsqueeze(-1)  # [1, eq_H, eq_W] → [eq_H, eq_W, 1]
                    # envmap_dirs 使用 reflvec 约定 (+Y up, -Z forward)，
                    # 转 world space：diag(1, -1, -1)
                    world_dirs = envmap_dirs * envmap_dirs.new_tensor([1.0, -1.0, -1.0])
                    hit_pos = position + world_dirs * depth  # [eq_H, eq_W, 3]
                    dist_to_min = hit_pos - scene_min
                    dist_to_max = scene_max - hit_pos
                    is_wall_faces = torch.cat([
                        dist_to_min[..., 0:1] < wall_margins[0],
                        dist_to_max[..., 0:1] < wall_margins[1],
                        dist_to_min[..., 1:2] < wall_margins[2],
                        dist_to_max[..., 1:2] < wall_margins[3],
                        dist_to_min[..., 2:3] < wall_margins[4],
                        dist_to_max[..., 2:3] < wall_margins[5],
                    ], dim=-1)
                    is_wall = (~is_bg) & is_wall_faces.any(dim=-1, keepdim=True)
                    occlu_mask = (is_bg | is_wall).float()
                else:
                    occlu_mask = is_bg.float()

                # 将遮挡掩码投影到 SH 基函数上：C_j = Σ_pixels (mask_p * ω_p * Y_j(dir_p))
                weighted_color = occlu_mask * solid_angles
                temp_coefficients = (weighted_color * components).sum(0).sum(0)
                occlusion_coefficients[grid_id] = temp_coefficients[:, None]
        # !SECTION
        # =================================================================
        # SECTION Cubemap 烘焙模式（原始行为，使用 diff-gaussian-rasterization）
        # =================================================================
        # 对每个有效体素，从体素中心渲染 6 个面（FOV=90° 的透视立方体贴图），
        # 通过 nvdiffrast 的 boundary_mode="cube" 将 6 面贴图采样到等距方向，
        # 用背景色检测遮挡并投影到 SH 基函数。
        else:  # cubemap mode (original behavior)
            for grid_id in trange(num_grid):
                quat = coords_of_id[grid_id]  # [3] voxel coords, O(1) vs O(res³)
                position = positions[quat[0] * args.occlu_res**2 + quat[1] * args.occlu_res + quat[2]]
                # position = torch.tensor([0.0, 1.5, 0.0]).to(position.device)
                rgb_cubemap = []
                opacity_cubemap = []
                depth_cubemap = []
                # NOTE: crop by position
                # 只保留 valid 范围内的高斯点左遮挡检测
                diff = means3D - position
                valid = (diff.abs() < args.valid).all(dim=1)
                valid_means3D = means3D[valid]
                valid_means2D = means2D[valid]
                valid_opacity = opacity[valid]
                valid_shs = shs[valid]
                valid_scales = scales[valid]
                valid_rots = rots[valid]
                # 渲染 6 个面（+X, -X, +Y, -Y, +Z, -Z），每个面 FOV=90°
                for r_idx, rotation in enumerate(rotations):
                    # SECTION 根据 c2w 求 w2c 
                    c2w = rotation
                    # 相机原点在世界坐标系下的坐标(体素中心)
                    c2w[:3, 3] = position
                    w2c = torch.inverse(c2w)
                    # w2c.T 转为列主序（供 CUDA 左乘）
                    world_view_transform = w2c.T
                    full_proj_transform = (
                        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
                    ).squeeze(0)
                    # 我服了
                    # camera_center = world_view_transform.inverse()[3, :3]
                    camera_center = position
                    # !SECTION

                    input_args = (
                        bg_color,
                        # bg_colors[r_idx],
                        valid_means3D,
                        torch.zeros_like(valid_means3D),
                        valid_opacity,
                        valid_scales,
                        valid_rots,
                        torch.Tensor([]),
                        shs,
                        camera_center,  # campos,
                        world_view_transform,  # viewmatrix,
                        full_proj_transform,  # projmatrix,
                        1.0,  # scale_modifier
                        1.0,  # tanfovx,
                        1.0,  # tanfovy,
                        res,  # image_height,
                        res,  # image_width,
                        gaussians.active_sh_degree,
                        False,  # prefiltered,
                        False,  # argmax_depth,
                    )
                    (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(
                        *input_args
                    )
                    rgb_cubemap.append(rendered_image.permute(1, 2, 0))
                    opacity_cubemap.append(opacity_map.permute(1, 2, 0))
                    depth_map = depth_map * (opacity_map > 0.5).float()  # NOTE: import to filter out the floater
                    depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)

                # 用 nvdiffrast 将 6 面 cubemap 采样到等距方向网格
                # boundary_mode="cube" 根据方向 (envmap_dirs) 在 6 个面间做纹理查找
                depth_envmap = dr.texture(
                    torch.stack(depth_cubemap)[None, ...],
                    envmap_dirs[None, ...].contiguous(),
                    # filter_mode="linear",
                    filter_mode="nearest",
                    boundary_mode="cube",
                )[
                    0
                ]  # [H, W, 1]

                rgb_envmap = dr.texture(
                    torch.stack(rgb_cubemap)[None, ...],
                    envmap_dirs[None, ...].contiguous(),
                    # filter_mode="linear",
                    filter_mode="nearest",
                    boundary_mode="cube",
                )[
                    0
                ][..., 0:1]  # [H, W, 1]

                # print(rgb_envmap.shape)
                # print(depth_envmap.shape)

                # use SH to store the HDRI
                # occlu_mask = (1 - (depth_envmap < occlusion_threshold).float()) + (depth_envmap == 0).float()  # [H, W, 1]
                # occlu_mask = (rgb_envmap > 0.5).float()
                
                # 遮挡判定：白色背景 (1,1,1) → 无高斯遮挡，视为可见
                # 使用单个通道（灰度）判断，>0.5 为可见

                is_bg = (rgb_envmap > 0.5)  # [H, W, 1] — no surface hit, fully visible
                if args.skip_walls:
                    # Compute hit positions and check proximity to scene AABB boundary (per-face)
                    # envmap_dirs uses reflvec convention (+Y up, -Z forward, nvdiffrast cubemap).
                    # position/scene_min/scene_max are in COLMAP world space (+Y down, +Z forward).
                    # Convert direction to world space before computing hit_pos:
                    #   n_world = diag(1, -1, -1) @ n_reflvec
                    world_dirs = envmap_dirs * envmap_dirs.new_tensor([1.0, -1.0, -1.0])
                    hit_pos = position + world_dirs * depth_envmap  # [H, W, 3]
                    dist_to_min = hit_pos - scene_min  # [H, W, 3]
                    dist_to_max = scene_max - hit_pos  # [H, W, 3]
                    is_wall_faces = torch.cat([
                        dist_to_min[..., 0:1] < wall_margins[0],  # near min_x
                        dist_to_max[..., 0:1] < wall_margins[1],  # near max_x
                        dist_to_min[..., 1:2] < wall_margins[2],  # near min_y
                        dist_to_max[..., 1:2] < wall_margins[3],  # near max_y
                        dist_to_min[..., 2:3] < wall_margins[4],  # near min_z
                        dist_to_max[..., 2:3] < wall_margins[5],  # near max_z
                    ], dim=-1)  # [H, W, 6]
                    is_wall = (~is_bg) & is_wall_faces.any(dim=-1, keepdim=True)  # [H, W, 1]
                    occlu_mask = (is_bg | is_wall).float()
                else:
                    occlu_mask = is_bg.float()

                # 将遮挡掩码投影到 SH 基函数：C_j = Σ 掩码 * 立体角 * Y_j
                weighted_color = occlu_mask * solid_angles  # [H, W, 1]
                temp_coefficients = (weighted_color * components).sum(0).sum(0)  # [d2]
                occlusion_coefficients[grid_id] = temp_coefficients[:, None]
        # !SECTION
        # =================================================================
        # 体素空洞填充：将未标记（-1）的空体素用最近的邻居扩张填充
        # 使遮挡插值（recon_occlusion）不会遇到无效体素
        # =================================================================
        while (occlusion_ids == -1).sum() > 0:
            gs_ir_ext.dialate_occlusion_ids(occlusion_ids)

    # =================================================================
    # 保存遮挡体素数据
    # =================================================================
    save_dict = {
        "occlusion_ids": occlusion_ids,           # [occlu_res³] 体素索引映射
        "occlusion_coefficients": occlusion_coefficients,  # [num_grid, d², 1] SH 系数
        "bound": args.bound,                      # 场景边界（对称）
        "degree": occlu_sh_degree,                # SH 度数 (=3)
        "occlusion_threshold": occlusion_threshold,
    }
    if args.auto_bound:
        save_dict["aabb"] = torch.cat([auto_aabb_min.cpu(), auto_aabb_max.cpu()])  # [6] 非对称 AABB
    save_file = args.occlusion_path or os.path.join(os.path.dirname(args.checkpoint), "occlusion_volumes.pth")
    torch.save(save_dict, save_file)
    print(f"save occlusion volumes as {save_file}")

