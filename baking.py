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


def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res: List[int] = [256, 512]) -> Tuple[torch.Tensor, torch.Tensor]:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )
    d_theta, d_phi = np.pi / res[0], 2 * np.pi / res[1]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    # get solid angles
    solid_angles = ((costheta - torch.cos(gy * np.pi + d_theta)) * d_phi)[..., None]  # [H, W, 1]
    print(f"solid_angles_sum error: {solid_angles.sum() - 4 * np.pi}")

    return solid_angles, reflvec


def lookAt(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    Z = F.normalize(eye - center, dim=0)
    Y = up
    X = F.normalize(torch.cross(Y, Z), dim=0)
    Y = F.normalize(torch.cross(Z, X), dim=0)

    matrix = torch.tensor(
        [
            [X[0], Y[0], Z[0]],
            [X[1], Y[1], Z[1]],
            [X[2], Y[2], Z[2]],
        ]
    )

    return matrix


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
    """Equirectangular pixel → world-space ray direction vectors [H, W, 3]."""
    ys = torch.linspace(0.5 * math.pi, -0.5 * math.pi, H, device=device)
    xs = torch.linspace(-math.pi, math.pi, W, device=device)
    lat, lon = torch.meshgrid(ys, xs, indexing="ij")
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
    rotations: List[torch.Tensor] = [
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.0, 1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
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

    # init occlusion volume
    occlu_sh_degree = 3
    occlusion_threshold = args.occlusion
    valid_mask = torch.zeros([args.occlu_res, args.occlu_res, args.occlu_res]).bool().cuda()
    points = gaussians.get_xyz
    quat = ((points - aabb_min) // grid).long()
    qx0, qx1 = quat[..., 0].clamp(min=0, max=args.occlu_res - 1), (quat[..., 0] + 1).clamp(
        min=0, max=args.occlu_res - 1
    )
    qy0, qy1 = quat[..., 1].clamp(min=0, max=args.occlu_res - 1), (quat[..., 1] + 1).clamp(
        min=0, max=args.occlu_res - 1
    )
    qz0, qz1 = quat[..., 2].clamp(min=0, max=args.occlu_res - 1), (quat[..., 2] + 1).clamp(
        min=0, max=args.occlu_res - 1
    )
    valid_mask[qx0, qy0, qz0] = True
    valid_mask[qx0, qy0, qz1] = True
    valid_mask[qx0, qy1, qz0] = True
    valid_mask[qx0, qy1, qz1] = True
    valid_mask[qx1, qy0, qz0] = True
    valid_mask[qx1, qy0, qz1] = True
    valid_mask[qx1, qy1, qz0] = True
    valid_mask[qx1, qy1, qz1] = True
    xyz_ids = torch.where(valid_mask)
    num_grid = valid_mask.sum()
    occlusion_ids = (
        torch.ones(
            [args.occlu_res, args.occlu_res, args.occlu_res],
            dtype=torch.int32,
        )
        * -1
    ).cuda()
    occlusion_ids[xyz_ids[0].tolist(), xyz_ids[1].tolist(), xyz_ids[2].tolist()] = torch.arange(
        num_grid, dtype=torch.int32
    ).cuda()
    occlusion_coefficients = torch.zeros(
        [num_grid, occlu_sh_degree**2, 1], dtype=torch.float32
    ).cuda()

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

    (
        solid_angles,  # [H, W, 1]
        envmap_dirs,  # [H, W, 3]
    ) = get_envmap_dirs()
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

    # --vis_walls: render equirect panorama using SGS rasterizer for wall detection verification
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
        rendered_image, radii, depth_raw, acc, normal_raw = rasterizer(
            means3D=means3D,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rots,
            cov3D_precomp=None,
        )  # rendered_image: [3, H, W], depth_raw: [1, H, W], acc: [1, H, W]

        rendered_image = rendered_image.permute(1, 2, 0)  # [H, W, 3]
        depth = (depth_raw.squeeze(0) / acc.squeeze(0).clamp_min(1e-6)).unsqueeze(-1)  # [H, W, 1]
        alpha = acc.unsqueeze(-1)  # [H, W, 1]

        # Equirect ray directions for hit position computation
        equi_dirs = _equirect_ray_dirs(H, W).cuda()  # [H, W, 3]
        hit_pos = dummy_cam.camera_center.view(1, 1, 3) + equi_dirs * depth  # [H, W, 3]

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

        def save_img(tensor, path):
            imageio.imwrite(path, (tensor.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8))

        save_img(rendered_image, os.path.join(model_path, "vis_walls_rgb.png"))
        save_img(overlay, os.path.join(model_path, "vis_walls_overlay.png"))

        n_wall = is_wall.sum().item()
        n_surface = (~is_bg).sum().item()
        print(f"[vis_walls] Stats: wall={n_wall}, non-wall={n_surface - n_wall}, bg={is_bg.sum().item()}")
        print(f"[vis_walls] Saved to {model_path}/vis_walls_rgb.png and vis_walls_overlay.png")
        # exit(0)

    with torch.no_grad():
        for grid_id in trange(num_grid):
            quat = torch.cat(torch.where(occlusion_ids == grid_id))
            position = positions[(quat[0] * args.occlu_res**2 + quat[1] * args.occlu_res + quat[2],)]
            # position = torch.tensor([0.0, 1.5, 0.0]).to(position.device)
            rgb_cubemap = []
            opacity_cubemap = []
            depth_cubemap = []
            # NOTE: crop by position
            diff = means3D - position
            valid = (diff.abs() < args.valid).all(dim=1)
            valid_means3D = means3D[valid]
            valid_means2D = means2D[valid]
            valid_opacity = opacity[valid]
            valid_shs = shs[valid]
            valid_scales = scales[valid]
            valid_rots = rots[valid]
            for r_idx, rotation in enumerate(rotations):
                c2w = rotation
                c2w[:3, 3] = position
                w2c = torch.inverse(c2w)
                T = w2c[:3, 3]
                R = w2c[:3, :3].T
                world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
                full_proj_transform = (
                    world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
                ).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

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

            # convert cubemap to HDRI
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
            is_bg = (rgb_envmap > 0.5)  # [H, W, 1] — no surface hit, fully visible
            if args.skip_walls:
                # Compute hit positions and check proximity to scene AABB boundary (per-face)
                hit_pos = position + envmap_dirs * depth_envmap  # [H, W, 3]
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

            weighted_color = occlu_mask * solid_angles  # [H, W, 1]
            temp_coefficients = (weighted_color * components).sum(0).sum(0)  # [d2]
            occlusion_coefficients[grid_id] = temp_coefficients[:, None]

        # dialate coefficient ids
        while (occlusion_ids == -1).sum() > 0:
            gs_ir_ext.dialate_occlusion_ids(occlusion_ids)

    save_dict = {
        "occlusion_ids": occlusion_ids,
        "occlusion_coefficients": occlusion_coefficients,
        "bound": args.bound,
        "degree": occlu_sh_degree,
        "occlusion_threshold": occlusion_threshold,
    }
    if args.auto_bound:
        save_dict["aabb"] = torch.cat([auto_aabb_min.cpu(), auto_aabb_max.cpu()])  # [6]
    save_file = os.path.join(os.path.dirname(args.checkpoint), "occlusion_volumes.pth")
    torch.save(save_dict, save_file)
    print(f"save occlusion volumes as {save_file}")

