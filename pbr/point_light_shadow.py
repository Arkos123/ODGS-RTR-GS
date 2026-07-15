"""
点光源阴影模块。

透视模式：从光源位置渲染 6 张深度图（cubemap），用 nvdiffrast 查询。
Equirect 模式：从光源位置渲染单张全景深度图，用方向→UV 映射查询。

参考 GS-IR/shadow_map.py:get_depth_cubemap()
"""
from typing import Callable

import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import _C as diff_C
from scene.gaussian_model import GaussianModel
from spherical_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
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
    def shadow_func(points: torch.Tensor) -> torch.Tensor:
        """逐像素阴影查询（equirect）"""
        _, H, W = points.shape
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
