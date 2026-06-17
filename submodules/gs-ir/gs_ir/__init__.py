import torch

from . import _C
from .volumes import IrradianceVolumes

@torch.no_grad()
def recon_occlusion(
    H: int,
    W: int,
    bound: float,
    points: torch.Tensor,  # [HW, 3]
    normals: torch.Tensor,  # [HW, 3]
    roughness: torch.Tensor,  # [HW, 1]
    occlusion_coefficients: torch.Tensor,
    occlusion_ids: torch.Tensor,
    aabb: torch.Tensor,
    sample_rays: int = 256,
    degree: int = 4,
) -> torch.Tensor:
    occlu_res = occlusion_ids.shape[0]
    # half_grid = bound / float(occlu_res)
    # shift_points = points + normals * half_grid
    # 根据 aabb 计算 grid_step
    grid_step = (aabb[3:] - aabb[:3]) / float(occlu_res - 1)  # per-axis spacing [3]
    shift_points = points + normals * (grid_step * 0.5)  # shift by half grid along normal
    # shift_points = points
    (
        coefficients,  # [HW, d2, 1]
        coeff_ids,  # [HW, 8]
    ) = _C.sparse_interpolate_coefficients(
        occlusion_coefficients,
        occlusion_ids,
        aabb,
        shift_points,
        normals,
        degree,
    )
    coefficients = coefficients.permute(0, 2, 1)  # [HW, 1, d2]

    roughness = torch.ones([H * W, 1], dtype=torch.float32).cuda()

    # baking 的 SH 系数使用 reflvec 空间计算:
    #   envmap_dirs = (sinθ·sinφ, cosθ, -sinθ·cosφ)
    # 其中 +Y = 上, φ=0 → -Z = "前" (即 nvdiffrast cubemap 约定)
    #
    # 传入的 normals 是 COLMAP 世界空间 (+Y = 下, +Z = 前),
    # 必须在 SH 求值前转换到 reflvec 空间:
    #   n_reflvec = diag(1, -1, -1) @ n_colmap
    reflvec_normals = normals.clone()
    reflvec_normals[:, 1] *= -1.0  # flip Y: COLMAP +Y down → reflvec +Y up
    reflvec_normals[:, 2] *= -1.0  # flip Z: COLMAP +Z forward → reflvec -Z forward

    occlusion = _C.SH_reconstruction(
        coefficients, reflvec_normals, roughness, sample_rays, degree
    )  # [HW, 1]

    return occlusion


__all__ = ["_C", "recon_occlusion", "IrradianceVolumes"]
