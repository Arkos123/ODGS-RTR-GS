import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from pbr.shade import get_reflectance_color_forward, pbr_shading
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.prt_utils import PRTutils
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, tv_loss, est_wsmap
from utils.image_utils import psnr
from utils.graphics_utils import linear2srgb_torch
from spherical_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs_ir import recon_occlusion

def _ensure_pano_losses():
    """Lazy-load pano_losses inside function to avoid sys.path pollution at module level."""
    if _ensure_pano_losses.cache is not None:
        return _ensure_pano_losses.cache
    import sys, os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'spherical-gaussian-splatting'))
    if os.path.isdir(_root):
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from pano_losses import (
            pano_row_area_weight, pano_rgb_nonedge_weight,
            geometry_iter_ramp, pano_normal_smoothness_loss, pano_alpha_hole_loss,
        )
        _ensure_pano_losses.cache = (pano_row_area_weight, pano_rgb_nonedge_weight, geometry_iter_ramp, pano_normal_smoothness_loss, pano_alpha_hole_loss)
    else:
        _ensure_pano_losses.cache = (None, None, None, None, None)
    return _ensure_pano_losses.cache
_ensure_pano_losses.cache = None


def _erp_edge_aware_loss(data, img):
    """Edge-aware smoothness with cyclic horizontal boundary for ERP.

    Replaces first_order_edge_aware_loss for equirect mode.
    Horizontal gradients use torch.roll (cyclic) to avoid left-right seam.
    Vertical gradients use standard padding (non-cyclic, ERP vertical is finite).
    """
    dx = (torch.roll(data, shifts=-1, dims=-1) - data).abs()
    wx = torch.exp(-(torch.roll(img, shifts=-1, dims=-1) - img).abs())
    dy = torch.zeros_like(data)
    wy = torch.zeros_like(img)
    if data.shape[-2] > 1:
        dy[:, :-1, :] = (data[:, 1:, :] - data[:, :-1, :]).abs()
        wy[:, :-1, :] = torch.exp(-(img[:, 1:, :] - img[:, :-1, :]).abs())
    return (dx * wx + dy * wy).sum(dim=0).mean()


def _equirect_ray_dirs(H, W, device='cuda'):
    """Equirectangular pixel → ray directions in COLMAP view space (+Y down)."""
    ys = torch.linspace(0.5 * math.pi, -0.5 * math.pi, H, device=device)
    xs = torch.linspace(-math.pi, math.pi, W, device=device)
    lat, lon = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([
        torch.sin(lon) * torch.cos(lat),
        -torch.sin(lat),
        torch.cos(lon) * torch.cos(lat),
    ], dim=-1)


def _project_lat_lon(means3D, viewmatrix):
    """Per-Gaussian lon/lat in ERP convention (for densification only)."""
    with torch.no_grad():
        if means3D.numel() == 0:
            z = torch.empty(0, device=means3D.device, dtype=means3D.dtype)
            return z, z, z
        ones = torch.ones((means3D.shape[0], 1), dtype=means3D.dtype, device=means3D.device)
        pts_h = torch.cat([means3D, ones], dim=-1)
        p_view = pts_h @ viewmatrix.to(device=means3D.device, dtype=means3D.dtype)
        x, y, z = p_view[:, 0], p_view[:, 1], p_view[:, 2]
        dist_xz = torch.sqrt(torch.clamp(x * x + z * z, min=1e-12))
        lat = torch.atan2(y, dist_xz)
        lon = torch.atan2(x, z)
        psi = torch.zeros_like(lat)
    return psi, lat, lon


# NOTE: _run_sgs_rasterizer wrapper removed in V2 multi-pass consolidation.
#       Direct rasterizer() call is used in render_view().

# ── ERP depth-to-normal helpers (from SGS gaussian_renderer) ────────────────

def _erp_edge_aware_smooth_depth(depth: torch.Tensor, alpha: torch.Tensor | None,
                                 iters: int = 2, rel_sigma: float = 0.035,
                                 eps: float = 1.0e-6):
    if depth is None or depth.numel() == 0:
        return depth
    d = depth
    valid = torch.ones_like(d, dtype=d.dtype)
    if alpha is not None:
        valid = (alpha > 1.0e-5).to(dtype=d.dtype, device=d.device)
    sigma = max(float(rel_sigma), 1.0e-4)
    for _ in range(max(0, int(iters))):
        acc = d * valid
        wsum = valid.clone()
        for shift, dim, cyclic in [(-1, -1, True), (1, -1, True), (-1, -2, False), (1, -2, False)]:
            dn = torch.roll(d, shifts=shift, dims=dim)
            vn = torch.roll(valid, shifts=shift, dims=dim)
            if not cyclic:
                if dim == -2 and shift < 0:
                    vn[..., -1:, :] = 0.0
                elif dim == -2 and shift > 0:
                    vn[..., :1, :] = 0.0
            rel = (dn.detach() - d.detach()).abs() / (0.5 * (dn.detach().abs() + d.detach().abs()) + eps)
            w = vn * valid / (1.0 + (rel / sigma) ** 2)
            acc = acc + w * dn
            wsum = wsum + w
        d = acc / wsum.clamp_min(eps)
    return d


def _shift_with_spatial_mask(t: torch.Tensor, shift: int, dim: int, cyclic: bool):
    out = torch.roll(t, shifts=shift, dims=dim)
    mask = torch.ones_like(t[:1])
    if not cyclic and dim == -2 and shift != 0:
        k = abs(int(shift))
        if shift < 0:
            mask[..., -k:, :] = 0.0
        else:
            mask[..., :k, :] = 0.0
    return out, mask


def _relative_depth_gate(d0: torch.Tensor, d1: torch.Tensor, sigma: float, eps: float = 1.0e-6):
    sig = max(float(sigma), 1.0e-4)
    rel = (d1.detach() - d0.detach()).abs() / (0.5 * (d1.detach().abs() + d0.detach().abs()) + eps)
    return 1.0 / (1.0 + (rel / sig) ** 4)


def _erp_tangent_from_same_surface_neighbors(pts: torch.Tensor, depth: torch.Tensor,
                                             valid: torch.Tensor, step: int, dim: int,
                                             edge_sigma: float):
    st = max(1, int(step))
    cyclic = (dim == -1)
    p_f, mf = _shift_with_spatial_mask(pts, -st, dim, cyclic)
    p_b, mb = _shift_with_spatial_mask(pts, st, dim, cyclic)
    d_f, _ = _shift_with_spatial_mask(depth, -st, dim, cyclic)
    d_b, _ = _shift_with_spatial_mask(depth, st, dim, cyclic)
    v_f, _ = _shift_with_spatial_mask(valid, -st, dim, cyclic)
    v_b, _ = _shift_with_spatial_mask(valid, st, dim, cyclic)

    wf = valid * v_f * mf * _relative_depth_gate(depth, d_f, edge_sigma)
    wb = valid * v_b * mb * _relative_depth_gate(depth, d_b, edge_sigma)
    tangent = (wf * (p_f - pts) + wb * (pts - p_b)) / (wf + wb).clamp_min(1.0e-8)
    confidence = (wf + wb).clamp(0.0, 1.0)
    return tangent, confidence


def _erp_smooth_normals_same_surface(normal: torch.Tensor, depth: torch.Tensor,
                                     confidence: torch.Tensor, iters: int,
                                     edge_sigma: float, eps: float = 1.0e-8):
    if normal is None or normal.numel() == 0:
        return normal
    n = normal
    c = confidence.clamp(0.0, 1.0)
    for _ in range(max(0, int(iters))):
        acc = n * c
        wsum = c.clone()
        for shift, dim, cyclic in [(-1, -1, True), (1, -1, True), (-1, -2, False), (1, -2, False)]:
            nn, mm = _shift_with_spatial_mask(n, shift, dim, cyclic)
            dn, _ = _shift_with_spatial_mask(depth, shift, dim, cyclic)
            cn, _ = _shift_with_spatial_mask(c, shift, dim, cyclic)
            dg = _relative_depth_gate(depth, dn, edge_sigma)
            ng = ((n.detach() * nn.detach()).sum(dim=0, keepdim=True).clamp_min(0.0)) ** 2
            w = c * cn * mm * dg * ng
            acc = acc + w * nn
            wsum = wsum + w
        n = F.normalize(acc / wsum.clamp_min(eps), dim=0, eps=eps)
    return n


def _erp_depth_to_normal(depth: torch.Tensor, alpha: torch.Tensor,
                         eps: float = 1.0e-8, step: int = 8,
                         smooth_iters: int = 3,
                         smooth_rel_sigma: float = 0.055,
                         edge_rel_sigma: float = 0.075,
                         min_confidence: float = 0.035):
    """Depth → normal in COLMAP view space (+Y down).

    Internally the ray Y component uses -sin(lat) (instead of +sin(lat))
    so the output normals follow the COLMAP +Y-down convention directly.
    The caller only needs the C2W rotation to reach world space.
    """
    if depth is None or depth.numel() == 0 or depth.ndim != 3:
        return None, None
    _, H, W = depth.shape
    valid = torch.ones_like(depth, dtype=depth.dtype)
    if alpha is not None:
        valid = (alpha > 1.0e-5).to(dtype=depth.dtype, device=depth.device)

    d = _erp_edge_aware_smooth_depth(depth, alpha, iters=int(smooth_iters), rel_sigma=float(smooth_rel_sigma))
    device, dtype = d.device, d.dtype

    ys = torch.linspace(0.5 * math.pi, -0.5 * math.pi, H, device=device, dtype=dtype)
    xs = torch.linspace(-math.pi, math.pi, W, device=device, dtype=dtype)
    lat, lon = torch.meshgrid(ys, xs, indexing='ij')
    rays = torch.stack([torch.sin(lon) * torch.cos(lat), -torch.sin(lat), torch.cos(lon) * torch.cos(lat)], dim=0)
    pts = rays * d.clamp_min(1.0e-6)

    st = max(1, min(int(step), max(1, min(H, W) // 8)))
    steps = sorted(set([1, max(1, st // 4), max(1, st // 2), st]))
    n_acc = torch.zeros_like(pts)
    c_acc = torch.zeros_like(d)
    scale_w_sum = 0.0
    for s in steps:
        tx, cx = _erp_tangent_from_same_surface_neighbors(pts, d, valid, s, -1, edge_rel_sigma)
        ty, cy = _erp_tangent_from_same_surface_neighbors(pts, d, valid, s, -2, edge_rel_sigma)
        cross = torch.cross(tx, ty, dim=0)
        cross_norm = torch.linalg.norm(cross, dim=0, keepdim=True)
        n = F.normalize(cross, dim=0, eps=eps)
        n = torch.where((n * rays).sum(dim=0, keepdim=True) > 0.0, -n, n)
        c = (cx * cy * (cross_norm.detach() / (cross_norm.detach() + 1.0e-6))).clamp(0.0, 1.0)
        sw = math.sqrt(float(s))
        n_acc = n_acc + sw * c * n
        c_acc = c_acc + sw * c
        scale_w_sum += sw

    confidence = (c_acc / max(scale_w_sum, 1.0e-6)).clamp(0.0, 1.0)
    n = F.normalize(n_acc / c_acc.clamp_min(eps), dim=0, eps=eps)
    post_iters = max(1, int(smooth_iters) // 3)
    n = _erp_smooth_normals_same_surface(n, d, confidence, post_iters, edge_rel_sigma)
    keep = (valid > 0.0) & (confidence > float(min_confidence))
    n = torch.where(keep.expand_as(n), n, torch.zeros_like(n))
    return n


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None,
                fast_pbr=False):
    gamma_func = lambda x: linear2srgb_torch(x)

    refmap = dict_params.get("refmap")
    cubemap = dict_params.get("cubemap") if pc.use_pbr else None
    transfer_net = dict_params.get("transfer_net")
    occlusion_volumes = dict_params.get("occlusion_volumes")
    aabb = dict_params.get("aabb")
    brdf_lut = dict_params.get("brdf_lut")

    if is_training:
        if refmap is not None:
            refmap.train()
            refmap.build_mips()
        if cubemap is not None:
            cubemap.train()
            cubemap.build_mips()
    else:
        if refmap is not None:
            refmap.eval()
        if cubemap is not None:
            cubemap.eval()

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    c2w = viewpoint_camera.c2w
    H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

    # SGS rasterizer settings for equirect mode (camera_type=3)
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=0.0,
        tanfovy=0.0,
        bg=torch.zeros(3, device='cuda'),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        camera_type=3,
        render_depth=False, # deprecated, only used by OmniGS pinhole path
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    ref_tint = pc.get_ref_tint
    ref_roughness = pc.get_ref_roughness
    ref_strength = pc.get_ref_strength
    normal = pc.get_min_axis(viewpoint_camera.camera_center)

    if not fast_pbr:
        only_diffuse = dict_params.get("iteration", 0) < pipe.diffuse_iteration
        if pipe.compute_with_prt and override_color is None and transfer_net is not None:
            viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
            if only_diffuse:
                prt_color = PRTutils.cal_diffuse(pc)
            else:
                prt_color = PRTutils.cal_color(pc, transfer_net, viewdirs, normal, is_training)
            override_color = prt_color

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(
                viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D, dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    # ---- Compute forward-shaded color (PRT + forward reflection) ----
    viewdirs_gauss = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    if fast_pbr:
        colors_precomp_pass1 = None
    elif pipe.forward_shading and refmap is not None:
        refl_color_forward = get_reflectance_color_forward(
            refmap, normal, viewdirs_gauss, ref_roughness, ref_tint, brdf_lut=brdf_lut)
        colors_precomp_pass1 = (1.0 - ref_strength) * colors_precomp + ref_strength * refl_color_forward
    else:
        colors_precomp_pass1 = colors_precomp

    # ---- Compute PBR per-Gaussian attributes for extra_features ----
    if pc.use_pbr:
        base_color = pc.get_base_color
        roughness = pc.get_roughness
        metallic = pc.get_metallic
        if not getattr(pipe, 'relight', False):
            incidents = pc.get_incidents
            incidents_rgb = torch.clamp(eval_sh(
                pc.active_sh_degree,
                incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2),
                normal,
            ), 0.0, 1.0)
        elif getattr(pipe, 'transfer_light', False) and cubemap is not None:
            transfer_shs = pc.get_incidents.permute(0, 2, 1)
            light_shs = cubemap.shs
            incidents = light_shs * transfer_shs
            incidents = incidents.permute(0, 2, 1)
            incidents_rgb = torch.clamp(eval_sh(
                pc.active_sh_degree,
                incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2),
                normal,
            ), 0.0, 1.0)
        else:
            incidents_rgb = torch.zeros_like(base_color)

    # ---- Build extra_features (V2 multi-channel: all non-color attrs in one pass) ----
    extra_list = [normal * 0.5 + 0.5]  # [P, 3], normal encoded for [0,1] range
    if not fast_pbr:
        extra_list.extend([ref_strength, ref_roughness, ref_tint])  # [P, 1+1+3]
    if pc.use_pbr:
        extra_list.extend([base_color, roughness.clamp(0.04, 1.0), metallic, incidents_rgb])
    extra_features_tensor = torch.cat(extra_list, dim=-1)  # [P, N]

    # ---- Single V2 rasterizer call (color + extra + geometry) ----
    rendered_image, rendered_extra, radii, depth_raw, alpha, normal_raw = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        shs=shs, colors_precomp=colors_precomp_pass1,
        extra_features=extra_features_tensor,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
    )
    rendered_opacity = alpha
    visibility_filter = radii > 0
    depth = depth_raw

    # ---- Pseudo-normal from depth (loss supervision for Gaussian normals) ----
    pseudo_normal = None
    if not fast_pbr:
        with torch.no_grad():
            pseudo_normal = _erp_depth_to_normal(depth, rendered_opacity)
        c2w_rot = c2w[:3, :3].to(device=pseudo_normal.device, dtype=pseudo_normal.dtype)
        pseudo_normal = c2w_rot @ pseudo_normal.reshape(3, -1)              # view → world
        pseudo_normal = F.normalize(pseudo_normal.reshape(3, H, W), dim=0)

    # ---- Alpha-normalize extra features ----
    alpha_mask = (rendered_opacity > 0).float()
    opacity_for_div = rendered_opacity.clamp_min(1e-5)
    rendered_extra = rendered_extra / opacity_for_div * alpha_mask  # [N, H, W]

    # ---- Slice extra into individual attribute maps ----
    offset = 0
    rendered_normal = rendered_extra[0:3] * 2.0 - 1.0
    rendered_normal = F.normalize(rendered_normal, dim=0)
    offset += 3

    rendered_ref_strength_map = None
    rendered_ref_roughness_map = None
    rendered_ref_tint = None
    if not fast_pbr:
        rendered_ref_strength_map = rendered_extra[offset+0:offset+1]
        rendered_ref_roughness_map = rendered_extra[offset+1:offset+2]
        rendered_ref_tint = rendered_extra[offset+2:offset+5]
        offset += 5

    rendered_base_color_img = None
    rendered_incident_img = None
    rendered_packed = None
    if pc.use_pbr:
        rendered_base_color_img = rendered_extra[offset+0:offset+3]
        roughness_single = rendered_extra[offset+3:offset+4]
        metallic_single = rendered_extra[offset+4:offset+5]
        rendered_incident_img = rendered_extra[offset+5:offset+8]
        # PBR packed tensor for backward compat with downstream code
        rendered_packed = torch.cat([
            roughness_single, metallic_single,
            torch.zeros_like(roughness_single),
        ], dim=0)
        offset += 8

    # ---- Normal-facing visualization (red=back-facing, blue=front-facing) ----
    out_feature_dict = {}
    facing_vis = None
    if not fast_pbr:
        ray_dirs_vis = _equirect_ray_dirs(H, W)
        cam_to_point = F.normalize(
            (ray_dirs_vis.reshape(-1, 3) @ c2w[:3, :3].T).reshape(H, W, 3), dim=-1)
        normal_hw = rendered_normal.permute(1, 2, 0)
        cos_angle_nml = (normal_hw * cam_to_point).sum(dim=-1)
        facing_vis = torch.where(
            cos_angle_nml.unsqueeze(-1) > 0,
            torch.tensor([1.0, 0.1, 0.1], device="cuda"),
            torch.tensor([0.1, 0.3, 1.0], device="cuda"),
        )
        facing_vis = torch.where(
            (rendered_opacity > 0.5).permute(1, 2, 0).expand(-1, -1, 3),
            facing_vis,
            torch.tensor([0.5, 0.5, 0.5], device="cuda"),
        )
        out_feature_dict["normal_facing"] = facing_vis.permute(2, 0, 1)

    # ---- Background blending for main render ----
    opacity_map = rendered_opacity.permute(1, 2, 0)
    radiance_map = rendered_image.permute(1, 2, 0)
    # SGS rasterizer with bg=0 outputs sum(T_i * alpha_i * c_i).
    # Correct final: sum(T_i * alpha_i * c_i) + T_final * bg
    # = rendered_image + (1 - rendered_opacity) * bg_color
    ref_rgb = radiance_map + (1.0 - opacity_map) * bg_color


    # ---- PBR shading ----
    if pc.use_pbr:
        normal_map = rendered_normal.permute(1, 2, 0)
        base_color_map = rendered_base_color_img.permute(1, 2, 0)
        roughness_map = rendered_packed[0:1].permute(1, 2, 0).clamp(0.04, 1.0)
        metallic_map = rendered_packed[1:2].permute(1, 2, 0)

        # Equirect-specific view direction (COLMAP view space, +Y down).
        ray_dirs = _equirect_ray_dirs(H, W)
        view_dirs = F.normalize(
            -(ray_dirs.reshape(-1, 3) @ c2w[:3, :3].T).reshape(H, W, 3), dim=-1)

        if aabb is not None:
            clamp_min, clamp_max = aabb[:3], aabb[3:]
        else:
            cbound = dict_params.get("occlusion_volumes", {}).get("bound", 1.5)
            clamp_min, clamp_max = -cbound, cbound

        # Depth from renderGeometryCUDA is already alpha-weighted normalised:
        #   out_depth = Σ(depth * vis) / Σ(vis)  (radial distance in ERP mode)
        # No need to divide by opacity again.
        surf_depth = depth
        points = (
            (-view_dirs.reshape(-1, 3) * surf_depth.reshape(-1, 1) + c2w[:3, 3])
            .clamp(min=clamp_min, max=clamp_max)
            .contiguous()
        )

        occlusion_map = None
        if occlusion_volumes is not None:
            if aabb is None:
                cbound = occlusion_volumes["bound"]
                aabb = torch.tensor([-cbound, -cbound, -cbound, cbound, cbound, cbound], device="cuda")
            occlusion_map = recon_occlusion(
                H=H, W=W,
                bound=occlusion_volumes["bound"],
                points=points,
                normals=normal_map.reshape(-1, 3).contiguous(),
                roughness=roughness_map.reshape(-1, 1).contiguous(),
                occlusion_coefficients=occlusion_volumes["occlusion_coefficients"],
                occlusion_ids=occlusion_volumes["occlusion_ids"],
                aabb=aabb,
                degree=occlusion_volumes["degree"],
            ).reshape(H, W, 1)

        incident_light_map = rendered_incident_img.permute(1, 2, 0)  # [H, W, 3]
        pbr_result = pbr_shading(
            light=cubemap,
            normals=normal_map,
            view_dirs=view_dirs,
            albedo=base_color_map,
            roughness=roughness_map,
            metallic=metallic_map if pipe.metallic else None,
            occlusion=occlusion_map,
            irradiance=incident_light_map,
            brdf_lut=brdf_lut,
        )
        rendered_pbr = pbr_result["render_rgb"]
        diffuse_pbr = pbr_result["diffuse_rgb"]
        specular_pbr = pbr_result["specular_rgb"]

        # PBR output is NOT pre-multiplied; blend with background via opacity.
        # Detach opacity from PBR gradient — PBR should not optimize opacity.
        rendered_pbr = rendered_pbr * opacity_map + (1.0 - opacity_map) * bg_color

        if pipe.tone_mapping:
            rendered_pbr = torch.clamp(rendered_pbr, 0.0, 1.0)

        out_feature_dict.update({
            "base_color": base_color_map.permute(2, 0, 1),
            "roughness": roughness_map.permute(2, 0, 1),
            "metallic": metallic_map.permute(2, 0, 1),
            "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None
                          else torch.zeros_like(roughness_map).permute(2, 0, 1),
            "diffuse_pbr": diffuse_pbr.permute(2, 0, 1),
            "specular_pbr": specular_pbr.permute(2, 0, 1),
            "image_pbr": rendered_pbr.permute(2, 0, 1),
        })
        if not fast_pbr:
            out_feature_dict.update({
                "incidents_light": pbr_result.get("incidents_light", torch.zeros_like(roughness_map)).permute(2, 0, 1),
                "incident_light_raw": incident_light_map.permute(2, 0, 1),
            })
            if cubemap is not None:
                out_feature_dict["env_export_base"] = cubemap.export_envmap(return_img=True).permute(2, 0, 1)
                out_feature_dict["env_export_diffuse"] = cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1)

    if not fast_pbr:
        out_feature_dict.update({
            "ref_roughness": rendered_ref_roughness_map,
            "ref_strength": rendered_ref_strength_map,
            "ref_tint": rendered_ref_tint,
        })

    # ---- psi, lat, lon for densification ----
    psi, lat, lon = _project_lat_lon(means3D, viewpoint_camera.world_view_transform)

    # ---- Results assembly ----
    results = {
        "render": ref_rgb.permute(2, 0, 1),
        "depth": depth,
        "normal": rendered_normal,
        "opacity": rendered_opacity,
        "pseudo_normal": pseudo_normal,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "num_rendered": 0,
        # Note: no "weights" key — equirect mode uses _equirect_prune_mask
        # which does not rely on weights_accum.  See scene/gaussian_model.py.
        "psi": psi,
        "lat": lat,
        "lon": lon,
    }
    if not fast_pbr:
        results.update({
            "ref_roughness": rendered_ref_roughness_map,
            "ref_strength": rendered_ref_strength_map,
        })
    results.update(out_feature_dict)

    if pc.use_pbr:
        results["pbr"] = gamma_func(rendered_pbr.permute(2, 0, 1))

    if not is_training and not fast_pbr:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        vis_dict = {
            "depth": depth_norm,
            "normal": rendered_normal * 0.5 + 0.5,
            "radiance_color": rendered_image,
            "normal_facing": facing_vis.permute(2, 0, 1),
        }
        vis_dict["pseudo_normal"] = pseudo_normal * 0.5 + 0.5
        vis_dict["ref_strength"] = rendered_ref_strength_map
        vis_dict["ref_roughness"] = rendered_ref_roughness_map
        vis_dict["ref_tint"] = rendered_ref_tint
        if refmap is not None:
            vis_dict["ref_export_base"] = refmap.export_envmap(return_img=True).permute(2, 0, 1)
        if pc.use_pbr:
            normal_map = rendered_normal.permute(1, 2, 0)
            base_color_map = rendered_base_color_img.permute(1, 2, 0) if pc.use_pbr else None
            vis_dict.update({
                "base_color": gamma_func(base_color_map.permute(2, 0, 1)),
                "base_color_rgb": base_color_map.permute(2, 0, 1),
                "roughness": rendered_packed[0:1],
                "metallic": rendered_packed[1:2],
                "diffuse_pbr": gamma_func(diffuse_pbr.permute(2, 0, 1)),
                "specular_pbr": gamma_func(specular_pbr.permute(2, 0, 1)),
                "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None
                              else torch.zeros_like(roughness_map).permute(2, 0, 1),
                "incidents_light": pbr_result.get("incidents_light", torch.zeros_like(roughness_map)).permute(2, 0, 1),
                "incident_light_raw": incident_light_map.permute(2, 0, 1),
                "image_pbr": gamma_func(rendered_pbr.permute(2, 0, 1)),
            })
            if cubemap is not None:
                vis_dict["env_export_base"] = cubemap.export_envmap(return_img=True).permute(2, 0, 1)
                vis_dict["env_export_diffuse"] = cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1)
        results["vis_dict"] = vis_dict

    return results


def calculate_loss(viewpoint_camera, pc, results, opt, env_map=None, use_ws_ssim=False, iteration=0):
    """计算 equirect 全景模式下的所有损失项。

    损失体系分三层，从通用到专用逐步叠加：

    ── 层1: 图像重建（通用） ──────────────────────────────────
      L1 + SSIM：渲染与 GT 之间的光度一致性。

    ── 层2: 全景图几何优化（equirect 特有） ───────────────────
      纬度面积加权:     ERP 极地像素实际球面面积小，用 row_weight
                       让极地 smoothness 自动降权，避免过度平滑天花板。
      RGB 边缘门控:     用 gt_image 梯度检测物体边界，smoothness
                       跨边界时自动降权，保护边缘锐度。
      几何 loss 渐变:   geometry_loss_from_iter 前不生效，之后逐步
                       warmup，防止训练早期几何不稳定时 loss 干扰。
      Alpha 空洞:       (可选) 惩罚 opacity 低于 target 的区域。
      Attribute smooth: ref_roughness/ref_strength 的 edge-aware
                       平滑（cyclic 水平边界防止左右接缝）。
      Normal MSE:       将光栅化最短轴法线拉向深度法线（pseudo_normal），
                       辅助法线学习但不反向传播到深度。
      Normal TV:        最短轴法线的 L1 全变分（cyclic 水平边界），
                       防止 MSE 把法线往噪声方向拉。

    ── 层3: PBR 材质分解（Stage 2 特有） ────────────────────
      PBR recon:        PBR 渲染结果的 L1 + SSIM。
      Attribute smooth: roughness / base_color / metallic 的 edge-aware
                       平滑，驱动物理属性在空间上连续变化。
      环境贴图:         TV 平滑 + 各通道趋于灰色的先验。
    """
    tb_dict = {"num_points": pc.get_xyz.shape[0]}
    rendered_image = results["render"]
    rendered_opacity = results["opacity"]
    gt_image = viewpoint_camera.original_image.cuda()

    loss = 0
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════
    # 层1: 图像重建损失（L1 + SSIM）
    # ════════════════════════════════════════════════════════════
    Ll1 = F.l1_loss(rendered_image, gt_image)
    if use_ws_ssim:
        ws_map = est_wsmap(rendered_image)
        ssim_val, ws_ssim_val = ssim(rendered_image, gt_image, ws_map=ws_map)
        tb_dict["l1"] = Ll1.item()
        tb_dict["ws_ssim"] = ws_ssim_val.item()
        tb_dict["ssim"] = ssim_val.item()
        ssim_loss = 1.0 - ws_ssim_val
    else:
        ssim_val = ssim(rendered_image, gt_image)
        tb_dict["l1"] = Ll1.item()
        tb_dict["ssim"] = ssim_val.item()
        ssim_loss = 1.0 - ssim_val

    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

    # ════════════════════════════════════════════════════════════
    # 层2: 全景图几何优化（equirect 特有）
    # ════════════════════════════════════════════════════════════

    # ── 预计算纬度权重、RGB 边缘门控、几何 loss 渐变系数 ────
    _pano = _ensure_pano_losses()
    _pano_row, _pano_rgb, _geom_ramp_fn, _pano_normal_smooth, _pano_alpha = _pano

    _, H, W = rendered_image.shape

    # row_weight[1,H,1] = cos(lat) 纬度面积加权。
    # ERP 投影中极地像素对应很小的球面面积，赤道像素对应大球面面积。
    # 在 smoothness loss 中乘以该权重，使得极地约束自动减弱，
    # 避免天花板/地板被过度平滑。
    row_weight = _pano_row(H, rendered_image.device, rendered_image.dtype) if _pano_row is not None else None

    # rgb_nonedge[1,H,W] ∈ [0,1]: RGB 边缘门控。
    # 从 gt_image 梯度检测物体边界，在边界处 weight≈0（抑制
    # smoothness 跨过边界），在平滑区域 weight≈1（正常 smooth）。
    # 注意：该 weight 不参与 normal smooth（避免极地和纹理丰富区域
    # 的 normal 约束被过度削弱导致正反馈漂移）。
    # rgb_nonedge = _pano_rgb(gt_image) if _pano_rgb is not None else None
    rgb_nonedge = None

    # geom_ramp[0,1]: 几何 loss 渐变激活系数。
    #   iter < geometry_loss_from_iter        → ramp = 0
    #   iter >> geometry_loss_from_iter+warmup → ramp = 1
    # 防止训练早期几何不稳定时 normal/ref 平滑过多干扰。
    # geom_ramp = _geom_ramp_fn(opt, iteration, "geometry_loss_from_iter") if _geom_ramp_fn is not None else 1.0
    # tb_dict["geom_ramp"] = float(geom_ramp)
    geom_ramp = 1.0

    # ── Mask 熵（透明度 vs 已知前景 mask） ────────────────────
    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    # ── Alpha 空洞损失（自监督，默认关闭 lambda_alpha_hole=0） ─
    # 惩罚 rendered_opacity 低于 target（0.985）的区域。
    # residual_weight 让残差大的像素获得更高权重，推动填补空洞。
    if opt.lambda_alpha_hole > 0 and _pano_alpha is not None:
        loss_alpha_hole = _pano_alpha(
            rendered_opacity, rendered_image, gt_image,
            target=0.985, residual_weight=0.5)
        tb_dict["loss_alpha_hole"] = loss_alpha_hole.item()
        loss = loss + geom_ramp * opt.lambda_alpha_hole * loss_alpha_hole

    # ── 反射粗糙度平滑（ref_roughness） ───────────────────────
    # 目标: 让反射粗糙度在空间上连续，避免椒盐噪声。
    # 使用 _erp_edge_aware_loss = cyclic 水平方向 + 边缘保护。
    # image_mask * rgb_nonedge 确保 smoothness 只在有效区域且不跨边界。
    if opt.lambda_ref_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        rendered_ref_roughness = results.get("ref_roughness")
        if rendered_ref_roughness is not None:
            w = image_mask
            if rgb_nonedge is not None:
                w = w * rgb_nonedge
            loss_ref_roughness_smooth = _erp_edge_aware_loss(
                rendered_ref_roughness * w, gt_image)
            tb_dict["loss_ref_roughness_smooth"] = loss_ref_roughness_smooth.item()
            loss = loss + geom_ramp * opt.lambda_ref_roughness_smooth * loss_ref_roughness_smooth

    # ── 反射强度平滑（ref_strength） ───────────────────────────
    # 同上：让反射强度贴图在空间上连续变化。
    if opt.lambda_ref_strength_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        rendered_ref_strength = results.get("ref_strength")
        if rendered_ref_strength is not None:
            w = image_mask
            if rgb_nonedge is not None:
                w = w * rgb_nonedge
            loss_ref_strength_smooth = _erp_edge_aware_loss(
                rendered_ref_strength * w, gt_image)
            tb_dict["loss_ref_strength_smooth"] = loss_ref_strength_smooth.item()
            loss = loss + geom_ramp * opt.lambda_ref_strength_smooth * loss_ref_strength_smooth

    # ── Normal 一致性（MSE 拉向深度法线） ──────────────────────
    # 把光栅化器输出的最短轴法线（rendered_normal）向深度推导
    # 法线（pseudo_normal）拉近。pseudo_normal 使用 detach()
    # 切断梯度，避免此 loss 反向传播到深度和 position。
    # 注意：rotation 参数仍通过其他路径（PBR 渲染梯度）更新，
    # 因此需要下方的 TV smooth 来防止法线无约束漂移。
    if opt.lambda_normal_render_depth > 0:
        rendered_normal = results["normal"]
        pseudo_normal = results["pseudo_normal"]
        loss_normal_render_depth = F.mse_loss(rendered_normal, pseudo_normal.detach())
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + geom_ramp * opt.lambda_normal_render_depth * loss_normal_render_depth

    # ── Normal TV 平滑（替代 SGS 的二次型 smoothness） ────────
    # L2 TV with cyclic horizontal boundary: L2 penalizes small bumps
    # quadratically, giving stronger suppression than L1 for the same lambda.
    # 使用 row_weight 加权：极地（lat≈±90°, row_weight≈0）平滑约束
    # 自动减轻，避免极地像素密集导致的过度平滑。
    if opt.lambda_normal_smooth > 0:
        rendered_normal = results["normal"]
        dx = (torch.roll(rendered_normal, shifts=-1, dims=-1) - rendered_normal).pow(2)
        dy = torch.zeros_like(rendered_normal)
        if rendered_normal.shape[-2] > 1:
            dy[:, :-1, :] = (rendered_normal[:, 1:, :] - rendered_normal[:, :-1, :]).pow(2)

        # 纬度面积加权：极地平滑约束降权，避免深度/法线极地畸变
        if row_weight is not None:
            w = row_weight.to(dtype=rendered_normal.dtype, device=rendered_normal.device)
            dx = dx * w
            dy = dy * w

        loss_normal_smooth = dx.mean() + dy.mean()
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + geom_ramp * opt.lambda_normal_smooth * loss_normal_smooth

    # ════════════════════════════════════════════════════════════
    # 层3: PBR 材质分解损失（Stage 2 特有）
    # ════════════════════════════════════════════════════════════
    if pc.use_pbr:
        # ── PBR 图像重建（L1 + SSIM） ──────────────────────────
        # rendered_pbr 已经过 gamma_correction
        rendered_pbr = results["pbr"]
        Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
        ssim_val_pbr = ssim(rendered_pbr, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_pbr * loss_pbr

        # ── PBR 粗糙度平滑 ─────────────────────────────────────
        # 使用 _erp_edge_aware_loss（cyclic 水平 + 边缘保护），
        # image_mask * rgb_nonedge 限制在有效区域且不跨 RGB 边界。
        if opt.lambda_roughness_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_roughness = results.get("roughness")
            if rendered_roughness is not None:
                w = image_mask
                if rgb_nonedge is not None:
                    w = w * rgb_nonedge
                loss_roughness_smooth = _erp_edge_aware_loss(rendered_roughness * w, gt_image)
                tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
                loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

        # ── PBR 基本色平滑 ─────────────────────────────────────
        if opt.lambda_base_color_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_base_color = results.get("base_color")
            if rendered_base_color is not None:
                w = image_mask
                if rgb_nonedge is not None:
                    w = w * rgb_nonedge
                loss_base_color_smooth = _erp_edge_aware_loss(rendered_base_color * w, gt_image)
                tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
                loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

        # ── PBR 金属度平滑 ─────────────────────────────────────
        if opt.lambda_metallic_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_metallic = results.get("metallic")
            if rendered_metallic is not None:
                w = image_mask
                if rgb_nonedge is not None:
                    w = w * rgb_nonedge
                loss_metallic_smooth = _erp_edge_aware_loss(rendered_metallic * w, gt_image)
                tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
                loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth

        # ── 环境贴图 TV 平滑 ──────────────────────────────────
        if opt.lambda_env_smooth > 0 and env_map is not None:
            env = env_map.get_env_map()
            loss_env_smooth = tv_loss(env.permute(2, 0, 1))
            tb_dict["loss_env_smooth"] = loss_env_smooth
            loss = loss + opt.lambda_env_smooth * loss_env_smooth

        # ── 环境光照白化（各通道趋于中性灰） ──────────────────
        # 先验：环境光照应在各颜色通道上接近灰色，防止偏色。
        if opt.lambda_white_light > 0 and env_map is not None:
            env_base = env_map.base
            white = (env_base[..., 0:1] + env_base[..., 1:2] + env_base[..., 2:3]) / 3.0
            loss_light_white_blance = torch.mean(torch.abs(env_base - white))
            tb_dict["loss_light_white_blance"] = loss_light_white_blance.item()
            loss = loss + opt.lambda_white_light * loss_light_white_blance

    tb_dict["loss"] = loss.item()
    return loss, tb_dict


def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
           is_training=False, dict_params=None):
    fast_pbr = dict_params.get("fast_pbr", False) if dict_params else False
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params,
                          fast_pbr=fast_pbr)
    if is_training:
        use_ws_ssim = False  # Disabled for memory: weighted SSIM adds ~100+ MB of conv2d intermediates at 4K ERP
        iteration = dict_params.get("iteration", 0) if dict_params else 0
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt,
                                       env_map=dict_params.get('cubemap') if pc.use_pbr else None,
                                       use_ws_ssim=use_ws_ssim, iteration=iteration)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    return results
