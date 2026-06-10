import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from pbr.shade import get_reflectance_color_forward, pbr_shading
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.prt_utils import PRTutils
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, tv_loss, first_order_edge_aware_loss, est_wsmap
from utils.image_utils import psnr
from utils.graphics_utils import linear2srgb_torch
from spherical_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs_ir import recon_occlusion


def _equirect_ray_dirs(H, W, device='cuda'):
    """Equirectangular pixel → world-space ray directions (view-space)."""
    ys = torch.linspace(0.5 * math.pi, -0.5 * math.pi, H, device=device)
    xs = torch.linspace(-math.pi, math.pi, W, device=device)
    lat, lon = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([
        torch.sin(lon) * torch.cos(lat),
        torch.sin(lat),
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


def _run_sgs_rasterizer(means3D, means2D, colors_precomp, opacities, scales, rotations,
                        rasterizer, shs=None, cov3D_precomp=None):
    rendered_image, radii, depth_raw, alpha, normal_raw = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    return rendered_image, depth_raw, alpha, radii, normal_raw


def _normal_from_raw(normal_raw, alpha, eps=1e-8):
    """Normalize raw rasterizer normal output and mask with alpha."""
    opacity_for_div = alpha.clamp_min(1e-5)
    normal = F.normalize(normal_raw / opacity_for_div, dim=0, eps=eps)
    alpha_mask = (alpha > 0).float()
    normal = normal * alpha_mask
    return normal


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
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
        render_depth=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    ref_tint = pc.get_ref_tint
    ref_roughness = pc.get_ref_roughness
    ref_strength = pc.get_ref_strength
    normal = pc.get_min_axis(viewpoint_camera.camera_center)

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]

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

    # ---- Pass 1: Forward-shaded rendering (PRT + forward reflection) ----
    viewdirs_gauss = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    if pipe.forward_shading and refmap is not None:
        refl_color_forward = get_reflectance_color_forward(
            refmap, normal, viewdirs_gauss, ref_roughness, ref_tint, brdf_lut=brdf_lut)
        colors_precomp_pass1 = (1.0 - ref_strength) * colors_precomp + ref_strength * refl_color_forward
    else:
        colors_precomp_pass1 = colors_precomp

    rendered_image, depth, acc, radii, pass1_normal_raw = _run_sgs_rasterizer(
        means3D, means2D, colors_precomp_pass1, opacity, scales, rotations,
        rasterizer, shs=shs, cov3D_precomp=cov3D_precomp,
    )
    rendered_opacity = acc
    visibility_filter = radii > 0

    # ---- Pseudo-normal from rasterizer normal_raw ----
    pseudo_normal = _normal_from_raw(pass1_normal_raw, rendered_opacity)

    # ---- Alpha normalization mask for multi-pass outputs ----
    alpha_mask = (rendered_opacity > 0).float()
    opacity_for_div = rendered_opacity.clamp_min(1e-5)

    # ---- Pass 2: Normal map (encode as n*0.5+0.5 → [0,1] range) ----
    colors_precomp_normal = normal * 0.5 + 0.5
    rendered_normal_img, _, _, _, _ = _run_sgs_rasterizer(
        means3D, means2D, colors_precomp_normal, opacity, scales, rotations,
        rasterizer, cov3D_precomp=cov3D_precomp,
    )
    rendered_normal_img = rendered_normal_img / opacity_for_div * alpha_mask
    rendered_normal = rendered_normal_img * 2.0 - 1.0
    rendered_normal = F.normalize(rendered_normal, dim=0)

    # ---- Pass 3: Ref_strength + ref_roughness + ref_tint (for visualization) ----
    colors_precomp_refs = torch.cat([ref_strength, ref_roughness, torch.zeros_like(ref_strength)], dim=-1)
    rendered_refs, _, _, _, _ = _run_sgs_rasterizer(
        means3D, means2D, colors_precomp_refs, opacity, scales, rotations,
        rasterizer, cov3D_precomp=cov3D_precomp,
    )
    rendered_refs = rendered_refs / opacity_for_div * alpha_mask
    rendered_ref_strength_map = rendered_refs[0:1]
    rendered_ref_roughness_map = rendered_refs[1:2]

    # ---- Pass 3b: Ref_tint map (RGB) ----
    rendered_ref_tint, _, _, _, _ = _run_sgs_rasterizer(
        means3D, means2D, ref_tint, opacity, scales, rotations,
        rasterizer, cov3D_precomp=cov3D_precomp,
    )
    rendered_ref_tint = rendered_ref_tint / opacity_for_div * alpha_mask

    # ---- Pass 4,5: PBR attributes (PBR mode only) ----
    if pc.use_pbr:
        base_color = pc.get_base_color
        roughness = pc.get_roughness
        metallic = pc.get_metallic

        # Pass 4: Base color (3 channels)
        colors_precomp_base = base_color
        rendered_base_color_img, _, _, _, _ = _run_sgs_rasterizer(
            means3D, means2D, colors_precomp_base, opacity, scales, rotations,
            rasterizer, cov3D_precomp=cov3D_precomp,
        )
        rendered_base_color_img = rendered_base_color_img / opacity_for_div * alpha_mask

        # Pass 5: Packed (R=roughness, G=metallic, B=depth_normalized)
        depth_attr = depths / depths.max().clamp_min(1e-6)
        colors_precomp_packed = torch.cat([
            roughness.clamp(0.04, 1.0),
            metallic,
            depth_attr,
        ], dim=-1)
        rendered_packed, _, _, _, _ = _run_sgs_rasterizer(
            means3D, means2D, colors_precomp_packed, opacity, scales, rotations,
            rasterizer, cov3D_precomp=cov3D_precomp,
        )
        rendered_packed = rendered_packed / opacity_for_div * alpha_mask

        # ---- Pass 6: Incident light (3 channels) ----
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

        rendered_incident_img, _, _, _, _ = _run_sgs_rasterizer(
            means3D, means2D, incidents_rgb, opacity, scales, rotations,
            rasterizer, cov3D_precomp=cov3D_precomp,
        )
        rendered_incident_img = rendered_incident_img / opacity_for_div * alpha_mask

    # ---- Background blending for main render ----
    opacity_map = rendered_opacity.permute(1, 2, 0)
    radiance_map = rendered_image.permute(1, 2, 0)
    # SGS rasterizer with bg=0 outputs sum(T_i * alpha_i * c_i).
    # Correct final: sum(T_i * alpha_i * c_i) + T_final * bg
    # = rendered_image + (1 - rendered_opacity) * bg_color
    ref_rgb = radiance_map + (1.0 - opacity_map) * bg_color

    out_feature_dict = {}

    # ---- PBR shading ----
    if pc.use_pbr:
        normal_map = rendered_normal.permute(1, 2, 0)
        base_color_map = rendered_base_color_img.permute(1, 2, 0)
        roughness_map = rendered_packed[0:1].permute(1, 2, 0).clamp(0.04, 1.0)
        metallic_map = rendered_packed[1:2].permute(1, 2, 0)

        # Equirect-specific view direction
        ray_dirs = _equirect_ray_dirs(H, W)
        view_dirs = F.normalize(
            -(ray_dirs.reshape(-1, 3) @ c2w[:3, :3].T).reshape(H, W, 3), dim=-1)

        if aabb is not None:
            clamp_min, clamp_max = aabb[:3], aabb[3:]
        else:
            cbound = dict_params.get("occlusion_volumes", {}).get("bound", 1.5)
            clamp_min, clamp_max = -cbound, cbound

        points = (
            (-view_dirs.reshape(-1, 3) * depth.reshape(-1, 1) + c2w[:3, 3])
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
        rendered_pbr = rendered_pbr * opacity_map.detach() + (1.0 - opacity_map.detach()) * bg_color

        if pipe.tone_mapping:
            rendered_pbr = torch.clamp(rendered_pbr, 0.0, 1.0)

        out_feature_dict.update({
            "base_color": base_color_map.permute(2, 0, 1),
            "roughness": roughness_map.permute(2, 0, 1),
            "metallic": metallic_map.permute(2, 0, 1),
            "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None
                          else torch.zeros_like(roughness_map).permute(2, 0, 1),
            "incidents_light": pbr_result.get("incidents_light", torch.zeros_like(roughness_map)).permute(2, 0, 1),
            "incident_light_raw": incident_light_map.permute(2, 0, 1),
            "diffuse_pbr": diffuse_pbr.permute(2, 0, 1),
            "specular_pbr": specular_pbr.permute(2, 0, 1),
            "image_pbr": rendered_pbr.permute(2, 0, 1),
        })

        if cubemap is not None:
            out_feature_dict["env_export_base"] = cubemap.export_envmap(return_img=True).permute(2, 0, 1)
            out_feature_dict["env_export_diffuse"] = cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1)

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
        "ref_roughness": rendered_ref_roughness_map,
        "ref_strength": rendered_ref_strength_map,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "num_rendered": 0,
        "weights": opacity,
        "psi": psi,
        "lat": lat,
        "lon": lon,
    }
    results.update(out_feature_dict)

    if pc.use_pbr:
        results["pbr"] = rendered_pbr.permute(2, 0, 1)

    if not is_training:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        vis_dict = {
            "depth": depth_norm,
            "normal": rendered_normal * 0.5 + 0.5,
            "radiance_color": rendered_image,
        }
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
                "roughness": rendered_packed[0:1],
                "metallic": rendered_packed[1:2],
                "diffuse_pbr": diffuse_pbr.permute(2, 0, 1),
                "specular_pbr": specular_pbr.permute(2, 0, 1),
            })
            if cubemap is not None:
                vis_dict["env_export_base"] = cubemap.export_envmap(return_img=True).permute(2, 0, 1)
        results["vis_dict"] = vis_dict

    return results


def calculate_loss(viewpoint_camera, pc, results, opt, env_map=None, use_ws_ssim=False):
    tb_dict = {"num_points": pc.get_xyz.shape[0]}
    rendered_image = results["render"]
    rendered_opacity = results["opacity"]
    gt_image = viewpoint_camera.original_image.cuda()

    loss = 0
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

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_ref_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        rendered_ref_roughness = results.get("ref_roughness")
        if rendered_ref_roughness is not None:
            loss_ref_roughness_smooth = first_order_edge_aware_loss(
                rendered_ref_roughness * image_mask, gt_image)
            tb_dict["loss_ref_roughness_smooth"] = loss_ref_roughness_smooth.item()
            loss = loss + opt.lambda_ref_roughness_smooth * loss_ref_roughness_smooth

    if opt.lambda_ref_strength_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        rendered_ref_strength = results.get("ref_strength")
        if rendered_ref_strength is not None:
            loss_ref_strength_smooth = first_order_edge_aware_loss(
                rendered_ref_strength * image_mask, gt_image)
            tb_dict["loss_ref_strength_smooth"] = loss_ref_strength_smooth.item()
            loss = loss + opt.lambda_ref_strength_smooth * loss_ref_strength_smooth

    if opt.lambda_normal_render_depth > 0:
        rendered_normal = results["normal"]
        pseudo_normal = results["pseudo_normal"]
        loss_normal_render_depth = F.mse_loss(rendered_normal, pseudo_normal.detach())
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_smooth > 0:
        rendered_normal = results["normal"]
        loss_normal_smooth = tv_loss(rendered_normal)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth

    if pc.use_pbr:
        rendered_pbr = results["pbr"]
        Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
        ssim_val_pbr = ssim(rendered_pbr, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_pbr * loss_pbr

        if opt.lambda_roughness_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_roughness = results.get("roughness")
            if rendered_roughness is not None:
                loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
                tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
                loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

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
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)
    if is_training:
        use_ws_ssim = getattr(pipe, 'equirect', False)
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt,
                                       env_map=dict_params.get('cubemap') if pc.use_pbr else None,
                                       use_ws_ssim=use_ws_ssim)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    return results
