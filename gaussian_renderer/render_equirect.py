import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from pbr.shade import get_reflectance_color, get_reflectance_color_forward, pbr_shading
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.prt_utils import PRTutils
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, tv_loss, first_order_edge_aware_loss, est_wsmap
from utils.image_utils import psnr
from utils.graphics_utils import linear2srgb_torch
from odgs_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs_ir import recon_occlusion


def _compute_equirect_view_dirs(H, W, c2w, device='cuda'):
    row = torch.arange(H, device=device).float()
    col = torch.arange(W, device=device).float()
    y, x = torch.meshgrid(row, col, indexing='ij')
    lon = (x / W) * 2 * math.pi - math.pi
    lat = math.pi / 2 - (y / H) * math.pi
    cos_lat = torch.cos(lat)
    dir_x = cos_lat * torch.sin(lon)
    dir_y = torch.sin(lat)
    dir_z = cos_lat * torch.cos(lon)
    ray_dirs = torch.stack([dir_x, dir_y, dir_z], dim=-1)
    view_dirs = -(ray_dirs.reshape(-1, 3) @ c2w[:3, :3].T).reshape(H, W, 3)
    return F.normalize(view_dirs, dim=-1)


def _run_odgs_rasterizer(means3D, means2D, colors_precomp, opacities, scales, rotations,
                         rasterizer, shs=None, cov3D_precomp=None):
    if shs is None:
        shs = torch.Tensor([]).cuda()
    if cov3D_precomp is None:
        cov3D_precomp = torch.Tensor([]).cuda()
    return rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    gamma_func = lambda x: linear2srgb_torch(x)

    refmap = dict_params.get("refmap")
    cubemap = dict_params.get("cubemap") if pc.use_pbr else None
    transfer_net = dict_params.get("transfer_net")
    occlusion_volumes = dict_params.get("occlusion_volumes")
    aabb = dict_params.get("aabb")
    brdf_lut = dict_params.get("brdf_lut")
    canonical_rays = dict_params.get("canonical_rays")

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

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        bg=torch.zeros(3, device='cuda'),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
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
    depths2 = depths.square()

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

    # ODGS uses isotropic scaling [N,1]; extract mean of 3 anisotropic axes
    scales_odgs = scales.mean(dim=-1, keepdim=True) if scales is not None else None

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

    # ---- Pass 1: Forward-shaded rendering (PRT + optional forward reflection) ----
    viewdirs_gauss = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    if pipe.forward_shading and refmap is not None:
        refl_color_forward = get_reflectance_color_forward(
            refmap, normal, viewdirs_gauss, ref_roughness, ref_tint, brdf_lut=brdf_lut)
        colors_precomp_pass1 = (1.0 - ref_strength) * colors_precomp + ref_strength * refl_color_forward
    else:
        colors_precomp_pass1 = colors_precomp

    rendered_image, depth, acc, radii, psi, lat, lon = _run_odgs_rasterizer(
        means3D, means2D, colors_precomp_pass1, opacity, scales_odgs, rotations,
        rasterizer, shs=shs, cov3D_precomp=cov3D_precomp,
    )
    rendered_opacity = acc
    visibility_filter = radii > 0

    render_depth_expected = depth / rendered_opacity.clamp_min(1e-5)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    surf_depth = render_depth_expected

    # ---- Multi-pass attribute rendering (reuse rasterizer, different colors_precomp) ----
    # Pass 2: Normal map (encode as n*0.5+0.5 → [0,1] range)
    colors_precomp_normal = normal * 0.5 + 0.5
    rendered_normal_img, _, _, _, _, _, _ = _run_odgs_rasterizer(
        means3D, means2D, colors_precomp_normal, opacity, scales_odgs, rotations,
        rasterizer, shs=torch.Tensor([]).cuda(), cov3D_precomp=cov3D_precomp,
    )
    rendered_normal = rendered_normal_img * 2.0 - 1.0
    rendered_normal = F.normalize(rendered_normal, dim=0)

    # Pass 3: Base color + ref_tint (PBR mode)
    if pc.use_pbr:
        base_color = pc.get_base_color
        colors_precomp_base = base_color
        rendered_base_color_img, _, _, _, _, _, _ = _run_odgs_rasterizer(
            means3D, means2D, colors_precomp_base, opacity, scales_odgs, rotations,
            rasterizer, shs=torch.Tensor([]).cuda(), cov3D_precomp=cov3D_precomp,
        )

        roughness = pc.get_roughness
        roughness_clamped = torch.clamp(roughness, 0.04, 1.0)
        metallic = pc.get_metallic
        # Pass 4: Packed attributes (roughness, metallic, depth)
        # Encode as RGB: R=roughness, G=metallic, B=depth_normalized
        depth_attr = depths / depths.max().clamp_min(1e-6)
        colors_precomp_packed = torch.cat([
            roughness_clamped,
            metallic,
            depth_attr,
        ], dim=-1)
        rendered_packed, _, _, _, _, _, _ = _run_odgs_rasterizer(
            means3D, means2D, colors_precomp_packed, opacity, scales_odgs, rotations,
            rasterizer, shs=torch.Tensor([]).cuda(), cov3D_precomp=cov3D_precomp,
        )

    # ---- Deferred shading setup ----
    opacity_map = rendered_opacity.permute(1, 2, 0)
    radiance_map = rendered_image.permute(1, 2, 0)
    normal_map = rendered_normal.permute(1, 2, 0)

    if pc.use_pbr:
        base_color_map = rendered_base_color_img.permute(1, 2, 0)
        roughness_map = rendered_packed[0:1].permute(1, 2, 0).clamp(0.04, 1.0)
        metallic_map = rendered_packed[1:2].permute(1, 2, 0)

    # ---- Reflection rendering (forward-shaded, deferred not needed) ----
    ref_rgb = radiance_map * opacity_map + (1.0 - opacity_map) * bg_color

    out_feature_dict = {}

    # ---- PBR shading ----
    if pc.use_pbr:
        view_dirs = _compute_equirect_view_dirs(H, W, c2w)
        points = (
            (-view_dirs.reshape(-1, 3) * depth.reshape(-1, 1) + c2w[:3, 3])
            .clamp(min=-1.5, max=1.5)
            .contiguous()
        )

        occlusion_map = None
        if occlusion_volumes is not None:
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

        pbr_result = pbr_shading(
            light=cubemap,
            normals=normal_map,
            view_dirs=view_dirs,
            albedo=base_color_map,
            roughness=roughness_map,
            metallic=metallic_map if pipe.metallic else None,
            occlusion=occlusion_map,
            brdf_lut=brdf_lut,
        )
        rendered_pbr = pbr_result["render_rgb"]
        diffuse_pbr = pbr_result["diffuse_rgb"]
        specular_pbr = pbr_result["specular_rgb"]

        rendered_pbr = rendered_pbr * opacity_map + (1.0 - opacity_map) * bg_color

        if pipe.tone_mapping:
            rendered_pbr = torch.clamp(rendered_pbr, 0.0, 1.0)

        out_feature_dict.update({
            "base_color": base_color_map.permute(2, 0, 1),
            "roughness": roughness_map.permute(2, 0, 1),
            "metallic": metallic_map.permute(2, 0, 1),
            "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None
                          else torch.zeros_like(roughness_map).permute(2, 0, 1),
        })

    # ---- Results assembly ----
    results = {
        "render": ref_rgb.permute(2, 0, 1),
        "depth": depth,
        "normal": rendered_normal,
        "opacity": rendered_opacity,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "num_rendered": rendered_image.shape[0],
    }
    results.update(out_feature_dict)

    if pc.use_pbr:
        results["pbr"] = rendered_pbr.permute(2, 0, 1)

    # ---- Visualization dict (non-training mode) ----
    if not is_training:
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        vis_dict = {
            "depth": depth_norm,
            "normal": rendered_normal * 0.5 + 0.5,
            "radiance_color": rendered_image,
        }
        if refmap is not None:
            vis_dict["ref_export_base"] = refmap.export_envmap(return_img=True).permute(2, 0, 1)
        if pc.use_pbr:
            vis_dict.update({
                "base_color": gamma_func(base_color_map.permute(2, 0, 1)),
                "roughness": roughness_map.permute(2, 0, 1),
                "metallic": metallic_map.permute(2, 0, 1),
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

    if opt.lambda_normal_render_depth > 0:
        rendered_normal = results["normal"]
        image_mask = viewpoint_camera.image_mask.cuda()
        pseudo_normal = F.normalize(
            torch.nn.functional.conv2d(rendered_opacity, torch.ones(1, 1, 3, 3, device='cuda') / 9, padding=1).detach(),
            dim=0)
        loss_normal_render_depth = F.mse_loss(rendered_normal * image_mask, pseudo_normal * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

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
            rendered_roughness = results["roughness"]
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
