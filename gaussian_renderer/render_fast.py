import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from pbr.shade import pbr_shading
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.prt_utils import PRTutils
from utils.sh_utils import eval_sh
from utils.graphics_utils import linear2srgb_torch
from .rtr_gs_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs_ir import recon_occlusion


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    """_summary_

    Args:
        viewpoint_camera (Camera): _description_
        pc (GaussianModel): _description_
        pipe (_type_): _description_
        bg_color (torch.Tensor): _description_
        scaling_modifier (float, optional): _description_. Defaults to 1.0.
        override_color (_type_, optional): _description_. Defaults to None.
        is_training (bool, optional): _description_. Defaults to False.
        dict_params (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    gamma_func = lambda x : linear2srgb_torch(x)

    # Setup environment maps
    if pc.use_pbr:
        cubemap = dict_params["cubemap"]
        if is_training:
            cubemap.train()
            cubemap.build_mips()
        else:
            cubemap.eval()
            # build_mips() should be called once before rendering
            # Assuming it's already called during initialization
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # set attribuates
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    normal = pc.get_min_axis(viewpoint_camera.camera_center)

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    
    # PRT: Compute radiance transfer color if enabled
    only_diffuse = False # dict_params["iteration"] < pipe.diffuse_iteration
    if pipe.compute_with_prt:
        net = dict_params.get("transfer_net", None)
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
        if only_diffuse:
            override_color = PRTutils.cal_diffuse(pc)
        else:
            override_color = PRTutils.cal_color(pc, net, viewdirs, normal, is_training)
    


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Use PRT color if computed, otherwise use zero tensor
    colors_precomp = override_color if override_color is not None else torch.zeros_like(pc.get_base_color)


    features = torch.cat([depths, depths2, normal], dim=-1) # [1, 1, 3]
    base_color = pc.get_base_color
    roughness = pc.get_roughness
    metallic = pc.get_metallic

    # Handle incident light for PBR - support relighting mode
    if pc.use_pbr:
        if not pipe.relight:
            # Use trained incident light
            incidents = pc.get_incidents
            incidents_light = torch.clamp(eval_sh(pc.active_sh_degree, incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2), normal), 0.0, 1.0)
        else:
            # Relighting mode: either use transfer_light or zero out incident light
            if pipe.transfer_light:
                transfer_shs = pc.get_incidents.permute(0, 2, 1)
                light_shs = cubemap.shs
                incidents = light_shs * transfer_shs
                incidents = incidents.permute(0, 2, 1)
                incidents_light = torch.clamp(eval_sh(pc.active_sh_degree, incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2), normal), 0.0, 1.0)
            else:
                # For relighting, zero out incident light (will be computed by PBR shading)
                incidents_light = torch.zeros_like(base_color)

    features = torch.cat([features, base_color, roughness, metallic, incidents_light], dim=-1) # [1, 1, 3, 3, 1, 1, 3]


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    # FOR DEFFERED SHADING
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask   #[N, H, W]

    rendered_depth, rendered_depth2, rendered_normal, rendered_base_color, rendered_roughness, rendered_metallic, rendered_incident_lights \
        = rendered_feature.split([1, 1, 3, 3, 1, 1, 3], dim=0)

    rendered_var = rendered_depth2 - rendered_depth.square()    # [1, H, W]

    # Radiance Shading
    opacity_map = rendered_opacity.permute(1, 2, 0)                                         # [H, W, 1]
    normal_map = rendered_normal.permute(1, 2, 0)                                           # [H, W, 3]
    normal_map = F.normalize(normal_map, dim=-1)

    canonical_rays = dict_params["canonical_rays"]
    c2w = viewpoint_camera.c2w
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]



    roughness_map = rendered_roughness.permute(1, 2, 0)                                 # [H, W, 1]
    roughness_map = torch.clamp(roughness_map, 0.04, 1.0)                               # [H, W, 1]
    metallic_map = rendered_metallic.permute(1, 2, 0)                                   # [H, W, 1]
    base_color_map = rendered_base_color.permute(1, 2, 0)                               # [H, W, 3]
    incident_light_map = rendered_incident_lights.permute(1, 2, 0)                      # [H, W, 3]

    # get occulsion
    aabb = dict_params.get("aabb")
    if aabb is not None:
        clamp_min, clamp_max = aabb[:3], aabb[3:]
    else:
        cbound = dict_params.get("occlusion_volumes", {}).get("bound", 1.5)
        clamp_min, clamp_max = -cbound, cbound
        aabb = torch.tensor([-cbound, -cbound, -cbound, cbound, cbound, cbound], device="cuda")
    points = (
        (-view_dirs.reshape(-1, 3) * rendered_depth.reshape(-1, 1) + c2w[:3, 3])
            .clamp(min=clamp_min, max=clamp_max)
                .contiguous()
            )  # [HW, 3]

    if "occlusion_volumes" in dict_params.keys() and dict_params.get("enable_occlusion", True):
        occlusion_volumes = dict_params["occlusion_volumes"]
        aabb = dict_params["aabb"]
        occlusion_map = recon_occlusion(
                        H=H,
                        W=W,
                        bound = occlusion_volumes["bound"],
                        points = points,
                        normals = normal_map.reshape(-1, 3).contiguous(),
                        roughness = roughness_map.reshape(-1, 1).contiguous(),
                        occlusion_coefficients = occlusion_volumes["occlusion_coefficients"],
                        occlusion_ids= occlusion_volumes["occlusion_ids"],
                        aabb = aabb,
                        degree = occlusion_volumes["degree"],
                    ).reshape(H, W, 1)
        
        # # 调试：检查 occlusion_map 的值范围
        # print("Occlusion map min:", occlusion_map.min().item())
        # print("Occlusion map max:", occlusion_map.max().item())
        # print("Occlusion map mean:", occlusion_map.mean().item())
        
        # 如果 occlusion_map 的值接近 1，说明没有阴影
        if occlusion_map.mean().item() > 0.95:
            print("Warning: Occlusion map is almost all 1 (no shadow effect)")
    else:
        occlusion_map = None
    
    pbr_result = pbr_shading(
        light=cubemap,
        normals = normal_map,  # [H, W, 3]
        view_dirs = view_dirs,   # [H, W, 3]
        albedo = base_color_map,  # [H, W, 3]
        roughness = roughness_map,  # [H, W, 1]
        metallic = metallic_map if pipe.metallic else None,    # [H, W, 1]
        occlusion = occlusion_map if occlusion_map is not None else None,  # [H, W, 1]
        irradiance = incident_light_map if not pipe.relight else None,     # [H, W, 3]
        brdf_lut=dict_params["brdf_lut"],
    )

    rendered_pbr = pbr_result["render_rgb"] # [H, W, 3]
    rendered_pbr = rendered_pbr * opacity_map + (1.0 - opacity_map) * bg_color

    if pipe.tone_mapping:
        rendered_pbr = torch.clamp(rendered_pbr, 0.0, 1.0)
        




        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": rendered_pbr.permute(2, 0, 1),
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "normal": normal_map.permute(2, 0, 1),
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib,
               "weights": weights
               }
    results['pbr'] = gamma_func(rendered_pbr.permute(2, 0, 1))
    results['vis_dict'] = {}
    return results






def render_fast(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None, **kwargs):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)
    return results