import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from pbr.shade import get_reflectance_color, get_reflectance_color_forward, pbr_shading, point_light_shading
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.prt_utils import PRTutils
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, tv_loss, first_order_edge_aware_loss
from utils.image_utils import psnr
from utils.graphics_utils import  linear2srgb_torch
from .rtr_gs_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs_ir import recon_occlusion
import nvdiffrast.torch as dr


def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    """渲染指定视角的高斯模型，支持基础反射渲染和PBR渲染

    Args:
        viewpoint_camera (Camera): 相机参数，包含视角、位置、内参等
        pc (GaussianModel): 高斯模型，包含3D高斯点的属性
        pipe: 渲染管线配置参数，控制渲染行为
            - compute_with_prt: 是否使用PRT计算反射
            - diffuse_iteration: 漫反射迭代次数
            - forward_shading: 是否前向渲染
            - relight: 重光照模式，关闭时使用 incidents_light 计算 pbr 间接漫反射
            - transfer_light: 仅relight模式有效。是否使用传输光照
            - metallic: 是否开启金属性
            - tone_mapping: 是否限制rendered_pbr在0-1之间
        bg_color (torch.Tensor): 背景颜色，形状为[3]
        scaling_modifier (float, optional): 缩放修改器，用于调整高斯点大小。Defaults to 1.0.
        override_color (torch.Tensor, optional): 覆盖颜色，如果提供则使用此颜色替代SH计算。Defaults to None.
        is_training (bool, optional): 是否为训练模式，影响反射贴图和环境贴图的模式。Defaults to False.
        dict_params (dict, optional): 参数字典，包含渲染所需的各种参数。Defaults to None.
            - refmap: 反射贴图 (混合渲染里的延迟反射渲染)
            - iteration: 当前迭代次数 (< pipe.diffuse_iteration则prt仅计算漫反射)
            - canonical_rays: 规范光线
            - brdf_lut: BRDF查找表
            - transfer_net: PRT传输网络（可选）
            - cubemap: 环境贴图（当pc.use_pbr=True时）
                - cubemap.diffuse 是预滤波后的漫反射cubemap，沿法线查询
                - cubemap.specular 沿反射方向查询，根据粗糙度决定采样哪个 mip
            - enable_occlusion: 是否开启遮挡计算
            - occlusion_volumes: 遮挡体积（可选）
            - aabb: 轴对齐包围盒（可选）
            - iso_mode: 是否开启各项同性scale（仅viewer里测试使用）

    Returns:
        dict: 包含渲染结果的字典，包含以下键：
            - vis_dict: 可视化字典，包含各种中间结果（仅在非训练模式时用于可视化查看），包含以下键：
                - surf_depth: alpha 归一化的深度图 [1, H, W]
                - depth: alpha 归一化+归一化(0-1)深度图 [1, H, W]
                - normal: 归一化法线图 [3, H, W]（0.5+0.5，0~1）
                - pseudo_normal: 归一化伪法线图 [3, H, W]（0.5+0.5，0~1）
                混合渲染-PRT：
                - radiance_color: PRT辐射分量 [3, H, W]
                - blended_radiance: PRT辐射*(1-反射强度) [3, H, W]
                混合渲染-延迟反射：
                - ref_color: 延迟反射分量 [3, H, W]
                - blended_ref_color: P延迟反射分量*反射强度 [3, H, W]
                - ref_tint: 反射色调图 [3, H, W]
                - ref_roughness: 反射粗糙度图 [1, H, W]
                - ref_strength: 反射强度图 [1, H, W]
                延迟反射refmap：
                - ref_export_base: 反射环境贴图导出 [3, H, W]
                pbr：
                - base_color_rgb: 基础颜色图（原始RGB）[3, H, W]（当pc.use_pbr=True时）
                - roughness: 粗糙度图 [1, H, W]（当pc.use_pbr=True时）
                - metallic: 金属度图 [1, H, W]（当pc.use_pbr=True时）
                - visibility: 可见性/遮挡图 [1, H, W]（当pc.use_pbr=True时）
                pbr(gamma校正)：
                - base_color: 基础颜色图（gamma校正）[3, H, W]（当pc.use_pbr=True时）
                - diffuse_pbr: PBR漫反射分量 [3, H, W]（当pc.use_pbr=True时）
                - specular_pbr: PBR镜面反射分量 [3, H, W]（当pc.use_pbr=True时）
                - image_pbr: PBR完整图像 [3, H, W]（当pc.use_pbr=True时）
                pbr其他：
                - incidents_light: (遮挡后)间接漫反射光照 [1, H, W]（当pc.use_pbr=True时）
                - incident_light_raw: (原始)间接漫反射光照 [1, H, W]（当pc.use_pbr=True时）
                - env_export_base: 环境贴图导出 [3, H, W]（当pc.use_pbr=True时）
                - env_export_diffuse: 环境贴图漫反射导出 [3, H, W]（当pc.use_pbr=True时）
            - render: 混合渲染分支图像 [3, H, W]
            - depth: (α归一化)深度图 [1, H, W]
            - depth_var: 深度方差 [1, H, W]
            - normal: (α归一化)法线图 [3, H, W]
            - pseudo_normal: 伪法线图 [3, H, W]
            - surface_xyz: 表面3D坐标 [3, H, W]
            - opacity: 透明度图 [1, H, W]
            - viewspace_points: 视图空间点 [N, 2]
            - visibility_filter: 可见性过滤(r>0) [N]
            - radii: 高斯半径 [N]
            - num_rendered: 渲染的高斯数量
            - num_contrib: 贡献的高斯数量
            - weights: 权重 [N, H, W]
            - pbr: (gamma校正)PBR渲染结果 [3, H, W]（当pc.use_pbr=True时）
            特征：
            - ref_roughness: 反射粗糙度图 [1, H, W]
            - ref_strength: 反射强度图 [1, H, W]
            - base_color: 基础颜色图 [3, H, W]（当pc.use_pbr=True时）
            - roughness: 粗糙度图 [1, H, W]（当pc.use_pbr=True时）
            - metallic: 金属度图 [1, H, W]（当pc.use_pbr=True时）
            - visibility: 可见性/遮挡图 [1, H, W]（当pc.use_pbr=True时）
            - pbr_env: PBR与环境贴图的混合结果 [3, H, W]（当pc.use_pbr=True且非训练模式时）
            - env_only: 纯环境贴图 [3, H, W]（当pc.use_pbr=True且非训练模式时）
    """
    # gamma校正函数：将线性空间颜色转换到sRGB空间
    gamma_func = lambda x : linear2srgb_torch(x)
    
    # 反射贴图（用于延迟反射渲染）
    refmap = dict_params["refmap"]

    # 如果启用了PBR模式，从参数字典中获取环境贴图（cubemap）
    if pc.use_pbr:
        cubemap = dict_params["cubemap"]
    
    # 根据训练/推理模式设置refmap和cubemap的状态
    # 训练模式：启用dropout等随机性，构建多分辨率金字塔（MIP）
    if is_training:
        refmap.train()
        refmap.build_mips()
        if pc.use_pbr:
            cubemap.train()
            cubemap.build_mips()
    # 推理模式：关闭dropout，使用确定性推理
    else:
        refmap.eval()
        if pc.use_pbr:
            cubemap.eval()

    
    # 创建屏幕空间点张量，用于接收CUDA光栅器返回的2D屏幕空间均值梯度
    # 这个零张量会在反向传播时接收屏幕空间位置的梯度
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 保留该张量的梯度，用于后续的高斯分裂/克隆决策
        screenspace_points.retain_grad()
    except:
        pass

    # 计算视角的半角正切值，用于构建透视投影矩阵
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    # 获取相机内参矩阵（包含主点坐标）
    intrinsic = viewpoint_camera.intrinsics

    # 构建光栅化配置对象，设置渲染所需的相机和投影参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 图像高度
        image_width=int(viewpoint_camera.image_width),    # 图像宽度
        tanfovx=tanfovx,                                  # X方向视场角正切
        tanfovy=tanfovy,                                  # Y方向视场角正切
        cx=float(intrinsic[0, 2]),                        # 主点X坐标（光心）
        cy=float(intrinsic[1, 2]),                        # 主点Y坐标（光心）
        bg=torch.zeros_like(bg_color),                    # 背景色（初始化为零）
        scale_modifier=scaling_modifier,                  # 高斯缩放修改因子
        viewmatrix=viewpoint_camera.world_view_transform, # 世界坐标到视图坐标的变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,  # 完整的投影变换矩阵
        sh_degree=pc.active_sh_degree,                    # 当前激活的球谐函数阶数
        campos=viewpoint_camera.camera_center,            # 相机中心位置（世界坐标）
        prefiltered=False,                                # 是否使用预滤波（关闭）
        backward_geometry=True,                           # 启用几何反向传播（用于优化）
        computer_pseudo_normal=True,                      # 计算伪法线（用于法线监督）
        debug=pipe.debug                                  # 调试模式开关
    )

    # 创建光栅化器实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # 准备高斯属性：从模型中提取3D位置、透明度、反射相关属性
    means3D = pc.get_xyz                                    # 3D高斯中心位置 [N, 3]
    means2D = screenspace_points                            # 2D屏幕空间位置（初始为零，由光栅器计算）
    opacity = pc.get_opacity                                # 高斯不透明度 [N, 1]
    ref_tint = pc.get_ref_tint                              # 反射色调 [N, 3]
    ref_roughness = pc.get_ref_roughness                    # 反射粗糙度 [N, 1]
    ref_strength = pc.get_ref_strength                      # 反射强度 [N, 1]
    # 计算法线：通过高斯协方差矩阵的最短轴确定，并朝向相机方向
    normal = pc.get_min_axis(viewpoint_camera.camera_center)


    # 计算从相机中心到高斯点的方向向量
    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    # 归一化方向向量，用于后续的球谐函数求值
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    

    # 计算深度：将3D点变换到视图空间，提取Z分量作为深度
    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)  # 齐次坐标 [N, 4]
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]       # 视图空间Z深度 [N, 1]
    depths2 = depths.square()                                                 # 深度平方，用于计算深度方差
    
    # 判断是否仅使用漫反射模式（早期迭代阶段，避免高频反射干扰几何优化）
    only_diffuse = dict_params["iteration"] < pipe.diffuse_iteration
    # 如果使用PRT（预计算辐射传输）计算颜色，且没有提供覆盖颜色
    # ANCHOR[id=prt] prt辐射度计算
    if pipe.compute_with_prt and override_color is None:
        net = dict_params["transfer_net"]  # 获取神经辐射传输网络（MLP）
        # 计算从相机到高斯点的视线方向
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
        if only_diffuse:
            # 仅漫反射模式：使用PRTutils计算漫反射颜色（低频光照）
            prt_color = PRTutils.cal_diffuse(pc)
        else:
            # 完整模式：使用MLP解码视角相关的辐射传输特征
            prt_color = PRTutils.cal_color(pc, net, viewdirs,  normal, is_training)
        # 将PRT计算的颜色作为覆盖颜色
        override_color = prt_color
    # 如果同时提供了PRT计算和覆盖颜色，触发除零错误（逻辑冲突）
    elif pipe.compute_with_prt and override_color is not None:
        1 / 0




    # 处理3D协方差矩阵：可以选择在Python中预计算，或由CUDA光栅器实时计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # 在Python中预计算协方差矩阵（从缩放和旋转参数）
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 使用原始缩放和旋转，由CUDA光栅器实时计算协方差
        scales = pc.get_scaling
        # 仅 viewer 触发：scale各向同性化模式（将各向异性缩发放平为各向同性）
        if dict_params and dict_params.get('iso_mode', False):
            scales = scales.mean(dim=-1, keepdim=True).repeat(1, 3)
        rotations = pc.get_rotation

    # 处理颜色：可以选择在Python中从SH转换，或由CUDA光栅器实时转换
    shs = None
    colors_precomp = None
    if override_color is None:
        # 没有覆盖颜色（如非PRT），需要从SH系数计算颜色
        if pipe.compute_SHs_python:
            # 在Python中计算SH到RGB的转换
            # 重新计算视线方向（从相机到高斯点）
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            # 重排SH系数：[N, 3, (D+1)^2]，D为SH阶数
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            # 评估球谐函数，得到RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 截断负值，确保颜色非负
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 使用原始SH系数，由CUDA光栅器实时转换
            shs = pc.get_shs
    else:
        # 使用提供的覆盖颜色（例如PRT计算的颜色）
        colors_precomp = override_color


    # 获取规范光线（用于计算视图方向）
    canonical_rays = dict_params["canonical_rays"]
    # 获取相机到世界的变换矩阵
    c2w = viewpoint_camera.c2w
    # 获取图像尺寸
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    # 构建特征张量：将深度、法线、反射属性拼接，用于延迟渲染
    # 特征维度：[1(depth), 1(depth²), 3(normal), 3(ref_tint), 1(ref_roughness), 1(ref_strength)] = 10维
    # ANCHOR 拼特征通道
    features = torch.cat([depths, depths2, normal, ref_tint, ref_roughness, ref_strength], dim=-1) # [1, 1, 3, 3, 1, 1]

    # 计算视图方向（从相机到高斯点）
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)

    # 如果使用前向着色模式（forward shading）：在光栅化前计算反射颜色
    if pipe.forward_shading and colors_precomp is not None:
        # 高斯级：高斯指向相机
        view_dirs = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D, dim=-1)
        # 使用前向着色计算反射颜色（基于反射贴图、法线、粗糙度、色调）
        refl_color_forward =  get_reflectance_color_forward(refmap, normal, view_dirs, ref_roughness, ref_tint, brdf_lut=dict_params["brdf_lut"])
        # 混合辐射颜色和反射颜色：ref_strength控制混合比例
        ref_rgb = (1.0 - ref_strength) * colors_precomp + ref_strength * refl_color_forward
        # 将混合后的颜色作为预计算颜色
        colors_precomp = ref_rgb



    
    # 如果启用了PBR模式，提取PBR相关属性
    if pc.use_pbr:
        base_color = pc.get_base_color      # 基础颜色（albedo）[N, 3]
        roughness = pc.get_roughness        # PBR粗糙度 [N, 1]
        metallic = pc.get_metallic          # PBR金属度 [N, 1]

        # for editing test
        # roughness = torch.ones_like(roughness) - 0.4
        # metallic = torch.ones_like(metallic) - 0.2
        # base_color = base_color[:, [2, 1, 0]]

        # roughness = torch.where(roughness > 0.2, 0, 0.4)

        # r_channel = torch.clamp(base_color[:, 0] + 0.7, 0.0, 1.0)
        # g_channel = torch.clamp(base_color[:, 1] + 0.7, 0.0, 1.0)
        # b_channel = torch.clamp(base_color[:, 1] + 0.7, 0.0, 1.0)
        # base_color[:, 0] = r_channel
        # base_color[:, 1] = g_channel
        # base_color[:, 2] = b_channel
        # end

        # 计算pbr 间接光（incident light，但更贴切的名字是 indirect_diffuse）
        # ANCHOR incidents_light定义处（实际用于pbr间接光照）
        if not pipe.relight:
            # 间接光 SHS (N, (d+1)^2, 3)
            incidents = pc.get_incidents  # incident shs
            # 根据法向，得到每个高斯点的间接光 RGB (N, 3)
            incidents_light = torch.clamp(eval_sh(pc.active_sh_degree, incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2), normal), 0.0, 1.0)
        else:
            # 重光照模式：使用新的光照条件
            if pipe.transfer_light:
                # 可能有bug？
                # pass
                # 传输光照模式：将环境贴图的SH系数与传输SH系数相乘
                transfer_shs = pc.get_incidents.permute(0, 2, 1)  # 传输SH系数
                light_shs = cubemap.shs                           # 环境贴图SH系数
                incidents = light_shs * transfer_shs              # 相乘得到新的入射光
                incidents = incidents.permute(0, 2, 1)
                # 评估SH系数，得到重光照后的入射光
                incidents_light = torch.clamp(eval_sh(pc.active_sh_degree, incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2), normal), 0.0, 1.0)
            else:
                # 非传输模式：入射光为零
                incidents_light = torch.zeros_like(base_color)

        # 将PBR属性追加到特征张量：[base_color(3), roughness(1), metallic(1), incidents_light(3)] = 8维
        # 总特征维度：10 + 8 = 18维
        # ANCHOR 拼特征通道(增加pbr通道)
        features = torch.cat([features, base_color, roughness, metallic, incidents_light], dim=-1) # [..., 3, 1, 1, 3]


    # 调用CUDA光栅器：将3D高斯投影到2D屏幕，生成图像和特征图
    # 返回值：渲染的高斯数量、贡献的高斯数量、颜色图像、透明度、深度、特征图、伪法线、表面XYZ坐标、权重、半径
    # CUDA光栅化器内部没有处理alpha归一化。 它只执行标准的front-to-back alpha混合
    # ANCHOR 调用CUDA光栅器
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    # 延迟着色后处理：对特征图进行alpha归一化
    # 只有有贡献的像素才进行归一化，避免除以零
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask   #[N, H, W]
    feature_size = rendered_feature.shape[0]  # 特征通道数

    # 计算期望深度：对深度图进行alpha归一化
    render_depth_expected = rendered_depth
    render_depth_expected = (render_depth_expected / rendered_opacity)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)  # 处理NaN值
    surf_depth = render_depth_expected


    # ANCHOR 拆特征通道
    # 拆分特征图：将拼接的特征张量按通道拆分为各个属性图
    # 前10个通道：depth(1), depth²(1), normal(3), ref_tint(3), ref_roughness(1), ref_strength(1)
    rendered_depth, rendered_depth2, rendered_normal, rendered_ref_tint, rendered_ref_roughness, rendered_ref_strength_map, rendered_feature_rest \
        = rendered_feature.split([1, 1, 3, 3, 1, 1, feature_size - 10], dim=0)

    cur_feature_len = 0
    # 如果启用了PBR，继续拆分剩余的PBR特征
    if pc.use_pbr:
        cur_feature_len = cur_feature_len + 10
        # ANCHOR 拆特征通道(pbr通道)
        # PBR特征：base_color(3), roughness(1), metallic(1), incident_lights(3)
        rendered_base_color, rendered_roughness, rendered_metallic, rendered_incident_lights, rendered_feature_rest_2 \
            = rendered_feature_rest.split([3, 1, 1, 3, feature_size - 18], dim=0)


    # 计算深度方差：用于评估深度不确定性
    #  - rendered_depth2 = sum(d_i² * w_i) / sum(w_i) → 先平方再加权平均
    #  - rendered_depth.square() = (sum(d_i * w_i) / sum(w_i))² → 先加权平均再平方
    # rendered_var = E[d²] - E[d]² = Var[d] 得到的就是方差
    rendered_var = rendered_depth2 - rendered_depth.square()    # [1, H, W]


    # 辐射着色：将光栅化结果转换为屏幕空间映射图
    depth_map = rendered_depth.permute(1, 2, 0)                 # [H, W, 1]
    opacity_map = rendered_opacity.permute(1, 2, 0)             # [H, W, 1]
    
    ref_roughness_map = rendered_ref_roughness.permute(1, 2, 0) # [H, W, 1]
    ref_tint_map = rendered_ref_tint.permute(1, 2, 0)           # [H, W, 3]
    ref_strength_map = rendered_ref_strength_map.permute(1, 2, 0) # [H, W, 1]
    normal_map = rendered_normal.permute(1, 2, 0)               # [H, W, 3]
    normal_map = F.normalize(normal_map, dim=-1)                # 归一化法线
    radiance_map = rendered_image.permute(1, 2, 0)              # [H, W, 3]

    # 计算每个像素的视图方向（世界坐标）
    # canonical_rays是相机坐标系中的规范光线，通过c2w矩阵变换到世界坐标
    # 像素级，场景指向相机（世界空间）
    view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]

    # 计算原始射线长度 norm（canonical_rays 是 [dx,dy,1] 未归一化，norm = ||[dx,dy,1]||）
    # 用于从 depth（视空间 Z）重建正确的欧几里得距离
    raw_ray_norm = torch.norm(canonical_rays, dim=-1).reshape(H, W)  # [H, W]

    # 延迟反射着色：根据模式选择使用前向或延迟反射计算
    # 用于非pbr的普通render结果。
    if not pipe.forward_shading:
        # 延迟模式：在屏幕空间计算反射颜色（保留高频细节）
        refl_color = get_reflectance_color(refmap, normal_map, view_dirs, ref_roughness_map, ref_tint_map, brdf_lut=dict_params["brdf_lut"])
        # 混合辐射(之前提前计算的PRT)和反射颜色
        # LINK ./render.py#prt
        ref_rgb = (1.0 - ref_strength_map) * radiance_map + ref_strength_map * refl_color
        # 应用透明度，混合背景色
        ref_rgb = ref_rgb * opacity_map + (1.0 - opacity_map) * bg_color
    else:
        # 前向模式：反射颜色已在光栅化前计算，直接使用辐射颜色
        refl_color = radiance_map
        # TODO: opacity_map 可能被乘了两次，也许这里应该去掉？
        ref_rgb = radiance_map * opacity_map + (1.0 - opacity_map) * bg_color
    



    # 构建输出特征字典：存储反射相关属性
    out_feature_dict = {}
    out_feature_dict.update({
        "ref_roughness": ref_roughness_map.permute(2, 0, 1),    # [1, H, W]
        "ref_strength": ref_strength_map.permute(2, 0, 1),      # [1, H, W]
    })


    # SECTION PBR
    # PBR着色：如果启用了PBR模式，进行基于物理的渲染
    if pc.use_pbr:
        roughness_map = rendered_roughness.permute(1, 2, 0)     # [H, W, 1]
        roughness_map = torch.clamp(roughness_map, 0.04, 1.0)   # 限制粗糙度范围（避免极端值）
        metallic_map = rendered_metallic.permute(1, 2, 0)       # [H, W, 1]
        base_color_map = rendered_base_color.permute(1, 2, 0)   # [H, W, 3]

        incident_light_map = rendered_incident_lights.permute(1, 2, 0)  # [H, W, 3]

        # 计算遮挡（occlusion）：用于自阴影和环境光遮蔽
        # 如果提供了遮挡体积数据，计算遮挡图
        if "occlusion_volumes" in dict_params.keys() and dict_params.get("enable_occlusion", True):
            # 获取AABB（轴对齐包围盒）边界
            aabb = dict_params.get("aabb")
            if aabb is not None:
                clamp_min, clamp_max = aabb[:3], aabb[3:]
            else:
                # 默认边界：以原点为中心，cbound为半径的立方体
                cbound = dict_params.get("occlusion_volumes", {}).get("bound", 1.5)
                clamp_min, clamp_max = -cbound, cbound
                aabb = torch.tensor([-cbound, -cbound, -cbound, cbound, cbound, cbound], device="cuda")
            # 计算每个像素的3D世界坐标点：通过深度和视图方向反投影
            # points = (-view_dirs * depth + camera_position)
            points = (
                (-view_dirs.reshape(-1, 3) * raw_ray_norm.reshape(-1, 1) * rendered_depth.reshape(-1, 1) + c2w[:3, 3])
                    .clamp(min=clamp_min, max=clamp_max)
                        .contiguous()
                    )  # [HW, 3]

            occlusion_volumes = dict_params["occlusion_volumes"]
            # 调用recon_occlusion计算遮挡：基于体素网格的SH系数插值
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
            
            # 调试：检查 occlusion_map 的值范围
            # print("Occlusion map min:", occlusion_map.min().item())
            # print("Occlusion map max:", occlusion_map.max().item())
            # print("Occlusion map mean:", occlusion_map.mean().item())
            
            # # 如果 occlusion_map 的值接近 1，说明没有阴影
            # if occlusion_map.mean().item() > 0.95:
            #     print("Warning: Occlusion map is almost all 1 (no shadow effect)")
        else:
            # 没有遮挡数据：遮挡图为None
            occlusion_map = None

        # 调用PBR着色函数：计算基于物理的渲染结果
        pbr_result = pbr_shading(
            light=cubemap,                                  # 环境贴图
            normals = normal_map,                           # 法线图 [H, W, 3]
            view_dirs = view_dirs,                          # 视图方向 [H, W, 3]
            albedo = base_color_map,                        # 基础颜色（反照率）[H, W, 3]
            roughness = roughness_map,                      # 粗糙度 [H, W, 1]
            metallic = metallic_map if pipe.metallic else None,    # 金属度 [H, W, 1]
            occlusion = occlusion_map if occlusion_map is not None else None,  # 遮挡图 [H, W, 1]

            # 间接光照：只参与漫反射部分，作用是补偿遮挡区域的间接光照。
            # ANCHOR pbr 漫反射间接光照
            irradiance = incident_light_map if not pipe.relight or (pipe.relight and pipe.transfer_light) else None,     # 辐照度 [H, W, 3]

            # irradiance = incident_light_map,     # [H, W, 3]
            brdf_lut=dict_params["brdf_lut"],               # BRDF查找表
        )

        # 提取PBR渲染结果
        rendered_pbr = pbr_result["render_rgb"]             # 最终PBR颜色 [H, W, 3]

        diffuse_pbr = pbr_result["diffuse_rgb"]             # 漫反射分量 [H, W, 3]
        specular_pbr = pbr_result["specular_rgb"]           # 镜面反射分量 [H, W, 3]
        occulusion_incident_light = pbr_result["incidents_light"]
        # ── Point light overlay ──────────────────────────────────────────────
        point_rgb = None
        point_lights = dict_params.get("point_lights", None) if dict_params else None
        if point_lights and len(point_lights) > 0:
            # 从 depth + view_dirs + raw_ray_norm 重建表面世界坐标
            surf_points = (
                -view_dirs.reshape(-1, 3) * raw_ray_norm.reshape(-1, 1) * rendered_depth.reshape(-1, 1) + c2w[:3, 3]
            ).reshape(H, W, 3)  # [H, W, 3]
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
            point_rgb = point_rgb * opacity_map
            rendered_pbr = rendered_pbr + point_rgb  # 遮挡后的间接漫反射光照 [H, W, 3]


        # 应用透明度，混合背景色
        rendered_pbr = rendered_pbr * opacity_map + (1.0 - opacity_map) * bg_color

        # 如果启用了色调映射，限制颜色范围
        if pipe.tone_mapping:
            rendered_pbr = torch.clamp(rendered_pbr, 0.0, 1.0)
        

        # 将PBR属性添加到输出字典
        out_feature_dict.update({
            "base_color": base_color_map.permute(2, 0, 1),  # [3, H, W]
            "roughness": roughness_map.permute(2, 0, 1),    # [1, H, W]
            "metallic": metallic_map.permute(2, 0, 1),      # [1, H, W]
        })


        # 将遮挡图添加到输出字典
        out_feature_dict.update({
            "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None else torch.zeros_like(roughness_map).permute(2, 0, 1),
        })
        if point_rgb is not None:
            out_feature_dict["point_light"] = point_rgb.permute(2, 0, 1)
    # !SECTION



    # 构建可视化字典：仅在推理模式下生成
    vis_dict = {}
    if not is_training:
        # 分解混合渲染结果
        # PRT 分量
        blended_radiance = (1.0 - ref_strength_map) * radiance_map
        # 反射分量
        blended_ref_color = ref_strength_map * refl_color
        # 归一化深度图：缩放到[0, 1]范围
        normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        vis_dict.update({
                "surf_depth": surf_depth,                   # alpha 归一化表面深度图
                "depth": normalized_depth_map.permute(2, 0, 1),        # 归一化深度图 [1, H, W]
                "normal": (normal_map.permute(2, 0, 1) * 0.5 + 0.5),  # 法线图（映射到[0,1]）[3, H, W]
                "pseudo_normal" : rendered_pseudo_normal * 0.5 + 0.5,   # 伪法线图 [3, H, W]
                "ref_roughness": ref_roughness_map.permute(2, 0, 1),    # 反射粗糙度 [1, H, W]
                "ref_strength": ref_strength_map.permute(2, 0, 1),      # 反射强度 [1, H, W]
                "radiance_color": radiance_map.permute(2, 0, 1),        # 辐射颜色 [3, H, W]
                "ref_color": refl_color.permute(2, 0, 1),               # 反射颜色 [3, H, W]
                "ref_export_base": refmap.export_envmap(return_img=True).permute(2, 0, 1),  # 反射环境贴图导出
                "ref_tint": ref_tint_map.permute(2, 0, 1),              # 反射色调 [3, H, W]
                "blended_radiance": blended_radiance.permute(2, 0, 1),  # 混合辐射 [3, H, W]
                "blended_ref_color": blended_ref_color.permute(2, 0, 1) # 混合反射 [3, H, W]
            }
        )

        # 如果启用了PBR，添加PBR相关的可视化
        if pc.use_pbr:
            vis_dict.update({
                "base_color": gamma_func(base_color_map.permute(2, 0, 1)),      # 基础颜色（sRGB）
                "base_color_rgb": base_color_map.permute(2, 0, 1),              # 基础颜色（线性）
                "roughness": roughness_map.permute(2, 0, 1),                    # 粗糙度
                "metallic": metallic_map.permute(2, 0, 1),                      # 金属度

                "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None else torch.zeros_like(rendered_image),  # 遮挡图
                "diffuse_pbr": gamma_func(diffuse_pbr.permute(2, 0, 1)),    # PBR漫反射（sRGB）
                "specular_pbr": gamma_func(specular_pbr.permute(2, 0, 1)),  # PBR镜面反射（sRGB）
                "image_pbr": gamma_func(rendered_pbr.permute(2, 0, 1)),     # PBR完整图像（sRGB）
                "incidents_light": (occulusion_incident_light.permute(2, 0, 1)),  # 遮挡后入射光
                "incident_light_raw": (incident_light_map.permute(2, 0, 1)),      # 原始入射光
            })

            vis_dict.update({
                "env_export_base": cubemap.export_envmap(return_img=True).permute(2, 0, 1),       # 环境贴图导出（完整）
                "env_export_diffuse": cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1),  # 环境贴图导出（漫反射）
            })

        # 对可视化结果应用透明度掩码：将背景区域混合为背景色
        # 排除不需要透明度掩码的键（环境贴图导出、表面深度）
        without_opacity_mask_keys = ["env_export_base", "env_export_diffuse", "ref_export_base", "surf_depth"] 
        for key in vis_dict.keys():
            if key not in without_opacity_mask_keys:
                # 应用透明度：前景 * opacity + 背景 * (1 - opacity)
                vis_dict[key] = (vis_dict[key].permute(1,2,0) * opacity_map + (1.0 - opacity_map) * bg_color).permute(2, 0, 1)
        


        
    # 构建最终结果字典
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": ref_rgb.permute(2, 0, 1),              # 最终渲染图像 [3, H, W]
               "depth": rendered_depth,                         # 深度图 [1, H, W]
               "depth_var": rendered_var,                       # 深度方差 [1, H, W]
               "normal": normal_map.permute(2, 0, 1),           # 法线图 [3, H, W]
               "pseudo_normal": rendered_pseudo_normal,         # 伪法线图 [3, H, W]
               "surface_xyz": rendered_surface_xyz,             # 表面3D坐标 [3, H, W]
               "opacity": rendered_opacity,                     # 透明度图 [1, H, W]
               "viewspace_points": screenspace_points,          # 视图空间点（用于梯度计算）
               "visibility_filter": radii > 0,                  # 可见性过滤（半径>0的高斯）
               "radii": radii,                                  # 高斯半径
               "num_rendered": num_rendered,                    # 渲染的高斯数量
               "num_contrib": num_contrib,                      # 贡献的高斯数量
               "weights": weights                               # 权重 [N, H, W]
               }
    


    # 如果启用了PBR，添加PBR渲染结果（应用gamma校正）
    if pc.use_pbr:
        results['pbr'] = gamma_func(rendered_pbr.permute(2, 0, 1))

    # 合并输出特征字典
    results.update(out_feature_dict)
    
    # 添加可视化字典
    results['vis_dict'] = vis_dict
    
    

    # 非训练模式下，生成环境贴图相关的可视化
    if not is_training:
        if pc.use_pbr:
            # 获取世界坐标系中的光线方向
            directions = viewpoint_camera.get_world_directions()
            directions = directions.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 3]
            # 如果环境贴图有旋转变换，应用旋转
            if cubemap.mtx is not None:
                directions = cubemap.rotate_dirs(directions)
            # 使用nvdiffrast采样环境贴图
            direct_env = dr.texture(
                cubemap.base[None, ...],  # [1, 6, 16, 16, 3] 环境贴图基础层
                directions.contiguous(),  # [1, H, W, 3] 采样方向
                filter_mode="linear",     # 线性插值
                boundary_mode="cube",     # 立方体贴图边界模式
            )[0]

            # 将PBR结果与环境贴图混合，生成最终可视化
            results["pbr_env"] = gamma_func((rendered_pbr * opacity_map + (1 - opacity_map)) * direct_env).permute(2, 0, 1)
            # 纯环境贴图可视化
            results["env_only"] = gamma_func(direct_env.permute(2, 0, 1))
        

    return results



def calculate_loss(viewpoint_camera, pc, results, opt, env_map):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }

    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_opacity = results["opacity"]

    rendered_ref_roughness = results["ref_roughness"]
    rendered_ref_strength = results["ref_strength"]

    loss = 0
    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = opt.lambda_rgb * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val))


    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth


    if opt.lambda_normal_smooth > 0:
        loss_normal_smooth = tv_loss(rendered_normal * image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth

    if opt.lambda_ref_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_ref_roughness_smooth = first_order_edge_aware_loss(rendered_ref_roughness * image_mask, gt_image)
        tb_dict["loss_ref_roughness_smooth"] = loss_ref_roughness_smooth.item()
        loss = loss + opt.lambda_ref_roughness_smooth * loss_ref_roughness_smooth

    if opt.lambda_ref_strength_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_ref_strength_smooth = first_order_edge_aware_loss(rendered_ref_strength * image_mask, gt_image)
        tb_dict["loss_ref_strength_smooth"] = loss_ref_strength_smooth.item()
        loss = loss + opt.lambda_ref_strength_smooth * loss_ref_strength_smooth


    if pc.use_pbr:
        rendered_pbr = results["pbr"]

        Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
        ssim_val_pbr = ssim(rendered_pbr, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_pbr * loss_pbr

        # for metallic roughness workflow
        if opt.lambda_base_color_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_base_color = results["base_color"]
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
            tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
            loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

        if opt.lambda_roughness_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_roughness = results["roughness"]
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
            tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
            loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

        if opt.lambda_metallic_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_metallic = results["metallic"]
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
            tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
            loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth


        if opt.lambda_env_smooth > 0:
            env = env_map.get_env_map()
            loss_env_smooth = tv_loss(env.permute(2, 0, 1))
            tb_dict["loss_env_smooth"] = loss_env_smooth
            loss = loss + opt.lambda_env_smooth * loss_env_smooth

        if opt.lambda_white_light > 0:
            env_base = env_map.base
            white = (env_base[..., 0:1] + env_base[..., 1:2] + env_base[..., 2:3]) / 3.0
            loss_light_white_blance = torch.mean(torch.abs(env_base - white))
            tb_dict["loss_light_white_blance"] = loss_light_white_blance.item()
            loss = loss + opt.lambda_white_light * loss_light_white_blance

        if opt.lambda_reflect_strength_equal_metallic > 0:
            loss_reflect_strength_equal_metallic = F.l1_loss(rendered_metallic, rendered_ref_strength)
            tb_dict["loss_reflect_strength_equal_metallic"] = loss_reflect_strength_equal_metallic.item()
            loss = loss + opt.lambda_reflect_strength_equal_metallic * loss_reflect_strength_equal_metallic


    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, 
                                       env_map=dict_params['cubemap'] if pc.use_pbr else None)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results