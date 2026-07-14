"""
RTR-GS 训练入口

该文件实现了两阶段逆渲染训练流程的主循环：

  Stage 1 (render_ref / render_ref_equirect):
    训练几何（3D高斯）+ 反射贴图（reflection map）
    输出：基础几何形状 + 反射属性（粗糙度、金属感、反射强度）

  Stage 2 (render_ref_pbr / render_ref_pbr_equirect):
    在Stage 1基础上增加PBR材料分解
    输入：预先烘焙的遮挡体积（occlusion volumes）
    新增组件：CubemapLight（环境光照）、PBR shading分支

关键概念：
  - Hybrid Rendering：前向PRT（低频频段）+ 延迟反射贴图（高频频段）混合
  - equirect模式：使用SGS球面高斯光栅化器处理360°全景图
  - 几何冻结：从SGS/ODGS预训练模型加载时锁定xyz/scaling/rotation/opacity
"""

import os
import json
import math
import time
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.graphics_utils import latlong_to_cubemap_equirect
from tqdm import tqdm
from utils.image_utils import psnr
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from lpipsPyTorch import get_lpips_model
from pbr import CubemapLight, get_brdf_lut
from scene.transfer_mlp import TransferMLP


start_time = 0
end_time = 0


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
    """
    主训练循环

    参数:
        dataset: 数据集参数（模型路径、分辨率、白背景等）
        opt:     优化参数（学习率、densification参数、损失权重等）
        pipe:    渲染管线参数（是否启用PRT、反射贴图、equirect模式等）
        is_pbr:  是否为PBR模式（Stage 2），决定是否初始化和训练环境光照

    训练流程：
        1. 初始化3D高斯（从checkpoint/PLY/PCD加载）
        2. 初始化PBR组件（transfer_net, cubemap light, refmap, occlusion volumes）
        3. 主循环：渲染 → 损失计算 → 反向传播 → densification → 保存
    """
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # 环境范围约束：用于真实场景，将场景外的反射强度强制降低
    USE_ENV_SCOPE = opt.use_env_scope # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4

    """
    设置 Gaussians
    ---------------
    加载或初始化3D高斯属性，支持多种加载方式（按优先级）：
      1. args.checkpoint      → 全量checkpoint（含优化器状态）← 恢复训练
      2. args.ply_checkpoint  → PLY文件（几何属性）         ← SGS/ODGS转换后导入
      3. scene.loaded_iter    → 从输出目录的PLY文件加载      ← 继续训练
      4. 无                   → 从SfM点云初始化（随机初始化） ← 新训练
      Geometry freezing: 从预训练模型加载时，锁定位置/缩放/旋转/不透明度，
      保留SGS/ODGS已有的几何质量，不进行densification
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif args.ply_checkpoint:
        print("Loading Gaussian geometry from PLY checkpoint: {}".format(args.ply_checkpoint))
        gaussians.load_ply(args.ply_checkpoint)

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    gaussians.training_setup(opt)

    # 注意：不再冻结几何（freeze_geometry 已被移除）。
    # Equirect 模式使用 SGS 风格的保守 densification（见下方训练循环），
    # 避免过度生长 floater，同时允许 RTR-GS 继续优化几何。


    """
    设置 PBR 组件
    ---------------
    根据渲染类型（is_pbr 和 pipe 配置）初始化可训练组件。

    组件概览：
      - transfer_net       : 神经辐射传输MLP（PRT模式），解码传输特征为视角相关颜色
      - canonical_rays     : 规范光线（非equirect模式），用于计算屏幕空间视图方向
      - brdf_lut           : Cook-Torrance BRDF 预积分查找表
      - occlusion_volumes  : 预烘焙的遮挡体积（从baking.py加载），用于阴影计算
      - cubemap light      : 可训练的环境光照贴图（Stage 2 PBR模式）
      - refmap             : 反射贴图（reflection map），存储场景反射信息
    """
    # 存储PBR相关参数，通过 dict_params 传递给 render 函数
    pbr_kwargs = dict()

    # ── PRT 传输网络 ──────────────────────────────────────────────
    # 如果启用PRT计算：初始化神经辐射传输 MLP（transfer_net）
    # 作用：将每个高斯的辐射传输特征解码为视角相关的颜色，详见
    # LINK ./doc/RTR-GS/compute_with_prt.md
    if pipe.compute_with_prt:
        # 激活全部SH阶数（PRT需要高阶SH以表示完整辐射传输）
        gaussians.active_sh_degree = gaussians.max_sh_degree
        # 创建 transfer_net：输入 = 视角方向 + 传输特征 → 输出 = 颜色
        transfer_net = TransferMLP(sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres)
        # 从checkpoint加载预训练权重（Stage 1 → Stage 2 间传递）
        if args.checkpoint:
            transfer_net_checkpoint = os.path.dirname(args.checkpoint) + "/transfer_net_" + os.path.basename(args.checkpoint)
            if os.path.exists(transfer_net_checkpoint):
                transfer_net.create_from_ckpt(transfer_net_checkpoint)
                print("Successfully loaded transfer net!")
            else:
                print("Failed to load transfer net!")

        transfer_net.training_setup(opt)
        pbr_kwargs["transfer_net"] = transfer_net

    # ── 反射与PBR 公共组件 ─────────────────────────────────────────
    # ref_map：用于混合渲染分支，记录光照信息
    # LINK doc/RTR-GS/ref_map介绍.md
    if is_pbr or pipe.ref_map:
        # 非全景模式：获取规范光线，用于在屏幕空间计算每个像素的视图方向
        if not pipe.equirect:
            canonical_rays = scene.get_canonical_rays()
            pbr_kwargs["canonical_rays"] = canonical_rays
        # 预计算BRDF LUT：用于Cook-Torrance BRDF的快速拆分和近似积分
        brdf_lut = get_brdf_lut().cuda()
        pbr_kwargs["brdf_lut"] = brdf_lut

    # ── PBR 环境光照 + 遮挡 ───────────────────────────────────────
    if is_pbr:
        # 加载预烘焙的遮挡体积（由baking.py生成）
        # 包含：SH系数的3D体素网格，用于实时计算阴影（近似环境光遮挡）
        if args.occlusion_path is not None:
            occlusion_volumes = torch.load(args.occlusion_path)
            if "aabb" in occlusion_volumes:
                aabb = occlusion_volumes["aabb"].clone().cuda()
            else:
                bound = occlusion_volumes["bound"]
                aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
            pbr_kwargs["occlusion_volumes"] = occlusion_volumes
            pbr_kwargs["aabb"] = aabb

        # 可训练的环境光照贴图（CubemapLight）
        cubemap = CubemapLight(base_res=128).cuda()
        cubemap.train()
        if args.checkpoint:
            cubemap_checkpoint = os.path.dirname(args.checkpoint) + "/cubemap_" + os.path.basename(args.checkpoint)
            if os.path.exists(cubemap_checkpoint):
                cubemap.create_from_ckpt(cubemap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                print("Failed to load!")
        cubemap.build_mips()
        cubemap.training_setup(opt)
        pbr_kwargs["cubemap"] = cubemap

    # ── 反射贴图（Reflection Map） ────────────────────────────────
    # 作用：存储环境反射信息，用于高光反射的渲染（split-sum近似）
    # 与cubemap light不同：refmap专门负责反射频段，在第Stage 1就和几何一起训练
    if pipe.ref_map:
        refmap = CubemapLight(base_res=128).cuda()
        refmap.train()

        if args.checkpoint:
            refmap_checkpoint = os.path.dirname(args.checkpoint) + "/refmap_" + os.path.basename(args.checkpoint)
            if os.path.exists(refmap_checkpoint):
                refmap.create_from_ckpt(refmap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                print("Failed to load!")

        refmap.build_mips()
        refmap.training_setup(opt, light_type="ref")
        pbr_kwargs["refmap"] = refmap

    """
    准备渲染函数和背景色
    --------------------
    render_fn 由 args.type 从 render_fn_dict 选择：
      - render_ref / render_ref_pbr         → render.py（透视模式）
      - render_ref_equirect / render_ref_pbr_equirect → render_equirect.py（全景模式）
      - render_ref_fast                      → render_fast.py（透视轻量模式）
    """
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    start_time = time.time()

    """ 保存初始状态可视化（第0次迭代） """
    with torch.no_grad():
        save_vis_images(scene, render_fn, pipe, background,
                        iteration=0, dict_params=pbr_kwargs, is_pbr=is_pbr)

    """
    开始训练循环
    ============
    主循环迭代流程（每个iteration）:
      1. SH阶数升级（每1000次迭代增加一阶）
      2. 随机选取一个训练视角
      3. 调用 render_fn 执行渲染（前向PRT + 延迟反射/PBR）
      4. 计算损失（L1 + SSIM + 辅助损失）
      5. 反向传播
      6. Densification（克隆/分裂/修剪高斯）
      7. 优化器步进（更新所有可训练参数）
      8. 周期性保存checkpoint和可视化结果
    """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        # ── SH 阶数递进 ──────────────────────────────────────────
        # 每 1000 次迭代增加一阶SH（从0阶逐步到max_sh_degree）
        # 策略：低阶先收敛大致形状，再逐步增加高频细节
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ── 随机选择训练视角 ──────────────────────────────────────
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # 从指定迭代开始启用debug输出
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        # ── 渲染 ──────────────────────────────────────────────────
        # 调用对应的render函数，输出：
        #   - render          : 最终渲染图像（hybrid: PRT + reflection）
        #   - depth / normal  : 几何辅助信息
        #   - pbr             : PBR渲染结果（Stage 2）
        #   - loss            : 各项损失之和
        #   - tb_dict         : TensorBoard 日志字典
        #   - vis_dict        : 可视化中间结果
        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                               opt=opt, is_training=True, dict_params=pbr_kwargs)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # ── 损失计算 ──────────────────────────────────────────────
        # render_pkg["loss"] 由render函数内部已经聚合了所有损失项：
        #   - L1 + SSIM（图像重建损失）
        #   - 法线平滑损失（TV smoothness）
        #   - PBR正则化损失（如果启用）
        #   - 环境贴图正则化
        tb_dict = render_pkg["tb_dict"] # TensorBoard 日志字典
        loss += render_pkg["loss"]

        # 环境范围外反射强度抑制损失（USE_ENV_SCOPE）
        # 用于真实场景：强制场景边界外的高斯反射强度趋近于0，
        # 防止浮游高斯的虚假反射
        def get_outside_msk():
            return None if not USE_ENV_SCOPE else \
                torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_ref_strength
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += REFL_MSK_LOSS_W * refl_msk_loss

        # ── 反向传播 ──────────────────────────────────────────────
        loss.backward()

        with torch.no_grad():

            # ── 进度条更新（PSNR EMA平滑） ────────────────────────
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            progress_bar.set_postfix(pbar_dict)

            # ── TensorBoard 日志 + 周期性验证 ────────────────────
            # 每 test_interval 次迭代在测试集上评估L1/PSNR
            training_report(tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,
                            bg_color=background, dict_params=pbr_kwargs)

            # ── 可视化图像保存（每 vis_interval 次迭代） ─────────
            if iteration % args.vis_interval == 0:
                save_vis_images(scene, render_fn, pipe, background,
                                iteration, pbr_kwargs, is_pbr)

            
            # ── 保存场景（PLY格式） ───────────────────────────────
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ── 保存checkpoint（完整训练状态，含优化器） ─────────
            # 支持从中断点恢复训练：
            #   - 高斯模型的状态 + 优化器状态
            #   - 各PBR组件的状态 + 优化器状态
            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                os.makedirs(os.path.join(scene.model_path, "checkpoint"),exist_ok=True)
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "checkpoint/chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"checkpoint/{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

            # ── 可选的几何冻结（freeze_geo_from_iter） ─────────────
            # 当达到指定迭代次数后，冻结位置优化并停止增删高斯点，
            # 只允许微调 scale/rotation 来优化法线（最短轴）。
            if opt.freeze_geo_from_iter > 0 and iteration == opt.freeze_geo_from_iter:
                print(f"\n[ITER {iteration}] Freezing geometry: "
                      f"xyz LR → 0, densification → disabled")
                for group in gaussians.optimizer.param_groups:
                    if group["name"] == "xyz":
                        group['lr'] = 0
                # 终止 densification 和 post-densification 修剪
                opt.densify_until_iter = 0

            # ── Densification（高斯密化/修剪） ────────────────────
            if iteration < opt.densify_until_iter:
                if pipe.equirect:
                    # Equirect 模式：使用 SGS 风格的保守 densification
                    # (lat-aware 阈值、capped 修剪、initial point 保护)
                    gaussians.add_densification_stats(
                        viewspace_point_tensor, visibility_filter,
                        weights=None, lat=render_pkg.get('lat'))
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter])

                    if (iteration > opt.densify_from_iter
                            and iteration % opt.densification_interval == 0):
                        gaussians.equirect_densify_and_prune(
                            opt, scene.cameras_extent,
                            lat=render_pkg.get('lat'), iteration=iteration)
                else:
                    # 透视模式：标准 3DGS densification
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter,
                                                        render_pkg['weights'])
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter])

                    if (iteration > opt.densify_from_iter
                            and iteration % opt.densification_interval == 0):
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            opt.densify_grad_threshold, 0.005,
                            scene.cameras_extent, size_threshold)

                # ── 不透明度和反射属性重置 ────────────────────────
                HAS_RESET0 = False
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity()
                    if not opt.without_normal_propagation:
                        gaussians.reset_refl(exclusive_msk=outside_msk)

                # 法线传播（normal propagation）
                if (opt.init_iter < iteration <= opt.propagation_until_iter) and iteration % 1000 == 0 and pipe.ref_map:
                    if not HAS_RESET0 and not opt.without_normal_propagation:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        gaussians.reset_scale(exclusive_msk=outside_msk)


            # ── Equirect 模式的独立修剪（densification结束后） ────
            # 透视模式的 densification 结束后，全景模式仍需继续修剪
            # 低不透明度高斯，防止 floater 积累。
            # 使用 SGS 风格的保守修剪（capped、保护初始点）。
            if pipe.equirect and iteration >= opt.densify_until_iter \
                    and not (opt.freeze_geo_from_iter > 0 and iteration >= opt.freeze_geo_from_iter):
                if iteration > 500 and iteration % opt.densification_interval == 0:
                    gaussians.equirect_prune(
                        opt, scene.cameras_extent, iteration=iteration)

                if iteration % opt.opacity_reset_interval == 0:
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity()
                    if not opt.without_normal_propagation:
                        gaussians.reset_refl(exclusive_msk=outside_msk)

                if (opt.init_iter < iteration <= opt.propagation_until_iter) and iteration % 1000 == 0 and pipe.ref_map:
                    if not opt.without_normal_propagation:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        gaussians.reset_scale(exclusive_msk=outside_msk)

            # ── 优化器步进 ────────────────────────────────────────
            # 更新所有可训练参数：
            #   - gaussians：xyz, SH系数, 缩放, 旋转, 不透明度, PBR属性等
            #   - pbr_kwargs组件：transfer_net, cubemap light, refmap
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass

    # ── 训练完成后的处理 ─────────────────────────────────────────
    if is_pbr:
        # 将环境光照转换为传输特征存储到高斯中
        # 这样在推理阶段可以加载高斯模型独立渲染，不再需要cubemap light
        cubemap.build_sh(3)
        gaussians.incident_to_transfer(cubemap.shs)

    # 记录总训练时间
    end_time = time.time()
    with open(os.path.join(args.model_path, "trainint_time.txt"), "w") as f:
        f.write(f"training time(seconds): {end_time - start_time}\n")
        minutes = (end_time - start_time) / 60.0
        f.write(f"training time(minutes): {minutes}\n")

    # ── 最终评估 ──────────────────────────────────────────────────
    if dataset.eval and not args.skip_eval:
        eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)

    


def save_vis_images(scene, render_fn, pipe, background, iteration, dict_params, is_pbr, num_views=4):
    """
    轻量级可视化保存函数

    对测试集的前 num_views 个视角，渲染并保存中间结果图像。
    不计算指标——仅用于训练过程中的快速视觉检查。

    保存内容：
      核心：render（最终渲染）、gt（真实值）、depth（深度）、opacity（不透明度）
      几何：normal（CUDA光栅化法线）、pseudo_normal（深度推导法线）
      PBR（Stage 2）：pbr（PBR渲染）、base_color、roughness、metallic 等
      反射：radiance_color（PRT辐射）、ref_strength/roughness/tint（反射属性）
      Equirect模式额外：将全景图转换为6张cubemap面
    """
    test_cameras = scene.getTestCameras()
    if not test_cameras or len(test_cameras) == 0:
        return

    n_views = min(num_views, len(test_cameras))
    vis_dir = os.path.join(scene.model_path, "vis", f"iteration_{iteration:05d}")

    for idx in range(n_views):
        viewpoint = test_cameras[idx]
        render_kwargs = dict(dict_params)
        render_kwargs["iteration"] = iteration
        # LINK ./gaussian_renderer/render_equirect.py:231
        # LINK ./gaussian_renderer/render.py:18
        view_dir = os.path.join(vis_dir, f"view_{viewpoint.image_name}")
        os.makedirs(view_dir, exist_ok=True)

        def save_pkg(render_pkg, save_dir):
            # 核心渲染结果
            render_img = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_img = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)

            # 深度图（归一化到[0, 1]以便保存为图像）
            depth_raw = render_pkg["depth"]
            depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min() + 1e-8)

            # 法线（从[-1, 1]映射到[0, 1]以便可视化）
            opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
            normal = torch.clamp(render_pkg["normal"] * 0.5 + 0.5, 0.0, 1.0)
            pseudo_normal = torch.clamp(render_pkg["pseudo_normal"] * 0.5 + 0.5, 0.0, 1.0)

            save_image(render_img, os.path.join(save_dir, "render.png"))
            save_image(gt_img, os.path.join(save_dir, "gt.png"))
            save_image(depth_norm, os.path.join(save_dir, "depth.png"))
            save_image(opacity, os.path.join(save_dir, "opacity.png"))
            save_image(normal, os.path.join(save_dir, "normal.png"))
            save_image(pseudo_normal, os.path.join(save_dir, "pseudo_normal.png"))

            # PBR 主结果
            if is_pbr and "pbr" in render_pkg:
                pbr_img = torch.clamp(render_pkg["pbr"], 0.0, 1.0)
                save_image(pbr_img, os.path.join(save_dir, "pbr.png"))

            # 从 vis_dict 保存更多可视化中间结果
            vis_dict = render_pkg.get("vis_dict", {})

            # 通用可视化key（所有模式都需要）
            core_vis_keys = [
                "radiance_color",
                "ref_strength", "ref_roughness", "ref_tint",
                "ref_export_base",
                "normal_facing", "normal_prior",
            ]
            # PBR专用可视化key
            pbr_vis_keys = [
                "base_color", "roughness", "metallic",
                "diffuse_pbr", "specular_pbr", "image_pbr",
                "visibility",
                "incidents_light", "incident_light_raw",
                "env_export_base", "env_export_diffuse",
            ]

            keys_to_save = core_vis_keys + (pbr_vis_keys if is_pbr else [])
            for key in keys_to_save:
                if key in vis_dict:
                    save_image(
                        torch.clamp(vis_dict[key], 0.0, 1.0),
                        os.path.join(save_dir, f"{key}.png"))



        render_pkg = render_fn(viewpoint, scene.gaussians, pipe, background,
                               is_training=False, dict_params=render_kwargs)
        save_pkg(render_pkg, view_dir)
        if getattr(pipe, 'equirect', False):
            # 额外渲染perspective：复制equirect相机的位姿(R,T)，改用针孔FOV与分辨率
            from scene.cameras import Camera
            fov_x = math.radians(120.0)   # ≈ 1.0472
            fov_y = 2 * math.atan(math.tan(fov_x * 0.5) * 500 / 1000)  # 保持方形像素 fx=fy
            persp_viewpoint = Camera(
                colmap_id=viewpoint.colmap_id, R=viewpoint.R, T=viewpoint.T,
                FoVx=fov_x, FoVy=fov_y,
                fx=None, fy=None, cx=None, cy=None,
                image_name=viewpoint.image_name, uid=viewpoint.uid,
                trans=viewpoint.trans, scale=viewpoint.scale,
                height=500, width=1000, render_only=True,
            )
            render_kwargs["canonical_rays"] = persp_viewpoint.get_canonical_rays()
            render_pkg2 = render_fn_dict["render_ref"](persp_viewpoint, scene.gaussians, pipe, background,
                                is_training=False, dict_params=render_kwargs)
            view_dir2 = os.path.join(view_dir, "perspective")
            os.makedirs(view_dir2, exist_ok=True)
            save_pkg(render_pkg2, view_dir2)
            # # Equirect → cubemap 六面转换
            # # 方便在标准3D查看器中检查全景渲染质量
            # cubemap_dir = os.path.join(view_dir, "cubemap")
            # os.makedirs(cubemap_dir, exist_ok=True)
            # face_names = ["posx", "negx", "posy", "negy", "posz", "negz"]
            # cubemap = latlong_to_cubemap_equirect(
            # render_img.permute(1, 2, 0), [512, 512])
            # for face_idx in range(6):
            #     face_img = cubemap[face_idx].permute(2, 0, 1)
            #     save_image(face_img, os.path.join(
            #         cubemap_dir, f"{face_names[face_idx]}.png"))

    torch.cuda.empty_cache()


def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    """
    日志记录和周期性验证评估

    职责：
      1. 将每次迭代的训练损失写入 TensorBoard（自动执行）
      2. 每 test_interval 次迭代在测试集和训练集上评估L1/PSNR
      3. 将评估结果写入 TensorBoard 和文本文件

    评估时保存的图像（前10个视角）：
      - image / gt_image: 渲染结果与原始图像
      - opacity / depth: 不透明度和深度
      - vis_dict中的各中间结果（法线、BRDF分量等）
    """
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # 周期性评估：在测试集和训练集上计算指标
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if scene.gaussians.use_pbr:
                    l1_pbr_test = 0.0
                    psnr_pbr_test = 0.0

                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    write_image_dict = {}

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    # 0~1 归一化
                    depth = (depth - depth.min()) / (depth.max() - depth.min())

                    write_image_dict.update({
                        "image": image, "gt_image": gt_image,
                        "opacity": opacity, "depth": depth,
                    })

                    vis_dict = render_pkg["vis_dict"]
                    write_image_dict.update(vis_dict)

                    # TensorBoard 保存前10个视角的完整可视化
                    if tb_writer and (idx < 10):
                        for key in write_image_dict:
                            tb_writer.add_images(config['name'] + "_view_{}_{}/{}".format(viewpoint.image_name, idx, key),
                                                torch.clamp(write_image_dict[key][None], 0.0, 1.0), global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    if scene.gaussians.use_pbr:
                        l1_pbr_test += F.l1_loss(render_pkg["pbr"], gt_image).mean().double()
                        psnr_pbr_test += psnr(render_pkg["pbr"], gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test))

                if scene.gaussians.use_pbr:
                    psnr_pbr_test /= len(config['cameras'])
                    l1_pbr_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR_PBR {}".format(iteration, config['name'], l1_pbr_test,
                                                                                psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                    if scene.gaussians.use_pbr:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_pbr', l1_pbr_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                # 最后一次迭代：将最终指标写入文件
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {}".format(l1_test, psnr_test))
                    if scene.gaussians.use_pbr:
                        with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                            f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))
        if tb_writer:
            # 记录不透明度直方图和高斯总数（用于监控densification状态）
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()




def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    """
    最终评估函数：在测试集上计算 PSNR / SSIM / LPIPS

    在训练完成后（或是脚本单独调用时），对全部测试视角进行渲染，
    计算标准的图像质量指标。

    额外导出：
      - eval/render/  : 每视角的渲染结果
      - eval/gt/      : 对应的真实图像
      - eval/normal/  : 法线图
      - eval/*/       : vis_dict 中各中间结果
      - envmap.png    : PBR模式下的环境光照贴图（仅首次）
      - incidents_light.png : 入射光照图（仅首次）
      - eval.txt      : 汇总指标
    """
    LPIPS = get_lpips_model(net_type='vgg').cuda()

    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()

    mkdir_flag = False

    if gaussians.use_pbr:
        psnr_pbr_test = 0.0
        ssim_pbr_test = 0.0
        lpips_pbr_test = 0.0

        os.makedirs(os.path.join(args.model_path, 'eval'), exist_ok=True)
        env_cubemap = pbr_kwargs['cubemap']
        envmap = env_cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, 'eval', 'envmap.png')
        torchvision.utils.save_image(envmap, envmap_path)

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)

            image = results["render"]
            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += LPIPS(image, gt_image).mean().double()

            if gaussians.use_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)

                psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()
                ssim_pbr_test += ssim(image_pbr, gt_image).mean().double()
                lpips_pbr_test += LPIPS(image_pbr, gt_image).mean().double()

            # 保存渲染结果和中间可视化
            write_image_dict = {}
            write_image_dict.update({
                "render": image,
                "gt": gt_image,
            })

            vis_dict = results["vis_dict"]
            write_image_dict.update(vis_dict)
            # 这些key是辅助渲染用的中间结果，不保存为单独文件
            ban_image_keys = ["ref_export_base", "ref_tint",\
                              "radiance_color", "ref_roughness", "ref_strength", "ref_color"]

            # 首次迭代保存全局环境贴图/入射光
            if not mkdir_flag:
                for key in ["env_export_base", "env_export_diffuse", "incidents_light", "incident_light_raw"]:
                    if key in write_image_dict:
                        save_image(torch.clamp(write_image_dict[key], 0.0, 1.0),
                                   os.path.join(args.model_path, f"{key}.png"))

            # 首次迭代创建输出目录结构
            if not mkdir_flag:
                mkdir_flag = True
                os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
                os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
                os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
                for key in vis_dict:
                    if key in write_image_dict.keys() and key not in ban_image_keys:
                        os.makedirs(os.path.join(args.model_path, 'eval', key), exist_ok=True)

            for key in write_image_dict:
                if key not in ban_image_keys:
                    save_image(torch.clamp(write_image_dict[key], 0.0, 1.0),
                            os.path.join(args.model_path, 'eval', key, f"{viewpoint.image_name}_{idx}.png"))

    # 汇总指标
    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)

    if gaussians.use_pbr:
        psnr_pbr_test /= len(test_cameras)
        ssim_pbr_test  /= len(test_cameras)
        lpips_pbr_test /= len(test_cameras)

    # 写入评估结果文件
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")

        if gaussians.use_pbr:
            f.write(f"psnr_pbr: {psnr_pbr_test}\n")
            f.write(f"ssim_pbr: {ssim_pbr_test}\n")
            f.write(f"lpips_pbr: {lpips_pbr_test}\n")

    if gaussians.use_pbr:
        print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {} PSNR_pbr {} SSIM_pbr {} LPIPS_pbr {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test,  psnr_pbr_test, ssim_pbr_test, lpips_pbr_test))
    else:
        print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))


if __name__ == "__main__":
    """
    命令行入口
    ==========
    支持的五种渲染类型（-t / --type）：
      - render_ref              Stage 1: 几何 + 反射贴图（透视模式）
      - render_ref_pbr          Stage 2: 几何 + 反射 + PBR材料分解（透视模式）
      - render_ref_fast         Stage 2变体：轻量级PBR（透视模式）
      - render_ref_equirect     Stage 1: 几何 + 反射贴图（全景360°模式）
      - render_ref_pbr_equirect Stage 2: 几何 + 反射 + PBR分解（全景360°模式）

    常用参数示例：
      python train.py -t render_ref -m output/scene_name           # Stage 1（透视）
      python train.py -t render_ref_pbr -m output/scene_name \\   # Stage 2（透视）
        --occlusion_path output/scene_name/occlusion_volumes.pth
      python train.py -t render_ref_equirect -m output/scene \\   # Stage 1（全景）
        --ply_checkpoint output/sgs_model.ply
      python train.py -c checkpoint/chkpnt30000.pth               # 从checkpoint恢复
    """
    # ── 参数解析 ──────────────────────────────────────────────────
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render_ref', 'render_ref_pbr', 'render_ref_fast',
                                                   'render_ref_equirect', 'render_ref_pbr_equirect'], default='render_ref')
    parser.add_argument("--test_interval", type=int, default=4000)
    parser.add_argument("--vis_interval", type=int, default=2000,
                        help="Interval for saving visualization images (first 4 test views) to disk during training")
    parser.add_argument("--save_interval", type=int, default=30000)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=30000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--ply_checkpoint", type=str, default=None,
                        help="Path to a .ply file with pre-trained Gaussian geometry (e.g. converted from SGS/ODGS)")
    parser.add_argument("--occlusion_path", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # ── Equirect 模式自动配置 ───────────────────────────────────
    # 全景模式下强制使用 forward_shading（因为SGS光栅化器不支持deferred shading）
    if args.type in ['render_ref_equirect', 'render_ref_pbr_equirect']:
        args.equirect = True
        args.forward_shading = True
        print("Equirectangular mode enabled: forward_shading=True")

    # 将命令行参数保存到输出目录，方便复现
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Saved args to {os.path.join(args.model_path, 'args.json')}")

    # 初始化系统状态（随机种子等）
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # 判断是否为PBR模式（Stage 2）
    is_pbr = args.type in ['render_ref_pbr', 'render_ref_fast', 'render_ref_pbr_equirect']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)

    # 全部完成
    print("\nTraining complete.")
