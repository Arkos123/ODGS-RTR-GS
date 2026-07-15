import copy
import os
from typing import List
import numpy as np
import torch
from scene.cameras import Camera
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from pbr import CubemapLight, PointLight, get_brdf_lut
from pbr.point_light_shadow import get_depth_cubemap, get_depth_equirect, make_shadow_func_cubemap, make_shadow_func_equirect
from scene.transfer_mlp import TransferMLP
import imageio
from utils.graphics_utils import  read_hdr, latlong_to_cubemap


def load_point_lights(config_path: str):
    """从 JSON 加载点光源列表。
    每个光源支持可选字段：
      radius: float — 视频逐帧水平旋转半径（默认 0，不旋转）
      rotate_total: float — 视频总旋转角度（度，默认 360）
      shadow_bias: float — 阴影深度偏移（默认 0.3），越大越不容易产生自阴影黑斑
    COLMAP 空间：+Y 向下，旋转在 XZ 水平面进行。
    """
    import json
    with open(config_path) as f:
        data = json.load(f)
    lights = []
    orbits = []
    biases = []
    for item in data["lights"]:
        pos = torch.tensor(item["position"], dtype=torch.float32, device="cuda")
        col = torch.tensor(item["color"], dtype=torch.float32, device="cuda")
        lights.append(PointLight(position=pos, color=col, intensity=item.get("intensity", 1.0)))
        radius = item.get("radius", 0.0)
        rotate_total = item.get("rotate_total", 360.0)
        orbits.append((radius, rotate_total))
        biases.append(item.get("shadow_bias", 0.3))
    return lights, orbits, biases


def _draw_light_indicator(
    image: torch.Tensor,  # [3, H, W] RGB 图像，值域 [0,1]
    lights: List[PointLight],
    camera,
    is_equirect: bool = False,
    color: tuple = (1.0, 0.2, 0.1),
    radius: int = 20,
) -> torch.Tensor:
    """在渲染图像上叠加点光源位置指示光球，方便确认光源位置。"""
    import math
    H, W = image.shape[1:]
    img = image.clone()
    for light in lights:
        pos = light.position
        if is_equirect:
            # Equirect 投影：方向 → (lon, lat)
            # 相机在原点时，方向向量 = normalize(light_pos - camera_pos)
            cam_center = camera.camera_center
            d = pos - cam_center
            d = d / d.norm()
            lat = torch.asin((-d[1]).clamp(-1.0, 1.0))
            lon = torch.atan2(d[0], d[2])
            px = (lon / math.pi + 1.0) * 0.5 * W
            py = (0.5 - lat / math.pi) * H
        else:
            # Perspective 投影：使用 view-projection 矩阵
            import torch.nn.functional as F
            p_h = torch.cat([pos, torch.tensor([1.0], device=pos.device)])
            p_clip = camera.full_proj_transform @ p_h
            p_ndc = p_clip[:2] / p_clip[3]
            px = (p_ndc[0] + 1.0) * 0.5 * W
            py = (p_ndc[1] + 1.0) * 0.5 * H
        px = int(px.item())
        py = int(py.item())
        # 在图像边界内才绘制
        if 0 <= px < W and 0 <= py < H:
            # 绘制外圈光晕（高斯衰减）
            import numpy as np
            ys = torch.arange(max(0, py - radius * 2), min(H, py + radius * 2 + 1), device=image.device)
            xs = torch.arange(max(0, px - radius * 2), min(W, px + radius * 2 + 1), device=image.device)
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            dist = ((gx - px) ** 2 + (gy - py) ** 2).float()
            glow = torch.exp(-dist / (radius * radius))
            for c in range(3):
                img[c, gy, gx] = img[c, gy, gx] * (1 - glow * 0.5) + color[c] * glow * 0.5
            # 绘制内核（实心圆）
            inner = dist < radius * radius * 0.25
            for c in range(3):
                img[c, gy, gx] = torch.where(inner, color[c], img[c, gy, gx])
    return img


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False, is_equirect=False):

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians, shuffle=False)
    if args.ply_path:
        print("Loading Gaussians from PLY {}".format(args.ply_path))
        gaussians.load_ply(args.ply_path)
        # 从路径中提取迭代号（如 iteration_30000/point_cloud.ply），没有则默认为 0
        first_iter = 0
        for part in args.ply_path.replace('\\', '/').split('/'):
            if part.startswith('iteration_'):
                try:
                    first_iter = int(part.split('_')[1])
                except ValueError:
                    pass
                break
    elif args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        NotImplementedError("No checkpoint or loaded iteration found")


    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    pbr_kwargs["iteration"] = first_iter

    if pipe.compute_with_prt:
        transfer_net = TransferMLP(sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres)
        if args.checkpoint:
            transfer_net_checkpoint = os.path.dirname(args.checkpoint) + "/transfer_net_" + os.path.basename(args.checkpoint)
            if os.path.exists(transfer_net_checkpoint):
                transfer_net.create_from_ckpt(transfer_net_checkpoint)
                print("Successfully loaded transfer net!")
            else:
                NotImplementedError("No checkpoint or loaded iteration found")

        pbr_kwargs["transfer_net"] = transfer_net


    # equirect mode doesn't need canonical_rays (SGS rasterizer computes its own)
    if is_pbr or pipe.ref_map:
        if not is_equirect:
            canonical_rays = scene.get_canonical_rays()
            pbr_kwargs["canonical_rays"] = canonical_rays

        brdf_lut = get_brdf_lut().cuda()
        pbr_kwargs["brdf_lut"] = brdf_lut

    if is_pbr:
        if args.occlusion_path is not None:
            occlusion_volumes = torch.load(args.occlusion_path)
            if "aabb" in occlusion_volumes:
                aabb = occlusion_volumes["aabb"].clone().cuda()
            else:
                bound = occlusion_volumes["bound"]
                aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
            pbr_kwargs["occlusion_volumes"] = occlusion_volumes
            pbr_kwargs["aabb"] = aabb

        cubemap = CubemapLight(base_res=128).cuda()
        cubemap.train()
        if args.checkpoint:
            cubemap_checkpoint = os.path.dirname(args.checkpoint) + "/cubemap_" + os.path.basename(args.checkpoint)
            if os.path.exists(cubemap_checkpoint):
                cubemap.create_from_ckpt(cubemap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                NotImplementedError("No checkpoint or loaded iteration found")
        pbr_kwargs["cubemap"] = cubemap
        
    if pipe.ref_map:
        refmap = CubemapLight(base_res=128).cuda()
        refmap.train()

        if args.checkpoint:
            refmap_checkpoint = os.path.dirname(args.checkpoint) + "/refmap_" + os.path.basename(args.checkpoint)
            if os.path.exists(refmap_checkpoint):
                refmap.create_from_ckpt(refmap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                NotImplementedError("No checkpoint or loaded iteration found")

        refmap.build_mips()
        refmap.training_setup(opt, light_type="ref")
        pbr_kwargs["refmap"] = refmap

    # Point lights
    if args.point_lights_config:
        point_lights, orbits, biases = load_point_lights(args.point_lights_config)
        pbr_kwargs["point_lights"] = point_lights
        has_orbit = any(r > 0 for r, _ in orbits)
        if has_orbit:
            pbr_kwargs["_point_light_orbits"] = orbits
            pbr_kwargs["_point_light_pivots"] = [light.position.clone() for light in point_lights]
            # 静态渲染用初始位置，深烘焙用初始位置
            print(f"[point light] Loaded {len(point_lights)} point light(s) with video orbit")
        else:
            print(f"[point light] Loaded {len(point_lights)} point light(s)")

        # Pre-bake shadow maps for fixed lights
        is_equirect_mode = getattr(args, 'equirect_width', None) is not None
        shadow_funcs = []
        for i, light in enumerate(point_lights):
            bias = biases[i] if i < len(biases) else 0.3
            if is_equirect_mode:
                depth_map = get_depth_equirect(gaussians, light.position)
                shadow_fn = make_shadow_func_equirect(depth_map, light.position, threshold=bias)
            else:
                depth_map = get_depth_cubemap(gaussians, light.position)
                shadow_fn = make_shadow_func_cubemap(depth_map, light.position, threshold=bias)
            shadow_funcs.append(shadow_fn)
            print(f"  [shadow] Light {i}: depth map ready (bias={bias})")
        pbr_kwargs["point_light_shadow_funcs"] = shadow_funcs
        pbr_kwargs["_point_light_biases"] = biases

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 检查属性是否存在
    # if   not pipe.specular_workflow:
    capture_list = ["pbr", "base_color", "diffuse_pbr", "specular_pbr"]
    # else:
        # capture_list = ["pbr", "diffuse_color", "specular_color", "diffuse_pbr", "specular_pbr"]

    envmap_base_dir = args.envmap_path
    task_dict = {
        "directional_front_top": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "directional_front_top.hdr",
        },
        "directional_front": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "directional_front.hdr",
        },
        "directional_left": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "directional_left.hdr",
        },
        "directional_right": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "directional_right.hdr",
        },
        "directional_top": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "directional_top.hdr",
        },
        "TCom_ColorfulAlley": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "TCom_ColorfulAlley_colorful_alley_2K_hdri_sphere.exr",
        },
        "studio": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "big-studio-01_4K.exr",
        },
        # "rock-theatre": {
        #     "capture_list": capture_list,
        #     "envmap_path": envmap_base_dir + "rock-theatre-viewpoint_4K.exr",
        # },
        "sunset": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "sunset.hdr",
        },
        "bridge": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "bridge.hdr",
        },
        "city": {
            "capture_list": capture_list,
            "envmap_path":  envmap_base_dir + "city.hdr",
        },
        "fireplace": {
            "capture_list": capture_list,
            "envmap_path":  envmap_base_dir + "fireplace.hdr",
        },
        "forest": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "forest.hdr",
        },
        "night": {
            "capture_list": capture_list,
            "envmap_path":  envmap_base_dir + "night.hdr",
        }
    }


    task_names = ["directional_front_top"]
    # task_names = ['studio','directional_front_top','directional_front','directional_left','directional_right','directional_top','TCom_ColorfulAlley','bridge', 'city', 'fireplace', 'forest', 'night']
    # task_names = ['studio']
    for task_name in task_names:
        cubemap = None
        hdri = read_hdr(task_dict[task_name]["envmap_path"])
        hdri = torch.from_numpy(hdri).cuda()
        res = 256
        cubemap = CubemapLight(base_res=res).cuda()
        cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
        cubemap.build_mips()
        cubemap.eval()

        pbr_kwargs["cubemap"] = cubemap

        # 导出当前环境贴图为 PNG（方便查看 relighting 用了什么光照）
        envmap_dir = os.path.join(args.model_path, 'test_rli', task_name)
        os.makedirs(envmap_dir, exist_ok=True)
        envmap = cubemap.export_envmap(return_img=True)  # [H, W, 3], HDR
        # Reinhard tone mapping: HDR → [0,1] 使亮部可见
        envmap_tm = (envmap / (envmap + 1.0)).permute(2, 0, 1).clamp(0.0, 1.0)
        save_image(envmap_tm, os.path.join(envmap_dir, "envmap.png"))

        eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, task_name,
                    equirect_width=getattr(args, 'equirect_width', None))

        if args.save_video:
            full_video = getattr(args, 'full_video_output', False)
            if is_equirect:
                eval_render_video_equirect(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, task_name,
                                            equirect_width=getattr(args, 'equirect_width', None),
                                            full_video_output=full_video)
            else:
                eval_render_video(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, task_name,
                                  full_video_output=full_video)
    



def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, env_name,
                save_video=False, equirect_width=None):
    test_cameras = scene.getTestCameras()

    mkdir_flag = False
    progress_bar = tqdm(range(0, len(test_cameras)), desc="Relighting",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]

            # Override camera resolution for equirect mode if --equirect_width is set
            if equirect_width is not None:
                viewpoint.image_width = equirect_width
                viewpoint.image_height = equirect_width // 2

            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            image = results["render"]
            image = torch.clamp(image, 0.0, 1.0)

            if gaussians.use_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)

            # ── 点光源位置指示光球 ──
            point_lights = pbr_kwargs.get("point_lights", None) if pbr_kwargs else None
            if args.point_light_vis and point_lights and len(point_lights) > 0:
                image_pbr = _draw_light_indicator(
                    image_pbr, point_lights, viewpoint,
                    is_equirect=(equirect_width is not None))

            write_image_dict = {}
            write_image_dict.update({
                "render": image_pbr, 
            })

            if not mkdir_flag:
                os.makedirs(os.path.join(args.model_path, 'test_rli', env_name), exist_ok=True)
                mkdir_flag = True


            for key in write_image_dict:
                    save_image(torch.clamp(write_image_dict[key], 0.0, 1.0), 
                            os.path.join(args.model_path, 'test_rli', env_name, f"{viewpoint.image_name}_{idx}.png"))
                    

# ---- Intermediate video output helpers ----

# Standard intermediate attributes for video output.
# Each entry: (vis_dict_key, filename_suffix, is_color_or_1ch)
#   is_color_or_1ch: "color" (3-ch already), "1ch" (single-ch -> replicate to 3)
INTERMEDIATE_VIDEO_ATTRS = [
    ("base_color",          "albedo",           "color"),
    ("diffuse_pbr",         "diffuse",          "color"),
    ("specular_pbr",        "specular",         "color"),
    ("normal",              "normal",           "color"),
    ("pseudo_normal",       "pseudo_normal",    "color"),
    ("depth",               "depth",            "1ch"),
    ("roughness",           "roughness",        "1ch"),
    ("metallic",            "metallic",         "1ch"),
    ("visibility",          "occlusion",        "1ch"),
    ("incidents_light",  "incident_light",   "color"),
    ("radiance_color",      "radiance",         "color"),
    ("ref_strength",        "ref_strength",     "1ch"),
    ("ref_roughness",       "ref_roughness",    "1ch"),
    ("ref_tint",            "ref_tint",         "color"),
]


def _extract_intermediate_frame(results, vis_key, H_even, W_even, kind):
    """Extract a single intermediate result frame -> numpy uint8 [H, W, 3] or None."""
    img = None
    vis_dict = results.get("vis_dict")
    if vis_dict is not None and vis_key in vis_dict:
        img = vis_dict[vis_key]
    elif vis_key in results:
        img = results[vis_key]
    if img is None:
        return None

    img = img.detach().cpu().float()
    # Single-channel -> replicate to 3-ch for video encoding
    if kind == "1ch" and img.dim() == 3 and img.shape[0] == 1:
        img = img.expand(3, -1, -1)
    img = img.clamp(0.0, 1.0)
    img = img[:, :H_even, :W_even]
    img_np = img.permute(1, 2, 0).numpy()
    return (img_np * 255).astype("uint8")


def _save_intermediate_videos(all_frames, video_path, env_name):
    """Save all intermediate attribute videos."""
    for suffix, frames in all_frames.items():
        if len(frames) > 0:
            imageio.mimsave(
                os.path.join(video_path, f"{env_name}_{suffix}.mp4"),
                np.stack(frames), fps=24, macro_block_size=1,
            )


def eval_render_video(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, env_name,
                      full_video_output=False):
    test_cameras = scene.getTrainCameras()
    video_images_dict = []
    intermediate_frames = {suffix: [] for _, suffix, _ in INTERMEDIATE_VIDEO_ATTRS} if full_video_output else None
    
    camera = test_cameras[0]
    H = camera.image_height
    W = camera.image_width
    fovx = camera.FoVx
    fovy = focal2fov(fov2focal(fovx, W), H)

    n_frames = 180
    radius = 1  # toycar
    radius = 0.4 #garden
    

    cycle_cameras = []
    def circular_poses(viewpoint_cam, radius, angle=0.0):
        translate_x = radius * np.cos(angle)
        translate_y = radius * np.sin(angle)
        translate_z = 0
        translate = np.array([translate_x, translate_y, translate_z])
        
        custom_cam = Camera(colmap_id=0, R=viewpoint_cam.R, T=viewpoint_cam.T,
            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
            image=torch.zeros(3, H, W), image_name=None, uid=0,
            trans=translate
        )
        return custom_cam

    for idx in range(n_frames):
        # view = copy.deepcopy(test_cameras[25]) # toycar
        # view = copy.deepcopy(test_cameras[120]) # kitchen
        # view = copy.deepcopy(test_cameras[180]) # kitchen

        view = copy.deepcopy(test_cameras[160]) # garden

        angle = 2 * np.pi * idx / n_frames
        cam = circular_poses(view, radius, angle)
        cycle_cameras.append(cam)
    
    test_cameras = cycle_cameras



    # n_test = 180
    # R_list = []
    # T_list = []

    # fs = [0,  len(test_cameras) - 1, len(test_cameras) // 2, 0]
    # R = test_cameras[fs[0]].R
    # t = test_cameras[fs[0]].T
    # Rt = getWorld2View(R,t)
    # pose0 = Rt
    # for i in range(1, len(fs)):
    #     R = test_cameras[fs[i]].R
    #     t = test_cameras[fs[i]].T
    #     pose1 = getWorld2View(R,t)
    #     rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
    #     slerp = Slerp([0, 1], rots)
    #     for i in range(n_test + 1):
    #         ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
    #         pose = np.eye(4, dtype=np.float32)
    #         pose[:3, :3] = slerp(ratio).as_matrix()
    #         pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
            
    #         R = np.transpose(pose[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    #         T = pose[:3, 3]

    #         R_list.append(R)
    #         T_list.append(T)
            
    #         # custom_cam = Camera(colmap_id=0, R=R, T=T,
    #         #         FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
    #         #         image=torch.zeros(3, H, W), image_name=None, uid=0)
    #         # trace_cameras.append(custom_cam)
    #     pose0 = pose1


    # progress_bar = tqdm(range(0, len(R_list)), desc="Relighting",
    #                     initial=0, total=len(test_cameras))

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Relighting",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            # custom_cam = Camera(colmap_id=0, R=R_list[idx], T=T_list[idx],
            #         FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
            #         image=torch.zeros(3, H, W), image_name=None, uid=0)
            
            # viewpoint = custom_cam

            viewpoint = test_cameras[idx]


            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            image = results["render"]
            image = torch.clamp(image, 0.0, 1.0)

            if gaussians.use_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)

            H, W = image_pbr.shape[1], image_pbr.shape[2]
            H_resize, W_resize = H, W
            if H % 2 != 0:
                H_resize = H - 1
            if W % 2 != 0:
                W_resize = W -1

            tmp_image_pbr = image_pbr[:,:H_resize, :W_resize]
            video_image_pbr = torch.clamp(tmp_image_pbr, 0.0, 1.0).permute(1,2,0).detach().cpu()
            video_image_pbr = (video_image_pbr.numpy() * 255).astype('uint8')
            video_images_dict.append(video_image_pbr)

            # ---- Full video output: collect intermediate frames ----
            if full_video_output:
                for vis_key, suffix, kind in INTERMEDIATE_VIDEO_ATTRS:
                    frame = _extract_intermediate_frame(results, vis_key, H_resize, W_resize, kind)
                    if frame is not None:
                        intermediate_frames[suffix].append(frame)



                    
        video_path = os.path.join(args.model_path, 'test_rli', "video")
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, f"{env_name}_pbr_video.mp4"), np.stack(video_images_dict), fps=24, macro_block_size=1)

        if full_video_output:
            _save_intermediate_videos(intermediate_frames, video_path, env_name)
            print(f"  Saved {len([v for v in intermediate_frames.values() if len(v)>0])} intermediate videos to {video_path}")


def eval_render_video_equirect(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, env_name, equirect_width=None, full_video_output=False):
    """全景模式重光照视频：相机小幅圆周运动产生视差效果"""
    train_cameras = scene.getTrainCameras()
    if len(train_cameras) == 0:
        train_cameras = scene.getTestCameras()

    video_images_dict = []
    intermediate_frames = {suffix: [] for _, suffix, _ in INTERMEDIATE_VIDEO_ATTRS} if full_video_output else None

    ref_cam = train_cameras[0]
    H = ref_cam.image_height
    W = ref_cam.image_width
    if equirect_width is not None:
        W = equirect_width
        H = equirect_width // 2
    print('equirect_size: H:', H, 'W:', W)

    # 按场景尺寸的~10%作为运动半径，保持相机朝向不变，产生视差效果
    radius = scene.cameras_extent * 0.03
    n_frames = 120  # 5s at 24fps

    # ---- 光源旋转参数 ----
    cubemap = pbr_kwargs.get("cubemap")
    light_rotate_yaw = getattr(args, 'light_rotate_yaw', 0.0)
    light_rotate_pitch = getattr(args, 'light_rotate_pitch', 0.0)
    light_rotate_roll = getattr(args, 'light_rotate_roll', 0.0)
    has_light_rotation = (light_rotate_yaw != 0 or light_rotate_pitch != 0 or light_rotate_roll != 0)

    progress_bar = tqdm(range(n_frames), desc="Relighting (Equirect Video)")

    with torch.no_grad():
        for idx in progress_bar:
            # 水平面小幅圆周运动（保持相机朝向不变）
            angle = 2 * np.pi * idx / n_frames
            translate = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.0,
            ])

            custom_cam = Camera(
                colmap_id=0, R=ref_cam.R, T=ref_cam.T,
                FoVx=ref_cam.FoVx, FoVy=ref_cam.FoVy,
                fx=None, fy=None, cx=None, cy=None,
                image=torch.zeros(3, H, W), image_name=None, uid=0,
                trans=translate,
            )

            # ---- 光源逐帧旋转 ----
            if cubemap is not None and has_light_rotation:
                fraction = idx / (n_frames - 1) if n_frames > 1 else 1.0  # 0 → 1
                yaw_r = np.radians(light_rotate_yaw * fraction)
                pitch_r = np.radians(light_rotate_pitch * fraction)
                roll_r = np.radians(light_rotate_roll * fraction)

                cos_y, sin_y = np.cos(yaw_r), np.sin(yaw_r)
                cos_p, sin_p = np.cos(pitch_r), np.sin(pitch_r)
                cos_r, sin_r = np.cos(roll_r), np.sin(roll_r)

                R_y = torch.tensor([
                    [cos_y, 0.0, -sin_y],
                    [0.0, 1.0, 0.0],
                    [sin_y, 0.0, cos_y],
                ], dtype=torch.float32, device=cubemap.base.device)

                R_x = torch.tensor([
                    [1.0, 0.0, 0.0],
                    [0.0, cos_p, -sin_p],
                    [0.0, sin_p, cos_p],
                ], dtype=torch.float32, device=cubemap.base.device)

                R_z = torch.tensor([
                    [cos_r, -sin_r, 0.0],
                    [sin_r, cos_r, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=torch.float32, device=cubemap.base.device)

                # 旋转顺序：pitch → yaw → roll
                cubemap.xfm(R_z @ R_y @ R_x)
            elif cubemap is not None:
                cubemap.xfm(None)  # 无旋转时清除上一帧可能残留的 mtx

            # ── 点光源逐帧水平轨道旋转 ──
            # 轨道参数：pivot 为旋转中心，radius 为 XZ 水平面半径
            # COLMAP 空间：+Y 向下，旋转在 XZ 平面
            point_light_orbits = pbr_kwargs.get("_point_light_orbits", None)
            if point_light_orbits and "_point_light_pivots" in pbr_kwargs:
                fraction = idx / (n_frames - 1) if n_frames > 1 else 1.0
                eq_mode = equirect_width is not None
                rotated_lights = []
                rotated_shadow_funcs = []
                for li, (pivot) in enumerate(pbr_kwargs["_point_light_pivots"]):
                    orbit_radius, orbit_total = point_light_orbits[li]
                    if orbit_radius <= 0 or orbit_total == 0:
                        rotated_lights.append(pbr_kwargs["point_lights"][li])
                        rotated_shadow_funcs.append(pbr_kwargs["point_light_shadow_funcs"][li])
                        continue
                    theta = np.radians(orbit_total * fraction)
                    c, s = np.cos(theta), np.sin(theta)
                    new_pos = pivot + torch.tensor([orbit_radius * c, 0.0, orbit_radius * s], dtype=pivot.dtype, device=pivot.device)
                    rotated_lights.append(PointLight(position=new_pos, color=pbr_kwargs["point_lights"][li].color, intensity=pbr_kwargs["point_lights"][li].intensity))
                    # Re-bake shadow map from rotated position
                    _biases = pbr_kwargs.get("_point_light_biases", [])
                    _bias = _biases[li] if li < len(_biases) else 0.3
                    if eq_mode:
                        dm = get_depth_equirect(gaussians, new_pos)
                        rotated_shadow_funcs.append(make_shadow_func_equirect(dm, new_pos, threshold=_bias))
                    else:
                        dm = get_depth_cubemap(gaussians, new_pos)
                        rotated_shadow_funcs.append(make_shadow_func_cubemap(dm, new_pos, threshold=_bias))
                pbr_kwargs["point_lights"] = rotated_lights
                pbr_kwargs["point_light_shadow_funcs"] = rotated_shadow_funcs

            results = render_fn(custom_cam, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            image_pbr = results["pbr"]
            image_pbr = torch.clamp(image_pbr, 0.0, 1.0)

            # ── 点光源位置指示光球（视频每帧） ──
            v_point_lights = pbr_kwargs.get("point_lights", None) if pbr_kwargs else None
            if args.point_light_vis and v_point_lights and len(v_point_lights) > 0:
                image_pbr = _draw_light_indicator(
                    image_pbr, v_point_lights, custom_cam,
                    is_equirect=(equirect_width is not None))

            # 确保宽高为偶数（视频编码要求）
            H_actual, W_actual = image_pbr.shape[1], image_pbr.shape[2]
            H_resize, W_resize = H_actual, W_actual
            if H_actual % 2 != 0:
                H_resize = H_actual - 1
            if W_actual % 2 != 0:
                W_resize = W_actual - 1

            video_image = image_pbr[:, :H_resize, :W_resize].permute(1, 2, 0).detach().cpu()
            video_image = (video_image.numpy() * 255).astype('uint8')
            video_images_dict.append(video_image)

            # ---- Full video output: collect intermediate frames ----
            if full_video_output:
                for vis_key, suffix, kind in INTERMEDIATE_VIDEO_ATTRS:
                    frame = _extract_intermediate_frame(results, vis_key, H_resize, W_resize, kind)
                    if frame is not None:
                        intermediate_frames[suffix].append(frame)

    video_path = os.path.join(args.model_path, 'test_rli', "video")
    os.makedirs(video_path, exist_ok=True)
    imageio.mimsave(os.path.join(video_path, f"{env_name}_equirect_video.mp4"),
                    np.stack(video_images_dict), fps=24, macro_block_size=1)

    if full_video_output:
        _save_intermediate_videos(intermediate_frames, video_path, env_name)
        print(f"  Saved {len([v for v in intermediate_frames.values() if len(v)>0])} intermediate videos to {video_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render_ref', 'render_ref_pbr', 'render_ref_fast',
                                                   'render_ref_equirect', 'render_ref_pbr_equirect'], default='render_ref')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to .ply file (alternative to --checkpoint)")
    parser.add_argument("--occlusion_path", type=str, default=None)
    parser.add_argument('-e', '--envmap_path', default="/home/huangpengyue/projects/RTR-GS/data/env_maps/", help="Env map path")
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--equirect_width", type=int, default=None,
                        help="Equirect output width (height=width/2). If not set, uses camera native resolution.")
    parser.add_argument("--light_rotate_yaw", type=float, default=0.0,
                        help="Total light yaw rotation (degrees) over video duration")
    parser.add_argument("--light_rotate_pitch", type=float, default=0.0,
                        help="Total light pitch rotation (degrees) over video duration")
    parser.add_argument("--light_rotate_roll", type=float, default=0.0,
                        help="Total light roll rotation (degrees) over video duration")
    parser.add_argument("--full_video_output", action="store_true", default=False,
                        help="Also output videos of intermediate results (albedo, normal, depth, "
                             "diffuse, specular, roughness, metallic, occlusion, incident light, etc.)")
    parser.add_argument("--point_lights_config", type=str, default=None,
                        help="点光源 JSON 配置文件路径")
    parser.add_argument("--point_light_vis", action="store_true", default=False,
                        help="在渲染图像上绘制点光源位置指示光球")

    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf', 'neilf_blend', 'neilf_forward', 'neilf_ref_pbr', 'render_ref_pbr', 'render_ref_pbr_equirect']
    is_equirect = 'equirect' in args.type
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr, is_equirect=is_equirect)

    # All done
    print("\nTraining complete.")
