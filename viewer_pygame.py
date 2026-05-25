import copy
import os
import cv2
import torch
import numpy as np
import pygame
from gaussian_renderer import render_fn_dict
from pbr import CubemapLight, get_brdf_lut
from scene import GaussianModel, Scene
from scene.transfer_mlp import TransferMLP
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal
from utils.general_utils import load_json_config
from utils.sh_utils import eval_sh
import torch.nn.functional as F
import math

# mipnerf/counter
# mipnerf/room
# mipnerf/garden
# ./data/mipnerf/360_v2/garden
# refnerf/helmet
# tensoIR/toaster
# 803-hdri-skies-com.hdr
# big-studio-01_4K.exr
# data\env_maps\high_res_envmaps_1k\square.hdr
# directional_front_top.hdr
    # -s ./data/mipnerf/360_v2/kitchen/ \
"""
source E:/Anaconda/etc/profile.d/conda.sh
conda activate odgs-rtr
python viewer_pygame.py \
    -c lab_output/tensoIR/lego/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path lab_output/tensoIR/lego/stage1/checkpoint/occlusion_volumes.pth \
    --envmap_path "./data/env_maps/directional_front_top.hdr" \
    --image_width 512 \
    --image_height 512
"""

"""
source E:/Anaconda/etc/profile.d/conda.sh
conda activate odgs-rtr
python viewer_pygame.py \
    -s ./data/mipnerf/360_v2/counter \
    -c lab_output/mipnerf/360_v2/counter/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path lab_output/mipnerf/360_v2/counter/stage1/checkpoint/occlusion_volumes.pth \
    --envmap_path "d:/localSpace/relighting/env_maps/big-studio-01_4K.exr" \
    --image_width 512 \
    --image_height 512
"""

def to_4x4_rot(R):
    """将3x3旋转矩阵扩展为4x4齐次矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    return T

def get_a2b_matrix(a=np.array([0, 1, 0]), b=np.array([0, 1, 0])):
    """计算旋转变换满足 b = R @ a
    
    Args:
        a: 源空间的方向
        b: 目标空间的方向
    
    Returns:
        3x3 旋转矩阵，将a空间中的向量变换到b空间
    """
    a = np.array(a, dtype=np.float64)
    a = a / np.linalg.norm(a)
    
    if b is None:
        b = np.array([0, 1, 0], dtype=np.float64)
    else:
        b = np.array(b, dtype=np.float64)
        b = b / np.linalg.norm(b)
    
    # 计算旋转轴（a_up 和 b_up 的叉积）
    rotation_axis = np.cross(a, b)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    # 如果两个向量平行或反平行，无需旋转
    if rotation_axis_norm < 1e-6:
        if np.dot(a, b) > 0:
            return np.eye(3)
        else:
            # 选择一个垂直于 a 的轴
            if abs(a[0]) < 0.9:
                v = np.cross(a, [1, 0, 0])
            else:
                v = np.cross(a, [0, 1, 0])
            v = v / np.linalg.norm(v)
            # 旋转180度：R = 2 v v^T - I
            return 2 * np.outer(v, v) - np.eye(3)
    
    rotation_axis = rotation_axis / rotation_axis_norm
    
    # 计算旋转角度
    cos_angle = np.dot(a, b)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # 使用罗德里格斯公式构建旋转矩阵
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ], dtype=np.float64)
    
    R = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R

class FPSCamera():
    """FPS 风格的相机控制器，支持 WASD 移动和鼠标视角控制，支持 Orbit 模式切换"""
    
    def __init__(self, position, target, up=np.array([0, 1, 0]), FoVy=0.5, FoVx=0.5, image_width=512, image_height=512):
        
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up = np.array(up, dtype=np.float64)
        self.update_true_global_down()
        self.FoVy = FoVy
        self.FoVx = FoVx
        self.image_width = image_width
        self.image_height = image_height
        self.trans=np.array([0.0, 0.0, 0.0])
        
        # 计算初始视角方向
        self.forward = self.target - self.position
        dist = np.linalg.norm(self.forward)
        if dist > 1e-6:
            self.forward = self.forward / dist
        else:
            self.forward = np.array([0, 0, -1], dtype=np.float64)
        
        # 速度参数
        self.move_speed = 1
        self.look_speed = 0.002
        
        # Orbit 模式参数
        self.mode = 'orbit'  # 'fps' or 'orbit'
        self.orbit_radius = max(0.1, np.linalg.norm(self.position - self.target))
        self.orbit_theta = np.arctan2(self.forward[0], self.forward[2])  # 水平角
        self.orbit_phi = np.arcsin(self.forward[1])  # 垂直角
        self.orbit_speed = 0.01
        
        # FPS 模式俯仰角度限制
        self.yaw = np.arctan2(self.forward[0], self.forward[2])  # 初始偏航角
        self.pitch = np.arcsin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))  # 初始俯仰角
        self.max_pitch = np.pi / 2 - 0.01  # 最大俯仰角，避免极点

    def update_RT(self):
        """更新相机变换矩阵"""
        c2w = self.get_c2w_matrix()
        
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        self.R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        self.T = w2c[:3, 3]
        
    def move_forward(self, delta):
        """前后移动"""
        self.position += self.forward * delta * self.move_speed
        
    def move_right(self, delta):
        """左右平移"""
        right = np.cross(self.forward, self.up)
        right = right / np.linalg.norm(right)
        self.position += right * delta * self.move_speed
        
    def move_up(self, delta):
        """上下移动"""
        self.position += self.up * delta * self.move_speed
        
    def look(self, dx, dy):
        """鼠标视角控制 - 使用增量旋转避免万向锁"""
        if self.mode == 'orbit':
            # Orbit 模式：绕目标点旋转
            self.orbit_theta += dx * self.orbit_speed
            self.orbit_phi += dy * self.orbit_speed
            # 限制 phi 范围，避免极点和翻转
            self.orbit_phi = np.clip(self.orbit_phi, -np.pi/2 + 0.01, np.pi/2 - 0.01)

            # 根据球坐标更新位置
            x = self.orbit_radius * np.cos(self.orbit_phi) * np.sin(self.orbit_theta)
            y = self.orbit_radius * np.sin(self.orbit_phi)
            z = self.orbit_radius * np.cos(self.orbit_phi) * np.cos(self.orbit_theta)
            self.position = self.target + np.array([x, y, z])
            
            # 更新 forward 方向
            self.forward = self.target - self.position
            self.forward = self.forward / np.linalg.norm(self.forward)
        else:
            # 根据之前 forward 方向更新 yaw 和 pitch 角度
            self.yaw = np.arctan2(self.forward[0], self.forward[2])
            self.pitch = np.arcsin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))

            # FPS 模式：直接更新欧拉角
            self.yaw += dx * self.look_speed
            self.pitch += dy * self.look_speed
            
            # 限制俯仰角，避免万向锁和翻转
            self.pitch = np.clip(self.pitch, -self.max_pitch, self.max_pitch)

            self.forward = np.array([
                np.sin(self.yaw) * np.cos(self.pitch),
                np.sin(self.pitch),
                np.cos(self.yaw) * np.cos(self.pitch)
            ])
            self.forward = self.forward / np.linalg.norm(self.forward)
    
    def toggle_mode(self):
        """切换 FPS/Orbit 模式"""
        self.mode = 'orbit' if self.mode == 'fps' else 'fps'
        print(f"Switched to {self.mode} mode")
        
        if self.mode == 'orbit':
            # 切换到 Orbit 时，保存当前状态
            self.orbit_radius = np.linalg.norm(self.position - self.target)
            self.orbit_theta = np.arctan2(self.forward[0], self.forward[2])
            self.orbit_phi = np.arcsin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))
        else:
            # 切换到 FPS 时，更新 forward 方向和 pitch 角度
            self.forward = self.target - self.position
            forward_norm = np.linalg.norm(self.forward)
            if forward_norm > 1e-6:
                self.forward = self.forward / forward_norm
            else:
                # 如果位置重合，给一个默认的向前方向
                self.forward = np.array([0, 0, -1], dtype=np.float64)
            # 更新 pitch 角度
            self.pitch = np.arcsin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))
        
    def get_c2w_matrix(self):
        """获取相机到世界的变换矩阵（c2w）"""
        # 计算相机坐标系
        forward = self.forward  # 相机看向 forward 方向
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward = forward / forward_norm
        else:
            forward = np.array([0, 0, -1], dtype=np.float64)
        
        up = self.up
        true_up_norm = np.linalg.norm(up)
        if true_up_norm > 1e-6:
            up = up / true_up_norm
        else:
            up = np.array([0, 1, 0], dtype=np.float64)
        
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            # 如果 forward 和 true_up 共线，换一个 right 方向
            right = np.array([1, 0, 0], dtype=np.float64)
        
        up = np.cross(right, forward)
        
        # 应用 global 到 true_global 的变换矩阵
        # R = self.a2b
        # right = R @ right
        # up = R @ up
        # forward = R @ forward
        
        c2wc = np.eye(4, dtype=np.float32)
        c2wc[:3, 0] = right
        c2wc[:3, 1] = up
        c2wc[:3, 2] = -forward
        c2wc[:3, 3] = self.position
        
        c2w = np.linalg.inv(self.w2wc) @ c2wc
        
        return c2w

    def set_cam_pos_by_c2w_matrix(self, c2w):
        """反向计算"""
        c2wc = self.w2wc @ c2w
        self.position = c2wc[:3, 3]
        self.forward = -c2wc[:3, 2]
    
    def get_camera_center(self):
        """获取相机位置"""
        return self.position

    def update_true_global_down(self, true_down=np.array([0, -1, 0])):
        self.w2wc = to_4x4_rot(get_a2b_matrix(true_down, np.array([0, -1, 0])))


def load_scene_data(checkpoint_path, occlusion_path, envmap_path, resolution=2):
    """加载场景数据（复用 eval_relighting_tensorIR.py 的逻辑）"""
    
    # 加载高斯模型
    gaussians = GaussianModel(sh_degree=3, render_type='render_ref_pbr')
    iteration = gaussians.create_from_ckpt(checkpoint_path, restore_optimizer=False)
    
    # 设置 base_color_scale（与 eval_relighting_tensorIR.py 保持一致）
    gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    print("Albedo scale:", gaussians.base_color_scale)
    
    # 加载传输网络
    transfer_net = TransferMLP(sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres)
    transfer_net_checkpoint = os.path.dirname(checkpoint_path) + "/transfer_net_" + os.path.basename(checkpoint_path)
    if os.path.exists(transfer_net_checkpoint):
        transfer_net.create_from_ckpt(transfer_net_checkpoint)
        print("Successfully loaded transfer net!")
    else:
        print("Warning: transfer net checkpoint not found!")
    
    # 加载 occlusion volumes
    occlusion_volumes = torch.load(occlusion_path)
    bound = occlusion_volumes["bound"]
    
    # 加载环境光
    from utils.graphics_utils import read_hdr, latlong_to_cubemap
    hdri = read_hdr(envmap_path)
    hdri = torch.from_numpy(hdri).cuda()
    res = 256
    cubemap = CubemapLight(base_res=res).cuda()
    cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
    cubemap.build_mips()
    cubemap.eval()
    # if pipe.transfer_light:  # 不需要，因为我们用纯 PBR 模式
    # cubemap.build_sh(3)
    # gaussians.incident_to_transfer(cubemap.shs)
    
    # 加载 BRDF LUT
    brdf_lut = get_brdf_lut().cuda()
    
    return {
        'gaussians': gaussians,
        'iteration': iteration,
        'transfer_net': transfer_net,
        'occlusion_volumes': occlusion_volumes,
        'bound': bound,
        'cubemap': cubemap,
        'brdf_lut': brdf_lut
    }

def get_canonical_rays(image_width: int, image_height: int, FoVx: float, FoVy: float):
    # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
    # get reference camera
    # ref_camera = self.train_cameras[scale][0]
    # TODO: inject intrinsic
    H, W = image_height, image_width
    cen_x = W / 2
    cen_y = H / 2
    tan_fovx = math.tan(FoVx * 0.5)
    tan_fovy = math.tan(FoVy * 0.5)
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = torch.meshgrid(
        torch.arange(W, device='cuda'),
        torch.arange(H, device='cuda'),
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
    camera_dirs = camera_dirs / torch.norm(camera_dirs, dim=1, keepdim=True)
    print("camera_dirs_shape: {}".format(camera_dirs.shape))
    return camera_dirs



def render_frame(fps_cam: FPSCamera, scene_data, canonical_rays: torch.Tensor):
    """渲染单帧画面"""
    
    
    gaussians = scene_data['gaussians']
    transfer_net = scene_data['transfer_net']
    occlusion_volumes = scene_data['occlusion_volumes']
    cubemap = scene_data['cubemap']
    brdf_lut = scene_data['brdf_lut']
    bound = scene_data['bound']
    enable_occlusion = scene_data.get('enable_occlusion', True)

    viewpoint_camera = Camera(
        colmap_id=0,
        R=fps_cam.R,
        T=fps_cam.T,
        FoVx=fps_cam.FoVx,
        FoVy=fps_cam.FoVy,
        trans=fps_cam.trans if fps_cam.trans is not None else np.array([0, 0, 0]),
        fx=None,
        fy=None,
        cx=None,
        cy=None,
        image = None,
        # image=torch.zeros(3, image_height, image_width, device='cuda'),
        width=fps_cam.image_width,
        height=fps_cam.image_height,
        image_name="view",
        render_only=True,
        uid=0)

    # 准备渲染参数
    pipe = type('Pipe', (), {
        'debug': False,
        'compute_with_prt': False,
        # 'compute_with_prt': False,
        'diffuse_iteration': 3000,
        'compute_cov3D_python': False,
        'compute_SHs_python': False,
        'metallic': True,      # 启用金属材质
        'ref_map': True,       # 使用反射图
        'compute_pseudo_normal': False,
        'relight': True,       # 启用重光照
        'tone_mapping': True,  # 启用色调映射
        'transfer_light': False,  # 是否使用传输光照
        'white_background': False,  # 背景颜色，根据数据集调整
        'forward_shading': False
    })()
    
    bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    
    # 准备 aabb
    aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
    
    # 渲染
    render_kwargs = {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": bg_color,
        "is_training": False,
        "dict_params": {
            "transfer_net": transfer_net,
            "occlusion_volumes": occlusion_volumes,
            "aabb": aabb,
            "cubemap": cubemap,
            "refmap": cubemap,  # 启用反射贴图
            "brdf_lut": brdf_lut,
            "canonical_rays": canonical_rays,
            "iteration": 999999999,  # 用于判断 diffuse/specular 阶段
            "relight": True,
            "enable_occlusion": enable_occlusion,  # 遮挡开关
        },
    }
    
    # render_fn = render_fn_dict['neilf_ref_fast']
    render_fn = render_fn_dict['neilf_ref']
    render_pkg = render_fn(viewpoint_camera=viewpoint_camera, **render_kwargs)
    
    # 获取渲染结果（使用 pbr 键而不是 render 键）
    image = render_pkg["pbr"]
    # image = render_pkg["render"]
    
    # 转换为 numpy 格式
    image_np = image.detach().permute(1, 2, 0).cpu().numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    image_np = (image_np * 255).astype(np.uint8)

    env_bg_np = None
    opacity_np = None
    if "env_only" in render_pkg and "opacity" in render_pkg:
        env_img = render_pkg["env_only"]
        env_bg_np = env_img.detach().permute(1, 2, 0).cpu().numpy()
        env_bg_np = np.clip(env_bg_np, 0.0, 1.0)
        env_bg_np = (env_bg_np * 255).astype(np.uint8)

        opacity = render_pkg["opacity"].detach().cpu().numpy()
        if opacity.ndim == 3 and opacity.shape[0] == 1:
            opacity = opacity[0]
        opacity = np.clip(opacity, 0.0, 1.0)
        opacity_np = np.repeat(opacity[..., None], 3, axis=-1)

    return image_np, env_bg_np, opacity_np

    cycle_cameras = []
def circular_poses(radius, angle=0.0):
    translate_x = radius * np.cos(angle)
    translate_y = radius * np.sin(angle)
    translate_z = 0
    translate = np.array([translate_x, translate_y, translate_z])
    
    # custom_cam = Camera(colmap_id=0, R=viewpoint_cam.R, T=viewpoint_cam.T,
    #     FoVx=viewpoint_cam.FoVx, FoVy=viewpoint_cam.FoVy, fx=None, fy=None, cx=None, cy=None,
    #     width=viewpoint_cam.image_width,
    #     height=viewpoint_cam.image_height, image_name=None, uid=0,
    #     render_only=True,
    #     trans=translate,
    #     image=None,
    # )
    return translate

def main():
    import argparse

    # 命令行参数
    parser = argparse.ArgumentParser(description="RTR-GS FPS Viewer (Pygame Version)")
    # parser.add_argument("-m", "--model_path", type=str, required=True,
    #                     help="Path to model directory")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("-s", "--source_path", type=str, required=False,
                        help="Path to scene source directory")
    parser.add_argument("--occlusion_path", type=str, required=True,
                        help="Path to occlusion volumes")
    parser.add_argument("--envmap_path", type=str, 
                        default="d:/localSpace/relighting/data/env_maps/big-studio-01_4K.exr",
                        help="Path to environment map")
    parser.add_argument("--resolution", type=int, default=2,
                        help="Resolution scale")
    parser.add_argument("--image_width", type=int, default=512,
                        help="Output image width")
    parser.add_argument("--image_height", type=int, default=512,
                        help="Output image height")
    parser.add_argument("--transform_path", type=str, default=None,
                        help="Path to transforms.json")

    args = parser.parse_args()

    # 从 args.source_path 加载场景(只加载相机)
    if args.source_path is None:
        scene = None
        test_cams = None
        is_colmap = False
    else:
        scene = Scene(args, read_cam_only=True, shuffle=False)
        test_cams = scene.getTestCameras()
        is_colmap = os.path.exists(os.path.join(args.source_path, "sparse"))
        if is_colmap:
            cycle_cameras = []
            n_frames = 180
            radius = 1  # toycar
            radius = 0.4 #garden
            for idx in range(n_frames):
                # view = copy.deepcopy(test_cameras[25]) # toycar
                # view = copy.deepcopy(test_cameras[120]) # kitchen
                # view = copy.deepcopy(test_cameras[180]) # kitchen

                cam = copy.deepcopy(test_cams[0]) # garden

                angle = 2 * np.pi * idx / n_frames
                cam.trans = circular_poses(radius, angle)
                cycle_cameras.append(cam)
            
            test_cams = cycle_cameras
    
    print("Loading scene data...")
    scene_data = load_scene_data(
        args.checkpoint,
        args.occlusion_path,
        args.envmap_path,
        args.resolution
    )
    
    # 初始化相机（从场景上方开始，默认 Orbit 模式）
    gaussians = scene_data['gaussians']
    scene_data['enable_occlusion'] = True
    scene_center = gaussians.get_xyz.detach().mean(dim=0).cpu().numpy()
    
    # 计算场景边界
    scene_min = gaussians.get_xyz.detach().min(dim=0).values.cpu().numpy()
    scene_max = gaussians.get_xyz.detach().max(dim=0).values.cpu().numpy()
    scene_size = np.maximum(scene_max - scene_min, 0.1)
    scene_radius = np.linalg.norm(scene_size) / 2.0
    
    # FPS 模式：在场景中心前方一点的位置
    # Orbit 模式：在场景上方
    # if fps_cam.mode == 'fps':
    # camera_distance = 0  # 更近一些
    # initial_position = scene_center + np.array([0, 0, camera_distance])  # 从 Z 轴前方开始
    # else:
    # Orbit 模式：在场景上方开始
    camera_distance = max(2.0, scene_radius * 2.5)
    initial_position = scene_center + np.array([0, 0, camera_distance])
    
    print(f"Scene center: {scene_center}")
    print(f"Scene radius: {scene_radius:.2f}")
    print(f"Camera distance: {camera_distance:.2f}")
    
    # ref_camera = scene.train_cameras[1.0][0]
    fps_cam = FPSCamera(
        position=initial_position,
        target=scene_center,
        FoVy=0.5,
        FoVx=0.5,
        image_height=args.image_height,
        image_width=args.image_width
    )
    
    # canonical_rays = scene.get_canonical_rays()
    canonical_rays = get_canonical_rays(args.image_width, args.image_height, fps_cam.FoVx, fps_cam.FoVy)
    
    env_rotation_y = 0.0  # 环境光绕Y轴旋转角度（弧度）
    env_rotation_x = 0.0  # 环境光绕X轴旋转角度（弧度）
    
    print("\nControls:")
    print("  FPS Mode (press M to toggle):")
    print("    W/S: Move forward/backward")
    print("    A/D: Move left/right")
    print("    Q/E: Move up/down")
    print("    Right mouse + drag: Rotate camera")
    print("    SPACE: Reset forward direction")
    print("  Orbit Mode (press M to toggle):")
    print("    Right mouse + drag: Rotate around target")
    print("  Both modes:")
    print("    Mouse wheel: Zoom in/out (Orbit only)")
    print("    LEFT/RIGHT: Rotate environment map")
    print("    R: Reset environment rotation")
    print("    B: Toggle envmap background")
    print("    O: Toggle occlusion (AO)")
    print("    ESC: Exit")
    print(f"\nStarting in {fps_cam.mode} mode...")
    
    # 初始化 Pygame
    pygame.init()
    screen = pygame.display.set_mode((args.image_width, args.image_height))
    pygame.display.set_caption("RTR-GS Viewer (Pygame)")
    clock = pygame.time.Clock()
    
    # 显示鼠标光标
    pygame.mouse.set_visible(True)
    
    # 鼠标状态 - 右键拖动
    right_mouse_pressed = False
    last_mouse_pos = pygame.mouse.get_pos()
    
    playing_transforms = False
    play_index = 0
    show_env_bg = False  # 是否在背景绘制环境光贴图
    running = True
    with torch.no_grad():
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        playing_transforms = not playing_transforms
                        if not test_cams:
                            playing_transforms = False
                            print("No transforms available for playback.")
                    if event.key == pygame.K_ESCAPE:
                        if playing_transforms:
                            playing_transforms = False
                        else:
                            running = False
                    elif event.key == pygame.K_m:
                        fps_cam.toggle_mode()
                    elif event.key == pygame.K_SPACE:
                        # 空格键现在可以用来重置视角
                        fps_cam.forward = np.array([0, 0, -1], dtype=np.float64)
                    elif event.key == pygame.K_r:
                        env_rotation_y = 0.0
                        env_rotation_x = 0.0
                        update_env_rotation(scene_data['cubemap'], env_rotation_y, env_rotation_x, fps_cam.w2wc)
                    elif event.key == pygame.K_b:
                        show_env_bg = not show_env_bg
                        print(f"Env background: {'ON' if show_env_bg else 'OFF'}")
                    elif event.key == pygame.K_o:
                        scene_data['enable_occlusion'] = not scene_data['enable_occlusion']
                        print(f"Occlusion: {'ON' if scene_data['enable_occlusion'] else 'OFF'}")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 3:  # 右键按下
                        right_mouse_pressed = True
                        last_mouse_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3:  # 右键释放
                        right_mouse_pressed = False
                elif event.type == pygame.MOUSEWHEEL:
                    # 滚轮缩放（仅在 Orbit 模式）
                    if fps_cam.mode == 'orbit':
                        zoom_speed = 0.2
                        if event.y > 0:  # 向上滚动
                            fps_cam.orbit_radius = max(0.1, fps_cam.orbit_radius - zoom_speed)
                        else:  # 向下滚动
                            fps_cam.orbit_radius += zoom_speed
                        
                        # 更新位置
                        x = fps_cam.orbit_radius * np.cos(fps_cam.orbit_phi) * np.sin(fps_cam.orbit_theta)
                        y = fps_cam.orbit_radius * np.sin(fps_cam.orbit_phi)
                        z = fps_cam.orbit_radius * np.cos(fps_cam.orbit_phi) * np.cos(fps_cam.orbit_theta)
                        fps_cam.position = fps_cam.target + np.array([x, y, z])
                        fps_cam.forward = fps_cam.target - fps_cam.position
                        fps_cam.forward = fps_cam.forward / np.linalg.norm(fps_cam.forward)
            
            if playing_transforms:
                ref_cam = test_cams[play_index]
                fps_cam.R = ref_cam.R
                fps_cam.T = ref_cam.T
                fps_cam.trans = ref_cam.trans
                play_index += 1
                play_index = play_index % len(test_cams)
            else: 
                # 处理持续按键（Pygame 的优势！）
                keys = pygame.key.get_pressed()
                if fps_cam.mode == 'fps':
                    if keys[pygame.K_w]:
                        fps_cam.move_forward(1.0)
                    if keys[pygame.K_s]:
                        fps_cam.move_forward(-1.0)
                    if keys[pygame.K_a]:
                        fps_cam.move_right(-1.0)
                    if keys[pygame.K_d]:
                        fps_cam.move_right(1.0)
                    if keys[pygame.K_q]:
                        fps_cam.move_up(-1.0)
                    if keys[pygame.K_e]:
                        fps_cam.move_up(1.0)
                if keys[pygame.K_u]:
                    # 更新为当前视角的down
                    fps_cam.update_true_global_down(fps_cam.forward)
                    update_env_rotation(scene_data['cubemap'], env_rotation_y, env_rotation_x, fps_cam.w2wc)
                if keys[pygame.K_LEFT]:
                    env_rotation_y -= 0.1
                    update_env_rotation(scene_data['cubemap'], env_rotation_y, env_rotation_x, fps_cam.w2wc)
                if keys[pygame.K_RIGHT]:
                    env_rotation_y += 0.1
                    update_env_rotation(scene_data['cubemap'], env_rotation_y, env_rotation_x, fps_cam.w2wc)
                if keys[pygame.K_UP]:
                    env_rotation_x += 0.1
                    update_env_rotation(scene_data['cubemap'], env_rotation_y, env_rotation_x, fps_cam.w2wc)
                if keys[pygame.K_DOWN]:
                    env_rotation_x -= 0.1
                    update_env_rotation(scene_data['cubemap'], env_rotation_y, env_rotation_x, fps_cam.w2wc)
                
                # 处理鼠标移动 - 右键拖动
                if right_mouse_pressed:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - last_mouse_pos[0]
                    dy = current_mouse_pos[1] - last_mouse_pos[1]
                    if dx != 0 or dy != 0:
                        fps_cam.look(dx, dy)
                    last_mouse_pos = current_mouse_pos
                
                # 根据forward、position更新相机 RT
                fps_cam.update_RT()
            
            # 渲染当前帧
            image_np, env_bg_np, opacity_np = render_frame(fps_cam, scene_data, canonical_rays)
            
            # 背景环境光: 替换白色背景为env_only
            # renderer中 pbr = pbr_raw*α + white*(1-α), 所以:
            #   display = pbr_raw*α + env*(1-α) = pbr + (env - 1.0)*(1-α)
            if show_env_bg and env_bg_np is not None and opacity_np is not None:
                pbr_f = image_np.astype(np.float32) / 255.0
                env_f = env_bg_np.astype(np.float32) / 255.0
                display_f = pbr_f + (env_f - 1.0) * (1.0 - opacity_np)
                display_np = (np.clip(display_f, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                display_np = image_np
            
            # 转换为 Pygame 表面
            image_surface = pygame.surfarray.make_surface(np.transpose(display_np, (1, 0, 2)))
            
            # 显示信息
            font = pygame.font.SysFont('Arial', 20)
            mode_text = font.render(f"Mode: {fps_cam.mode.upper()}", True, (0, 255, 0))
            pos_text = font.render(f"Pos: {fps_cam.position[0]:.2f}, {fps_cam.position[1]:.2f}, {fps_cam.position[2]:.2f}", True, (0, 255, 0))
            fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (0, 255, 0))
            yaw_text = font.render(f"Yaw: {fps_cam.yaw * 180 / np.pi:.2f}", True, (0, 255, 0))
            pitch_text = font.render(f"Pitch: {fps_cam.pitch * 180 / np.pi:.2f}", True, (0, 255, 0))
            env_rot_text = font.render(f"Env Rot: {env_rotation_y * 180 / math.pi:.1f}° [←→]", True, (0, 255, 0))
            env_bg_text = font.render(f"Env BG: {'ON' if show_env_bg else 'OFF'} [B]", True, (0, 255, 0))
            
            # 绘制到屏幕
            screen.blit(image_surface, (0, 0))
            screen.blit(mode_text, (10, 10))
            screen.blit(fps_text, (10, 40))
            screen.blit(yaw_text, (10, 70))
            screen.blit(pitch_text, (10, 100))
            screen.blit(env_rot_text, (10, 130))
            screen.blit(env_bg_text, (10, 160))
            if fps_cam.mode == 'orbit':
                radius_text = font.render(f"Radius: {fps_cam.orbit_radius:.2f}", True, (0, 255, 0))
                screen.blit(radius_text, (10, 190))
                screen.blit(pos_text, (10, 220))
            else:
                screen.blit(pos_text, (10, 190))
            
            # 更新显示
            pygame.display.flip()
            
            # 控制帧率
            clock.tick(60)
        
    pygame.quit()
    print("Viewer closed.")


def update_env_rotation(cubemap, angle_y, angle_x = 0.0, w2wc=None):
    cos_y = math.cos(angle_y)
    sin_y = math.sin(angle_y)
    cos_x = math.cos(angle_x)
    sin_x = math.sin(angle_x)

    rot_x = torch.tensor([
        [1.0,    0.0,     0.0    ],
        [0.0,  cos_x,  -sin_x   ],
        [0.0,  sin_x,   cos_x   ]
    ], dtype=torch.float32)

    rot_y = torch.tensor([
        [cos_y,  0.0,  -sin_y  ],
        [0.0,    1.0,   0.0    ],
        [sin_y,  0.0,   cos_y  ]
    ], dtype=torch.float32)

    rotation_matrix = rot_y @ rot_x

    if w2wc is not None:
        w2wc_rot = torch.tensor(w2wc[:3, :3], dtype=torch.float32)
        rotation_matrix = rotation_matrix @ w2wc_rot

    cubemap.xfm(rotation_matrix)


if __name__ == "__main__":
    main()
