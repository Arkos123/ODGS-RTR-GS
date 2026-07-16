"""
RTR-GS 交互式查看器（clean 重写版）

基于 train.py / eval_relighting_*.py 的渲染调用方式，复用 scene.cameras.Camera 类。
全部工作在 COLMAP 世界空间（+Y 下, +Z 前），无 w2wc / a2b / true_global_down 等复杂转换。

用法:
    # Perspective 模式（MipNeRF / Blender / TensoIR 场景）
    python viewer_clean.py \\
        -c output/checkpoint/chkpnt40000.pth \\
        --occlusion_path output/occlusion_volumes.pth \\
        --envmap_path data/env_maps/big-studio-01_4K.exr

    # Equirect 模式（OmniBlender 场景）
    python viewer_clean.py \\
        -c lab_output/.../stage2/checkpoint/chkpnt36000.pth \\
        --occlusion_path lab_output/.../occlusion_volumes.pth \\
        --envmap_path data/env_maps/big-studio-01_4K.exr \\
        --equirect
"""

import os
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pygame
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Tuple, List, Optional

from scene.cameras import Camera
from scene import GaussianModel
from scene.transfer_mlp import TransferMLP
from gaussian_renderer import render_fn_dict
from pbr import CubemapLight, PointLight, get_brdf_lut
from pbr.point_light_shadow import (
    get_depth_cubemap, get_depth_equirect,
    make_shadow_func_cubemap, make_shadow_func_equirect,
)
from utils.graphics_utils import read_hdr, latlong_to_cubemap, latlong_to_cubemap_equirect
from torchvision.utils import save_image
from plyfile import PlyData

# ══════════════════════════════════════════════════════════════════════════════
#  Camera Controller — 纯粹工作在 COLMAP 世界空间
#  +Y = down, +Z = forward（视觉上方向 = [0, -1, 0]）
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_UP = np.array([0, -1, 0], dtype=np.float64)  # COLMAP +Y=down, 视觉上方向=-Y


class CameraController:
    """FPS / Orbit 双模式相机控制器。

    全部空间在 COLMAP world frame（+Y 下, +Z 前）。
    与 eval_relighting_*.py 相同的 (R, T) 构造方式。
    """

    def __init__(self, position, target, up_ref=_DEFAULT_UP,
                 FoVx=0.6, FoVy=0.6, width=512, height=512,
                 render_mode="pbr"):
        self.render_mode = render_mode
        self.position = np.array(position, dtype=np.float64)
        self.target = np.array(target, dtype=np.float64)
        self.up_ref = np.array(up_ref, dtype=np.float64)
        self.up_ref /= np.linalg.norm(self.up_ref)

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.width = width
        self.height = height

        # forward = 归一化的视线方向（COLMAP 世界空间）
        delta = self.target - self.position
        self.forward = (delta / np.linalg.norm(delta)) if np.linalg.norm(delta) > 1e-8 \
            else np.array([0, 0, -1], dtype=np.float64)

        # 速度
        self.move_speed = 0.3
        self.look_speed = 0.002
        self.zoom_speed = 0.2

        # 点光源位置（IKJL UO 控制）
        self.point_light_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # 模式
        self.mode = 'orbit'  # 'orbit' | 'fps'

        # FPS Euler angles 缓存（避免万向锁）
        self._yaw = math.atan2(self.forward[0], self.forward[2])
        self._pitch = math.asin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))
        self._max_pitch = math.pi / 2 - 0.01

        # Orbit 参数
        self._orbit_radius = max(0.1, np.linalg.norm(self.position - self.target))
        self._orbit_theta = math.atan2(self.forward[0], self.forward[2])
        self._orbit_phi = math.asin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))

    # ── 运动 ────────────────────────────────────────────────────────────────

    def move_forward(self, delta):
        self.position += self.forward * delta * self.move_speed

    def move_right(self, delta):
        right = np.cross(self.forward, self.up_ref)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right /= rn
        self.position += right * delta * self.move_speed

    def move_up(self, delta):
        self.position += self.up_ref * delta * self.move_speed

    def look(self, dx, dy):
        """鼠标拖动旋转视角。"""
        if self.mode == 'orbit':
            self._orbit_theta += dx * self.look_speed
            self._orbit_phi += dy * self.look_speed
            self._orbit_phi = np.clip(self._orbit_phi, -math.pi / 2 + 0.01, math.pi / 2 - 0.01)

            x = self._orbit_radius * math.cos(self._orbit_phi) * math.sin(self._orbit_theta)
            y = self._orbit_radius * math.sin(self._orbit_phi)
            z = self._orbit_radius * math.cos(self._orbit_phi) * math.cos(self._orbit_theta)
            self.position = self.target + np.array([x, y, z])
            self.forward = self.target - self.position
            self.forward /= np.linalg.norm(self.forward)
        else:
            self._yaw += dx * self.look_speed
            self._pitch += dy * self.look_speed
            self._pitch = np.clip(self._pitch, -self._max_pitch, self._max_pitch)

            self.forward = np.array([
                math.sin(self._yaw) * math.cos(self._pitch),
                math.sin(self._pitch),
                math.cos(self._yaw) * math.cos(self._pitch),
            ])
            self.forward /= np.linalg.norm(self.forward)

    def orbit_zoom(self, direction):
        """Orbit 模式滚轮缩放。direction=1 放大, direction=-1 缩小。"""
        if self.mode == 'orbit':
            self._orbit_radius = max(0.1, self._orbit_radius + direction * self.zoom_speed)
            x = self._orbit_radius * math.cos(self._orbit_phi) * math.sin(self._orbit_theta)
            y = self._orbit_radius * math.sin(self._orbit_phi)
            z = self._orbit_radius * math.cos(self._orbit_phi) * math.cos(self._orbit_theta)
            self.position = self.target + np.array([x, y, z])
            self.forward = self.target - self.position
            self.forward /= np.linalg.norm(self.forward)

    def toggle_mode(self):
        self.mode = 'orbit' if self.mode == 'fps' else 'fps'
        print(f"Camera mode: {self.mode.upper()}")
        if self.mode == 'orbit':
            self._orbit_radius = max(0.1, np.linalg.norm(self.position - self.target))
            self._orbit_theta = math.atan2(self.forward[0], self.forward[2])
            self._orbit_phi = math.asin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))
        else:
            self._yaw = math.atan2(self.forward[0], self.forward[2])
            self._pitch = math.asin(np.clip(self.forward[1], -1 + 1e-6, 1 - 1e-6))

    def move_point_light(self, forward_delta=0.0, right_delta=0.0, up_delta=0.0):
        """基于当前相机朝向移动点光源位置。"""
        self.point_light_pos += self.forward * forward_delta * self.move_speed
        right = np.cross(self.forward, self.up_ref)
        rn = np.linalg.norm(right)
        if rn > 1e-8:
            right /= rn
        self.point_light_pos += right * right_delta * self.move_speed
        self.point_light_pos += self.up_ref * up_delta * self.move_speed

    # ── Camera 构造 ─────────────────────────────────────────────────────────

    def build_camera(self) -> Tuple[Camera, torch.Tensor]:
        """从当前状态构造 (Camera 实例, canonical_rays 张量)。

        Camera 的 (R, T) 在 COLMAP 空间构造，与 eval_relighting_*.py 一致。
        canonical_rays 用于 perspective render 的视图方向计算。
        """
        f = self.forward / np.linalg.norm(self.forward)

        # 正交相机基: OpenGL 相机沿 -Z 看
        right = np.cross(f, self.up_ref)
        rn = np.linalg.norm(right)
        if rn < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= rn
        up = np.cross(right, f)  # 重正交化

        # c2w 旋转矩阵: 列 = [right, up, -forward]
        c2w_rot = np.column_stack([right, up, -f]).astype(np.float32)

        # Camera 存储的 (R, T):
        #   R = w2c[:3,:3].T = c2w_rot           (c2w 旋转矩阵)
        #   T = w2c[:3,3]    = -c2w_rot.T @ pos  (w2c 平移)
        R = c2w_rot
        T = (-c2w_rot.T @ self.position).astype(np.float32)

        camera = Camera(
            colmap_id=0, R=R, T=T,
            FoVx=self.FoVx, FoVy=self.FoVy,
            trans=np.array([0.0, 0.0, 0.0]),
            fx=None, fy=None, cx=None, cy=None,
            image=None,
            width=self.width, height=self.height,
            image_name="viewer",
            render_only=True, uid=0,
        )

        # 规范光线 = 相机空间的归一化射线方向 [H*W, 3]
        canonical_rays = _make_canonical_rays(
            self.width, self.height, self.FoVx, self.FoVy
        )
        return camera, canonical_rays

    def build_equirect_camera(self) -> Camera:
        """构造用于 equirect 全景渲染的 Camera（位置不变，FoV=360°）。"""
        R = np.eye(3, dtype=np.float32)
        T = (-self.position).astype(np.float32)

        return Camera(
            colmap_id=0, R=R, T=T,
            FoVx=2 * math.pi, FoVy=math.pi,
            trans=np.array([0.0, 0.0, 0.0]),
            fx=None, fy=None, cx=None, cy=None,
            image=None,
            width=2048, height=1024,
            image_name="equirect",
            render_only=True, uid=0,
        )

    def get_camera_center(self):
        return self.position.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  渲染配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RenderConfig:
    """渲染管线配置，替代之前内联的 type('Pipe', ...)。"""
    compute_with_prt: bool = True
    diffuse_iteration: int = 3000
    compute_cov3D_python: bool = False
    compute_SHs_python: bool = False
    metallic: bool = True
    ref_map: bool = True
    compute_pseudo_normal: bool = False
    relight: bool = True
    tone_mapping: bool = True
    transfer_light: bool = True
    white_background: bool = False
    forward_shading: bool = True
    debug: bool = False

    def to_pipe(self) -> SimpleNamespace:
        """转换为 SimpleNamespace 供 render 函数使用。"""
        return SimpleNamespace(**{
            k: v for k, v in self.__dict__.items()
        })


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _make_canonical_rays(W: int, H: int, FoVx: float, FoVy: float) -> torch.Tensor:
    """生成相机空间的规范光线方向 [H*W, 3]（与 scene.get_canonical_rays 一致）。"""
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * math.tan(FoVx * 0.5))
    focal_y = H / (2.0 * math.tan(FoVy * 0.5))

    x, y = torch.meshgrid(
        torch.arange(W, device='cuda'),
        torch.arange(H, device='cuda'),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    dirs = F.pad(
        torch.stack([
            (x - cen_x + 0.5) / focal_x,
            (y - cen_y + 0.5) / focal_y,
        ], dim=-1),
        (0, 1), value=1.0,
    )  # [H*W, 3]
    return F.normalize(dirs, dim=-1)


def _make_pipe_from_config(cfg: RenderConfig) -> SimpleNamespace:
    """构造 Pipe 对象（兼容旧的 pipe.xxx 访问方式）。"""
    return cfg.to_pipe()


def _build_render_kwargs(gaussians, scene_data, pipe, bg_color, canonical_rays, fast_pbr=True,
                         point_lights=None, point_light_shadow_funcs=None):
    """构造传递给 render_fn 的 kwargs（与 eval_relighting_tensorIR.py 一致）。"""
    return {
        "pc": gaussians,
        "pipe": pipe,
        "bg_color": bg_color,
        "is_training": False,
        "dict_params": {
            "transfer_net": scene_data["transfer_net"],
            "occlusion_volumes": scene_data["occlusion_volumes"],
            "aabb": scene_data["aabb"],
            "cubemap": scene_data["cubemap"],
            "refmap": scene_data["cubemap"],  # refmap = cubemap（重光照模式）
            "brdf_lut": scene_data["brdf_lut"],
            "canonical_rays": canonical_rays,
            "iteration": 999999999,  # 标记为最终迭代（跳过 diffuse-only 阶段）
            "enable_occlusion": scene_data.get("enable_occlusion", True),
            "fast_pbr": fast_pbr,
            "point_lights": point_lights,
            "point_light_shadow_funcs": point_light_shadow_funcs,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  场景数据加载
# ══════════════════════════════════════════════════════════════════════════════

def load_scene_data(checkpoint_path=None, occlusion_path=None, envmap_path=None, ply_path=None):
    """加载高斯模型、传输网络、遮挡体积、环境光照等。

    支持从 checkpoint (.pth) 或 .ply 加载，两者至少提供一个。

    与 eval_relighting_tensorIR.py 一致的结构。
    """
    if ply_path is not None:
        # 先 peek PLY 字段，判断是否含 PBR 属性
        plydata = PlyData.read(ply_path)
        has_pbr = any(p.name.startswith("base_color") for p in plydata.elements[0].properties)
        render_type = 'render_ref_pbr' if has_pbr else 'render_ref'
        print(f"PLY has PBR attributes: {has_pbr} → using render_type={render_type}")

        gaussians = GaussianModel(sh_degree=3, render_type=render_type)
        gaussians.load_ply(ply_path)
        iteration = 0
        for part in ply_path.replace('\\', '/').split('/'):
            if part.startswith('iteration_'):
                try:
                    iteration = int(part.split('_')[1])
                except ValueError:
                    pass
                break
        print(f"Loaded Gaussians from PLY {ply_path}")
    elif checkpoint_path is not None:
        gaussians = GaussianModel(sh_degree=3, render_type='render_ref_pbr')
        iteration = gaussians.create_from_ckpt(checkpoint_path, restore_optimizer=False)
        print(f"Loaded Gaussians from {checkpoint_path} (iteration {iteration})")
    else:
        raise ValueError("Either checkpoint_path or ply_path must be provided")
    gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    # 传输网络（仅 checkpoint 模式有相邻的 transfer_net 文件）
    transfer_net = TransferMLP(
        sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres
    )
    if checkpoint_path is not None:
        tn_ckpt = (os.path.dirname(checkpoint_path) + "/transfer_net_"
                   + os.path.basename(checkpoint_path))
        if os.path.exists(tn_ckpt):
            transfer_net.create_from_ckpt(tn_ckpt)
            print("Loaded transfer net.")
        else:
            print("Warning: transfer net not found, using zeros.")
    else:
        print("PLY mode: transfer net not loaded, using zeros.")

    # 遮挡体积
    occlusion_volumes = torch.load(occlusion_path)
    if "aabb" in occlusion_volumes:
        occ_aabb = occlusion_volumes["aabb"].clone()
        bound = max(occ_aabb[:3].abs().max().item(), occ_aabb[3:].abs().max().item())
    else:
        bound = occlusion_volumes["bound"]
        occ_aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound])
    occ_aabb = occ_aabb.cuda()
    print(f"Loaded occlusion volumes (bound={bound:.2f}).")

    # 环境光照
    cubemap = _load_lighting(checkpoint_path, envmap_path)

    # BRDF LUT
    brdf_lut = get_brdf_lut().cuda()

    return {
        "gaussians": gaussians,
        "iteration": iteration,
        "transfer_net": transfer_net,
        "occlusion_volumes": occlusion_volumes,
        "aabb": occ_aabb,
        "bound": bound,
        "cubemap": cubemap,
        "brdf_lut": brdf_lut,
        "enable_occlusion": True,
    }


def _load_lighting(checkpoint_path=None, envmap_path=None):
    """加载环境光照贴图。

    优先级: 1) 指定 HDR 文件 → 2) 指定 .pth cubemap 文件 → 3) 训练好的 cubemap（仅 checkpoint 模式）→ 4) 报错。
    """
    if envmap_path is not None and os.path.exists(envmap_path):
        if envmap_path.endswith(".pth"):
            cubemap = CubemapLight(base_res=128).cuda()
            cubemap.create_from_ckpt(envmap_path, restore_optimizer=False)
            cubemap.build_mips()
            cubemap.build_sh(degree=3)
            cubemap.eval()
            print(f"Loaded cubemap from {envmap_path}")
            return cubemap
        else:
            hdri = read_hdr(envmap_path)
            hdri_t = torch.from_numpy(hdri).cuda()
            cubemap = CubemapLight(base_res=256).cuda()
            cubemap.base.data = latlong_to_cubemap(hdri_t, [256, 256])
            cubemap.build_mips()
            cubemap.build_sh(degree=3)  # 预计算 SH 系数供 render.py 使用
            cubemap.eval()
            print(f"Loaded envmap from {envmap_path}")
            return cubemap

    if checkpoint_path is not None:
        cubemap_ckpt = (os.path.dirname(checkpoint_path) + "/cubemap_"
                        + os.path.basename(checkpoint_path))
        if os.path.exists(cubemap_ckpt):
            cubemap = CubemapLight(base_res=128).cuda()
            cubemap.create_from_ckpt(cubemap_ckpt, restore_optimizer=False)
            cubemap.build_mips()
            cubemap.eval()
            print(f"Loaded trained cubemap from {cubemap_ckpt}")
            return cubemap

    raise FileNotFoundError(
        "No envmap given. Use --envmap_path to specify an HDR file or .pth cubemap."
        if checkpoint_path is None else
        f"No envmap path given and trained cubemap not found at {cubemap_ckpt}. "
        "Use --envmap_path to specify an HDR file or .pth cubemap."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  渲染函数
# ══════════════════════════════════════════════════════════════════════════════

def render_perspective(cam_ctrl: CameraController, scene_data: dict,
                       cfg: RenderConfig, render_mode: str,
                       fast_pbr: bool = True,
                       point_lights=None, point_light_shadow_funcs=None,
                       ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """渲染当前视角的透视画面。

    Returns:
        (display_image, env_bg, opacity) 均为 [H, W, 3] uint8 numpy,
        env_bg / opacity 可为 None（当 render_pkg 中不存在时）。
    """
    camera, canonical_rays = cam_ctrl.build_camera()
    pipe = _make_pipe_from_config(cfg)
    bg = 1.0 if cfg.white_background else 0.0
    bg_color = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")

    render_kwargs = _build_render_kwargs(
        scene_data["gaussians"], scene_data, pipe, bg_color, canonical_rays,
        fast_pbr=fast_pbr,
        point_lights=point_lights,
        point_light_shadow_funcs=point_light_shadow_funcs,
    )

    render_fn = render_fn_dict["render_ref_pbr"]
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_camera=camera, **render_kwargs)

    image_np = _extract_display_image(render_pkg, render_mode)

    # 提取 env_only / opacity 用于 show_env_bg 合成
    env_bg_np = None
    opacity_np = None
    if "env_only" in render_pkg and "opacity" in render_pkg:
        env = render_pkg["env_only"]
        env_bg_np = env.detach().permute(1, 2, 0).cpu().numpy()
        env_bg_np = np.clip(env_bg_np, 0.0, 1.0)
        opacity = render_pkg["opacity"].detach().cpu().numpy()
        if opacity.ndim == 3 and opacity.shape[0] == 1:
            opacity = opacity[0]
        opacity = np.clip(opacity, 0.0, 1.0)
        opacity_np = np.repeat(opacity[..., None], 3, axis=-1)

    return image_np, env_bg_np, opacity_np


def render_equirect_crop(cam_ctrl: CameraController, scene_data: dict,
                         cfg: RenderConfig, render_mode: str,
                         equirect_width: int = 2048,
                         env_angle_y: float = 0.0, env_angle_x: float = 0.0,
                         fast_pbr: bool = True,
                         point_lights=None, point_light_shadow_funcs=None,
                         point_light_pos=None) -> np.ndarray:
    """渲染当前位置 equirect 全景图 → 裁切为透视视口。

    位置未显著变化时复用缓存的 equirect 以节省时间。
    """
    global _EQUIRECT_CACHE
    cam_pos = cam_ctrl.get_camera_center().astype(np.float64)

    eq_w = min(equirect_width, 4096)
    eq_h = eq_w // 2

    # 检测是否需要重新渲染
    cache = _EQUIRECT_CACHE

    # 点光源位置变化也触发重新渲染
    cached_pl = cache.get("point_light_pos")
    pl_changed = (
        (point_light_pos is not None and cached_pl is None)
        or (point_light_pos is not None and cached_pl is not None
            and np.linalg.norm(point_light_pos - cached_pl) > 0.01)
        or (point_light_pos is None and cached_pl is not None)
    )

    pos_changed = (
        cache["pos"] is None
        or np.linalg.norm(cam_pos - cache["pos"]) > 0.01
        or cache["eq_w"] != eq_w
        or cache.get("render_mode") != render_mode
        or cache.get("env_angle_y") != env_angle_y
        or cache.get("env_angle_x") != env_angle_x
        or pl_changed
    )

    if pos_changed:
        cam = cam_ctrl.build_equirect_camera()
        # 覆写宽高到当前 equirect 分辨率
        cam.image_width = eq_w
        cam.image_height = eq_h

        pipe = _make_pipe_from_config(cfg)
        bg = 1.0 if cfg.white_background else 0.0
        bg_color = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")

        # equirect render 不需要 canonical_rays
        render_kwargs = _build_render_kwargs(
            scene_data["gaussians"], scene_data, pipe, bg_color, None,
            fast_pbr=fast_pbr,
            point_lights=point_lights,
            point_light_shadow_funcs=point_light_shadow_funcs,
        )

        render_fn = render_fn_dict["render_ref_pbr_equirect"]
        with torch.no_grad():
            render_pkg = render_fn(viewpoint_camera=cam, **render_kwargs)

        equirect_img = _extract_raw_tensor(render_pkg, render_mode)

        # 更新缓存
        _EQUIRECT_CACHE = {
            "pos": cam_pos.copy(),
            "eq_w": eq_w,
            "render_mode": render_mode,
            "env_angle_y": env_angle_y,
            "env_angle_x": env_angle_x,
            "point_light_pos": point_light_pos.copy() if point_light_pos is not None else None,
            "image": equirect_img,
        }
    else:
        equirect_img = cache["image"]

    # 裁切透视视口
    with torch.no_grad():
        perspective_img = _equirect_to_perspective(
            equirect_img,
            cam_ctrl.forward, cam_ctrl.up_ref,
            cam_ctrl.FoVx, cam_ctrl.width, cam_ctrl.height,
        )

    img_np = perspective_img.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    return (img_np * 255).astype(np.uint8)


# Equirect 缓存（模块级）
_EQUIRECT_CACHE = {
    "pos": None,
    "eq_w": None,
    "render_mode": None,
    "image": None,
}

# 点光源阴影深度图缓存（模块级）
_POINT_LIGHT_SHADOW_CACHE = {"pos": None, "func": None}


def _get_point_light_shadow(scene_data: dict, light_pos_np: np.ndarray,
                            render_backend: str):
    """获取或更新点光源阴影深度图。

    光源位置变化超过阈值（0.05）时重新渲染深度图，否则复用缓存。
    """
    cache = _POINT_LIGHT_SHADOW_CACHE
    if (cache["pos"] is not None
            and np.linalg.norm(light_pos_np - cache["pos"]) < 0.05):
        return cache["func"]

    light_pos_t = torch.from_numpy(light_pos_np.astype(np.float32)).cuda()
    gaussians = scene_data["gaussians"]
    with torch.no_grad():
        if render_backend == "equirect":
            depth_map = get_depth_equirect(gaussians, light_pos_t)
            shadow_fn = make_shadow_func_equirect(depth_map, light_pos_t)
        else:
            depth_map = get_depth_cubemap(gaussians, light_pos_t)
            shadow_fn = make_shadow_func_cubemap(depth_map, light_pos_t)
    cache["pos"] = light_pos_np.copy()
    cache["func"] = shadow_fn
    return shadow_fn


def _extract_raw_tensor(render_pkg: dict, render_mode: str) -> torch.Tensor:
    """从 render_pkg 提取指定模式的原始图像张量 [3, H, W]。"""
    if render_mode == "pbr":
        img = render_pkg.get("pbr", render_pkg.get("render"))
    elif render_mode == "render":
        img = render_pkg.get("render")
    elif "vis_dict" in render_pkg and render_mode in render_pkg["vis_dict"]:
        img = render_pkg["vis_dict"][render_mode]
    else:
        img = render_pkg.get(render_mode, render_pkg.get("pbr", render_pkg.get("render")))

    if img is None:
        raise KeyError(f"Render mode '{render_mode}' not found in output.")

    # 单通道 → 3 通道
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    return img


def _extract_display_image(render_pkg: dict, render_mode: str) -> np.ndarray:
    """从 render_pkg 提取并转换为 uint8 numpy 显示图像。"""
    img = _extract_raw_tensor(render_pkg, render_mode)
    img_np = img.detach().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0.0, 1.0)
    return (img_np * 255).astype(np.uint8)


# ── Equirect → 透视裁切（复用了 nvdiffrast 风格的空间变换） ──────────────

def _equirect_to_perspective(equirect_img, forward, up_ref,
                              fovx_rad, target_w, target_h):
    """从等距柱状投影中提取透视视口。"""
    device = equirect_img.device
    H, W = target_h, target_w

    aspect = W / H
    fovy_rad = 2 * math.atan(math.tan(fovx_rad * 0.5) / aspect)
    tan_hfovx = math.tan(fovx_rad * 0.5)
    tan_hfovy = math.tan(fovy_rad * 0.5)

    xs = torch.linspace(-tan_hfovx, tan_hfovx, W, device=device)
    ys = torch.linspace(tan_hfovy, -tan_hfovy, H, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    dirs_cam = F.normalize(torch.stack([gx, gy, torch.ones_like(gx)], dim=-1), dim=-1)

    forward_n = forward / np.linalg.norm(forward)
    right = np.cross(forward_n, up_ref)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-8 else np.array([1.0, 0.0, 0.0])
    cam_up = np.cross(right, forward_n)
    cam_up /= np.linalg.norm(cam_up)

    R_world = np.float32([right, cam_up, forward_n])  # [3, 3]
    R_world_t = torch.from_numpy(R_world).to(device=device)

    dirs_world = (dirs_cam.reshape(-1, 3) @ R_world_t.T).reshape(H, W, 3)

    lon = torch.atan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = torch.asin(torch.clamp(dirs_world[..., 1], -1.0, 1.0))

    grid_x = lon / math.pi
    grid_y = -2 * lat / math.pi
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    perspective = F.grid_sample(
        equirect_img.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=False
    )
    return perspective[0]


# ── 纯环境光天空盒（快速 cubemap 采样，不渲染场景） ──────────────────

def render_env_skybox(cubemap, forward, up_ref, fovx_rad, width, height):
    """直接采样 cubemap 渲染天空盒，不经过高斯渲染管线。"""
    import nvdiffrast.torch as dr
    device = cubemap.base.device
    H, W = height, width
    aspect = W / H
    fovy_rad = 2 * math.atan(math.tan(fovx_rad * 0.5) / aspect)
    tan_hfovx = math.tan(fovx_rad * 0.5)
    tan_hfovy = math.tan(fovy_rad * 0.5)

    xs = torch.linspace(-tan_hfovx, tan_hfovx, W, device=device)
    ys = torch.linspace(tan_hfovy, -tan_hfovy, H, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    dirs_cam = F.normalize(torch.stack([gx, gy, torch.ones_like(gx)], dim=-1), dim=-1)

    forward_n = forward / np.linalg.norm(forward)
    right = np.cross(forward_n, up_ref)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-8 else np.array([1.0, 0.0, 0.0])
    cam_up = np.cross(right, forward_n)
    cam_up /= np.linalg.norm(cam_up)

    R_world = np.float32([right, cam_up, forward_n])
    R_world_t = torch.from_numpy(R_world).to(device=device)

    dirs_world = (dirs_cam.reshape(-1, 3) @ R_world_t.T).reshape(1, H, W, 3)

    if cubemap.mtx is not None:
        dirs = cubemap.rotate_dirs(dirs_world)
    else:
        dirs = dirs_world

    sky = dr.texture(
        cubemap.base[None, ...],
        dirs.contiguous(),
        filter_mode="linear",
        boundary_mode="cube",
    )[0]
    return sky.permute(2, 0, 1)


# ── 环境光旋转 ─────────────────────────────────────────────────────

def rotate_envmap(cubemap, angle_y: float, angle_x: float = 0.0):
    """旋转环境光照（绕 Y 轴和 X 轴）。"""
    cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
    cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)

    rot_x = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x],
    ], dtype=torch.float32, device=cubemap.base.device)

    rot_y = torch.tensor([
        [cos_y, 0.0, -sin_y],
        [0.0, 1.0, 0.0],
        [sin_y, 0.0, cos_y],
    ], dtype=torch.float32, device=cubemap.base.device)

    cubemap.xfm(rot_y @ rot_x)


# ══════════════════════════════════════════════════════════════════════════════
#  快照保存
# ══════════════════════════════════════════════════════════════════════════════

def save_snapshot(cam_ctrl: CameraController, scene_data: dict,
                  cfg: RenderConfig, output_dir="snapshots"):
    """保存高分辨率 equirect + cubemap 六面 + 当前透视视角。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 根据当前 FOV 计算适配的 equirect 分辨率
    fov_fraction = cam_ctrl.FoVx / (2 * math.pi)
    eq_w = min(max(int(cam_ctrl.width / fov_fraction), 2048), 4096)
    eq_h = eq_w // 2

    eq_cam = cam_ctrl.build_equirect_camera()
    eq_cam.image_width = eq_w
    eq_cam.image_height = eq_h

    pipe = _make_pipe_from_config(cfg)
    bg = 1.0 if cfg.white_background else 0.0
    bg_color = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")

    render_kwargs = _build_render_kwargs(
        scene_data["gaussians"], scene_data, pipe, bg_color, None,
        fast_pbr=True,
    )

    render_fn = render_fn_dict["render_ref_pbr_equirect"]
    with torch.no_grad():
        render_pkg = render_fn(viewpoint_camera=eq_cam, **render_kwargs)
    equirect_img = _extract_raw_tensor(render_pkg, "pbr")

    snap_dir = os.path.join(output_dir, f"snapshot_{ts}")
    os.makedirs(snap_dir, exist_ok=True)

    # 保存全景图
    save_image(equirect_img, os.path.join(snap_dir, "equirect.png"))

    # 保存 cubemap 六面
    face_names = ["posx", "negx", "posy", "negy", "posz", "negz"]
    cubemap_faces = latlong_to_cubemap_equirect(
        equirect_img.permute(1, 2, 0), [512, 512]
    )
    c_dir = os.path.join(snap_dir, "cubemap")
    os.makedirs(c_dir, exist_ok=True)
    for fi, name in enumerate(face_names):
        save_image(cubemap_faces[fi].permute(2, 0, 1), os.path.join(c_dir, f"{name}.png"))

    # 保存当前透视视口
    persp = _equirect_to_perspective(
        equirect_img, cam_ctrl.forward, cam_ctrl.up_ref,
        cam_ctrl.FoVx, cam_ctrl.width, cam_ctrl.height,
    )
    save_image(persp, os.path.join(snap_dir, "perspective.png"))

    print(f"Snapshot saved to {snap_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  可视化模式
# ══════════════════════════════════════════════════════════════════════════════

RENDER_MODES: List[Tuple[str, str]] = [
    ("pbr", "PBR"),
    ("render", "Render"),
    ("base_color", "Base Color"),
    ("normal", "Normal"),
    ("pseudo_normal", "Pseudo Normal"),
    ("roughness", "Roughness"),
    ("metallic", "Metallic"),
    ("depth", "Depth"),
    ("visibility", "Occlusion"),
    ("diffuse_pbr", "Diffuse"),
    ("specular_pbr", "Specular"),
    ("incidents_light", "Incident Light"),
    ("point_light", "Point Light"),
]

RENDER_MODE_NAMES = {k: v for k, v in RENDER_MODES}


# ══════════════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RTR-GS Clean Viewer")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Path to checkpoint (.pth)")
    parser.add_argument("--ply_path", type=str, default=None,
                        help="Path to .ply file (alternative to --checkpoint)")
    parser.add_argument("--occlusion_path", type=str, required=True)
    parser.add_argument("--envmap_path", type=str, default=None)
    parser.add_argument("-s", "--source_path", type=str, default=None,
                        help="场景源目录（用于读取 FOV 和相机位置）")
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--equirect", action="store_true", default=False,
                        help="使用 equirect→裁切模式")
    parser.add_argument("--equirect_width", type=int, default=2048,
                        help="Equirect 全景分辨率")
    parser.add_argument("--white_background", action="store_true", default=None)
    args = parser.parse_args()

    # ── 自动检测 white_background ────────────────────────────────────
    white_bg = False
    if args.white_background is None and args.source_path is not None:
        for tf_file in ["transforms_train.json", "transforms_test.json"]:
            tf_path = os.path.join(args.source_path, tf_file)
            if os.path.exists(tf_path):
                try:
                    with open(tf_path) as f:
                        import json
                        tf_data = json.load(f)
                    if "white_bg" in tf_data:
                        white_bg = bool(tf_data["white_bg"])
                        break
                except Exception:
                    pass
    elif args.white_background is not None:
        white_bg = args.white_background

    # ── 加载场景数据 ─────────────────────────────────────────────────
    print("Loading scene data...")
    if not args.checkpoint and not args.ply_path:
        parser.error("Either --checkpoint or --ply_path must be provided")
    scene_data = load_scene_data(args.checkpoint, args.occlusion_path, args.envmap_path, args.ply_path)
    gaussians = scene_data["gaussians"]

    # 场景边界
    pos_all = gaussians.get_xyz.detach()
    scene_center = pos_all.mean(dim=0).cpu().numpy()
    scene_min = pos_all.min(dim=0).values.cpu().numpy()
    scene_max = pos_all.max(dim=0).values.cpu().numpy()
    scene_radius = max(np.linalg.norm(scene_max - scene_min) / 2.0, 0.1)
    print(f"Scene center: {scene_center}, radius: {scene_radius:.2f}")

    # ── 从场景读取 FOV ──────────────────────────────────────────────
    is_equirect_dataset = False
    test_cams = None
    if args.source_path is not None:
        from scene import Scene
        scene = Scene(args, read_cam_only=True, shuffle=False)
        test_cams = scene.getTestCameras()
        is_equirect_dataset = os.path.exists(
            os.path.join(args.source_path, "data_extrinsics.json")
        )

    # ---- FOV 推导链路 ----
    # 固定垂直 FOV（默认 60°），水平 FOV 按视角宽高比自动推导：
    #   fovx = 2 * atan(tan(fovy/2) * width/height)
    fovy = 1.047197551  # 60° in radians
    if test_cams and len(test_cams) > 0 and not is_equirect_dataset:
        # 透视数据集：从第一帧相机取垂直 FOV
        cam_fovx = test_cams[0].FoVx
        cam_fovy = test_cams[0].FoVy
        cam_aspect = math.tan(cam_fovx / 2) / math.tan(cam_fovy / 2)
        win_aspect = args.image_width / max(args.image_height, 1)
        if abs(cam_aspect / win_aspect - 1.0) < 0.05:
            fovy = cam_fovy
        print(f"Scene camera FOV: {math.degrees(cam_fovx):.1f}°×{math.degrees(cam_fovy):.1f}°, "
              f"using fovy={math.degrees(fovy):.1f}°")
    else:
        print(f"Default FOV: fovy={math.degrees(fovy):.1f}°")

    # 水平 FOV = 2 * atan(tan(fovy/2) * aspect) — 保持垂直固定
    aspect = args.image_width / max(args.image_height, 1)
    fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
    print(f"Viewport FOV: {math.degrees(fovx):.1f}°×{math.degrees(fovy):.1f}° (aspect={aspect:.3f})")

    # ── 初始化相机 ─────────────────────────────────────────────────
    if test_cams and len(test_cams) > 0:
        c0 = test_cams[0]
        init_pos = -c0.R @ c0.T
        init_forward = (c0.R @ np.array([0, 0, -1], dtype=np.float64)).astype(np.float64)
        init_target = scene_center.copy()
        init_up = c0.R[:, 1].astype(np.float64)
        init_up /= np.linalg.norm(init_up)
        print(f"Initial position from first camera: {init_pos}, up: {init_up}")
    else:
        init_pos = scene_center + np.array([0, 0, scene_radius * 2.5])
        init_target = scene_center
        init_up = _DEFAULT_UP

    render_mode = "pbr"

    cam_ctrl = CameraController(
        position=init_pos, target=init_target, up_ref=init_up,
        FoVx=fovx, FoVy=fovy,
        width=args.image_width, height=args.image_height,
        render_mode=render_mode,
    )

    # ── 渲染配置 ────────────────────────────────────────────────────
    render_cfg = RenderConfig(white_background=white_bg)

    # ── 状态 ─────────────────────────────────────────────────────────
    render_backend = "equirect" if args.equirect else "perspective"
    env_angle_y = 0.0
    env_angle_x = 0.0
    show_env_bg = False
    show_env_only = False
    playing_transforms = False
    play_index = 0

    # ── 点光源 ───────────────────────────────────────────────────────
    point_light = PointLight(
        position=torch.zeros(3, dtype=torch.float32, device="cuda"),
        color=torch.ones(3, dtype=torch.float32, device="cuda"),
        intensity=20.0,
    )
    point_light_enabled = True

    if render_backend == "equirect":
        print("Backend: EQUIRECT (full panorama → perspective crop)")
    else:
        print("Backend: PERSPECTIVE (direct rendering)")

    # ── Pygame 初始化 ────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((args.image_width, args.image_height))
    pygame.display.set_caption("RTR-GS Viewer (Clean)")
    clock = pygame.time.Clock()
    pygame.mouse.set_visible(True)

    right_mouse_down = False
    last_mouse_pos = pygame.mouse.get_pos()

    # ── 主循环 ─────────────────────────────────────────────────────
    running = True
    while running:
        # ── 事件处理 ─────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_m:
                    cam_ctrl.toggle_mode()

                elif event.key == pygame.K_v:
                    keys_list = [k for k, _ in RENDER_MODES]
                    idx = keys_list.index(render_mode) if render_mode in keys_list else 0
                    render_mode = keys_list[(idx + 1) % len(keys_list)]
                    print(f"View: {RENDER_MODE_NAMES.get(render_mode, render_mode.upper())}")

                elif event.key == pygame.K_n:
                    render_backend = "equirect" if render_backend == "perspective" else "perspective"
                    print(f"Backend: {render_backend.upper()}")
                    _EQUIRECT_CACHE["pos"] = None  # 清除缓存
                    _POINT_LIGHT_SHADOW_CACHE["pos"] = None  # 阴影深度图也需重建

                elif event.key == pygame.K_x:
                    scene_data["enable_occlusion"] = not scene_data["enable_occlusion"]
                    print(f"Occlusion: {'ON' if scene_data['enable_occlusion'] else 'OFF'}")

                elif event.key == pygame.K_b:
                    show_env_bg = not show_env_bg
                    print(f"Env BG: {'ON' if show_env_bg else 'OFF'}")

                elif event.key == pygame.K_h:
                    show_env_only = not show_env_only
                    print(f"Env Only: {'ON' if show_env_only else 'OFF'}")

                elif event.key == pygame.K_r:
                    env_angle_y = 0.0
                    env_angle_x = 0.0
                    rotate_envmap(scene_data["cubemap"], 0.0, 0.0)

                elif event.key == pygame.K_p:
                    if test_cams is None or len(test_cams) == 0:
                        print("No transforms for playback.")
                    else:
                        playing_transforms = not playing_transforms
                        print(f"Playback: {'ON' if playing_transforms else 'OFF'}")
                        if not playing_transforms:
                            # 从当前 R/T 恢复 position/forward/up
                            fp = test_cams[play_index % len(test_cams)]
                            cam_ctrl.position = (-fp.R @ fp.T).astype(np.float64)
                            cam_ctrl.forward = (fp.R @ np.array([0, 0, -1], dtype=np.float64)).astype(np.float64)
                            cam_ctrl.forward /= np.linalg.norm(cam_ctrl.forward)
                            cam_ctrl.up_ref = fp.R[:, 1].astype(np.float64)
                            cam_ctrl.up_ref /= np.linalg.norm(cam_ctrl.up_ref)

                elif event.key == pygame.K_k:
                    print("Saving snapshot...")
                    save_snapshot(cam_ctrl, scene_data, render_cfg)

                elif event.key == pygame.K_SPACE:
                    if not playing_transforms:
                        cam_ctrl.forward = np.array([0, 0, -1], dtype=np.float64)

                elif event.key == pygame.K_t:
                    point_light_enabled = not point_light_enabled
                    print(f"Point Light: {'ON' if point_light_enabled else 'OFF'}")

                elif event.key == pygame.K_g:
                    # 重置点光源位置到原点
                    cam_ctrl.point_light_pos[:] = 0.0
                    _POINT_LIGHT_SHADOW_CACHE["pos"] = None
                    print("Point light reset to origin.")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # 右键
                    right_mouse_down = True
                    last_mouse_pos = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    right_mouse_down = False

            elif event.type == pygame.MOUSEWHEEL:
                cam_ctrl.orbit_zoom(event.y)

        # ── 持续按键 ─────────────────────────────────────────────────
        keys = pygame.key.get_pressed()

        if playing_transforms and test_cams is not None:
            ref_cam = test_cams[play_index % len(test_cams)]
            # 从 ref_cam.R/T 恢复 c2w 位置、朝向和上方向
            cam_ctrl.position = (-ref_cam.R @ ref_cam.T).astype(np.float64)
            cam_ctrl.forward = (ref_cam.R @ np.array([0, 0, -1], dtype=np.float64)).astype(np.float64)
            cam_ctrl.forward /= np.linalg.norm(cam_ctrl.forward)
            cam_ctrl.up_ref = ref_cam.R[:, 1].astype(np.float64)
            cam_ctrl.up_ref /= np.linalg.norm(cam_ctrl.up_ref)
            play_index += 1
        else:
            if cam_ctrl.mode == 'fps':
                if keys[pygame.K_w]:
                    cam_ctrl.move_forward(1.0)
                if keys[pygame.K_s]:
                    cam_ctrl.move_forward(-1.0)
                if keys[pygame.K_a]:
                    cam_ctrl.move_right(-1.0)
                if keys[pygame.K_d]:
                    cam_ctrl.move_right(1.0)
                if keys[pygame.K_q]:
                    cam_ctrl.move_up(-1.0)
                if keys[pygame.K_e]:
                    cam_ctrl.move_up(1.0)

            # 点光源移动（IKJL UO，基于相机朝向）
            if point_light_enabled:
                if keys[pygame.K_i]:
                    cam_ctrl.move_point_light(forward_delta=1.0)
                if keys[pygame.K_k]:
                    cam_ctrl.move_point_light(forward_delta=-1.0)
                if keys[pygame.K_j]:
                    cam_ctrl.move_point_light(right_delta=-1.0)
                if keys[pygame.K_l]:
                    cam_ctrl.move_point_light(right_delta=1.0)
                if keys[pygame.K_u]:
                    cam_ctrl.move_point_light(up_delta=-1.0)
                if keys[pygame.K_o]:
                    cam_ctrl.move_point_light(up_delta=1.0)

            # 环境光旋转
            if keys[pygame.K_LEFT]:
                env_angle_y -= 0.05
                rotate_envmap(scene_data["cubemap"], env_angle_y, env_angle_x)
            if keys[pygame.K_RIGHT]:
                env_angle_y += 0.05
                rotate_envmap(scene_data["cubemap"], env_angle_y, env_angle_x)
            if keys[pygame.K_UP]:
                env_angle_x += 0.05
                rotate_envmap(scene_data["cubemap"], env_angle_y, env_angle_x)
            if keys[pygame.K_DOWN]:
                env_angle_x -= 0.05
                rotate_envmap(scene_data["cubemap"], env_angle_y, env_angle_x)

            # 鼠标右键拖动
            if right_mouse_down:
                cur = pygame.mouse.get_pos()
                dx = cur[0] - last_mouse_pos[0]
                dy = cur[1] - last_mouse_pos[1]
                if dx != 0 or dy != 0:
                    cam_ctrl.look(dx, dy)
                last_mouse_pos = cur

        # ── 渲染 ───────────────────────────────────────────────────
        # 更新点光源位置和阴影
        if point_light_enabled:
            pl_pos = cam_ctrl.point_light_pos
            point_light.position = torch.from_numpy(pl_pos.astype(np.float32)).cuda()
            shadow_fn = _get_point_light_shadow(scene_data, pl_pos, render_backend)
            active_lights = [point_light]
            active_shadow = [shadow_fn]
        else:
            pl_pos = None
            active_lights = None
            active_shadow = None

        env_bg_np = None
        opacity_np = None
        if show_env_only:
            with torch.no_grad():
                sky = render_env_skybox(
                    scene_data["cubemap"], cam_ctrl.forward, cam_ctrl.up_ref,
                    cam_ctrl.FoVx, cam_ctrl.width, cam_ctrl.height,
                )
            display_np = sky.permute(1, 2, 0).cpu().numpy()
            display_np = np.clip(display_np, 0.0, 1.0)
            display_np = (display_np * 255).astype(np.uint8)
        else:
            # 纯 pbr 模式用 fast_pbr 加速（跳过 PRT 混合渲染+反射）；
            # render/vis_dict 模式需要完整渲染
            fast_pbr = (render_mode == "pbr")
            if render_backend == "equirect":
                display_np = render_equirect_crop(
                    cam_ctrl, scene_data, render_cfg, render_mode,
                    equirect_width=args.equirect_width,
                    env_angle_y=env_angle_y, env_angle_x=env_angle_x,
                    fast_pbr=fast_pbr,
                    point_lights=active_lights,
                    point_light_shadow_funcs=active_shadow,
                    point_light_pos=pl_pos,
                )
            else:
                display_np, env_bg_np, opacity_np = render_perspective(
                    cam_ctrl, scene_data, render_cfg, render_mode,
                    fast_pbr=fast_pbr,
                    point_lights=active_lights,
                    point_light_shadow_funcs=active_shadow,
                )

        # ── Env BG 合成 ────────────────────────────────────────────
        # 将白色背景替换为环境光照背景: display = pbr*α + env*(1-α)
        if show_env_bg and env_bg_np is not None and opacity_np is not None:
            pbr_f = display_np.astype(np.float32) / 255.0
            env_f = env_bg_np.astype(np.float32) / 255.0
            display_f = pbr_f + (env_f - 1.0) * (1.0 - opacity_np)
            display_np = (np.clip(display_f, 0.0, 1.0) * 255).astype(np.uint8)

        # ── 显示到屏幕 ──────────────────────────────────────────────
        surf = pygame.surfarray.make_surface(np.transpose(display_np, (1, 0, 2)))
        # surf = pygame.transform.flip(surf, False, True)  # Y-flip (temp fix for upside-down)
        screen.blit(surf, (0, 0))

        # HUD
        font = pygame.font.SysFont('Arial', 20)
        fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (0, 255, 0))
        mode_text = font.render(f"Mode: {cam_ctrl.mode.upper()}", True, (0, 255, 0))
        backend_text = font.render(f"Backend: {render_backend.upper()} [N]", True, (0, 255, 255))
        view_text = font.render(f"View: {RENDER_MODE_NAMES.get(render_mode, render_mode.upper())} [V]", True, (0, 255, 0))
        occ_text = font.render(f"Occlusion: {'ON' if scene_data['enable_occlusion'] else 'OFF'} [X]", True, (0, 255, 0))
        env_bg_text = font.render(f"Env BG: {'ON' if show_env_bg else 'OFF'} [B]", True, (0, 255, 0))
        env_only_text = font.render(f"Env Only: {'ON' if show_env_only else 'OFF'} [H]", True, (255, 255, 0))
        env_rot_text = font.render(f"Env Rot: {env_angle_y * 180 / math.pi:.1f}° [←→]", True, (0, 255, 0))
        pl_pos = cam_ctrl.point_light_pos
        pl_text = font.render(
            f"Point Light: {'ON' if point_light_enabled else 'OFF'} [T] "
            f"({pl_pos[0]:.2f},{pl_pos[1]:.2f},{pl_pos[2]:.2f}) IKJL UO [G=reset]",
            True, (255, 200, 0))

        screen.blit(fps_text, (10, 10))
        screen.blit(mode_text, (10, 35))
        screen.blit(backend_text, (10, 60))
        screen.blit(view_text, (10, 85))
        screen.blit(occ_text, (10, 110))
        screen.blit(env_bg_text, (10, 135))
        screen.blit(env_only_text, (10, 160))
        screen.blit(env_rot_text, (10, 185))
        screen.blit(pl_text, (10, 210))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("Viewer closed.")


if __name__ == "__main__":
    main()
