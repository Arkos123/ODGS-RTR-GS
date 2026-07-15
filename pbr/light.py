from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

from arguments import OptimizationParams
from utils.sh_utils import eval_sh_basis

from .renderutils import diffuse_cubemap, specular_cubemap


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap: torch.Tensor) -> torch.Tensor:
        # avg_pool_nhwc
        y = cubemap.permute(0, 3, 1, 2)  # NHWC -> NCHW
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> torch.Tensor:
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                indexing="ij",
            )
            v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )
        return out


class CubemapLight(nn.Module):
    # for nvdiffrec
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(
        self,
        base_res: int = 512,
        scale: float = 0.5,
        bias: float = 0.25,
    ) -> None:
        super(CubemapLight, self).__init__()
        self.mtx = None
        base = (
            torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale + bias
        )
        # base = (
        #     torch.ones(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale + bias
        # )
        self.base = nn.Parameter(base)
        self.register_parameter("env_base", self.base)
        self.envmap_dirs = self.get_envmap_dirs()
        self.sh_dirs = self.get_sh_dirs()

    def training_setup(self, training_args: OptimizationParams, light_type = "env"):
        assert light_type in ["env", "ref"]
        if light_type == "env":
            lr = training_args.env_lr
        elif light_type == "ref":
            lr = training_args.ref_lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def step(self):
        self.base.grad *= 64
        self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.clamp_(min=0.0)

    def xfm(self, mtx) -> None:
        self.mtx = mtx

    def rotate_dirs(self, directions: torch.Tensor) -> torch.Tensor:
        if self.mtx is None:
            return directions
        orig_shape = directions.shape
        flat = directions.reshape(-1, 3)
        if not isinstance(self.mtx, torch.Tensor):
            mtx_t = torch.tensor(self.mtx, dtype=torch.float32, device=directions.device)
        else:
            mtx_t = self.mtx.to(dtype=torch.float32, device=directions.device)
        rotated = (mtx_t @ flat.T).T
        return rotated.reshape(orig_shape)

    def clamp_(self, min: Optional[float]=None, max: Optional[float]=None) -> None:
        self.base.clamp_(min, max)

    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS)
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / (1.0 - self.MAX_ROUGHNESS)
            + len(self.specular)
            - 2,
        )

    def build_sh(self, degree: int = 3):
        res = [512, 1024]
        lat_step_size = np.pi / res[0]
        lng_step_size = 2 * np.pi / res[1]
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, res[0],device="cuda"), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, res[1],device="cuda"  )], indexing='ij')


        reflvec = torch.stack([  torch.cos(theta) * torch.cos(phi), 
                                torch.sin(theta) * torch.cos(phi), 
                                torch.sin(phi)], dim=-1).view(res[0], res[1], 3)    # [envH, envW, 3]
        
        color = dr.texture(
            self.base[None, ...],
            reflvec[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        self.shs = eval_sh_basis(degree, reflvec, color).permute(0, 2, 1)

        return color


    
    
    def build_mips(self, cutoff: float = 0.99) -> None:
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (
                self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
            ) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

    def export_envmap(
        self,
        filename: Optional[str] = None,
        res: List[int] = [512, 1024],
        return_img: bool = False,
        base: bool = True,
    ) -> Optional[torch.Tensor]:
        """将 cubemap 环境贴图导出为等距柱状投影图（equirectangular panorama），便于可视化。

        cubemap（6 张方形图）存储在 self.base 或 self.diffuse 中，人眼不便直接查看。
        此函数在经纬度网格上采样 cubemap，生成一张可以直观看的 ERP 图像。

        Args:
            filename: 保存路径（路径需存在）。None 且 return_img=False 时不保存也不返回。
            res: ERP 分辨率 [H, W]，默认 [512, 1024]。
            return_img: True 则返回 tensor，False 则保存到文件。
            base: True 时采样 self.base（完整环境贴图，含高光细节），
                  False 时采样 self.diffuse（预滤波漫反射，仅低频）。
        Returns:
            return_img=True 时返回 color tensor [H, W, 3]，否则 None。
        """
        # 构建等距柱状投影的经纬度网格
        # lat（纬度）范围 [+π/2, -π/2]（从上到下），lng（经度）范围 [+π, -π]（从左到右）
        lat_step_size = np.pi / res[0]
        lng_step_size = 2 * np.pi / res[1]
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, res[0], device="cuda"),
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, res[1], device="cuda")], indexing='ij')

        # 将经纬度转换为世界空间中的方向向量（reflvec 坐标系：+X 右, +Y 上, +Z 后）
        # 对于 nvdiffrast cubemap 采样，boundary_mode="cube" 会将方向向量映射到 6 个面
        reflvec = torch.stack([
            torch.cos(theta) * torch.cos(phi),   # X: 右
            torch.sin(theta) * torch.cos(phi),   # Y: 上
            torch.sin(phi)                       # Z: 后
        ], dim=-1).view(res[0], res[1], 3)        # [envH, envW, 3]

        # 如果设置了环境旋转矩阵（self.mtx），先将方向旋转后再采样
        sample_dirs = self.rotate_dirs(reflvec) if self.mtx is not None else reflvec

        # 用 nvdiffrast 从 cubemap 采样：传入 [1, 6, H, W, 3] cubemap 和 [1, H, W, 3] 方向，
        # boundary_mode="cube" 负责将方向向量路由到对应面进行插值
        if base:
            color = dr.texture(
                self.base[None, ...],            # [1, 6, face_h, face_w, 3]
                sample_dirs[None, ...].contiguous(),  # [1, H, W, 3]
                filter_mode="linear",
                boundary_mode="cube",
            )[0]  # [H, W, 3]
        else:
            color = dr.texture(
                self.diffuse[None, ...],         # [1, 6, face_h, face_w, 3]
                sample_dirs[None, ...].contiguous(),  # [1, H, W, 3]
                filter_mode="linear",
                boundary_mode="cube",
            )[0]  # [H, W, 3]

        if return_img:
            return color
        else:
            # 保存为图像文件，cv2 需要 BGR 顺序，所以 RGB→BGR 翻转
            cv2.imwrite(filename, color.clamp(min=0.0).detach().cpu().numpy()[..., ::-1])

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))
    
    

    @staticmethod
    def get_sh_dirs():
        pass


    @staticmethod
    def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
        lat_step_size = np.pi / res[0]
        lng_step_size = 2 * np.pi / res[1]
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, res[0], device="cuda"), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, res[1], device="cuda")], indexing='ij')


        view_dirs = torch.stack([  torch.cos(theta) * torch.cos(phi), 
                                torch.sin(theta) * torch.cos(phi), 
                                torch.sin(phi)], dim=-1).view(res[0], res[1], 3)    # [envH, envW, 3]
        
        return view_dirs
    
    def get_env_map(self):
        dirs = self.envmap_dirs
        if self.mtx is not None:
            dirs = self.rotate_dirs(dirs)
        envmap = dr.texture(
            self.base[None, ...],
            dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        return envmap
    
    
    def capture(self):
        captured_list = [
            self.base,
            self.optimizer.state_dict(),
        ]

        return captured_list
    
    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.base,
         opt_dict) = model_args[:2]
        
        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter


@dataclass
class PointLight:
    """点光源：位置 + 颜色 + 强度"""
    position: torch.Tensor  # [3] 世界坐标
    color: torch.Tensor     # [3] RGB 线性颜色
    intensity: float = 1.0
