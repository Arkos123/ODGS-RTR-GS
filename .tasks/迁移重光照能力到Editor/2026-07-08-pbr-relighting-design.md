# PBR 重光照渲染迁移设计 — Phase 3

## 元信息

- **日期**: 2026-07-08
- **状态**: 已批准
- **关联任务**: `.tasks/迁移重光照能力到Editor/task_plan.md` Phase 3
- **参考实现**: `RTR-GS/viewer_clean.py`（equirect PBR 渲染流程）
- **参考模块**: `RTR-GS/gaussian_renderer/render_equirect.py`（V2 extra_features + PBR shading）
- **范围**: 仅 Phase 3，不含 PRT(transfer_net)、不含 GUI 光照编辑面板

## 目标

在 3DGS-Editor-3.0 中增加 PBR 重光照渲染能力：

1. 加载 RTR-GS Stage2 PLY（已有，Phase 2 完成）
2. 加载 HDR 环境贴图 → CubemapLight
3. 加载 occlusion_volumes（可选）
4. 通过 equirect 渲染管线执行 PBR shading
5. 交互式查看重光照结果（视角旋转/环境光旋转/遮挡开关）

## 架构

### 数据流总览

```
                        ┌──────────────────┐
                        │   Scene 对象       │
                        │  (状态容器)         │
                        └──┬──────┬──────┬──┘
                           │      │      │
              ┌────────────┘      │      └──────────────┐
              ▼                   ▼                     ▼
     ┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐
     │ GaussianObject  │  │ CubemapLight │  │ occlusion_volumes    │
     │ 列表 (N个对象)   │  │ (envmap)     │  │ (baking输出，可选)    │
     │ 含PBR属性        │  │ + brdf_lut   │  │                      │
     └────────┬───────┘  └──────────────┘  └──────────────────────┘
              │                 │                     │
              └──────┬──────────┴──────────┬──────────┘
                     ▼                     ▼
           ┌──────────────────┐  ┌────────────────────┐
           │  合并属性 (cat)   │  │ pbr_shading()       │
           │  + extra_features│  │ (屏幕空间，后处理)    │
           │  + SGS光栅化     │  │ cubemap+normals+BRDF│
           └────────┬─────────┘  └─────────┬──────────┘
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────┐
                    │ PBR Equirect 图   │
                    │ [3, EqH, EqW]     │
                    └────────┬─────────┘
                             │
                    ┌────────▼────────┐
                    │ extract_viewport│
                    │ → 显示          │
                    └─────────────────┘
```

### 渲染流程（详细）

以下为 `render_sgs_equirect(..., is_pbr=True)` 的完整流程：

```
1. 合并可见 GaussianObject 的属性
   └─ positions, scales, rotations, opacities (cat)
   └─ shs_dc, shs_rest (cat → SH features)
   └─ base_color, roughness, metallic (cat)
   └─ incidents_dc, incidents_rest (cat → SH transfer)
   └─ normals = get_min_axis() 用 camera_center 计算

2. 计算 incident_light_rgb
   └─ incidents_sh = combine(dc + rest) → [N, 3, (sh+1)^2]
   └─ transfer_shs = incidents_sh.permute(0,2,1)  # [N, (sh+1)^2, 3]
   └─ light_shs = cubemap.shs  # [3, (sh+1)^2]
   └─ incidents = light_shs * transfer_shs  # broadcast
   └─ incidents_rgb = eval_sh(sh_degree, incidents, normals)  # [N, 3]

3. 构建 extra_features → [N, 11] (V2 格式)
   └─ [0:3] normal * 0.5 + 0.5 (编码到[0,1])
   └─ [3:6] base_color
   └─ [6:7] roughness.clamp(0.04, 1.0)
   └─ [7:8] metallic
   └─ [8:11] incidents_rgb

4. SGS Equirect 光栅化 (camera_type=3)
   └─ rasterizer(means3D, opacities, shs, extra_features, scales, rotations)
   └─ 返回: rendered_image [3,HW], rendered_extra [11,HW], radii, depth, alpha, normal_raw

5. Alpha-归一化 + 切片
   └─ rendered_extra = rendered_extra / opacity.clamp_min(1e-5) * alpha_mask
   └─ normal_map = rendered_extra[0:3] * 2 - 1 → normalize
   └─ base_color_map = rendered_extra[3:6].permute(1,2,0)
   └─ roughness_map = rendered_extra[6:7].permute(1,2,0)
   └─ metallic_map = rendered_extra[7:8].permute(1,2,0)
   └─ incident_map = rendered_extra[8:11].permute(1,2,0)

6. 计算遮挡 (可选)
   └─ if occlusion_volumes:
       └─ 从 depth + view_dirs → 表面点
       └─ recon_occlusion(points, normals, roughness, occlusion_coefficients)
       └─ → occlusion_map [H,W,1]

7. PBR Shading (屏幕空间)
   └─ pbr_shading(light=cubemap, normals=normal_map, view_dirs=view_dirs,
                   albedo=base_color_map, roughness=roughness_map,
                   metallic=metallic_map, occlusion=occlusion_map,
                   brdf_lut=brdf_lut)
   └─ → rendered_pbr [H,W,3]
   └─ 输出 = rendered_pbr * opacity + (1-opacity) * bg_color

8. 返回: rgb(rendered_pbr), normal, depth, alpha
```

## 文件改动清单

### 1. `scene.py` — Scene 类扩展

新增字段：
```python
# PBR 重光照状态
self.cubemap = None              # CubemapLight | None
self.occlusion_volumes = None    # dict | None (from baking)
self.brdf_lut = None             # Tensor [1,256,256,2] | None
self.envmap_path = None          # str | None
self.enable_occlusion = True     # bool
self.env_angle_y = 0.0           # float
self.env_angle_x = 0.0           # float
```

新增方法：
- `load_envmap(self, hdr_path)` — 加载 HDR → CubemapLight
- `load_occlusion(self, occ_path)` — 加载 occlusion_volumes.pth
- `rotate_envmap(self, angle_y, angle_x)` — 旋转环境光

### 2. `render.py` — GaussianRenderer 扩展

修改 `render_sgs_equirect()`：
- 增加 `is_pbr=False` 参数
- `is_pbr=True` 时执行 PBR 分支（extra_features + pbr_shading）
- 返回结构和现有格式一致 `{'rgb': ..., 'normal': ..., 'depth': ..., 'alpha': ...}`

### 3. `control.py` — ControlWidget 扩展

新增交互元素（Phase 3 最小版本）：
- 渲染通道切换：`"rgb"` / `"normal"` / `"depth"` / `"pbr"`
- 加载 HDR 按钮（文件对话框）
- 加载遮挡体积按钮（可选）

## 依赖导入

从 RTR-GS 通过 `sys.path` 导入（无副作用，纯 PyTorch 函数/类）：

```python
# render.py / scene.py
import sys
sys.path.insert(0, "/home/huangpengyue/projects/RTR-GS")

from pbr import CubemapLight, get_brdf_lut                     # 环境光照
from pbr.shade import pbr_shading, saturate_dot                # PBR 着色
from utils.graphics_utils import read_hdr, latlong_to_cubemap   # HDR 加载
from utils.sh_utils import eval_sh                             # SH 评估
```

⚠️ `gs_ir.recon_occlusion`（CUDA 扩展）需要 **有条件导入**，仅在 `occlusion_volumes` 存在时调用。

## 现有模式兼容性

- `render_mode == "sgs_equirect"` + `render_channel == "pbr"` → PBR 渲染
- `render_mode == "sgs_equirect"` + `render_channel == "rgb"` → 现有 RGB 渲染（不变）
- 其他模式（`"gaussians"`, `"points"`, `"sgs_pinhole"`）→ 不变
- 无 cubemap 时 PBR 模式降级为普通 RGB 渲染

## Equirect 缓存

复用现有 `RenderWidget` 的 equirect 缓存机制：

```
缓存 key = (camera position, scene content_version, render_mode, render_channel)
失效条件：位置变化 / 对象增删 / envmap 旋转 / 遮挡开关切换
```

## 已知限制

- 位置改变需要重新渲染整张 equirect（2048×1024）
- 透视渲染尚未支持 PBR（需要时可通过 SGS pinhole + extra_features 增加）
- transfer_net/PRT 渲染不在本阶段范围
