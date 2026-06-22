# Submodules & CUDA Rasterizers

## 概述

RTR-GS 项目包含 5 个子模块目录和 3 个不同的 CUDA 光栅化器，分别服务于透视渲染、等距渲染和遮挡评估。

## 子模块一览

| 目录 | 类型 | 用途 | 编译方式 |
|------|------|------|---------|
| `submodules/rtr_gs-rasterization/` | CUDA rasterizer | **透视模式**混合渲染（PRT + 反射贴图 + PBR） | `pip install submodules/rtr_gs-rasterization/` |
| `submodules/spherical-gaussian-splatting/submodules/spherical-gaussian-rasterization/` | CUDA rasterizer | **等距模式**渲染（360° equirect） | `pip install submodules/spherical-gaussian-rasterization/`（在 SGS 子模块内） |
| `submodules/diff-gaussian-rasterization/` | CUDA rasterizer | 原始 3DGS 光栅化器（baking 的轻量 cubemap 渲染用） | `pip install submodules/diff-gaussian-rasterization/` |
| `submodules/gs-ir/` | CUDA kernels | 遮挡体素（SH 插值 + 重建） | `pip install submodules/gs-ir/` |
| `submodules/simple-knn/` | CUDA kernel | KNN 初始化（`distCUDA2` 计算初始 scale） | `pip install submodules/simple-knn/` |
| `submodules/spherical-gaussian-splatting/` | 完整训练框架 | SGS 预训练（含 geometry + SH 优化） | 见其 `CLAUDE.md` |

## CUDA 光栅化器详解

### 1. rtr_gs-rasterization（透视模式）

**用途**：`render.py`（`-t render_ref`/`render_ref_pbr`）、`render_fast.py`

**Python 接口**：`gaussian_renderer/rtr_gs_rasterization.py` → `rtr_gs_rasterization._C`

**关键特性**：
- 支持 feature tensor（depth、depth²、normal、ref_tint、ref_roughness、ref_strength 等打包在一个 tensor 里）
- `renderPseudoNormalCUDA`（`forward.cu:428-491`）：从 depth buffer 计算伪法线，通过 viewmatrix 转换到世界空间
- `computer_pseudo_normal` 开关控制是否计算伪法线
- 使用 `getWorld2View2` 构建的 view matrix

**CUDA 文件**：
- `cuda_rasterizer/forward.cu` — 前向渲染 + 伪法线计算
- `cuda_rasterizer/backward.cu` — 反向传播
- `cuda_rasterizer/rasterizer_impl.cu` — 渲染器实现 + 调度

**渲染管线**：single-pass 光栅化 → deferred PBR shading（深度、法线、PBR 属性同时输出）

### 2. spherical-gaussian-rasterization（等距模式）

**位置**：`submodules/spherical-gaussian-splatting/submodules/spherical-gaussian-rasterization/`

**用途**：`render_equirect.py`（`-t render_ref_equirect`/`render_ref_pbr_equirect`）、`baking.py`（`--vis_walls`、`--equirect` 模式）

**Python 接口**：`spherical_gaussian_rasterization`（直接 import）

**关键特性**：
- `camera_type=3`（equirectangular），`camera_type=1`（pinhole）
- 使用径向深度（radial distance），非 Z-depth
- 输出：`rendered_image`, `radii`, `depth_raw`, `alpha`, `normal_raw`
- 不支持 feature tensor → `render_equirect.py` 使用多 pass 渲染策略

**camera_type 分发机制**（`rasterize_points.cu:108-142`）：
- `camera_type == 1`：Pinhole，调用 `CudaRasterizer::Rasterizer::forward()`（标准 3DGS 光栅化器，与 `diff-gaussian-rasterization` 同源但代码独立，已知有 bug）
- `camera_type == 3`：Lonlat/Equirect，调用 `CudaRasterizer::LonlatRasterizer::forward()`（专用的 equirect 光栅化器，SGS 训练使用的路径）
- 两个路径的 CUDA 代码完全独立，不共享 kernel

**LonlatRasterizer 双 pass 架构**（`rasterizer_impl.cu:705-729`）：
每次 `rasterizer()` 调用内部串行执行两个独立的 CUDA kernel：
1. **`render`**（`renderCUDA`）：输出 RGB 颜色，标准 alpha blending
2. **`renderGeometry`**（`renderGeometryCUDA`）：输出 depth + alpha + normal，单独的 geometry pass

Python 端看到的是一次调用返回全部输出，GPU 上两个 kernel 串行执行。这对 baking 的 equirect 模式很重要——单次调用同时获得 RGB（遮挡掩码）和 depth（墙壁检测），无需重复渲染。

**深度语义（重要！）**：
- `depth_raw` 来自 `renderGeometryCUDA`（`forward.cu:593-700`），输出已 alpha 归一化：
  ```c
  out_depth[pix_id] = depth_sum / denom;  // denom = weight_sum = alpha_acc
  ```
  即 `depth_raw = Σ(depth_i * vis_i) / Σ(vis_i)`，**不是原始累加值**。
- `alpha` = `alpha_acc = 1 - Π(1 - α_i)` = `Σ(vis_i)`（相同值）。
- Python 侧**不要**再除 `alpha`，否则双除导致 depth 偏小（半透明边缘尤其明显）。

**行序约定**（所有三个 rasterizer/direction 函数一致，无 Y-flip）：
- SGS `point3ToLonlatScreen`：pix.y=0 对应 lat=-π/2（南极），pix.y=H-1 对应 lat=+π/2（北极）
- `_equirect_ray_dirs`（`render_equirect.py`）：row 0 = lat=+π/2（北极），使用 `-sin(lat)` 作为 Y（COLMAP view space +Y down）
- `get_envmap_dirs`（`baking.py`）：row 0 = +Y（北极），使用 reflvec 约定（+Y up, -Z forward）
- `baking.py` 中的 `_equirect_ray_dirs`：与 `render_equirect.py` 版本不同，使用 `+sin(lat)`（equirect space +Y up），然后通过 `diag(1, -1, 1)` 转换到 COLMAP view space

**与透视模式 depth 的对比**：
- 透视使用 `rtr_gs-rasterization` 的特征渲染（feature pass），输出 **未归一化** 的 `Σ(depth * vis)`，Python 需手动 `depth / opacity`。
- 全景使用 `renderGeometryCUDA` 的 depth 输出，**已归一化**，Python 直接用 `depth`。

**p_view.w vs p_view.z**：
- 全景（`camera_type=3`）：`depths[idx] = p_view.w`（径向距离），见 `forward.cu:934`
- 透视（`camera_type=1`）：`depths[idx] = p_view.z`（Z-depth），见 `forward.cu:448`
- 全景模式 `p_view` 的齐次 w 分量是到相机的欧氏距离，乘以单位射线方向即 3D 位置。

**渲染管线**：多 pass 渲染（每 pass 一个颜色属性）：
1. 前向着色 PRT 颜色
2. 法线贴图（编码为 RGB）
3. reflection 属性（strength + roughness + tint）
4. PBR base color
5. PBR packed（roughness + metallic + depth）
6. Incident light

### 3. diff-gaussian-rasterization（原始 3DGS）

**用途**：`baking.py`（`_C.lite_rasterize_gaussians`，用于体素遮挡烘焙中的 cubemap 渲染）

**特性**：
- 原始 3DGS 光栅化器，轻量级
- 不输出法线、特征等扩展属性

### 4. gs-ir（遮挡体素）

**用途**：所有模式（perspective / equirect / fast）的 PBR 遮挡评估

**Python 接口**：`gs_ir.recon_occlusion`

**CUDA 文件**：
- `src/occlusion_kernel.cu` — 核心遮挡评估：
  - `sparse_interpolate_coefficients_kernel`：稀疏插值（三线性权重 + 余弦掩码防止自遮挡）
  - `SH_reconstruction_kernel`：使用 GGX importance sampling 重建 SH 可见性
  - `dialate_occlusion_ids_kernel`：膨胀空缺体素（baking 用）
- `src/irradiance_kernel.cu` — irradiance 相关

### 5. simple-knn

**用途**：`scene/gaussian_model.py` → `distCUDA2`，计算点云 KNN 距离用于初始 scale

## 调用关系

```
render.py (perspective)
  ├── rtr_gs-rasterization → CUDA 光栅化（feature tensor + 伪法线）
  └── gs-ir → recon_occlusion（遮挡评估）

render_equirect.py (equirect)
  ├── spherical-gaussian-rasterization → CUDA 光栅化（多 pass）
  └── gs-ir → recon_occlusion（遮挡评估）

baking.py
  ├── spherical-gaussian-rasterization → vis_walls 可视化渲染 / --equirect 模式烘焙
  ├── diff-gaussian-rasterization → _C.lite_rasterize_gaussians（cubemap 渲染，默认模式）
  └── gs-ir → _C（dialate_occlusion_ids 等）

scene/gaussian_model.py
  └── simple-knn → distCUDA2（初始化）
```

## 编译顺序

```bash
# 按依赖顺序编译：
pip install submodules/diff-gaussian-rasterization/
pip install submodules/simple-knn/
pip install submodules/rtr_gs-rasterization/
pip install submodules/gs-ir/
# SGS 子模块内部的 rasterizer：
cd submodules/spherical-gaussian-splatting
pip install submodules/spherical-gaussian-rasterization/
pip install submodules/simple-knn/
```

## 注意

- `rtr_gs-rasterization` 和 `spherical-gaussian-rasterization` 是两个**独立的 CUDA 光栅化器**，代码不互通
- `gs-ir` 的 `recon_occlusion` 在 `render.py` 和 `render_equirect.py` 中用法一致（相同的 `shift_points` 自遮挡偏移和 SH 重建逻辑）
- `diff-gaussian-rasterization` 仅 baking 使用，不在训练/推理管线中
