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

**用途**：`render_equirect.py`（`-t render_ref_equirect`/`render_ref_pbr_equirect`）、`baking.py`（vis_walls）

**Python 接口**：`spherical_gaussian_rasterization`（直接 import）

**关键特性**：
- `camera_type=3`（equirectangular），`camera_type=1`（pinhole）
- 使用径向深度（radial distance），非 Z-depth
- 输出：`rendered_image`, `radii`, `depth_raw`, `alpha`, `normal_raw`
- 不支持 feature tensor → `render_equirect.py` 使用多 pass 渲染策略

**深度语义（重要！）**：
- `depth_raw` 来自 `renderGeometryCUDA`（`forward.cu:593-700`），输出已 alpha 归一化：
  ```c
  out_depth[pix_id] = depth_sum / denom;  // denom = weight_sum = alpha_acc
  ```
  即 `depth_raw = Σ(depth_i * vis_i) / Σ(vis_i)`，**不是原始累加值**。
- `alpha` = `alpha_acc = 1 - Π(1 - α_i)` = `Σ(vis_i)`（相同值）。
- Python 侧**不要**再除 `alpha`，否则双除导致 depth 偏小（半透明边缘尤其明显）。

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
  ├── spherical-gaussian-rasterization → vis_walls 可视化渲染
  ├── diff-gaussian-rasterization → _C.lite_rasterize_gaussians（cubemap 渲染）
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
