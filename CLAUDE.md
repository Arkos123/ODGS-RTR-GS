# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTR-GS is a 3D Gaussian Splatting framework for inverse rendering with radiance transfer and reflection. It enables novel view synthesis, BRDF/lighting decomposition, and relighting of objects with arbitrary reflectance properties, including reflective surfaces.

This repository also includes **Spherical Gaussian Splatting (SGS)** as a submodule at `submodules/spherical-gaussian-splatting/`, enabling RTR-GS to train with omnidirectional (equirectangular) 360° images. See [README.md](README.md) for unified environment setup and read `submodules/spherical-gaussian-splatting/CLAUDE.md` for SGS usage.
- The old ODGS (Omni3DGS) submodule has been replaced by the new SGS(Spherical3DGS) submodule. See `doc/关于融合RTRGS&ODGS的方案/implement/*.md` for historical integration details.

## Documentation

The [doc/](doc/) directory contains detailed documentation for this project, including:
- **Development log**: Detailed development process and key decisions in `doc/dev_log/*.md`.
- **Technical deep dives**: In-depth analysis of rendering internals, occlusion baking, and key features (e.g. `doc/RTR-GS/ref_map介绍.md`)

Check the `doc/` folder for details if you need more information.

## RTR-GS Key Algorithmic Concepts

> For more details, please refer to `paper/full.md`.

### Hybrid Rendering Model (Core Innovation)

The method separates **high-frequency** and **low-frequency** appearances:

1. **Radiance (low-frequency)**: Computed via **forward rendering** using Precomputed Radiance Transfer (PRT)
   - View-independent: `C_d ≈ ρ_d · Σ(c_j · c_j^t)` - transfer vector dot product with SH lighting
   - View-dependent: Uses neural radiance transfer via MLP `G(f_t, o)` to decode transfer features
   - All Gaussians share global SH lighting `c_j` and MLP `G`, providing stronger low-frequency constraints

2. **Reflection (high-frequency)**: Computed via **deferred rendering** using reflection map
   - Formula: `C_ref = R_t · F_ref(E_r, R_r, n, v)` (split-sum approximation)
   - Deferred rendering preserves BRDF sharpness better than forward rendering

3. **Final blending**: `I_rgb = C_r · (1 - R_i) + C_ref · R_i` (screen-space)

### Why PRT over Spherical Harmonics?

- SH lacks directional resolution for specular reflections, causing overfitting
- PRT connects Gaussians through shared global components (lighting + MLP)
- Prevents floating artifacts from high-frequency overfitting
- Better maintains geometric smoothness

### Normal Modeling

- Normal = shortest axis of Gaussian, oriented toward viewing direction
- Optimized via: (1) consistency with pseudo-normals from depth map, (2) gradients from reflection rendering
- Simplified normal propagation periodically enhances opacity for robustness

### Two-Branch Rendering for Decomposition

1. **Hybrid Rendering Branch**: Reconstructs geometry, stores reflection attributes
2. **PBR Branch**: Decomposes materials (albedo, metallic, roughness) and lighting

Both branches run simultaneously during Stage 2 - freezing geometry or using PBR alone degrades quality (see ablation in paper).

### Occlusion Baking

- Precomputes visibility in 3D voxel grid for shadow computation (`baking.py` → `occlusion_volumes.pth`)
- Uses spherical harmonics coefficients stored per voxel (SH degree=3 by default)
- **`recon_occlusion`** (`submodules/gs-ir/gs_ir/__init__.py`): During PBR rendering, interpolates per-pixel occlusion SH coefficients from the voxel grid and evaluates visibility in the surface normal direction using GGX importance sampling
- Self-occlusion prevention: evaluation point is shifted half a grid step along the normal direction before voxel interpolation (`shift_points = points + normals * grid_step * 0.5`)
- Cosine mask in CUDA kernel (`occlusion_kernel.cu`) excludes voxel corners below the surface
- **Coordinate system**: Baking's SH coefficients are computed in **reflvec space** (nvdiffrast cubemap convention: +Y=up, -Z=forward), while normals from `get_min_axis` are in **COLMAP world space** (+Y=down, +Z=forward). A `diag(1, -1, -1)` conversion is applied in `recon_occlusion` before SH evaluation. See `doc/dev_log/20260617-001-equirect-camera-centric-to-world-space-fix.md` for details.
- `--skip_walls`: Excludes scene-boundary geometry (walls/floor/ceiling) from occlusion via distance threshold (`--wall_margin`)
- `--auto_bound`: Automatically computes the voxel grid AABB from Gaussian positions
- Enables indirect lighting modeling via parameter `L_ind`

## Coordinate Conventions

Multiple coordinate systems exist in the codebase. **Mixing them is the most common source of bugs** (normal color inversion, occlusion misalignment).

| Space | +X | +Y | +Z | Where it appears |
|-------|----|----|----|-----------------|
| **COLMAP world** | right | **down** | **forward** | Gaussian xyz/normals, `scene_min/max`, SGS rasterizer (when fed COLMAP viewmatrix) |
| **reflvec / cubemap** | right | up | **-forward (-Z)** | `get_envmap_dirs`, nvdiffrast `boundary_mode="cube"`, baking's SH coefficients storage |
| **Equirect view (SGS internal)** | right | up | forward | SGS `_erp_ray_grid`, `_erp_depth_to_normal` raw output (SGS submodule only). Uses `+sin(lat)` → +Y up |

> `render_equirect.py` (RTR-GS project) has its own `_equirect_ray_dirs` and `_erp_depth_to_normal` that use **COLMAP** (+Y down) directly — they are not in Equirect view space.

**Key conversions:**
- `COLMAP → reflvec`: `diag(1, -1, -1)` (flip Y and Z)
- `COLMAP → equirect view (SGS)`: `diag(1, -1, 1)` (flip Y only)

**Equirect baking direction consistency:**
The SGS equirect rasterizer (fed a COLMAP viewmatrix) and `get_envmap_dirs` use different coordinate systems but the **same pixel `(i,j)` corresponds to the same physical direction** after `diag(1,-1,-1)` conversion. No mask remapping is needed. The column-remap "fix" previously suggested in [doc/RTR-GS/equirect-baking-z-flip.md](doc/RTR-GS/equirect-baking-z-flip.md) was based on comparing vector components across coordinate systems (COLMAP +Z=forward vs reflvec -Z=forward) and has been removed. Verify with `scripts/test_equirect_baking_dirs.py`.

### 行主序 vs 列主序

CUDA 光栅化器使用 **OpenGL 列主序** 约定，而 PyTorch 默认行主序。`world_view_transform` 传到 CUDA 前必须 `.T` 转成列主序。详见 [doc/RTR-GS/row-major-column-major.md](doc/RTR-GS/row-major-column-major.md)。

**Files where conventions collide (be careful):**
- `gaussian_renderer/render_equirect.py` — RTR-GS 版本的 `_equirect_ray_dirs` / `_erp_depth_to_normal` 直接在 COLMAP space 中工作（`-sin(lat)`），输出的是 COLMAP 空间法线 → 只需 C2W 旋转即可到 world space。而 SGS 子模块的对应函数使用 Equirect view（`+sin(lat)`），需要额外 Y flip。
- `submodules/gs-ir/gs_ir/__init__.py` — `recon_occlusion`: baking 的 SH 系数在 reflvec 空间，传入的 normals 是 COLMAP 空间 → 内部用 `diag(1, -1, -1)` 转换后再做 SH 重建
- `baking.py` — `envmap_dirs` (reflvec) used with world-space `position` for `hit_pos` → requires diag(1,-1,-1) conversion
- `baking.py --equirect` — occlusion mask (COLMAP space) and SH components (reflvec space) correspond to the same physical direction per pixel; no remapping needed

**Cubemap rotation order** (nvdiffrast convention):
`+X(0), -X(1), +Y(2), -Y(3), +Z(4), -Z(5)`

### Python vs CUDA 旋转矩阵约定

`build_rotation` (Python) 与 `computeCov3D` (CUDA) 的四元数→旋转矩阵约定**互为转置**：`R_cuda = R_py^T`。因此 `get_min_axis` 取 R_py 的列 = R_cuda 的行，匹配 CUDA 的 `computeShortAxisNormalView`。详见 [doc/RTR-GS/rotation-convention.md](doc/RTR-GS/rotation-convention.md)。

RTR-GS CUDA rasterizer (`rtr_gs-rasterization`): `renderPseudoNormalCUDA` computes normals in camera space then transforms to world space using the view matrix. See `forward.cu:488-490`.



## Pipeline

Activate environment before running any commands: `conda activate odgs-rtr`

### Full SGS → RTR-GS Equirectangular Pipeline (in dev)

The codebase supports **equirectangular (360° panorama)** training via the SGS spherical CUDA rasterizer. This is a 5-stage pipeline for omnidirectional images:

1. **SGS training**: Reconstruct geometry from equirectangular images using the spherical Gaussian rasterizer (camera_type=3)
2. **PLY conversion** (`script/sgs2rtrgs.py`): Convert SGS format PLY → RTR-GS format (adds default PBR/reflection attributes)
3. **RTR-GS Stage 1** (`train.py --t render_ref_equirect`): Geometry + reflection pre-training in equirect mode
4. **Occlusion baking** (`baking.py`): Precompute visibility. Use `--equirect` flag for SGS equirect rasterizer (avoids perspective-rasterizer artifacts on equirect-trained Gaussians). See `doc/RTR-GS/occlusion_baking.md`.
5. **RTR-GS Stage 2** (`train.py --t render_ref_pbr_equirect`): PBR material decomposition in equirect mode

> See `script/run_sgs_rtr.sh` for details.

**Key differences from perspective mode**:
- Uses `spherical_gaussian_rasterization` instead of `diff_gaussian_rasterization`
- Camera type 3 (equirectangular) in the rasterizer settings
- Depth-derived pseudo-normals (`_erp_depth_to_normal` in `render_equirect.py`) for normal supervision
- Uses SGS-style equirect densification (latitude-aware thresholds, conservative capped pruning, initial-point protection) — see `doc/dev_log/20260705-002-equirect-densification-sgs-port.md`
- V2 single-pass extra_features: all non-color attributes (normal, reflection, PBR) rasterized via one extra_features tensor

### Perspective Training Pipeline (traditional)

> See `script/run_real_scene.sh` for example.

3. **RTR-GS Stage 1** (`train.py -t render_ref`): Geometry + reflection pre-training in perspective mode
4. **Occlusion baking** (`baking.py`): Precompute visibility
5. **RTR-GS Stage 2** (`train.py -t render_ref_pbr`): PBR material decomposition in perspective mode

### Scripts
```bash
# TensoIR or ShinyBlender Synthetic (perpective mode)
sh script/run_synthetic.sh

# Stanford ORB (perpective mode)
sh script/run_orb.sh

# MipNerf360 or Shiny Blender Real (perpective mode)
sh script/run_real_scene.sh

# 360Roam real scene (perspective mode, cube faces from equirect)
sh script/run_360roam.sh

# Equirectangular mode
sh script/run_sgs_rtr.sh
```
Others:
- `script/sgs2rtrgs.py`: SGS → RTR-GS PLY
- `scripts/equi2blender.py`: OpenMVG equirect dataset → Perspective blender format (split into 6 cube faces)


## Key Architecture

### Submodules and Rasterizers

The project has 6 submodule directories and 3 CUDA rasterizers:

| Rasterizer | Mode | Used by |
|---|---|---|
| `rtr_gs-rasterization` | **Perspective** (pinhole) | `render.py`, `render_fast.py` |
| `spherical-gaussian-rasterization` (inside SGS) | **Equirect** (360°) | `render_equirect.py` |
| `diff-gaussian-rasterization` | Original 3DGS (lightweight) | `baking.py` (cubemap voxel render) |

Others: `gs-ir` (occlusion voxel SH interpolation), `simple-knn` (initial scale via `distCUDA2`).

> See `doc/submodules-and-rasterizers.md` for detailed descriptions, CUDA file structure, depth semantics, and full call graph.

### Core Modules

- **`train.py`**: Main training entry point. Handles two-stage training pipeline with PBR components.
- **`scene/gaussian_model.py`**: `GaussianModel` class storing 3D Gaussian attributes (position, SH coefficients, opacity, scaling, rotation) plus PBR-specific attributes (base_color, roughness, metallic, reflection properties).
- **`scene/__init__.py`**: `Scene` class managing dataset loading (supports Colmap, Blender, StanfordORB, NeILF, Synthetic4Relight, OpenMVG formats).
- **`gaussian_renderer/`**: Rendering pipeline with three modules, selected via `render_fn_dict`:
  - **`__init__.py`**: Registry mapping `-t` flags to render functions: `render_ref`/`render_ref_pbr` → `render.py`, `render_ref_fast` → `render_fast.py`, `render_ref_equirect`/`render_ref_pbr_equirect` → `render_equirect.py`.
  - **`render.py`** (perspective mode): Full hybrid rendering with deferred reflection. Uses the `rtr_gs_rasterization.py` with:
    - Forward-shaded PRT color pass (diffuse → view-dependent)
    - Deferred reflection map shading (`get_reflectance_color`)
    - Multi-pass feature rendering: depth, depth², normal, ref_tint, ref_roughness, ref_strength, plus PBR attributes (base_color, roughness, metallic, incident_light) all in one CUDA feature tensor
    - PBR shading with occlusion, Cook-Torrance BRDF, and environment map
    - Cubernap-based relighting support (`transfer_light` mode)
  - **`render_fast.py`** (perspective lightweight variant): Simplified rendering with deferred PBR-only shading
  - **`render_equirect.py`** (equirectangular 360° mode): Full equirect rendering using the **SGS spherical CUDA rasterizer** (`spherical_gaussian_rasterization`). Key characteristics:
    - **V2 single-pass extra_features**: The SGS V2 rasterizer supports multi-channel `extra_features` in a single call. All non-color attributes (normal, ref_strength/roughness/tint, base_color, roughness, metallic, incident light) are concatenated into a per-Gaussian `extra_features: [P, N]` tensor and rasterized in one pass alongside the main color. The result is alpha-normalized (`rendered_extra / opacity_for_div * alpha_mask`) post-rasterization, then sliced into individual attribute maps. This replaces the previous multi-pass (3-6 separate rasterizer calls) approach.
    - **Depth-derived pseudo-normals** (`_erp_depth_to_normal`): Multi-scale, edge-safe normal estimation from depth via tangent cross-products, only across same-surface neighbors. Provides high-quality geometric normal supervision when CUDA rasterizer's analytical normal (shortest-axis) is unreliable in ERP space.
    - **Equirect ray geometry**: Uses `_equirect_ray_dirs()` for world-space ray directions and `_project_lat_lon()` for densification in ERP coordinates.
    - **Normal-facing visualization**: Red=back-facing, blue=front-facing, gray=background debug overlay.
    - **Loss functions** in `calculate_loss()`: L1+SSIM + optional mask entropy + ref_roughness/ref_strength edge-aware smoothness + normal-from-depth MSE + normal TV smoothness + PBR losses and environment map regularization.
    - Camera type=3 for SGS rasterizer.
  - **`rtr_gs_rasterization.py`** (perspective rasterizer wrapper): Python wrapper around the CUDA rasterizer (`rtr_gs_rasterization._C`). 
- **`pbr/`**: Physically-based rendering components:
  - `light.py`: `CubemapLight` for environment lighting
  - `shade.py`: PBR shading functions (BRDF evaluation, environment map sampling)
- **`baking.py`**: Precomputes occlusion volumes for shadow computation.

### Rendering Types (`-t` flag)

- `render_ref` / `render_ref_fast`: Perspective hybrid rendering with reflection map
- `render_ref_pbr`: Perspective PBR branch for BRDF/lighting decomposition
- `render_ref_equirect`: Equirectangular (360° panorama) hybrid rendering using SGS rasterizer
- `render_ref_pbr_equirect`: Equirect PBR mode
- `neilf_ref` / `neilf_ref_pbr` / `neilf_ref_fast`: (legacy, same as render_xx)

### Dataset Formats

The codebase supports multiple dataset formats detected automatically:

- **Colmap**: `sparse/` directory present
- **Blender/NeRF-Synthetic**: `transforms_train.json` file
- **Stanford ORB**: Path contains "stanford_orb"
- **NeILF**: `inputs/sfm_scene.json` file
- **Synthetic4Relight**: Path contains "Synthetic4Relight"
- **OpenMVG**: `data_extrinsics.json` file (equirectangular/panoramic datasets for the SGS omnidirectional pipeline)

## Output Structure

See `lab_output/OmniBlender/barbershop/` for equirectangular pipeline output example.

See `lab_output/360Roam/base_blender/` for perspective pipeline output example.

**语言要求：所有思考、计划、分析过程及回答全部使用中文！**