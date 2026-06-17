# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTR-GS is a 3D Gaussian Splatting framework for inverse rendering with radiance transfer and reflection. It enables novel view synthesis, BRDF/lighting decomposition, and relighting of objects with arbitrary reflectance properties, including reflective surfaces.

This repository also includes **Spherical Gaussian Splatting (SGS)** as a submodule at `submodules/spherical-gaussian-splatting/`, enabling RTR-GS to train with omnidirectional (equirectangular) 360° images. See [README.md](README.md) for unified environment setup and read `submodules/spherical-gaussian-splatting/CLAUDE.md` for SGS usage.
- The original ODGS submodule has been replaced by the new SGS submodule. See `doc/关于融合RTRGS&ODGS的方案/implement/*.md` for historical integration details.

## Documentation

The [doc/](doc/) directory contains detailed documentation for this project, including:
- **Technical deep dives**: In-depth analysis of rendering internals, occlusion baking, and key features (e.g. `ref_map`)
- **Training pipeline overview**: Complete walkthrough of the two-stage training process
- **Integration notes**: RTR-GS & ODGS/SGS fusion analysis and research records
- **Historical commit explanations**: Context and rationale behind important changes

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
- **Coordinate system**: Baking's SH coefficients are computed in **reflvec space** (nvdiffrast cubemap convention: +Y=up, -Z=forward), while normals from `get_min_axis` are in **COLMAP world space** (+Y=down, +Z=forward). A `diag(1, -1, -1)` conversion is applied in `recon_occlusion` before SH evaluation. See `doc/20260610-occlusion-skip-walls-and-auto-bound.md` for details.
- `--skip_walls`: Excludes scene-boundary geometry (walls/floor/ceiling) from occlusion via distance threshold (`--wall_margin`)
- `--auto_bound`: Automatically computes the voxel grid AABB from Gaussian positions
- Enables indirect lighting modeling via parameter `L_ind`

## Pipeline

Activate environment before running any commands: `conda activate odgs-rtr`

### Full SGS → RTR-GS Equirectangular Pipeline (in dev)

The codebase supports **equirectangular (360° panorama)** training via the SGS spherical CUDA rasterizer. This is a 5-stage pipeline for omnidirectional images:

1. **SGS training**: Reconstruct geometry from equirectangular images using the spherical Gaussian rasterizer (camera_type=3)
2. **PLY conversion** (`script/sgs2rtrgs.py`): Convert SGS format PLY → RTR-GS format (adds default PBR/reflection attributes)
3. **RTR-GS Stage 1** (`train.py --t render_ref_equirect`): Geometry + reflection pre-training in equirect mode
4. **Occlusion baking** (`baking.py`): Precompute visibility
5. **RTR-GS Stage 2** (`train.py --t render_ref_pbr_equirect`): PBR material decomposition in equirect mode

> See `script/run_sgs_rtr.sh` for details.

**Key differences from perspective mode**:
- Uses `spherical_gaussian_rasterization` instead of `diff_gaussian_rasterization`
- Camera type 3 (equirectangular) in the rasterizer settings
- Depth-derived pseudo-normals (`_erp_depth_to_normal` in `render_equirect.py`) for normal supervision
- Multi-pass alpha-normalized rendering for normal, reflection attributes, and PBR attributes
- Geometry is frozen when loading from SGS pre-training (xyz/scaling/rotation/opacity locked)

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
    - **Multi-pass alpha-normalized rendering**: Because the SGS rasterizer handles one color attribute per rasterization call, the equirect path splits rendering into 3–6 separate GPU rasterization passes: (1) forward-shaded PRT color, (2) normal map, (3) ref_strength + ref_roughness, (4) ref_tint, (5) PBR base color, (6) PBR packed (roughness+metallic+depth), (7) incident light. Each pass is alpha-normalized (`rendered / opacity_for_div * alpha_mask`) post-rasterization.
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
