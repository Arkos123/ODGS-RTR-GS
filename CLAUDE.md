# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RTR-GS is a 3D Gaussian Splatting framework for inverse rendering with radiance transfer and reflection. It enables novel view synthesis, BRDF/lighting decomposition, and relighting of objects with arbitrary reflectance properties, including reflective surfaces.

**Paper**: "RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection" (MM 2025). Full text available at `paper/full.md`.

This repository also includes **Spherical Gaussian Splatting (SGS)** as a submodule at `submodules/spherical-gaussian-splatting/`. SGS extends 3DGS to omnidirectional (equirectangular) 360° images with a custom spherical CUDA rasterizer, building on ODGS (NeurIPS 2024) and omniGS features. See [README.md](README.md) for unified environment setup and [submodules/spherical-gaussian-splatting/CLAUDE.md](submodules/spherical-gaussian-splatting/CLAUDE.md) for SGS usage.
- The original ODGS submodule (`submodules/odgs/`) has been replaced by the new SGS submodule (`submodules/spherical-gaussian-splatting/`). See `doc/关于融合RTRGS&ODGS的方案/implement/*.md` for historical integration details.
- `scripts/equi2blender.py`: Convert ODGS OpenMVG equirect datasets to RTR-GS Blender format (see [doc](doc/关于融合RTRGS&ODGS的方案/implement/5-26-001-equi2blender转换工具.md))

## Documentation

The [doc/](doc/) directory contains detailed documentation for this project, including:
- **Technical deep dives**: In-depth analysis of rendering internals, occlusion baking, and key features (e.g. `ref_map`)
- **Training pipeline overview**: Complete walkthrough of the two-stage training process
- **Integration notes**: RTR-GS & ODGS/SGS fusion analysis and research records
- **Historical commit explanations**: Context and rationale behind important changes

Check the `doc/` folder for details if you need more information.

## RTR-GS Key Algorithmic Concepts

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

- Precomputes visibility in 3D voxel grid for shadow computation
- Uses spherical harmonics coefficients stored per voxel
- Enables indirect lighting modeling via parameter `L_ind`

## Commands

**运行环境**: `conda activate odgs-rtr`

### Training Pipeline

The training is a **two-stage process**:

**Stage 1 - Geometry and Reflection Prewarning (30k iterations):**
```bash
python train.py --eval \
    -s <data_path> \
    -m <output_path>/stage1 \
    --lambda_mask_entropy 0.1 \
    --diffuse_iteration 3000 \
    --ref_map \
    --skip_eval \
    -t render_ref \
    --compute_with_prt
```

**Baking Occlusion Volumes:**
```bash
python baking.py \
    --checkpoint <output_path>/stage1/checkpoint/chkpnt30000.pth \
    --bound 1.5 \
    --occlu_res 128
```

**Stage 2 - PBR Refinement (40k iterations):**
```bash
python train.py --eval \
    -s <data_path> \
    -m <output_path>/stage2 \
    -c <output_path>/stage1/checkpoint/chkpnt30000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    --iterations 40000 \
    --ref_map \
    -t render_ref_pbr \
    --compute_with_prt
```

### Evaluation and Relighting
```bash
# Render and evaluate
python render_and_eval.py \
    -m <output_path>/stage2 \
    -c <output_path>/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    --ref_map \
    --compute_with_prt \
    -t render_ref_pbr

# Relighting with new environment maps
python eval_relighting_tensorIR.py \
    -m <output_path>/stage2 \
    -c <output_path>/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    -e <env_maps_path> \
    --ref_map \
    --relight \
    --compute_with_prt \
    -t render_ref_pbr
```

### Run Scripts
```bash
# TensoIR or ShinyBlender Synthetic
sh script/run_synthetic.sh

# Stanford ORB
sh script/run_orb.sh

# MipNerf360 or Shiny Blender Real
sh script/run_real_scene.sh
```

## Key Architecture

### Core Modules

- **`train.py`**: Main training entry point. Handles two-stage training pipeline with PBR components.
- **`scene/gaussian_model.py`**: `GaussianModel` class storing 3D Gaussian attributes (position, SH coefficients, opacity, scaling, rotation) plus PBR-specific attributes (base_color, roughness, metallic, reflection properties).
- **`scene/__init__.py`**: `Scene` class managing dataset loading (supports Colmap, Blender, StanfordORB, NeILF, Synthetic4Relight formats).
- **`gaussian_renderer/`**: Rendering pipeline with two modes:
  - `render.py`: Full hybrid rendering (forward + deferred for reflections)
  - `render_fast.py`: Fast rendering variant
- **`pbr/`**: Physically-based rendering components:
  - `light.py`: `CubemapLight` for environment lighting
  - `shade.py`: PBR shading functions (BRDF evaluation, environment map sampling)
- **`baking.py`**: Precomputes occlusion volumes for shadow computation.

### Rendering Types (`-t` flag)

- `render_ref` / `render_ref_fast`: Hybrid rendering with reflection map
- `render_ref_pbr`: Adds PBR branch for BRDF/lighting decomposition
- `render_ref` / `render_ref_pbr`: NeILF-style rendering variants (used in training)

### Key Pipeline Flags

- `--compute_with_prt`: Use Precomputed Radiance Transfer (PRT) instead of spherical harmonics
- `--ref_map`: Enable reflection map for high-frequency reflections
- `--metallic`: Enable metallic BRDF model
- `--relight`: Enable relighting mode with new environment maps

### Dataset Formats

The codebase supports multiple dataset formats detected automatically:
- **Colmap**: `sparse/` directory present
- **Blender/NeRF-Synthetic**: `transforms_train.json` file
- **Stanford ORB**: Path contains "stanford_orb"
- **NeILF**: `inputs/sfm_scene.json` file
- **Synthetic4Relight**: Path contains "Synthetic4Relight"

## Loss Functions

Key losses defined in paper Section 3.5:

- **Rendering losses**: `L = (1-λ)L1 + λL_D-SSIM` for both hybrid rendering and PBR
- **Normal consistency**: `L_n = ||n - n̂_d||_2` (pseudo-normal from depth)
- **Light regularization**: `L_light = Σ(L_c - 1/3·ΣL_c)` (white light assumption)
- **Metal reflection prior**: `L_m = L1(m, R_i)` (metallic ≈ reflection intensity)
- **Smoothness terms**: Bilateral smoothness for BRDF parameters

### Training Parameters (in `arguments/__init__.py`)
- `--iterations`: Training iterations (default: 30,000 for stage 1, 40,000 for stage 2)
- `--densify_from_iter` / `--densify_until_iter`: Gaussian densification window (500-10,000)
- `--lambda_*`: Various loss weights
  - `lambda_normal_render_depth`: 0.02 (normal consistency)
  - `lambda_white_light`: 0.003 (light regularization)
  - `lambda_reflect_strength_equal_metallic`: 0.1 (metal prior)

## Output Structure

```
<model_path>/
├── checkpoint/
│   ├── chkpnt30000.pth          # Gaussian model checkpoint
│   ├── transfer_net_chkpnt30000.pth  # PRT transfer network
│   ├── cubemap_chkpnt30000.pth  # Environment cubemap
│   └── occlusion_volumes.pth    # Baked occlusion
├── point_cloud/
│   └── iteration_XXXXX/
│       └── point_cloud.ply
└── eval/
    ├── render/                  # Rendered images
    ├── normal/                  # Normal maps
    └── eval.txt                 # Metrics (PSNR, SSIM, LPIPS)
```

