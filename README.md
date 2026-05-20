# RTR-GS + ODGS Unified Workspace

This repository integrates two projects in a single unified environment:

| Project | Description | Reference |
|---------|-------------|-----------|
| **RTR-GS** | 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection | MM 2025 |
| **ODGS** (submodule) | Omnidirectional 3D Gaussian Splatting for 360-degree equirectangular images | NeurIPS 2024 |

- Original RTR-GS README: [README_orig_RTR-GS.md](README_orig_RTR-GS.md)
- Original ODGS README/CLAUDE: [submodules/odgs/CLAUDE.md](submodules/odgs/CLAUDE.md)

## Directory Structure

```
RTR-GS/
├── submodules/
│   ├── simple-knn/                     # KNN density estimation (shared by both projects)
│   ├── rtr_gs-rasterization/           # RTR-GS CUDA rasterizer (PRT + reflection)
│   ├── gs-ir/                          # Irradiance / occlusion CUDA kernels
│   ├── diff-gaussian-rasterization/    # RTR-GS's modified diff rasterizer
│   └── odgs/                           # ODGS submodule
│       └── submodules/
│           ├── odgs-gaussian-rasterization/  # ODGS equirectangular CUDA rasterizer
│           └── simple-knn/                   # (ignored, use the shared one above)
├── environment.yml                     # Unified conda environment (name: odgs-rtr)
├── README.md                           # This file
└── README_orig_RTR-GS.md               # Original RTR-GS documentation
```

## Prerequisites

- Linux (tested on Ubuntu)
- NVIDIA GPU with CUDA 11.8 support (e.g. RTX 3090, A100, etc.)
- NVIDIA driver supporting CUDA 11.8 (driver version >= 520)
- Conda (Miniconda or Anaconda)
- GCC (for compiling CUDA extensions)

## Installation

### Step 1: Clone with submodules

```bash
git clone <your-repo-url> RTR-GS
cd RTR-GS

# Initialize all submodules recursively
git submodule update --init --recursive
```

Note: ODGS's `submodules/simple-knn/` is intentionally left empty. The shared `simple-knn` at `submodules/simple-knn/` is used instead.

### Step 2: Create conda environment

```bash
conda env create -f environment.yml
conda activate odgs-rtr
```

### Step 3: Install extra pip packages

Some packages require manual installation with specific version pins.

#### kornia (needed by RTR-GS for image filtering)

```bash
pip install kornia==0.7.3
```

#### torch-scatter (needed by RTR-GS)

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

#### nvdiffrast (needed by RTR-GS for differentiable rendering)

> **Compatibility note**: PyTorch 2.1.2 requires `setuptools<70` for its CUDA extension build system.
> The `pip install "setuptools<70"` step below ensures this.

```bash
pip install "setuptools<70" wheel ninja
git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
pip install /tmp/nvdiffrast --no-build-isolation
rm -rf /tmp/nvdiffrast
```

#### protobuf (fix tensorboard compatibility)

> `environment.yml` installs `tensorboard=2.10.0`, which requires an older protobuf version.
> Without this fix, running ODGS or RTR-GS training will fail with:
> `TypeError: Descriptors cannot be created directly.`

```bash
pip install "protobuf>=3.20,<4"
```

### Step 4: Compile CUDA extensions

All CUDA extensions must be compiled for the current PyTorch + CUDA version. Compile in this order:

> **Note**: All `pip install .` commands below use `--no-build-isolation` to ensure
> the build process can access PyTorch and CUDA from the current environment.

```bash
# (4a) simple-knn – shared by both projects, install only once
cd submodules/simple-knn
pip install . --no-build-isolation
cd ../..

# (4b) RTR-GS extensions
cd submodules/rtr_gs-rasterization
pip install . --no-build-isolation
cd ../..

cd submodules/gs-ir
pip install . --no-build-isolation
cd ../..

cd submodules/diff-gaussian-rasterization
pip install . --no-build-isolation
cd ../..

# (4c) ODGS extension – equirectangular rasterizer
cd submodules/odgs/submodules/odgs-gaussian-rasterization
pip install . --no-build-isolation
cd ../../..

cd submodules/odgs/submodules
# 如果没有需下载
git clone https://github.com/Cekavis/diff-gaussian-rasterization-pinhole.git
cd diff-gaussian-rasterization-pinhole
pip install . --no-build-isolation
cd ../..

```

### Step 5: Verify installation

```bash
conda activate odgs-rtr

python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, Available: {torch.cuda.is_available()}')

import simple_knn
print('simple_knn: OK')

import rtr_gs_rasterization
print('rtr_gs_rasterization: OK')

import gs_ir
print('gs_ir: OK')

import diff_gaussian_rasterization
print('diff_gaussian_rasterization: OK')

import odgs_gaussian_rasterization
print('odgs_gaussian_rasterization: OK')

import nvdiffrast.torch as dr
print('nvdiffrast: OK')

from torch_scatter import scatter
print('torch_scatter: OK')
"
```

## Usage

### RTR-GS: Training + Inverse Rendering

Refer to the original documentation in [README_orig_RTR-GS.md](README_orig_RTR-GS.md) for full details.

#### Stage 1 – Geometry and Reflection (30k iterations)

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

#### Bake occlusion volumes

```bash
python baking.py \
    --checkpoint <output_path>/stage1/checkpoint/chkpnt30000.pth \
    --bound 1.5 \
    --occlu_res 128
```

#### Stage 2 – PBR Refinement (40k iterations)

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

### ODGS: Omnidirectional Training

ODGS is located at `submodules/odgs/`. Run training from the repo root:

```bash
# Train
cd submodules/odgs
python train.py -s <dataset_path> -m <output_path> --eval
cd ../..

# Render omnidirectional (equirectangular)
cd submodules/odgs
python render.py -m <output_path> --iteration <N>
cd ../..

# Render perspective (pinhole projection)
cd submodules/odgs
python render_perspective.py -m <output_path> --iteration <N>
cd ../..
```

## Environment Details

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 (CUDA 11.8) |
| CUDA Toolkit | 11.8 |
| simple-knn | Compiled from source |
| odgs-gaussian-rasterization | Compiled from source |
| rtr_gs-rasterization | Compiled from source |
| gs-ir | Compiled from source |
| diff-gaussian-rasterization | Compiled from source |
| nvdiffrast | Compiled from source |

## Citation

If you use RTR-GS, please cite:

```bibtex
@inproceedings{10.1145/3746027.3755197,
    author = {Zhou, Yongyang and Zhang, Fanglue and Wang, Zichen and Zhang, Lei},
    title = {RTR-GS: 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection},
    year = {2025},
    booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
    pages = {6888–6897}
}
```

If you use ODGS, please cite the ODGS NeurIPS 2024 paper accordingly.
