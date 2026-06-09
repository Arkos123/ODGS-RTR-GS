# RTR-GS + SGS Unified Workspace

This repository integrates two projects in a single unified environment:

| Project                           | Description                                                                                 | Reference    |
| --------------------------------- | ------------------------------------------------------------------------------------------- | ------------ |
| **RTR-GS**                        | 3D Gaussian Splatting for Inverse Rendering with Radiance Transfer and Reflection           | MM 2025      |
| **SGS** (submodule)               | Omnidirectional Spherical Gaussian Splatting for 360° equirectangular images                | Based on ODGS (NeurIPS 2024) + omniGS |

- Original RTR-GS README: [README\_orig\_RTR-GS.md](README_orig_RTR-GS.md)
- SGS module CLAUDE: [submodules/spherical-gaussian-splatting/CLAUDE.md](submodules/spherical-gaussian-splatting/CLAUDE.md)

## Directory Structure

```
RTR-GS/
├── submodules/
│   ├── simple-knn/                     # KNN density estimation (shared by both projects)
│   ├── rtr_gs-rasterization/           # RTR-GS CUDA rasterizer (PRT + reflection)
│   ├── gs-ir/                          # Irradiance / occlusion CUDA kernels
│   ├── diff-gaussian-rasterization/    # RTR-GS's modified diff rasterizer
│   └── spherical-gaussian-splatting/           # SGS submodule
│       └── submodules/
│           ├── spherical-gaussian-rasterization/  # SGS spherical CUDA rasterizer
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

Note: SGS's `submodules/simple-knn/` is intentionally left empty. The shared `simple-knn` at `submodules/simple-knn/` is used instead.

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

> **⚠️ CUDA 版本确认**
> 编译 nvdiffrast 等 CUDA 扩展时，需要系统 nvcc 版本与 PyTorch 编译用的 CUDA 版本一致（本项目使用 **CUDA 11.8**）。
>
> 安装前确认：
>
> ```bash
> nvcc --version               # 查看系统 CUDA 编译器版本
> python -c "import torch; print(torch.version.cuda)"  # 查看 PyTorch 对应的 CUDA 版本
> ```
>
> 如果系统 nvcc 版本不是 11.8，在创建并激活 conda 环境后，先安装 CUDA 11.8 工具包：
>
> ```bash
> # 2. 从 conda-forge 安装完整的 CUDA 11.8 工具包
> conda install -c conda-forge cudatoolkit=11.8
>
> # 3. 重新安装 nvcc 11.8（因为 conda-forge 的 cudatoolkit 可能不带 nvcc）
> conda install -c nvidia cuda-nvcc=11.8
> # 1. 安装 CCCL（CUB + Thrust + libcudacxx 头文件）
> conda install -c conda-forge cccl
> export CUDA_HOME=$CONDA_PREFIX
>
> conda install nvidia/label/cuda-11.8.0::cuda-cudart-dev -y
> conda install nvidia/label/cuda-11.8.0::libcurand-dev -y
>
> # 验证 11.8
> nvcc --version
>
> # 如果遇到错误：/usr/include/crt/host_config.h:138:2: error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!
> # 这时可安装 gcc 8 和 g++ 8 到 conda 环境
> conda install -c conda-forge gxx_linux-64=8.5.0
> ```
>
> 之后安装的包不受 conda 环境的 CUDA 编译器影响。

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
> Without this fix, running SGS or RTR-GS training will fail with:
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

# (4c) SGS extension – spherical equirectangular rasterizer
cd submodules/spherical-gaussian-splatting/submodules/spherical-gaussian-rasterization
pip install . --no-build-isolation
cd ../../..

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

import spherical_gaussian_rasterization
print('spherical_gaussian_rasterization: OK')

import nvdiffrast.torch as dr
print('nvdiffrast: OK')

from torch_scatter import scatter
print('torch_scatter: OK')
"
```
### CUDA Environment (required for renderutils_plugin compilation)

`pbr/renderutils/` 的 CUDA 扩展编译需要以下条件。这部分代码在 [ops.py](./pbr/renderutils/ops.py#L49-L54) 中通过 `CUDA_HOME` 环境变量自动寻找 CUDA 库。

**方案 A：系统 CUDA（推荐）**
- 安装 CUDA toolkit 到标准路径（如 `/usr/local/cuda-11.8`），并创建 `/usr/local/cuda` 符号链接指向它
- `ops.py` 默认以 `/usr/local/cuda` 为 fallback，无需额外配置

**方案 B：conda CUDA 包 + 手动设 CUDA_HOME**
如果使用 conda 安装的 CUDA toolkit（`cuda-nvcc` 等包）：
```bash
# 每次激活环境时手动设置
export CUDA_HOME=/usr/local/cuda-11.8

# 或用 conda activate 钩子自动设置（推荐）：
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
EOF

mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/deactivate.d/cuda_env.sh << 'EOF'
export PATH=${PATH#$CUDA_HOME/bin:}
unset CUDA_HOME
EOF
```

验证是否生效：
```bash
conda activate odgs-rtr
echo $CUDA_HOME     # 应显示 CUDA 安装路径
which nvcc          # 应显示 CUDA 11.8 的 nvcc
```

## Usage

### RTR-GS: Training + Inverse Rendering

Refer to the original documentation in [README\_orig\_RTR-GS.md](README_orig_RTR-GS.md) for full details.

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

### SGS: Omnidirectional Training (Spherical Gaussian Splatting)

SGS is located at `submodules/spherical-gaussian-splatting/`. Run training from the repo root:

```bash
# Train
cd submodules/spherical-gaussian-splatting
python train.py -s <dataset_path> -m <output_path> --eval
cd ../..

# Render omnidirectional (equirectangular)
cd submodules/spherical-gaussian-splatting
python render.py -m <output_path> --iteration <N>
cd ../..

# Render perspective (pinhole projection)
cd submodules/spherical-gaussian-splatting
python render_perspective.py -m <output_path> --iteration <N>
cd ../..

# Render pinhole (custom intrinsics)
cd submodules/spherical-gaussian-splatting
python render_pinhole.py -m <output_path> --iteration <N>
cd ../..
```

### Interactive Viewer

This project includes a Pygame-based interactive viewer (`viewer_pygame.py`) for exploring 3D scenes with WASD/free-look or Orbit controls.

- **If you have a desktop environment** (e.g., running locally or via a remote desktop like VNC/RDP): simply run `python viewer_pygame.py` directly with the appropriate arguments — a Pygame window will open on your desktop.
- **If you are on a headless server** (no physical display, e.g., a remote Linux server): follow the guide below to stream the visuals to your browser using **Xvfb** + **x11vnc** + **noVNC**.

#### Prerequisites

These tools can be installed without root/sudo permissions:

| Tool | Description | Install method |
|------|-------------|---------------|
| **Xvfb** | Virtual framebuffer (provides a fake display) | Usually pre-installed; check `/usr/bin/Xvfb` |
| **x11vnc** + **libvncserver1** | VNC server that captures the virtual display | Extract from official Ubuntu `.deb` packages |
| **noVNC** | Web-based VNC client | Clone from GitHub |
| **pygame** | Python GUI library for the viewer | `pip install` in the conda environment |

#### Installation

**Step 1: Install x11vnc and its library dependency** (no sudo required)

```bash
# Download both .deb packages
cd /tmp
apt download x11vnc libvncserver1

# Extract both into the same tools directory
mkdir -p ~/tools/x11vnc
dpkg -x x11vnc_*.deb ~/tools/x11vnc/
dpkg -x libvncserver1_*.deb ~/tools/x11vnc/

# Verify
ls ~/tools/x11vnc/usr/bin/x11vnc
ls ~/tools/x11vnc/usr/lib/x86_64-linux-gnu/libvncserver.so.1
```

**Step 2: Install noVNC** (no sudo required)

```bash
git clone https://github.com/novnc/noVNC.git ~/tools/noVNC
```

**Step 3: Install pygame** (in the conda environment)

```bash
conda activate odgs-rtr
pip install pygame
```

**Step 4: Verify Xvfb**

Xvfb is usually pre-installed on Ubuntu servers. Verify it exists:

```bash
ls /usr/bin/Xvfb
```

If missing, ask your administrator to install it (`sudo apt install xvfb`).

#### Usage

1. Edit the checkpoint paths in `scripts/start_viewer_novnc.sh`:

```bash
CHECKPOINT="lab_output/your_scene/stage2/checkpoint/chkpnt40000.pth"
OCCLUSION_PATH="lab_output/your_scene/stage1/checkpoint/occlusion_volumes.pth"
ENVMAP_PATH="./data/env_maps/your_envmap.hdr"
# Optional: set SOURCE_PATH to your scene data directory to load camera information.
# When set, the initial view will start at the first camera's position.
SOURCE_PATH="./data/your_scene"
IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
```

> **Note on environment lighting**: The viewer loads the environment lighting in the following priority:
> 1. If `ENVMAP_PATH` is set (non-empty), the specified HDR file is used.
> 2. Otherwise, the viewer looks for a trained cubemap checkpoint (`cubemap_chkpntXXXXX.pth`) next to the main checkpoint file — this is the lighting decomposed during training.
> 3. If neither is available, an error is raised.
>
> So if you want to use the **trained lighting**, simply leave `ENVMAP_PATH` empty (`""`).

2. Run the script:

```bash
bash scripts/start_viewer_novnc.sh
```

3. Open your local browser to the URL printed by the script (e.g., `http://<server_ip>:6080/vnc.html`).

4. (Recommended) For security, use an SSH tunnel:

```bash
# On your local machine:
ssh -L 6080:localhost:6080 user@server_ip
# eg. ssh -L 6080:localhost:6080 huangpengyue@10.108.11.10
# Then open http://localhost:6080/vnc.html
```

#### Viewer Controls

| Key/Input | Action |
|-----------|--------|
| **M** | Toggle between FPS and Orbit mode |
| **W/A/S/D** | Move forward/left/backward/right (FPS mode) |
| **Q/E** | Move up/down (FPS mode) |
| **Right mouse + drag** | Rotate camera / look around |
| **Mouse wheel** | Zoom in/out (Orbit mode) |
| **←/→/↑/↓** | Rotate environment map |
| **R** | Reset environment rotation |
| **B** | Toggle environment map background |
| **O** | Toggle occlusion (AO) |
| **P** | Play test camera transforms |
| **ESC** | Exit |

#### How It Works

```
Your browser (noVNC client)
    ↓ WebSocket
noVNC proxy (port 6080)
    ↓ VNC protocol
x11vnc (port 5900)
    ↓ captures
Xvfb (virtual display :99)
    ↓ Pygame renders to
viewer_pygame.py
```

All keyboard and mouse events from your browser are transparently forwarded to the Pygame application, providing a full interactive experience.

## Environment Details

| Component                   | Version              |
| --------------------------- | -------------------- |
| Python                      | 3.10                 |
| PyTorch                     | 2.1.2 (CUDA 11.8)    |
| CUDA Toolkit                | 11.8                 |
| simple-knn                  | Compiled from source |
| spherical-gaussian-rasterization | Compiled from source |
| rtr\_gs-rasterization       | Compiled from source |
| gs-ir                       | Compiled from source |
| diff-gaussian-rasterization | Compiled from source |
| nvdiffrast                  | Compiled from source |

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

If you use the SGS omnidirectional module, please cite ODGS (NeurIPS 2024) and omniGS accordingly.
