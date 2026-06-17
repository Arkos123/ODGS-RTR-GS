# RTR-GS 完整训练流程介绍

> RTR-GS: 基于3D高斯泼溅（3D Gaussian Splatting）的逆渲染框架，支持辐射传输与反射建模

## 概述

RTR-GS 采用 **两阶段训练** 策略，将几何重建与BRDF/材质分解解耦，最终实现新视角合成、重光照（Relighting）和材质编辑等功能。

整个流程分为以下主要步骤：

1. **Stage 1 — 几何与反射预训练**（30,000次迭代）
2. **遮挡烘培（Occlusion Baking）**
3. **Stage 2 — PBR 精细优化**（40,000次迭代）
4. **评估渲染（Render & Eval）**
5. **重光照（Relighting）**

---

## 1. 数据集格式

RTR-GS 支持多种数据集格式，自动检测方式如下：

| 数据集 | 检测条件 | 示例 |
|--------|---------|------|
| **Colmap** | 目录下存在 `sparse/` | MipNeRF 360, Shiny Blender(Real) |
| **Blender/NeRF-Synthetic** | 存在 `transforms_train.json` | TensoIR 合成数据, Shiny Blender(Synthetic) |
| **Stanford ORB** | 路径包含 `"stanford_orb"` | Stanford ORB 数据集 |
| **NeILF** | 存在 `inputs/sfm_scene.json` | NeILF 格式数据 |
| **Synthetic4Relight** | 路径包含 `"Synthetic4Relight"` | 合成可重光照数据 |

---

## 2. Stage 1 — 几何与反射预训练

### 目标

从多视角图像出发，重建场景的 **3D高斯几何**，同时训练 **反射分支（Reflection Branch）** 以捕获高光反射特征。

### 运行命令

```bash
python train.py --eval \
    -s <data_path> \
    -m <output_path>/stage1 \
    --lambda_normal_render_depth 0.01 \
    --diffuse_iteration 3000 \
    --skip_eval \
    --ref_map \
    -t render_ref \
    --compute_with_prt \
    --densify_grad_threshold 0.0005
```

### 核心参数说明

| 参数 | 作用 |
|------|------|
| `-t render_ref` | 使用混合渲染（Hybrid Rendering）模式，包含前向辐射传输 + 延迟反射渲染 |
| `--compute_with_prt` | 使用 **预计算辐射传输（Precomputed Radiance Transfer, PRT）** 替代球谐函数（SH），增强低频约束 |
| `--ref_map` | 启用反射贴图（Reflection Map），处理高光频率成分 |
| `--diffuse_iteration 3000` | 前3,000次迭代仅训练漫反射部分，之后再引入反射 |
| `--lambda_normal_render_depth 0.01` | 法线与深度伪法线的一致性损失权重 |
| `--skip_eval` | 跳过训练期间的评估，加速训练 |
| `--densify_grad_threshold 0.0005` | 高斯密化梯度阈值，控制高斯分裂/克隆的敏感度（MipNeRF场景用更小值） |

### 关键技术细节

- **混合渲染模型**：将渲染分为低频辐照度（PRT前向渲染）和高频反射（延迟渲染），再通过屏幕空间混合得到最终结果
- **法线建模**：高斯短轴方向作为法线，通过深度伪法线一致性损失和反射渲染梯度进行优化
- **PRT网络**：`TransferMLP` 解码传输特征，所有高斯共享，提供更强的低频约束，防止高频过拟合导致的漂浮伪影
- **密化策略**：迭代500-10,000次进行高斯密化和剪枝，后续冻结几何

### 输出

```
<model_path>/stage1/
├── checkpoint/
│   ├── chkpnt30000.pth              # 高斯模型检查点
│   ├── transfer_net_chkpnt30000.pth # PRT传输网络
│   ├── cubemap_chkpnt30000.pth      # 环境光照 Cubemap
│   └── refmap_chkpnt30000.pth       # 反射贴图
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
└── ...
```

---

## 3. 遮挡烘培（Occlusion Baking）

### 目标

在Stage 1训练完成后，预计算场景的 **遮挡体积（Occlusion Volumes）**，为Stage 2提供阴影计算支持。该步骤在三维体素网格中预先计算可见性，以球谐系数形式存储。

### 运行命令

```bash
python baking.py \
    --checkpoint <output_path>/stage1/checkpoint/chkpnt30000.pth \
    --bound 2.0 \
    --occlu_res 128
```

### 核心参数说明

| 参数 | 作用 |
|------|------|
| `--bound 2.0` | 遮挡体积的边界范围（场景坐标系单位）。MipNeRF场景通常用2.0，Stanford ORB场景用1.5 |
| `--occlu_res 128` | 遮挡体积分辨率（体素网格大小），如128³ |
| `--valid 1.5` | 标识有效区域，用于裁剪无效高斯以加速烘培 |
| `--occlusion 0.4` | 遮挡阈值，控制可见区域判定。值越小，环境光遮挡越轻 |
| `--cubemap_res 256` | 烘培过程中生成的Cubemap分辨率 |

### 技术细节

- 算法将场景空间划分为 `occlu_res³` 的体素网格
- 对每个体素，从该位置向6个方向（Cubemap面）渲染深度图
- 使用 `nvdiffrast` 将Cubemap转换为球面环境图并计算可见性
- 可见性以球谐系数形式存储在 `occlusion_coefficients` 中
- 对空缺体素执行膨胀操作以填补空洞

### 输出

```
<model_path>/stage1/checkpoint/
└── occlusion_volumes.pth    # 包含 occlusion_ids, occlusion_coefficients, bound, degree 等信息
```

---

## 4. Stage 2 — PBR 精细优化

### 目标

在Stage 1的几何基础上，加载 **BRDF/PBR分支**，分解材质的 **漫反射(base_color)、粗糙度(roughness)、金属度(metallic)** 以及 **环境光照**，同时保持高质量渲染。

### 运行命令

```bash
python train.py --eval \
    -s <data_path> \
    -m <output_path>/stage2 \
    -c <output_path>/stage1/checkpoint/chkpnt30000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    --iterations 40000 \
    --lambda_ref_strength_smooth 0.01 \
    --lambda_reflect_strength_equal_metallic 0.0 \
    --ref_map \
    -t render_ref_pbr \
    --compute_with_prt
```

> **注意**：对于 **Stanford ORB** 数据集，会增加 `--metallic` 标志和 `--lambda_reflect_strength_equal_metallic 0.1` 以支持金属材质分解。

### 核心参数说明

| 参数 | 作用 |
|------|------|
| `-c` / `--checkpoint` | 加载Stage 1的高斯模型检查点，恢复几何 |
| `--occlusion_path` | 加载预计算的遮挡体积，用于阴影计算 |
| `-t render_ref_pbr` | 使用PBR渲染模式，同时运行混合渲染分支和PBR分支 |
| `--iterations 40000` | 总训练迭代数（Stage 2通常40,000次） |
| `--lambda_ref_strength_smooth 0.01` | 反射强度平滑损失权重 |
| `--lambda_reflect_strength_equal_metallic 0.0/0.1` | 金属度与反射强度一致性损失权重（金属场景设为0.1） |
| `--metallic` | 启用金属BRDF模型（Stanford ORB等金属场景使用） |

### 双分支渲染架构

Stage 2 同时运行两个渲染分支：

```
┌─────────────────────────────────────────────┐
│              训练过程中的渲染                    │
├──────────────────────┬──────────────────────┤
│  混合渲染分支          │     PBR 分支          │
│  (Hybrid Rendering)  │  (Physically Based)   │
│                      │                      │
│  · 重建几何（冻结）      │  · 分解材质：          │
│  · 存储反射属性         │    - 漫反射 albedo    │
│  · 提供GT监督信号       │    - 粗糙度 roughness  │
│                      │    - 金属度 metallic   │
│                      │  · 分解环境光照：        │
│                      │    - CubemapLight    │
│                      │  · 输出PBR渲染结果      │
└──────────────────────┴──────────────────────┘
```

**为什么需要双分支？** 冻结几何或单独使用PBR都会导致质量下降。混合分支提供稳定的几何重建监督，PBR分支则负责材质分解。

### 损失函数体系

| 损失项 | 公式 | 作用 |
|--------|------|------|
| 渲染损失 | `L = (1-λ)L1 + λL_D-SSIM` | 混合渲染和PBR渲染均使用 |
| 法线一致性 | `L_n = \|n - n̂_d\|₂` | 高斯法线与深度伪法线对齐 |
| 白色光照正则 | `L_light = Σ(L_c - 1/3·ΣL_c)` | 约束光照为白光的先验 |
| 金属反射先验 | `L_m = L1(m, R_i)` | 金属度≈反射强度 |
| 平滑项 | 双边平滑（Bilateral Smoothness） | BRDF参数空间平滑 |

### 输出

```
<model_path>/stage2/
├── checkpoint/
│   ├── chkpnt40000.pth              # 高斯模型检查点
│   ├── transfer_net_chkpnt40000.pth # PRT传输网络
│   ├── cubemap_chkpnt40000.pth      # 分解后的环境光照
│   └── refmap_chkpnt40000.pth       # 反射贴图
├── point_cloud/
│   └── iteration_40000/
│       └── point_cloud.ply
├── eval/                            # 评估输出（如果启用--eval）
│   ├── render/                      # 渲染图像
│   ├── normal/                      # 法线图
│   ├── pbr/                         # PBR渲染结果
│   ├── base_color/                  # 漫反射 albedo
│   ├── roughness/                   # 粗糙度图
│   ├── envmap.png                   # 分解的环境贴图
│   └── eval.txt                     # PSNR/SSIM/LPIPS 指标
└── ...
```

---

## 5. 评估渲染（Render & Evaluation）

### 目标

在测试视角上渲染最终结果，计算定量指标（PSNR/SSIM/LPIPS），保存可视化结果。

### 运行命令

```bash
python render_and_eval.py \
    -m <output_path>/stage2 \
    -c <output_path>/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    --ref_map \
    --compute_with_prt \
    -t render_ref_pbr \
    --save_video
```

### 支持的功能

- **渲染输出**：混合渲染结果、PBR渲染结果、法线图、深度图、漫反射、粗糙度、金属度等
- **指标计算**：PSNR、SSIM、LPIPS（VGG）
- **视频生成**：环绕视角的视频
- **网格导出**（可选）：提取TSDF网格，生成场景mesh
- **材质编辑**（可选）：通过 `--editing_config_path` 加载编辑配置，修改材质属性后重渲染

---

## 6. 重光照（Relighting）

### 目标

将训练场景置于新的环境光照下，生成重光照后的新视角图像。这是逆渲染的核心应用。

### 运行命令

#### MipNeRF 360 / Shiny Blender（基于Colmap的实景）

```bash
python eval_relighting_colmap.py \
    -s <data_path> \
    -m <output_path>/stage2 \
    -c <output_path>/stage2/checkpoint/chkpnt40000.pth \
    -e "./data/env_maps/" \
    --ref_map \
    -t render_ref_pbr \
    --compute_with_prt \
    --save_video
```

#### TensoIR / 合成数据（基于Blender格式）

```bash
python eval_relighting_tensorIR.py \
    -m <output_path>/stage2 \
    -c <output_path>/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    -e "./data/env_maps/" \
    --ref_map \
    --relight \
    --compute_with_prt \
    -t render_ref_pbr \
    --save_video
```

#### Stanford ORB

```bash
python eval_relighting_stanford_orb.py \
    -m <output_path>/stage2 \
    -c <output_path>/stage2/checkpoint/chkpnt40000.pth \
    --occlusion_path <output_path>/stage1/checkpoint/occlusion_volumes.pth \
    --ref_map \
    --relight \
    --compute_with_prt \
    --metallic \
    -t render_ref_pbr
```

### 运行环境贴图

支持的环境贴图（位于 `./data/env_maps/`）：
- `big-studio-01_4K.exr` — 摄影棚光照
- `rock-theatre-viewpoint_4K.exr` — 剧场光照
- `sunset.hdr` — 日落
- `bridge.hdr` — 桥梁
- `city.hdr` — 城市
- `fireplace.hdr` — 壁炉
- `forest.hdr` — 森林
- `night.hdr` — 夜晚

---

## 7. 完整流程总结

### 数据流程图

```
多视角图像
    │
    ▼
┌──────────────────┐
│  Stage 1 训练     │  30,000次迭代
│  (几何+反射)       │  -t render_ref
└────────┬─────────┘
         │ checkpoint (chkpnt30000.pth)
         ▼
┌──────────────────┐
│  遮挡烘培          │  预计算可见性
│  (Occlusion Baking) │
└────────┬─────────┘
         │ occlusion_volumes.pth
         ▼
┌──────────────────┐
│  Stage 2 训练     │  40,000次迭代
│  (PBR材质分解)     │  -t render_ref_pbr
└────────┬─────────┘
         │ checkpoint (chkpnt40000.pth)
         ▼
┌──────────────────┐
│  评估渲染          │  PSNR/SSIM/LPIPS
│  (Render & Eval)  │  可视化结果
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  重光照            │  新环境光照下渲染
│  (Relighting)     │  新视角合成
└──────────────────┘
```

### 时间估计

| 步骤 | 场景规模 | 预估时间（单GPU） |
|------|---------|------------------|
| Stage 1 (30k iters) | 中等 | 1-2 小时 |
| 遮挡烘培 | 128³ | 30分钟 - 1小时 |
| Stage 2 (30k基础上+10k iters) | 中等 | 1-2 小时 |
| 评估渲染 | 测试集 | 10-30 分钟 |
| 重光照 | N个环境图 | 每个5-10分钟 |

---

## 8. 不同场景的命令差异

### MipNeRF 360 / Shiny Blender Real

```bash
sh script/run_real_scene.sh
```

特点：无 `--metallic`，遮挡边界 `--bound 2.0`，使用 `eval_relighting_colmap.py`，高斯基密化更敏感（`--densify_grad_threshold 0.0005`）

### TensoIR / Shiny Blender Synthetic

```bash
sh script/run_synthetic.sh
```

特点：使用 `--metallic` 和 `--lambda_reflect_strength_equal_metallic 0.1`，遮挡边界 `--bound 1.5`，使用 `eval_relighting_tensorIR.py`，有 `--lambda_mask_entropy 0.1`

### Stanford ORB

```bash
sh script/run_orb.sh
```

特点：与合成数据类似但使用 `eval_relighting_stanford_orb.py`，评估和重光照分开执行

---

## 9. 输出目录结构全貌

```
<model_path>/stage2/
├── checkpoint/
│   ├── chkpnt40000.pth
│   ├── transfer_net_chkpnt40000.pth
│   ├── cubemap_chkpnt40000.pth
│   ├── refmap_chkpnt40000.pth
│   └── occlusion_volumes.pth       # 来自stage1
├── point_cloud/
│   └── iteration_XXXXX/
│       └── point_cloud.ply
├── eval/
│   ├── render/                     # [渲染结果]
│   ├── gt/                         # [真值图像]
│   ├── normal/                     # [法线图]
│   ├── envmap.png                  # [环境贴图]
│   ├── pbr/                        # [PBR渲染]
│   ├── base_color/                 # [漫反射albdeo]
│   ├── roughness/                  # [粗糙度]
│   ├── depth/                      # [深度图]
│   ├── radiance_color/             # [辐照度颜色]
│   ├── refl_color/                 # [反射颜色]
│   ├── refl_strength/              # [反射强度]
│   ├── specular_pbr/              # [PBR高光]
│   ├── diffuse_pbr/               # [PBR漫反射]
│   ├── opacity/                    # [不透明度]
│   ├── metallic/                   # [金属度(金属场景)]
│   └── eval.txt                    # [量化指标]
├── test_rli/                       # [重光照结果]
│   ├── studio/                     # [各环境贴图下的渲染]
│   ├── rock-theatre/
│   ├── sunset/
│   ├── ...
├── trainint_time.txt               # [训练耗时记录]
├── exported_mesh/                  # [导出的网格(可选)]
│   ├── fuse_unbounded.ply
│   └── fuse_unbounded_post.ply
└── editing_config.json             # [材质编辑配置(可选)]
```
