# Equirect ERP-Aware Losses & Cyclic Boundary Fix

## 背景

RTR-GS 的 equirect 全景模式训练代码原来没有利用 ERP 投影的特殊几何性质：
1. 纬度无加权 — 极地（天花板/地板）被过度 smooth
2. 无边缘门控 — smoothness loss 模糊物体边界
3. Loss 无渐变 — 几何 loss 从一开始就生效
4. 水平边界无循环 — 左右接缝处渲染不连续
5. 伪法线计算无 `torch.no_grad()` — 占用显存

## 修改内容

### 1. SGS 子模块：提取 `pano_losses.py`

`submodules/spherical-gaussian-splatting/pano_losses.py` — 从 SGS `train.py` 中提取全套全景图优化函数（纯函数，无副作用），供 RTR-GS 复用。

`submodules/spherical-gaussian-splatting/train.py` — 改为 `from pano_losses import ...`

### 2. `arguments/__init__.py`

`OptimizationParams` 新增参数：
- `geometry_loss_from_iter = 3500` — 几何 loss 起始迭代
- `geometry_loss_warmup_iters = 1500` — 渐变步数
- `lambda_alpha_hole = 0.0` — alpha 空洞 loss（默认关闭）

### 3. `gaussian_renderer/render_equirect.py`

#### a. 懒加载 `pano_losses`（避免 sys.path 污染）

`_ensure_pano_losses()` 函数在 `calculate_loss()` 首次被调用时加载 `pano_losses`，避免模块顶层的 `sys.path.insert(0, ...)` 屏蔽 SGS 子模块的同名包（如 `lpipsPyTorch`）。

#### b. ERP 循环边界：`_erp_edge_aware_loss`

新增函数，替换 `first_order_edge_aware_loss` 用于所有 ERP 渲染属性的 smoothness loss：
- 水平方向：`torch.roll`（cyclic）— 消除左右接缝
- 垂直方向：标准 padding（ERP 垂直方向有限）
- 用于：ref_roughness, ref_strength, roughness, base_color, metallic

#### c. `calculate_loss()` 改进

| 改进 | 函数 | 效果 |
|------|------|------|
| 纬度面积加权 | `pano_row_area_weight` | 极地 smooth 权重低，赤道高 |
| RGB 边缘门控 | `pano_rgb_nonedge_weight` | smooth 在物体边界处自动降权 |
| Loss 渐变 | `geometry_iter_ramp` | 几何 loss 从 iter 3500 逐步开启 |
| ERP 法线平滑 | `pano_normal_smoothness_loss` | 替换 TV loss，cyclic 水平 + 边缘门控 |
| Alpha 空洞 | `pano_alpha_hole_loss` | 自监督 alpha 惩罚（默认关闭） |
| Weighted SSIM | 关闭 `use_ws_ssim` | 节省内存（4K ERP 显存不足） |

#### d. 伪法线：`torch.no_grad()`

`_erp_depth_to_normal` 调用包 `torch.no_grad()`，避免中间张量在计算图中滞留占用显存。

### 4. `baking.py`

SGS V2 光栅化器返回 6 元组（V1 返回 5 元组），baking.py 的两个调用点已适配。

### 5. SGS CUDA `backward.cu`

`BACKWARD::renderChannels()` 添加 `cudaFuncSetAttribute`，解决 16 通道 extra_features 的 backward kernel 需要 71KB 动态共享内存超过默认 48KB 限制的问题。

## 涉及文件

- `arguments/__init__.py`
- `gaussian_renderer/render_equirect.py`
- `baking.py`
- `submodules/spherical-gaussian-splatting/pano_losses.py`（新文件）
- `submodules/spherical-gaussian-splatting/train.py`
- `submodules/spherical-gaussian-splatting/submodules/spherical-gaussian-rasterization/cuda_rasterizer/backward.cu`

## 验证

- `conda run -n odgs-rtr python -c "from gaussian_renderer.render_equirect import _ensure_pano_losses; print(all(f is not None for f in _ensure_pano_losses()))"`
- `from lpipsPyTorch import lpips` 不再因 sys.path 污染报错
