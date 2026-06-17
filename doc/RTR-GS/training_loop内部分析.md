# RTR-GS 训练循环内部机制分析

> 基于代码阅读，梳理 `train.py` 中训练循环的内部实现细节，方便后续开发和调试。

---

## 1. `training()` 函数 — 整体流程

**文件**: `train.py:28`

```python
def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
```

由 `__main__` 在 514 行调用，`is_pbr` 根据 `--type` 参数推导。

### 1.1 Setup 阶段 (28-136)

| 步骤 | 行号 | 说明 |
|------|------|------|
| TensorBoard writer | 30 | `prepare_output_and_logger(dataset)`，路径为 `model_path` |
| 高斯模型初始化 | 43-61 | 按优先级：checkpoint -> PLY checkpoint -> 已加载场景 -> PCD |
| PBR 组件设置 | 67-124 | `TransferMLP`、`CubemapLight`、`brdf_lut`、`refmap`、occlusion volumes |
| 渲染函数选择 | 130 | `render_fn = render_fn_dict[args.type]` |
| 背景色 | 131-132 | 白背景 `[1,1,1]` 或黑背景 `[0,0,0]` |

### 1.2 主训练循环 (137-261)

```
for iteration in progress_bar:
```

每个 iteration 的执行顺序：

```
1. 更新学习率 (144)
2. SH degree 每1000步递增 (148-149)
3. 随机选取一个训练相机 (153-157)
4. 渲染 render_fn(..., is_training=True) (163-165)
5. 损失计算 → loss.backward() (171-186)
6. with torch.no_grad(): (188-261)
   a. 更新进度条 (190-197) — 仅显示 psnr/psnr_pbr，EMA平滑
   b. training_report() (200-202) — 写 TensorBoard
   c. 密化/剪枝/法线传播 (205-231)
   d. 优化器 step (235-240)
   e. 保存 checkpoint (243-258)
```

### 1.3 训练结束 (262-275)

- PBR 清理: `cubemap.build_sh(3)`, `gaussians.incident_to_transfer()`
- 写入 `trainint_time.txt`
- 如果 `--eval` 且未 `--skip_eval`: 调用 `eval_render()`

---

## 2. `training_report()` — 训练中的评估

**文件**: `train.py:279`

```python
def training_report(tb_writer, iteration, tb_dict, scene, renderFunc, pipe,
                    bg_color, scaling_modifier=1.0, override_color=None,
                    opt=None, is_training=False, **kwargs):
```

### 2.1 每轮迭代都做的事

```python
# 281-284
if tb_writer:
    for key in tb_dict:
        tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)
```

将当前 iteration 的所有 loss 标量写入 TensorBoard。

### 2.2 每 `test_interval` 做的事

```python
# 287
if iteration % args.test_interval == 0:
```

`--test_interval` 默认 **4000**。触发时：

1. **对 test 和 train 两个集合**分别评估
2. **遍历该集合的所有相机**，用 `is_training=False` 渲染
3. 对每个视角：
   - 构建 `write_image_dict`，包含 `image`, `gt_image`, `opacity`, `depth`
   - 合并 `vis_dict`（normal, radiance_color, ref_color 等）
   - 前 **10 个视角**写入 TensorBoard IMAGES
4. 计算并打印 L1 / PSNR（PBR 模式还有 PSNR_PBR）
5. 将 L1/PSNR 写入 TensorBoard SCALARS
6. 写 opacity histogram 和 total_points 到 TensorBoard
7. 最后一次 iteration 写 `test_loss.txt` / `train_loss.txt`

**关键**: 图像只写到 TensorBoard，不保存为磁盘文件。

---

## 3. `eval_render()` — 训练结束后的完整评估

**文件**: `train.py:371`

仅在 `--eval` 且未 `--skip_eval` 时调用。完整评估流程：

| 步骤 | 说明 |
|------|------|
| 初始化 LPIPS | 加载 VGG 网络 (`get_lpips_model`) |
| 遍历 test 所有相机 | 用 `is_training=False` 渲染 |
| 计算指标 | PSNR + SSIM + LPIPS |
| 保存图像 | 输出到 `{model_path}/eval/{key}/{view_name}.png` |
| 写指标文件 | `eval.txt` (PSNR/SSIM/LPIPS) |

### 保存的图像类型

```
eval/
  render/        # 混合渲染结果
  gt/            # 真值
  normal/        # 法线图
  depth/         # 深度图
  opacity/       # 不透明度
  pseudo_normal/ # 伪法线
  radiance_color/# 辐照度颜色
  ref_color/     # 反射颜色
  ref_roughness/ # 反射粗糙度
  ref_strength/  # 反射强度
  blended_radiance/ # 混合辐射
  blended_ref_color/ # 混合反射颜色
  envmap.png     # 环境贴图 (仅 PBR)
  pbr/           # PBR 渲染 (仅 PBR)
  base_color/    # 漫反射 albedo (仅 PBR)
  roughness/     # 粗糙度 (仅 PBR)
  metallic/      # 金属度 (仅 PBR)
  diffuse_pbr/   # PBR 漫反射 (仅 PBR)
  specular_pbr/  # PBR 高光 (仅 PBR)
  visibility/    # 可见性 (仅 PBR)
  incidents_light/ # 入射光 (仅 PBR)
```

---

## 4. Render 输出字段

**文件**: `gaussian_renderer/render.py`

### 4.1 `render_view()` 输出 (18)

| 字段 | 形状 | 说明 | PBR |
|------|------|------|-----|
| `render` | [3,H,W] | 混合渲染图像 | ✓ |
| `depth` | [1,H,W] | 深度图 | ✓ |
| `depth_var` | [1,H,W] | 深度方差 | ✓ |
| `normal` | [3,H,W] | 法线图 | ✓ |
| `pseudo_normal` | [3,H,W] | 伪法线 | ✓ |
| `surface_xyz` | [3,H,W] | 3D坐标 | ✓ |
| `opacity` | [1,H,W] | 不透明度 | ✓ |
| `ref_roughness` | [1,H,W] | 反射粗糙度 | ✓ |
| `ref_strength` | [1,H,W] | 反射强度 | ✓ |
| `pbr` | [3,H,W] | PBR 渲染 | ✓ |
| `base_color` | 见 vis_dict | 漫反射 albedo | ✓ |
| `roughness` | 见 vis_dict | 粗糙度 | ✓ |
| `metallic` | 见 vis_dict | 金属度 | ✓ |
| `visibility` | 见 vis_dict | 可见性 | ✓ |

### 4.2 `vis_dict` 字段 (仅 `is_training=False` 时生成, 411)

| 字段 | 说明 |
|------|------|
| `surf_depth` | 表面深度 |
| `depth` | 深度可视化 (colormap) |
| `normal` | 法线 (归一化到 [0,1]) |
| `pseudo_normal` | 伪法线 (归一化到 [0,1]) |
| `ref_roughness` | 反射粗糙度 |
| `ref_strength` | 反射强度 |
| `radiance_color` | 辐射颜色 |
| `ref_color` | 反射颜色 |
| `ref_tint` | 反射色调 |
| `blended_radiance` | 混合辐射 |
| `blended_ref_color` | 混合反射颜色 |
| `base_color` | 漫反射 albedo (PBR) |
| `base_color_rgb` | 同上 RGB |
| `roughness` | 粗糙度 (PBR) |
| `metallic` | 金属度 (PBR) |
| `visibility` | 可见性 (PBR) |
| `diffuse_pbr` | PBR 漫反射 |
| `specular_pbr` | PBR 高光 |
| `image_pbr` | PBR 图像 (gamma校正后) |
| `incidents_light` | 入射光 |
| `env_export_base` | 环境导出基础 |
| `env_export_diffuse` | 环境导出漫反射 |

### 4.3 `calculate_loss()` 输出的 `tb_dict` (511)

| 字段 | 说明 | PBR |
|------|------|-----|
| `l1` | L1损失 | ✓ |
| `psnr` | PSNR | ✓ |
| `ssim` | SSIM | ✓ |
| `num_points` | 高斯点数 | ✓ |
| `loss_depth` | 深度损失 | ✓ |
| `loss_mask_entropy` | 掩码熵 | ✓ |
| `loss_normal_render_depth` | 法线一致性 | ✓ |
| `loss_normal_smooth` | 法线平滑 | ✓ |
| `loss_ref_roughness_smooth` | 反射粗糙度平滑 | ✓ |
| `loss_ref_strength_smooth` | 反射强度平滑 | ✓ |
| `loss` | 总损失 | ✓ |
| `l1_pbr` | PBR L1 | ✓ |
| `ssim_pbr` | PBR SSIM | ✓ |
| `psnr_pbr` | PBR PSNR | ✓ |
| 各平滑损失 | albedo/roughness/metallic/env 平滑 | ✓ |

---

## 5. 图像保存通用模式

代码库中所有图像保存都使用 `torchvision.utils.save_image`:

```python
from torchvision.utils import save_image

# 通用格式
save_image(torch.clamp(tensor, 0.0, 1.0), path)
```

参数：
- `tensor`: `[C, H, W]` 或 `[N, C, H, W]`，值域 `[0,1]`
- `path`: 输出的 PNG 文件路径

注意：
- 保存前始终 `torch.clamp(tensor, 0.0, 1.0)` 确保有效值域
- 法线图从 `[-1,1]` 映射到 `[0,1]`: `normal * 0.5 + 0.5`
- 深度图归一化: `(depth - depth.min()) / (depth.max() - depth.min())`

---

## 6. 相关参数表

**文件**: `train.py:474-493` (在 `__main__` 中定义)

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--test_interval` | 4000 | TensorBoard 写入图像 + 打印 L1/PSNR 的间隔 |
| `--save_interval` | 30000 | 保存 PLY point_cloud 的间隔 |
| `--checkpoint_interval` | 30000 | 保存完整 checkpoint (.pth) 的间隔 |
| `--skip_eval` | False | 是否跳过训练结束后的完整评估 |
| `--eval` (ModelParams) | False | 启用/禁用最终 `eval_render` |

**注意**: 这三个 interval 参数都在 `__main__` 的 `argparse` 中定义，不在 `arguments/__init__.py` 的参数组里。

---

## 7. 运行时的数据流

```
相机 → render_fn → render_pkg dict
                      ├── "render" → L1/SSIM 损失
                      ├── "tb_dict" → TensorBoard 标量 (每轮)
                      ├── "loss" → backward
                      └── (is_training=False 时) "vis_dict" → TensorBoard 图像 (test_interval)
                                                           → eval_render 磁盘保存 (训练结束)
```

TensorBoard 是训练中唯一可查看中间图像的方式。`eval_render()` 只在训练结束后才保存图像到磁盘。

---

## 8. helper 工具函数

| 文件 | 函数 | 用途 |
|------|------|------|
| `utils/image_utils.py` | `psnr()` | PSNR 计算 |
| `utils/image_utils.py` | `visualize_depth()` | 深度图 turco colormap 可视化 |
| `utils/image_utils.py` | `mse()` | MSE |
| `utils/image_utils.py` | `mae()` | MAE |
| `utils/loss_utils.py` | `ssim()` | SSIM |
| `utils/graphics_utils.py` | `linear2srgb_torch()` | gamma 校正 |
| `utils/system_utils.py` | `prepare_output_and_logger()` | TensorBoard writer 创建 |
| `gaussian_renderer/__init__.py` | `render_fn_dict` | 渲染函数映射表 |
