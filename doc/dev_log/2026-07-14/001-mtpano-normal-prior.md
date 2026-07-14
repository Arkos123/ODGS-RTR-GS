---
created_at: "2026-07-14"
updated_at: "2026-07-14"
---

# MTPano Normal Prior 集成

## 概述

将 MTPano（全景图多任务基础模型）的法线预测结果作为 normal prior 接入 RTR-GS 的 equirect 训练流程，替代原有的 depth-derived pseudo_normal 监督信号，提升法线估计质量。

## 背景

RTR-GS 的 equirect 模式原本通过 `_erp_depth_to_normal` 从 SGS 光栅化的深度图推导伪法线（pseudo_normal），但该法线质量完全取决于深度质量，而深度本身没有直接监督，仅通过 RGB 光度损失间接优化。在白墙、天空等无纹理区域，深度噪声大，导致伪法线不可靠。

MTPano 是全景图多任务模型（DINOv3 + PD-BridgeNet），可直接从单张 RGB 全景图预测高质量法线，不依赖深度。

## 坐标系转换

MTPano 输出的法线在模型内部空间中（`+Y上, -Z前`），需要转换到 RTR-GS 使用的 COLMAP 世界空间（`+Y下, +Z前`）。通过实测四面墙和地板/天花板对比确认，完整转换公式为：

```
MTPano 原始 (X, Y, Z)
  → X_new = -Z_mtp    (X←-Z)
  → Y_new = -Y_mtp    (Y←-Y)
  → Z_new =  X_mtp    (Z←X)
  → c2w_rot = rotation (原始 OpenMVG 旋转矩阵, 非 .T)
  → COLMAP world 法线
```

注意：c2w_rot 在 RTR-GS 管线中等价于 OpenMVG 原始 rotation 矩阵（经过 `getWorld2View(R.T) + transpose + inverse` 链后），而非 rotation.T。

## 修改的文件

### `script/convert_mtpano_normal.py` (新建)
MTPano 法线 → COLMAP world 坐标系转换脚本。读取 OpenMVG 数据集的相机位姿，对每张图像的法线执行坐标转换，并生成可视化 PNG 供对比。

### `scene/dataset_readers.py`
在 `readCamerasFromOpenMVG` 中增加 MTPano normal prior 的加载。数据集目录下如果有 `mtpano_results/{image_name}/{image_name}_normal_colmap.npy`，则加载并存入 `CameraInfo.normal` 字段。该字段通过已有的 `loadCam` → `Camera.normal` 链路自动传递到训练管线。

### `gaussian_renderer/render_equirect.py`
- `calculate_loss` 中修改 normal MSE 部分：优先检查 `viewpoint_camera.normal` 是否有有效数据，有则用作法线监督目标（`target_normal`），否则 fallback 到原有的 `results["pseudo_normal"]`
- `render_view` 中增加 `normal_prior` 到 `vis_dict`，使可视化可以输出对比图

### `train.py`
在 `core_vis_keys` 中添加 `"normal_prior"`，使其自动保存到可视化目录。

## 数据流

```
数据集加载:
  mtpano_results/0/0_normal_colmap.npy → CameraInfo.normal → Camera.normal

训练时:
  viewpoint_camera.normal (GPU, [3,H,W], COLMAP world)
    ↓ (如果存在且非零)
  target_normal = normal_prior.detach()  (+ device/dtype 对齐 + 分辨率适配)
    ↓ MSE
  loss_normal_render_depth = MSE(rendered_normal, target_normal)
```

## 验证方式

- 训练时检查 `vis/iteration_xxxx/view_xxx/normal_prior.png` 确认先验法线加载正确
- 对比 `normal.png`（渲染法线）与 `normal_prior.png`（MTPano 先验）在训练不同阶段的接近程度
- 观察 `loss_normal_render_depth` 数值是否合理收敛
