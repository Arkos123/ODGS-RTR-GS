# Equirect Baking Mode

## 日期

2026-06-23

## 背景

`baking.py` 原使用 6 面透视 cubemap（`diff_gaussian_rasterization._C.lite_rasterize_gaussians`）渲染遮挡体，对于 SGS equirect 流水线存在两个问题：

1. Equirect 训练的 Gaussians 包含大块拉伸的高斯，在透视投影下变成巨大 screen-space blob，导致遮挡判断错误
2. SGS `camera_type=1`（pinhole）有 bug

## 改动

### 新增 `--equirect` 模式（`baking.py`）

添加 `--equirect` 开关（由 `PipelineParams` 基类提供）和 `--equirect_res` 参数。启用时：

- 使用 SGS `GaussianRasterizer(camera_type=3)` 在体素中心**单次渲染** equirect 全景图
- 替代 6 面 cubemap + nvdiffrast `dr.texture(boundary_mode="cube")` 转换方案
- 深度输出为径向距离（`p_view.w`），已 alpha 归一化
- 背景色为白色（与 cubemap 路径一致，遮挡掩码依赖 `> 0.5` 判断）

### 分辨率选择

默认 `--equirect_res 128`（128×256 = 32K 像素）。由于遮挡最终编码为 SH degree=3（9 个系数），低频近似不需要高分辨率。对比 cubemap 的 6×256² = 393K 像素减少约 12 倍。

### `--vis_walls` depth 双除 bug 修复

SGS `renderGeometryCUDA` 输出的 depth 已 alpha 归一化：
```c
out_depth = Σ(depth_i × vis_i) / Σ(vis_i)
```

vis_walls 中原有 `depth_raw / acc` 的额外除法导致深度偏小，已修复。

### 透视流水线不变

cubemap 路径完全保留，作为默认模式。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `baking.py` | 新增 equirect 渲染分支 + vis_walls depth bug 修复 |
| `doc/RTR-GS/occlusion_baking.md` | 更新文档，记录 equirect 模式和参数 |
| `doc/submodules-and-rasterizers.md` | 补充 SGS rasterizer camera_type 分发、LonlatRasterizer 双 pass 架构、行序约定 |
| `CLAUDE.md` | 流水线步骤 4 增加 --equirect 简要说明 |

## 验证

```bash
python baking.py \
    --checkpoint <stage1>/chkpnt30000.pth \
    --auto_bound --skip_walls --equirect \
    --occlu_res 96
```
