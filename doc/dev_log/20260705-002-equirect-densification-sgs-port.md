# Equirect 模式训练优化：移植 SGS 纬度感知 Densification

## 背景

全景（equirectangular/ERP）训练模式下，RTR-GS 出现了明显的 floater 现象：
- 细长半透明的 Gaussians
- 特定角度可见的大面积半透明遮挡

## 根因分析

RTR-GS 全景模式原本使用了与透视模式相同的 densification 逻辑，但该逻辑不适合 equirect 投影：

| 问题 | 细节 | 后果 |
|------|------|------|
| **`weights` 语义错误** | `render_equirect.py` 返回 `"weights": opacity`（原始 sigmoid 不透明度），但标准 3DGS 中 `weights` 应该是高斯在像素上的累计贡献权重 | `weight_mask = weights_accum < 1e-4` 的 pruning 完全无效（opacity ≈ 0.5，远 > 1e-4） |
| **缺少纬度感知** | 极地 ERP 像素覆盖的球面面积比赤道小得多，但 densification 使用统一的梯度阈值 | 极地过度高斯克隆/分裂 |
| **修剪无保护** | 标准 `prune()` 没有观测次数要求、无上限、不保护初始点 | 低贡献高斯堆积且无法清理 |
| **Post-densification 修剪被注释** | 原本的 equirect 专用后修剪代码被注释掉了 | 后期无法清理 floater |

## 解决方案

以 SGS 子模块（`submodules/spherical-gaussian-splatting/scene/gaussian_model.py`）的 densification 系统为基准，移植到 RTR-GS 的 `GaussianModel`。

### 移植的核心方法

| 方法 | 来源 (SGS) | 功能 |
|------|-----------|------|
| `_dynamic_grad_threshold` | `_dynamic_grad_threshold` | 纬度感知阈值：`cos(lat)` 插值 `[min_thresh, max_thresh]`，极地阈值高 → 抑制过度 densification |
| `_limit_mask_by_score` | `_limit_mask_by_score` | `torch.topk` 限制候选数 ≤ `max_ratio * total` |
| `_equirect_prune_mask` | `_direct_prune_mask` | 保守修剪：需 ≥ `min_prune_obs` 次观测、cap `max_prune_ratio`、保护初始点、延迟到 `prune_start_iter` |
| `_equirect_densify_and_clone` | `densify_and_clone` (lat-aware) | 带纬度感知阈值 + capped 的克隆 |
| `_equirect_densify_and_split` | `densify_and_split` (lat-aware) | 带纬度感知阈值 + capped 的分裂 |
| `equirect_densify_and_prune` | `densify_and_prune` (SGS) | 先 prune 后 clone/split（与 SGS 顺序一致） |
| `equirect_prune` | — | 仅修剪，用于 post-densification 清理 |

### 训练循环改动 (`train.py`)

- `pipe.equirect=True` 时走 equirect 分支（SGS 风格），`else` 透视模式不变
- Equirect 分支：`weights=None` 传入 `add_densification_stats`，避免 `weights_accum` 污染
- Post-densification：`iteration >= densify_until_iter` 后继续 `equirect_prune` 清理 floater

### 参数 (`arguments/__init__.py`)

新增 `equirect_*` 参数，默认值与 SGS 一致：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `equirect_grad_threshold_min` | 0.0002 | 赤道最低梯度阈值 |
| `equirect_grad_threshold_max` | 0.002 | 极地最高梯度阈值 |
| `equirect_prune_start_iter` | 7000 | 开始修剪的迭代 |
| `equirect_min_prune_obs` | 6 | 最低观测次数要求 |
| `equirect_max_prune_ratio` | 0.03 | 单次最大修剪比例 |
| `equirect_densify_update_ratio` | 0.08 | 分位数更新比率 |
| `equirect_max_clone_ratio` | 0.18 | 单次最大克隆比率 |
| `equirect_max_split_ratio` | 0.12 | 单次最大分裂比率 |
| `equirect_split_shrink` | 1.15 | 分裂缩放因子 |
| `equirect_min_new_scale_ratio` | 0.0 | 子高斯最小缩放比例 |

## 安全性设计

- 所有新增方法仅在 `pipe.equirect=True` 时进入
- 修改的现有方法（`prune_points`, `densification_postfix`, `add_densification_stats`）均使用 `hasattr` 守卫 → 透视模式完全不受影响
- `_is_initial_point` 的同步在 `prune_points` 和 `densification_postfix` 内处理，避免尺寸不匹配

## 关键文件

- `arguments/__init__.py` — 新增 equirect 参数
- `scene/gaussian_model.py` — 新增 equirect densification 方法 + 3 处 `hasattr` 守卫修改
- `gaussian_renderer/render_equirect.py` — 移除 `"weights": opacity`
- `train.py` — equirect/perspective 分支 densification + post-pruning

## 后续可能改进

- 将 `_is_initial_point` 加入 `capture()`/`restore()`，支持 checkpoint 恢复后保护状态延续
- 考虑加入 SGS 的 geometry regularization losses（depth thickness、planar depth 等）进一步提升 equirect 几何质量
