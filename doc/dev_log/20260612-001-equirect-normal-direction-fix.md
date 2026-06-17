# Equirect 法线方向修复 + Depth-derived Pseudo-normal

## 问题

在 SGS (Spherical Gaussian Splatting) 的 equirect 渲染模式下，Stage 2 的 occlusion visibility 出现异常——水平面（地板、天花板）基本全黑。通过 `baking.py --vis_walls` 的 normal-facing 可视化确认：**大部分水平面的法线显示为背向视野（红色）**，违反了"可见面法线应朝向相机"的原则。

## 根因分析

### 问题 1：`get_min_axis` 的 axis convention 与 CUDA rasterizer 不一致

**所有 CUDA rasterizer**（原始 3DGS `diff-gaussian-rasterization`、RTR-GS 自定义 `rtr_gs-rasterization`、SGS `spherical-gaussian-rasterization`）的 `computeCov3D` 都使用同一约定：

```cuda
// CUDA computeCov3D (forward.cu)
glm::mat3 M = S * R;                     // 缩放在前，旋转在后
glm::mat3 Sigma = glm::transpose(M) * M; // Σ = R^T @ S² @ R
```

在此约定下，协方差矩阵 Σ 的特征向量（即 Gaussian 的主轴方向）是 **旋转矩阵 R 的行（rows of R）**。

但 RTR-GS 的 Python 函数 `get_min_axis` 使用的是 **旋转矩阵 R 的列（columns of R）**：

```python
# 修改前：取 columns of R
ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)
```

对于 identity rotation（R=I），行列相同，问题不暴露。但对于**任意旋转**（如水平面的 tilt），column i ≠ row i，最短轴的**世界方向**完全不同。这导致水平面的法线指向错误方向（背向视野）。

### 问题 2：Equirect 路径的 pseudo_normal 来源不正确

RTR-GS 的 normal consistency loss 在 `render_equirect.py` 中：

```python
loss_normal_render_depth = F.mse_loss(rendered_normal, pseudo_normal.detach())
```

在**透视路径**（`render.py` + `rtr_gs-rasterization`），`pseudo_normal` 来自 CUDA 的 `renderPseudoNormal`——一个**从深度图推导的法线**（depth-derived normal）。该法线始终几何正确，能给 `get_min_axis` 提供正确的监督信号。

在 **equirect 路径**（`render_equirect.py` + SGS rasterizer），`pseudo_normal` 直接使用了 CUDA 的 `normal_raw`（即 `computeShortAxisNormalView` 输出的**最短轴法线**）。这导致：
- `pseudo_normal` = CUDA 最短轴法线（rows of R，view space）
- `rendered_normal` = Python 最短轴法线（columns of R，world space）
- 两者在 axis convention 上直接冲突，loss 把法线拉向错误方向

SGS 子模块自身提供了正确的 depth-to-normal 函数 `_erp_depth_to_normal`，但 RTR-GS 的 equirect 路径没有使用它。

### 为什么透视路径不受影响

透视路径的 `pseudo_normal` 来自 `rtr_gs-rasterization` 的 CUDA `renderPseudoNormalCUDA`，该函数通过 depth buffer 的 3×3 邻域梯度 + cross product 计算 normal，**不依赖 Gaussian 的 axis convention**。所以即使 `get_min_axis` 的 column/row 不匹配，depth-derived 的信号也能将 normals 拉向正确的几何方向。

## 改动

### 1. `scene/gaussian_model.py` — `get_min_axis` 修复

```python
# 修改前：columns of R
ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)

# 修改后：rows of R（匹配 CUDA S*R 约定）
ndir = torch.bmm(rot_matrix.transpose(1, 2), min_axis.unsqueeze(-1)).squeeze(-1)
```

### 2. `utils/general_utils.py` — `get_minimum_axis` 修复

```python
# 取 rows of R + 修复了旧版跨 column 取值的 bug
min_axis_id = torch.argmin(scales, dim=-1)
batch_idx = torch.arange(R.shape[0], device=R.device)
ndir = R[batch_idx, min_axis_id, :]
```

### 3. `render_equirect.py` — pseudo_normal 改为 depth-derived

- 从 SGS `gaussian_renderer/__init__.py` 复制了 `_erp_depth_to_normal` 及其辅助函数（`_erp_edge_aware_smooth_depth`、`_shift_with_spatial_mask`、`_relative_depth_gate`、`_erp_tangent_from_same_surface_neighbors`、`_erp_smooth_normals_same_surface`）
- 将 `pseudo_normal` 的计算从 `_normal_from_raw(normal_raw, opacity)` 改为 `_erp_depth_to_normal(depth, opacity)`
- 清理了不再使用的 `_normal_from_raw` 函数和 `pass1_normal_raw` 变量

### 4. `baking.py` — 新增 normal-facing 可视化（用于调试）

在 `--vis_walls` 模式下新增 3 张调试输出图：
- `vis_walls_normal_facing.png`：红/蓝二值图（红=背向，蓝=朝向）
- `vis_walls_normal_cos.png`：余弦灰度热力图
- `vis_walls_normal_map.png`：原始法线方向图（RGB 编码）

## 涉及的公式

| 组件 | CUDA 约定 | Python 约定（改前） | Python 约定（改后） |
|---|---|---|---|
| 协方差 | `Σ = R^T @ S² @ R` | `Σ = R @ S² @ R^T` | 未改（不用于渲染） |
| 主轴方向 | rows of R | columns of R | **rows of R** |
| 最短轴法线 | `computeShortAxisNormalView` | `get_min_axis` | `get_min_axis` ✅ |
| Pseudo-normal | — | `normal_raw`（CUDA 最短轴） | **depth-derived** ✅ |

## 涉及的 CUDA Rasterizer（全部使用相同约定）

| 包名 | 源码位置 |
|---|---|
| `diff_gaussian_rasterization` | `RTR-GS/submodules/diff-gaussian-rasterization/` |
| `rtr_gs_rasterization` | `RTR-GS/submodules/rtr_gs-rasterization/` |
| `spherical_gaussian_rasterization` | `RTR-GS/submodules/spherical-gaussian-splatting/submodules/spherical-gaussian-rasterization/` |

## 关键函数对照

| 函数 | 所在文件 | 用途 | 约定 |
|---|---|---|---|
| `computeCov3D` | 各 CUDA rasterizer `forward.cu` | 计算 3D 协方差 | `M=S*R` → `Σ=M^T M` |
| `computeShortAxisNormalView` | SGS `forward.cu` | CUDA 最短轴法线 | rows of R |
| `get_min_axis` | `scene/gaussian_model.py` | Python 最短轴法线 | **rows of R** ✅ |
| `get_minimum_axis` | `utils/general_utils.py` | Python 最短轴法线（无相机翻转） | **rows of R** ✅ |
| `_erp_depth_to_normal` | `render_equirect.py`（复制自 SGS） | Depth-to-normal（监督信号） | 几何正确 |
| `renderPseudoNormalCUDA` | `rtr_gs-rasterization forward.cu` | 透视路径 depth-to-normal | 几何正确 |
