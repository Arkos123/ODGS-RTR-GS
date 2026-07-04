# 旋转矩阵约定：Python build_rotation vs CUDA computeCov3D

## 核心事实

Python `build_rotation` 与 CUDA `computeCov3D` 使用**互为转置**的四元数→旋转矩阵约定。

```
R_py = build_rotation(q)    # Python
R_cuda = computeCov3D(q)    # CUDA（转换为标准行主序后）
R_cuda = R_py^T              # 互为转置！
```

## 根源

Python 和 CUDA 的公式在符号上存在差异（所有非对角元素的 `r*z`、`r*y` 等交叉项符号相反），导致 `R_py[i,j] = R_cuda[j,i]`。

对于单位四元数 q=(1,0,0,0) 两者相同（R=I），但对于任意旋转，两者不同。

## 重要性：get_min_axis 的正确实现

这个差异直接影响 `get_min_axis` 中法线方向的提取。

CUDA `computeShortAxisNormalView` 提取最短轴的方式（`forward.cu:303-308`）：

```cuda
// axis=0: glm::vec3(R[0][0], R[1][0], R[2][0])  = R_cuda 行0
// axis=1: glm::vec3(R[0][1], R[1][1], R[2][1])  = R_cuda 行1
// axis=2: glm::vec3(R[0][2], R[1][2], R[2][2])  = R_cuda 行2
```

由于 `R_cuda = R_py^T`，CUDA 取 `R_cuda` 的行 = `R_py` 的列。

| 提取方式 | Python 中 | = R_cuda 的 | 匹配 CUDA? |
|----------|----------|-------------|-----------|
| 列（原版） | `R_py @ onehot` | **行** | **✓** |
| 行（错误修复） | `R_py^T @ onehot` | **列** |  ✗ |

```python
# 正确：取 R_py 的列 = R_cuda 的行，匹配 CUDA computeShortAxisNormalView
ndir = torch.bmm(rot_matrix, min_axis.unsqueeze(-1)).squeeze(-1)

# 错误：取 R_py 的行 = R_cuda 的列，与 CUDA 约定相反
ndir = torch.bmm(rot_matrix.transpose(1, 2), min_axis.unsqueeze(-1)).squeeze(-1)
```

## 影响范围

| 函数 | 文件 | 当前状态 |
|------|------|---------|
| `get_min_axis` | `scene/gaussian_model.py` | 正确（取列） |
| `get_minimum_axis` | `utils/general_utils.py` | 有错误注释但未被调用（死代码） |

## 验证方法

用非单位旋转的四元数验证：

```python
q = (0.5, 0.5, 0.5, 0.0)  # 旋转 > 0°
# R_py = build_rotation(q)
# R_cuda_std[i,j] 通过 CUDA 的 GLM 列主序构造 + 转换得到
# 断言: R_py.T ≈ R_cuda_std
```

## 相关 CUDA 代码

- `spherical-gaussian-rasterization/cuda_rasterizer/forward.cu:computeCov3D` — R 构建
- `spherical-gaussian-rasterization/cuda_rasterizer/forward.cu:computeShortAxisNormalView` — 法线提取
- `rtr_gs-rasterization/.../forward.cu` — RTR-GS 版本（使用相同约定）
- `diff-gaussian-rasterization/.../forward.cu` — 原始 3DGS（使用相同约定）

## 历史

- Commit `b0789a3` 误将 `get_min_axis` 从列改为行，声称 "CUDA eigenvectors = rows of R"。对于 CUDA 自身的 R 这个说法正确，但忽略了 Python `build_rotation` 的转置关系。
- Commit `86238b6` 回滚了该修复，恢复为列（正确版本）。
