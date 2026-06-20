# Equirect PBR Gamma Correction Fix

## 问题

全景 renderer (`render_equirect.py`) 的 PBR 输出缺少 sRGB gamma 校正，与透视 renderer (`render.py`) 行为不一致。

## 影响

### 训练 loss 错误（核心 bug）

`results["pbr"]` 直接输出线性空间值，而 GT 图像是 sRGB 编码的，导致 PBR loss 在**不一致的色彩空间**中计算：

```
# render_equirect.py (修复前)
results["pbr"] = rendered_pbr.permute(2, 0, 1)       # 线性
Ll1_pbr = F.l1_loss(results["pbr"], gt_image)          # gt_image 是 sRGB → 色彩空间不匹配!
```

这会影响 PBR 参数（base_color, roughness, metallic, env map）的训练质量。

### vis_dict 可视化不一致

`diffuse_pbr`、`specular_pbr`、`image_pbr` 写入 `vis/` 目录和 TensorBoard 时没有 gamma 校正，而透视模式有。

## 修复

给全景 renderer 的 PBR 输出加上 `gamma_func`（sRGB gamma ≈ 2.2），与透视模式对齐：

```python
# L569 — 训练 & 非训练都 gamma
results["pbr"] = gamma_func(rendered_pbr.permute(2, 0, 1))

# L592-598 — vis_dict 中 PBR 可视化输出也 gamma
vis_dict["diffuse_pbr"]  = gamma_func(diffuse_pbr.permute(2, 0, 1))
vis_dict["specular_pbr"] = gamma_func(specular_pbr.permute(2, 0, 1))
vis_dict["image_pbr"]    = gamma_func(rendered_pbr.permute(2, 0, 1))
```

同时新增 `base_color_rgb` 键（线性原始值），与透视模式对称。

## 修改文件

- `gaussian_renderer/render_equirect.py`: 5 insertions, 4 deletions

## 背景：色彩空间

| 数据 | 色彩空间 | 来源 |
|------|:--------:|------|
| GT 图像 (PNG/JPEG) | sRGB (gamma≈2.2) | 相机管线编码 |
| PBR shading 输出 | 线性 | 物理渲染公式 |
| `results["pbr"]` (修前) | 线性 ❌ | 缺少 gamma 转换 |
| `results["pbr"]` (修后) | sRGB ✅ | `gamma_func()` 转换 |

透视模式 (`render.py`) 一直有 gamma 校正，全景模式遗漏了。

---

# Coordinate Convention Refactoring: Y-down 内聚

## 动机

`_equirect_ray_dirs` 和 `_erp_depth_to_normal` 输出的射线/法线在 equirect 空间（+Y 向上），而 COLMAP view space 约定 +Y 向下。调用处需要重复做 Y flip：

```python
# 三个地方都有这段
n = n * [1, -1, 1]   # equirect → COLMAP view space
```

这既冗余又容易遗漏。

## 方案

把 Y-down 约定下沉到函数内部，修改射线方向的 Y 分量 `sin(lat)` → `-sin(lat)`，使函数直接输出 COLMAP view space 的法线/射线。

```python
# 改前
rays = [sin(lon)*cos(lat),  sin(lat), cos(lon)*cos(lat)]  # +Y up
# 改后
rays = [sin(lon)*cos(lat), -sin(lat), cos(lon)*cos(lat)]  # +Y down
```

## 效果

- 删除 3 处 Y-flip 行（`pseudo_normal`、`facing_vis`、PBR `view_dirs`）
- 删除对应的长注释
- net -9 行，调用处只保留 C2W 旋转

## 修改文件

- `gaussian_renderer/render_equirect.py`: 14 insertions, 23 deletions

---

# Depth 双除 alpha 修复

## 问题

`surf_depth = depth / rendered_opacity` 对 CUDA 已归一化的 depth 再次除以 alpha。

## 根因

`renderGeometryCUDA`（`forward.cu:694`）输出 depth 时已经做了 alpha 归一化：

```c
out_depth[pix_id] = depth_sum / denom;  // denom = weight_sum = alpha_acc
```

但 `render_equirect.py:466` 又除了一次 `rendered_opacity`（= `alpha_acc`），导致：

```
surf_depth = (depth_sum / α) / α = depth_sum / α²
```

透视模式（`render.py`）没有此问题，因为它的 depth 来自 feature pass（原始 `Σ(depth * vis)`，未归一化）。

## 影响

- 不透明像素（α≈1.0）无影响
- 半透明边缘（α<1.0）的 3D 点偏向相机 → `recon_occlusion` 在错误位置采样遮挡

## 修复

```python
# 改前
surf_depth = depth / rendered_opacity.clamp_min(1e-5)
# 改后
surf_depth = depth  # CUDA 已归一化
```

## 修改文件

- `gaussian_renderer/render_equirect.py`: 1 insertion, 1 deletion
