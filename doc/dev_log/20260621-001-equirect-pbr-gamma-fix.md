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
