现在完全清楚了。答案是：**不是 bug，是一个有明确记录的、有意的设计选择**。让我解释完整推理过程和原因。

---

## 关键发现

### 1. `forward_shading` 默认是 False

`arguments/__init__.py:70`:
```python
self.forward_shading = False
```

### 2. 但在 equirect 模式下被强制覆盖为 True

`train.py:644-648`:
```python
# Equirect mode: force forward_shading
if args.type in ['render_ref_equirect', 'render_ref_pbr_equirect']:
    args.equirect = True
    args.forward_shading = True
```

这意味着 equirect 训练永远走 per-Gaussian 的**前向着色**路径（`render_equirect.py:312-315`），不会进入你发现的缺失的延迟路径。

---

## 对比 perspective 和 equirect 的着色路径

| 路径                           | `render.py` (perspective)                                               | `render_equirect.py`                        |
| ------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------- |
| `forward_shading=True`         | Per-Gaussian 前向反射 → alpha blend                                     | ✅ Per-Gaussian 前向反射 → alpha blend      |
| `forward_shading=False` (默认) | ✅ **延迟反射** — 光栅化 G-buffer → per-pixel `get_reflectance_color()` | ❌ **完全没有反射** — 只有 raw PRT radiance |

在 perspective 模式下，`forward_shading=False`（默认值）仍然输出带反射的图像，因为延迟路径补上了反射计算（`render.py:310-313`）。在 equirect 模式下，相同的配置就会漏掉反射——但好在 **训练代码强制 forward_shading=True**，所以实际运行时不会触发这个路径。

---

## 为什么 deliberately 选择 forward shading？

这是三个技术因素共同作用的结果：

### 原因 1：V2 extra_features 已消除光栅化约束，但 deferred 仍然没必要

> **更新说明**：SGS V2 光栅化器已支持通过 `extra_features: [P, N]` 参数一次性渲染任意通道的非颜色属性（与 perspective 的 feature tensor 类似），不再受限于"一次只能渲染一个颜色属性"。但即便这个约束已消除，deferred 反射在 equirect 中仍然不是必要的选择。

在 V2 下，所有 G-buffer 数据（normal、ref_roughness、ref_tint、PBR base_color/roughness/metallic/incident）都在单次光栅化调用中合并为 extra_features tensor 一次性拿到。但做 deferred 反射需要在 `get_reflectance_color()` 中调用 `dr.texture()` 从 cubemap 采样——这在 4K 分辨率的 equirect 图像上**逐像素跑开销非常大**，而前向方案只需在 N 个 Gaussian 上计算一次。

### 原因 2：SGS 预训练使几何体已经稳定

Equirect 管线有 5 个阶段，其中 RTR-GS Stage 1 的几何体是**从 SGS PLY 加载并冻结的**：
```
SGS 训练 → PLY 转换 → RTR-GS Stage 1 (几何冻结) → 遮挡烘焙 → RTR-GS Stage 2 (PBR)
```

几何体冻结意味着 Gaussians 位置精确。在 `get_reflectance_color_forward()` 中，每个 Gaussian 用自身的 normal/roughness/tint 独立计算反射，然后通过 alpha blending 合成——当高斯位置精确时，forward 和 deferred 在非边界区域几乎没有视觉差异。

### 原因 3：前向着色在 equirect 中更高效

Per-Gaussian 的前向反射只有一次张量运算（`render_equirect.py:313-315`）：
```python
refl_color_forward = get_reflectance_color_forward(
    refmap, normal, viewdirs_gauss, ref_roughness, ref_tint, brdf_lut=brdf_lut)
colors_precomp_pass1 = (1.0 - ref_strength) * colors_precomp + ref_strength * refl_color_forward
```

它只需要对 N 个 Gaussian 计算一次，而不是对 H×W 个**像素**。在 4096×2048 的 equirect 分辨率下，N ≈ 200K-500K，而 H×W ≈ 8.4M，差距巨大。

---

## 结论

| 你的观察                          | 结论                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| "equirect 里只有 forward shading" | ✅ 确实如此，外部表现正确                                                            |
| "没有延迟着色"                    | ✅ 确实没有实现延迟反射路径                                                          |
| "是不是 bug"                      | ❌ **不是 bug** — `train.py` 明确强制 `forward_shading=True`，确保不会走向缺失的路径 |

**但严格来说这也算一个 "弱化"**：perspective 模式可以自由切换前向/延迟反射，而 equirect 模式被固定在前向。如果你需要高质量的反射效果，可以尝试在 equirect 模式下添加延迟反射路径——所有 G-buffer 数据从 V2 extra_features 切片后已经就位，只需要在 `render_equirect.py` 约 552 行后加一个 `get_reflectance_color()` 调用即可。