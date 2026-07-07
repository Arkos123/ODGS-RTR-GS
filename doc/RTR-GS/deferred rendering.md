## RTR-GS 延迟渲染架构详解

RTR-GS 采用的是 **混合 forward + deferred 渲染** 策略，核心设计思想在 CLAUDE.md 中已有概述：

> **Hybrid Rendering Model**: 分离高频和低频外观。
> - **Radiance (low-frequency)**: 通过 PRT 前向着色
> - **Reflection (high-frequency)**: 通过 reflection map **延迟渲染**

下面基于代码分步解释：

---

### Phase 1: Forward 几何光栅化 — 生成 G-buffer

在 `render.py:255-267`，CUDA 光栅化器一次调用`即生成所有 G-buffer 通道`：

```python
# 把所有属性打包成一个 feature tensor
features = torch.cat([depths, depths2, normal, ref_tint, ref_roughness, ref_strength], dim=-1)
if pc.use_pbr:
    features = torch.cat([features, base_color, roughness, metallic, incidents_light], dim=-1)

# 一次光栅化调用，生成所有属性
(num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
 rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(...)
```

这是**一个"假"前向渲染**——光栅化器看似输出了 `rendered_image`（辐射颜色），但它其实是 PRT 预计算的低频颜色（SH 辐射），并不是最终像素。**所有需要逐像素计算的属性都打包在 `features` 中一次性光栅化**，生成的是一个完整的 G-buffer：

| G-buffer 通道                        | 来源            | 分辨率      |
| ------------------------------------ | --------------- | ----------- |
| `rendered_image` (辐射)              | SH/PRT 前向着色 | `[3, H, W]` |
| `rendered_depth` / `rendered_depth2` | 深度 + 深度²    | `[1, H, W]` |
| `rendered_normal`                    | 高斯最短轴法线  | `[3, H, W]` |
| `rendered_ref_tint`                  | 反射色调        | `[3, H, W]` |
| `rendered_ref_roughness`             | 反射粗糙度      | `[1, H, W]` |
| `rendered_ref_strength`              | 反射强度        | `[1, H, W]` |
| `rendered_base_color`                | PBR 基础颜色    | `[3, H, W]` |
| `rendered_roughness`                 | PBR 粗糙度      | `[1, H, W]` |
| `rendered_metallic`                  | PBR 金属度      | `[1, H, W]` |
| `rendered_incident_lights`           | 入射光照 SH     | `[3, H, W]` |

`forward_shading` 模式下（`render.py:210-214`），辐射 + 反射的融合也在高斯级别（逐顶点）完成，本质仍是每个高斯独立着色，然后通过 alpha blending 合成。

---

### Phase 2: Deferred 反射高光 (Equation 1 in paper)

关闭 `forward_shading` 时，反射部分走**延迟着色**，在 `render.py:310-313`：

```python
# 延迟：在屏幕空间逐像素计算反射颜色
refl_color = get_reflectance_color(
    refmap,           # 预滤波的 cubemap 环境贴图
    normal_map,       # G-buffer 法线
    view_dirs,        # 屏幕空间视线方向
    ref_roughness_map, # G-buffer 粗糙度
    ref_tint_map,     # G-buffer 反射色调
    brdf_lut=dict_params["brdf_lut"]
)
# Split-sum 融合：radiance (低频) + reflection (高频)
ref_rgb = (1.0 - ref_strength_map) * radiance_map + ref_strength_map * refl_color
```

调用 `get_reflectance_color`（`shade.py:208-251`）实现的是 **Split-sum 近似**：

1. **反射方向计算**：`ref_dirs = 2 * dot(n, v) * n - v`（标准反射向量）
2. **BRDF 查找**：使用预计算的 BRDF LUT（2D 纹理），通过 `fv_uv = (NoV, roughness)` 查找 `F` 和 `G` 项
3. **环境贴图采样**：用粗糙度选择 MIP level 从 cubic map 中采样
4. **合并**：`spec * (spec_col * fg[0] + fg[1])` —— 这是 split-sum 的核心公式

---

### Phase 3: Deferred PBR 着色

当 `pc.use_pbr=True` 时，在 `render.py:329-401` 执行完整的延迟 PBR 着色：

```python
pbr_result = pbr_shading(
    light=cubemap,       # 可学习的环境贴图
    normals=normal_map,  # G-buffer
    view_dirs=view_dirs,
    albedo=base_color_map,  # G-buffer
    roughness=roughness_map,  # G-buffer
    metallic=metallic_map,
    occlusion=occlusion_map,  # 从遮挡体素插值（可选）
    irradiance=incident_light_map,  # G-buffer 入射光
    brdf_lut=dict_params["brdf_lut"],
)
```

`pbr_shading`（`shade.py:255-360`）是完整的 **Cook-Torrance BRDF** 延迟着色：

- **漫反射**：`diffuse_rgb = kd * diffuse_light * albedo`
  - `diffuse_light` 从环境贴图采样（在法线方向）
  - 被遮挡的区域用 `incident_light`（PRT 预计算）补偿：`(1-occlusion) * irradiance`
- **镜面反射**：`specular_rgb = spec * reflectance`
  - 也是 split-sum 框架：预滤波环境贴图 + BRDF LUT
  - `F0` 根据金属度混合：`F0 = (1-metallic) * 0.04 + albedo * metallic`
- **最终**：`render_rgb = diffuse_rgb + specular_rgb`
- **输出分解**：同时返回 `diffuse_rgb` 和 `specular_rgb` 以便可视化

---

### 为什么选择 Hybrid Forward + Deferred？

从 paper 和代码可以总结出 RTR-GS 的混合架构决策：

| 阶段                    | 实现方式                       | 为什么                                                    |
| ----------------------- | ------------------------------ | --------------------------------------------------------- |
| **PRT 辐射 (低频)**     | Forward (逐顶点 → alpha blend) | 需要跨高斯共享全局分量（SH 光照 + MLP），forward 自然     |
| **反射高光 (高频)**     | Deferred (逐像素)              | 保持 BRDF 锐度，split-sum 需要像素连续的 normal/roughness |
| **PBR 材质 (光照分解)** | Deferred (逐像素)              | Cook-Torrance BRDF 需要完整 G-buffer 输入                 |

关键洞察：**3D Gaussian 的 alpha blending 本质上是 weighted sum，会模糊高频信号**。如果把高光反射和 PBR 材质放进去 forward 走逐顶点 → alpha blend，反射高光的锐利边缘会被模糊掉。延迟渲染在 alpha-normalized 后的 G-buffer 上逐像素计算，**完全避免了这个问题**。

`render.py:269-271` 中也明确注释了这是"FOR DEFFERED SHADING"：

```python
# FOR DEFFERED SHADING
mask = num_contrib > 0
rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask
```

这一步**alpha normalization**将光栅化的 feature 从 $`\sum w_i f_i`$ 转换为 $`\sum w_i f_i / \sum w_i`$，得到正确的 G-buffer 值。

---

### 可视化输出分解

在非训练模式下（`render.py:419-468`），你能看到完整的延迟渲染分解：

| 可视化项                                 | 来源                      |
| ---------------------------------------- | ------------------------- |
| `radiance_color`                         | 前向 PRT 辐射             |
| `ref_color`                              | 延迟反射着色              |
| `blended_radiance` / `blended_ref_color` | 融合前的分量              |
| `diffuse_pbr` / `specular_pbr`           | PBR 的漫反射/镜面反射分离 |
| `image_pbr`                              | 完整 PBR 结果             |
| `incidents_light`                        | 遮挡补偿环境光照          |

最终输出就是这两大延迟着色分支的混合：`ref_rgb`（辐射+反射）和 `rendered_pbr`（完整 PBR）各自独立计算，再分别用于 Stage 1 和 Stage 2 的损失监督。