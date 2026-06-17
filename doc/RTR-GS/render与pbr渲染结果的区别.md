训练时**两个都用**，但用途不同。看训练循环：

```python
# 一次渲染同时得到两个输出
render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                       opt=opt, is_training=True, dict_params=pbr_kwargs)

# calculate_loss() 内部同时计算两个损失：
#   1. render 的 L1 + SSIM  → 优化 PRT属性 + 几何 + 反射属性
#   2. pbr 的 L1 + SSIM    → 优化 BRDF属性(base_color, roughness, metallic) + PBR cubemap
#   3. 两者合并：loss = loss_hybrid + lambda_pbr * loss_pbr
loss += render_pkg["loss"]
loss.backward()
```

论文 3.4 节明确解释了为什么要两个分支**同时**使用：

> we use both the previously mentioned hybrid rendering and PBR branches simultaneously, rather than freezing the geometric parameters or enabling only the PBR branch. This approach is adopted for two main reasons.
>
> 1) Different rendering models still require corresponding geometric adjustments for proper adaptation, so completely freezing the geometric parameters is undesirable.
> 2) Since the PBR-related parameters are initialized randomly, using only PBR can easily lead to drastic changes in the geometric structure.

## 所以各自干嘛的

| 分支 | 输出 | 优化目标 | 用途 |
|---|---|---|---|
| **Hybrid (`render`)** | PRT辐射度 + 反射混合 | 几何重建、反射属性 (ref_tint, roughness, strength)、PRT参数 | 主力重建几何，处理任意反射率 |
| **PBR (`pbr`)** | BRDF渲染 | 材质分解 (albedo, metallic, roughness)、PBR光照cubemap | 解耦出物理材质和光照 |

两者共享几何（位置、缩放、旋转、法线），但各自优化不同的外观参数。`render` 负责把几何和反射重建好，`pbr` 负责把材质和光照分解出来。**没有 PBR 分支，就分解不出 albedo、roughness、metallic 这些物理材质属性**。