`ref_map` 是 RTR-GS 中一个 **布尔类型的管线配置参数**，控制是否启用 **反射贴图（Reflection Map）** 来处理高光（镜面）反射。下面详细解释它的完整含义和机制。

## 一、定义位置

定义在 [arguments/__init__.py](file:///home/huangpengyue/projects/RTR-GS/arguments/__init__.py#L74) 的 `PipelineParams` 中，默认值 `False`：

```python
self.ref_map = False
```

命令行传入方式：`--ref_map`

## 二、它的本质是什么

当 `--ref_map` 为 `True` 时，系统会创建一个 **额外的** **`CubemapLight`** **实例**（名为 `refmap`），专门用来表示 **场景的高频反射环境光照**。

在 [train.py:L106-119](file:///home/huangpengyue/projects/RTR-GS/train.py#L106-L119) 中可以看到：

```python
if pipe.ref_map:
    refmap = CubemapLight(base_res=128).cuda()   # 创建一个Cubemap
    refmap.train()
    # ... 尝试从checkpoint加载
    refmap.training_setup(opt, light_type="ref")
    pbr_kwargs["refmap"] = refmap                # 传入渲染器
```

## 三、它和另一个 `cubemap` 有什么区别？

这里 **有两个** `CubemapLight`，角色完全不同：

| 组件            | 何时创建                                   | 用途                                    |
| ------------- | -------------------------------------- | ------------------------------------- |
| **`cubemap`** | `is_pbr=True` 时（即 `-t render_ref_pbr`） | PBR 分支的 **环境光照**，用于 BRDF 材质分解（漫反射+镜面） |
| **`refmap`**  | `--ref_map` 为 True 时                   | 混合渲染分支中的 **反射图**，专门用于计算高频镜面反射颜色       |

它们可以同时存在，比如 Stage 2 中两者都启用。

## 四、在渲染中的具体作用

在 [render.py:L290-305](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L290-L305) 中：

```python
if not pipe.forward_shading:
    refl_color = get_reflectance_color(
        refmap,                   # 传入refmap作为光照
        normal_map, 
        view_dirs, 
        ref_roughness_map, 
        ref_tint_map, 
        brdf_lut=dict_params["brdf_lut"]
    )
    # 混合：最终颜色 = (1 - 反射强度) × 漫反射 + 反射强度 × 反射颜色
    ref_rgb = (1.0 - ref_strength_map) * radiance_map + ref_strength_map * refl_color
```

**`get_reflectance_color`** 是 [pbr/shade.py:L208](file:///home/huangpengyue/projects/RTR-GS/pbr/shade.py#L208-L251) 中的核心函数，它做的事情就是 **基于 Split-Sum 近似的环境贴图反射**：

1. 根据法线和视线方向计算 **反射方向**：`ref_dirs = 2*(n·v)*n - v`
2. 用粗糙度查询 **BRDF LUT**（预先计算好的 BRDF 积分查找表）
3. 从 `refmap` 的 **mipmap 链**（预过滤的环境贴图）中采样对应的镜面反射颜色
4. 将采样颜色与 BRDF 项相乘得到最终反射颜色

这本质上是 **延迟渲染（deferred rendering）** 的思路——先通过前向渲染得到法线、粗糙度、反射强度等逐像素属性，再用这些属性从反射贴图中采样得到高频反射。

## 五、为什么需要 `--ref_map`？

这是 RTR-GS **混合渲染模型** 的核心设计理念——将场景外观拆解为两部分：

| 成分             | 方法       | 特点                   |
| -------------- | -------- | -------------------- |
| **低频辐照度（漫反射）** | PRT 前向渲染 | 共享网络，低频约束强，防止漂浮伪影    |
| **高频反射（镜面）**   | 反射贴图延迟渲染 | 保持高频细节，避免PRT分辨率不足的问题 |

最终的屏幕空间混合：

```
I_rgb = C_radiance × (1 - R_i) + C_reflection × R_i
```

如果不加 `--ref_map`，则退化为纯 PRT 前向渲染（即 `pipe.forward_shading = True` 的分支），无法处理高频镜面反射。

## 六、小结

**`--ref_map`** **就是一个开关，决定是否启用一个独立的** **`CubemapLight`** **作为反射贴图，配合延迟渲染管线来处理场景中的高光反射成分。** 它在 Stage 1（几何预训练）和 Stage 2（PBR 优化）中都需要开启，是所有实物数据集（MipNeRF 360、Stanford ORB、TensoIR 等）的标准配置。

# 和重光照的关系

> 重光照中开启，并设置 refmap=cubemap

你说得对。`ref_map` 提供的就是**环境的高频反射光**，和物体本身无关。数据流很清楚：

## refmap 是环境光，不是物体属性

[render.py:L90](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L90) — 从 dict\_params 取出

```python
refmap = dict_params["refmap"]
```

[shade.py:L208-L233](file:///home/huangpengyue/projects/RTR-GS/pbr/shade.py#L208-L233) — 在反射方向采样 cubemap

```python
def get_reflectance_color(light, normals, view_dirs, roughness, specular_color, brdf_lut):
    ref_dirs = 2*(n·v)n - v                      # 反射方向
    spec = dr.texture(light.specular, ref_dirs, ...)  # 采样 cubemap（split-sum）
    ...
    return spec * specular_color                  # × 反射色调（物体属性）
```

所以 `refmap` 的分量：

| 参与方                               | 例子          | 含义                |
| --------------------------------- | ----------- | ----------------- |
| **`refmap`（cubemap）**             | 新的 HDR 环境贴图 | **环境光**，重光照时可自由更换 |
| **`ref_tint`**（per Gaussian）      | 金属反射偏蓝/偏金   | **物体属性**，重光照时不变   |
| **`ref_roughness`**（per Gaussian） | 光滑/粗糙       | **物体属性**，重光照时不变   |
| **`ref_strength`**（per Gaussian）  | 反射强度        | **物体属性**，重光照时不变   |

## 重光照时需要 `ref_map` 吗？

需要，原因有两个：

**原因 1：不设置** **`ref_map=True`，完全没有反射**

[render\_and\_eval.py:L335-L347](file:///home/huangpengyue/projects/RTR-GS/render_and_eval.py#L335-L347) 中 `pipe.ref_map` 控制是否把 `refmap` 放入 dict\_params：

```python
if pipe.ref_map:
    pbr_kwargs["refmap"] = refmap
```

而 render.py 第 90 行直接取 `dict_params["refmap"]`，不传会 crash。即使设置了 `refmap=cubemap`，也要 `ref_map=True` 才能让 dict\_params 有这个 key。

**原因 2：反射是物体外观的重要组成部分**

即使 PBR 分支能独立计算镜面反射（通过 `cubemap` 的 specular 查询），混合渲染分支（`render` 键）中的反射和 PBR 分支共同优化。关闭 `ref_map` 训练时模型就没法学到反射属性（`ref_tint`、`ref_roughness`）。

## eval 脚本怎么做

所有三个重光照脚本都是 `--ref_map` + `refmap=cubemap`：

```python
# 用新 HDR 创建 cubemap
cubemap = CubemapLight(base_res=256)
cubemap.base.data = latlong_to_cubemap(hdri, [256, 256])

# cubemap 同时作为 PBR 环境光和反射贴图
dict_params = {
    "cubemap": cubemap,     # PBR 漫反射+镜面
    "refmap": cubemap,      # 混合渲染分支的反射
}
```

训练时两者是独立的（`cubemap_` vs `refmap_` checkpoint），重光照时都用同一个新 envmap。**`ref_map`** **提供的只是环境的光，不是物体本身的属性。**
