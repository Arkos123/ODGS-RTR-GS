## 什么是Split-Sum Approximation？

**Split-Sum Approximation** 是由Epic Games的Brian Karis在2013年提出的一种**加速基于图像的光照（Image-Based Lighting, IBL）渲染**的技术，最初用于Unreal Engine的PBR渲染管线。

### 核心思想

在PBR渲染中，镜面反射的渲染方程是一个复杂的积分：

$$L_s(x, o) = \int_{\Omega} L_i(x, i) f_s(i, o) (n \cdot i) di$$

这个积分需要同时对**入射光方向**和**BRDF**进行积分，计算量非常大。

Split-Sum的核心创新是：**将这个积分拆分成两个独立的项相乘**：

$$L_s(x, o) \approx \underbrace{\int_{\Omega} f_s(i, o) (n \cdot i) di}_{\text{BRDF项}} \times \underbrace{\int_{\Omega} L_i(x, i) D(i, o) (n \cdot i) di}_{\text{环境光照项}}$$

- **第一项（BRDF项）**：只与材质属性（粗糙度、法线与视线夹角）有关，可以**预计算成2D LUT表**
- **第二项（环境光照项）**：只与环境贴图和法线分布函数有关，可以**预滤波成Mipmap Cubemap**

这样运行时只需要两次纹理采样相乘即可，大幅提升了性能。

---

## 该项目中如何使用Split-Sum

在RTR-GS中，Split-Sum被用于**反射贴图（Reflection Map）的高频镜面反射计算**。

### 1. 公式定义

根据 [CLAUDE.md](file:///home/huangpengyue/projects/RTR-GS/CLAUDE.md#L34) 和 [paper/full.md](file:///home/huangpengyue/projects/RTR-GS/paper/full.md#L233)：

```
C_ref = R_t · F_ref(E_r, R_r, n, v)  (split-sum approximation)
```

其中：
- `R_t`: 反射强度（per-Gaussian属性）
- `E_r`: 可学习的反射贴图（CubemapLight）
- `R_r`: 反射粗糙度
- `n`: 法线
- `v`: 视线方向
- `F_ref`: Split-Sum近似函数

### 2. 代码实现

核心函数是 [pbr/shade.py:L208-251](file:///home/huangpengyue/projects/RTR-GS/pbr/shade.py#L208-L251) 的 `get_reflectance_color`：

```python
def get_reflectance_color(light, normals, view_dirs, roughness, specular_color, brdf_lut):
    # 1. 计算反射方向
    ref_dirs = 2*(n·v)n - v
    
    # 2. 查询BRDF LUT（Split-Sum第一项）
    fg_lookup = dr.texture(brdf_lut, fg_uv)  # 预计算的2D LUT
    
    # 3. 从mipmap cubemap采样环境光（Split-Sum第二项）
    miplevel = light.get_mip(roughness)  # 根据粗糙度选择mipmap层级
    spec = dr.texture(light.specular, ref_dirs, mip_level_bias=miplevel)
    
    # 4. 合并两项
    reflectance = spec_col * fg_lookup[..., 0:1] + fg_lookup[..., 1:2]
    return spec * reflectance
```

### 3. 在渲染管线中的位置

根据[doc/RTR-GS/ref_map介绍.md](file:///home/huangpengyue/projects/RTR-GS/doc/RTR-GS/ref_map介绍.md#L57)，Split-Sum用于**混合渲染分支的延迟反射计算**：

1. **前向渲染**：先渲染出法线、粗糙度、反射强度等逐像素属性
2. **延迟反射**：用这些属性通过Split-Sum从`refmap`采样得到高频反射颜色
3. **最终混合**：`I_rgb = C_radiance × (1 - R_i) + C_reflection × R_i`

### 4. 为什么用Split-Sum而不是直接积分？

根据论文和文档：
- **性能**：实时渲染需要高效计算，预计算LUT+Mipmap可以在运行时只做纹理采样
- **高频细节保留**：PRT（预计算辐射传输）适合低频漫反射，但无法处理高频镜面反射，Split-Sum通过延迟渲染保留了BRDF的锐利度
- **避免过拟合**：如果直接用SH表示高频反射，会出现漂浮伪影（floating artifacts）

---

**总结**：Split-Sum是RTR-GS混合渲染模型的关键组件，它将复杂的镜面反射积分拆分为BRDF LUT和预滤波环境贴图两项，实现了高效且高质量的高频反射渲染。