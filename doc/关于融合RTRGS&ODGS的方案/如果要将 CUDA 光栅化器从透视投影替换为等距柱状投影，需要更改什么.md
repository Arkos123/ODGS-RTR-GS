基于 ODGS 和 RTR-GS 光栅化器的对比分析，以下是所需变更的分层描述：

#### 5.1 光栅化器接口层

**当前 RTR-GS 光栅化器（`rtr_gs_rasterization`）vs ODGS 光栅化器（`odgs_gaussian_rasterization`）：**

| 方面 | RTR-GS（透视） | ODGS（等距柱状） |
|------|----------------|-------------------|
| 投影参数 | `tanfovx`, `tanfovy`, `cx`, `cy`, `projmatrix` | 无 `tanfovx/tanfovy`，无 `projmatrix` |
| 设置参数 | 17 个字段 | 仅 8 个字段 |
| forward 输出 | 10 个返回值（含 `feature`, `normal`, `surface_xyz`, `weights`） | 7 个返回值（`color,depth,acc,radii,psi,lat,lon`），**无 `feature`/`normal`/`surface_xyz`/`weights`** |
| `markVisible` | 使用 `viewmatrix + projmatrix` 做锥体裁剪 | 使用 `viewmatrix` 做球壳裁剪 |
| 关键特性 | 支持 `computer_pseudo_normal`，支持 `features` 张量的 alpha blending | 支持 `psi/lat/lon` 等距柱状坐标 |

**所需变更：**

**A) [render.py L119-L136] `GaussianRasterizationSettings` 必须大幅简化：**
- 移除 `tanfovx`, `tanfovy`, `cx`, `cy`, `projmatrix`
- 移除 `computer_pseudo_normal`（ODGS 不支持）
- 添加任何 ODGS 特有的字段（当前 ODGS 没有额外字段，但需要检查）

**B) [render.py L253-L264] 光栅化器调用签名必须适配：**
```python
# 当前（RTR-GS 透视版本）：
(num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
 rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(...)

# 改为（ODGS 等距柱状版本）：
(color, depth, acc, radii, psi, lat, lon) = rasterizer(...)
```

#### 5.2 features 张量机制 — 最核心的变更

**这是最关键的问题。** RTR-GS 的 deferred shading 依赖于 `features` 张量通过 CUDA 光栅化器的逐通道 alpha blending 传播到像素级别 [L203, L249, L268-L278]。然而，**ODGS 的光栅化器根本不支持 `features` 参数**。

ODGS 的光栅化器签名 (`odgs_gaussian_rasterization/__init__.py`) 只有：
```python
def rasterize_gaussians(means3D, means2D, sh, colors_precomp, opacities, scales, rotations, cov3Ds_precomp, raster_settings):
```
没有 `features` 参数。CUDA 内核中也没有对额外特征通道做 alpha blending 的逻辑。

**解决方案有两个方向：**

**方案 A（推荐，但工作量大）：扩展 ODGS 的 CUDA 光栅化器**
- 修改 ODGS 的 CUDA 内核，添加 `features` 张量的支持
- 让 CUDA kernel 对每个特征通道做加权 alpha blending
- 在 forward 输出中添加 `rendered_feature`
- 实现 backward 中对 `features` 的梯度传导

**方案 B（**仅使用 forward shading**）：只在 Python 中完成所有颜色计算**
- 在 `render.py` 中强制使用 forward shading（`colors_precomp` 包含最终颜色）
- 完全移除对 `rendered_feature` 的依赖
- 问题的后果：无法做 deferred PBR shading，但可以保持 basic reflection（因为 forward shading 已经在 L207-L211 实现了）

#### 5.3 其他需要修改的模块

**C) [render.py L140-L148] 法线计算**
当前 `normal = pc.get_min_axis(viewpoint_camera.camera_center)` 使用最短轴方向作为法线。这个方法与投影模型无关，**可以保留**。但 `computer_pseudo_normal=True` 需要移除。

**D) [render.py L153-L155] 深度计算**
当前通过 `xyz_homo @ world_view_transform` 计算相机空间深度，**与投影无关**，可以保留。

**E) [render.py L100-L136] 相机参数访问**
需要确保 `Camera` 对象在等距柱状投影模式下仍能提供有效的 `world_view_transform`、`camera_center`、`c2w` 等。ODGS 的 `Camera` 类已经提供了这些。

**F) 前向 Shading 路径的调整 [L207-L211]**
如果采用方案 B（forward shading only），需要强制 `pipe.forward_shading = True`，使得所有颜色计算在光栅化前完成。但 `get_reflectance_color_forward` 需要每个高斯的法线、视角方向等，这些都已经在 Python 端可用。

#### 5.4 影响范围总表

| 文件 | 行号 | 需要修改的内容 | 重要性 |
|------|------|---------------|--------|
| `render.py` | L119-L136 | `GaussianRasterizationSettings` 移除透视参数 | **必须** |
| `render.py` | L253-L264 | 光栅化器调用签名适配 | **必须** |
| `render.py` | L267-L278 | `rendered_feature` 拆分逻辑 | **必须**（方案 A 需适配维度；方案 B 需删除）|
| `render.py` | L281-L284 | PBR feature 拆分 | **方案 A 必须；方案 B 删除** |
| `render.py` | L287-L313 | Deferred 反射渲染 | **方案 B 不需要；方案 A 保留** |
| `render.py` | L326-L394 | PBR deferred shading | **方案 A 保留；方案 B 移除** |
| `rtr_gs_rasterization.py` | 整个文件 | 替换为 ODGS 或扩展版 | **必须** |
| CUDA `.cu` 文件 | — | 添加 `features` 支持（方案 A） | **方案 A 必须** |
| `scene/cameras.py` | L63-L81 | 确保等距柱状投影矩阵 | **可能需要** |
| `render_fast.py` | 整个文件 | 做同样的适配 | 同步修改 |

#### 5.5 整体策略建议

**推荐路径：方案 A（扩展 ODGS CUDA 光栅化器）**

步骤：
1. 在 ODGS 的 CUDA 内核中添加 `features` 张量的 alpha blending 支持（参考 3DGS 代码中 `feature` 的处理方式）
2. 修改 `GaussianRasterizationSettings` 移除透视相机参数，添加必要的等距柱状投影参数
3. 适配 `rtr_gs_rasterization.py` 的 Python 绑定层
4. 修改 `render.py` 中 `rendered_feature` 的拆分逻辑（维度保持不变，因为 features 内容与投影模型无关）
5. 保留完整的 deferred shading 管线（反射 + PBR）
6. `render_fast.py` 同步修改
        