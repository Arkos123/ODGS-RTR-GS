# render_equirect.py PBR 损失项补充与 depth 归一化修复

## 背景

`gaussian_renderer/render_equirect.py` 是 RTR-GS 的等距柱状投影（equirectangular/360°）渲染模块，用于 SGS（Spherical Gaussian Splatting）训练的渲染和前向传播。

它与 `gaussian_renderer/render.py`（标准 pinhole 透视渲染器）共享相同的 `GaussianModel` 和 `Camera` 接口，但使用不同的 CUDA 栅格化器：
- `render.py` → `rtr_gs_rasterization`（RTR-GS 定制 pinhole 栅格化器，支持 feature tensor 和 单 pass 渲染）
- `render_equirect.py` → `spherical_gaussian_rasterization`（SGS 球形栅格化器，支持 equirect 投影，需多 pass 渲染）

## 问题

### 1. 缺失 PBR 平滑损失项

`render_equirect.py` 的 `calculate_loss()` 在 PBR 分支中缺少以下默认开启的损失项（这些项在 `render.py` 中存在，且有非零默认值）：

| 损失项 | 默认权重 | 来源（render.py） | 作用 |
|---|---|---|---|
| `lambda_base_color_smooth` | 0.03 | render.py:599-604 | 对反照率（base_color）做边缘感知平滑，防止反照率噪声 |
| `lambda_metallic_smooth` | 0.01 | render.py:613-618 | 对金属度（metallic）做边缘感知平滑 |
| `lambda_env_smooth` | 0.01 | render.py:621-625 | 对环境贴图做总变分（TV）平滑 |

由于这些权重的默认值均非零，使用 `-t render_ref_pbr_equirect` 训练时这些约束被静默跳过，导致材质分解结果（base_color、metallic）比标准训练更粗糙/噪声更多。

**注意**：`lambda_white_light` 和 `lambda_roughness_smooth` 在 `render_equirect.py` 中已有实现，不受影响。

### 2. depth 未作 alpha 归一化用于遮挡计算

SGS 球形栅格化器输出的 depth 是 alpha 合成的原始值：`Σ(T_i * α_i * z_i)`，而非归一化的表面深度 `Σ(T_i * α_i * z_i) / Σ(T_i * α_i)`。

在原始代码中，这个未归一化的 depth 直接用于遮挡体积计算的世界坐标重建：

```python
# 修复前：使用未归一化的 depth
points = (-view_dirs.reshape(-1, 3) * depth.reshape(-1, 1) + c2w[:3, 3])
```

这导致在物体边缘处（alpha 过渡区域），深度值偏小，计算出的 3D 坐标偏离真实表面，进而使遮挡阴影的边缘位置产生微小偏移。

## 修复内容

### 修复 1：添加缺失的损失项

在 `calculate_loss()` 的 `if pc.use_pbr:` 分支内，`lambda_roughness_smooth` 块之后，按 `render.py` 的相同模式添加：

```python
# base_color 边缘感知平滑
if opt.lambda_base_color_smooth > 0:
    image_mask = viewpoint_camera.image_mask.cuda()
    rendered_base_color = results.get("base_color")
    if rendered_base_color is not None:
        loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
        tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
        loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

# metallic 边缘感知平滑（同上模式）

# 环境贴图 TV 平滑
if opt.lambda_env_smooth > 0 and env_map is not None:
    env = env_map.get_env_map()
    loss_env_smooth = tv_loss(env.permute(2, 0, 1))
    tb_dict["loss_env_smooth"] = loss_env_smooth
    loss = loss + opt.lambda_env_smooth * loss_env_smooth
```

### 修复 2：depth alpha 归一化

在计算遮挡点云之前，将 depth 除以 opacity：

```python
surf_depth = depth / rendered_opacity.clamp_min(1e-5)
points = (-view_dirs.reshape(-1, 3) * surf_depth.reshape(-1, 1) + c2w[:3, 3])
```

`rendered_opacity` 已在栅格化第一个 pass 后计算得到，只有在该处之后才能使用。

## 参数字典

### 涉及的优化参数

所有参数定义在 `arguments/__init__.py`：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--lambda_base_color_smooth` | float | 0.03 | 反照率边缘感知平滑权重 |
| `--lambda_metallic_smooth` | float | 0.01 | 金属度边缘感知平滑权重 |
| `--lambda_env_smooth` | float | 0.01 | 环境贴图总变分平滑权重 |

### 等距柱状渲染类型选择

在 `train.py` 中通过 `-t` 参数选择渲染类型：

```bash
# 仅等距柱状几何+反射预训练（无 PBR）
-t render_ref_equirect

# 等距柱状 PBR 材质分解训练
-t render_ref_pbr_equirect
```

使用上述类型时，`train.py` 会自动设置 `forward_shading=True` 和 `equirect=True`。

## 验证方式

1. **语法检查**：`python -c "from gaussian_renderer.render_equirect import render_view, calculate_loss, render"`
2. **对比测试**：使用 `-t render_ref_pbr_equirect` 训同一场景，对比修复前后的 vis 图（base_color、metallic、roughness）的平滑度
3. **遮挡验证**：比较修复前后 occlusion_map 在物体边缘处的差异

## 相关文件

- `gaussian_renderer/render_equirect.py` — 等距柱状渲染器（修复对象）
- `gaussian_renderer/render.py` — 标准透视渲染器（参考实现）
- `gaussian_renderer/__init__.py` — 渲染函数注册表
- `arguments/__init__.py` — 所有损失权重的默认值定义

## 参考

- `render.py:599-604` — `lambda_base_color_smooth` 实现
- `render.py:613-618` — `lambda_metallic_smooth` 实现
- `render.py:621-625` — `lambda_env_smooth` 实现
- `render.py:343-349` — depth alpha 归一化参考实现
