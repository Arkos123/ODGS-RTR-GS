# Equirect 等距空间 → COLMAP 世界空间坐标系转换修复

## 问题

在 equirect 渲染模式下，`pseudo_normal`（depth-derived）、`normal_facing`（法线可见性热力图）和 PBR `view_dirs` 的颜色/值存在系统性错误：

| 视觉输出 | 错误表现 |
|---|---|
| `pseudo_normal` | 水平面（地板/天花板）颜色与 `normal`（高斯最短轴）正好相反（Y 轴反转） |
| `normal_facing` | 天花板/地板等水平面全部显示为红色（背向），违反"朝向相机应为蓝色"的预期 |
| PBR `view_dirs` | 视线方向 Y 分量错误，间接影响 PBR 镜面反射 |

## 根因

`_equirect_ray_dirs()` 和 `_erp_depth_to_normal()` 从等距经纬度网格构造射线：

```python
# render_equirect.py
ys = torch.linspace(0.5 * math.pi, -0.5 * math.pi, H)  # 顶部→底部
xs = torch.linspace(-math.pi, math.pi, W)                # 左→右
lat, lon = torch.meshgrid(ys, xs, indexing='ij')
rays = torch.stack([
    torch.sin(lon) * torch.cos(lat),   # +X 右
    torch.sin(lat),                     # +Y **上** ← 关键差异
    torch.cos(lon) * torch.cos(lat),    # +Z 前
], dim=-1)
```

在此坐标系中：
- **`+Y = 上方向`**（lat=+π/2 图像顶部 → `sin(lat)=+1`）
- **`+Z = 前方向`**（镜头中心）

但整个代码库（SGS/RTR-GS）使用 **COLMAP 约定**作为世界空间：
- **`+Y = 下方向`**（COLMAP 相机模型的 Y 轴朝下）
- **`+Z = 前方向`**

因此等距空间和 COLMAP 世界空间的 Y 轴完全相反。而 RTR-GS 透视版的 CUDA rasterizer 在计算伪法线时，最后用 `C2W @ viewspace_normal` 转到了世界空间（见 `rtr_gs-rasterization/forward.cu:488-490`），所以透视版是正确的。

等距版 Python 代码的所有 `_equirect_ray_dirs` / `_erp_depth_to_normal` 输出在被 `c2w` / `C2W` 矩阵转换前，都缺少了这个 Y 轴翻转。

## 修复

在**所有将等距空间向量乘以 C2W 的地方，先做 Y 轴翻转**：

```
n_world = C2W[:3,:3] @ diag(1, -1, 1) @ n_equirect
```

| 位置 | 用途 | 修改内容 |
|---|---|---|
| `render_equirect.py:315` 后 | `pseudo_normal` 坐标转换 | Y-flip + C2W 旋转到世界空间 |
| `render_equirect.py:354` | `normal_facing` cam_to_point | rays 先 Y-flip 再乘 C2W^T |
| `render_equirect.py:463` | PBR `view_dirs` | rays 先 Y-flip 再乘 C2W^T |

### 涉及的等距 → 世界空间转换模式

代码中有 3 个不同的向量需要从等距空间转换到世界空间：

```python
# 1. 法向量：等距空间 n → 世界空间 n
n_equirect  → Y-flip → n_view(COLMAP)  → C2W @ n_view → n_world

# 2. 射线：等距空间 r → 世界空间 r_direction
r_equirect  → Y-flip → r_view(COLMAP)  → r_view @ C2W^T → r_world

# 3. 视线方向（PBR）：r_world → -r_world（从表面指向相机）
```

## 透视版 vs 等距版对比

| | 透视版 `render.py` + `rtr_gs-rasterization` | 等距版 `render_equirect.py` + SGS rasterizer |
|---|---|---|
| `pseudo_normal` 来源 | CUDA `renderPseudoNormalCUDA` | Python `_erp_depth_to_normal` |
| 法线计算空间 | camera view space | equirect camera-centric space (+Y up) |
| → 世界空间 | CUDA 内 `C2W @ n` (`forward.cu:488-490`) ✅ | Python 曾**缺少** `C2W @ Y_flip @ n` ❌→✅ |
| `normal_facing` 视线 | CUDA 用 viewmatrix 计算 | Python `_equirect_ray_dirs @ C2W^T` 曾缺少 Y-flip ❌→✅ |
| `view_dirs` (PBR) | 透视 canoncial rays | 同上 ❌→✅ |

## 验证方法

1. 运行 `train.py -t render_ref_equirect` 迭代 0，对比 `normal` 和 `pseudo_normal` 可视化：
   - 地板：两者应均为紫红色（`(0.5, 0.0, 0.5)`，法线指向 -Y 方向）
   - 天花板：两者应均为绿色（`(0.5, 1.0, 0.5)`，法线指向 +Y 方向）
2. 检查 `normal_facing` 可视化：
   - 所有可见面应为蓝色（法线朝向相机）
   - 仅物体轮廓/边缘可能出现红色（silhouette 处法线突变）

## 相关文件

- `gaussian_renderer/render_equirect.py` — 修复文件
- `submodules/rtr_gs-rasterization/cuda_rasterizer/forward.cu:488-490` — 透视版正确实现参考
