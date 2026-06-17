# compute_with_prt：预计算辐射传输（PRT）

## 一句话总结

`compute_with_prt` 控制渲染时使用 **PRT（Precomputed Radiance Transfer，预计算辐射传输）** 还是标准 **SH（Spherical Harmonics，球谐函数）** 来计算每个 Gaussian 的出射辐射度。模型训练时必须开启，viewer 渲染时必须与训练一致，否则结果异常。

## 为什么需要 PRT

来自论文第 3.3 节：

> Compared to spherical harmonics, radiance transfer ... provides stronger global low-frequency constraints.
>
> In the shading process, all Gaussians share two global components: the spherical harmonics lighting $c_j$ and the MLP $G$. This design enables shading across Gaussians to be connected through shared components, promoting the representation of overall low-frequency variations.

PRT 的核心优势：
- **共享全局光照**：所有 Gaussian 共享同一组 SH 光照系数和同一个 MLP 网络，提供强低频约束
- **防止高频过拟合**：SH 在每个 Gaussian 上独立，容易过拟合高频细节导致 floating artifacts
- **更好的几何平滑性**：低频约束有助于保持几何结构的平滑

## PRT 在代码中的实现

### 总入口

[render.py:L158-L165](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L158-L165)

```python
if pipe.compute_with_prt and override_color is None:
    net = dict_params["transfer_net"]
    viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
    if only_diffuse:
        prt_color = PRTutils.cal_diffuse(pc)
    else:
        prt_color = PRTutils.cal_color(pc, net, viewdirs, normal, is_training)
    override_color = prt_color
```

当 `compute_with_prt=True` 时，`override_color` 被设置为 PRT 计算的颜色，后续光栅化会直接使用这个颜色（跳过 SH 系数）：
```python
colors_precomp = override_color  # 使用 PRT 颜色，不走 SH
```

当 `compute_with_prt=False` 时，颜色回退到标准 SH 求值：
```python
shs = pc.get_shs  # 原始 SH 系数（未针对 PRT 训练充分优化）
```

### PRT 计算的两个分量

[prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py)

#### 漫反射分量（view-independent）

```python
@staticmethod
def cal_diffuse(gaussian, mask=None):
    # C_d = diffuse_tint · transport
    # 其中 transport = relu( transfer · light + 0.5 )
    
    diffuse_tint = gaussian.get_diffuse_tint          # 漫反射色调
    shs_direct_light = ...                             # 全局 SH 光照
    shs_diffust_transfer = gaussian.get_diffuse_transfer  # 每个Gaussian的漫反射传输向量
    transport = relu((transfer · light).sum(-1) + 0.5)
    cd = diffuse_tint * transport
    return cd
```

对应论文公式 5：
$$C_d \approx \rho_d \sum_{j=0}^{n^2} c_j c_j^t$$

其中 $c_j$ 是全局共享的 SH 光照，$c_j^t$ 是每个 Gaussian 各自的传输向量。

#### 镜面反射分量（view-dependent）

```python
@staticmethod
def cal_specular(gaussian, net, dir, normal, mask=None):
    # 1. 计算反射方向
    reflect_dir = 2.0 * (normal · view_dir) * normal - view_dir
    
    # 2. MLP 解码神经辐射传输向量
    LT_coeff = cal_spec_coff(gaussian, net, reflect_dir)
    
    # 3. 传输向量 · 光照 = 反射辐射度
    direct_color = relu((LT_coeff × light).sum(-1))
    cs = specular_tint * direct_color
    return cs
```

对应论文公式 6：
$$C_s(o) \approx \rho_s \sum_{j=0}^{n^2} c_j c_j^t(o)$$

其中 $c_j^t(o) = G(f_t, o)$ 由 MLP 解码。

### MLP 架构

[transfer_mlp.py](file:///home/huangpengyue/projects/RTR-GS/scene/transfer_mlp.py)

```python
class TransferMLP:
    # 3 层 MLP，64 隐藏单元
    net = [
        Linear(3 → 64) + ReLU,       # 输入：反射方向
        Concat + Linear(64+feat → 64) + ReLU,  # 拼接 specular feature
        Linear(64 → 16)              # 输出：SH 传输系数
    ]
```

输入是反射方向 + 每个 Gaussian 的 specular feature，输出 16 个 SH 传输系数（degree 3）。

### 训练流程中的 PRT

两个阶段都用 `--compute_with_prt`：

```
Stage 1:  python train.py ... --ref_map -t render_ref --compute_with_prt
Stage 2:  python train.py ... --ref_map -t render_ref_pbr --compute_with_prt
```

在 Stage 1 中，PRT 计算完整的混合渲染颜色（`override_color`）。Stage 2 中 PRT 继续为混合渲染分支提供辐射度颜色，而 PBR 分支则使用额外的 BRDF 属性。

## 为什么 viewer 与训练不一致

如果模型是用 `--compute_with_prt` 训练的，**但 viewer 中关闭了它**：

1. `override_color` 保持 `None`
2. 渲染器回退到 `shs = pc.get_shs`（标准 3DGS SH 系数）
3. 这些 SH 系数在 PRT 训练过程中没有被充分优化——因为训练时 `override_color = prt_color` 直接覆盖了颜色，SH 系数没有收到有效的渲染损失梯度
4. 结果是：SH 系数可能包含噪声或不完整的颜色信息，渲染出奇怪的结果

**这就是 viewer 渲染和训练结果不一致的根本原因。**

## 相关命令和参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--compute_with_prt` | `False` | 启用 PRT 替代 SH |
| `--diffuse_iteration 3000` | `0` | 前 N 次迭代仅用漫反射 PRT |

在代码中对应 [PipelineParams](file:///home/huangpengyue/projects/RTR-GS/arguments/__init__.py#L69)：
```python
class PipelineParams:
    self.compute_with_prt = False
    self.diffuse_iteration = 0
```

## 关键代码定位

| 文件 | 行号 | 内容 |
|---|---|---|
| [render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py) | 158-165 | `compute_with_prt` 分支入口 |
| [render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py) | 186-196 | SH 回退路径 |
| [prt_utils.py](file:///home/huangpengyue/projects/RTR-GS/utils/prt_utils.py) | 1-63 | PRT 计算（diffuse + specular） |
| [transfer_mlp.py](file:///home/huangpengyue/projects/RTR-GS/scene/transfer_mlp.py) | 1-55 | 传输 MLP 网络 |
| [gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py) | 269-273 | `get_incidents` SH 入射光 |
| [arguments/\_\_init\_\_.py](file:///home/huangpengyue/projects/RTR-GS/arguments/__init__.py) | 69 | PipelineParams 定义 |

---
# 重光照到底要不要开

你说得对。我仔细追踪了 `render.py` 中 `rendered_image` 和 `rendered_pbr` 的数据流，**之前我的分析有误**。

## 数据流核实

```
rasterizer 输出
  ├── rendered_image     → radiance_map → results["render"]  (混合渲染)
  ├── rendered_feature   → base_color, roughness, metallic,
  │                        incidents_light 等 → pbr_shading → results["pbr"]  (PBR)
  ├── rendered_normal    → normal_map → pbr_shading
  └── rendered_opacity   → opacity_map
```

`rendered_image` 和 `rendered_feature` 是光栅化器**独立的输出通道**。`compute_with_prt` 只改变了 `rendered_image`（通过 `override_color` 替换 SH 颜色），**对 `rendered_feature` 没有任何影响**。

所以：

| 输出 | compute_with_prt 影响 | 重光照 viewer 显示 |
|---|---|---|
| `render` 键（混合渲染） | ✅ PRT vs SH | ❌ viewer 不用 |
| `pbr` 键（PBR 渲染） | **❌ 无影响** | ✅ viewer 用 |

## 那 viewer 奇怪的根本原因是什么？

重新审视 viewer 和训练 `render.png` 的差异：

1. **比较的对象不同**：训练保存的 `render.png` = `render`（混合渲染），viewer 显示的是 `pbr`
2. **`relight=True`** → `incidents_light = 0`，PBR 缺少间接光照

第一条才是主因。如果你把训练时的 `pbr.png`（[train.py:L350](file:///home/huangpengyue/projects/RTR-GS/train.py#L350) 保存）和 viewer 对比，应该就是一致的了。

**`compute_with_prt` 对于只看 PBR 的输出确实"没用"。** 我之前的说法是错的。