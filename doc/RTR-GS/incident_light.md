**代码中的 `incident_light` 对应论文公式 7 中的 $L_{ind}(\mathbf{x})$**。(学习到的间接漫反射光照。保存在球谐函数中)

逐行对照代码和数据流：

---

## Paper 公式 7

$$L_d = \frac{\mathbf{c}}{\pi} \big[ \underbrace{V}_{遮挡} \cdot \underbrace{L_d^{dir}}_{cubemap漫反射预积分} + \underbrace{(1-V)}_{1-遮挡} \cdot \underbrace{L_{ind}}_{间接光照} \big]$$

## Code 实现

[shade.py:L299-L321](file:///home/huangpengyue/projects/RTR-GS/pbr/shade.py#L299-L321)

```python
# ───── L_d^{dir} ───── 来自 cubemap 的漫反射预积分
diffuse_light = dr.texture(light.diffuse, normals)          # = L_d^{dir}

# ───── V × L_d^{dir} ───── 可见性调制
diffuse_light = diffuse_light * occlusion                    # = V · L_d^{dir}

# ───── (1-V) × incident_light_map ───── 加间接光照
#                    ↓ 传入的 incident_light_map 就是 L_ind
diffuse_light = diffuse_light + (1.0 - occlusion) * irradiance
#            = V · L_d^{dir}  +  (1-V) · L_ind               ← 完全对应公式7

# ───── 最终漫反射 ─────
diffuse_rgb = kd * diffuse_light * albedo                    # = c/π · [V·L_d^{dir} + (1-V)·L_ind]
```

所以各符号对应关系：

| 符号 | 代码中的来源 | 含义 |
|---|---|---|
| $L_d^{dir}$ | `light.diffuse` — cubemap 预积分的漫反射贴图 | 直接环境光照 |
| $L_{ind}$ | **`incident_light_map`** = 从每个 Gaussian 的 `incidents` SH 系数沿法线渲染得到 | **间接漫反射光照** |
| $V$ | `occlusion` — 从体素格中烘焙的可见性 | 可见性/遮挡 |
| $\frac{\mathbf{c}}{\pi}$ | `albedo` | 漫反射 base color |

**一句话：`incident_light` 是 $L_{ind}$，即每个 Gaussian 学习的局部间接光照，通过 `(1-V) × L_ind` 的方式补偿被遮挡区域的间接光。**

---

## 存储形式

每个 Gaussian 有自己的一组球谐系数（SH coefficients），专门用于表示该位置接收到的入射光：

[gaussian_model.py:L269-L273](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py#L269-L273)
```python
@property
def get_incidents(self):
    incidents_dc = self._incidents_dc     # shape [N, 1, 3]   DC分量
    incidents_rest = self._incidents_rest  # shape [N, 15, 3]  高阶分量
    return torch.cat((incidents_dc, incidents_rest), dim=1)  # [N, 16, 3]
```

每个 Gaussian 存了 3 通道 × 16 个 SH 系数（degree 3），总维度 `16×3=48`。

---

## 渲染时的用途

在 [render.py:L236-L237](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L236-L237)，每一个 Gaussian 的 SH 入射光系数会**沿着该 Gaussian 的法线方向求值**，得到一个 3 通道的 `incidents_light`：

```python
incidents = pc.get_incidents  # 每个Gaussian的入射光SH系数
incidents_light = eval_sh(pc.active_sh_degree, incidents, normal)
# ↑ 沿法线方向求值，得到 [N, 3] 的 RGB 入射光强
```

然后这组 `incidents_light` 值和 base_color、roughness、metallic 一起被拼接到特征向量中，通过 CUDA 光栅化器渲染到屏幕上，得到逐像素的 `incident_light_map` [H, W, 3]。

最终在 `pbr_shading` 中，[shade.py:L307-L308](file:///home/huangpengyue/projects/RTR-GS/pbr/shade.py#L307-L308) 它作为**额外的漫反射照明项**被加到 PBR 结果中：

```python
# 漫反射 = cubemap_diffuse(主光照) + (1-occlusion) × irradiance(局部入射光)
diffuse_light = diffuse_light + (1.0 - occlusion) * irradiance
```

---

## 直观理解

| 概念 | 类比 |
|---|---|
| **cubemap** | 房间天花板上的大灯 → 全局光照方向 |
| **incident_light** | 每个 Gaussian 位置的小夜灯 → 局部环境光 |

`incident_light` 捕捉的是**场景中不同空间位置接收到的额外环境光差异**：

- 桌子底下的 Gaussian：incident_light ≈ 很暗（遮挡多）
- 桌面上的 Gaussian：incident_light ≈ 较亮（遮挡少）
- 墙角处的 Gaussian：incident_light ≈ 中等（间接反射）

---

## 为什么 relighting 时它会被清零

在 viewer 和 eval_relighting 脚本中，当使用新的 HDR envmap 进行重光照时，[render.py:L246-L247](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L246-L247) 会清零 `incident_light`：

```python
if pipe.transfer_light:
    incidents_light = ...  # 用传输光照（保留空间信息）
else:
    incidents_light = torch.zeros_like(base_color)  # 清零
```

这是因为**原来的 `incident_light` 是在旧光照下学到的**，直接用在新的 envmap 上会不正确。但这也意味着新 envmap 的 PBR 渲染完全依赖 cubemap 的全局光照，丢失了每个位置原先捕捉到的局部光照差异——这是目前重光照的一个近似处理。开启 `--transfer_light` 可以缓解这个问题（通过光照传输保留空间分布）。