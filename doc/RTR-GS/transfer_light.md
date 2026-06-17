论文里**没有提到它（疑似也从来没开启过）**。全文唯一出现 "incident" 的地方在第 3.5 节的光照正则化（行 259）：

> We apply a light regularization assuming a natural white incident light ...

以及参考文献中引用 NeILF（Neural Incident Light Field）。

`transfer_light` 是代码中额外实现的**后训练**功能，不涉及论文中描述的端到端训练过程。整个 `transfer_light` 的数据流（`incident_to_transfer + build_sh + light_shs * transfer_shs`）是一个**推理时的重光照增强技巧**，不是论文核心方法的组成部分。

这也能解释为什么 `transfer_light` 默认是 `False`——它只是一个可选的推理后处理，不是标准流程的一部分。

***

`transfer_light` 是一种**保留间接光空间分布的重光照技术**。让我用完整的数据流来解释。

## 问题背景

重光照时，如果直接清零 `incidents_light`：

```python
# relight=True, transfer_light=False
incidents_light = torch.zeros_like(base_color)  # → PBR 失去所有局部间接光
```

这样每个点的漫反射光照完全来自新 cubemap，场景中原来的阴影和间接光分布全部丢失。

***

## transfer\_light 的核心思路

它在训练好的 `incidents`（旧光照下的间接光）和新 cubemap 之间做一个**光照传输**：

### 步骤 1：训练完成后，分解 incidents

[gaussian\_model.py:L1131-L1136](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py#L1131-L1136)

```python
def incident_to_transfer(self, light_shs):
    # incidents = light_old × transfer
    # 所以：transfer = incidents / light_old
    out_shs = self.get_incidents          # [N, 16, 3]  学到的间接光
    transfer_shs = out_shs / light_shs    # [N, 16, 3]  分解出"传输函数"
    self._incidents_dc = transfer_shs[:, :1, :]
    self._incidents_rest = transfer_shs[:, 1:, :]
```

这相当于把 $L\_{ind}$ 分解为：

$$L\_{ind}(\mathbf{x}) = \underbrace{L\_{env}^{old}}_{\text{旧环境光}} \times \underbrace{T(\mathbf{x})}_{\text{空间传输函数}}$$

$T(\mathbf{x})$ 编码的是该位置相对于环境光的**空间特征**——比如墙角处暗一些、桌面亮一些。

### 步骤 2：重光照时重新合成

[render.py:L240-L245](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L240-L245)

```python
if pipe.transfer_light:
    transfer_shs = pc.get_incidents       # 现在存的是 T(x)
    light_shs = cubemap.shs               # 新环境光的SH投影
    incidents = light_shs * transfer_shs  # L_ind_new = L_env_new × T(x)
    incidents_light = eval_sh(incidents, normal)
```

相当于：

$$L\_{ind}^{new}(\mathbf{x}) = L\_{env}^{new} \times T(\mathbf{x})$$

> 在新的环境光下，每个位置接收到的间接光 = 新环境光 × 该位置原先学到的空间传输函数

***

## 三种模式的对比

| 模式                                                | `incidents_light` 的值         | 效果           |
| ------------------------------------------------- | ---------------------------- | ------------ |
| **训练/普通评估** (`relight=False`)                     | `eval_sh(incidents, normal)` | 正确地使用学到的间接光  |
| **重光照无传输** (`relight=True, transfer_light=False`) | `0`                          | 间接光丢失，画面偏平   |
| **重光照有传输** (`relight=True, transfer_light=True`)  | `light_new × transfer`       | 保留空间分布，适配新光照 |

从 [shade.py:L378](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L378) 可以看出，`transfer_light=True` 时 `incidents_light_map` 才会被传给 `pbr_shading` 作为 `irradiance`：

```python
irradiance = incident_light_map if not pipe.relight or (pipe.relight and pipe.transfer_light) else None
```

***

## 一张图总结

```
训练时:
  incidents SH 系数 = L_env_old × T(x)
         ↑              ↑          ↑
    每个Gaussian    旧cubemap   空间位置特征
  
重光照 transfer_light=True:
  incidents_new = L_env_new × T(x)
                  ↑              ↑
              新cubemap     位置特征保留
```

**所以** **`transfer_light`** **的作用就是：将原先学到的间接光空间分布（阴影、间接反射模式）传输到新的环境光照下，避免重光照时丢失局部光照信息。**
