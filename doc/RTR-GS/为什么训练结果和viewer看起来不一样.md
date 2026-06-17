
我发现了 **4 个关键差异** 以及 **1 个关键混淆点**，这些共同解释了 viewer 和训练输出看起来不一样的原因。让我逐一说明：

---

## 混淆点：两个"渲染结果"本质不同

训练时保存的 `render.png` 是 **混合渲染结果**（`render_pkg["render"]` = 辐射度 + 反射），而 viewer 显示的是 **PBR 渲染结果**（`render_pkg["pbr"]` = BRDF 材质分解）。两者走的是不同的 shading 路径：

| | `render`（训练保存） | `pbr`（viewer 显示） |
|---|---|---|
| **公式** | `(1-ref_strength)×radiance + ref_strength×reflection` | BRDF (GGX + Lambertian) |
| **gamma校正** | ❌ 无 | ✅ 有 (`gamma_func`) |
| **依赖** | 依赖 PRT/SH 辐射度 + refmap | 依赖 cubemap + incident_light |

---

## 关键差异 ①：`compute_with_prt`（影响最大）

| | Viewer | 训练 |
|---|---|---|
| `compute_with_prt` | **`False`** | **`True`** |

训练时使用 PRT（含 MLP 网络）计算辐射度颜色，这是论文的核心方法。Viewer 中关闭了 PRT，退化为**原始球谐函数 (SH)** 着色。PRT 与 SH 的结果差异会很大，因为 PRT 通过共享全局光照 + MLP 提供了更强的低频约束。

在 [viewer.py:L429-444](file:///home/huangpengyue/projects/RTR-GS/viewer_pygame.py#L429-L444) 中：
```python
pipe = type('Pipe', (), {
    'compute_with_prt': False,  # ← 这里应该改为 True
    ...
})()
```

而在训练脚本中是通过 `--compute_with_prt` 命令行参数传入的。

---

## 关键差异 ②：`relight` 模式（PBR 渲染无入射光）

| | Viewer | 训练/Eval |
|---|---|---|
| `pipe.relight` | **`True`** | `False`（默认） |

当 `pipe.relight=True` 时，[render.py:L236-L247](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L236-L247) 中的 PBR 分支会将 `incidents_light` 置零：
```python
if not pipe.relight:
    incidents_light = ...  # 使用学习到的入射光
else:
    if pipe.transfer_light:
        incidents_light = ...  # 使用传输光照
    else:
        incidents_light = torch.zeros_like(base_color)  # ← viewer 走这里，无入射光！
```

接着在 [render.py:L378](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L378) 传给 `pbr_shading` 时：
```python
irradiance = incident_light_map if not pipe.relight or (...) else None
# 当 pipe.relight=True 时 irradiance = None
```

所以 viewer 的 PBR 渲染**完全没有入射光信息**，只依赖 cubemap 的环境光照，导致材质看起来非常奇怪。

---

## 关键差异 ③：`refmap` 来源不同

| | Viewer | 训练/Eval |
|---|---|---|
| `refmap` | **= cubemap（同一对象）** | **从单独 checkpoint 加载** |

Viewer 中 [viewer.py:L462](file:///home/huangpengyue/projects/RTR-GS/viewer_pygame.py#L462) 将 `refmap` 设置为和 `cubemap` 同一对象：
```python
"refmap": cubemap,  # refmap = cubemap (共享同一环境贴图)
```

而训练时是从单独的 `refmap_chkpntXXXXX.pth` checkpoint 加载的，两者在训练过程中分别优化，得到的是不同的环境贴图。这会导致反射效果完全不同。

---

## 关键差异 ④：最终显示的图像不同

| | Viewer | 训练 `render.png` |
|---|---|---|
| **取哪个键** | `render_pkg["pbr"]`（[viewer.py:L476](file:///home/huangpengyue/projects/RTR-GS/viewer_pygame.py#L476)） | `render_pkg["render"]`（[train.py:L328](file:///home/huangpengyue/projects/RTR-GS/train.py#L328)） |

Viewer 显示的是 `pbr` 键（PBR 渲染+gamma校正），而训练保存的 `render.png` 是 `render` 键（混合渲染，无gamma校正）。

---

## 总结与修复建议

为了让 viewer 显示与训练一致的渲染结果，需要修改以下几点：

1. **显示 `render` 而不是 `pbr`**（与训练保存的一致）
2. **`compute_with_prt = True`**（使用 PRT+MLP）
3. **`relight = False`**（使用学习到的入射光）
4. **`refmap` 从单独的 checkpoint 加载**（如果存在 `refmap_chkpnt` 文件）

此外，如果用户在 viewer 中使用自定义 envmap（`--envmap_path`），则 `relight=True` 是合理的（用于重光照），但此时：
- 应确保加载了单独的 `refmap` checkpoint（保留训练好的反射贴图）
- 若不加载单独的 refmap，反射会使用自定义的 envmap，与训练时不一致

要我现在修复 `viewer_pygame.py` 使其与训练输出一致吗？

---

分析完毕。三个 `eval_relighting_*.py` 的重光照实现模式**高度一致**。我来跟你对比 viewer 和它们的差异。