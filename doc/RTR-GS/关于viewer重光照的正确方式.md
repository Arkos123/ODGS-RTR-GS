
## eval_relighting 三兄弟的共同模式

所有三个脚本的重光照都遵循完全相同的模式，以下是 [tensorIR](file:///home/huangpengyue/projects/RTR-GS/eval_relighting_tensorIR.py#L194-L210) 的核心代码（[stanford_orb](file:///home/huangpengyue/projects/RTR-GS/eval_relighting_stanford_orb.py#L308-L324)、[colmap](file:///home/huangpengyue/projects/RTR-GS/eval_relighting_colmap.py#L164-L166) 完全一致）：

```python
# 1. 从 HDR 创建新的 cubemap
cubemap = CubemapLight(base_res=res).cuda()
cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
cubemap.build_mips()
cubemap.eval()

# 2. refmap = cubemap（共享同一个对象）
render_kwargs = {
    "pc": gaussians,
    "pipe": pipe,                    # 从命令行解析，compute_with_prt 由 --compute_with_prt 控制
    "bg_color": background,
    "is_training": False,
    "dict_params": {
        "transfer_net": transfer_net,
        "occlusion_volumes": occlusion_volumes,
        "aabb": aabb,
        "cubemap": cubemap,
        "refmap": cubemap,           # ← 注意：refmap = cubemap（viewer 已经做对了）
        "brdf_lut": brdf_lut,
        "canonical_rays": canonical_rays,
        "iteration": iteration,
        "relight": True,
    },
}
```

---

## Viewer 与 Eval 的逐项对比

| 方面 | viewer_pygame.py | eval_relighting_*.py | 结论 |
|---|---|---|---|
| **`compute_with_prt`** | **`False`** ❌ | `True`（通过 `--compute_with_prt`） | **致命差异** |
| **`refmap`** | `= cubemap` | `= cubemap` | ✅ 一致 |
| **`relight`** | `True` | 由 `--relight` 控制 | ✅ 正确（重光照） |
| **`transfer_light`** | `False` | `False`（默认） | ✅ 一致 |
| **显示图像** | `render_pkg["pbr"]` | `render_pkg["pbr"]` | ✅ 一致 |
| **`metallic`** | `True` | 由 `--metallic` 控制 | ✅ 正确 |
| **加载 transfer_net** | ✅ 已加载 | ✅ 加载（条件同） | ✅ 已实现 |
| **`tone_mapping`** | `True` | 默认 `True` | ✅ 一致 |

---

## 唯一问题的根源

核心问题只有一个：**`compute_with_prt = False`**。

为什么这是致命的：

1. 模型是用 `--compute_with_prt` 训练的。Stage 2 PBR 训练中，颜色计算走的是 PRT 路径：`override_color = PRTutils.cal_color(pc, net, viewdirs, normal)`，**原始 SH 系数没有被优化用于最终渲染**。

2. 当 viewer 设置 `compute_with_prt = False` 时，[render.py:L186-L196](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L186-L196) 会退回到标准 3DGS 的 SH 渲染：
   ```python
   if override_color is None:
       # 没有 PRT，用未充分优化的 SH 系数
       shs = pc.get_shs
   ```
   这些 SH 系数在 PRT 训练过程中没有被充分优化，所以渲染结果"看起来很怪"。

**修复方案只需要一行变动：**

在 [viewer.py:L432](file:///home/huangpengyue/projects/RTR-GS/viewer_pygame.py#L432) 将：
```python
'compute_with_prt': False,
```
改为：
```python
'compute_with_prt': True,
```

其他所有参数 viewer 已经和 eval_relighting 脚本一致了。要我直接改掉吗？