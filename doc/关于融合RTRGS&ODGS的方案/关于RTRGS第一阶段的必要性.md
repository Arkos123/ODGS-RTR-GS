好问题。让我结合代码和文档来分析。

## Stage 1 到底在训练什么？

从我们的 pipeline 看，**ODGS 已经提供了几何**（xyz, opacity, scales, rotations, SH 颜色），Stage 1 训练的是 RTR-GS **独有的新组件**：

| 组件 | 来源 | 需要训练吗？ |
|------|------|------------|
| **xyz / opacity / scales / rotations** | ODGS 提供 ✅ | 轻微微调 |
| **SH 颜色** | ODGS 提供 ✅ | 轻微微调 |
| **ref_tint / ref_roughness / ref_strength**（反射属性） | `odgs2rtrgs.py` 初始化为默认值 ⚠️ | **需要训练** |
| **diffuse_transfer**（PRT 传输向量） | 初始化为零 ⚠️ | **需要训练** |
| **PRT TransferMLP 网络** | 随机初始化 ⚠️ | **需要训练** |
| **Reflection Map / CubemapLight**（环境贴图） | 随机初始化 ⚠️ | **需要训练** |
| **法线**（高斯的短轴方向） | 从 ODGS scales 计算，需优化 | **需要训练** |

## 30k 迭代的必要性

30k 是 RTR-GS **从零训练**（从 Colmap SfM 点云起步）的设置。我们的场景有 ODGS 的良好初始化，所以**理论上可以减少**。但有两个因素：

1. **PRT 传输网络和反射贴图是从零开始练**的，它们需要足够的迭代来收敛
2. 训练日志显示约 4k 次后 PSNR 就达到 24+，**说明后期迭代主要是收敛这些新组件**

从日志看 PSNR 增长曲线：

```
~4k iters: PSNR 24.2 (已经不错了，因为 ODGS 几何好)
~33k iters (Stage 2 最后): PSNR 24.15 (略有提升)
```

所以**可以试试把 Stage 1 降到 15k-20k 次迭代**，不过 occlusion baking 需要几何稳定，太早停可能影响遮挡质量。要改一下脚本试试？