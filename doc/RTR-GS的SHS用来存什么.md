## RTR-GS 的 SHS（球谐系数）含义及变化

### 一、SHS 存的是什么？

`_shs_dc` + `_shs_rest`（合称 `get_shs`，形状 `[N, (D+1)², 3]`）在代码注释中被标注为 **"output radiance"（输出辐射度）**，但具体的**语义取决于渲染模式**：

#### 模式 A：非 PRT 模式（`--compute_with_prt` 关闭，默认 False）
SHS 就是标准 3DGS 的球谐颜色系数 → 直接通过 `eval_sh(视角方向)` 计算每个高斯在该视角下的 RGB 颜色。

#### 模式 B：PRT 模式（`--compute_with_prt` 打开，实际训练中启用）
这是**理解核心**。SHS **不再代表"视角相关颜色"**，而是被当作 **"该高斯位置处的入射光照 SH 系数"**，与传输系数（`diffuse_transfer`）配合使用：

```python
# PRTutils.cal_diffuse 中的核心计算（prt_utils.py:23-31）
shs = gaussian.get_shs                   # → 作为入射光照 c_j
shs_transfer = gaussian.get_diffuse_transfer  # → 传输系数 c_j^t
transport = ReLU(Σ(c_j · c_j^t) + 0.5)  # 传输函数
cd = diffuse_tint * transport            # 最终漫反射颜色 = 反照率 × 传输
```

同时高光部分（`cal_specular`）也用 SHS 作为光照，与 MLP 生成的传输系数 `LT_coeff` 点积。

**直观理解**：在 PRT 模式下，SHS 的角色从"视角颜色"变成了"位置相关的光照/辐射场"，而每个高斯的独立 `diffuse_transfer` + `specular_feature` 编码了材质/传输信息。光照和传输在 SH 域逐分量点积得出最终颜色。

### 二、三阶段中各阶段 SHS 的变化

| 阶段                                            | SHS 角色                  | SHS 是否被优化？                                                | 可视化效果                              |
| ----------------------------------------------- | ------------------------- | --------------------------------------------------------------- | --------------------------------------- |
| **SGS 训练后**（刚转换时）                      | 标准 3DGS 视角颜色        | SGS 已经训练完毕                                                | ✅ 正常，SHS → `eval_sh` 能看到物体颜色 |
| **RTR-GS Stage 1**（`render_ref_equirect`）     | 入射光照 SH               | ✅ 是！优化器包含 `f_dc`（lr=0.0025）和 `f_rest`（lr=0.000125） | ❌ 继续优化后会偏离原始颜色             |
| **RTR-GS Stage 2**（`render_ref_pbr_equirect`） | 同 Stage1                 | ✅ 继续优化                                                     | ❌ 进一步偏离原始颜色，趋向"光照"       |
| **训练完成后**                                  | 编码了场景光照/辐射场信息 | —                                                               | ❌ 直接用 SHS 显示出的不是正常外观      |

注意：发光的浮游高斯（floater）的 SHS 容易捕获高频局部光照，这也是为什么在 PRT 模式中仍然需要对 SHS 做约束。

### 三、关键结论

> **SGS 转换后的点云 → 放到普通 3DGS 查看器中显示正常**
> **经过 RTR-GS 训练后的点云 → 在普通查看器中直接用 SHS 显示会得到"光照图"，而不是物体的正常外观**

原因：
1. **训练期间 PRT 模式下 SHS 的语义发生了漂移**——它不再编码视角颜色，而是编码位置相关的入射光照
2. RTR-GS 的最终渲染不是 `eval_sh(shs, dir)`，而是 `diffuse_tint * ReLU(Σ(shs · transfer) + 0.5) + specular_tint * ReLU(Σ(MLP(feat, refl_dir) · shs))`，再在前向渲染中与反射分量混合
3. 即使保留扩展属性，普通 3DGS 查看器只读取 SHS → `eval_sh`，这完全绕过了 RTR-GS 的 PRT 核心机制

**如果你想把 RTR-GS 训练后的点云放到普通查看器中显示近似外观**，有两种思路：
- **方案 A**：在导出前，用 PRT 管线给每个视角"烘焙"出颜色，反写回 SHS（压缩回标准 3DGS 颜色表示），但这会有质量损失
- **方案 B**：让查看器理解扩展属性（如 `diffuse_tint`、`diffuse_transfer`、`specular_feature`），实现 PRT 计算——也就是 3DGS Editor 正在做的事情