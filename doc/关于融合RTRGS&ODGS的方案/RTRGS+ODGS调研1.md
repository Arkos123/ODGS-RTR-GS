
综合调研完成，以下是完整的可行性报告。

---

## 融合可行性报告：ODGS（全景重建）+ RTR-GS（光照材质分解）

### 一、提议的 Pipeline

```
全景图序列 (equirectangular)
        │
        ▼
    ODGS 训练 ───► .ply 文件 (6个基础属性: xyz, SH, scale, rot, opacity)
        │
        ▼
   属性转换 + RTR-GS 扩展属性初始化
        │
        ▼
   RTR-GS 逆渲染训练 ───► 材质/光照分解结果
        │
        └── GT 来源: 方案A 透视 / 方案B 全景图
```

### 二、可行性评估：3 个关键环节

---

#### 环节 1：ODGS 输出 → RTR-GS 输入（✅ 可行，需写一个转换脚本）

**现状差异**：

| 属性 | ODGS 的 .ply | RTR-GS 期望的 .ply |
|------|:---:|:---:|
| xyz | ✅ | ✅ |
| SH (f_dc, f_rest) | ✅ | ✅（但命名 `shs_dc/shs_rest`） |
| scaling, rotation, opacity | ✅ | ✅ |
| diffuse_tint, specular_tint | ❌ 无 | ✅ 需要 |
| ref_tint, ref_strength, ref_roughness | ❌ 无 | ✅ 需要 |
| specular_feature (10维) | ❌ 无 | ✅ 需要 |
| diffuse_transfer_dc/rest (PRT) | ❌ 无 | ✅ 需要 |
| PBR属性 (base_color等) | ❌ 无 | ✅ 可选 |

**ODGS 的 .ply 只有 6 个基础属性**，而 RTR-GS 需要 14 个（非 PBR）或 19 个（PBR）属性 [RTR-GS gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py#L570-L609)。

**解决方案**：写一个约 50 行的 Python 转换脚本，功能是：
1. 用 RTR-GS 的 `load_ply` 读取 ODGS 输出的 .ply（只读 xyz/SH/scale/rot/opacity）
2. 调用 `create_from_pcd` 类似的逻辑初始化所有 RTR-GS 扩展属性（diffuse_tint、ref_tint 等设为默认值）
3. 用 RTR-GS 的 `save_ply` 写出完整格式

**工作量**：低（~1 小时代码）

---

#### 环节 2：RTR-GS 以全景图为 GT 进行训练（⚠️ 有条件可行，但需要修改）

**两大挑战**：

**挑战 A：Camera 模型不兼容**

ODGS 和 RTR-GS 使用完全不同的渲染管线：

| 维度 | ODGS (equirectangular) | RTR-GS (perspective) |
|------|:---:|:---:|
| CUDA栅格化器 | `odgs_gaussian_rasterization` | `diff_gaussian_rasterization` (RTR版) |
| 投影模型 | 球面投影 (`computeOmniCov2D`) | 针孔透视投影 |
| 相机参数传递 | **不**传 `tanfovx/tanfovy/projmatrix` | 传 `tanfovx/tanfovy/projmatrix` |
| 额外输出 | `psi, lat, lon` | 无 |
| 法线计算 | 无 | 有（`computer_pseudo_normal`） |
| PRT/反射渲染 | 无 | 有 |

ODGS 的 CUDA rasterizer 完全无视 `projmatrix` [ODGS forward.cu](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/submodules/odgs-gaussian-rasterization/cuda_rasterizer/forward.cu#L75-L100)，使用自己的球面 Jacobian。而 RTR-GS 的渲染器 [gaussian_renderer/render.py](file:///home/huangpengyue/projects/RTR-GS/gaussian_renderer/render.py#L115-L136) 依赖 `tanfovx/tanfovy` 和 `full_proj_transform` 做视锥剔除。

**挑战 B：纬度权重损失**

全景图的 MSE/SSIM 损失需要纬度加权（`ws_map = cos(lat)`），否则极地像素会被过度优化。ODGS 通过 [loss_utils.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/utils/loss_utils.py#L6-L11) 中的 `est_wsmap()` 实现，RTR-GS 的损失函数没有这个。

**解决方案**（按推荐优先级）：

| 方案 | 做法 | 工作量 | 推荐度 |
|------|------|:---:|:---:|
| **A. 只用 ODGS 重建几何，RTR-GS 仍用透视 GT** | ODGS→PLY→RTR-GS 加载，但训练时 RTR-GS 仍使用原有的透视图像做 GT | **低** | ⭐⭐⭐ |
| **B. 新增 equirectangular 渲染分支** | 在 RTR-GS 中增加一个分支，检测到全景图时使用 ODGS 的 CUDA rasterizer 渲染，其余 PRT/反射/PBR 逻辑保持不变 | **中高** | ⭐⭐ |
| **C. 全景图→Cubemap 分片** | 把全景图展开为 6 张透视面片，每个面片用标准透视光栅化，损失在各面片上计算 | **中** | ⭐ |

---

#### 环节 3：两阶段串接训练（✅ 整体可行）

推荐路线图如下：

```
阶段 1 (ODGS):
  输入: 全景图序列 + OpenMVG 格式 (data_extrinsics.json)
  输出: .ply + checkpoint (checkpoint/chkpnt30000.pth)
  功能: 重建场景几何 (3DGS)

   ↓ 转换脚本: odgs_ply_to_rtrgs_ply.py

阶段 2 (RTR-GS):
  输入: ODGS 转换后的 .ply + 训练/测试相机信息
  输出: 材质分解 (albedo, roughness, metallic) + 光照
  功能: BRDF/光照逆渲染
  ⚠️ 可选的 GT 来源:
       - 方案A: 用已有的透视图像 (最简单)
       - 方案B: 用全景图 (需做渲染管线的修改)
```

### 三、具体需要修改/添加的文件

| 文件 | 改动量 | 说明 |
|------|:---:|------|
| **新建:** `scripts/odgs2rtrgs.py` | ~50行 | ODGS .ply → RTR-GS .ply 格式转换 |
| **新建:** `scene/dataset_readers_odgs.py` | ~100行 | 复用 ODGS 的 OpenMVG reader，整合到 RTR-GS 的 Scene 类 |
| 仅备选: `gaussian_renderer/render_equirect.py` | ~200行 | 如果要用全景 GT，则需要封装 ODGS 的 equirect 栅格化器 |
| 仅备选: `utils/loss_utils.py` (修改) | ~10行 | 添加纬度加权损失函数 `est_wsmap` |

### 四、建议的优先级路线

```
第一阶段 (低工作量, 立即可用):
  ODGS 训练 → 转换脚本 → RTR-GS 加载几何 → 用已有透视图像 继续材质分解
  └── 不改任何 CUDA 代码，不改渲染管线

第二阶段 (可选, 更高整合度):
  让 RTR-GS 支持全景 GT 输入 → 使用 equirect 渲染分支
  └── 需要改渲染器代码，可复用 ODGS 的 CUDA kernel
```

### 五、总结

| | 结论 |
|---|------|
| **是否可行？** | **总体可行 ✅**。核心路径（ODGS 重建 + RTR-GS 材质分解）没有不可逾越的技术障碍 |
| **主要工作** | (1) PLY 格式转换脚本  (2) 数据集加载器适配  (3) 可选的全景渲染分支 |
| **最大风险** | RTR-GS 的 `load_ply` 函数存在变量名 bug（`ref_roughness` 从错误字段读取，`diffuse_transfer` 读取使用了 `incidents_*` 变量名），需要先修复才能正确加载 |
| **推荐起步** | 先做 **方案A** —— ODGS 跑出 .ply → 转换脚本 → RTR-GS 加载几何后用原有的透视图像做逆渲染。这条路改动最小，能快速验证全流程 |
        

总结一下如果你决定推进，下一步需要做的事情：

### 下一步做什么（按顺序）

1. **先写转换脚本**（简单的开始）：将 ODGS 的 .ply 转为 RTR-GS 格式，并修复 `load_ply` 中的两个变量名 bug
2. **试跑全流程**：ODGS 训练 → 转换 → RTR-GS 加载几何 → RTR-GS 用现有透视图像 继续材质分解
3. **验证效果**：对比直接 RTR-GS 训练 vs ODGS+转换+RTR-GS 的结果

要开始做步骤 1 吗？


# 调研2
明白了，场景里只有全景图，没有透视 GT 可用。那确实得走方案 B 的路。让我重新梳理一下真正可行的方案。

你说得对，既然只有全景图，Option A 不可行。让我重新评估方案 B 的实际改动量。
        
好的，调研完毕。核心发现是：**RTR-GS 的渲染流程天然分为三层，CUDA 只做 alpha blending，Python 做颜色计算**，所以替换光栅化器是可行的。让我把完整分析更新到文档中。

[ODGS-RTR-GS-Integration.md](file:///home/huangpengyue/projects/RTR-GS/doc/ODGS-RTR-GS-Integration.md)
          
好，全部更新完毕。总结一下核心结论：

### 结论：完全可行，不需要改 CUDA 代码

RTR-GS 的渲染架构有个很巧妙的**三层分离**设计：
- **Layer 1 (Python)**：逐高斯计算 PRT 颜色、反射颜色 → **与投影无关**
- **Layer 2 (CUDA)**：Splatting + alpha blending → **唯一与投影相关的一层**
- **Layer 3 (Python)**：像素级 Deferred PBR → **与投影无关**

所以只需要把 Layer 2 的 CUDA 光栅化器从 `diff_gaussian_rasterization` 换成 `odgs_gaussian_rasterization` 就行。

**关键挑战**：ODGS 的光栅化器不支持 RTR-GS 用的 `features` 多通道张量机制（用于传递法线、BRDF 参数到像素级别）。

**解决方案**：多趟前向渲染 —— 用同一套 ODGS 光栅化器跑 4 趟，每趟渲染不同的伪彩色编码数据（法线、BRDF 参数、深度），跑完后解码出来做 PBR。不需要改 CUDA 代码。

技术细节和分阶段计划都写在了 [ODGS-RTR-GS-Integration.md](./ODGS-RTR-GS-Integration.md) 里。你要从 Phase 1（PLY 转换脚本 + 修复 load_ply bug）开始做吗？