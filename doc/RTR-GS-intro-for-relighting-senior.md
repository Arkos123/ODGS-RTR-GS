# RTR-GS 简介（给师兄看）

## RTR-GS 是什么？

RTR-GS (Real-Time Relighting Gaussian Splatting) 是一个基于 **3D Gaussian Splatting** 的**逆渲染 + 可重光照**框架。输入一组多视角照片 → 重建场景的几何、材质、光照 → 支持重光照（relighting）和新视角合成。

## 核心问题

3DGS 本质上是一个辐射场（radiance field），学到的颜色是"光照+材质"的耦合结果，无法分离光照和材质，因此无法重光照。

RTR-GS 做的事情是：**在 3DGS 的框架下解耦光照、材质、几何**，使重建的场景可以在新的光照条件下重新渲染。

## 核心方法

### 双分支渲染架构（Two-Branch Hybrid Rendering）

把场景外观分解为**低频（漫反射）**和**高频（镜面反射）**两部分：

**1. Radiance Branch（PRT 分支）— 处理低频**
- 用 **Precomputed Radiance Transfer (PRT)** 表达漫反射和视差依赖
- 世界共享的球谐光照（SH lighting）+ 每个 Gaussian 独立的传输特征向量
- 结合一个小 MLP（TransferMLP）解码视差依赖的传输特征
- 本质上是**前向渲染**：光照 × 传输 = 颜色

**2. Reflection Branch（反射分支）— 处理高频**
- 用**反射图（Reflection Map）**做 deferred rendering
- **Split-sum 近似**：BRDF 积分拆分为预积分的环境光 + 菲涅尔项
- 每个 Gaussian 存储反射属性：tint（镜面色）、roughness（粗糙度）、strength（强度）

**3. 最终融合**：`I = C_diffuse * (1 - R_strength) + C_reflection * R_strength`（逐像素 alpha blend）

### PBR 分支（Stage 2）— 材质分解

Stage 2 开启 PBR 分支，进一步分解出：
- **Albedo（base_color）**：漫反射基础色
- **Roughness / Metallic**：粗糙度、金属度
- **Incident Light**：incident light SH 系数（可迁移光照）
- 使用 **Cook-Torrance BRDF** 做微表面着色
- 结合 Occlusion Volume（预烘焙可见性）模拟间接光照

### 两阶段训练

1. **Stage 1**（`-t render_ref`）：训练几何 + 反射属性，不做材质分解
2. **Baking**（`baking.py`）：预计算可见性到 3D 体素网格（SH 系数存储）
3. **Stage 2**（`-t render_ref_pbr`）：固定几何，训练 PBR 材质 + 光照分解

### 多数据集格式

支持 Colmap、Blender、NeILF、Stanford ORB、OpenMVG（全景）等格式。

## 法线模型

- Gaussian 的**最短轴**近似为法线，朝向观察方向翻转
- 通过深度图生成**伪法线**做监督，约束几何平滑度
- 反射渲染的梯度也反馈到法线
- 定期传播法线（normal propagation）增强优化鲁棒性

## 全景模式（Equirectangular 360°）

- 用 SGS（Spherical Gaussian Splatting）子模块的**球面 CUDA 光栅化器**
- 支持等距柱状投影（equirectangular）图像的直接训练
- 深度导出伪法线、纬度感知剪枝（防原图中的 floater）
- 5 阶段管线：SGS 训练 → PLY 转换 → RTR-GS Stage1 → Baking → RTR-GS Stage2

## 当前关注的几个问题

1. **材质分解的稳定性**：在高光/反射区域容易相互混淆（roughness 和 metallic 分解不干净）
2. **Normal 精度**：最短轴法线的近似在复杂几何上不够准确，且全景模式下尤其明显
3. **Occlusion/间接光照**：预烘焙的可见性无法表示动态阴影
4. **重光照的一致性**：不同光照条件下的渲染质量需要进一步提升
5. **Equirect 模式的 floater**：360° 全景场景中存在一些悬浮高斯伪影，已经部分修复但仍需改进
