
# ODGS ↔ RTR-GS 集成方案

**目标**：使用 ODGS 从等距柱状（360°）图像序列重建 3DGS 几何，然后将重建结果输入到 RTR-GS 进行逆向渲染（BRDF/光照分解）。

**约束**：所有输入数据均为等距柱状全景图 —— 没有可作为真值的透视图像。

---

## 流水线概览

```
                   ┌──────────────────────────────────────┐
                   │  ODGS 训练（等距柱状）                │
 输入：全景图序列  │  - 重建 3DGS 几何                     │
 ─────────────────►│  - 输出 .ply（6 属性）+ 检查点        │
                   └──────────┬───────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────────┐
                   │  PLY 转换 +                          │
                   │  RTR-GS 属性初始化                   │
                   └──────────┬───────────────────────────┘
                              │
                              ▼
                   ┌──────────────────────────────────────┐
                   │  RTR-GS 等距柱状逆向渲染              │
                   │  - 使用 ODGS CUDA 光栅化器           │
 输入：全景图序列  │  - 多通道前向渲染                     │
 ─────────────────►│  - PRT + 反射 + PBR                  │
                   │  - 纬度加权损失函数                   │
                   └──────────────────────────────────────┘
```

---

## 架构分析：可行性说明

RTR-GS 的渲染流水线具有**清晰的三层分离**，这使得更换光栅化器非常实用：

```
[第 1 层] Python：逐高斯颜色计算
    - PRT 颜色（漫反射 + 镜面反射）
    - 反射颜色（前向着色）
    - 组合逐高斯属性
          │
          ▼
[第 2 层] CUDA：溅射 + 透明度混合（投影相关）
    - 基于瓦片的排序和剔除
    - 逐像素透明度合成：
      • colors_precomp（RGB）      → rendered_image
      • features（10~18D 张量）    → rendered_feature（如果支持）
          │
          ▼
[第 3 层] Python：逐像素延迟着色（投影无关）
    - 法线图提取
    - 反射渲染
    - PBR 着色
```

**关键洞察**：只有第 2 层（CUDA 光栅化器）与投影模型有关。第 1 层和第 3 层与投影无关 —— 无论输入是透视投影还是等距柱状投影，它们都基于逐高斯或逐像素数据进行计算。

---

## 主要挑战：特征混合

RTR-GS 的延迟 PBR 依赖于 CUDA 光栅化器的 `features` 张量机制 —— 它将逐高斯属性（深度、法线、BRDF 参数）打包成一个多通道张量，然后 CUDA 内核独立地对每个通道进行透明度混合。

**ODGS 的 CUDA 光栅化器不支持 `features` 机制**。它只处理 `colors_precomp`（RGB）和 `sh`（球谐系数）。

### 解决方案：多通道前向渲染

与其大费周章地修改 ODGS CUDA 内核，不如将每个所需通道分别作为一个独立的前向通道进行渲染：

| 通道 | colors_precomp = | 输出 =            | 用途                     |
|------|------------------|-------------------|--------------------------|
| 1    | PRT 辐射度（RGB） | `radiance_map`    | 基础漫反射 + 镜面反射颜色 |
| 2    | 法线（编码为 RGB）| `normal_map`      | 逐像素表面法线           |
| 3    | BRDF 参数（RGB）  | `brdf_map`        | base_color, roughness, metallic |
| 4    | 深度（float→RGB）| `depth_map`       | 逐像素深度               |

所有通道都使用**同一个** ODGS CUDA 光栅化器和**同一组**相机参数。逐高斯数据完全相同（相同的位置、不透明度、缩放、旋转）—— 只有 `colors_precomp` 发生变化。

渲染完成后，将每个通道解码回其语义含义，然后运行标准的 RTR-GS 逐像素 PBR 着色。

**性能代价**：ODGS 光栅化器速度很快（在 1024×512 分辨率下约 2ms/通道）。4 个通道约 8ms，可以接受。

---

## 任务分解

### 阶段 1：PLY 桥接（必需，低难度）

编写 `scripts/odgs2rtrgs.py`：
1. 加载 ODGS .ply 文件（6 属性：xyz、SH、scale、rot、opacity）
2. 创建 RTR-GS 的 GaussianModel
3. 将 RTR-GS 扩展属性初始化为默认值：
   - `diffuse_tint`, `specular_tint`, `ref_tint` = zeros
   - `ref_strength` = sigmoid⁻¹(0.01)
   - `ref_roughness` = sigmoid⁻¹(0.65)
   - `specular_feature` = zeros
   - `diffuse_transfer_dc/rest` = zeros
   - PBR：`base_color`, `roughness`, `metallic` = 默认值
4. 保存为 RTR-GS 格式的 .ply 文件

同时修复 RTR-GS `load_ply` 中的变量名错误（位于 [scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py#L718-L746)）。

### 阶段 2：等距柱状相机支持（必需，低难度）

1. 在 RTR-GS 的 [scene/dataset_readers.py](file:///home/huangpengyue/projects/RTR-GS/scene/dataset_readers.py) 中添加等距柱状数据集读取器 —— 复用 ODGS 的 OpenMVG 解析器
2. 相机类自然适配：ODGS 使用与 RTR-GS 相同的 `Camera(R, T, FoVx, FoVy, ...)`，其中 `FoVx = π` 硬编码

### 阶段 3：等距柱状渲染器（核心工作，中等难度）

创建 `gaussian_renderer/render_equirect.py`：
1. 复用 RTR-GS 的 PRT 计算（与投影无关）
2. 将 `diff_gaussian_rasterization` 替换为 `odgs_gaussian_rasterization`
3. 移除仅适用于透视的参数：`tanfovx/tanfovy/cx/cy/projmatrix`
4. 实现多通道渲染以获取延迟着色所需的数据
5. 添加纬度加权损失函数

### 阶段 4：训练流水线（中等难度）

修改 `train.py` 以支持等距柱状模式：
1. 加载 ODGS 的 .ply 或检查点作为初始几何
2. 禁用致密化（这已经是阶段 2 的行为）
3. 使用等距柱状渲染器
4. 使用纬度加权的 L1 + SSIM 损失

---

## 进度检查清单

- [ ] 阶段 1：PLY 转换脚本 + load_ply 错误修复
- [ ] 阶段 2：等距柱状数据集读取器 + 相机
- [ ] 阶段 3：`render_equirect.py`（多通道前向渲染）
- [ ] 阶段 3：纬度加权损失
- [ ] 阶段 4：训练脚本修改
- [ ] 端到端测试

---

## 工作量估算

| 阶段 | 需修改的文件 | 代码行数 | 难度 |
|------|--------------|:--------:|:----:|
| 阶段 1 | 1 新增 + 1 修改 | ~80 | 低 |
| 阶段 2 | 1 修改 | ~100 | 低 |
| 阶段 3 | 1 新增 | ~300 | 中 |
| 阶段 4 | 1 修改 | ~100 | 中 |
| **总计** | **约 5 个文件** | **约 580** |      |