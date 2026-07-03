# 修复 Baking/遮罩中的 reflvec/COLMAP 坐标系不一致

## 背景

baking.py 涉及三个坐标系：**COLMAP world**（+Y下, +Z前）、**reflvec**（+Y上, -Z前）、**Equirect view**（+Y上, +Z前）。代码中有两处坐标系不一致，恰好互相抵消，但存在很久未被发现。

## 修复内容

### 修复 1：`gs_ir/__init__.py` — 去注释法线方向转换

`recon_occlusion` 中，baking 的 SH 系数在 **reflvec** 空间计算，但传入的 `normals` 是 **COLMAP** 世界空间。SH 基函数在错误的空间求值会导致水平面（地板、天花板）出现系统性遮挡误判。

**修复**：在 `SH_reconstruction` 前将 normals 从 COLMAP 空间转换到 reflvec 空间：

```python
reflvec_normals = normals.clone()
reflvec_normals[:, 1] *= -1.0  # COLMAP +Y下 → reflvec +Y上
reflvec_normals[:, 2] *= -1.0  # COLMAP +Z前 → reflvec -Z前
```

（这个修复实际上在 dev_log 20260617 已经记录了，但在后续代码中被注释掉了。）

### 修复 2：`baking.py` — 交换 cubemap faces 2↔3, 4↔5

cubemap 的六面中，X 轴在两个空间同向（`diag(1,-1,-1)` 不改 X），但 Y 和 Z 轴翻转了。`rotations[k]` 定义的朝向在 COLMAP 空间，但 nvdiffrast 的 cubemap 采样在 reflvec 空间。导致：

```
Face   | COLMAP 朝向 | reflvec 朝向 | nvdiffrast 期望  
+Z(4)  | +Z          | -Z           | +Z → ✗
-Z(5)  | -Z          | +Z           | -Z → ✗
+Y(2)  | +Y          | -Y           | +Y → ✗
-Y(3)  | -Y          | +Y           | -Y → ✗
```

**修复**：互换 cubemap stack 中 faces 2↔3, 4↔5 的渲染顺序，使 nvdiffrast 采样时能取到正确方向的内容。

### 关于两个 Bug 的抵消

这两个 Bug 恰好代表同一个绕 X 轴 180° 旋转 $R = \text{diag}(1,-1,-1)$：
- Bug A（baking）：SH 系数经过 R 旋转（cubemap face 取样方向取反）
- Bug B（求值）：法线未经 R 旋转就送入 SH 求值

由于 SH 是旋转等变的（rotationally equivariant），两个 Bug 同时存在时 $\sum c_{\text{swap}} \cdot Y(n_{\text{colmap}}) = \sum c_{\text{true}} \cdot Y(n_{\text{reflvec}})$ 恰好正确。但**只修其中一个会破坏正确性**，必须同时修改。

### 顺便做的清理

1. **删除了冗余的 `getWorld2ViewTorch` 函数**，两处调用简化为 `w2c.T`
2. **更新了 `rotations` 的注释**，删除了误导的 `lookAt(...)` 行末注释
3. **更新了 `occlusion_baking.md`** 中的代码片段，反映当前实现
4. **在 CLAUDE.md 添加了行主序/列主序的说明**

## 验证方法

```bash
# 比较修复前后的 occlu_mask 输出或 PBR 结果
# 或使用 --skip_walls 运行 baking，观察墙壁边缘的遮挡是否更准确
```

## 影响范围

| 组件 | 影响 |
|------|------|
| `baking.py` — cubemap 路径 | 修复：Y/Z face 方向更正 |
| `baking.py` — equirect 路径 | 无影响（不涉及 cubemap face 翻转） |
| `gs_ir/__init__.py` — `recon_occlusion` | 修复：法线正确转换到 reflvec 空间 |
| `doc/RTR-GS/row-major-column-major.md` | 新建文档 |
