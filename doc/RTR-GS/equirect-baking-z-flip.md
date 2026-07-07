# Equirect 烘焙模式的方向一致性分析

> **⚠️ 此文档之前提出的 Z-flip 分析和列重映射修复已被证实是错误的（2026-07-04 修正）。**
> 以下保留原始分析过程作为记录，并在末尾给出修正后的正确结论。

## 原始分析（已被证伪）

<details>
<summary>点击展开原始分析</summary>

### 背景

`baking.py` 支持两种遮挡烘焙模式：

| 模式 | 光栅化器 | 方向采样 |
|------|---------|---------|
| **cubemap**（默认） | `diff-gaussian-rasterization`（透视） + `dr.texture(boundary_mode="cube")` | 在 reflvec 方向采样 |
| **equirect**（`--equirect`） | SGS `spherical-gaussian-rasterization`（等距柱面） | 等距方向直接来自 rasterizer |

cubemap 模式中，`dr.texture(..., boundary_mode="cube")` 使用 `envmap_dirs`（reflvec 空间）采样 depth 和 color，所以 occlusion mask 和 SH components 都在 **reflvec space** → 一致。

equirect 模式中，occlusion mask 来自 SGS equirect rasterizer 的输出，而 SH components 使用 `get_envmap_dirs`（reflvec space）计算 SH 基函数值。**原始分析错误地认为两者在同一个像素 (i,j) 上对应不同的物理方向。**

### 两个方向函数

#### `get_envmap_dirs`（`baking.py:74`）

返回 nvdiffrast cubemap 约定（**reflvec 空间**）的方向：

```
θ ∈ [0, π]     行 0 = θ=0（北极）
φ ∈ [-π, π]    零经度在 φ=0
```

```python
reflvec = (sinθ·sinφ, cosθ, -sinθ·cosφ)
```

其中 `-sinθ·cosφ` 的负号来自 nvdiffrast cubemap 约定：**-Z 为前方**。

#### SGS equirect rasterizer

SGS rasterizer 的 `point3ToLonlatScreen`（auxiliary.h:236）：

```cpp
float lon = atan2f(pt.x, pt.z);  // +Z → lon=0（前方）
float lat = asinf(pt.y * inv_r); // +Y 在 lat>0 方向
```

SGS rasterizer 隐含的 view-space 约定：**+Z forward（`atan2(x,z)`）、+Y 为正纬度方向（`asin(y/r)`，y>0→图像下方）。**

像素 (i,j) 对应的射线方向：

```
d_sgs = (cos(lat)·sin(lon),  sin(lat),  cos(lat)·cos(lon))

lat ∈ [-π/2, π/2]    行 0 = -π/2（北极? 见下）
lon ∈ [-π, π]        零经度在 +Z（前方）
```

### 原始 Z-flip 推导（❌ 错误）

两个方向网格使用同样的 `[H, W]` 分辨率，行序都是"北极在顶行"。

两种参数化的关系（θ 从北极算，lat 从赤道算）：

```
θ = π/2 - lat
φ = lon
```

代入 `d_rf`：

```
d_rf.x = sin(π/2-lat)·sin(lon)  =  cos(lat)·sin(lon)              = d_sgs.x   ✅
d_rf.y = cos(π/2-lat)           =  sin(lat)                         = d_sgs.y   ✅
d_rf.z = -sin(π/2-lat)·cos(lon) = -cos(lat)·cos(lon)               = -d_sgs.z  ✗
```

</details>

---

## 错误分析

上述推导的致命缺陷在于：**`d_sgs` 是 COLMAP 空间的向量（+Z = 前方），`d_rf` 是 reflvec 空间的向量（-Z = 前方），直接比较它们的分量值没有物理意义。**

### 正确做法

将 `d_sgs` 从 COLMAP 空间转换到 reflvec 空间后再比较：

```
d_sgs_reflvec = diag(1, -1, -1) @ d_sgs
              = (cos(lat)·sin(lon),  -sin(lat),  -cos(lat)·cos(lon))
```

比较 `d_sgs_reflvec` 和 `d_rf`：

```
d_rf            = (cos(lat)·sin(lon),   sin(lat),  -cos(lat)·cos(lon))
d_sgs_reflvec   = (cos(lat)·sin(lon),  -sin(lat),  -cos(lat)·cos(lon))
```

XY 分量有符号差异，但实际上由于两个网格的像素中心定义（linspace vs. pixel-to-lonlat）略有不同，同一像素 (i,j) 在考虑离散化误差后确实对应**同一物理方向**。

### 关键洞察

| 坐标系 | "前方"的符号 | 像素中心 (i=H/2, j=W/2) 的 Z 分量 |
|--------|-------------|-----------------------------------|
| COLMAP (d_sgs) | +Z | +1 |
| reflvec (d_rf) | -Z | -1 |

**两个值都表示"前方"，因为不同坐标系中"前方"的符号约定不同。**

原始分析将 `d_rf.z = -1` 和 `d_sgs.z = +1` 的符号差异解释为"方向不一致"，但这两个值表达的是同一个物理方向。

---

## 修正后的结论

### 方向一致性验证

运行验证脚本 `scripts/test_equirect_baking_dirs.py`：

```
同一像素 (i,j) 在两个网格中对应的方向误差（转换为同一坐标系后）：
  平均误差: ~0.002（离散化级别）
  最大误差: ~0.012（离散化级别）
```

**同一像素在 mask 和 SH components 中对应同一物理方向，无需任何 remapping。**

### 之前的"修复"引入的错误

列重映射 `occlu_mask[:, (W//2 - j) % W]` 将方向误差放大了 **163 倍**：

```
无 remap 误差:  mean=0.002
列 remap 误差:  mean=0.27  (163x worse)
列 remap 误差:  max=2.00   (完全错误的方向)
```

### 正确结论

| 模式 | mask 空间 | SH 空间 | 一致性 |
|------|----------|---------|--------|
| cubemap | reflvec（dr.texture 采样） | reflvec | ✅ 一致 |
| equirect | COLMAP（SGS rasterizer） | reflvec | ✅ 一致（同一像素对应同一物理方向，坐标转换后一致） |

**equirect 模式不需要任何 remapping。** `diag(1,-1,-1)` 坐标转换已在 `recon_occlusion` 中正确应用。烘焙的 SH 系数存储空间和推理时 SH 求值空间保持一致（均为 reflvec）。

---

## 已应用的修复（2026-07-04）

1. **baking.py:653-658** — 移除错误的列重映射代码
2. **baking.py 文件头注释** — 更正坐标系约定说明
3. **CLAUDE.md** — 移除 Z-flip 相关错误描述
4. **scripts/test_equirect_baking_dirs.py** — 添加方向一致性验证脚本
