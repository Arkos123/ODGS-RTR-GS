# 行主序与列主序：3DGS 的矩阵约定

## 背景

本仓库基于 3D Gaussian Splatting，其 CUDA 光栅化器继承自 OpenGL 约定，而 Python/NumPy/PyTorch 默认使用行主序。两者之间需要显式转换，这是代码中大量 `.T` / `.transpose(0, 1)` 的来源。

不理解这个约定是阅读代码时最常见的困惑之一。

## 基本概念

一个 4×4 矩阵在内存中的两种排列方式：

```
矩阵本身（数学表示）：
⎡ a b c d ⎤
⎢ e f g h ⎥
⎢ i j k l ⎥
⎣ m n o p ⎦

行主序（row-major）：按行存  → a,b,c,d, e,f,g,h, i,j,k,l, m,n,o,p
列主序（column-major）：按列存 → a,e,i,m, b,f,j,n, c,g,k,o, d,h,l,p
```

- **行主序**：NumPy / PyTorch 默认
- **列主序**：OpenGL 约定，本项目的 CUDA 光栅化器使用

## 核心链路：从相机参数到 CUDA 光栅化器

### 原始 3DGS 链路

```
R = c2w 旋转, T = w2c 平移
          ↓
getWorld2View(R, T)       ← 构建中间格式 [R^T | t; 0 1]（行主序行向量右乘格式）
          ↓
.transpose(0, 1)          ← 转列主序，供 CUDA 左乘
          ↓
CUDA rasterizer            ← 用 v_out = M * v_in（列主序左乘列向量）
```

见 `scene/cameras.py:63`：
```python
self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
```

### 简化后的 baking.py 链路

```python
c2w[:3, 3] = position
w2c = torch.inverse(c2w)
world_view_transform = w2c.T        # 行主序 w2c → 列主序 w2c（等价于上面两步的合并）
```

见 `baking.py:706-710`。

## 为什么 getWorld2View 要设计成中间格式？

`getWorld2View(R, t)` 返回的不是标准的 w2c 矩阵 `[R_w2c | t_w2c; 0 1]`，而是：

```
Rt = [R^T | t]     其中 R = c2w 旋转, t = w2c 平移
     [ 0  | 1]
```

**这不是因为"旋转矩阵转置=逆"**，而是因为这是在为**行向量右乘**构建中间格式。在 NumPy/PyTorch 中，点云默认存为 `N×3` 的行向量，行向量右乘矩阵 `X * M` 时，标准的 w2c 矩阵不能直接使用，需要旋转部分转置、平移放在最后一列。

这个中间格式**不会直接用于变换计算**——它随后会被 `.transpose(0, 1)` 转为列主序，CUDA 光栅化器收到后以 `v_out = M_colmajor * v_in`（左乘列向量）的方式使用。

## 常见陷阱

### 1. 传入 getWorld2View 的 t 必须是 w2c 平移

`getWorld2View(R, t)` 直接把 `t` 放到矩阵的第 4 列。如果传入的是相机位置（c2w 平移），结果根本不是 w2c 矩阵。

所有调用处传入的都是 w2c 平移：
- **Colmap**：`extr.tvec` 天然是 w2c 平移
- **Blender/NeRF**：`w2c[:3, 3]` = `-c2w_rot.T @ 相机位置`
- **OpenMVG**：`-R @ frame["value"]["center"]`（显式计算）

### 2. 不要被函数命名迷惑

函数名 `getWorld2View` 暗示输出是 w2c 矩阵，但实际上其输出**只有在 t 是 w2c 平移时**才是行主序的 w2c。如果 t 是相机位置（c2w 平移），输出的就是行主序的 c2w。原始 3DGS 靠 `getWorld2View2`（在其中对结果求逆再求逆）来吸收这个不一致。

### 3. 中间格式不是最终格式

不要试图直接使用 `getWorld2View` 返回的矩阵做变换——它行主序、旋转部分转置、平移在最后一列的布局不适合直接运算。它只是通往最终列主序格式的中间产物。

### 4. `.T` 和 `.transpose(0, 1)` 在数学上是等价的

两者在 PyTorch 中都是矩阵转置。`w2c.T` 是 `w2c.transpose(0, 1)` 的简写。

## 检查清单

当修改涉及 `world_view_transform` 或 `viewmatrix` 的代码时：

- [ ] 传入 `getWorld2View` 的 `t` 是 w2c 平移还是相机位置？
- [ ] 结果是否有 `.transpose(0, 1)` / `.T` 转为列主序？
- [ ] 如果删除了中间函数直接求 `w2c.T`，w2c 的旋转和平移部分是否正确？
- [ ] `camera_center` 是从矩阵反推（`world_view_transform.inverse()[3, :3]`）还是直接使用 `position`？两者等价，选一种保持即可。

## 术语速查

| 术语 | 含义 |
|------|------|
| 行主序 (row-major) | 按行存储，PyTorch/NumPy 默认 |
| 列主序 (column-major) | 按列存储，OpenGL/CUDA 光栅化器使用 |
| w2c (world-to-camera) | 世界坐标系 → 相机坐标系 |
| c2w (camera-to-world) | 相机坐标系 → 世界坐标系（= w2c⁻¹） |
| 中间格式 | `[R^T | t; 0 1]`，行主序行向量右乘格式 |
| world_view_transform | 列主序的 w2c 矩阵，直接传给 CUDA 光栅化器 |
