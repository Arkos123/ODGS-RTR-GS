# 环境光贴图旋转功能 — 实现文档

## 概述

在 `viewer_pygame.py` 中新增环境光贴图（Environment Map）旋转功能。用户可通过键盘实时旋转环境光方向，观察物体在不同光照角度下的重光照效果。

## 实现方案：方案一 — 旋转采样方向

**原理**：所有 cubemap 采样均通过 `nvdiffrast.dr.texture(cubemap, directions, ..., boundary_mode="cube")` 完成。对采样方向向量 `directions` 施加旋转矩阵后采样，等效于旋转环境光贴图本身，无需修改贴图数据或重建 mipmap。

**优点**：
- 计算开销极低（仅一次 `3x3 @ [N, 3]` 矩阵乘法）
- 不修改原始 cubemap 数据
- 不需要重建 mipmap 链
- 适合实时交互

## 修改文件清单

| # | 文件 | 修改类型 | 说明 |
|---|------|---------|------|
| 1 | `pbr/light.py` | 新增方法 + 修改现有方法 | 添加 `rotate_dirs()` 核心旋转方法；`export_envmap()`、`get_env_map()` 应用旋转 |
| 2 | `pbr/shade.py` | 修改渲染逻辑 | 3 个函数中采样前旋转方向向量 |
| 3 | `gaussian_renderer/render.py` | 修改渲染逻辑 | 直接环境可视化采样添加旋转 |
| 4 | `viewer_pygame.py` | 新增交互逻辑 | 添加键盘控制 + HUD 显示 |

---

## 详细修改

### 1. `pbr/light.py` — 核心旋转能力

#### 1.1 新增 `rotate_dirs()` 方法（L104-L114）

```python
def rotate_dirs(self, directions: torch.Tensor) -> torch.Tensor:
    if self.mtx is None:
        return directions
    orig_shape = directions.shape
    flat = directions.reshape(-1, 3)
    if not isinstance(self.mtx, torch.Tensor):
        mtx_t = torch.tensor(self.mtx, dtype=torch.float32, device=directions.device)
    else:
        mtx_t = self.mtx.to(dtype=torch.float32, device=directions.device)
    rotated = (mtx_t @ flat.T).T
    return rotated.reshape(orig_shape)
```

- 利用 `CubemapLight` 已有的 `self.mtx` 属性（通过 `xfm()` 设置）
- 处理 numpy array → torch tensor 转换
- 保持输入 shape 不变（适配任意维度：`[N,3]`、`[H,W,3]`、`[1,H,W,3]` 等）

#### 1.2 `export_envmap()` 应用旋转（L189）

```python
sample_dirs = self.rotate_dirs(reflvec) if self.mtx is not None else reflvec
```

- 导出环境贴图时（base 和 diffuse 分支均生效）采样方向添加旋转

#### 1.3 `get_env_map()` 应用旋转（L240-L242）

```python
dirs = self.envmap_dirs
if self.mtx is not None:
    dirs = self.rotate_dirs(dirs)
```

---

### 2. `pbr/shade.py` — 3 个渲染函数

所有三个函数中，在 `dr.texture()` 采样 cubemap 之前旋转方向向量。

#### 2.1 `get_reflectance_color_forward` — 前向着色反射（L181-L182）

```python
if light.mtx is not None:
    ref_dirs = light.rotate_dirs(ref_dirs)
```

方向：反射方向 `R = 2(N·V)N - V`

#### 2.2 `get_reflectance_color` — 延迟着色反射（L227-L228）

```python
if light.mtx is not None:
    ref_dirs = light.rotate_dirs(ref_dirs)
```

方向：反射方向，同上。

#### 2.3 `pbr_shading` — PBR 着色（L291-L296）

```python
sampling_dirs = ref_dirs
sampling_normals = normals
if light.mtx is not None:
    sampling_dirs = light.rotate_dirs(ref_dirs)
    sampling_normals = light.rotate_dirs(normals)
```

需要同时旋转两个方向：
- **漫反射采样方向** `sampling_normals` → 法线方向（查找 `light.diffuse`）
- **镜面反射采样方向** `sampling_dirs` → 反射方向（查找 `light.specular`）

---

### 3. `gaussian_renderer/render.py` — 直接环境可视化（L494-L495）

```python
if cubemap.mtx is not None:
    directions = cubemap.rotate_dirs(directions)
```

- 非训练模式下，直接使用世界空间视线方向采样 `cubemap.base` 用于可视化
- 添加旋转使可视化也跟随环境光旋转

---

### 4. `viewer_pygame.py` — 交互控制

#### 4.1 新增辅助函数（L744-L755）

```python
def update_env_rotation(cubemap, angle_y):
    cos_a = math.cos(angle_y)
    sin_a = math.sin(angle_y)
    mtx = torch.tensor([
        [cos_a,  0, -sin_a],
        [0,      1,  0    ],
        [sin_a,  0,  cos_a]
    ], dtype=torch.float32)
    cubemap.xfm(mtx)
```

- 构建绕 **Y 轴**旋转矩阵（最直观的旋转轴，水平旋转环境光）
- 调用 `cubemap.xfm(mtx)` 设置变换矩阵

#### 4.2 新增状态变量（L614）

```python
env_rotation_y = 0.0  # 环境光绕Y轴旋转角度（弧度）
```

#### 4.3 键盘事件处理（L634-L642）

```python
elif event.key == pygame.K_LEFT:
    env_rotation_y -= 0.05     # 逆时针 5°/次
    update_env_rotation(scene_data['cubemap'], env_rotation_y)
elif event.key == pygame.K_RIGHT:
    env_rotation_y += 0.05     # 顺时针 5°/次
    update_env_rotation(scene_data['cubemap'], env_rotation_y)
elif event.key == pygame.K_r:
    env_rotation_y = 0.0       # 重置
    update_env_rotation(scene_data['cubemap'], env_rotation_y)
```

#### 4.4 HUD 显示（L724, L727）

```python
env_rot_text = font.render(f"Env Rot: {env_rotation_y * 180 / math.pi:.1f}° [←→]", True, (0, 255, 0))
# ...
screen.blit(env_rot_text, (10, 130))
```

#### 4.5 控制台提示（L592-L593）

```python
print("    LEFT/RIGHT: Rotate environment map")
print("    R: Reset environment rotation")
```

---

## 数据流链路

```
用户按键 (← → R)
  → update_env_rotation(cubemap, angle)
    → cubemap.xfm(rot_matrix)          # 设置旋转矩阵到 self.mtx
      → render_frame() 每帧调用
        → render_fn() → render.py
          → shade.py 各函数:
              light.rotate_dirs(directions)  # 旋转采样方向
                → dr.texture(cubemap, rotated_dirs, boundary_mode="cube")
                  → 等效于旋转了环境贴图
```

## 使用说明

| 按键 | 功能 |
|------|------|
| `←` | 环境光逆时针旋转 5° |
| `→` | 环境光顺时针旋转 5° |
| `R` | 重置环境光旋转为 0° |
| `M` | 切换飞行/轨道模式 |
| `ESC` | 退出 |

HUD 显示格式：`Env Rot: 45.0° [←→]`

## 兼容性说明

- 旋转矩阵通过 `self.mtx` 传递，**默认 `None` 时不执行任何旋转**，完全向后兼容
- 训练流程不受影响（`render.py` 中 `is_training=True` 时跳过环境可视化代码）
- `export_envmap()` 在导出时如有旋转矩阵也会反映旋转后的效果

## 性能影响

| 项目 | 开销 |
|------|------|
| 设置旋转矩阵 | CPU 端 3 个三角函数 + 9 个 float 赋值 |
| 每帧旋转方向 | GPU 端 1 次 `[3,3] @ [..., 3]` 矩阵乘法 |
| 额外显存 | 无 |

**结论**：对渲染帧率几乎无影响。
