# ODGS GaussianModel vs RTR-GS GaussianModel
以下是对两个文件中 `GaussianModel` 类的完整详细对比分析。

---

## 对比总结

### 一、ODGS GaussianModel (`submodules/odgs/scene/gaussian_model.py`)

**文件路径：** [odgs/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene/gaussian_model.py)

#### 1. 属性列表

| 属性名 | 形状 | 说明 |
|--------|------|------|
| `_xyz` | (N, 3) | 高斯位置坐标 |
| `_features_dc` | (N, 3, 1) | SH DC 系数（原始3DGS命名） |
| `_features_rest` | (N, 3, 15) | SH 高阶系数（原始3DGS命名） |
| `_scaling` | (N, 3) | 各向异性缩放 |
| `_rotation` | (N, 4) | 四元数旋转 |
| `_opacity` | (N, 1) | 不透明度 |

仅有 **6 个可学习张量属性**，完全遵循原始 3D Gaussian Splatting 设计。

#### 2. `save_ply` / `load_ply` 函数签名

```python
def save_ply(self, path)      # 保存到 PLY 文件
def load_ply(self, path)      # 从 PLY 文件加载
```

#### 3. PLY 属性列表（`construct_list_of_attributes`）

```python
['x', 'y', 'z',                     # 位置
 'nx', 'ny', 'nz',                  # 法线（始终为零占位）
 'f_dc_0', 'f_dc_1', 'f_dc_2',     # SH DC (3)
 'f_rest_0' ... 'f_rest_N',        # SH 高阶 (45 当 sh_degree=3)
 'opacity',                         # 不透明度
 'scale_0', 'scale_1', 'scale_2',   # 缩放 (3)
 'rot_0', 'rot_1', 'rot_2', 'rot_3' # 旋转四元数 (4)
]
```

**关键细节：**
- 包含 `nx, ny, nz` 法线字段（始终为零占位）。
- `save_ply` 会过滤掉带有 NaN 位置的高斯点。
- 将法线矩阵 `normals` 拼接到属性列表中，但法线始终为零数组。
- 属性拼接顺序：`xyz -> normals -> f_dc -> f_rest -> opacities -> scale -> rotation`。

---

### 二、RTR-GS GaussianModel (`scene/gaussian_model.py`)

**文件路径：** [RTR-GS/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py)

#### 1. 核心 3DGS 属性（与 ODGS 类似但命名不同）

| 属性名 | 形状 | ODGS 对应名 | 说明 |
|--------|------|-------------|------|
| `_xyz` | (N, 3) | `_xyz` | 位置 |
| `_shs_dc` | (N, 3, 1) | `_features_dc` | **命名不同**，但含义相同（输出辐射 SH DC） |
| `_shs_rest` | (N, 3, 15) | `_features_rest` | **命名不同**，但含义相同（输出辐射 SH 高阶） |
| `_scaling` | (N, 3) | `_scaling` | 缩放 |
| `_rotation` | (N, 4) | `_rotation` | 旋转 |
| `_opacity` | (N, 1) | `_opacity` | 不透明度 |

#### 2. RTR-GS 新增的属性（始终存在）

| 属性名 | 形状 | 说明 |
|--------|------|------|
| `_diffuse_tint` | (N, 3) | 漫反射色调，经 `torch.sigmoid` 激活 |
| `_specular_tint` | (N, 3) | 镜面反射色调，经 `torch.sigmoid` 激活 |
| `_ref_tint` | (N, 3) | 反射色调，经 `torch.sigmoid` 激活 |
| `_ref_strength` | (N, 1) | 反射强度，经 `torch.sigmoid` 激活 |
| `_ref_roughness` | (N, 1) | 反射粗糙度，经 `torch.sigmoid` 激活 |
| `_specular_feature` | (N, 10) | 镜面特征向量，无激活函数 |
| `_diffuse_transfer_dc` | (N, 1, 1) | 漫反射传输 SH DC（PRT 核心） |
| `_diffuse_transfer_rest` | (N, 1, 15) | 漫反射传输 SH 高阶（PRT 核心） |

此外还有：
- `n_featres = 10`：镜面特征向量的维度。
- `base_color_scale`：形状 (3,) 的缩放因子，用于 `base_color` 和 `diffuse_tint` 的缩放。

#### 3. RTR-GS 新增的属性（仅当 `use_pbr=True` 时）

| 属性名 | 形状 | 说明 |
|--------|------|------|
| `_base_color` | (N, 3) | PBR 基础色/反照率，经 `torch.sigmoid` 激活 |
| `_roughness` | (N, 1) | PBR 粗糙度，经 `torch.sigmoid` 激活 |
| `_metallic` | (N, 1) | 金属度，经 `torch.sigmoid` 激活 |
| `_incidents_dc` | (N, 3, 1) | 入射光照 SH DC |
| `_incidents_rest` | (N, 3, 15) | 入射光照 SH 高阶 |

#### 4. `save_ply` / `load_ply` 函数签名

```python
def save_ply(self, path)      # 保存到 PLY 文件
def load_ply(self, path)      # 加载 PLY 文件
```

签名与 ODGS 相同，但内部逻辑存在很大差异。

#### 5. PLY 属性列表（`construct_list_of_attributes`）

**非 PBR 模式（默认 `render_ref`）：**

```python
['x', 'y', 'z',                              # 位置
 'f_dc_0', 'f_dc_1', 'f_dc_2',              # SH DC (输出辐射)
 'f_rest_0' ... 'f_rest_N',                  # SH 高阶 (3*15=45)
 'diffuse_tint_0', 'diffuse_tint_1', 'diffuse_tint_2',      # 漫反射色调 (3)
 'specular_tint_0', 'specular_tint_1', 'specular_tint_2',   # 镜面色调 (3)
 'ref_tint0', 'ref_tint1', 'ref_tint2',      # 反射色调 (3)     ← 注意：没有下划线
 'ref_strength',                              # 反射强度 (1)
 'ref_roughness',                             # 反射粗糙度 (1)
 'specular_feature_0' ... 'specular_feature_9',              # 镜面特征 (10)
 'diffuse_transfer_dc_0',                     # 传输 SH DC (1)
 'diffuse_transfer_rest_0' ... 'diffuse_transfer_rest_N',    # 传输 SH 高阶 (15)
 'opacity',                                   # 不透明度
 'scale_0', 'scale_1', 'scale_2',            # 缩放 (3)
 'rot_0', 'rot_1', 'rot_2', 'rot_3'           # 旋转 (4)
]
```

**PBR 模式（`render_ref_pbr` 或 `render_fast`）：** 在上述基础上额外添加：

```python
 'base_color_0', 'base_color_1', 'base_color_2',  # PBR 基础色 (3)
 'roughness',                                      # PBR 粗糙度 (1)
 'metallic',                                       # 金属度 (1)
 'incidents_dc_0', 'incidents_dc_1', 'incidents_dc_2',  # 入射SH DC (3)
 'incidents_rest_0' ... 'incidents_rest_N',              # 入射SH 高阶 (3*15=45)
```

---

### 三、核心差异对比表

| 对比维度 | ODGS (3DGS 变体) | RTR-GS |
|----------|-------------------|--------|
| **可学习参数数量** | **6** 个 | **14** 个 (基础) / **19** 个 (PBR) |
| **SH 属性命名** | `_features_dc` / `_features_rest` | `_shs_dc` / `_shs_rest` |
| **输出辐射 SH** | 有 | 有（但合并了 PRT 传输属性） |
| **额外反射属性** | 无 | `_diffuse_tint`, `_specular_tint`, `_ref_tint`, `_ref_strength`, `_ref_roughness`, `_specular_feature` |
| **PRT 传输属性** | 无 | `_diffuse_transfer_dc`, `_diffuse_transfer_rest` |
| **PBR 属性** | 无 | `_base_color`, `_roughness`, `_metallic`, `_incidents_dc`, `_incidents_rest` |
| **法线 (nx, ny, nz)** | 始终存在 (占位零) | **不存在** |
| **PLY NaN 过滤** | 有 (`save_ply` 中过滤) | 无 |
| **属性拼接顺序** | `xyz -> normals -> f_dc -> f_rest -> opacity -> scale -> rotation` | `xyz -> shs_dc -> shs_rest -> diffuse_tint -> specular_tint -> ref_tint -> ref_strength -> ref_roughness -> specular_feature -> diffuse_transfer_dc -> diffuse_transfer_rest -> opacity -> scale -> rotation` |
| **激活函数数量** | 4 个 (`scaling`, `opacity`, `rotation`, `covariance`) | **15 个** (+ `normal_activation`, `diffuse_tint_activation`, `specular_tint_activation`, `ref_tint_activation`, `ref_roughness_activation`, `ref_strength_activation`, 以及 PBR 的 `base_color_activation`, `roughness_activation`, `metallic_activation`) |
| **`ref_tint` PLY 字段命名** | N/A | `ref_tint0`, `ref_tint1`, `ref_tint2` (缺少下划线，与 `diffuse_tint_0` 格式不一致) |
| **`load_ply` 潜在 bug** | N/A | 第 718-719 行：`ref_roughness` 从 `"ref_strength"` 字段读取（应为 `"ref_roughness"`）；第 735-746 行：使用了 `incidents_dc`/`incidents_extra` 变量名而非 `diffuse_transfer_dc`/`diffuse_transfer_extra`，可能是复制粘贴导致的错误 |

---

### 四、详细函数差异

#### `save_ply` 对比

| 步骤 | ODGS | RTR-GS |
|------|------|--------|
| NaN 过滤 | 用 `np.isnan` 检查 xyz，过滤掉无效点 | **不做** NaN 过滤 |
| 法线处理 | 生成 `np.zeros_like(xyz)` 占位 | **不**包含法线 |
| SH 处理 | `_features_dc`/`_features_rest` → `transpose(1,2).flatten()` | `_shs_dc`/`_shs_rest` → `transpose(1,2).flatten()` |
| 额外属性 | 无 | 逐个提取 `diffuse_tint`, `specular_tint`, `ref_tint`, `ref_strength`, `ref_roughness`, `specular_feature`, `diffuse_transfer_dc`, `diffuse_transfer_rest` |
| PBR 属性 | 无 | 条件性添加 `base_color`, `roughness`, `metallic`, `incidents_dc`, `incidents_rest` |
| SH 传输属性 flatten | N/A | `_diffuse_transfer_dc`/`_diffuse_transfer_rest` 同样做 `transpose(1,2).flatten()` |

#### `load_ply` 对比

| 步骤 | ODGS | RTR-GS |
|------|------|--------|
| PLY 读取 | `PlyData.read(path)` | 相同 |
| xyz 读取 | `stack((x,y,z), axis=1)` | 相同 |
| SH DC 读取 | 读取 `f_dc_0/1/2` → reshape (N,3,1) | 相同，但读入后赋值给 `_shs_dc` |
| SH 高阶读取 | 读取 `f_rest_*` → reshape (N,3,SH-1) | 相同，但读入后赋值给 `_shs_rest` |
| 新属性读取 | 无 | 读取 `diffuse_tint_*`, `specular_tint_*`, `ref_strength`, `ref_roughness`, `ref_tint*`, `specular_feature_*`, `diffuse_transfer_dc_*`, `diffuse_transfer_rest_*` |
| PBR 读取 | 无 | 条件性读取 `base_color_*`, `roughness`, `metallic`, `incidents_dc_0/1/2`, `incidents_rest_*` |
| **已知 bug** | 无 | 第 718-719 行：`ref_roughness` 从 `"ref_strength"` 读取；第 735-746 行：变量名使用 `incidents_dc`/`incidents_extra` 而非 `diffuse_transfer_dc`/`diffuse_transfer_extra` |
| SH 阶数验证 | `assert len(extra_f_names) == 3*(max_sh_degree+1)^2 - 3` | 相同 |

---

### 五、关键总结

1. **命名差异：** RTR-GS 将输出辐射 SH 命名为 `_shs_dc`/`_shs_rest`，而 ODGS 采用原始 3DGS 的命名 `_features_dc`/`_features_rest`。两者语义完全相同（存储输出辐射的球谐系数）。

2. **PLY 格式核心差异：**
   - ODGS 的 PLY 包含法线字段 `nx, ny, nz`（始终为零），而 RTR-GS **不包含**法线字段。
   - RTR-GS 在 PLY 中额外存储 **8 个（非 PBR）/ 13 个（PBR）** 额外属性字段，用于反射和 PRT/PBR 参数。
   - ODGS 在保存时会过滤 NaN 位置的点，RTR-GS 不做此处理。

3. **属性总数差异：** ODGS 的 GaussianModel 仅有 6 个可学习张量属性，而 RTR-GS 在非 PBR 模式下有 14 个、PBR 模式下有 19 个可学习张量属性，是前者的 2-3 倍。

4. **RTR-GS 的 `load_ply` 中存在疑似 bug：** `ref_roughness` 被错误地从 `"ref_strength"` PLY 属性名读取（第 718-719 行），且漫反射传输属性的读取代码中变量名使用了 `incidents_*` 而非 `diffuse_transfer_*`（第 735-746 行），这可能是复制粘贴导致的问题。
        
          