# Notes: RTR-GS 重光照能力迁移到 3DGS-Editor-3.0

## 项目结构对比

### 3DGS-Editor-3.0 架构

```
3DGS_Editor-3.0/
├── main.py                  # PyQt5 入口: MainWindow
├── scene.py                 # GaussianObject + Scene (核心数据模型)
├── render.py                # GaussianRenderer + RenderWidget (PyQt5 视口)
├── control.py               # ControlWidget (左侧控制面板)
├── painter.py               # ControlPainter + GuidePainter (视口内叠加绘制)
├── camera.py                # Camera (四元数相机)
├── utils.py                 # 投影辅助函数
├── src/
│   └── gaussian_render.cu   # 自定义 CUDA 渲染内核 (render/render_point/render_mask)
└── third_party/             # SAM2 模型检查点
```

**关键类：**
- `GaussianObject`（`scene.py:6-92`）：单组高斯点云，每点属性为 positions/scales/rotations/shs_dc/shs_rest/opacities
- `Scene`（`scene.py`）：持有 `gaussian_objects` 列表、场景状态（render_mode, select_mode 等）
- `GaussianRenderer`（`render.py:22`）：统一渲染入口，4 种模式通过 `scene.render_mode` 切换
- `ControlWidget`（`control.py`）：两个 QTabWidget 标签页（渲染设置 + 对象编辑）
- `RenderWidget`（`render.py`）：QWidget，16ms 定时器（~60 FPS），处理鼠标/键盘事件

**已有渲染模式：**
1. `"gaussians"` — 自定义 CUDA 3DGS 泼溅（`gaussian_render.cu` `render_cuda`）
2. `"points"` — 自定义 CUDA 点云渲染（`render_point_cuda`）
3. `"sgs_pinhole"` — 通过 `diff_gaussian_rasterization_pinhole` 包的 pinhole 光栅化
4. `"sgs_equirect"` — 通过 `spherical_gaussian_rasterization` 包（SGS OmniGS lonlat 光栅化），渲染全景图后再提取透视视口

---

### RTR-GS 架构（重光照相关部分）

```
RTR-GS/
├── scene/
│   ├── gaussian_model.py     # GaussianModel: 基础属性 + Stage1 扩展 + Stage2 PBR 扩展
│   └── transfer_mlp.py       # TransferMLP: 视点相关镜面转移
├── gaussian_renderer/
│   ├── __init__.py           # render_fn_dict 分发
│   ├── render.py             # 透视模式（PRT + deferred reflection + PBR）
│   ├── render_fast.py        # 透视轻量版
│   └── render_equirect.py    # 全景模式（SGS rasterizer + PRT + forward reflection + PBR）
├── pbr/
│   ├── light.py              # CubemapLight（训练/可切换环境贴图 + mipmap）
│   ├── shade.py              # BRDF: split-sum 反射, Cook-Torrance PBR
│   └── __init__.py
├── utils/
│   └── prt_utils.py          # PRTutils: 漫反射/镜面 PRT 颜色计算
├── arguments/
│   └── __init__.py           # 所有命令行参数定义
├── baking.py                 # 遮挡体素烘焙（可选，用于间接光照）
├── train.py                  # 两阶段训练流水线
└── script/
    └── sgs2rtrgs.py          # SGS → RTR-GS PLY 转换（补全默认扩展属性）
```

---

## 关键技术细节

### 1. GaussianModel 属性体系

| 属性 | Shape | 所属阶段 | 激活 | RTR-GS PLY 字段名 |
|------|-------|---------|------|------------------|
| `_xyz` | [N,3] | 基础 | 无 | x, y, z |
| `_shs_dc` | [N,3,1] | 基础 | 无 | f_dc_0, f_dc_1, f_dc_2 |
| `_shs_rest` | [N,3,15] | 基础 | 无 | f_rest_0 ~ f_rest_44 |
| `_opacity` | [N,1] | 基础 | sigmoid | opacity (logit 存储) |
| `_scaling` | [N,3] | 基础 | exp | scale_0, scale_1, scale_2 (log 存储) |
| `_rotation` | [N,4] | 基础 | normalize | rot_0, rot_1, rot_2, rot_3 |
| `_diffuse_tint` | [N,3] | Stage1 | sigmoid | diffuse_tint_0,1,2 |
| `_specular_tint` | [N,3] | Stage1 | sigmoid | specular_tint_0,1,2 |
| `_ref_tint` | [N,3] | Stage1 | sigmoid | ref_tint_0,1,2 |
| `_ref_strength` | [N,1] | Stage1 | sigmoid | ref_strength |
| `_ref_roughness` | [N,1] | Stage1 | sigmoid | ref_roughness |
| `_specular_feature` | [N,10] | Stage1 | 无 | specular_feature_0 ~ _9 |
| `_diffuse_transfer_dc` | [N,1,1] | Stage1 | 无 | diffuse_transfer_dc_0 |
| `_diffuse_transfer_rest` | [N,1,15] | Stage1 | 无 | diffuse_transfer_rest_0 ~ _14 |
| `_base_color` | [N,3] | Stage2 | sigmoid | base_color_0,1,2 |
| `_roughness` | [N,1] | Stage2 | sigmoid | roughness |
| `_metallic` | [N,1] | Stage2 | sigmoid | metallic |
| `_incidents_dc` | [N,3,1] | Stage2 | 无 | incidents_dc_0,1,2 |
| `_incidents_rest` | [N,3,15] | Stage2 | 无 | incidents_rest_0 ~ _44 |

### 2. SGS PLY 格式（原始训练结果）

只包含 3DGS 基础字段 + SH 颜色：
```
x, y, z, nx, ny, nz,
f_dc_0, f_dc_1, f_dc_2,
f_rest_0 ~ f_rest_44,
opacity,
scale_0, scale_1, scale_2,
rot_0, rot_1, rot_2, rot_3
```
→ 共 59 个 float32 字段（含法线 nx,ny,nz，但 RTR-GS 的 `load_ply` 从第 3 个字段开始解析）

**sgs2rtrgs.py 的默认值：**
- `diffuse_tint` = torch.ones(N, 3) * 0.5
- `specular_tint` = torch.ones(N, 3) * 0.5
- `ref_tint` = torch.ones(N, 3) * 0.5
- `ref_strength` = torch.ones(N, 1) * 0.1
- `ref_roughness` = torch.ones(N, 1) * 0.4
- `specular_feature` = torch.zeros(N, 10)
- `diffuse_transfer_dc` = torch.ones(N, 1, 1)
- `diffuse_transfer_rest` = torch.zeros(N, 1, 15)

### 3. PRT 渲染流程

**view-independent 漫反射 PRT:**
```
C_d = diffuse_tint * ReLU(Σ(c_j * c_j^t) + 0.5)
```
其中 c_j = 场景 SH 光照系数，c_j^t = 逐高斯漫反射转移系数

**view-dependent 镜面 PRT:**
```
r = reflect(-v, n)     // 反射方向
LT = TransferMLP(specular_feature, r)  // → 16 SH 系数
C_s = specular_tint * ReLU(Σ(LT * scene_sh))
```
TransferMLP 架构：`Linear(3→64) → ReLU → Concat(spec_feature, 64→64) → ReLU → Linear(64→16)`

### 4. 反射渲染

split-sum 近似：
```
reflectance = specular_tint * LUT_scale + LUT_bias
specular_rgb = env_sample(reflection_dir, roughness) * reflectance
```
- LUT 是预计算的 BRDF 积分查找表
- env_sample 从 CubemapLight 的 mipmap 中采样

### 5. PBR 渲染

Cook-Torrance BRDF:
```
diffuse = (1-metallic) * albedo / π * diffuse_light * occlusion
F0 = lerp(0.04, albedo, metallic)
specular = specular_light * (F0 * LUT_scale + LUT_bias)
render_rgb = diffuse + specular
```

### 6. 重光照模式

| `pipe.relight` | `pipe.transfer_light` | 行为 |
|---|---|---|
| False | — | 使用存储的 incident SH（训练时固定的光照） |
| True | True | incidents = cubemap.shs * pc.get_incidents（相乘实现 relighting） |
| True | False | 无入射光（zero） |

### 7. equirect 渲染 bridge 注意事项

3DGS-Editor 已有 `render_sgs_equirect()` 使用 `spherical_gaussian_rasterization` 但不带 extra_features。RTR-GS 的 `render_equirect.py` 使用 V2 multi-channel extra_features 单次光栅化所有属性。

**核心差异：** 3DGS-Editor 的 SGS equirect 渲染只用 SH 颜色，而 RTR-GS 额外光栅化了 normal / ref_* / PBR 属性。需要确认 Editor 的 SGS rasterizer 版本是否支持 extra_features 参数。若不支持，需要升级 SGS 子模块或回到多 pass 方式。

---

## 关键设计决策点

1. **渲染集成方式**
   - 方案 A：直接在 3DGS-Editor 中 import RTR-GS 模块（如 `sys.path.append` + `from gaussian_renderer.render_equirect import ...`）
   - 方案 B：将 RTR-GS 渲染核心封装成独立的 Python package，通过 pip install 引入
   - 方案 C：将需要的代码复制/重构到 3DGS-Editor 中

2. **3DGS-Editor 已有的 SGS submodule**
   - 项目已依赖 `spherical_gaussian_rasterization` 包（`render_sgs_equirect`）
   - 需检查 extra_features 支持情况，若不支持需要升级或回退到多 pass

3. **GUI 架构**
   - `control.py` 目前只有两个标签页，可以新增第三个"光照编辑"标签页
   - 光源编辑可采用与对象编辑类似的 gizmo 交互模式

---

## Phase 1-2 实现记录（2026-07-06）

### 提交 `1a42b4b` (3DGS_Editor-3.0)

### 改动的文件

**scene.py：**
- 新增模块级 `_RTRGS_EXT_ATTRS` 常量（13 个扩展属性名）
- 新增 `_read_ply_with_format()` — 单次 PLY 读取 + 格式自动检测（standard_3dgs / rtr_gs_stage1 / rtr_gs_stage2）
- 新增 `_detect_ply_format(properties)` — 从属性名列表判断格式，用多字段联合判定加固
- `GaussianObject.__init__` — 新增 13 个扩展属性字段（diffuse_tint, specular_tint, ref_tint, ref_strength, ref_roughness, specular_feature, diffuse_transfer_dc, diffuse_transfer_rest, base_color, roughness, metallic, incidents_dc, incidents_rest）+ `has_extended_attrs` 标记
- 重写 `load_from_file()` — 动态 PLY 头部解析，格式感知加载
- 新增 `Scene._copy_extended_attrs()` — 按索引复制扩展属性（选择传播）
- 新增 `Scene._merge_extended_attrs()` — torch.cat 合并扩展属性

**control.py：**
- 新增类级 `_EXT_ATTR_SPEC` — 13 个扩展属性的规格表（stem, channels, is_pbr, ply_name）
- 新增 `_collect_attr()` — 统一收集单属性（标量返回 [N]，多通道返回 [N, C]）
- 重写 `export_merged_ply()` — dict-of-lists 驱动收集/赋值，格式感知导出

**render.py：**
- 修复 SGS equirect rasterizer 返回值解包（第 2 位 `extra` → `_`，第 3 位 `radii` → `_`）

### 存储约定
- 扩展属性存 raw 值（预激活），与 PLY 文件一致
- 属性不存在时为 None（标准 3DGS 格式）
- 选择/合并操作只复制非 None 的属性

### 验证结果
- Stage2（149 字段）加载 ✅
- Stage1（96 字段）加载 ✅
- 标准 3DGS（59 字段）向后兼容 ✅
- 选择子集 → 扩展属性保留 ✅
- 导出 → 再导入 → 属性值一致 ✅

