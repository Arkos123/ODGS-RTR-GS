# Occlusion Baking 墙壁跳过与自适应网格

## 日期

2026-06-10

## 背景

在完全封闭的室内场景（四面墙壁+天花板+地板）中，Stage 1 后的 occlusion baking 会遇到核心问题：从场景内部任意体素沿任意方向发射的射线，最终都会碰到墙壁 Gaussian。`occlu_mask` 全为 0，导致 PBR 光照公式 `diffuse_light = env_light * occlusion + (1-occlusion) * irradiance` 退化为 `diffuse_light = irradiance`，环境光照被完全绕过，材质-光照分解效果不佳。

此外，场景坐标不一定居中或标准化（如 OmniBlender 数据集），而 occlusion volume 的体素网格默认使用对称的 `[-bound, bound]³`，大量体素落在空白区域，分辨率浪费。

## 改动概览

### 1. `--skip_walls`：墙壁检测（`baking.py`）

基于距离阈值的墙壁检测，在现有单次 cubemap 渲染基础上，通过**射线到场景实际边界的出口距离**来判断墙壁。

**算法**：
- 从体素位置渲染 cubemap（不变）
- 计算每个方向的命中点坐标 `hit_pos = voxel_pos + dir * depth`
- 计算命中点到场景 AABB 边界的最短距离
- 如果该距离 < `wall_margin` → 命中点在场景边界附近 → 是墙壁 → 视为未被遮挡（`mask=1`）
- 场景实际范围由 Gaussians 位置的百分位数确定（`--extent_percentile`），避免离群点干扰

**相比之前 `t_out * threshold` 方案的改进**：
- 阈值有明确物理意义：场景单位距离，而非距离比例
- 不受射线方向影响：同一墙面上不同角度的命中点，距离边界一致
- 更直观：`wall_margin=0.3` 即"离场景边界 0.3 单位内的表面是墙"

### 2. `--auto_bound`：自适应网格范围（`baking.py`）

自动根据场景实际范围计算 occlusion volume 的网格 AABB，无需手动调 `--bound`。

**算法**：
- 从 Gaussians 位置计算 `scene_min/max`（百分位统计）
- 加 padding 后得到 `aabb = [scene_min - pad, scene_max + pad]`
- 同步将 `--valid` 设为相同值（确保 cubemap 渲染能看到完整场景）

### 3. 非对称 AABB 支持（`baking.py` + 所有加载点）

当 `--auto_bound` 启用时，`.pth` 文件中额外保存 `"aabb"` 字段（6 元素 `[min_x, min_y, min_z, max_x, max_y, max_z]`）。
同时保留 `"bound"` 字段（最大对称半边长）向后兼容。

所有加载 occlusion volumes 的代码点已更新为优先读取 `"aabb"`，没有则 fallback 到对称 `[-bound, bound]³`。

**涉及文件**：
- `train.py` — Stage 2 训练加载
- `render.py`、`render_fast.py`、`render_equirect.py` — 渲染时的点 clamp
- `render_and_eval.py`、`render_checkpoint.py` — 评估
- `eval_relighting_*.py` — 光照迁移评估
- `viewer_pygame.py` — 交互查看器

### 4. Equirect PBR Incident Light 补全（`render_equirect.py`）

`render_equirect.py` 原始的 PBR 路径缺失了 incident light（每高斯局部 irradiance）的光栅化和传递，原因是 ODGS/SGS 光栅化器不支持原始 RTR-GS 的 feature tensor 打包机制，只能用多 pass 渲染。之前只实现了 base_color、roughness、metallic 的渲染 pass，incident light 被遗漏。

**修改**：
- 新增第 6 个光栅化 Pass（incident light，3 通道）
- 非 relight 模式：评估 `pc.get_incidents` SH 系数 → RGB
- Relight+transfer 模式：`cubemap.shs * transfer_shs` → RGB
- 结果传给 `pbr_shading(irradiance=incident_light_map)`

同时修复了 clamp 硬编码 1.5 的问题（改用 AABB）、补全了 vis_dict（incidents_light、incident_light_raw、env_export 等）。

## 新增 CLI 参数

| 参数 | 默认值 | 所属文件 | 说明 |
|------|--------|---------|------|
| `--skip_walls` | False | `baking.py` | 墙壁检测开关 |
| `--wall_margin` | 0.3 | `baking.py` | 距离阈值（场景单位），命中点距 AABB 边界小于此值则视为墙壁 |
| `--vis_walls` | False | `baking.py` | 墙壁检测可视化：从场景中心渲染全景图，红色=墙壁、绿色=非墙壁、白色=背景，保存为 PNG 后退出 |
| `--extent_percentile` | 0.01 | `baking.py` | 场景范围百分位（切掉两端离群点） |
| `--auto_bound` | False | `baking.py` | 自适应网格范围 |
| `--bound_padding` | 1.1 | `baking.py` | 网格范围余量因子 |

## 向后兼容

- 不加 `--skip_walls` 时，`baking.py` 的 occlusion mask 生成逻辑不变
- 不加 `--auto_bound` 时，`.pth` 不保存 `"aabb"` 字段，格式与之前完全一致
- 所有加载点优先读 `"aabb"`，没有则 fallback 到对称 bound
- `render.py/fast/equirect` 的 clamp 优先用 AABB，没有则 fallback 到 bound

## 使用方式

```bash
# 封闭室内场景（barbershop），完整参数
python baking.py \
    --checkpoint lab_output/OmniBlender/barbershop/stage1/checkpoint/chkpnt30000.pth \
    --auto_bound \
    --occlu_res 96 \
    --skip_walls \
    --wall_margin 0.3 \
    --extent_percentile 0.01

# 普通场景（不使用新功能）
python baking.py \
    --checkpoint .../chkpnt30000.pth \
    --bound 1.5 \
    --occlu_res 128
```