# Equirect 模式优化：纬度加权 Normal TV + Post-processing 清理 + Envmap 导出

## 改动

### 1. `render_equirect.py`: Normal TV 纬度面积加权

将 `row_weight`（`cos(lat)`）接入 Normal TV 平滑损失，极地平滑约束自动减弱，避免极地像素密集导致的过度平滑。

### 2. `run_sgs_rtr.sh`: Stage 2 禁用 Opacity Reset

添加 `--opacity_reset_interval 30000`（大于总迭代数），防止 Stage 2 中 opacity 骤降到 0.01 导致深度渲染异常发白。

### 3. `script/clean_gs_ply.py`: Floater 后处理清理脚本

独立的后处理脚本，从训练好的 checkpoint 中移除 floater 高斯点。支持：
- 按不透明度阈值（`--opacity`, 默认 0.01）
- 按各向异性比（`--anisotropy`, 默认 15）
- 按世界空间尺度（`--world_size_ratio`, 默认 0.2）
- 按 Z-score 离群点（`--outlier_std`, 默认禁用）
- 输出 PLY 或 `.pth` checkpoint（`--save-as ply|checkpoint`）
- `--dry-run` 预览效果

### 4. `eval_relighting_colmap.py`: Envmap PNG 导出

每个 relighting 任务输出 `test_rli/{task_name}/envmap.png`，自动应用 Reinhard tone mapping 将 HDR 压缩到可见范围。

## 关键文件

- `gaussian_renderer/render_equirect.py` — Normal TV row_weight
- `script/run_sgs_rtr.sh` — Stage 2 opacity_reset_interval
- `script/clean_gs_ply.py` — Floater 清理脚本
- `eval_relighting_colmap.py` — Envmap PNG 导出

## 后续可能改进

- `clean_gs_ply.py` 可扩展更多启发式指标（如多视角深度一致性）
- 考虑将清理脚本集成到训练流程中作为可选步骤
