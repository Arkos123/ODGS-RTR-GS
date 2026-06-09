# SGS 整合与 ODGS 替换

## 背景

将 `submodules/odgs/`（旧等矩形3DGS模块）替换为 `submodules/spherical-gaussian-splatting/`（SGS，融合ODGS + omniGS）。SGS 有更完善的 equirect 渲染管线（边缘感知深度平滑、深度→法线、几何正则化损失等），重建质量显著提升（PSNR ~39）。

## 改动的核心文件

### `gaussian_renderer/render_equirect.py`
- **导入**: `odgs_gaussian_rasterization` → `spherical_gaussian_rasterization`
- **光栅化设置**: 新增 SGS 需要的 `projmatrix`, `camera_type=3`, `tanfovx/tanfovy=0`
- **移除的 ODGS workaround**: `_compute_pseudo_normal()`（改用 SGS 原生 normal_raw）、`scales.mean()` 各向同性 hack
- **新增可视化 pass**: `ref_tint` 渲染 pass，`ref_strength`/`ref_roughness`/`ref_tint` 加入 `vis_dict`
- **PBR gradient fix**: PBR blending 时 `opacity_map.detach()` 防止梯度泄漏到 opacity

### `train.py`
- **几何冻结**: equirect + 加载预训练 PLY/checkpoint 时自动冻结 xyz/scaling/rotation/opacity，禁用稠密化
- **条件**: `--ply_checkpoint`（Stage1）或 `-c + equirect`（Stage2）时触发

### 管线脚本
| 文件 | 说明 |
|------|------|
| `sgs.sh` | SGS 独立训练脚本 |
| `script/sgs2rtrgs.py` | SGS PLY → RTR-GS PLY 转换 |
| `script/run_sgs_rtr.sh` | 完整 5 阶段流水线 |

### 其他
- `.gitmodules`: 移除 ODGS 条目
- `CLAUDE.md`/`README.md`: ODGS → SGS 文档更新
- `.claude/commands/`: 命令文件英文化（commit-and-doc, save-research）

## 使用方式

```bash
# 完整流水线
bash script/run_sgs_rtr.sh

# 各阶段单独执行
conda activate odgs-rtr

# SGS 训练
cd submodules/spherical-gaussian-splatting
python train.py -s <data_dir> -m <output_path>/sgs --eval

# PLY 转换
python script/sgs2rtrgs.py <sgs_ply> <output_rtrgs_ply>

# RTR-GS Stage1 (几何自动冻结)
python train.py --eval -s <data_dir> -m <output_path>/stage1 \
    --ply_checkpoint <converted_ply> -t render_ref_equirect --compute_with_prt --ref_map

# RTR-GS Stage2 (几何自动冻结)
python train.py --eval -s <data_dir> -m <output_path>/stage2 \
    -c <stage1_checkpoint> --occlusion_path <occlusion> \
    -t render_ref_pbr_equirect --compute_with_prt --ref_map
```

## 已知限制

- PBR 分支用 `CubemapLight`（无限远环境光照），对完全封闭场景（如 barbershop）光照分解效果有限
- 封闭场景建议以 hybrid 渲染分支为主，PBR 作参考
