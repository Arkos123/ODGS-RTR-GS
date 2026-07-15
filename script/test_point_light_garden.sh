#!/bin/bash
# 在 garden 场景上测试点光源重光照（perspective 模式）
# bash script/test_point_light_garden.sh
#
# 分辨率选项：
#   -i images_2   使用 2x 下采样图像（更快）
#   -r 2          输出缩小一半

set -e
cd "$(dirname "$0")/.."

# ── 路径配置 ──────────────────────────────────────────
DATA="/home/huangpengyue/projects/RTR-GS/data/mipnerf/garden"
ODIR="/home/huangpengyue/projects/RTR-GS/data/fromA6000/home/wangzichen/lab_output/rtr-gs-3-19-full/360_v2/garden"
STAGE2_CKPT="$ODIR/stage2/checkpoint/chkpnt40000.pth"
OCCLUSION="$ODIR/stage1/checkpoint/occlusion_volumes.pth"
ENVMAP_DIR="/home/huangpengyue/projects/RTR-GS/data/env_maps/"

# ── conda 环境 ──────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate odgs-rtr 2>/dev/null || source activate odgs-rtr

# ── 运行 ────────────────────────────────────────────────
python eval_relighting_colmap.py --eval \
    -s "$DATA" \
    -m "$ODIR" \
    --data_device cpu \
    -c "$STAGE2_CKPT" \
    -i images_4 \
    --occlusion_path "$OCCLUSION" \
    -e "$ENVMAP_DIR" \
    --ref_map \
    -t render_ref_pbr_equirect \
    --compute_with_prt \
    --point_lights_config test_data/test_point_light.json \
    --point_light_vis \
    --save_video --full_video_output

echo "Done. Output in $OUTPUT/test_rli/"
