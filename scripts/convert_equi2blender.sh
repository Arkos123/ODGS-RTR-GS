#!/bin/bash
set -e
cd "$(dirname "$0")/.."
# bash scripts/convert_equi2blender.sh

# ── 配置（按需修改）──
SRC=data/OmniBlender/barbershop
DST=data/OmniBlender/barbershop_blender
FACE_SIZE=800
FACES="F B L R U D"
STEP=0
PITCH=0        # 单个值=固定, 多个值空格分隔=按view循环, 留空=无偏移
YAW="0"         # 同上
# ──────────────────

python scripts/equi2blender.py \
    --source_path "$SRC" \
    --output_path "$DST" \
    --face_size "$FACE_SIZE" \
    --faces $FACES \
    --step "$STEP" \
    ${PITCH:+--pitch $PITCH} \
    ${YAW:+--yaw $YAW} \
    --force
