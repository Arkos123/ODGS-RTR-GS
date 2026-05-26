#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# ── 配置（按需修改）──
SRC=data/360Roam/base
DST=data/360Roam/base_blender
# FACE_SIZE=1024
FACE_SIZE=800
FACES="F B L R"
STEP=2         # 每 N 个视角处理一次（1=全部，3=每3个处理1个）
PITCH=0        # 俯仰偏移角度（正=抬头）
YAW=0          # 偏航偏移角度（正=右转）
# ──────────────────

python scripts/equi2blender.py \
    --source_path "$SRC" \
    --output_path "$DST" \
    --face_size "$FACE_SIZE" \
    --faces $FACES \
    --step "$STEP" \
    --pitch "$PITCH" \
    --yaw "$YAW" \
    --force
