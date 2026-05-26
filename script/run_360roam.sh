#!/bin/bash
# for 360Roam real scene (converted from equirect to Blender-format cube faces)
# data_dir 应指向转换后的 Blender 格式目录（含 transforms_train.json）
data_dir="./data/360Roam/base_blender"
dataset_name="360Roam"
scene_name="base"
output_path="./lab_output/"
gpu_id="0"

# 遮挡烘培参数
occlu_bound=2.5
occlu_res=128

scene_output="${output_path}/${dataset_name}/${scene_name}"
mkdir -p "${scene_output}"

export CUDA_VISIBLE_DEVICES=$gpu_id

# --densify_grad_threshold 0.0005
# -------- Stage 1: 几何 + 反射预训练 --------
python train.py --eval \
    -s "${data_dir}" \
    -m "${scene_output}/stage1" \
    --lambda_normal_render_depth 0.01 \
    --diffuse_iteration 3000 \
    --skip_eval \
    --ref_map \
    -t render_ref \
    --compute_with_prt \
    --densify_grad_threshold 0.0008

# -------- 遮挡烘培 --------
python baking.py \
    --checkpoint "${scene_output}/stage1/checkpoint/chkpnt30000.pth" \
    --bound ${occlu_bound} \
    --occlu_res ${occlu_res}

# -------- Stage 2: PBR 材质分解 --------
python train.py --eval \
    -s "${data_dir}" \
    -m "${scene_output}/stage2" \
    -c "${scene_output}/stage1/checkpoint/chkpnt30000.pth" \
    --occlusion_path "${scene_output}/stage1/checkpoint/occlusion_volumes.pth" \
    --save_training_vis \
    --iterations 40000 \
    --lambda_ref_strength_smooth 0.01 \
    --lambda_reflect_strength_equal_metallic 0.0 \
    --ref_map \
    -t render_ref_pbr \
    --compute_with_prt

# # -------- 评估 + 重光照 --------
# python render_and_eval.py \
#     -m "${scene_output}/stage2" \
#     -c "${scene_output}/stage2/checkpoint/chkpnt40000.pth" \
#     --occlusion_path "${scene_output}/stage1/checkpoint/occlusion_volumes.pth" \
#     --ref_map \
#     --compute_with_prt \
#     -t render_ref_pbr \
#     --save_video

# python eval_relighting_tensorIR.py \
#     -m "${scene_output}/stage2" \
#     -c "${scene_output}/stage2/checkpoint/chkpnt40000.pth" \
#     --occlusion_path "${scene_output}/stage1/checkpoint/occlusion_volumes.pth" \
#     -e "./data/env_maps/" \
#     --ref_map \
#     --relight \
#     --compute_with_prt \
#     -t render_ref_pbr \
#     --save_video
