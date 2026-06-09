#!/bin/bash
# ============================================================
# SGS + RTR-GS 全景图（等距柱状投影）训练流程
# ============================================================
# 完整5阶段流程：
#   1. SGS 训练            — 从全景图重建3DGS几何
#   2. PLY 转换            — SGS → RTR-GS格式
#   3. RTR-GS Stage1      — 几何+反射预训练（等距柱状）
#   4. 遮挡烘培            — 预计算可见性
#   5. RTR-GS Stage2      — PBR材质分解（等距柱状）
# ============================================================
# 数据集要求（OpenMVG格式）:
#   <data_dir>/data_extrinsics.json, data_views.json,
#   train.txt, test.txt, pcd.ply, images/
# ============================================================
# 使用方法：修改下方参数后直接运行
#  conda activate odgs-rtr
#  bash script/run_sgs_rtr.sh
# ============================================================

set -e

# ===================== 用户配置区域 =====================
# 数据集路径（包含 data_extrinsics.json 的目录）

# 自动生成路径
dataset_name="OmniBlender" # # 360Roam
scene_name="barbershop" # base
data_dir="/home/huangpengyue/projects/RTR-GS/data/${dataset_name}/${scene_name}"
# 输出根目录
output_path="./lab_output"
scene_output="${output_path}/${dataset_name}/${scene_name}" # lab_output/360Roam/base

# GPU ID
gpu_id="0"

# SGS SH degree（通常为3）
sgs_sh_degree=3

# 遮挡烘培参数
occlu_res=128
occlu_bound=2.0
# ========================================================

echo "=========================================="
echo " SGS + RTR-GS 全景图训练"
echo " 数据: ${data_dir}"
echo " 输出: ${scene_output}"
echo " GPU: ${gpu_id}"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$gpu_id

# -------- Step 1: SGS 训练 --------
# echo ""
# echo ">>> [1/5] SGS 训练（等距柱状3DGS几何重建）"

# mkdir -p "${scene_output}/sgs"
# cd submodules/spherical-gaussian-splatting
# python train.py \
#     -s "${data_dir}" \
#     -m "${scene_output}/sgs" \
#     --eval \
#     --iterations 30000
# cd - > /dev/null

# 查找SGS输出的PLY
sgs_ply="${scene_output}/sgs/point_cloud/iteration_30000/point_cloud.ply"
if [ ! -f "$sgs_ply" ]; then
    sgs_iter_dir=$(ls -d "${scene_output}/sgs/point_cloud/iteration_"* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    if [ -n "$sgs_iter_dir" ]; then
        sgs_ply="${sgs_iter_dir}/point_cloud.ply"
    else
        echo "错误：未找到 SGS 输出的 PLY 文件"
        exit 1
    fi
fi

# -------- Step 2: PLY 转换 --------
echo ""
echo ">>> [2/5] PLY 转换（SGS → RTR-GS）"

converted_ply_dir=$(dirname "$sgs_ply")
converted_ply="${scene_output}/sgs/$(basename "$converted_ply_dir")/point_cloud_rtrgs.ply"
# python script/sgs2rtrgs.py "$sgs_ply" "$converted_ply" --sh_degree ${sgs_sh_degree}

# -------- Step 3: RTR-GS Stage1 等距柱状训练 --------
echo ""
echo ">>> [3/5] RTR-GS Stage1（几何+反射预训练）"

# python train.py --eval \
#     -s "${data_dir}" \
#     -m "${scene_output}/stage1" \
#     --ply_checkpoint "${converted_ply}" \
#     --data_device cpu \
#     --lambda_normal_render_depth 0.01 \
#     --lambda_normal_smooth 0.01 \
#     --diffuse_iteration 3000 \
#     --checkpoint_interval 4000 \
#     --skip_eval \
#     --ref_map \
#     -t render_ref_equirect \
#     --vis_interval 1000 \
#     --compute_with_prt \
#     --without_normal_propagation

# # -------- Step 4: 遮挡烘培 --------
# echo ""
# echo ">>> [4/5] 遮挡烘培"

# python baking.py \
#     --checkpoint "${scene_output}/stage1/checkpoint/chkpnt30000.pth" \
#     --bound ${occlu_bound} \
#     --occlu_res ${occlu_res}

# -------- Step 5: RTR-GS Stage2 等距柱状 PBR 训练 --------
echo ""
echo ">>> [5/5] RTR-GS Stage2（PBR材质分解）"

python train.py --eval \
    -s "${data_dir}" \
    -m "${scene_output}/stage2" \
    -c "${scene_output}/stage1/checkpoint/chkpnt30000.pth" \
    --occlusion_path "${scene_output}/stage1/checkpoint/occlusion_volumes.pth" \
    --data_device cpu \
    --skip_eval \
    --iterations 40000 \
    --checkpoint_interval 3000 \
    --lambda_ref_strength_smooth 0.01 \
    --ref_map \
    --vis_interval 1000 \
    -t render_ref_pbr_equirect \
    --compute_with_prt \
    --without_normal_propagation

echo ""
echo "=========================================="
echo "训练完成！最终模型:"
echo "  ${scene_output}/stage2/checkpoint/chkpnt40000.pth"
echo "=========================================="
