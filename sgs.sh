# 运行
# bash sgs.sh

eval "$(conda shell.bash hook)"
conda activate odgs-rtr

# 切换到 SGS 目录
cd /home/huangpengyue/projects/RTR-GS/submodules/spherical-gaussian-splatting

DATA_DIR=/home/huangpengyue/projects/RTR-GS/data
OUTPUT_DIR=/home/huangpengyue/projects/RTR-GS/lab_output

# data/OmniBlender/barbershop
# export CUDA_VISIBLE_DEVICES=0,1
# 训练 base 场景（推荐先试这个）
python train.py \
    -s $DATA_DIR/OmniBlender/barbershop \
    -m $OUTPUT_DIR/OmniBlender/barbershop/sgs \
    --eval

# # 训练后可以查看结果
# # 全向渲染
# python render.py -m $OUTPUT_DIR/OmniBlender/barbershop/sgs/equi_render --iteration 30000

# # 透视渲染
# # python render_perspective.py -m $OUTPUT_DIR/OmniBlender/barbershop --iteration 30000

# # 从预设六面视角观察
# python render_pinhole.py -m $OUTPUT_DIR/OmniBlender/barbershop/sgs/pinhole --iteration 30000 --preset

# # 随机视角漫游
# python render_pinhole.py -m $OUTPUT_DIR/OmniBlender/barbershop/sgs/rand_pinhole --iteration 30000 --rand
