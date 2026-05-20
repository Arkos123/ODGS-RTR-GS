conda activate odgs-rtr

# 切换到 odgs 目录
cd /home/huangpengyue/projects/RTR-GS/submodules/odgs

# export CUDA_VISIBLE_DEVICES=0,1
# 训练 base 场景（推荐先试这个）
python train.py \
    -s /home/huangpengyue/projects/RTR-GS/data/360Roam/base \
    -m ./output/360Roam/base \
    --eval

# 训练后可以查看结果
# 全向渲染
python render.py -m ./output/360Roam/base --iteration 30000

# 透视渲染
python render_perspective.py -m ./output/360Roam/base --iteration 30000

# 从预设六面视角观察
python render_pinhole.py -m ./output/360Roam/base --iteration 30000 --preset

# 随机视角漫游
python render_pinhole.py -m ./output/360Roam/base --iteration 30000 --rand