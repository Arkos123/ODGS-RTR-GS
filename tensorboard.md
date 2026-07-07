SGS 自带 TensorBoard 日志，直接跑就行：


  tensorboard --logdir ./lab_output/OmniBlender/barbershop/stage1-new --port 6006

  tensorboard --logdir ./lab_output/OmniBlender/barbershop/stage1 --port 6006

  tensorboard --logdir /home/huangpengyue/projects/RTR-GS/lab_output/mipnerf/kitchen/stage1 --port 6006

  tensorboard --logdir=/home/huangpengyue/data/fromA6000/kitchen --port 6006
  tensorboard --logdir=/home/huangpengyue/projects/RTR-GS/lab_output/mipnerf/kitchen/stage1 --port 6006

  tensorboard --logdir=/home/huangpengyue/data/fromA6000/home/wangzichen/lab_output/rtr-gs-3-19-full/360_v2/kitchen/stage2 --port 6006
  tensorboard --logdir=/home/huangpengyue/projects/RTR-GS/lab_output/mipnerf/kitchen/stage2 --port 6006
  

  SGS 记录了很多指标，包括：
  - 损失曲线：train_loss_patches/loss、train_loss_patches/ssim
  - 几何正则损失：depth_thickness、planar_depth、ray_plane、normal_smooth 等
  - 渲染评估：PSNR、SSIM、LPIPS
  - 可视化图：render、depth、normal

  如果你在远程服务器上跑，用 SSH 隧道看：

  ssh -L 6006:localhost:6006 huangpengyue@10.108.11.10

  然后浏览器打开 http://localhost:6006。