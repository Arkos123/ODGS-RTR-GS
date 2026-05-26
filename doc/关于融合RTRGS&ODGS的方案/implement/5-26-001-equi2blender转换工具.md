# Equirectangular → Blender 透视数据集转换工具

## 背景

ODGS 的全景图（等距柱状投影）数据是 OpenMVG 格式（`data_extrinsics.json` + `data_views.json`），这种 equirectangular 投影存在极地畸变、像素密度不均等问题，影响重建质量。

本工具将 OpenMVG 全景数据转换为 RTR-GS 原生支持的 Blender/NeRF-Synthetic 格式（4 面或 6 面 CubeMap），从而可以用标准 pinhole 渲染器 + RTR-GS 完整流水线训练。

## 核心文件

| 文件 | 说明 |
|------|------|
| `scripts/equi2blender.py` | Python 转换脚本 |
| `scripts/convert_equi2blender.sh` | Shell 封装（修改顶部变量即可运行） |
| `script/run_360roam.sh` | 转换后数据的训练脚本示例 |

## 转换原理

### 坐标变换

对于每张全景图，OpenMVG 提供 `rotation`（世界→相机旋转）和 `center`（相机位置）：

```python
R_c2w = rotation.T          # 相机→世界
R_face_c2w = R_c2w @ R_offset @ face_rot[face_name]
```

其中 `R_offset` 支持 `--pitch`/`--yaw` 视角偏移，`face_rot` 是 cube face 朝向矩阵。

### 图像采样

用 OpenCV `cv2.remap` 从 equirect 图像中采样 cube face：
- 每个 face 像素计算 3D 方向向量 → 旋转到相机坐标系 → 经纬度映射到 equirect 坐标
- 插值方式：`INTER_LANCZOS4`（高质量）

### JSON 格式

输出 `transforms_train.json` / `transforms_test.json`，与 Blender/NeRF-Synthetic 格式完全兼容：

```json
{
    "camera_angle_x": 1.5707963,
    "frames": [
        {
            "file_path": "./images/0_0000_F",
            "transform_matrix": [[4x4]]
        }
    ]
}
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source_path` | 必填 | OpenMVG 数据集路径 |
| `--output_path` | 必填 | 输出路径 |
| `--face_size` | 1024 | Cube face 分辨率（正方形） |
| `--faces` | F B L R | Cube face 列表（可选 U D） |
| `--step` | 1 | 每隔 N 个视角处理 1 个 |
| `--pitch` | 0 | 俯仰偏移角度（正=抬头） |
| `--yaw` | 0 | 偏航偏移角度（正=右转） |
| `--force` | false | 覆盖已存在的输出目录 |

### face_size 推荐

| 原始全景图 | Cube Face | 说明 |
|-----------|-----------|------|
| 4096×2048 | 1024×1024 | 水平角分辨率匹配 |
| 2048×1024 | 512×512 | 同上比例 |
| 8192×4096 | 2048×2048 | 保持细节 |

### step 采样行为

先按文件名排序（`0_0000 < 0_0001 < ...`，相邻帧视角接近），再均匀 step 采样。

### 视角偏移（pitch/yaw）

整个 cube 作为刚体旋转，F 抬头 → B 自然低头，L/R 不受 pitch 影响，保持 multi-view 一致性。

## 使用方法

```bash
# 1. 修改 scripts/convert_equi2blender.sh 中的配置
# 2. 运行（需在 odgs-rtr 环境中）
bash scripts/convert_equi2blender.sh

# 或直接调用
python scripts/equi2blender.py \
    --source_path data/360Roam/base \
    --output_path data/360Roam/base_blender \
    --face_size 1024 \
    --faces F B L R \
    --step 2 \
    --force

# 3. 用 RTR-GS 训练
bash script/run_360roam.sh
```

## 注意事项

1. 输出目录不含 `train.txt`/`test.txt`（RTR-GS Blender reader 只读 `transforms_*.json`）
2. 点云从 OpenMVG 的 `colorized.ply` 或 `pcd.ply` 自动复制
3. 若无点云会自动生成随机点云（Blender reader 行为）
4. 评估脚本用 `render_and_eval.py` + `eval_relighting_tensorIR.py`（Blender 格式和合成数据一致）
