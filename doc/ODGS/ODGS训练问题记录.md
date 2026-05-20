# ODGS 训练问题记录

记录在运行 ODGS 训练时遇到的问题及解决方案。

---

## 问题 1：protobuf 版本不兼容

**现象：** 运行 `python train.py` 时立即报错：

```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date
and must be regenerated with protoc >= 3.19.0.
```

**原因：** `environment.yml` 固定了 `tensorboard=2.10.0`，但 conda 安装时没有固定 protobuf 版本，导致装上了 protobuf 6.x。tensorboard 2.x 使用旧版 protobuf 生成的 `_pb2.py` 文件，与新版本不兼容。

**解决方案：** 将 protobuf 降级到 3.20.x：

```bash
pip install "protobuf>=3.20,<4"
```

**影响范围：** 所有使用 tensorboard 的训练（ODGS 和 RTR-GS）。

---

## 问题 2：`dtype=np.byte` 导致 PIL 图像加载失败

**现象：** 运行 `python train.py` 时，在读取数据集阶段报错：

```
File "scene/dataset_readers.py", line 306, in readCamerasFromOpenMVG
    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
TypeError: Cannot handle this data type: (1, 1, 3), |i1
```

**原因：** `np.byte` 等价于 `np.int8`（有符号 8 位整数，范围 -128~127）。像素值范围是 0~255，当值超过 127 时会溢出变为负数。PIL 的 `"RGB"` 模式要求 `np.uint8`（无符号 8 位整数），不接受 `np.int8`。

**解决方案：** 将 `dtype=np.byte` 改为 `dtype=np.uint8`。

**涉及文件（共 4 处）：**

| 文件 | 行号 | 函数 |
|------|------|------|
| `scene/dataset_readers.py` | 230 | `readCamerasFromTransforms`（Colmap 格式） |
| `scene/dataset_readers.py` | 306 | `readCamerasFromOpenMVG`（OpenMVG 格式） |
| `scene_perspective/dataset_readers.py` | 116 | 透视渲染读取（OpenMVG） |
| `scene_perspective/dataset_readers.py` | 214 | 透视渲染读取（带 ERP 名称） |

**根因分析：** 这是 ODGS 代码本身的 bug，与依赖版本无关。在所有 NumPy + Pillow 组合下都会出现。原版 ODGS 的 conda environment 中同样会触发此错误。

---

## 环境状态（修复后）

| 组件 | 版本 |
|------|------|
| protobuf | 3.20.x |
| tensorboard | 2.10.0 |
| CUDA | 11.8 |
| PyTorch | 2.1.2 |
