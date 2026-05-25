```
feat: add ODGS PLY bridge and OpenMVG equirectangular camera support

Phase 1 - PLY Bridge:
- Add script/odgs2rtrgs.py to convert ODGS .ply (6 attrs) to RTR-GS .ply (16+ attrs)
- Fix 4 variable name bugs in scene/gaussian_model.py load_ply (ref_roughness,
  incidents_dc/incidents_extra -> diffuse_transfer_dc/diffuse_transfer_extra)
- Fix ref_tint naming in construct_list_of_attributes (add missing underscore)

Phase 2 - OpenMVG Camera Support:
- Add readCamerasFromOpenMVG and readOpenMVGInfo to scene/dataset_readers.py
- Register OpenMVG format in sceneLoadTypeCallbacks
- Add data_extrinsics.json detection in scene/__init__.py"
```
阶段1和阶段2实施完成。以下是变更的详细说明：
---

## 阶段1：PLY 桥接

### 新增文件：[script/odgs2rtrgs.py](file:///home/huangpengyue/projects/RTR-GS/script/odgs2rtrgs.py)

将 ODGS 训练输出的 `.ply`（6属性：xyz, SH, scale, rot, opacity）转换为 RTR-GS 格式（16属性），并自动初始化扩展属性为默认值：

```bash
# 用法1: 直接传入PLY文件
python script/odgs2rtrgs.py /path/to/odgs/point_cloud.ply /path/to/output.ply

# 用法2: 传入模型路径+迭代次数
python script/odgs2rtrgs.py -m /path/to/odgs/model -i 30000 -o /path/to/output
```

扩展属性初始化默认值：
- `ref_strength` = sigmoid⁻¹(0.01) — 极低的反射强度
- `ref_roughness` = sigmoid⁻¹(0.65) — 中等粗糙度
- `diffuse_tint`, `specular_tint`, `ref_tint`, `specular_feature` = 0
- `diffuse_transfer_dc/rest` = 0

### 修复文件：[scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py)

修复了 `load_ply` 中的 **4处bug**：

| 位置 | 原代码 | 问题 | 修复 |
|------|-------|------|------|
| L719 | `ref_roughness` 读取 `"ref_strength"` | 复制粘贴错误，读到错误属性 | 改为读取 `"ref_roughness"` |
| L735 | 读取 `"incidents_dc_0"` | 错误的属性名（PBR的incidents混入了diffuse_transfer） | 改为 `"diffuse_transfer_dc_0"` |
| L742 | 引用未定义的 `extra_incidents_names` | 变量名错误 | 改为 `extra_diffuse_transfer_names` |
| L746 | 引用未定义的 `incidents_extra.shape[0]` | 变量名错误 | 改为 `diffuse_transfer_extra.shape[0]` |
| L779-783 | `incidents_dc`/`incidents_extra`→Parameter | 变量名错误 | 改为 `diffuse_transfer_dc`/`diffuse_transfer_extra` |
| L540 | `ref_tint{}`无下划线 | 与load_ply的`startswith("ref_tint")`解析不兼容 | 改为 `ref_tint_{}` |

---

## 阶段2：等距柱状相机支持

### 修改文件：[scene/dataset_readers.py](file:///home/huangpengyue/projects/RTR-GS/scene/dataset_readers.py)

新增两个函数：

- **`readCamerasFromOpenMVG()`** — 解析 `data_extrinsics.json`，读取全景图位姿和图像，设置 `FoVx ≈ π`（等距柱状投影特征）
- **`readOpenMVGInfo()`** — 完整的数据集读取器，读取 `data_views.json`（文件名映射）+ `train.txt/test.txt`（划分）+ `pcd.ply`/`colorized.ply`（点云）

同时注册 `"OpenMVG"` 到 `sceneLoadTypeCallbacks`。

### 修改文件：[scene/__init__.py](file:///home/huangpengyue/projects/RTR-GS/scene/__init__.py)

在 `Scene.__init__` 的数据集检测逻辑中新增分支：

```python
elif os.path.exists(os.path.join(source_path, "data_extrinsics.json")):
    # 自动识别为 OpenMVG 全景图格式
    scene_info = sceneLoadTypeCallbacks["OpenMVG"](...)
```

RTR-GS 现在支持的全景图数据集目录结构：
```
dataset_root/
├── data_extrinsics.json    # OpenMVG 外参
├── data_views.json         # 视图→文件名映射
├── train.txt               # 训练集图像名列表
├── test.txt                # 测试集图像名列表
├── pcd.ply 或 colorized.ply # 初始点云
└── images/                 # 全景图文件
```

---

这两步完成后，RTR-GS 已经能够加载 ODGS 输出的点云（通过 PLY 转换脚本）和 OpenMVG 格式的全景图数据集（通过场景自动检测）。后续的阶段3和4（等距柱状渲染器和训练流水线）需要替换 CUDA 光栅化器并实现多通道前向渲染，才能进行实际的逆渲染训练。