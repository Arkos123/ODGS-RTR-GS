   
以下是关于 RTR-GS 加载机制的详细调查结果，包含五个部分，每部分均附有代码片段和行号。

---

## 1. `train.py` 如何加载 checkpoint（`-c` 标志）

在 [train.py](file:///home/huangpengyue/projects/RTR-GS/train.py) 第 43-55 行，训练入口 `training()` 函数中，有三个分支来决定高斯模型的初始化方式：

```python
# train.py:L43-L55
gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
scene = Scene(dataset, gaussians)
if args.checkpoint:
    print("Create Gaussians from checkpoint {}".format(args.checkpoint))
    first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

elif scene.loaded_iter:
    gaussians.load_ply(os.path.join(dataset.model_path,
                                    "point_cloud",
                                    "iteration_" + str(scene.loaded_iter),
                                    "point_cloud.ply"))
else:
    gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)
```

**逻辑说明**：
- 如果指定了 `-c` / `--checkpoint` 参数，则调用 `gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)`，并返回 `first_iter`（用于从断点继续训练）。
- 如果没有 `-c` 但有 `scene.loaded_iter`（即 `--load_iteration` 参数指定了加载某个迭代号的 point_cloud），则调用 `gaussians.load_ply()` 从 `.ply` 文件加载。
- 否则，从数据集初始点云调用 `create_from_pcd()` 初始化。

另外，该 checkpoint 机制还会联动加载其他 PBR 组件（第 65-119 行）：

```python
# train.py:L65-L77  -- 加载 transfer_net
if pipe.compute_with_prt:
    transfer_net = TransferMLP(...)
    if args.checkpoint:
        transfer_net_checkpoint = os.path.dirname(args.checkpoint) + "/transfer_net_" + os.path.basename(args.checkpoint)
        if os.path.exists(transfer_net_checkpoint):
            transfer_net.create_from_ckpt(transfer_net_checkpoint)

# train.py:L94-L104  -- 加载 cubemap
if is_pbr:
    cubemap = CubemapLight(base_res=128).cuda()
    if args.checkpoint:
        cubemap_checkpoint = os.path.dirname(args.checkpoint) + "/cubemap_" + os.path.basename(args.checkpoint)
        if os.path.exists(cubemap_checkpoint):
            cubemap.create_from_ckpt(cubemap_checkpoint, restore_optimizer=True)

# train.py:L106-L119  -- 加载 refmap
if pipe.ref_map:
    refmap = CubemapLight(base_res=128).cuda()
    if args.checkpoint:
        refmap_checkpoint = os.path.dirname(args.checkpoint) + "/refmap_" + os.path.basename(args.checkpoint)
        if os.path.exists(refmap_checkpoint):
            refmap.create_from_ckpt(refmap_checkpoint, restore_optimizer=True)
```

**checkpoint 保存逻辑**（第 242-253 行）：

```python
# train.py:L242-L253
if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
    torch.save((gaussians.capture(), iteration),
               os.path.join(scene.model_path, "checkpoint/chkpnt" + str(iteration) + ".pth"))

    for com_name, component in pbr_kwargs.items():
        try:
            torch.save((component.capture(), iteration),
                       os.path.join(scene.model_path, f"checkpoint/{com_name}_chkpnt" + str(iteration) + ".pth"))
        except:
            pass
```

---

## 2. `create_from_ckpt` 方法（checkpoint 恢复）

位于 [scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py) 第 351-415 行：

```python
# scene/gaussian_model.py:L351-L415
def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
    (model_args, first_iter) = torch.load(checkpoint_path)

    (self.active_sh_degree,
     self._xyz,
     self._shs_dc,
     self._shs_rest,
     self._diffuse_tint,
     self._specular_tint,
     self._ref_tint,
     self._ref_strength,
     self._ref_roughness,
     self._specular_feature,
     self._diffuse_transfer_dc,
     self._diffuse_transfer_rest,
     self._scaling,
     self._rotation,
     self._opacity,
     self.max_radii2D,
     weights_accum,
     xyz_gradient_accum,
     denom,
     opt_dict,
     self.spatial_lr_scale) = model_args[:21]

    self.weights_accum = weights_accum
    self.xyz_gradient_accum = xyz_gradient_accum
    self.denom = denom

    if self.use_pbr:
        if len(model_args) > 21:
            (self._base_color, self._roughness, self._metallic,
             self._incidents_dc, self._incidents_rest) = model_args[21:26]
        else:
            # 旧的 checkpoint 没有 PBR 属性 => 初始化默认值
            base_color = torch.ones_like(self._xyz)
            self._base_color = nn.Parameter(base_color.requires_grad_(True))
            roughness = inverse_sigmoid(torch.ones_like(self._xyz[..., :1]) * 0.65)
            self._roughness = nn.Parameter(roughness.requires_grad_(True))
            metallic = inverse_sigmoid(torch.ones_like(self._xyz[..., :1]) * 0.001)
            self._metallic = nn.Parameter(metallic.requires_grad_(True))
            incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._incidents_dc = nn.Parameter(incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    if restore_optimizer:
        try:
            self.optimizer.load_state_dict(opt_dict)
        except:
            print("Not loading optimizer state_dict!")

    return first_iter
```

**关键点**：
- 使用 `torch.load` 加载 `.pth` 文件，它保存的是 `(capture_tuple, iteration)` 元组。
- `capture()` 方法（第 116-151 行）将 21 个（非 PBR）或 26 个（PBR）张量/状态打包为元组。
- 从 checkpoint 恢复时，直接从保存的张量赋值到 `self._*` 属性，**不需要通过 `nn.Parameter` 包装**，因为张量本身在保存时已经是 `nn.Parameter`。
- PBR 属性（`_base_color`, `_roughness`, `_metallic` 等）作为可选部分，兼容旧 checkpoint。

---

## 3. `create_from_pcd` 方法（从点云初始化）

位于 [scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py) 第 417-470 行：

```python
# scene/gaussian_model.py:L417-L470
def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    self.spatial_lr_scale = spatial_lr_scale
    fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
    shs = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    shs[:, :3, 0] = RGB2SH(fused_color)       # DC 分量由 RGB 转换
    shs[:, 3:, 1:] = 0.0                       # 高阶 SH 归零

    # 使用 distCUDA2 计算点云密度来决定初始 scale
    dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1                             # 单位四元数

    opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), ...))

    self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    self._rotation = nn.Parameter(rots.requires_grad_(True))
    self._scaling = nn.Parameter(scales.requires_grad_(True))
    self._opacity = nn.Parameter(opacities.requires_grad_(True))
    self._shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
    self._shs_rest = nn.Parameter(shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    # RTR-GS 专用属性：初始化默认值
    self._diffuse_tint = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
    self._specular_tint = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
    self._ref_tint = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
    ref_strength = inverse_sigmoid(torch.ones(...) * 0.01)  # 默认反射强度 0.01
    self._ref_strength = nn.Parameter(ref_strength.requires_grad_(True))
    ref_roughness = inverse_sigmoid(torch.ones(...) * 0.65)  # 默认粗糙度 0.65
    self._ref_roughness = nn.Parameter(ref_roughness.requires_grad_(True))
    self._specular_feature = nn.Parameter(torch.zeros((..., self.n_featres), ...).requires_grad_(True))
    self._diffuse_transfer_dc = nn.Parameter(...)
    self._diffuse_transfer_rest = nn.Parameter(...)

    if self.use_pbr:
        # 初始化 base_color、roughness、metallic、incidents 为默认值
        ...
```

**关键点**：
- 从 BasicPointCloud（COLMAP 或随机生成的点云）提取位置和颜色。
- 颜色转换为 SH 的 DC 分量，高阶 SH 归零。
- `distCUDA2` 计算点云 KNN 距离，用于初始化各向异性 scale。
- RTR-GS 特有的 `_diffuse_tint`, `_specular_tint`, `_ref_tint`, `_ref_strength`, `_ref_roughness`, `_specular_feature`, `_diffuse_transfer_dc/rest` 均初始化为默认值。
- 所有属性都用 `nn.Parameter` 包装为可学习参数。

---

## 4. RTR-GS 是否支持加载预训练的 `.ply` 文件

**支持，但路径有限制。**

RTR-GS 主 [GaussianModel](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py) 的第 678-822 行有完整的 `load_ply` 方法。在 `train.py` 中，只有当满足 `scene.loaded_iter`（即用户指定了 `--load_iteration` 参数，或 `load_iteration=-1` 自动搜索最大迭代）时，才会从输出目录的 `point_cloud/iteration_N/point_cloud.ply` 加载，参见上面第 49-53 行。

在 [scene/__init__.py](file:///home/huangpengyue/projects/RTR-GS/scene/__init__.py) 第 34-39 行：

```python
# scene/__init__.py:L34-L39
if load_iteration:
    if load_iteration == -1:
        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
    else:
        self.loaded_iter = load_iteration
```

**结论**：RTR-GS 可以加载 `.ply`，但只能从特定目录结构 `{model_path}/point_cloud/iteration_{N}/point_cloud.ply` 中加载。不能直接传入任意 `.ply` 文件路径。如果要加载任意 `.ply`，需要外部自行调用 `gaussians.load_ply(path)`。

---

## 5. PLY 保存/加载逻辑

### `save_ply`（第 570-609 行）

```python
# scene/gaussian_model.py:L570-L609
def save_ply(self, path):
    mkdir_p(os.path.dirname(path))

    xyz = self._xyz.detach().cpu().numpy()
    sh_dc = self._shs_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    sh_rest = self._shs_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    diffuse_tint = self._diffuse_tint.detach().cpu().numpy()
    specular_tint = self._specular_tint.detach().cpu().numpy()
    ref_tint = self._ref_tint.detach().cpu().numpy()
    ref_strength = self._ref_strength.detach().cpu().numpy()
    ref_roughness = self._ref_roughness.detach().cpu().numpy()
    specular_feature = self._specular_feature.detach().cpu().numpy()
    diffuse_transfer_dc = self._diffuse_transfer_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    diffuse_transfer_rest = self._diffuse_transfer_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self._opacity.detach().cpu().numpy()
    scale = self._scaling.detach().cpu().numpy()
    rotation = self._rotation.detach().cpu().numpy()

    attributes_list = [xyz, sh_dc, sh_rest, diffuse_tint, specular_tint, ref_tint,
                       ref_strength, ref_roughness, specular_feature,
                       diffuse_transfer_dc, diffuse_transfer_rest,
                       opacities, scale, rotation]

    if self.use_pbr:
        attributes_list.extend([
            self._base_color.detach().cpu().numpy(),
            self._roughness.detach().cpu().numpy(),
            self._metallic.detach().cpu().numpy(),
            self._incidents_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            self._incidents_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
        ])

    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(attributes_list, axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
```

`construct_list_of_attributes`（第 527-568 行）定义了 PLY 文件中的属性名称列表，按顺序为：`x, y, z, f_dc_0..N, f_rest_0..N, diffuse_tint_0..2, specular_tint_0..2, ref_tint0..2, ref_strength, ref_roughness, specular_feature_0..9, diffuse_transfer_dc_0, diffuse_transfer_rest_0..N, opacity, scale_0..2, rot_0..3`。如果 `use_pbr`，则额外追加 `base_color_0..2, roughness, metallic, incidents_dc_0..N, incidents_rest_0..N`。

### `load_ply`（第 678-822 行）

```python
# scene/gaussian_model.py:L678-L822
def load_ply(self, path):
    plydata = PlyData.read(path)
    # 读取 xyz
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # 读取 SH DC（3 个通道）
    shs_dc = np.zeros((xyz.shape[0], 3, 1))
    shs_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    shs_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    shs_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # 读取 SH rest，动态查找属性名
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    ...
    shs_extra = shs_extra.reshape((..., 3, (self.max_sh_degree + 1) ** 2 - 1))

    # 读取 RTR-GS 特有属性（diffuse_tint, specular_tint, ref_tint, ref_strength, ref_roughness, specular_feature, diffuse_transfer）
    ...

    # 读取 scale 和 rotation
    ...

    # 将所有读取的 numpy 数组转为 nn.Parameter 并赋值给 self
    self._xyz = nn.Parameter(torch.tensor(xyz, ...).requires_grad_(True))
    self._rotation = nn.Parameter(torch.tensor(rots, ...).requires_grad_(True))
    self._scaling = nn.Parameter(torch.tensor(scales, ...).requires_grad_(True))
    self._opacity = nn.Parameter(torch.tensor(opacities, ...).requires_grad_(True))
    self._shs_dc = nn.Parameter(torch.tensor(shs_dc, ...).transpose(1, 2).contiguous().requires_grad_(True))
    self._shs_rest = nn.Parameter(torch.tensor(shs_extra, ...).transpose(1, 2).contiguous().requires_grad_(True))
    self._diffuse_tint = nn.Parameter(...)
    self._specular_tint = nn.Parameter(...)
    self._ref_tint = nn.Parameter(...)
    self._ref_strength = nn.Parameter(...)
    self._ref_roughness = nn.Parameter(...)
    self._specular_feature = nn.Parameter(...)
    self._diffuse_transfer_dc = nn.Parameter(...)
    self._diffuse_transfer_rest = nn.Parameter(...)

    self.active_sh_degree = self.max_sh_degree

    # PBR 属性加载
    if self.use_pbr:
        # base_color, roughness, metallic, incidents_dc, incidents_rest
        ...
```

**重要：`load_ply` 中存在变量名 Bug**

在 [scene/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/scene/gaussian_model.py) 第 734-746 行，存在明显的变量名错误：

```python
# L734-L746 - BUG: 变量名不一致
diffuse_transfer_dc = np.zeros((xyz.shape[0], 1, 1))
diffuse_transfer_dc[:, 0, 0] = np.asarray(plydata.elements[0]["incidents_dc_0"])  # ← 属性名是 incidents_dc_0

extra_diffuse_transfer_names = [p.name for p in plydata.elements[0].properties if
                                 p.name.startswith("diffuse_transfer_rest_")]       # ← 这里查 diffuse_transfer_rest_
extra_diffuse_transfer_names = sorted(extra_diffuse_transfer_names, ...)
diffuse_transfer_extra = np.zeros((xyz.shape[0], len(extra_diffuse_transfer_names)))
for idx, attr_name in enumerate(extra_incidents_names):      # ← BUG: 用了 extra_incidents_names（未定义）
    diffuse_transfer_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

diffuse_transfer_extra = diffuse_transfer_extra.reshape((incidents_extra.shape[0], ...))  # ← BUG: 用了 incidents_extra（未定义）
```

这两处应该是 `extra_diffuse_transfer_names` 和 `diffuse_transfer_extra`，而不是 `extra_incidents_names` 和 `incidents_extra`。**这意味着当前的 `load_ply` 在非 PBR 模式下加载 diffuse_transfer 属性时很可能崩溃**（`NameError`）。

---

## 6. `scene_perspective` 模块如何加载 `.ply` 进行推理

`scene_perspective/` 是 ODGS 子模块中的推理专用模块，位于 [submodules/odgs/scene_perspective/](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene_perspective/)。

入口脚本为 [render_perspective.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/render_perspective.py)：

```python
# submodules/odgs/render_perspective.py:L41-L44
def render_sets(dataset, iteration, pipeline, skip_train, skip_test):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)    # 注意：不传 scale_max
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
```

[scene_perspective/__init__.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene_perspective/__init__.py) 第 85-91 行的加载逻辑：

```python
# submodules/odgs/scene_perspective/__init__.py:L85-L91
if self.loaded_iter:
    self.gaussians.load_ply(os.path.join(self.model_path,
                                                   "point_cloud",
                                                   "iteration_" + str(self.loaded_iter),
                                                   "point_cloud.ply"))
else:
    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
```

这里的 GaussianModel 是 ODGS 的简化版 [submodules/odgs/scene_perspective/gaussian_model.py](file:///home/huangpengyue/projects/RTR-GS/submodules/odgs/scene_perspective/gaussian_model.py)：

```python
# submodules/odgs/scene_perspective/gaussian_model.py:L249-L291
def load_ply(self, path):
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # 读取 features_dc (SH DC)
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # 读取 features_rest (SH 高阶)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((..., 3, (self.max_sh_degree + 1) ** 2 - 1))

    # 读取 scale 和 rotation
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    ...

    self._xyz = nn.Parameter(torch.tensor(xyz, ...).requires_grad_(True))
    self._features_dc = nn.Parameter(torch.tensor(features_dc, ...).transpose(1, 2).contiguous().requires_grad_(True))
    self._features_rest = nn.Parameter(torch.tensor(features_extra, ...).transpose(1, 2).contiguous().requires_grad_(True))
    self._opacity = nn.Parameter(torch.tensor(opacities, ...).requires_grad_(True))
    self._scaling = nn.Parameter(torch.tensor(scales, ...).requires_grad_(True))
    self._rotation = nn.Parameter(torch.tensor(rots, ...).requires_grad_(True))
    self.prune_large()       # 根据 scale_max 裁剪过大的高斯

    self.active_sh_degree = self.max_sh_degree
```

**`scene_perspective` 版本的关键特点**：
- **仅包含最基本的 3DGS 属性**：`xyz`, `features_dc`, `features_rest`, `scaling`, `rotation`, `opacity`。没有 RTR-GS 的 `diffuse_tint`, `ref_tint`, `ref_strength`, `specular_feature` 等属性。
- `construct_list_of_attributes`（第 183-196 行）的输出属性列表仅 8 个字段：`x, y, z, nx, ny, nz, f_dc_0..N, f_rest_0..N, opacity, scale_0..2, rot_0..3`（带有 normals 占位）。
- `save_ply` 保存前会过滤 `NaN` 位置的高斯（第 203-205 行）。
- `load_ply` 加载后调用 `prune_large()`，裁剪 scale 过大的高斯（第 234-246 行）。

---

## 总结对比

| 机制 | 函数 | 位置 | 用途 |
|------|------|------|------|
| **Checkpoint 恢复** | `create_from_ckpt()` | `scene/gaussian_model.py:L351` | 从 `.pth` 恢复完整状态（含 optimizer、梯度累积），用于继续训练 |
| **从点云初始化** | `create_from_pcd()` | `scene/gaussian_model.py:L417` | 从 COLMAP/随机点云初始化高斯，用于训练开始 |
| **PLY 加载（RTR-GS）** | `load_ply()` | `scene/gaussian_model.py:L678` | 从 `.ply` 恢复高斯几何和外观属性（含 RTR-GS 扩展属性） |
| **PLY 加载（ODGS 推理）** | `load_ply()` | `submodules/odgs/scene_perspective/gaussian_model.py:L249` | 仅加载基本 3DGS 属性，用于透视投影推理 |
| **PLY 保存** | `save_ply()` | `scene/gaussian_model.py:L570` | 将所有高斯属性保存为 ply 格式 |

**关键差异**：
- `.pth` checkpoint 保存完整训练状态（包括 optimizer、梯度累积），用于继续训练；`.ply` 仅保存高斯属性，用于推理和可视化。
- RTR-GS 的 `load_ply` 有变量名 Bug（`extra_incidents_names` 应为 `extra_diffuse_transfer_names`，`incidents_extra` 应为 `diffuse_transfer_extra`）。
- `scene_perspective` 模块的 GaussianModel 是简化版本，不支持 RTR-GS 扩展属性（diffuse_tint、ref_tint 等）。
        
       