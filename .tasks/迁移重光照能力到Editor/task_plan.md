# Task Plan: RTR-GS 重光照渲染能力迁移到 3DGS-Editor-3.0

## Goal
将 RTR-GS 的重光照（relighting）渲染能力完整迁移到 3DGS-Editor-3.0 编辑器中，使其支持 SGS / RTR-GS stage1 / RTR-GS stage2 等多种点云格式的加载、渲染和重光照编辑。

## 当前状态

### 3DGS-Editor-3.0 (`/home/huangpengyue/projects/3DGS_Editor-3.0/`)
- **GS 模型**：已扩展支持 RTR-GS Stage1 + Stage2 属性（共 13 个扩展字段）
- **格式支持**：标准 3DGS + RTR-GS Stage1（96 字段）+ RTR-GS Stage2（149 字段），自动检测
- **渲染管线**：4 种模式——自定义 CUDA、SGS pinhole、SGS equirect（尚未接入 PRT/PBR）
- **GUI 架构**：PyQt5，左侧 ControlWidget，右侧 RenderWidget

### RTR-GS (`/home/huangpengyue/projects/RTR-GS/`)
- **渲染管道**：PRT 重光照（TransferMLP + SH 光照）、反射渲染（CubemapLight + split-sum）、PBR 渲染（Cook-Torrance BRDF）
- **训练输出格式**：Stage1（96 字段）和 Stage2（149 字段）PLY

## Phases

### Phase 1: GS 模型扩展 ✅
- [x] 扩展 `GaussianObject`，增加 13 个 RTR-GS 属性字段
- [x] Stage1 属性：diffuse_tint, specular_tint, ref_tint, ref_strength, ref_roughness, specular_feature, diffuse_transfer_dc, diffuse_transfer_rest
- [x] Stage2 PBR 属性：base_color, roughness, metallic, incidents_dc, incidents_rest
- [x] `_copy_extended_attrs` / `_merge_extended_attrs` 支持选择/合并传播
- [x] PLY 导出同步更新（格式感知导出，保留扩展属性）

### Phase 2: PLY 格式兼容加载 ✅
- [x] 自动检测 3 种格式：standard_3dgs / rtr_gs_stage1 / rtr_gs_stage2
- [x] 动态 PLY 头部解析（`_read_ply_with_format`），不再硬编码字段
- [x] Stage1/Stage2 正确加载扩展属性，标准 3DGS 无扩展属性（向后兼容）
- [x] 导出格式自动切换（标准 3DGS / Stage1 / Stage2）
- [ ] 加载后初始化渲染组件（transfer_net, refmap, cubemap）— 待 Phase 3

### Phase 3: RTR-GS 渲染引擎桥接（进行中）
- [ ] 将 RTR-GS 的 rendering pipeline 模块化，使其可在 3DGS-Editor 环境中调用
  - PRT 颜色计算：view-independent diffuse + view-dependent specular
  - 反射渲染：CubemapLight + split-sum
  - PBR 渲染：Cook-Torrance BRDF
- [ ] 集成到 editor 渲染管线复用 `scene.render_mode` 机制
- [ ] 处理依赖：TransferMLP / CubemapLight 实例化
- [ ] 支持重光照参数传递

### Phase 4: GUI 光照编辑
- [ ] 光照编辑标签页/面板
- [ ] 光源管理（添加/删除/移动/颜色/强度）
- [ ] 环境贴图切换（加载 HDR / 强度 / 旋转）
- [ ] 材质参数调节（base_color / roughness / metallic）
- [ ] 编辑参数实时传递到渲染引擎

### Phase 5: 集成与测试
- [ ] 验证 3 种点云格式加载
- [ ] 渲染结果对比验证
- [ ] 重光照交互验证
- [ ] 原有功能回归测试

### Phase 6: 后续优化
- [ ] 性能优化（30+ FPS）
- [ ] 烘焙结果可视化
- [ ] 光照预设
- [ ] 重光照结果导出

## Key Questions
1. ~~RTR-GS 的 rendering pipeline 是否可以直接 import 到 3DGS-Editor？~~ 可以，通过 `sys.path.append` 或 PYTHONPATH 导入 RTR-GS 模块
2. ~~3DGS-Editor 的 SGS equirect 渲染模式能否复用？~~ 可以，SGS rasterizer 已在 editor 中，extra_features 参数已支持
3. GUI 交互中重光照的实时性要求多高？
4. 光源编辑系统是否需要物理光照模拟还是只有 relighting 级别？

## Decisions Made
- **属性存储约定**：扩展属性存 raw 值（预激活），与 PLY 文件一致，不应用 sigmoid/exp
- **SGS = Standard 3DGS**：SGS 训练输出与标准 3DGS 格式相同，已有 editor 已能导入
- **忽略 COLMAP SfM ASCII PLY**：稀疏点云无实用价值，不做专门支持
- **f_rest 固定 45**：标准/Stage1/Stage2 格式均使用 SH degree=3，45 个 f_rest 字段

## Errors Encountered
- Simplification agent 第一次跑时 CWD 在 RTR-GS，找不到 3DGS_Editor-3.0 的文件

## Status
**Phase 1-2 ✅ 已完成** — 提交 3DGS_Editor-3.0 `1a42b4b`
- 支持：标准 3DGS + RTR-GS Stage1(96字段) + Stage2(149字段)
- 下一步：Phase 3 — 渲染引擎桥接
