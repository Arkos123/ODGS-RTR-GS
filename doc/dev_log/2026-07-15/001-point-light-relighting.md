---
created_at: "2026-07-15"
updated_at: "2026-07-15"
---

# 点光源重光照功能实现

## 概述

为 RTR-GS 增加点光源重光照能力。支持点光源 PBR 着色、阴影、视频轨道旋转、指示器可视化，兼容 perspective 和 equirect 两种渲染模式。

## 架构

采用独立叠加层方案：`point_light_shading()` 在 `pbr_shading()` 之后运行，结果逐像素叠加，不修改现有 IBL 逻辑。

## 文件变更

| 文件 | 说明 |
|------|------|
| `pbr/light.py` | 新增 `PointLight` 数据类（位置、颜色、强度） |
| `pbr/shade.py` | 新增 `point_light_shading()`（完整 Cook-Torrance BRDF） |
| `pbr/__init__.py` | 导出新接口 |
| `pbr/point_light_shadow.py` | **新建** — 阴影模块（cubemap + equirect 双模式） |
| `gaussian_renderer/render.py` | 透视模式点光源叠加 + 表面点重建 norm 修复 |
| `gaussian_renderer/render_equirect.py` | Equirect 模式点光源叠加 |
| `eval_relighting_colmap.py` | 重光照 eval 脚本扩展（`--point_lights_config`, `--point_light_vis`, 视频旋转, 指示器） |
| `utils/graphics_utils.py` | `get_canonical_rays` 去掉意外添加的归一化 |
| `test_data/test_point_light.json` | 测试配置 |
| `doc/point-light-relighting-design.md` | 设计文档 |
| `.gitattributes` | 强制 LF 换行符 |

## 关键修复记录

### 1. cubemap batch 维度缺失
`make_shadow_func_cubemap` 中 `dr.texture` 需要 `[1, H, W, 3]` 格式的 UV，但传入的是 `[H, W, 3]`，添加 `.unsqueeze(0)` 修复。

### 2. 变量名冲突导致相机半径被覆盖
`eval_render_video_equirect` 中点光源轨道循环用了同名变量 `radius`，覆盖了相机运动半径。重命名为 `orbit_radius` 修复。

### 3. Float64 dtype 冲突
`torch.tensor([radius * c, 0.0, radius * s])` 默认 float64，与 float32 的 GPU 张量相加后 CUDA 光栅化器报错。添加 `dtype=pivot.dtype` 修复。

### 4. `lite_rasterize_gaussians` 参数不匹配
GS-IR 的 `lite_rasterize_gaussians` 有 18 个参数，RTR-GS 的版本多了 `shs` 参数共 19 个。对照 `baking.py` 的调用方式修复。

### 5. Equirect 投影 v 坐标翻转
`_draw_light_indicator` 中 equirect 的 `v = (lat / pi + 1) * 0.5` 上下颠倒，改为 `v = 0.5 - lat / pi` 匹配 SGS rasterizer 约定。

### 6. 透视投影 Y 轴方向
RTR-GS 的 `ndc2Pix` 不做 Y 翻转（NDC Y 向下 = 图像 Y 向下），所以 `py = (ndc_y + 1) / 2 * H` 是正确的，不需要 `(1 - ndc_y) / 2 * H`。

### 7. 表面点重建缺少 norm 因子（重要）
`render.py` 中表面点重建 `-view_dirs * depth + cam_pos` 缺少原始射线长度因子。`canonical_rays` 被意外归一化（本应返回未归一化的 `[dx, dy, 1]`），导致 `view_dirs` 是单位方向但 depth 是视空间 Z，重建的 3D 位置在 off-center 像素处偏近。修复：
- `get_canonical_rays` 去掉归一化，返回 `[dx, dy, 1]`
- `render.py` 增加 `raw_ray_norm = ||canonical_rays||` 并乘到表面重建中

### 8. 透视投影半像素偏移
`_project_to_screen` 的 NDC→像素公式缺少 `-0.5`（匹配 CUDA `ndc2Pix` 的半整数像素中心约定），添加后对齐。

## 使用方式

```bash
# 静态点光源重光照（perspective）
python eval_relighting_colmap.py --eval -s <data> -m <model> -c <ckpt> \
    --point_lights_config test_data/test_point_light.json --point_light_vis

# Equirect 模式 + 视频轨道旋转
python eval_relighting_colmap.py --eval -s <data> -m <model> -c <ckpt> \
    -t render_ref_pbr_equirect --equirect_width 2000 --save_video \
    --point_lights_config test_data/test_point_light.json --point_light_vis
```

JSON 配置格式：
```json
{
    "lights": [{
        "position": [0.0, -0.3, 0.0],
        "color": [1.0, 1.0, 1.0],
        "intensity": 20.0,
        "radius": 0.4,
        "rotate_total": 360.0,
        "shadow_bias": 0.3
    }]
}
```

## 已知注意事项

- `shadow_bias` 是场景坐标单位，需要根据场景尺度调整（默认 0.3）
- 视频旋转时每帧重新烘焙深度图，较慢；可降低 `--equirect_width` 加速
- 透视模式 garden 场景已验证（stage2 + point_light），已修复表面位置误差
