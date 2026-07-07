---
created_at: "2026-07-06"
updated_at: "2026-07-06"
---

# Phase 1-2: 3DGS-Editor 扩展属性 & PLY 兼容（已完成）

## 概要

扩展 3DGS-Editor-3.0 的 GaussianObject，支持加载/导出 RTR-GS 格式的点云。

## 详细内容

见 `.tasks/迁移重光照能力到Editor/task_plan.md` 和 `notes.md`。

**关键变更**（提交 `1a42b4b` 在 3DGS_Editor-3.0）：

| 文件 | 改动 |
|------|------|
| scene.py | 13 个扩展属性字段、动态 PLY 解析、格式自动检测、选择/合并传播 |
| control.py | `_EXT_ATTR_SPEC` dict-of-lists、统一收集/赋值、格式感知导出 |
| render.py | SGS equirect 返回值解包修复 |

**支持格式**：标准 3DGS（含 SGS）/ RTR-GS Stage1（96字段）/ Stage2（149字段）
