---
name: "commit-and-doc"
description: "Commit changes and update documentation"
disable-model-invocation: true
---
将本次的修改提交commit（注：忽略非本次的修改）。同时形成文档，方便后续查阅。文档也加入commit里
若本次修改是对上次commit的修正或补充，使用 `git commit --amend`

沉淀的内容包括但不限于：
- 本次commit的新功能/修复的描述。文档存放在：
    - `doc/dev_log/日期/序号-功能描述.md` 
    - 例如：`doc/dev_log/2026-06-17/001-做了什么.md`
    - 文档顶部包含yaml元数据例如：
`---
created_at: "2026-06-17"
updated_at: "2026-06-17"
---
`
- 如果你对某个常用模块/功能做了详细/全面的搜索调查，可将探索结果沉淀为说明文档保存到 `doc/`，以便后续复用，避免后续对同一模块重复调查浪费时间和token。（例如：`doc/render_equirect.md`）。如果你调研的内容在`doc/`已存在，或者修改的内容波及了doc里已有文档，需要更新已有的文档，以保持文档和实际代码的一致。
- (可选) 检查并更新`CLAUDE.md` 确保与修改后的代码保持一致。但是注意务必精简：这个文档是长期记忆，不是变更日志。只保留agent必须知道的记忆。