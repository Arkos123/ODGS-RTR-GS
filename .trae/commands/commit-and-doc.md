---
name: "commit-and-doc"
description: "Commit changes and create documentation"
disable-model-invocation: true
---
将本次的修改提交commit。同时形成文档，方便后续查阅。

沉淀的内容包括但不限于：
- 本次commit的新功能/修复的描述。文档存放在：
    - `doc/M-D-001-功能描述.md` 
    - 文档名称为日期+序号+主要功能描述

- 如果你对某个模块/功能做了调研（如调用子agent的调研）你也可以将调研结果沉淀为说明文档保存到 `./doc/` 以便后续复用。
