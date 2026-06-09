---
name: "save-research"
description: "Save project investigation results as documentation"
disable-model-invocation: true
---
在调查完毕后，将项目调查结果以文档的方式保存到 `./doc/` 目录下。

例如：
- 用户询问：项目里的incident_light是什么意思？
- 你需要：
    - 先检查 `./doc` 是否已经有相关说明文档，以便节省调查时间。
    - 如果文档缺少你需要的信息，则进行研究。
    - 将调查结果保存到 `./doc/`，并让用户直接查看文档。这样也方便以后随时查看。