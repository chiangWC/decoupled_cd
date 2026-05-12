# Session Bootstrap

这份文档现在只保留兼容入口和文档索引，不再作为流程约束源。

## 当前源头

- Trellis 任务流程: `.trellis/workflow.md`
- 项目执行约束: `.trellis/spec/backend/experiment-protocol.md`
- Python 代码规范: `.trellis/spec/backend/index.md`
- 当前主线、候选路线和实验结论: `docs/model_improvement_plan.md`
- 扩展交接和关键文件: `docs/handoff.md`

## 新会话规则

新会话应先遵循 Trellis 注入的 `<workflow-state>`。如果平台没有注入 Trellis 上下文，则读取 `.agents/skills/trellis-start/SKILL.md` 并执行其中的启动步骤。

不要在本文件新增新的流程、分支、远端运行或实验记录规则；这些规则应写入 `.trellis/workflow.md` 或 `.trellis/spec/backend/experiment-protocol.md`。

## 按需再读

- [docs/handoff.md](./handoff.md)
  - 需要看当前主线细节、关键判断、分支优先级和关键文件时再读
- [docs/model_improvement_plan.md](./model_improvement_plan.md)
  - 需要查历史实验、候选路线、失败路线或默认下一步时再读
- [docs/experiment_index.jsonl](./experiment_index.jsonl)
  - 已知实验号、分支名、状态或失败原因时，用作结构化索引
- [docs/experiments/](./experiments/)
  - 只有需要 seed、slice、诊断证据、命令或结果路径时再打开 detail doc
- [README_spec.md](../README_spec.md)
  - 需要核对模型语义和 Step 1-4 定义时再读
- [docs/transition_graph_notes.md](./transition_graph_notes.md)
  - 需要修改构图逻辑时再读
- [docs/reuse_plan.md](./reuse_plan.md)
  - 只有做工程复用或重构时再读
