# Workflow

这份文档现在只保留旧链接兼容说明。流程约束已经迁入 Trellis，不再在这里重复维护。

## 当前源头

- 任务阶段、skill routing、是否创建 task、何时 commit: `.trellis/workflow.md`
- 分支、远端运行、主线口径、实验节奏、实验台账规则: `.trellis/spec/backend/experiment-protocol.md`
- 代码组织、数据契约、错误处理、日志、质量要求: `.trellis/spec/backend/index.md`

## 文档分工

`docs/` 继续保留实验知识库，不承载 Trellis 运行流程:

- [docs/handoff.md](./handoff.md): 当前主线状态、稳定判断、当前优先分支和关键文件
- [docs/model_improvement_plan.md](./model_improvement_plan.md): 当前快照、路线判断、候选路线和默认下一步
- [docs/experiment_index.jsonl](./experiment_index.jsonl): agent-facing 结构化实验索引
- [docs/experiments/](./experiments/): seed 指标、切片、诊断、命令、结果路径和失败模式证据
- [docs/experiment_themes.md](./experiment_themes.md): 跨实验诊断和主题级结论
- [docs/archive_legacy_experiments.md](./archive_legacy_experiments.md): 低频复访的旧失败路线压缩归档

## 更新规则

- 新流程或协作规则写入 `.trellis/workflow.md`。
- 新代码/运行约束写入 `.trellis/spec/backend/`。
- 新实验结论写入实验台账文档；不要塞进 Trellis spec。
- 如果旧链接指向本文件，按上面的“当前源头”跳转到对应 Trellis 文件。
