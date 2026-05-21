# 清理模型代码冗余

## Goal

降低模型代码中的结构性冗余和维护风险，同时保持现有训练、评估、实验结果语义不变。第一阶段优先清理 `DecoupledCDM` 中重复的 evidence 统计、触发 mask、零初始化和参数校验模式，不做新模型机制。

## What I Already Know

* 用户先要求检查模型代码是否冗余，随后要求“清理一下”。
* 当前主要冗余集中在 `models/decoupled_cdm.py`。
* `DecoupledCDM.__init__` 约 627 行、112 个参数，存在大量成组实验开关和重复校验/赋值。
* `DecoupledCDM.forward` 约 261 行，串联多个可选 residual/prior 分支。
* 多个 helper 重复构造 `q_mask`、`target_attempts`、`target_correct`、`target_seen`、`seen_ratio`、mastery/confidence 和 `trigger_mask`。
* `scripts/train.py` 的 CLI/model kwargs/summary payload 也重复展开同一批模型字段，但这次应避免一次性大拆 CLI。
* 工作树在任务创建前是干净的。

## Assumptions

* 本任务是重构清理，不改变默认实验配置和模型数学输出。
* 优先做低风险、可测试的内部抽取，而不是删除历史实验开关。
* 旧 checkpoint 兼容性重要；不轻易重命名已有 `nn.Module` / `Parameter` 属性。

## Open Questions

* None for MVP. User confirmed scope: first clean model internals, do not tackle the large `scripts/train.py` parameter-passing refactor in this task.

## Requirements

* 保持现有 public constructor 参数、CLI flag、summary 字段基本兼容。
* 抽取重复 evidence 统计逻辑，让多个 readout/prior helper 复用同一实现。
* 抽取重复 trigger mask / min-max-count / seen-ratio 判定逻辑。
* 抽取重复 zero-init last layer 逻辑。
* 不改变默认模型输出、已有实验开关语义或 checkpoint key。
* 更新/新增聚焦测试，覆盖抽取后关键 residual/prior 输出仍符合现有行为。
* 暂不整理 `scripts/train.py` 的 argparse/model kwargs/summary 大段参数传递，只在必要时保持兼容。

## Acceptance Criteria

* [ ] `models/decoupled_cdm.py` 的重复 evidence 统计和 trigger mask 逻辑明显减少。
* [ ] 现有相关 tests 通过，尤其 `tests/test_decoupled_cdm.py`。
* [ ] 不引入新的外部依赖。
* [ ] 不重命名已有模型参数 key，除非有明确兼容层。

## Definition of Done

* Tests added/updated where risk warrants it.
* Lint/type-check or focused tests run according to repo constraints.
* Behavior-changing cleanup is explicitly avoided or documented.
* If broader cleanup remains, leave a concise follow-up note in final response.

## Out of Scope

* 删除历史实验分支或 CLI flags。
* 重新设计 `DecoupledCDM` 构造参数体系为完整 config dataclass。
* 改变实验默认 runner、训练目标、模型指标口径。
* 重写 `scripts/train.py` argparse 架构。

## Technical Notes

* Relevant files inspected:
  * `models/decoupled_cdm.py`
  * `models/hetero_propagation.py`
  * `models/ensemble_cdm.py`
  * `scripts/train.py`
  * `scripts/evaluate.py`
  * `configs/defaults.py`
* Trellis backend guidelines apply; before code edits, load backend spec via `trellis-before-dev`.
* Candidate low-risk extraction points:
  * evidence required checks
  * target concept evidence summary
  * bounded mastery/confidence summary
  * min/max concept count + seen-ratio trigger mask
  * zero-initialize final linear layer helper
