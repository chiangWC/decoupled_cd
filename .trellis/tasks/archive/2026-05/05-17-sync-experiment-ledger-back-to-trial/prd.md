# sync experiment ledger back to trial

## Goal

把当前分支上已经确认的实验结论整理成连续、可提交、可回灌到 `exp/trellis-trial` 的 ledger 记录，避免实验知识只停留在支线或工作区脏状态里。重点是保留 experiment 82 的已确认负结果，并新增一条后续结构诊断记录，覆盖 `q-local / no-expert / expert-coverage-gate` 这轮探索。

## What I already know

* 当前分支是 `exp/exp81-target-concept-interaction-rebase`，`docs/experiment_index.jsonl`、`docs/handoff.md`、`docs/model_improvement_plan.md` 已有未提交 ledger 改动。
* `docs/experiments/082_exp81_target_interaction_rebase.md` 目前是未跟踪文件，但内容完整，记录了 experiment 80 在 exp81 伪主线上的 rebase 失败结论。
* `exp/trellis-trial` 当前已提交的 ledger 还停在 experiment 81，因此从 `trial` 再切支线时无法继承 experiment 82 及后续结构诊断。
* 后续新结论尚未入账：`q-local` 直接叠回当前 exp81 主线三 seed 判负；`no-expert + sidecar + single-only + q-local` 在 matched family 恢复；`inverse/partial coverage gate` 退化成 no-expert 轨迹；`soft coverage gate` 更差。

## Assumptions (temporary)

* 这次任务只整理和同步 ledger，不改模型代码、不重跑实验。
* experiment 83 将用于记录 `expert x q-local` 结构诊断，而不是 promote 新主线。
* 回灌 `trial` 时只同步 ledger 相关提交，不把当前实验代码分支整体 merge 回去。

## Open Questions

* 无阻塞问题；范围已由当前分支结果和用户意图确定。

## Requirements (evolving)

* 保留并提交现有 experiment 82 相关 ledger 改动。
* 新增 experiment 83 详情文档，准确记录 `q-local / no-expert / coverage-gating` 的结果与结论。
* 更新 `docs/experiment_index.jsonl`、`docs/handoff.md`、`docs/model_improvement_plan.md`，使其覆盖 experiment 82 与 83。
* 形成一组纯 ledger 提交，便于后续 cherry-pick 或同步回 `exp/trellis-trial`。

## Acceptance Criteria (evolving)

* [ ] `docs/experiments/082_exp81_target_interaction_rebase.md` 进入提交历史
* [ ] 新增 `docs/experiments/083_*.md`，内容与已跑结果一致
* [ ] 三份总台账包含 experiment 82/83，并更新默认 follow-up 结论
* [ ] 当前分支形成清晰的 ledger-only 提交边界，便于回灌 `exp/trellis-trial`

## Definition of Done (team quality bar)

* 文档内容与结果文件、已报告指标一致
* 提交前完成基本一致性检查（diff、自查、必要时 grep）
* 不混入无关代码改动
* 回灌方案明确，知道哪些提交应同步回 `exp/trellis-trial`

## Out of Scope (explicit)

* 重新训练、补 seed、修改模型实现
* 直接把当前代码分支整体 merge 回 `exp/trellis-trial`
* 清理 `.trellis/tasks/archive/...` 之外的历史任务记录

## Technical Notes

* 参考 detail 模板: `docs/experiments/080_scoped_evidence_interaction_combo.md`, `docs/experiments/081_exp70_single_only_evidence_readout_rebase.md`
* 需要更新的 ledger 文件:
  * `docs/experiment_index.jsonl`
  * `docs/handoff.md`
  * `docs/model_improvement_plan.md`
* 需要写入的新结论来源于现有结果目录:
  * `results/q_local_single_only_combo/`
  * `results/q_local_single_only_no_expert/`
  * `results/expert_q_local_inverse_coverage/`
  * `results/expert_q_local_partial_coverage/`
  * `results/expert_q_local_soft_coverage/`
