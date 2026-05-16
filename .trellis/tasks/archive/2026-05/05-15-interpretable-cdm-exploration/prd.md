# 继续可解释 CDM 模型探索

## Goal

继续探索 decoupled cognitive diagnosis model 的可解释结构改进，直至出现较明显的增长信号。和上一轮不同，本轮暂不考虑 CF、student-exercise ID-aware residual、transductive ID side channel 等非可解释 CDM 改进；候选结构必须能落回知识点、Q 矩阵、TKC/UKC、guess/slip、概念图传播或学生-知识点历史证据。

## What I already know

* 用户要求继续探索直到出现明显信号，接受大结构改动与参数探索。
* 用户明确质疑 CF 的可解释性；本轮排除非可解释 CDM 改进。
* 当前正式主线是实验 70：实验 34 底座 + 实验 49 pairwise history residual + 实验 51 interpretable readout expert + 实验 70 student-conditioned UKC `none_seen` readout sidecar。
* 当前主线三 seed 均值约 `test_auc = 0.765517`，冲刺目标仍是 `test_auc ~= 0.780`。
* 实验 77 的 hybrid CF 有预测信号，但已标注为 ID-aware 上界候选，不作为纯 CDM 主线。
* 已知失败或降级路线：
  * recency-aware history state 拒绝。
  * q_repr 写回、target-conditioned student context、local expert、multi-hop propagation、difficulty weighting、learned attribution 都没有形成 clean overall 突破。
  * 实验 66 evidence-aware TKC propagation 有机制信号，但全量 history-feature 注入三 seed 不稳；复盘建议若复访，应只做更局部的 behavior gate 调节。
* 用户已纠正分支规则：当前项目是原始项目 worktree，Trellis 安装在 `exp/trellis-trial` 线上；后续把 `exp/trellis-trial` 作为当前“伪主线”。
* 本轮不能从 `master` 直接切分支，否则会丢失 Trellis 文件/hooks 的检测能力；只能从 `exp/trellis-trial` 或其后代继续。

## Assumptions

* “明显信号”按当前项目协议解释为：大结构单 seed `AUC +0.002` 左右，或 `AUC/ACC` 同向达到约 `1e-3` 以上且 `RMSE/Brier/ECE` 没有明显副作用，并有 slice 支撑。
* 可解释 CDM 允许使用学生级或学生-知识点级历史统计、学生级 guess/slip、概念图、Q 矩阵和题目难度；不允许使用低秩 student-exercise ID residual 或直接记忆特定学生-题目 pair。
* 若某条结构首轮弱但机制不崩盘，可以做收敛的小范围参数扫描；若仍弱，则继续下一条可解释结构假设。

## Requirements

* 从 `exp/trellis-trial` 伪主线或当前 Trellis-enabled 分支切新实验分支，暂定 `exp/evidence-calibrated-behavior-gate`。
* 第一条结构假设：evidence-calibrated TKC behavior gate。
  * 在数据侧构造可解释的 `student-concept` 历史证据张量：attempt、correct、incorrect、accuracy、log_attempt、seen。
  * 在 TKC propagation 中只用这些 `student-concept` 证据调节正题/错题行为消息的融合 gate。
  * 保持 `valid/test` 复用 train history，不引入 valid/test target 行为。
  * 新模块默认关闭；关闭时主线行为不变。
  * 初版必须 zero-init 或近似保持主线初始行为，避免一开关就破坏底座。
  * 训练输出、evaluate 输出和 slice 分析要记录该开关及关键参数。
* 第一轮至少跑当前 Exp70 seed=2024 口径的正式远端训练，并与 `results/assist_09_mainline_300ep.json` 对比。
* 如果第一轮没明显信号但不崩盘，允许小范围扫以下参数之一到两个：
  * residual scale / max logit
  * trigger：all concepts vs low-evidence concepts vs multi-concept targets
  * learning-rate rescue 仅限一个邻近点
* 如果 evidence-calibrated behavior gate 仍没有明显信号，继续下一条可解释结构假设，而不是停在弱结果。
* 本轮不得把实验 77 的 hybrid CF side channel 作为候选，也不得引入新的 student-exercise ID-aware residual。
* 远端训练、评估、测试通过 `bash scripts/remote_exec.sh <command>` 执行；远端运行前必须 commit 并 push 当前分支。
* 每条已执行实验都要记录命令、结果路径、指标与结论；完成阶段更新实验台账。

## Acceptance Criteria

* [ ] 新实验分支从 `exp/trellis-trial` 伪主线或其后代创建，且未继承实验 77 的 CF 作为默认底座。
* [ ] 实现一个可由 CLI/config 控制的 evidence-calibrated TKC behavior gate，默认关闭。
* [ ] 新增 `student-concept` history evidence tensors，且不改变 valid/test history visibility 语义。
* [ ] 默认关闭时当前主线行为不变。
* [ ] 添加 focused unit tests 覆盖 evidence tensor 构造、默认关闭路径和启用路径。
* [ ] 远端单测和 smoke 通过。
* [ ] 至少完成 seed=2024 远端正式训练并与当前主线同 seed 对比。
* [ ] 若未出现明显信号，继续推进下一条可解释结构假设或收敛扫描。
* [ ] 实验台账更新，包含结果路径、核心指标、slice/副作用判断和是否继续的理由。

## Definition of Done

* Tests added/updated where behavior changes are testable without long training.
* Syntax/type-level checks pass locally where cheap.
* Remote tests/smoke/formal training executed according to protocol after commit and push.
* Experiment ledger updated.
* Rollback path is clear: disabling the new flag restores current mainline behavior.

## Out of Scope

* 不引入 CF、matrix factorization、student-exercise ID-aware residual 或其他 transductive pair memory。
* 不把探索性代码直接合入 `master`。
* 不修改数据 split 或引入 valid/test target history。
* 不做无上限组合爆炸；每条假设只做必要的最小扫描。

## Technical Notes

* Workflow source: `.trellis/workflow.md` via `trellis-start`.
* Experiment protocol: `.trellis/spec/backend/experiment-protocol.md`.
* Current mainline and history: `docs/model_improvement_plan.md`, `docs/handoff.md`, `docs/experiment_index.jsonl`.
* Relevant prior docs:
  * `docs/experiments/066_evidence_aware_tkc_propagation.md`
  * `docs/experiments/072_representation_bottleneck_probes.md`
  * `docs/experiments/077_recency_history_and_hybrid_cf.md`
* Likely files:
  * `data/datasets.py`
  * `data/pipeline.py`
  * `models/hetero_propagation.py`
  * `models/decoupled_cdm.py`
  * `scripts/train.py`
  * `scripts/evaluate.py`
  * `scripts/analyze_prediction_slices.py`
  * `tests/test_*`
