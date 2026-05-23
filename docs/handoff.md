# Handoff

这份文档只回答新会话最需要的几个问题：当前默认从哪里出发、哪个 runner 是当前 trial、哪些高分结果不能误当默认主线。

流程约束仍以 Trellis 为准：`.trellis/workflow.md` 和 `.trellis/spec/backend/experiment-protocol.md`。实验细节查 `docs/model_improvement_plan.md`、`docs/experiment_index.jsonl` 和 `docs/experiments/`。

## 当前状态

- 当前分支: `exp/trellis-trial`。
- accepted `master` 模型主线: 实验 70，三 seed mean `test_auc = 0.765517`。
- Trellis pseudo-mainline / baseline runner: 实验 81，`scripts/run_assist09_baseline.sh`，在实验 70 上加入 single-only concept-evidence readout。
- 当前 pure-CDM trial runner: 实验 104，`scripts/run_assist09_history_alignment_trial.sh`。
- 实验 106: 已记录的 branch-BCE `0.18` refinement 候选；当前 HEAD 未把它提升为 runner 默认。
- 当前 practical target: `test_auc >= 0.778`；`0.780` 仍是 desirable headroom。
- 当前探索重点: 双塔路线已过线但显存约 `11.76GB`，后续优先找 `~7GB` 级别更稳定的 pure-CDM 信号或更高峰值；不要把 104 后的一串实验误读成默认主线连续 promote。

## 当前 Trial 主线

当前 trial 的 single-run / single-checkpoint pure-CDM 默认路线是:

```bash
bash scripts/run_assist09_history_alignment_trial.sh
```

该 runner 当前编码实验 104:

- 底座: 实验 81 baseline runner。
- 训练目标: 实验 95/103 系列的 `loss_only` history-evidence cognitive alignment，不加 output-logit sidecar。
- 结构: 实验 104 双塔 `DecoupledCDMEnsemble`，primary `concept_dim=64`，secondary `concept_dim=80`。
- branch BCE: `dual_cdm_branch_bce_weight=0.10`。
- 四 seed AUC: `0.778773/0.778250/0.778508/0.777948`。
- mean AUC: `0.778370`，stdev `0.000306`。
- 约束: 单次训练、单 checkpoint、pure CDM；没有 fixed checkpoint average、valid-trained combiner、hybrid train-history tabular side channel。

新的 pure-CDM single-checkpoint 想法默认应对比当前 active runner，也就是实验 104 mean `0.778370`。如果要把实验 106 作为新 baseline，需要先明确 promote runner/default contract。

## 已记录但未提升为默认的候选

- 实验 106 branch-BCE refinement: 在实验 104 双塔 `64x80` 底座上把 `dual_cdm_branch_bce_weight` 从 `0.10` 提到 `0.18`。
- 实验 106 四 seed AUC: `0.778890/0.778552/0.778256/0.778618`，mean `0.778579`，stdev `0.000226`。
- 这比实验 104 mean 高 `+0.000209`，但当前代码默认 runner/spec 仍保持实验 104；不要把实验 106 写成 active default，除非本轮任务明确要求 promote。

## 为什么 104 后还有实验

104 的问题不是 AUC 没过线，而是路线太重:

- 104/106 双塔 `64x80` peak CUDA 约 `11.76GB`。
- 用户想让后续 Codex 继续找的是 `~7GB` 显存水平下更稳定的信号，或能给出更高峰值的新机制。
- 实验 105-108 因此主要是 low-VRAM / single-tower / shared-branch / schedule / trigger follow-up，不是默认 runner promotion 链。
- 这些 follow-up 当前大多判负: single64 是轻量参考但四 seed mean 只有 `0.776736`；shared-branch、single72、dim64 output alignment、checkpoint smoothing、soft difficulty regularization、trigger threshold 收紧、cognitive focusing、validation `brier` checkpoint selection 都没形成可替代 104 的稳定路线。

后续如果继续这条线，目标不是继续微调这些已判负旋钮，而是提出新的 `~7GB` 机制假设。

## 不是默认主线的高分路线

- 实验 106 branch-BCE `0.18`: pure-CDM single-run refinement 候选；当前不是 HEAD runner 默认。
- 实验 99 hybrid stacker: 三 seed mean `0.786910`，是当前最高绝对 AUC 信号；但它是 valid-trained hybrid evaluator，不是默认 CDM runner。
- 实验 102 checkpoint average: 四 seed mean `0.778872`，是 pure-CDM evaluator 上界；但它是 two-checkpoint inference evaluator，不是 single-run/default-training 答案。
- 实验 105 single64: seed2024 `0.778379`、显存约 `6.33GB`；但历史四 seed mean `0.776736`，只作为轻量参考，不替代实验 104。

## 当前不要再混淆的路线

- 不要把 `scripts/run_assist09_baseline.sh` 改成 trial runner；baseline 是实验 81。
- 不要把实验 106 写成当前默认 runner，除非同时显式修改 runner/spec 并确认 promote 决策。
- 不要把 hybrid stacker 或 checkpoint average 写成默认训练主线。
- 不要把实验 76/78 deterministic evidence-prior 旧伪主线当作当前 trial 默认组件；它们已因 seed 退化回退。
- 不要继续围绕实验 100/103 的 dim80 / output-alignment / smooth loss / reliability weighting 做局部参数小扫，除非有新的机制假设。
- 不要把 student/exercise direct history terms 加回默认 trial；当前保留 cogonly 语义。
- 不要继续 exp105 single64 的 soft difficulty regularization、trigger threshold 收紧、cognitive residual focusing、validation `brier` checkpoint selection 等已判负 follow-up。

## 后续优先级

- 若继续 heavy pure-CDM 论文路线: 从当前 active runner 实验 104 出发；如果采用实验 106，需要先确认 promote。
- Heavy-route ablation: `64x64/64x80/80x80`、branch 单独 AUC、融合 AUC、branch BCE 邻域、`cognitive/guess/slip` 组件语义稳定性。
- 若继续用户当前更关心的低显存路线: 以实验 105 single64 作为轻量参考，目标是 `~7GB` 下找到比 `0.776736` 四 seed mean 更稳定、且尽量接近或超过 104/106 的新机制。
- 不要继续围绕 105-108 已判负的小旋钮拉长实验链；下一步需要机制级假设，例如更轻的双视角共享、蒸馏/一致性训练、低成本表示交互，或更可靠的 checkpoint/validation 选择语义。
- 若继续最高绝对 AUC: 从实验 99 hybrid stacker 出发，但必须明确标注为 hybrid evaluator / model-integration task。

## 固定数据与协议

- Dataset: `data/assist_09_ordered`。
- Train/valid/test:
  - `data/assist_09_ordered/train.csv`
  - `data/assist_09_ordered/valid.csv`
  - `data/assist_09_ordered/test.csv`
- Q matrix: `data/assist_09_ordered/Q_matrix.csv`。
- Graph: `data/assist_09_ordered/transition_graph/propagation_graph.csv`。
- Graph mode: single graph。dual graph 只作为 legacy ablation。
- valid/test 传播输入复用 train history；不要用 valid/test target rows 重建历史。
- 结构比较默认按 300 epoch 量级看，不用 20 epoch 下结论。

## 关键文件

- Baseline runner: `scripts/run_assist09_baseline.sh`。
- Trial runner: `scripts/run_assist09_history_alignment_trial.sh`。
- Train CLI: `scripts/train.py`。
- Main model: `models/decoupled_cdm.py`。
- Ensemble wrapper: `models/ensemble_cdm.py`。
- Trainer: `trainers/engine.py`。
- Protocol spec: `.trellis/spec/backend/experiment-protocol.md`。
- Current snapshot: `docs/model_improvement_plan.md`。
- Experiment index: `docs/experiment_index.jsonl`。
- Detail docs: `docs/experiments/`.

## 查证顺序

1. 只想知道当前 trial: 读本文件的 "当前 Trial 主线"。
2. 要改 runner 或默认协议: 读 `.trellis/spec/backend/experiment-protocol.md`。
3. 要查实验号和状态: 读 `docs/experiment_index.jsonl`。
4. 要复现实验命令/结果文件: 读对应 `docs/experiments/<id>_*.md`。
5. 要理解路线取舍: 读 `docs/model_improvement_plan.md`。
