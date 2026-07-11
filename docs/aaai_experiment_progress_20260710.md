# AAAI 双路线实验进度（2026-07-10）

## 当前结论：联合插件硬门 0/3

seed 42 的 joint standard/holdout campaign 已按预注册停止规则结束。ASSIST17、XES3G5M、ASSIST09 的冻结插件配方都在 validation 达到 `decision=shared` 及 `min_validation_auc_delta >= +0.001`，但最终 test-once 完整硬门为 **0/3 通过**。因此，旧版进展中“插件路线 3/3 达标/胜”的表述不再是当前结论；它只描述早期单划分历史 campaign，不能与本轮联合 standard/holdout 结果混合计数。

详细 base/plugin 指标、两项 AUC delta、ordinary/weighted DOA delta、绝对门槛及协议审计见 `docs/joint_split_no_regression_results_20260710.md`。

| 数据集 | validation 冻结路线 | test 相对门 | test 绝对门槛 | 最终结果 |
|---|---|---|---|---|
| ASSIST17 | random shared，`aux-w050` | 双 AUC、双 DOA delta 均通过 | ordinary DOA `0.7088566682`，要求 `>0.708857`，低 `3.318e-7` | **FAIL** |
| XES3G5M | baseline-finetune shared，`aux-w050-ft-lr025` | 双 AUC 通过；ordinary/weighted DOA delta 失败 | ordinary DOA `0.6424787965`，要求 `>=0.664573` | **FAIL** |
| ASSIST09 | baseline-finetune shared，`aux-w025-ft-lr025` | 双 AUC、双 DOA delta 均通过 | ordinary DOA `0.6688742557`，要求 `>0.670806`，低 `0.0019317443` | **FAIL** |

这里的 `shared=3/3` 是 validation 冻结判定，不是 test 成功数。Prediction-invariant adapter 没有运行，也没有 artifact 或结果；不得把 baseline-finetune 称为 adapter。最终 test 成功数只能记为 `0/3`。

## 当前联合 campaign 协议

- 统一设置为 `train_seed=42`、`doa_seed=42`、`split_seed=2024`、`min_responses=3`，validation 安全余量为 `+0.001`。
- 配方只由 standard/holdout validation 决定，冻结后才进入 test-once。三个数据集、两个 protocol、baseline/plugin 两臂共有 12 个唯一 frozen ID 和 12 个唯一 ledger claim。
- 本轮 12 个 actual test evaluation 各运行一次，没有重试；outer runner 未把 `test.csv` 放入 dataset hash，inner evaluator 在不可覆盖 claim 成功后才读取 test。
- holdout DOA 仅从一次评估留下的 cache 计算，cached DOA runner 没有 test path/split 参数，也未重读 test。
- 冻结后的 test 结果没有用于新搜索、调参或补跑。后续插件研究只能使用尚未打开 test 的数据集另立计划；完整模型路线也须作为独立阶段，不混入本轮计数。

## 历史插件 campaign（不纳入当前计数）

早期单划分 campaign 曾按“holdout DOA 提高且同骨干 AUC 下降不超过 0.002”的旧口径报告插件 3/3：ASSIST17 test holdout DOA `0.711672`、XES3G5M `0.664573`、ASSIST09 `0.677355`。这些结果属于另一套选择与门槛，不能覆盖本轮联合双划分的预注册 hard gate。

历史 ASSIST17 还有一项已知协议例外：旧 outer runner 在 claim 前将 `test.csv` 列为哈希输入。模型 test 推理仍只执行一次，后续 DOA 未重读 test，但该资产不是完美的 test-once 审计样本。本次 joint-split final campaign 的四个新 ASSIST17 outer runner 已修正该问题，均未在 claim 前读取或哈希 test。

历史实现问题仍需保留：r18 `plugin-decouple` 的 `ukc_gate` 曾漏入 optimizer，相关 no-op 对照不能作为机制证据；r22 所谓“密度”实际恒为 `log(2)`；旧 SVGCD 三阶段训练有阶段间残留梯度且 OneCycleLR 未按真实 step 更新；旧 ORCDF 固定 scale/插件 gate 的参数注册也有问题。本轮冻结配方使用修正后的实现和 fresh baseline。

## 完整模型路线：历史 2/3

| 数据集 | 冻结配置 | test overall AUC | test zero AUC | 目标 | 历史结论 |
|---|---|---:|---:|---:|---|
| ASSIST17 | v2 base, full batch, dim 64, 300 epochs | 0.786333 | 0.783969 | `>0.7808` | 通过 |
| MOOCRadar | hybrid+mono+UKC, student minibatch 64, dim 64, 30 epochs | 0.929329 | 0.946124 | `>0.9454` | 通过；overall 距 0.9300 约 0.000671 |

完整模型历史计数仍是 2/3，不能声称三个数据集胜出。XES3G5M base 长训当时因早期插件 campaign 触发停止条件而未启动；corrected-support 只有单元测试与 smoke 证据，不是性能结论。若补做完整模型标准实验，必须另立后续计划和计数，不能作为本阶段插件失败后的 test 调参。

## 数据与发布边界

- 联合 campaign artifact 位于 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-joint-splits-20260710/`，不进入 Git。
- NIPS34 等未开 test 数据集只能在新的预注册 campaign 中使用；历史结果不得混入当前计数。
- GPU 策略为优先空闲卡、显存低于 50% 时可共享；本轮 final evaluation 无 OOM，结束后无本方残留 GPU 作业。
- vendor ORCDF/SVGCD 快照缺少许可证，仅限远端内部实验，不得 push、公开或再分发。
