# 联合 standard/holdout 无回归实验最终结果（2026-07-10）

## 结论

本轮联合 campaign 的最终 test 硬门结果是 **0/3 通过**，不能表述为插件路线成功。三个冻结配方都在 validation 达到 `decision=shared` 和预注册的最小 AUC 安全余量 `+0.001`，但 test-once 后均未通过完整硬门：ASSIST17 只差绝对 DOA 门槛 `3.318e-7`；XES3G5M 的 standard/holdout AUC 都提升，但 ordinary/weighted DOA 差值及绝对门槛失败；ASSIST09 的四个相对改善门都通过，但绝对门槛仍差 `0.0019317443`。

因此，validation 上的 `shared` 只表示配方获准冻结并进入一次性 test，不能等同于 test 胜出。按预注册规则，本轮已经停止，没有使用 test 结果继续搜索、调参或重试。

## 固定协议

- `train_seed=42`、`doa_seed=42`、`split_seed=2024`、`min_responses=3`。
- 配方只使用 standard/holdout validation 选择；双 AUC 均须改善，holdout ordinary/weighted DOA 均须改善，且 `min_validation_auc_delta >= +0.001` 才能冻结为 `shared`。
- 三个数据集、两个 protocol、baseline/plugin 两臂共形成 12 个唯一 frozen ID；统一 ledger 中有且仅有 12 个 test-once claim。
- 12 个 actual test evaluation 各执行一次，没有重试。outer runner 的 `--dataset-file` 不包含 `test.csv`，也未在 claim 前哈希 test；test 由 inner evaluator 在 claim 成功后读取。
- holdout DOA 只从 baseline/plugin 一次评估生成的 evaluation cache 计算；cached DOA runner 不接收 test path 或 test split 参数，也不重读 test。standard 不计算 DOA。

## Validation 冻结结果

### 共享骨干路线（已运行）

| 数据集 | 骨干 | 初始化路线 | 冻结配方 | standard AUC delta | holdout AUC delta | ordinary DOA delta | weighted DOA delta | validation 判定 |
|---|---|---|---|---:|---:|---:|---:|---|
| ASSIST17 | ORCDF | random shared | `aux-w050` | +0.0014753827 | +0.0010801356 | +0.0098298001 | +0.0052414271 | `shared` |
| XES3G5M | ORCDF | baseline-finetune | `aux-w050-ft-lr025` | +0.0021497799 | +0.0021348435 | +0.0312341167 | +0.0127877238 | `shared` |
| ASSIST09 | SVGCD | baseline-finetune | `aux-w025-ft-lr025` | +0.0039326036 | +0.0029302412 | +0.0055219719 | +0.0021876799 | `shared` |

ASSIST17 是随机初始化的共享骨干配方；XES3G5M 和 ASSIST09 是从各自 fresh λ=0 baseline checkpoint 初始化的 baseline-finetune 共享骨干配方。二者都是共享骨干训练路线，不是 prediction-invariant adapter。

### Prediction-invariant adapter 路线（未运行）

| 数据集 | 状态 | 原因 |
|---|---|---|
| ASSIST17 | 未运行 | random shared 已通过 validation 冻结门，按停止规则直接进入 test-once。 |
| XES3G5M | 未运行 | baseline-finetune 在首个预注册倍率 0.25 通过 validation 冻结门，未进入 adapter。 |
| ASSIST09 | 未运行 | baseline-finetune 在首个预注册倍率 0.25 通过 validation 冻结门，未进入 adapter。 |

adapter 路线没有 artifact、test claim 或性能结果，不得计入成功或失败数量。

## Test-once 指标与硬门

下表的 absolute threshold 作用于 plugin ordinary holdout DOA。最终结果要求 standard AUC delta、holdout AUC delta、ordinary DOA delta、weighted DOA delta 及 absolute threshold 同时通过。

| 数据集 | std base | std plugin | std Δ | holdout base | holdout plugin | holdout Δ | ordinary base | ordinary plugin | ordinary Δ | weighted base | weighted plugin | weighted Δ | absolute threshold | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ASSIST17 | 0.7868747971 | 0.7877898869 | +0.0009150899 | 0.7836596181 | 0.7848584806 | +0.0011988625 | 0.6946270918 | 0.7088566682 | +0.0142295764 | 0.7016505209 | 0.7090716432 | +0.0074211223 | `> 0.708857` | **FAIL**；仅 threshold 失败，低 `0.0000003318`（`3.318e-7`） |
| XES3G5M | 0.7925144228 | 0.7955118929 | +0.0029974701 | 0.7848765165 | 0.7881561267 | +0.0032796101 | 0.6449238638 | 0.6424787965 | -0.0024450673 | 0.6998844275 | 0.6921671700 | -0.0077172576 | `>= 0.664573` | **FAIL**；两项 AUC 提升，但两项 DOA delta 与 threshold 失败 |
| ASSIST09 | 0.7756980781 | 0.7794331978 | +0.0037351197 | 0.7686488688 | 0.7717050169 | +0.0030561481 | 0.6652624498 | 0.6688742557 | +0.0036118059 | 0.6830105966 | 0.6865567454 | +0.0035461488 | `> 0.670806` | **FAIL**；四个相对门全过，仅 threshold 失败，低 `0.0019317443` |

逐数据集解释：

- ASSIST17 的双 AUC 和双 DOA 相对改善都为正，唯一失败项是严格绝对门槛；`0.7088566682` 没有满足 `>0.708857`，即使差距只有 `3.318e-7` 也不能四舍五入为通过。
- XES3G5M 的 standard/holdout AUC 分别提高 `0.0029974701` 和 `0.0032796101`，但 ordinary/weighted DOA 分别下降 `0.0024450673` 和 `0.0077172576`，ordinary DOA 也低于绝对门槛，故不是无回归诊断增强。
- ASSIST09 的两项 AUC 与两项 DOA 相对改善全部通过，但 ordinary DOA `0.6688742557` 仍未超过 `0.670806`，不能记为成功。

## Test-once 与历史例外

本次 joint-split final campaign 修正了旧 ASSIST17 campaign 的协议偏差：四个新 ASSIST17 outer runner 都没有在 claim 前读取或哈希 test，行为与 XES3G5M、ASSIST09 一致。历史旧 ASSIST17 artifact 曾在 claim 前把 `test.csv` 列为 outer runner 哈希输入；其模型 test 推理仍只运行一次，DOA 也未重读 test，但该资产仍是预哈希例外，不能包装成完美的 test-once 样本，也不用于本轮 0/3 以外的新计数。

本轮 final ledger 的 12 个 frozen ID、claim path 和 claim 文件均唯一；12 个 selection test evaluation 与 3 个 cache-only DOA root 均只有 `attempt-001=dry_run`、`attempt-002=completed/0`。冻结之后没有新配方、补跑或 test 驱动的选择。

## 后续边界

不得根据这三个已见 test 的结果继续调整当前插件配方。若继续研究插件路线，只能在尚未打开 test 的新数据集上另立预注册计划并重新完成 validation 冻结与 test-once；若继续当前数据集，应转入完整模型路线的独立计划，明确新的研究问题和计数口径。上述后续工作均不属于本阶段，也不得混入本轮 validation `shared=3/3` 或 test hard-gate `pass=0/3` 的计数。

## 最终状态

`DONE_WITH_HARD_GATE_FAILURES`：软件与 test-once 协议审计完成；validation 冻结为 `shared` 的三个配方在最终 test 硬门上均失败，最终成功数为 **0/3**。
