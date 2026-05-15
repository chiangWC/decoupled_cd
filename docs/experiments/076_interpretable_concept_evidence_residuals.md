# Experiment 76: interpretable concept evidence residuals

- 分支: `exp/evidence-calibrated-behavior-gate`
- base: `exp/trellis-trial` 伪主线及其后代
- 代码提交:
  - `2749d2c` evidence-calibrated behavior gate
  - `16a033e` concept evidence readout residual
  - `d5e85e6` deterministic concept evidence prior residual
- 代码状态: 探索性实现保留在实验分支；尚未合入正式主线
- 主线参考: `results/assist_09_mainline_300ep.json`

## 动机

本轮只探索可解释 CDM 结构，不使用 CF、student-exercise ID residual、transductive pair memory 等非 CDM side channel。可用信号限制在 Q 矩阵、知识点、TKC/UKC、概念图、difficulty 和 train-history student-concept evidence。

## 工程验证

- 本地语法检查:
  - `python3 -m py_compile models/decoupled_cdm.py scripts/train.py scripts/evaluate.py scripts/analyze_prediction_slices.py tests/test_decoupled_cdm.py`
  - `git diff --check`
- 远端单测:
  - `python -m unittest tests.test_decoupled_cdm tests.test_history_visibility tests.test_hetero_propagation tests.test_training_modes`
  - 结果: 通过，最终 `30 tests`
- 远端 smoke:
  - `OUTPUT=results/concept_evidence_prior/smoke_seed2024_min2_seen1_max05_strength2_cap20_1ep_2000.json bash scripts/run_assist09_concept_evidence_prior.sh --epochs 1 --max-rows 2000 --device cpu`
  - 结果: 通过

## 结构 1: evidence-calibrated TKC behavior gate

做法:

- 数据侧新增 `student_concept_evidence`，每个 student-concept 单元包含 `attempt/correct/incorrect/accuracy/log_attempt/seen`。
- TKC propagation 中用该 evidence 调节 correct/incorrect behavior gate logits。
- residual zero-init，默认关闭。

结果: 拒绝。

| config | test_auc | test_acc | test_rmse | test_brier | test_ece | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mainline | 0.765444 | 0.730404 | 0.427481 | 0.182740 | 0.051686 | reference |
| max0.5 all | 0.762811 | 0.730461 | 0.428099 | 0.183269 | 0.051342 | AUC 明显回撤 |
| max0.5 low_evidence | 0.762909 | 0.728615 | 0.428719 | 0.183800 | 0.050932 | AUC/ACC 回撤 |

## 结构 2: target-local concept evidence readout residual

做法:

- 对目标题 Q 概念聚合 student-concept evidence。
- 只在满足 `concept_count` 与 `seen_ratio` 的目标上启用 trainable zero-init cognitive-logit residual。
- 默认配置先保护 `seen_ratio=1.0`，避免原始 history residual 的 none_seen 副作用。

结果: 拒绝。val AUC 有提高，但 test 没有转成 clean gain。

| config | test_auc | delta AUC | test_acc | test_rmse | test_brier | test_ece | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| min2 seen1 max0.5 | 0.765391 | -0.000053 | 0.728292 | 0.428375 | 0.183505 | 0.054592 | 持平略降且副作用明显 |
| min2 seen1 max0.25 | 0.765296 | -0.000149 | 0.728596 | 0.428185 | 0.183343 | 0.053455 | 降幅后仍不 clean |
| min2 seen0.5 max0.25 | 0.765345 | -0.000099 | 0.728577 | 0.428164 | 0.183325 | 0.053387 | 扩作用域仍无信号 |

## 结构 3: deterministic concept evidence prior residual

做法:

- 对目标题 Q 概念上的 train-history `correct/attempt` 做 beta 平滑:
  - `smoothed_accuracy = (correct + 0.5 * prior_strength) / (attempt + prior_strength)`
- 再用 `seen_ratio` 与 `log1p(attempt_count) / log1p(confidence_cap)` 缩放。
- 作为有界 cognitive-logit residual:
  - `residual = max_logit * confidence * seen_ratio * (smoothed_accuracy - 0.5) * 2`
- 该结构是确定性、可解释、非 ID-pair memory；默认关闭。

结果: `min_count=1` 出现明显单 seed 信号。

| config | test_auc | delta AUC | test_acc | delta ACC | test_rmse | delta RMSE | test_brier | delta Brier | test_ece | delta ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| min2 seen1 max0.5 | 0.766404 | +0.000960 | 0.729985 | -0.000419 | 0.427394 | -0.000087 | 0.182666 | -0.000074 | 0.051922 | +0.000236 |
| min2 seen1 max0.25 | 0.764937 | -0.000507 | 0.729148 | -0.001256 | 0.427636 | +0.000155 | 0.182873 | +0.000133 | 0.049785 | -0.001902 |
| min2 seen1 max0.75 | 0.766605 | +0.001161 | 0.729186 | -0.001218 | 0.427462 | -0.000019 | 0.182724 | -0.000017 | 0.052701 | +0.001015 |
| min2 seen0.5 max0.5 | 0.766428 | +0.000984 | 0.729985 | -0.000419 | 0.427383 | -0.000098 | 0.182656 | -0.000084 | 0.051924 | +0.000237 |
| min1 seen1 max0.5 | 0.770505 | +0.005061 | 0.730651 | +0.000247 | 0.425893 | -0.001588 | 0.181385 | -0.001355 | 0.053886 | +0.002199 |

Best result path:

- `results/concept_evidence_prior/assist_09_seed2024_min1_seen1_max05_strength2_cap20_300ep.json`
- slice:
  - `results/concept_evidence_prior/assist_09_seed2024_min1_seen1_max05_strength2_cap20_slices.json`
  - `results/concept_evidence_prior/assist_09_mainline_slices_for_prior_compare.json`

## Slice 对照

相对当前主线 seed=2024:

- `concept_count=1`: `AUC 0.768689 -> 0.774700`，`ACC +0.000297`，`RMSE -0.002252`，`ECE +0.000314`
- `all_seen`: `AUC 0.762771 -> 0.768124`，`ACC +0.000158`，`RMSE -0.001681`，`ECE +0.001862`
- `none_seen`: `AUC 0.814497 -> 0.815608`，`ACC +0.008444`，`RMSE -0.002687`，`ECE -0.008072`
- `concept_count=2`: AUC 基本持平，但 `ACC/RMSE/ECE` 变差。
- `partial_seen`: 样本少，且 AUC/ACC/ECE 都变差；当前最佳配置实际收益来自单知识点和 all_seen 主体。

## 结论

- 可以停止本轮自由探索: 已出现明显增长信号。
- `concept_evidence_prior_residual` 的 `min_count=1, seen_ratio=1.0, max_logit=0.5, prior_strength=2.0, confidence_cap=20` 是新的可解释 CDM 候选。
- 单 seed 已达到 `AUC +0.005061`，且 `ACC/RMSE/Brier` 同向；主要风险是 `ECE +0.002199`。
- 下一步应优先补 2 个 seed，而不是继续扩同类 final-logit 变体。若多 seed 稳定，再考虑 calibration rescue，例如较小 `max_logit` + `min_count=1` 或温度/校准后处理；不要用 CF/ID side channel 解释这条收益。
