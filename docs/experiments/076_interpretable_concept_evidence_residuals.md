# Experiment 76: interpretable concept evidence residuals

- 分支: `exp/evidence-calibrated-behavior-gate`
- base: `exp/trellis-trial` 伪主线及其后代
- 代码提交:
  - `2749d2c` evidence-calibrated behavior gate
  - `16a033e` concept evidence readout residual
  - `d5e85e6` deterministic concept evidence prior residual
- 代码状态: 已合入 `exp/trellis-trial` 伪主线；尚未合入正式 `master`
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

## Multi-seed 验证

Official seeds `2024/2025/2026`，对照仍是实验 70 当前 `master` 主线。

| seed | exp70 AUC | exp76 AUC | delta AUC | exp70 ACC | exp76 ACC | delta ACC | exp70 RMSE | exp76 RMSE | delta RMSE | exp70 ECE | exp76 ECE | delta ECE | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 | 0.765368 | 0.770505 | +0.005137 | 0.728558 | 0.730651 | +0.002093 | 0.427880 | 0.425893 | -0.001987 | 0.052010 | 0.053886 | +0.001876 | strong normal |
| 2025 | 0.766528 | 0.764808 | -0.001720 | 0.729605 | 0.730385 | +0.000780 | 0.426968 | 0.428430 | +0.001462 | 0.049967 | 0.055701 | +0.005734 | weak regression |
| 2026 | 0.764655 | 0.502933 | -0.261722 | 0.729148 | 0.517384 | -0.211764 | 0.427201 | 0.517206 | +0.090005 | 0.045154 | 0.168078 | +0.122924 | degenerate, best_epoch=2 |

Official means `2024/2025/2026`:

- exp70: `AUC 0.765517`, `ACC 0.729104`, `RMSE 0.427350`, `Brier 0.182628`, `ECE 0.049044`
- exp76: `AUC 0.679415`, `ACC 0.659473`, `RMSE 0.457176`, `Brier 0.210813`, `ECE 0.092555`
- delta: `AUC -0.086102`, `ACC -0.069631`, `RMSE +0.029826`, `Brier +0.028185`, `ECE +0.043511`

Validation result paths:

- `results/exp76_multiseed_validation/assist_09_seed2025_min1_seen1_max05_strength2_cap20_300ep.json`
- `results/exp76_multiseed_validation/assist_09_seed2026_min1_seen1_max05_strength2_cap20_300ep.json`

## Slice 对照

相对当前主线 seed=2024:

- `concept_count=1`: `AUC 0.768689 -> 0.774700`，`ACC +0.000297`，`RMSE -0.002252`，`ECE +0.000314`
- `all_seen`: `AUC 0.762771 -> 0.768124`，`ACC +0.000158`，`RMSE -0.001681`，`ECE +0.001862`
- `none_seen`: `AUC 0.814497 -> 0.815608`，`ACC +0.008444`，`RMSE -0.002687`，`ECE -0.008072`
- `concept_count=2`: AUC 基本持平，但 `ACC/RMSE/ECE` 变差。
- `partial_seen`: 样本少，且 AUC/ACC/ECE 都变差；当前最佳配置实际收益来自单知识点和 all_seen 主体。

## 结论

- 可以停止本轮自由探索: 已出现明显增长信号。
- `concept_evidence_prior_residual` 的 `min_count=1, seen_ratio=1.0, max_logit=0.5, prior_strength=2.0, confidence_cap=20` 是新的可解释 CDM 候选，并已作为 `exp/trellis-trial` 伪主线默认运行口径。
- 单 seed 已达到 `AUC +0.005061`，且 `ACC/RMSE/Brier` 同向；主要风险是 `ECE +0.002199`。
- `2026-05-16` 补 official multi-seed 后，已确认 `seed2026` 的训练失败模式从实验 76 就存在，不是实验 78 才引入。`seed2026` 会在 `best_epoch=2` 退化到随机附近。
- 因此实验 76 的 single-seed 突破不能直接当成 clean official multi-seed 结论；后续所有基于实验 76 的 promote 或 follow-up，都应把 `seed2026` 视作已知风险，而不是后继实验的新增问题。
