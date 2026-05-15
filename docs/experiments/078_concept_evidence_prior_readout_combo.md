# Experiment 78: concept evidence prior plus readout correction

- 分支: `exp/concept-evidence-prior-next`
- base: `exp/trellis-trial` 伪主线 `409f334`
- 代码状态: 无新增代码；复用实验 76 已合入伪主线的 concept evidence prior/readout flags
- 伪主线参考: `results/concept_evidence_prior/assist_09_seed2024_min1_seen1_max05_strength2_cap20_300ep.json`

## 动机

实验 76 的 deterministic concept evidence prior 已形成明确可解释信号，但 ECE 略差。继续探索时先做小范围参数扫描和同源可解释修正，要求仍然只使用 student-concept train-history evidence，不引入 CF、student-exercise ID residual 或 transductive pair memory。

## 运行与结果

| config | result path | test_auc | delta AUC | test_acc | delta ACC | test_rmse | delta RMSE | test_brier | delta Brier | test_ece | delta ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp76 pseudo-mainline | `results/concept_evidence_prior/assist_09_seed2024_min1_seen1_max05_strength2_cap20_300ep.json` | 0.770505 | 0.000000 | 0.730651 | 0.000000 | 0.425893 | 0.000000 | 0.181385 | 0.000000 | 0.053886 | 0.000000 | reference |
| lr7e-4 es20 sched5 | `results/concept_evidence_prior_next/assist_09_seed2024_lr7e4_es20_sched5_300ep.json` | 0.770125 | -0.000381 | 0.732193 | +0.001541 | 0.425288 | -0.000606 | 0.180870 | -0.000516 | 0.050586 | -0.003300 | calibration rescue, not AUC signal |
| prior_strength=1.0 | `results/concept_evidence_prior_next/assist_09_seed2024_strength1_cap20_max05_300ep.json` | 0.771292 | +0.000787 | 0.723839 | -0.006813 | 0.428912 | +0.003019 | 0.183966 | +0.002581 | 0.068187 | +0.014301 | ranking signal with large side effects |
| prior_strength=1.5 | `results/concept_evidence_prior_next/assist_09_seed2024_strength15_cap20_max05_300ep.json` | 0.770421 | -0.000084 | 0.731108 | +0.000457 | 0.425877 | -0.000017 | 0.181371 | -0.000014 | 0.053580 | -0.000305 | near-neutral |
| max_logit=0.6 | `results/concept_evidence_prior_next/assist_09_seed2024_strength2_cap20_max06_300ep.json` | 0.770597 | +0.000092 | 0.727340 | -0.003311 | 0.427769 | +0.001876 | 0.182986 | +0.001601 | 0.061900 | +0.008014 | not clean |
| prior + readout min1 max0.5 | `results/concept_evidence_prior_next/assist_09_seed2024_prior_default_plus_readout_min1_max05_300ep.json` | 0.772562 | +0.002056 | 0.730823 | +0.000171 | 0.424811 | -0.001082 | 0.180464 | -0.000921 | 0.052361 | -0.001524 | clear single-seed signal |

Best command:

```bash
OUTPUT=results/concept_evidence_prior_next/assist_09_seed2024_prior_default_plus_readout_min1_max05_300ep.json \
  bash scripts/run_assist09_baseline.sh \
  --concept-evidence-readout-residual \
  --concept-evidence-readout-min-count 1 \
  --concept-evidence-readout-min-seen-ratio 1.0 \
  --concept-evidence-readout-max-logit 0.5
```

Slice report:

- `results/concept_evidence_prior_next/assist_09_seed2024_prior_plus_readout_min1_max05_slices.json`
- `results/concept_evidence_prior_next/assist_09_seed2024_prior_plus_readout_min1_max05_slices.csv`

## Slice 对照

相对实验 76 伪主线:

- `concept_count=1`: `AUC 0.774700 -> 0.777905`, `ACC +0.000891`, `RMSE -0.001691`, `ECE -0.002991`。
- `all_seen`: `AUC 0.768124 -> 0.770869`, `ACC +0.000752`, `RMSE -0.001328`, `ECE -0.001302`。
- `concept_count=2`: `AUC -0.002849`, `ACC -0.003067`, `RMSE +0.002226`, `ECE +0.006198`。
- `none_seen`: `AUC -0.010753`, `ACC -0.015682`, `RMSE +0.005943`, `Brier +0.004271`, `ECE -0.005559`。
- `partial_seen`: 样本少，`AUC/ACC/RMSE/Brier/ECE` 均回撤。

## 结论

- 已出现下一条明显增长信号: 在实验 76 deterministic prior 上叠加同源 `concept_evidence_readout_residual(min_count=1, seen_ratio=1.0, max_logit=0.5)`，单 seed 相对伪主线 `AUC +0.002056`，且 `ACC/RMSE/Brier/ECE` 同向。
- 这条仍然具有可解释性: 只使用 train-history student-concept evidence，经 Q 矩阵聚合到目标题知识点，不使用 CF 或 student-exercise pair memory。
- 主要风险是收益集中在单知识点/all_seen 大样本，`none_seen` 与小样本 `partial_seen` 回撤；不能直接替换伪主线默认，应优先补 seed 并跟踪这些 slice。
