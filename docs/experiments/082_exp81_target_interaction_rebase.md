# Experiment 82: exp81 exact-3 target interaction rebase

- 分支: `exp/exp81-target-concept-interaction-rebase`
- base: 当前 `exp/trellis-trial` 伪主线，已包含 experiment 81 的 `single-only concept-evidence readout`
- 对照: 同分支、同 seed、feature 关闭的 matched baseline

## 动机

实验 80 在已回退的 experiment 78 底座上给出过 `single-only readout + exact-3 target_concept_interaction_qrepr` 的正向 multi-seed 信号，但那条证据不能直接拿来作为当前 trial promote 依据。本轮要回答的是更严格的问题:

- 把 experiment 80 的核心机制重挂到当前 exp81 伪主线
- 保留 experiment 81 已成立的 `concept_count=1` readout
- 只在 `concept_count=3` 上追加 bounded q-repr interaction

这条组合在当前主线语义下还能否留下 clean overall gain。

## 实现状态

- 当前代码已支持:
  - `--target-concept-interaction-qrepr-adapter`
  - `--target-concept-interaction-min-count`
  - `--target-concept-interaction-max-count`
  - `--target-concept-interaction-max-scale`
- `train/evaluate/analyze_prediction_slices` 与单测入口已同步支持这些字段。
- 本轮 formal run 只比较 experiment 81 默认口径上再叠加 exact-3 interaction 的增量。

## 工程验证

- 远端单测:
  - `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - 结果: `38 tests`, `OK`
- 远端 smoke:
  - `bash scripts/run_assist09_baseline.sh --epochs 1 --max-rows 2000 --device cpu --target-concept-interaction-qrepr-adapter --target-concept-interaction-min-count 3 --target-concept-interaction-max-count 3 --target-concept-interaction-max-scale 0.25`
  - summary 已记录 `target_concept_interaction_*` 三个字段

## 正式训练

Matched baseline:

```bash
bash scripts/run_assist09_baseline.sh \
  --seed 2024 \
  --output results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_baseline_matched_300ep.json
```

Candidate configs:

```bash
bash scripts/run_assist09_baseline.sh \
  --target-concept-interaction-qrepr-adapter \
  --target-concept-interaction-min-count 3 \
  --target-concept-interaction-max-count 3 \
  --target-concept-interaction-max-scale 0.25 \
  --seed 2024 \
  --output results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_single_readout_plus_exact3_scale025_300ep.json

bash scripts/run_assist09_baseline.sh \
  --target-concept-interaction-qrepr-adapter \
  --target-concept-interaction-min-count 3 \
  --target-concept-interaction-max-count 3 \
  --target-concept-interaction-max-scale 0.125 \
  --seed 2024 \
  --output results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_single_readout_plus_exact3_scale0125_300ep.json
```

| config | result path | test_auc | delta AUC | test_acc | delta ACC | test_rmse | delta RMSE | test_brier | delta Brier | test_ece | delta ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp81 matched baseline | `results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_baseline_matched_300ep.json` | 0.770791 | 0.000000 | 0.731736 | 0.000000 | 0.425032 | 0.000000 | 0.180652 | 0.000000 | 0.047425 | 0.000000 | reference |
| exact-3 interaction, scale `0.25` | `results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_single_readout_plus_exact3_scale025_300ep.json` | 0.770613 | -0.000178 | 0.729700 | -0.002036 | 0.425904 | +0.000872 | 0.181395 | +0.000742 | 0.051044 | +0.003620 | rejected |
| exact-3 interaction, scale `0.125` | `results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_single_readout_plus_exact3_scale0125_300ep.json` | 0.770845 | +0.000054 | 0.730423 | -0.001313 | 0.425461 | +0.000429 | 0.181017 | +0.000365 | 0.048964 | +0.001540 | best but still not clean |

`scale=0.25` 直接从 matched baseline 全面回撤，因此没有扩 seed 价值。`scale=0.125` 是最小 rescue sweep，但最好点也只剩 `AUC +0.000054`，同时 `ACC/RMSE/Brier/ECE` 仍全反向，因此同样不扩 seed。

## Slice 观察

Slice outputs:

- baseline: `results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_baseline_matched_slices.json`
- candidate: `results/target_concept_interaction_qrepr_rebase/assist_09_seed2024_exp81_exact3_scale0125_slices.json`

关键切片对照（`scale=0.125` vs matched baseline）:

- `concept_count=3`
  - baseline: `AUC 0.705941`, `ACC 0.687708`, `RMSE 0.467791`, `ECE 0.093598`
  - candidate: `AUC 0.703280`, `ACC 0.682171`, `RMSE 0.470428`, `ECE 0.092686`
- `concept_count=1`
  - baseline: `AUC 0.775273`, `ACC 0.735275`, `RMSE 0.422063`, `ECE 0.042695`
  - candidate: `AUC 0.775157`, `ACC 0.733676`, `RMSE 0.422589`, `ECE 0.046372`
- `none_seen`
  - baseline: `AUC 0.805488`, `ACC 0.817853`, `RMSE 0.363121`, `ECE 0.065898`
  - candidate: `AUC 0.806531`, `ACC 0.820265`, `RMSE 0.364256`, `ECE 0.071634`

结论上最关键的是:

- 目标 `concept_count=3` 切片没有留下原假设应有的 clean 排序或误差收益
- `concept_count=1` 默认主线路径被轻微扰动，说明 interaction 没有做到“只在 count=3 带来净增益”
- `none_seen` 虽有极小排序/ACC 正向，但误差与校准仍更差，不足以抵消 overall 回撤

## 结论

- experiment 80 的核心组合在 experiment 78 底座上成立，不代表它能平移到当前 exp81 伪主线。
- 本轮 matched baseline 证明:
  - `scale=0.25` 在当前底座上已经是明确负向
  - `scale=0.125` 只能把结果拉回 near-neutral，但仍不是 clean win
  - 目标 `concept_count=3` 切片也没有保住原本想要的局部收益
- 当前决策:
  - 这条 exp81 rebase 路线拒绝，不扩 additional seeds
  - `exp/target-concept-interaction-qrepr` 保留为历史正向候选，但不再作为当前默认 follow-up
  - 下一步默认回到 experiment 81 promote 后计划: 补 `B49 seed=2024` 交叉复验，或直接继续更大的 representation-level 假设
