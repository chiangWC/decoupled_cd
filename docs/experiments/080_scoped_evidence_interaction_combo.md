# Experiment 80: scoped evidence plus exact-3 target interaction

- 分支: `exp/target-concept-interaction-qrepr`
- base: `exp/trellis-trial` pseudo-mainline `4e27684`
- 代码状态: candidate branch，已完成 matched multi-seed 验证，暂未 promote 到 `exp/trellis-trial`
- 对照: 实验 78 伪主线默认 `concept_evidence_prior + concept_evidence_readout(min_count=1, max_count=0, seen_ratio=1.0, max_logit=0.5)`

## 动机

实验 79 已经说明 `single-only` readout 是稳定但偏弱的正向信号；本轮又发现 `target_concept_interaction_qrepr` 在只作用于 `exact-3` 时会给出小幅正向。二者的语义正好互补:

- `concept_count=1` 交给可解释 student-concept evidence readout
- `concept_count=3` 交给 target-local concept interaction q representation
- `concept_count=2` / `4+` 保持当前主链，不强行替换

这个组合仍只使用 `Q / concept embedding / TKC / UKC / student-concept history evidence`，不引入 student-exercise ID residual，不是 CF。

## 实现

- 沿用实验 79 的 `concept_evidence_readout_max_count`:
  - `0` 表示不设上限
  - `1` 表示 readout residual 只在 `concept_count=1` 上触发
- 新增 `target_concept_interaction_qrepr` 的两个控制量:
  - `target_concept_interaction_max_scale`
  - `target_concept_interaction_max_count`
- 目标交互支线最终最佳触发是:
  - `--target-concept-interaction-min-count 3`
  - `--target-concept-interaction-max-count 3`
  - `--target-concept-interaction-max-scale 0.25`
- summary loader、train/evaluate CLI、单测都已同步支持新字段。

## 工程验证

- 本地:
  - `python3 -m py_compile models/decoupled_cdm.py scripts/train.py scripts/evaluate.py scripts/analyze_prediction_slices.py tests/test_decoupled_cdm.py`
  - `git diff --check`
- 远端单测:
  - `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - 结果: 通过，`37 tests`

## 单 seed 扫描

Matched baseline 是实验 78 伪主线 seed=2024。

| config | result path | test_auc | delta AUC | test_acc | delta ACC | test_rmse | delta RMSE | test_brier | delta Brier | test_ece | delta ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp78 pseudo-mainline | `results/concept_evidence_prior_next/assist_09_seed2024_prior_default_plus_readout_min1_max05_300ep.json` | 0.772562 | 0.000000 | 0.730823 | 0.000000 | 0.424811 | 0.000000 | 0.180464 | 0.000000 | 0.052361 | 0.000000 | reference |
| target interaction `3+`, scale `0.25` | `results/target_concept_interaction_qrepr/assist_09_seed2024_min3_scale025_300ep.json` | 0.772525 | -0.000037 | 0.731413 | +0.000590 | 0.424647 | -0.000164 | 0.180325 | -0.000139 | 0.051368 | -0.000993 | near-tie |
| target interaction `exact-3`, scale `0.125` | `results/target_concept_interaction_qrepr/assist_09_seed2024_exact3_scale0125_300ep.json` | 0.772618 | +0.000056 | 0.731051 | +0.000228 | 0.424710 | -0.000101 | 0.180378 | -0.000086 | 0.052413 | +0.000052 | weak positive |
| target interaction `exact-3`, scale `0.25` | `results/target_concept_interaction_qrepr/assist_09_seed2024_exact3_scale025_300ep.json` | 0.772985 | +0.000423 | 0.732060 | +0.001237 | 0.424533 | -0.000278 | 0.180228 | -0.000236 | 0.052485 | +0.000124 | best standalone interaction |
| target interaction `exact-3`, scale `0.5` | `results/target_concept_interaction_qrepr/assist_09_seed2024_exact3_scale05_300ep.json` | 0.772155 | -0.000407 | 0.730613 | -0.000209 | 0.424582 | -0.000229 | 0.180270 | -0.000195 | 0.050102 | -0.002259 | over-scaled |
| target interaction `2-3`, scale `0.25` | `results/target_concept_interaction_qrepr/assist_09_seed2024_count23_scale025_300ep.json` | 0.772066 | -0.000496 | 0.730290 | -0.000533 | 0.425295 | +0.000484 | 0.180876 | +0.000411 | 0.053559 | +0.001198 | widening to count2 hurts |
| no-expert control | `results/expert_isolation/assist_09_seed2024_no_expert_300ep.json` | 0.772283 | -0.000279 | 0.733563 | +0.002740 | 0.425336 | +0.000525 | 0.180911 | +0.000446 | 0.055943 | +0.003582 | expert removal alone not clean |
| no-expert + exact-3 scale `0.25` | `results/expert_isolation/assist_09_seed2024_no_expert_exact3_scale025_300ep.json` | 0.771915 | -0.000647 | 0.732878 | +0.002055 | 0.425203 | +0.000392 | 0.180797 | +0.000333 | 0.053553 | +0.001192 | expert suppression not the main driver |
| `single-only readout + exact-3 interaction` | `results/target_concept_interaction_qrepr/assist_09_seed2024_single_readout_plus_exact3_scale025_300ep.json` | 0.773258 | +0.000696 | 0.732707 | +0.001884 | 0.424133 | -0.000678 | 0.179889 | -0.000575 | 0.051236 | -0.001125 | best single-seed combo |

## Multi-seed 验证

Matched baseline 是实验 78 伪主线；candidate 是:

```bash
bash scripts/run_assist09_baseline.sh \
  --concept-evidence-readout-max-count 1 \
  --target-concept-interaction-qrepr-adapter \
  --target-concept-interaction-min-count 3 \
  --target-concept-interaction-max-count 3 \
  --target-concept-interaction-max-scale 0.25
```

| seed | baseline AUC | candidate AUC | delta AUC | baseline ACC | candidate ACC | delta ACC | baseline RMSE | candidate RMSE | delta RMSE | baseline ECE | candidate ECE | delta ECE | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 | 0.772562 | 0.773258 | +0.000696 | 0.730823 | 0.732707 | +0.001884 | 0.424811 | 0.424133 | -0.000678 | 0.052361 | 0.051236 | -0.001125 | normal |
| 2025 | 0.769170 | 0.772174 | +0.003004 | 0.734476 | 0.735390 | +0.000913 | 0.425773 | 0.424379 | -0.001394 | 0.054576 | 0.053549 | -0.001028 | strong normal |
| 2026 | 0.502804 | 0.502923 | +0.000120 | 0.511446 | 0.517232 | +0.005785 | 0.519218 | 0.517176 | -0.002042 | 0.173121 | 0.168050 | -0.005071 | both runs degenerate |
| 2027 | 0.771996 | 0.773480 | +0.001484 | 0.733068 | 0.730689 | -0.002379 | 0.424792 | 0.425004 | +0.000212 | 0.053035 | 0.054619 | +0.001584 | ranking win, calibration tradeoff |

All seeds `2024/2025/2026/2027`:

- mean `AUC +0.001326`
- mean `ACC +0.001551`
- mean `RMSE -0.000975`
- mean `Brier -0.000924`
- mean `ECE -0.001410`

Normal-learning seeds `2024/2025/2027`:

- mean `AUC +0.001728`
- mean `ACC +0.000140`
- mean `RMSE -0.000620`
- mean `Brier -0.000527`
- mean `ECE -0.000190`

## Slice 对照

Seed=2024 相对实验 78:

- `concept_count=1`: `AUC +0.000320`, `ACC +0.001896`, `RMSE -0.000484`, `ECE -0.000087`
- `concept_count=2`: `AUC +0.001889`, `ACC +0.002401`, `RMSE -0.001990`, `ECE -0.005762`
- `concept_count=3`: `AUC +0.000695`, 但 `ACC -0.004430`, `RMSE/Brier` 小幅回撤，`ECE -0.005177`
- `concept_count=4+`: `AUC +0.002261`, `ACC +0.005510`, `RMSE -0.000208`, `ECE +0.000990`
- `none_seen`: `AUC +0.001398`, `ACC +0.003619`, `RMSE -0.002417`, `ECE -0.003377`
- `partial_seen`: `AUC +0.003804`, `ACC +0.003030`, `RMSE -0.007201`, `ECE -0.019002`
- `all_seen`: `AUC +0.000606`, `ACC +0.001820`, `RMSE -0.000587`, `ECE -0.000500`

Slice result paths:

- `results/target_concept_interaction_qrepr/exp78_baseline_seed2024_slices.json`
- `results/target_concept_interaction_qrepr/assist_09_seed2024_single_readout_plus_exact3_scale025_slices.json`

## 结论

- 这是当前自由探索阶段第一个完成 multi-seed 复核后仍保持 clean 正向的结构候选。
- 真正成立的不是单个新模块，而是“按概念数分治”的组合:
  - `concept_count=1` 用 deterministic/local evidence readout
  - `concept_count=3` 用 bounded target-concept interaction q representation
  - 其它样本保留当前主链
- strongest evidence 来自:
  - `seed2025` 的强 AUC 提升
  - normal-learning seeds `2024/2025/2027` 的 mean `AUC +0.001728`
  - all-seed mean 上 `AUC/ACC/RMSE/Brier/ECE` 全同向
- 风险:
  - `seed2027` 的 `ACC/RMSE/ECE` 出现轻微反向
  - `seed2026` baseline/candidate 仍同步退化，不提供真正的语义区分
- 当前决策:
  - 保留为高优先级候选，不立即 promote 到 `exp/trellis-trial`
  - 若下一步继续，优先围绕这条组合做 very small confirmation sweep 或 matched promotion check，而不是回到无约束的 residual 扩线
