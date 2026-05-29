# Experiment 116: student-concept holdout coverage slice

## Status

`cross_dataset_holdout_stress_slice_completed`

## Verdict

The student-concept holdout split supports the coverage-bias story that the
ordinary random ASSIST09 split could not support. On this stress split,
low-coverage test interactions are numerous and materially harder than fully
covered interactions across four training seeds:

```text
Exp81 mean low AUC  = 0.723470
Exp81 mean full AUC = 0.768472
coverage gap        = +0.045001
```

Experiment 110 improves both overall AUC and low-coverage AUC:

```text
Exp110 full overall AUC delta vs Exp81 = +0.010490
Exp110 full low AUC delta vs Exp81     = +0.010450
Exp110 full full AUC delta vs Exp81    = +0.008317
```

This is a better evidence shape for a paper claim than experiment 115. The
strong claim should be tied to this split as a `student-concept holdout /
UKC-heavy stress split`, not to the default random interaction split.

The same seed2024 holdout protocol was extended to ASSIST17 and NIPS34. Both
datasets reproduce the core pattern: low-coverage interactions are harder, and
Exp110 improves low-coverage AUC while reducing the coverage gap.

## Split

Generated on the remote host with:

```bash
python scripts/split_student_concept_holdout.py \
  --source-dir data/assist_09_ordered \
  --output-dir /tmp/assist09_holdout_seed2024 \
  --seed 2024 \
  --overwrite
```

The split was moved outside the repository worktree to keep `remote_exec.sh`
usable.

Default split parameters:

- `holdout_student_frac = 0.5`
- `target_eval_ratio = 0.30`
- `min_eval_ratio = 0.15`
- `max_eval_ratio = 0.32`
- `test_ratio_within_eval = 2/3`
- `random_valid_ratio = 0.10`
- `random_test_ratio = 0.20`
- `min_student_interactions = 30`
- `min_student_concepts = 5`
- `min_train_interactions = 10`

Split summary:

| split | rows | ratio |
|---|---:|---:|
| train | 186714 | 0.698197 |
| valid | 26957 | 0.100803 |
| test | 53752 | 0.201000 |

Student assignment summary:

| mode | students |
|---|---:|
| strict holdout | 732 |
| fallback random | 2 |
| random | 1759 |

Test overlap from the split summary:

| overlap | count | ratio |
|---|---:|---:|
| none_seen | 21021 | 0.391074 |
| partial_seen | 4086 | 0.076016 |
| all_seen | 28645 | 0.532910 |

## Runs

Primary multi-seed runs retrained Exp81 baseline and Exp110 full on the
holdout split for seeds `2024`, `2025`, `2026`, and `2027`. Seed `2027` also
includes the three Exp110 component ablations as a staged diagnostic.

Outputs are on the remote host:

```text
results/coverage_holdout_exp116/
```

Primary multi-seed run summaries:

- `seed2024_exp81_baseline.json`
- `seed2024_exp110_full.json`
- `seed2025_exp81_baseline.json`
- `seed2025_exp110_full.json`
- `seed2026_exp81_baseline.json`
- `seed2026_exp110_full.json`
- `seed2027_exp81_baseline.json`
- `seed2027_exp110_full.json`

Primary multi-seed coverage artifacts:

- `multiseed_exp81_exp110_coverage_report.json`
- `multiseed_exp81_exp110_summary.csv`
- `multiseed_exp81_exp110_per_run.csv`
- `multiseed_exp81_exp110_bucket_metrics.csv`

Seed2027 ablation run summaries:

- `seed2027_exp81_baseline.json`
- `seed2027_exp110_full.json`
- `seed2027_exp110_no_dual_tower.json`
- `seed2027_exp110_no_cog_align.json`
- `seed2027_exp110_no_branch_bce.json`

Coverage-slice artifacts:

- `seed2027_coverage_report.json`
- `seed2027_coverage_summary.csv`
- `seed2027_bucket_metrics.csv`

## Multi-Seed Primary Result

Definitions:

- `low_coverage`: `target_coverage < 0.5`
- `partial_coverage`: `0 < target_coverage < 1`
- `full_coverage`: `target_coverage = 1`
- `coverage_gap_auc`: `full_auc - low_auc`

Four-seed means, population stdev in parentheses:

| model | overall AUC | low AUC | full AUC | coverage gap | low ECE | low Brier |
|---|---:|---:|---:|---:|---:|---:|
| Exp81 baseline | 0.744564 (0.004305) | 0.723470 (0.005541) | 0.768472 (0.005157) | 0.045001 (0.001315) | 0.099601 (0.007420) | 0.196805 (0.001658) |
| Exp110 full | 0.755054 (0.003375) | 0.733920 (0.001638) | 0.776789 (0.000520) | 0.042869 (0.001353) | 0.081303 (0.034149) | 0.190867 (0.006540) |

Primary deltas, Exp110 full minus Exp81 baseline:

| metric | delta |
|---|---:|
| overall AUC | +0.010490 |
| low AUC | +0.010450 |
| full AUC | +0.008317 |
| coverage gap | -0.002133 |
| low ECE | -0.018299 |
| low Brier | -0.005937 |

Per-seed primary rows:

| seed | model | overall AUC | low AUC | full AUC | coverage gap | low ECE | low Brier |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2024 | Exp81 baseline | 0.737375 | 0.713960 | 0.759657 | 0.045697 | 0.092418 | 0.197037 |
| 2024 | Exp110 full | 0.750079 | 0.732680 | 0.776036 | 0.043356 | 0.131363 | 0.201585 |
| 2025 | Exp81 baseline | 0.748482 | 0.727414 | 0.770169 | 0.042755 | 0.092255 | 0.194055 |
| 2025 | Exp110 full | 0.757517 | 0.736739 | 0.777318 | 0.040579 | 0.079949 | 0.188940 |
| 2026 | Exp81 baseline | 0.745260 | 0.725446 | 0.771526 | 0.046080 | 0.104732 | 0.197740 |
| 2026 | Exp110 full | 0.758717 | 0.733127 | 0.777224 | 0.044097 | 0.034920 | 0.183841 |
| 2027 | Exp81 baseline | 0.747137 | 0.727060 | 0.772533 | 0.045473 | 0.109001 | 0.198387 |
| 2027 | Exp110 full | 0.753902 | 0.733135 | 0.776578 | 0.043443 | 0.078979 | 0.189102 |

The key interpretation is stable across seeds: the stress split makes low
coverage materially harder than full coverage, and Exp110 improves the
low-coverage slice by roughly the same magnitude as its overall gain. Exp110
also narrows the mean coverage gap and improves low-coverage Brier/ECE.

## Cross-Dataset Seed2024 Extension

Remote source/split roots:

```text
/home/xph/jwc/research/local_data/cross_dataset_exp116_117/assist_17_holdout_seed2024
/home/xph/jwc/research/local_data/cross_dataset_exp116_117/nips34_holdout_seed2024
```

Remote result directory:

```text
results/cross_dataset_exp116_117/
```

Split summaries:

| dataset | train rows | valid rows | test rows | strict-holdout students | test none_seen | test full_seen |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST17 | 271713 | 39527 | 79041 | 840 | 43213 | 35828 |
| NIPS34 | 962172 | 140160 | 280395 | 2444 | 143041 | 134849 |

Coverage-slice artifacts:

- `assist_17_holdout_seed2024_coverage_report.json`
- `assist_17_holdout_seed2024_coverage_report_summary.csv`
- `assist_17_holdout_seed2024_coverage_report_slices.csv`
- `nips34_holdout_seed2024_coverage_report.json`
- `nips34_holdout_seed2024_coverage_report_summary.csv`
- `nips34_holdout_seed2024_coverage_report_slices.csv`

Seed2024 cross-dataset summary:

| dataset | model | overall AUC | low count | low AUC | full count | full AUC | coverage gap | low Brier | low ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSIST17 | Exp81 baseline | 0.730791 | 33212 | 0.684796 | 30413 | 0.784395 | 0.099599 | 0.237931 | 0.105130 |
| ASSIST17 | Exp110 full | 0.755751 | 33212 | 0.743160 | 30413 | 0.780608 | 0.037447 | 0.213228 | 0.097672 |
| NIPS34 | Exp81 baseline | 0.755259 | 143041 | 0.746311 | 134849 | 0.773741 | 0.027429 | 0.204529 | 0.035942 |
| NIPS34 | Exp110 full | 0.772583 | 143041 | 0.766900 | 134849 | 0.780103 | 0.013203 | 0.195253 | 0.025829 |

Cross-dataset deltas, Exp110 full minus Exp81 baseline:

| dataset | overall AUC delta | low AUC delta | full AUC delta | coverage gap delta | low Brier delta | low ECE delta |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST17 | +0.024960 | +0.058364 | -0.003787 | -0.062151 | -0.024704 | -0.007458 |
| NIPS34 | +0.017324 | +0.020589 | +0.006362 | -0.014227 | -0.009276 | -0.010113 |

This is stronger than the ASSIST09 four-seed mean on the coverage-bias
dimension. ASSIST17 in particular shows the desired story clearly: the baseline
coverage gap is almost `0.10` AUC, while Exp110 cuts it to about `0.04` and
raises low-coverage AUC by `+0.058`. NIPS34 shows the same direction with a
smaller but still clean gap reduction.

## Seed2027 Ablation Metrics

| model | overall AUC | ACC | Brier | ECE | best epoch |
|---|---:|---:|---:|---:|---:|
| Exp81 baseline | 0.747137 | 0.722206 | 0.190987 | 0.078813 | 194 |
| Exp110 full | 0.753902 | 0.725499 | 0.184500 | 0.047335 | 134 |
| Exp110 w/o dual tower | 0.754154 | 0.726094 | 0.183736 | 0.042235 | 167 |
| Exp110 w/o cognitive alignment | 0.750969 | 0.724010 | 0.187367 | 0.060094 | 154 |
| Exp110 w/o branch BCE | 0.497356 | 0.528650 | 0.256908 | 0.151764 | 3 |

The no-dual row has slightly higher overall AUC than Exp110 full on this seed,
but this does not carry into the within-slice AUCs below. Overall AUC is not a
weighted average of slice AUCs; cross-slice ranking can change the ordering.

## Seed2027 Coverage Summary

Definitions:

- `low_coverage`: `target_coverage < 0.5`
- `partial_coverage`: `0 < target_coverage < 1`
- `full_coverage`: `target_coverage = 1`
- `coverage_gap_auc`: `full_auc - low_auc`

| model | low count | low AUC | full count | full AUC | coverage gap | low ECE | low Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp81 baseline | 21467 | 0.727060 | 28645 | 0.772533 | 0.045473 | 0.109001 | 0.198387 |
| Exp110 full | 21467 | 0.733135 | 28645 | 0.776578 | 0.043443 | 0.078979 | 0.189102 |
| Exp110 w/o dual tower | 21467 | 0.728613 | 28645 | 0.773960 | 0.045347 | 0.052044 | 0.186720 |
| Exp110 w/o cognitive alignment | 21467 | 0.725731 | 28645 | 0.771353 | 0.045622 | 0.082966 | 0.192572 |
| Exp110 w/o branch BCE | 21467 | 0.496684 | 28645 | 0.497842 | 0.001158 | 0.172641 | 0.255145 |

Component read relative to Exp110 full:

| ablation | low AUC delta | full AUC delta | low ECE delta | low Brier delta |
|---|---:|---:|---:|---:|
| w/o dual tower | -0.004522 | -0.002618 | -0.026935 | -0.002382 |
| w/o cognitive alignment | -0.007404 | -0.005225 | +0.003987 | +0.003470 |
| w/o branch BCE | -0.236451 | -0.278736 | +0.093662 | +0.066043 |

The dual tower and cognitive alignment both help low-coverage AUC on this
staged seed. Branch BCE appears critical for preventing dual-tower collapse on
this stress split.

## Original Buckets

| model | bucket | count | AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| Exp81 baseline | `coverage=0` | 21021 | 0.727547 | 0.197307 | 0.108535 |
| Exp81 baseline | `0<coverage<0.5` | 446 | 0.654571 | 0.249285 | 0.138674 |
| Exp81 baseline | `0.5<=coverage<1` | 3640 | 0.716930 | 0.196437 | 0.085519 |
| Exp81 baseline | `coverage=1` | 28645 | 0.772533 | 0.184748 | 0.057696 |
| Exp110 full | `coverage=0` | 21021 | 0.733391 | 0.188020 | 0.077348 |
| Exp110 full | `0<coverage<0.5` | 446 | 0.690102 | 0.240120 | 0.155862 |
| Exp110 full | `0.5<=coverage<1` | 3640 | 0.737964 | 0.188333 | 0.073065 |
| Exp110 full | `coverage=1` | 28645 | 0.776578 | 0.180563 | 0.034693 |

The fractional low bucket is now usable as a diagnostic (`446` rows instead of
`15` under the random split), but it is still much smaller than `coverage=0`.

## Conclusion

- The student-concept holdout split is the right vehicle for the proposed
  coverage-bias story.
- Across four seeds, it shows the desired pattern: low coverage is much harder
  than full coverage, and Exp110 improves the low-coverage slice over Exp81.
- ASSIST17 and NIPS34 seed2024 extensions reproduce the same direction with
  enough low/full support: Exp110 raises low-coverage AUC and narrows the
  coverage gap on both datasets.
- Seed2027 ablations suggest cognitive alignment and dual tower both help
  low-coverage AUC, while branch BCE is critical for avoiding collapse on this
  stress split.
- If the paper needs component claims on the stress split, expand the three
  ablations beyond seed2027. The main Exp81-vs-Exp110 coverage-bias claim now
  has multi-seed support.
