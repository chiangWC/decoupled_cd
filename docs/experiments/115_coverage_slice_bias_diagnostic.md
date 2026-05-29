# Experiment 115: coverage slice bias diagnostic

## Status

`coverage_slice_completed_mixed_story`

## Verdict

The coverage-slice rerun supports a narrower story than the initial hypothesis:
experiment 110 improves the low-coverage aggregate more than the full-coverage
aggregate, but ASSIST09 ordered test does not show lower AUC on low coverage.
In fact, `target_coverage < 0.5` has higher AUC than `target_coverage = 1` for
all checked models.

The reason is distributional. The low-coverage aggregate is almost entirely
`coverage = 0` (`1658` rows) plus only `15` rows in `0 < coverage < 0.5`.
The `coverage = 0` rows have high label rate (`0.802`) and high AUC, while the
true fractional low-coverage bucket is too small to report as a stable slice.

The defensible paper claim from this diagnostic is:

```text
Experiment 110's gain over experiment 81 is larger on the train-history
low-coverage aggregate than on fully covered items, but the exact
"low coverage is harder by AUC" claim is not supported on ASSIST09 ordered test.
```

## Definition

For each test interaction `(student, exercise)`:

```text
Q_e = exercise concepts
Seen_s = concepts seen by the same student in train history
target_coverage = |Q_e intersect Seen_s| / |Q_e|
```

Report aggregates:

- `low_coverage`: `target_coverage < 0.5`.
- `full_coverage`: `target_coverage == 1`.
- `coverage_gap_auc`: `full_auc - low_auc`.

The four original buckets are also preserved:

- `coverage = 0`.
- `0 < coverage < 0.5`.
- `0.5 <= coverage < 1`.
- `coverage = 1`.

## Artifacts

Remote output directory:

```text
results/coverage_slice_exp115/
```

Files:

- `coverage_slice_report.json`
- `summary.csv`
- `per_run.csv`
- `bucket_metrics.csv`

The analysis used existing checkpoints only; no model was retrained.

## Model Summary

Four-seed means. Exp81 uses the three promotion checkpoints plus the available
seed2027 exp81 baseline checkpoint.

| model | overall AUC | low AUC | full AUC | coverage gap | low ECE | low Brier |
|---|---:|---:|---:|---:|---:|---:|
| Exp81 baseline | 0.770771 | 0.809060 | 0.768719 | -0.040340 | 0.053174 | 0.130412 |
| Exp110 full | 0.778321 | 0.822306 | 0.776289 | -0.046017 | 0.060933 | 0.124164 |
| Exp110 w/o dual tower | 0.774657 | 0.818108 | 0.773534 | -0.044574 | 0.098016 | 0.134954 |
| Exp110 w/o cognitive alignment | 0.774398 | 0.815461 | 0.772430 | -0.043031 | 0.049689 | 0.127193 |
| Exp110 w/o branch BCE | 0.776481 | 0.817545 | 0.774457 | -0.043088 | 0.057575 | 0.124223 |

## Delta Read

Exp110 full versus Exp81 baseline:

| slice | AUC delta | Brier delta | ECE delta |
|---|---:|---:|---:|
| overall | +0.007550 | -0.004723 | -0.019848 |
| low coverage | +0.013246 | -0.006248 | +0.007759 |
| full coverage | +0.007570 | -0.004593 | -0.018707 |

Component removals relative to Exp110 full:

| ablation | low AUC drop | full AUC drop | low Brier delta | low ECE delta |
|---|---:|---:|---:|---:|
| w/o dual tower | -0.004198 | -0.002755 | +0.010790 | +0.037083 |
| w/o cognitive alignment | -0.006845 | -0.003859 | +0.003029 | -0.011244 |
| w/o branch BCE | -0.004761 | -0.001832 | +0.000059 | -0.003358 |

This supports the component story more cleanly than the raw gap story:
removing dual tower, cognitive alignment, or branch BCE hurts low-coverage AUC
more than full-coverage AUC.

## Bucket Diagnostics

Bucket means across seeds:

| model | bucket | count | AUC | ECE | Brier |
|---|---|---:|---:|---:|---:|
| Exp81 baseline | `coverage=0` | 1658 | 0.809664 | 0.052754 | 0.129418 |
| Exp81 baseline | `0<coverage<0.5` | 15 | 0.657407 | 0.299224 | 0.240300 |
| Exp81 baseline | `0.5<=coverage<1` | 315 | 0.768554 | 0.084047 | 0.166785 |
| Exp81 baseline | `coverage=1` | 50561 | 0.768719 | 0.049065 | 0.182194 |
| Exp110 full | `coverage=0` | 1658 | 0.823468 | 0.060986 | 0.123244 |
| Exp110 full | `0<coverage<0.5` | 15 | 0.666667 | 0.308961 | 0.225926 |
| Exp110 full | `0.5<=coverage<1` | 315 | 0.794018 | 0.062997 | 0.149326 |
| Exp110 full | `coverage=1` | 50561 | 0.776289 | 0.030358 | 0.177601 |

The `0 < coverage < 0.5` bucket is too small to use as a headline metric.

## Conclusion

- Do not claim that ordinary overall AUC hides a monotonic low-coverage AUC
  failure on ASSIST09 ordered test. This slice does not show that.
- It is fair to claim that Exp110 improves the low-coverage aggregate more
  strongly than Exp81 and that removing core Exp110 components disproportionately
  hurts low-coverage AUC.
- If the desired story is "low historical coverage is harder," a better follow-up
  is to run the same diagnostic on datasets/splits with materially larger
  fractional low-coverage support, or to define the stress slice as explicit
  UKC-heavy / partially-covered items rather than `coverage = 0`.
