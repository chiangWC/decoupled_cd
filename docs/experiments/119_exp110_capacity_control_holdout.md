# Experiment 119: exp110 seed2027 capacity-control holdout

## Status

`seed2027_capacity_control_completed`

## Verdict

The seed2027 capacity-control supplement supports the dual-tower story. A
larger single-tower `single96` control nearly matches Exp110 full on overall
ASSIST09 student-concept holdout AUC, but it does not recover low-coverage AUC
or history-hiding robustness.

Key holdout coverage result:

```text
Exp110 full low AUC       = 0.733135
single96 capacity low AUC = 0.722636
delta                     = -0.010499
```

Key hide80 result:

```text
Exp110 full hidden AUC       = 0.717703
single96 capacity hidden AUC = 0.705737
Exp110 full delta AUC        = 0.036198
single96 capacity delta AUC  = 0.047521
```

This is enough for the seed2027-first gate. There is no need to run
`single128` or multi-seed capacity controls unless the paper later requires a
stronger fairness/capacity table.

## Design

The control keeps the Exp110 evidence/readout and cognitive-alignment protocol,
removes only the dual-tower ensemble and branch BCE, and increases the single
tower from `concept_dim=64` to `concept_dim=96`.

Dataset:

```text
/tmp/assist09_holdout_seed2024
```

Remote artifacts:

- `results/paper_robustness_followup/seed2027_exp110_single96_holdout.json`
- `results/paper_robustness_followup/seed2027_capacity_coverage_report.json`
- `results/paper_robustness_followup/seed2027_capacity_coverage_summary.csv`
- `results/paper_robustness_followup/seed2027_capacity_coverage_slices.csv`
- `results/paper_robustness_followup/seed2027_capacity_history_hiding_report.json`

Training summary:

| model | concept dim | dual tower | branch BCE | overall AUC | ACC | Brier | ECE | best epoch | peak CUDA GB |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single96 capacity | 96 | no | 0.00 | 0.753259 | 0.725443 | 0.185236 | 0.053034 | 133 | 4.548575 |

## Coverage Control

| model | overall AUC | low AUC | full AUC | coverage gap | low Brier | low ECE |
|---|---:|---:|---:|---:|---:|---:|
| Exp110 full | 0.753902 | 0.733135 | 0.776578 | 0.043443 | 0.189102 | 0.078979 |
| w/o dual tower | 0.754154 | 0.728613 | 0.773960 | 0.045347 | 0.186720 | 0.052044 |
| single96 capacity | 0.753259 | 0.722636 | 0.775477 | 0.052841 | 0.189765 | 0.062275 |

`single96` is close on overall AUC but worse than both Exp110 full and the
lower-dimensional w/o-dual baseline on low-coverage AUC. The wider coverage
gap makes the result useful specifically as a capacity-control answer: more
single-tower width does not reproduce the dual-branch stress-split behavior.

## History-Hiding Control

Holdout hide `80%`, mask seeds `11/13/17`:

| model | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---:|---:|---:|---:|---:|
| Exp110 full | 0.753902 | 0.717703 | 0.036198 | 0.198436 | 0.061857 |
| w/o dual tower | 0.754154 | 0.703029 | 0.051125 | 0.208222 | 0.078672 |
| single96 capacity | 0.753259 | 0.705737 | 0.047521 | 0.207829 | 0.088843 |

`single96` remains much weaker than Exp110 full under hidden histories:
hidden AUC is lower by `0.011966`, delta AUC is worse by `0.011323`, and
hidden ECE is worse by `0.026986`.

## Expansion Decision

Do not expand immediately to `single128` or multi-seed capacity controls.

Reasons:

- The seed2027 result is already favorable for the paper story.
- The control nearly matches overall AUC, so the negative stress/robustness
  result is more informative than a weak underfit baseline.
- Exp110 full keeps better low-coverage AUC, hide80 hidden AUC, hide80 delta
  AUC, and hide80 ECE.

Expand only if reviewers or paper framing require a dedicated capacity/fairness
table rather than a compact diagnostic.
