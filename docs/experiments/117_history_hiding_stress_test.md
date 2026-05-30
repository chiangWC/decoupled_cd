# Experiment 117: history hiding stress test

## Status

`cross_dataset_history_hiding_completed`

## Verdict

Evaluation-only history hiding supports the robustness story on both the
student-concept holdout split and the default ASSIST09 ordered split. Exp110
full has higher hidden AUC than Exp81 baseline at every hide ratio and a
smaller AUC drop at every hide ratio.

The cleanest table row for the paper story is hide `80%`:

| dataset | model | original AUC | hidden AUC | delta AUC |
|---|---|---:|---:|---:|
| holdout | Exp81 baseline | 0.747137 | 0.709184 | 0.037952 |
| holdout | Exp110 full | 0.753902 | 0.717703 | 0.036198 |
| ordered | Exp81 baseline | 0.772682 | 0.725930 | 0.046751 |
| ordered | Exp110 full | 0.779321 | 0.734223 | 0.045098 |

The component diagnostic is also useful: removing the dual tower makes the
model substantially more fragile under hidden histories on both splits. On the
holdout split at hide `80%`, `w/o dual tower` drops `0.051125` AUC, versus
`0.036198` for Exp110 full.

The seed2027 `single96` capacity-control supplement does not erase that
robustness gap. On the holdout split at hide `80%`, `single96` has hidden AUC
`0.705737` and delta AUC `0.047521`, versus Exp110 full hidden AUC `0.717703`
and delta AUC `0.036198`.

The seed2024 cross-dataset extension is more nuanced. On original ASSIST17 and
NIPS34 splits, Exp110 again has higher hidden AUC and smaller delta AUC at all
hide ratios. On the new student-concept holdout splits, Exp110 keeps a much
higher hidden AUC and better Brier/ECE under hidden histories, but delta AUC is
roughly tied on ASSIST17 and slightly worse at NIPS34 hide `80%`. The
cross-dataset robustness claim should therefore emphasize original-split
robustness plus better hidden-history absolute performance on holdout splits,
not a universal lower-delta statement on every stress split. Junyi was added as
a reduced-capacity supplement; on both its original split and zero-coverage
holdout split, Exp110-near `dual16x32` has higher hidden AUC and smaller
delta AUC at every hide ratio than the single16 baseline.

## Design

This is an evaluation perturbation only; no model is retrained.

For each model checkpoint and each hide ratio, the evaluator randomly hides a
fraction of each student's train-history interactions and rebuilds student-side
history tensors from the masked train frame:

- `student_exercise_mask`
- `student_tkc_mask`
- `student_ukc_mask`
- `response_matrix`
- `student_concept_evidence`

By default, exercise-level evidence remains based on the original train split.
This isolates incomplete student observation history rather than also changing
global exercise statistics.

Each hide ratio uses mask seeds `11`, `13`, and `17`; tables below report means
and population standard deviations over those three masks.

## Artifacts

Remote output directory:

```text
results/history_hiding_exp117/
```

Cross-dataset output directory:

```text
results/cross_dataset_exp116_117/
```

Artifacts:

- `holdout_seed2027_report.json`
- `holdout_seed2027_summary.csv`
- `holdout_seed2027_per_run.csv`
- `ordered_seed2027_report.json`
- `ordered_seed2027_summary.csv`
- `ordered_seed2027_per_run.csv`

Evaluator added in:

```text
scripts/evaluate_history_hiding_stress.py
```

Focused tests:

```text
tests/test_history_hiding_stress.py
```

Remote verification:

```bash
python -m py_compile scripts/evaluate_history_hiding_stress.py
python -m unittest tests.test_history_hiding_stress
```

Passed remotely: 4 tests OK.

The cross-dataset extension also verified:

```bash
python -m py_compile scripts/evaluate_coverage_slice.py scripts/evaluate_history_hiding_stress.py
python -m unittest tests.test_coverage_slice tests.test_history_hiding_stress
```

Passed remotely: 9 tests OK.

## Holdout Split

Dataset: `/tmp/assist09_holdout_seed2024`.

| model | hide | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---:|---:|---:|---:|---:|---:|
| Exp81 baseline | 0.2 | 0.747137 | 0.740785 (0.001107) | 0.006351 | 0.194419 | 0.083526 |
| Exp81 baseline | 0.4 | 0.747137 | 0.734350 (0.001908) | 0.012786 | 0.197684 | 0.088229 |
| Exp81 baseline | 0.6 | 0.747137 | 0.725098 (0.002141) | 0.022039 | 0.202852 | 0.097385 |
| Exp81 baseline | 0.8 | 0.747137 | 0.709184 (0.001967) | 0.037952 | 0.211076 | 0.108814 |
| Exp110 full | 0.2 | 0.753902 | 0.749267 (0.000724) | 0.004635 | 0.186443 | 0.049112 |
| Exp110 full | 0.4 | 0.753902 | 0.742933 (0.001231) | 0.010968 | 0.188889 | 0.052097 |
| Exp110 full | 0.6 | 0.753902 | 0.733131 (0.001815) | 0.020770 | 0.192642 | 0.055568 |
| Exp110 full | 0.8 | 0.753902 | 0.717703 (0.000641) | 0.036198 | 0.198436 | 0.061857 |
| w/o cognitive alignment | 0.2 | 0.750969 | 0.745764 (0.001149) | 0.005204 | 0.189646 | 0.062325 |
| w/o cognitive alignment | 0.4 | 0.750969 | 0.738637 (0.000476) | 0.012331 | 0.192591 | 0.066154 |
| w/o cognitive alignment | 0.6 | 0.750969 | 0.729898 (0.000669) | 0.021071 | 0.196039 | 0.069901 |
| w/o cognitive alignment | 0.8 | 0.750969 | 0.712135 (0.000918) | 0.038834 | 0.202703 | 0.076980 |
| w/o dual tower | 0.2 | 0.754154 | 0.745677 (0.000893) | 0.008477 | 0.187730 | 0.049190 |
| w/o dual tower | 0.4 | 0.754154 | 0.736655 (0.001707) | 0.017500 | 0.192119 | 0.057393 |
| w/o dual tower | 0.6 | 0.754154 | 0.724535 (0.002485) | 0.029620 | 0.197890 | 0.065137 |
| w/o dual tower | 0.8 | 0.754154 | 0.703029 (0.001988) | 0.051125 | 0.208222 | 0.078672 |

## Ordered Split

Dataset: `data/assist_09_ordered`.

| model | hide | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---:|---:|---:|---:|---:|---:|
| Exp81 baseline | 0.2 | 0.772682 | 0.765415 (0.000658) | 0.007267 | 0.183341 | 0.055071 |
| Exp81 baseline | 0.4 | 0.772682 | 0.757229 (0.000364) | 0.015452 | 0.186910 | 0.060098 |
| Exp81 baseline | 0.6 | 0.772682 | 0.745550 (0.000695) | 0.027132 | 0.192518 | 0.070218 |
| Exp81 baseline | 0.8 | 0.772682 | 0.725930 (0.001683) | 0.046751 | 0.202816 | 0.087641 |
| Exp110 full | 0.2 | 0.779321 | 0.773188 (0.000642) | 0.006133 | 0.177848 | 0.030399 |
| Exp110 full | 0.4 | 0.779321 | 0.764902 (0.000581) | 0.014419 | 0.180925 | 0.033762 |
| Exp110 full | 0.6 | 0.779321 | 0.753838 (0.001000) | 0.025483 | 0.185220 | 0.039200 |
| Exp110 full | 0.8 | 0.779321 | 0.734223 (0.002755) | 0.045098 | 0.193037 | 0.048387 |
| w/o cognitive alignment | 0.2 | 0.774025 | 0.767741 (0.000712) | 0.006283 | 0.180124 | 0.036225 |
| w/o cognitive alignment | 0.4 | 0.774025 | 0.759656 (0.001281) | 0.014369 | 0.183471 | 0.041492 |
| w/o cognitive alignment | 0.6 | 0.774025 | 0.748324 (0.001696) | 0.025701 | 0.188173 | 0.049550 |
| w/o cognitive alignment | 0.8 | 0.774025 | 0.729318 (0.002590) | 0.044707 | 0.196367 | 0.062049 |
| w/o dual tower | 0.2 | 0.775119 | 0.766149 (0.000877) | 0.008970 | 0.181301 | 0.040804 |
| w/o dual tower | 0.4 | 0.775119 | 0.755589 (0.000736) | 0.019530 | 0.186134 | 0.049604 |
| w/o dual tower | 0.6 | 0.775119 | 0.741053 (0.000618) | 0.034066 | 0.192899 | 0.060644 |
| w/o dual tower | 0.8 | 0.775119 | 0.718406 (0.002391) | 0.056712 | 0.203722 | 0.075459 |

## Cross-Dataset Seed2024 Extension

Artifacts:

- `assist_17_original_seed2024_history_hiding_report.json`
- `assist_17_original_seed2024_history_hiding_report_summary.csv`
- `assist_17_original_seed2024_history_hiding_report_per_run.csv`
- `nips34_original_seed2024_history_hiding_report.json`
- `nips34_original_seed2024_history_hiding_report_summary.csv`
- `nips34_original_seed2024_history_hiding_report_per_run.csv`
- `assist_17_holdout_seed2024_history_hiding_report.json`
- `assist_17_holdout_seed2024_history_hiding_report_summary.csv`
- `assist_17_holdout_seed2024_history_hiding_report_per_run.csv`
- `nips34_holdout_seed2024_history_hiding_report.json`
- `nips34_holdout_seed2024_history_hiding_report_summary.csv`
- `nips34_holdout_seed2024_history_hiding_report_per_run.csv`
- `junyi_original_seed2024_history_hiding_report.json`
- `junyi_holdout_seed2024_history_hiding_report.json`

Overall AUCs before hiding:

| dataset | split | Exp81 AUC | Exp110 AUC | delta |
|---|---|---:|---:|---:|
| ASSIST17 | original | 0.777001 | 0.779926 | +0.002925 |
| NIPS34 | original | 0.783783 | 0.784708 | +0.000924 |
| ASSIST17 | holdout | 0.730791 | 0.755751 | +0.024960 |
| NIPS34 | holdout | 0.755259 | 0.772583 | +0.017324 |

Hide `80%` summary:

| dataset | split | model | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---|---|---:|---:|---:|---:|---:|
| ASSIST17 | original | Exp81 baseline | 0.777001 | 0.721476 | 0.055525 | 0.213374 | 0.050534 |
| ASSIST17 | original | Exp110 full | 0.779926 | 0.732120 | 0.047807 | 0.207714 | 0.031026 |
| NIPS34 | original | Exp81 baseline | 0.783783 | 0.750299 | 0.033484 | 0.205844 | 0.062779 |
| NIPS34 | original | Exp110 full | 0.784708 | 0.754391 | 0.030317 | 0.202219 | 0.045086 |
| ASSIST17 | holdout | Exp81 baseline | 0.730791 | 0.690722 | 0.040070 | 0.225770 | 0.066205 |
| ASSIST17 | holdout | Exp110 full | 0.755751 | 0.715649 | 0.040103 | 0.215247 | 0.058250 |
| NIPS34 | holdout | Exp81 baseline | 0.755259 | 0.722845 | 0.032415 | 0.213995 | 0.046035 |
| NIPS34 | holdout | Exp110 full | 0.772583 | 0.739493 | 0.033091 | 0.205920 | 0.021949 |

Cross-dataset read:

- On original ASSIST17 and NIPS34 splits, Exp110 has smaller delta AUC at every
  hide ratio and higher hidden AUC, matching the ASSIST09 pattern.
- On holdout splits, Exp110's absolute hidden AUC remains much higher at every
  hide ratio. The delta-AUC robustness criterion is mixed: ASSIST17 is tied
  within about `0.00003` at hide `80%`, and NIPS34 is slightly worse by about
  `0.00068` at hide `80%`.
- Holdout calibration/error still favors Exp110 under hidden histories:
  NIPS34 hide `80%` hidden ECE improves from `0.046035` to `0.021949`, and
  hidden Brier improves from `0.213995` to `0.205920`.

## Junyi Reduced-Capacity Supplement

Junyi does not run the original Exp110 `dual64x80` configuration on the
available 24GB GPU. These rows compare an Exp81-style `single16` baseline with
the feasible Exp110-near `dual16x32` student-recompute family. They are
therefore a Junyi robustness supplement, not a direct ASSIST17/NIPS34-equivalent
Exp110 transfer.

Training summaries:

- `results/cross_dataset_exp116_117/junyi_original_seed2024_exp81_baseline_student_recompute2048.json`
- `results/junyi_memory_trials/junyi_seed2024_student_recompute2048_dual16x32_300ep.json`
- `results/cross_dataset_exp116_117/junyi_holdout_seed2024_exp81_baseline_student_recompute2048.json`
- `results/cross_dataset_exp116_117/junyi_holdout_seed2024_exp110_dual16x32_student_recompute2048.json`

Overall AUCs before hiding:

| dataset | split | baseline AUC | Exp110-near AUC | delta |
|---|---|---:|---:|---:|
| Junyi reduced-capacity | original | 0.819085 | 0.824503 | +0.005418 |
| Junyi reduced-capacity | zero-coverage holdout | 0.815124 | 0.819972 | +0.004848 |

Original split:

| model | hide | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---:|---:|---:|---:|---:|---:|
| Exp81-style baseline (single16) | 0.2 | 0.819085 | 0.815059 | 0.004026 | 0.165939 | 0.031871 |
| Exp81-style baseline (single16) | 0.4 | 0.819085 | 0.807594 | 0.011491 | 0.168860 | 0.032101 |
| Exp81-style baseline (single16) | 0.6 | 0.819085 | 0.793961 | 0.025125 | 0.174556 | 0.033365 |
| Exp81-style baseline (single16) | 0.8 | 0.819085 | 0.764977 | 0.054108 | 0.187405 | 0.048699 |
| Exp110-near dual16x32 | 0.2 | 0.824503 | 0.821455 | 0.003048 | 0.162226 | 0.013810 |
| Exp110-near dual16x32 | 0.4 | 0.824503 | 0.815909 | 0.008594 | 0.164565 | 0.010817 |
| Exp110-near dual16x32 | 0.6 | 0.824503 | 0.806271 | 0.018233 | 0.168693 | 0.011837 |
| Exp110-near dual16x32 | 0.8 | 0.824503 | 0.782792 | 0.041711 | 0.178663 | 0.023877 |

Zero-coverage holdout split:

| model | hide | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---:|---:|---:|---:|---:|---:|
| Exp81-style baseline (single16) | 0.2 | 0.815124 | 0.810419 | 0.004705 | 0.163775 | 0.021022 |
| Exp81-style baseline (single16) | 0.4 | 0.815124 | 0.802448 | 0.012676 | 0.166885 | 0.019981 |
| Exp81-style baseline (single16) | 0.6 | 0.815124 | 0.788530 | 0.026594 | 0.172851 | 0.021691 |
| Exp81-style baseline (single16) | 0.8 | 0.815124 | 0.757700 | 0.057425 | 0.187570 | 0.055921 |
| Exp110-near dual16x32 | 0.2 | 0.819972 | 0.816698 | 0.003274 | 0.161138 | 0.016856 |
| Exp110-near dual16x32 | 0.4 | 0.819972 | 0.811047 | 0.008925 | 0.163455 | 0.014771 |
| Exp110-near dual16x32 | 0.6 | 0.819972 | 0.800934 | 0.019038 | 0.167750 | 0.014197 |
| Exp110-near dual16x32 | 0.8 | 0.819972 | 0.776974 | 0.042998 | 0.177918 | 0.029197 |

Junyi is the cleanest cross-dataset stress-test supplement for the delta-AUC
robustness claim: the Exp110-near reduced-capacity model has smaller AUC drop
at every hide ratio on both the original and zero-coverage holdout split.

## Seed2027 Capacity Control

This is the history-hiding half of the seed2027 capacity-control supplement.
The `single96` control keeps the Exp110 training protocol and cognitive
alignment but removes the dual tower and branch BCE.

Remote artifacts:

- `results/paper_robustness_followup/seed2027_capacity_history_hiding_report.json`

Holdout hide `80%` summary:

| model | original AUC | hidden AUC | delta AUC | hidden Brier | hidden ECE |
|---|---:|---:|---:|---:|---:|
| Exp110 full | 0.753902 | 0.717703 | 0.036198 | 0.198436 | 0.061857 |
| w/o dual tower | 0.754154 | 0.703029 | 0.051125 | 0.208222 | 0.078672 |
| single96 capacity | 0.753259 | 0.705737 | 0.047521 | 0.207829 | 0.088843 |

The result argues against a simple capacity explanation. `single96` nearly
matches overall AUC, but under hide `80%` it is `-0.011966` hidden AUC behind
Exp110 full and loses `+0.011323` more AUC. Its hidden ECE is also worse than
both Exp110 full and the lower-dimensional w/o-dual baseline.

## Interpretation

- Exp110 full is consistently more robust than Exp81 under incomplete
  observation histories: lower delta AUC and higher hidden AUC at every hide
  ratio on both splits.
- The dual tower is important for robustness. `w/o dual tower` has a much
  larger AUC drop, especially at hide `60%` and `80%`.
- Cognitive alignment improves hidden AUC and calibration relative to removing
  it, although its delta-AUC curve is close to Exp110 full on the default
  ordered split. The robustness claim should therefore emphasize the full
  model first, then use the ablations as component diagnostics.
- The seed2027 `single96` capacity control supports the dual-tower robustness
  story: larger single-tower capacity does not recover hide80 hidden AUC,
  delta AUC, or ECE on the holdout split.
- Because this is evaluation-only seed2027 evidence, it is suitable as a small
  paper diagnostic. If the paper needs a primary robustness table, expand
  Exp81 and Exp110 full to multiple training seeds; keep ablations as seed2027
  diagnostics unless component robustness becomes a central claim.
- Cross-dataset evidence supports the stress-test diagnostic, but the wording
  should be precise: Exp110 is consistently better under hidden histories in
  absolute AUC/error terms, while "smaller AUC drop" is clean on original
  splits, mixed on ASSIST17/NIPS34 holdout splits, and clean again in the
  Junyi reduced-capacity supplement.
