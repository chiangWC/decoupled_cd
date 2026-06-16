# Experiment 123: TKC-only Direct Fusion Ablation

## Purpose

This records the direct TKC/UKC fusion ablation requested for the coverage-bias story.

The full model uses adaptive student-state fusion:

```text
Z_s = w_s z_s^TKC + (1 - w_s) z_s^UKC
```

`TKC-only` keeps the same exp110 training and model surface but replaces the final student-state fusion with:

```text
Z_s = z_s^TKC
```

This is the most direct ablation for the claim that relying only on tested-knowledge state is weaker when target concepts are under-covered in the student's train history.

## Run Scope

- Single seed only.
- Student-concept holdout split only.
- Junyi excluded.
- `Full adaptive` means the existing exp110-style model with adaptive TKC/UKC fusion.
- `TKC-only` changes only `student_fusion_mode=tkc_only`; other exp110 knobs are kept fixed.

## Overall AUC

| Dataset | Seed | Full adaptive | TKC-only | Delta |
|---|---:|---:|---:|---:|
| ASSIST09 holdout | 2027 | 0.753902 | 0.752956 | -0.000946 |
| ASSIST17 holdout | 2024 | 0.755751 | 0.754807 | -0.000944 |
| NIPS34 holdout | 2024 | 0.772583 | 0.772380 | -0.000203 |

Overall AUC moves in the expected direction on all three datasets, but the effect size is small.

## Coverage Slice

`low` means `target_coverage < 0.5`, including zero coverage. `gap = full AUC - low AUC`.

| Dataset | Model | Low AUC | Full AUC | Gap | Low Brier | Low ECE |
|---|---|---:|---:|---:|---:|---:|
| ASSIST09 holdout | Full adaptive | 0.733135 | 0.776578 | 0.043443 | 0.189102 | 0.078979 |
| ASSIST09 holdout | TKC-only | 0.731776 | 0.776582 | 0.044805 | 0.189430 | 0.078233 |
| ASSIST17 holdout | Full adaptive | 0.743160 | 0.780608 | 0.037447 | 0.213228 | 0.097672 |
| ASSIST17 holdout | TKC-only | 0.740388 | 0.781023 | 0.040635 | 0.214032 | 0.097381 |
| NIPS34 holdout | Full adaptive | 0.766900 | 0.780103 | 0.013203 | 0.195253 | 0.025829 |
| NIPS34 holdout | TKC-only | 0.766851 | 0.779556 | 0.012705 | 0.195471 | 0.031298 |

Readout:

- ASSIST09 and ASSIST17 support the intended story: TKC-only loses more on low coverage than on full coverage, and the coverage gap expands.
- NIPS34 is weak/mixed: low AUC barely changes, the gap does not expand, but low Brier and low ECE are worse under TKC-only.

## History Hiding

Evaluation-only stress test with `hide_ratio=0.8` and mask seeds `11,13,17`.

| Dataset | Model | Original AUC | Hidden AUC | Delta AUC | Hidden Brier | Hidden ECE |
|---|---|---:|---:|---:|---:|---:|
| ASSIST09 holdout | Full adaptive | 0.753902 | 0.717703 | 0.036198 | 0.198436 | 0.061857 |
| ASSIST09 holdout | TKC-only | 0.752956 | 0.715704 | 0.037252 | 0.199424 | 0.065218 |
| ASSIST17 holdout | Full adaptive | 0.755751 | 0.715649 | 0.040103 | 0.215247 | 0.058250 |
| ASSIST17 holdout | TKC-only | 0.754807 | 0.713119 | 0.041689 | 0.216500 | 0.060775 |
| NIPS34 holdout | Full adaptive | 0.772583 | 0.739493 | 0.033091 | 0.205920 | 0.021949 |
| NIPS34 holdout | TKC-only | 0.772380 | 0.740251 | 0.032130 | 0.206312 | 0.034510 |

Readout:

- ASSIST09 and ASSIST17 again support the story: TKC-only has lower hidden AUC and larger AUC drop under hidden-history stress.
- NIPS34 is mixed: TKC-only has slightly higher hidden AUC and smaller AUC drop, but worse hidden ACC, Brier, and ECE. Do not claim that the TKC-only ablation is uniformly worse on every metric and dataset.

## Paper-safe Interpretation

The safe statement is:

> Directly removing the UKC branch from student-state fusion consistently reduces overall AUC on the holdout splits. On ASSIST09 and ASSIST17, the degradation concentrates more on low-coverage or hidden-history conditions, supporting the need to complement tested-concept state with untested-concept state. NIPS34 shows weaker and mixed evidence, so the claim should be phrased as a trend rather than a universal law.

Avoid:

- claiming the adaptive gate is always better on every metric;
- claiming NIPS34 history hiding supports a larger AUC drop for TKC-only;
- mixing this single-seed ablation with multi-seed main-table claims.

## Artifacts

Remote code copy:

```text
/home/xph/jwc/research/local_data/tkc_fusion_ablation_code
```

Remote result directory:

```text
/home/xph/jwc/research/local_data/tkc_fusion_ablation_runs/results
```

Key summaries:

```text
assist09_holdout_tkc_fusion_coverage_summary.csv
assist09_holdout_tkc_fusion_hide80_summary.csv
assist17_holdout_tkc_fusion_coverage_report_summary.csv
assist17_holdout_tkc_fusion_hide80_report_summary.csv
nips34_holdout_tkc_fusion_coverage_report_summary.csv
nips34_holdout_tkc_fusion_hide80_report_summary.csv
```

Training summaries:

```text
assist17_holdout_tkc_only_seed2024.json
nips34_holdout_tkc_only_seed2024.json
```

Existing full-model summaries were copied with absolute checkpoint paths:

```text
assist17_holdout_full_seed2024_abs.json
nips34_holdout_full_seed2024_abs.json
```
