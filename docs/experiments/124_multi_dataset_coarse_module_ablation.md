# Experiment 124: multi-dataset coarse module ablation

## Status

`single_seed_cross_dataset_coarse_ablation_completed`

## Verdict

This experiment extends the coarse ablation framing beyond ASSIST09 to
ASSIST17 and NIPS34 holdout splits. It is single-seed and uses the current best
full-model口径 for each dataset:

- ASSIST17 holdout: seed 2028, exp110 `dual64x80`, learning rate `3e-4`.
- NIPS34 holdout: seed 2024, tuned `dim80x96`, learning rate `2e-4`.

The result is mixed:

- `w/o cognitive alignment objective` is useful for the coverage-bias story on
  ASSIST17: low AUC drops from `0.752372` to `0.739197`, coverage gap expands
  from `0.029172` to `0.042964`, and low ECE worsens from `0.042853` to
  `0.127166`.
- `w/o dual-branch ensemble` is mostly negative on overall AUC and low AUC for
  both datasets, so it can be described as a stability/performance component.
- `w/o evidence-aware readout` is not a robust positive module across datasets.
  It improves ASSIST17 and NIPS34 in this single-seed run, so it should not be
  presented as a core cross-dataset innovation.

For midterm slides, use this table as evidence that coarse ablation was checked
across datasets, but do not claim every coarse module is universally beneficial.

## Result Paths

Remote result directory:

`/home/xph/jwc/research/decoupled_cd/results/exp124_multi_dataset_coarse_ablation/`

Full-model references:

- ASSIST17 full: `results/exp122_best_seed_sweep/seed2028_assist17_holdout_exp110.json`
- NIPS34 full: `results/exp121_dataset_tuning/seed2024_nips34_holdout_dim80_lr2e4.json`

New ablation summaries:

- `seed2028_assist17_holdout_without_evidence_aware_readout.json`
- `seed2028_assist17_holdout_without_cognitive_alignment_objective.json`
- `seed2028_assist17_holdout_without_dual_branch_ensemble.json`
- `seed2024_nips34_holdout_without_evidence_aware_readout.json`
- `seed2024_nips34_holdout_without_cognitive_alignment_objective.json`
- `seed2024_nips34_holdout_without_dual_branch_ensemble.json`

Coverage reports:

- `assist17_holdout_coarse_coverage_report.json`
- `nips34_holdout_coarse_coverage_report.json`

## Module Definitions

### Evidence-aware readout

Removed together:

- high-concept logit adapter
- pairwise history interaction adapter
- interpretable readout expert adapter
- student-conditioned UKC readout residual
- concept evidence readout residual

### Cognitive alignment objective

Removed together:

- history evidence logit prior residual and cognitive-alignment loss
- concept evidence prior residual and train-only prior schedule

### Dual-branch ensemble

Removed together:

- dual CDM ensemble
- secondary tower
- branch BCE

## Overall Metrics

| Dataset | Variant | AUC | Delta AUC | ACC | RMSE | Brier | ECE | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assist17_holdout | Full model | 0.7637 | +0.0000 | 0.6991 | 0.4425 | 0.1958 | 0.0289 | 50 |
| assist17_holdout | w/o evidence-aware readout | 0.7890 | +0.0253 | 0.7206 | 0.4299 | 0.1849 | 0.0206 | 182 |
| assist17_holdout | w/o cognitive alignment objective | 0.7516 | -0.0121 | 0.6794 | 0.4517 | 0.2040 | 0.0641 | 50 |
| assist17_holdout | w/o dual-branch ensemble | 0.7523 | -0.0114 | 0.6904 | 0.4462 | 0.1991 | 0.0087 | 59 |
| nips34_holdout | Full model | 0.7728 | +0.0000 | 0.7041 | 0.4391 | 0.1928 | 0.0272 | 62 |
| nips34_holdout | w/o evidence-aware readout | 0.7755 | +0.0027 | 0.7073 | 0.4373 | 0.1912 | 0.0167 | 110 |
| nips34_holdout | w/o cognitive alignment objective | 0.7736 | +0.0008 | 0.7058 | 0.4379 | 0.1917 | 0.0086 | 56 |
| nips34_holdout | w/o dual-branch ensemble | 0.7714 | -0.0014 | 0.7046 | 0.4394 | 0.1930 | 0.0197 | 76 |

## Coverage Slice Metrics

| Dataset | Variant | Overall AUC | Low AUC | Full AUC | Gap | Low Brier | Low ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| assist17_holdout | Full model | 0.7637 | 0.7524 | 0.7815 | 0.0292 | 0.2018 | 0.0429 |
| assist17_holdout | w/o evidence-aware readout | 0.7890 | 0.7851 | 0.7949 | 0.0097 | 0.1873 | 0.0287 |
| assist17_holdout | w/o cognitive alignment objective | 0.7516 | 0.7392 | 0.7822 | 0.0430 | 0.2214 | 0.1272 |
| assist17_holdout | w/o dual-branch ensemble | 0.7523 | 0.7322 | 0.7772 | 0.0450 | 0.2075 | 0.0309 |
| nips34_holdout | Full model | 0.7728 | 0.7677 | 0.7799 | 0.0122 | 0.1960 | 0.0408 |
| nips34_holdout | w/o evidence-aware readout | 0.7755 | 0.7756 | 0.7758 | 0.0002 | 0.1913 | 0.0181 |
| nips34_holdout | w/o cognitive alignment objective | 0.7736 | 0.7663 | 0.7823 | 0.0160 | 0.1948 | 0.0134 |
| nips34_holdout | w/o dual-branch ensemble | 0.7714 | 0.7675 | 0.7774 | 0.0099 | 0.1953 | 0.0302 |

## Decision

- Keep `TKC/UKC decoupling` and `cognitive alignment` as the cleanest
  story-facing mechanisms.
- Treat dual branch as a stability/performance component rather than the core
  contribution.
- Do not use evidence-aware readout as a core ablation claim. In this
  cross-dataset single-seed check it behaves like a dataset-sensitive auxiliary
  component.
