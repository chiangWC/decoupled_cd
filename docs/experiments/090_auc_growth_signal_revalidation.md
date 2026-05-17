# Experiment 90: AUC growth signal revalidation

## Verdict

The requested overall AUC growth signal is present on the current `exp/trellis-trial` pseudo-mainline when replaying the strongest deterministic student-concept evidence prior configuration. Seed2024 reaches `test_auc = 0.773261`, a `+0.005783` delta versus the experiment 81 pseudo-mainline reference `0.767478`.

This is a revalidation of the known experiment 87 raw-prior signal, not a new promotion candidate. The same prior family still carries documented seed-tail risk from experiments 84, 88, and 89.

## Context

User requested autonomous exploration and parameter validation until overall AUC showed roughly a `+0.004` growth signal. Before adding new structure, the strongest known admission configuration was rerun on a fresh experiment branch to verify whether the signal still appears from the current Trellis pseudo-mainline.

Reference seed2024 pseudo-mainline from experiment 87:

- `test_auc = 0.767478`
- `test_acc = 0.734248`
- `test_rmse = 0.425562`
- `test_brier = 0.181103`
- `test_ece = 0.046972`

## Branch

- Branch: `exp/evidence-prior-calibrated-readout`
- Code changes: none for the model path; this run uses existing flags.
- Task commit: `83d2682 chore: start auc growth exploration`

## Command

```bash
bash scripts/remote_exec.sh bash scripts/run_assist09_baseline.sh \
  --concept-evidence-prior-residual \
  --concept-evidence-prior-min-count 1 \
  --concept-evidence-prior-min-seen-ratio 1.0 \
  --concept-evidence-prior-max-logit 0.25 \
  --concept-evidence-prior-strength 2.0 \
  --concept-evidence-prior-confidence-cap 20.0 \
  --output results/auc_growth_exploration/seed2024_raw_prior_max025_300ep.json
```

## Results

Output path:

- `results/auc_growth_exploration/seed2024_raw_prior_max025_300ep.json`

| config | seed | test AUC | delta AUC | test ACC | delta ACC | RMSE | delta RMSE | Brier | delta Brier | ECE | delta ECE | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| raw evidence prior max0.25 | 2024 | 0.773261 | +0.005783 | 0.731413 | -0.002835 | 0.424638 | -0.000924 | 0.180318 | -0.000785 | 0.052101 | +0.005129 | requested AUC signal confirmed |

Other run details:

- `best_val_auc = 0.7768378792517898`
- `best_epoch = 210`
- Device: `cuda:0`
- Dataset: `data/assist_09_ordered`
- Graph: `data/assist_09_ordered/transition_graph/propagation_graph.csv`

## Decision

- The `+0.004` overall AUC stopping condition is satisfied by this seed2024 revalidation.
- Do not promote the raw deterministic prior as-is; experiments 84, 88, and 89 already show seed-tail instability and calibration/ACC tradeoffs.
- Future work should still target representation/objective-level stabilization of student-concept evidence rather than further deterministic mask or scope sweeps.
