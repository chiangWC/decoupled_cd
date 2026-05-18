# Experiment 99: Hybrid Stacker Multi-Seed 0.778 Validation

## Status

`hybrid_multiseed_signal_confirmed`, not promoted to default CDM.

## Verdict

The experiment 91/97 hybrid hist-gradient stacker clears the adjusted practical sprint target `test_auc >= 0.778` on three matched seeds.

The three-seed stacker mean is `test_auc = 0.786910`, with every tested seed above `0.785`. Secondary metrics are also consistently strong: mean `ACC = 0.744423`, `RMSE = 0.413858`, `Brier = 0.171278`, and `ECE = 0.007453`.

This is the current strongest controlled route toward the requested growth target. It remains an explicitly hybrid evaluator: the combiner is trained on validation labels and uses train-history tabular side-channel features. Do not report it as the default CDM runner or as a pure-CDM promotion.

## Base And References

Branch: `exp/auc-078-exploration`

Base branch: `exp/trellis-trial`

Adjusted practical target:

- User revised `0.78` to `0.778`.
- Corrected high-water reference remains seed2027 `0.7726816125195809`; `+0.004` threshold remains `0.776682`.

Parent details:

- [091 corrected high-water hybrid stacker](./091_corrected_high_water_hybrid_stacker.md)
- [097 autonomous growth signal exploration](./097_autonomous_growth_signal_exploration.md)
- [098 pure CDM runner integration](./098_pure_cdm_runner_integration.md)

## Implementation

Restored the experiment 91 hybrid evaluator on the current trial descendant branch:

- `scripts/evaluate_ensemble.py`
- `tests/test_evaluate_ensemble.py`

Focused remote verification:

```bash
python -m py_compile scripts/evaluate_ensemble.py tests/test_evaluate_ensemble.py
python -m unittest tests.test_evaluate_ensemble
```

Result: `Ran 1 test ... OK`.

## Commands

Seed2024 recheck used the original experiment 91 member summaries:

```bash
python scripts/evaluate_ensemble.py \
  --summaries \
    results/auc_growth_exploration/seed2024_raw_prior_max025_300ep.json \
    results/correct_auc_baseline_continue/seed2024_raw_prior_max05_300ep.json \
    results/correct_auc_baseline_continue/seed2024_lr7e4_es20_sched5_raw_prior_max025_300ep.json \
    results/correct_auc_baseline_continue/seed2024_no_expert_raw_prior_max025_300ep.json \
  --combiner stack_hist_gradient \
  --stack-feature-set hybrid \
  --average prob \
  --stack-max-iter 200 \
  --stack-learning-rate 0.03 \
  --stack-l2-regularization 0.01 \
  --output results/auc_078_exploration/seed2024_stack_hist_gradient_4member_hybrid_prob_test.json
```

Seeds 2025 and 2026 were regenerated on this branch with the same four-member family:

- `raw_prior_max025`
- `raw_prior_max05`
- `lr7e4_es20_sched5_raw_prior_max025`
- `no_expert_raw_prior_max025`

Then each seed used the same hist-gradient hybrid stacker command shape as seed2024.

## Stacker Results

| seed | output | test_auc | ACC | RMSE | Brier | ECE | valid_auc | verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 2024 | `results/auc_078_exploration/seed2024_stack_hist_gradient_4member_hybrid_prob_test.json` | 0.787288 | 0.744201 | 0.413708 | 0.171154 | 0.006625 | 0.836768 | clears 0.778 |
| 2025 | `results/auc_078_exploration/seed2025_stack_hist_gradient_4member_hybrid_prob_test.json` | 0.788060 | 0.744067 | 0.413560 | 0.171032 | 0.009687 | 0.846259 | clears 0.778 |
| 2026 | `results/auc_078_exploration/seed2026_stack_hist_gradient_4member_hybrid_prob_test.json` | 0.785382 | 0.745000 | 0.414304 | 0.171648 | 0.006047 | 0.848005 | clears 0.778 |
| mean | - | 0.786910 | 0.744423 | 0.413858 | 0.171278 | 0.007453 | 0.843677 | confirmed |

## Member Diagnostics

Seed2026 is the important stress test:

- `raw_prior_max025`: `test_auc = 0.767959`
- `raw_prior_max05`: `test_auc = 0.502923`
- `lr7e4_es20_sched5_raw_prior_max025`: `test_auc = 0.767608`
- `no_expert_raw_prior_max025`: `test_auc = 0.750855`
- hybrid stacker: `test_auc = 0.785382`

The stacker remains above `0.778` even when one member collapses and another is weak. That supports the interpretation that the growth is mainly from validation-trained use of train-history tabular features, not from a single deterministic prior checkpoint.

## Decision

- Treat the hist-gradient hybrid stacker as the active `0.778+` growth signal.
- Keep it explicitly hybrid and optional; do not promote it into `scripts/run_assist09_baseline.sh`.
- The next useful follow-up is either seed2027 validation or a separate integration task that converts the same train-history feature family into a model-side readout/objective without validation-trained stacking.
- Stop pure-CDM micro-sweeps around experiment 95 unless a new mechanism-level reason appears.
