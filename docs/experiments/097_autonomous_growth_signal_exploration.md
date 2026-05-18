# Experiment 97: autonomous growth signal exploration

## Status

`hybrid_signal_reconfirmed`, pure-CDM follow-ups rejected as single-seed improvements.

## Verdict

Reliability-weighting and target-construction variants around the experiment 95 `cogonly loss_only` pure-CDM alignment objective did not produce a corrected high-water breakthrough on seed2027. The best pure-CDM variant in this round was `confidence_power=0.5, confidence_floor=0.2` at `alignment=0.05`, with `test_auc = 0.776264`, a tiny `+0.000101` over experiment 95's seed2027 `0.776163` but still below the corrected stop threshold `0.776682`.

The valid growth signal remains the explicitly hybrid experiment 91 stacker. After bringing `scripts/evaluate_ensemble.py` onto `exp/smooth-cognitive-alignment`, the seed2024 hist-gradient hybrid stacker was re-run and reproduced `test_auc = 0.787288`, a `+0.014606` delta versus the corrected high-water reference `0.772682`, with ACC/RMSE/Brier/ECE all strongly positive. Treat this as the active growth signal and next validation target; do not present it as a default pure-CDM model.

## Base And References

Branch: `exp/smooth-cognitive-alignment`

Corrected high-water reference:

- output: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`
- `test_auc = 0.7726816125195809`
- corrected `+0.004` target: `0.776682`

Pure-CDM reference:

- experiment 95 seed2027: `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_w005_300ep.json`
- `test_auc = 0.776163`

Parent details:

- [091 corrected high-water hybrid stacker](./091_corrected_high_water_hybrid_stacker.md)
- [095 history alignment CF-risk ablation](./095_history_alignment_cf_risk_ablation.md)
- [096 smooth cognitive alignment loss](./096_smooth_cognitive_alignment_loss.md)

## Implementation

Added an opt-in confidence-weighted cognitive alignment path:

- `--history-evidence-cognitive-alignment-confidence-power`
- `--history-evidence-cognitive-alignment-confidence-cap`
- `--history-evidence-cognitive-alignment-confidence-floor`
- runner: `scripts/run_assist09_history_alignment_reliability_trial.sh`

Default behavior is unchanged because `confidence_power = 0.0` disables weighting.

Also restored the hybrid evaluator on this branch:

- `scripts/evaluate_ensemble.py`
- `tests/test_evaluate_ensemble.py`

Commits:

- `c471e43 feat: add reliability-weighted alignment trial`
- `60fc8f8 fix: align reliability trial seed output name`
- `5863fbc feat: add hybrid ensemble evaluator`

Remote verification:

```bash
python -m py_compile trainers/engine.py scripts/train.py
python -m unittest tests.test_training_modes
bash scripts/run_assist09_history_alignment_reliability_trial.sh --epochs 1 --max-rows 2000 --output results/pure_cdm_hybrid_signal/smoke_history_alignment_confidence_weighted_seed2027.json
python -m py_compile scripts/evaluate_ensemble.py tests/test_evaluate_ensemble.py
python -m unittest tests.test_evaluate_ensemble
```

## Pure-CDM Seed2027 Probes

All runs use the experiment 95 `cogonly loss_only` family unless noted.

| variant | output | test_auc | vs exp95 seed2027 | vs high-water | ACC | RMSE | Brier | ECE | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| confidence `power=1.0`, floor `0.2`, `alignment=0.05` | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_confpow1_floor02_w005_300ep.json` | 0.775841 | -0.000322 | +0.003160 | 0.736646 | 0.422300 | 0.178337 | 0.044911 | below exp95 |
| confidence `power=0.5`, floor `0.2`, `alignment=0.05` | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_confpow05_floor02_w005_300ep.json` | 0.776264 | +0.000101 | +0.003582 | 0.736703 | 0.421584 | 0.177733 | 0.042198 | tiny positive, below threshold |
| confidence `power=0.5`, floor `0.2`, `alignment=0.0881` | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_confpow05_floor02_w00881_300ep.json` | 0.775493 | -0.000670 | +0.002811 | 0.737198 | 0.421123 | 0.177344 | 0.035757 | hot weight not rescued |
| target-concept only, `alignment=0.05` | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_targetonly_w005_300ep.json` | 0.775303 | -0.000860 | +0.002621 | 0.737045 | 0.422281 | 0.178322 | 0.044698 | target-only worse |

Interpretation:

- Confidence weighting can slightly improve seed2027 over experiment 95 when softened (`power=0.5`), but the gain is far below the threshold for expansion.
- Confidence weighting does not rescue the hot `alignment=0.0881` window.
- Removing the global concept and mastery terms hurts AUC, so the experiment 95 mixed target remains the better pure-CDM target among tested variants.

## Hybrid Recheck

Command:

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
  --output results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_prob_test_recheck_smooth_branch.json
```

Result:

| output | test_auc | delta vs high-water | ACC | RMSE | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_prob_test_recheck_smooth_branch.json` | 0.787288 | +0.014606 | 0.744201 | 0.413708 | 0.171154 | 0.006625 |

## Decision

- Stop the reliability-weighted pure-CDM micro-sweep unless a new mechanism-level reason appears.
- Keep experiment 95 as the current pure-CDM trial candidate.
- Treat the experiment 91/97 hybrid hist-gradient stacker as the active growth signal.
- Next useful step: controlled multi-seed hybrid validation, or a separate integration task to move the same train-history feature family into a trainable readout/objective without pretending it is the current default CDM runner.
