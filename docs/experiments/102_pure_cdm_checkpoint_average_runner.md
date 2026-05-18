# Experiment 102: Pure CDM Checkpoint Average Runner

## Status

`pure_cdm_checkpoint_average_signal_confirmed`

## Verdict

Prediction-only averaging of the experiment 95 cog-only checkpoint and experiment 100 dim80 + output-alignment checkpoint clears the user-adjusted `0.778` target on all four seeds.

This is not the experiment 99 hybrid stacker: it uses no train-history tabular feature side-channel, no validation-trained combiner, and no learned post-hoc model. The evaluator only averages two pure-CDM checkpoint probability predictions on the requested split.

The best mode is probability averaging. Four-seed test AUC is `0.779011/0.778059/0.778969/0.779448`, mean `0.778872`, which is `+0.002593` over experiment 95 and `+0.001841` over experiment 100's dim80 + output-alignment candidate. This removes the seed2024/2025 regression that blocked experiment 100 default promotion while preserving seed2026/2027 headroom.

This supports a new opt-in pure CDM runner/evaluator candidate. It does not by itself change `scripts/run_assist09_baseline.sh`, because it is a two-checkpoint inference runner rather than a single default training script.

## Base

Branch: `exp/pure-cdm-default-promotion`

Member checkpoints:

- experiment 95: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_cogonly_w005_300ep.json`
- experiment 100: `results/pure_cdm_default_promotion/seed{seed}_history_alignment_cogonly_dim80_outalign0004_300ep.json`

Evaluator:

- `scripts/evaluate_checkpoint_average.py`
- tests: `tests/test_evaluate_checkpoint_average.py`

Verification:

```bash
python -m py_compile scripts/evaluate_checkpoint_average.py tests/test_evaluate_checkpoint_average.py
python -m unittest tests.test_evaluate_checkpoint_average
```

## Admission Probes Before Average

These single-seed probes were run because experiment 101 left seed2024 as the default-promotion blocker. None fixed the tail, so they were not expanded.

| variant | seed | test_auc | delta vs exp95 | delta vs dim80+out004 | ACC | RMSE | Brier | ECE | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| concept evidence calibrated readout max logit `0.10` | 2024 | 0.776143 | -0.001700 | -0.000276 | 0.737864 | 0.421514 | 0.177674 | 0.045073 | rejected |
| exercise evidence difficulty adapter max logit `0.10` + output align | 2024 | 0.776835 | -0.001008 | +0.000416 | 0.740090 | 0.420791 | 0.177065 | 0.040510 | insufficient |
| student evidence GS prior max logit `0.10` | 2024 | 0.776308 | -0.001534 | -0.000111 | 0.739101 | 0.421504 | 0.177666 | 0.045000 | rejected |
| student ability prior max logit `0.10` | 2024 | 0.776442 | -0.001400 | +0.000023 | 0.739025 | 0.421341 | 0.177528 | 0.043972 | rejected |
| exercise difficulty adapter max logit `0.10`, no output align | 2024 | 0.776425 | -0.001418 | +0.000006 | 0.737578 | 0.421369 | 0.177551 | 0.043342 | rejected |
| output calibration max logit `0.10` | 2024 | 0.776839 | -0.001004 | +0.000420 | 0.740071 | 0.420937 | 0.177188 | 0.042697 | insufficient |
| output calibration max logit `0.25` | 2024 | 0.776351 | -0.001492 | -0.000069 | 0.739786 | 0.421393 | 0.177572 | 0.046755 | rejected |

Decision: do not continue these model-side readout/shortcut probes around dim80. The useful structural signal is checkpoint complementarity between experiment 95 and experiment 100.

## Four-Seed Checkpoint Average

Command shape:

```bash
python scripts/evaluate_checkpoint_average.py \
  --summaries \
    results/pure_cdm_hybrid_signal/seed${seed}_history_alignment_cogonly_w005_300ep.json \
    results/pure_cdm_default_promotion/seed${seed}_history_alignment_cogonly_dim80_outalign0004_300ep.json \
  --average prob \
  --split test \
  --output results/pure_cdm_default_promotion/seed${seed}_exp95_dim80out004_prob_average_eval_script.json
```

| seed | prob avg AUC | delta vs exp95 | delta vs dim80+out004 | ACC | RMSE | Brier | ECE | output |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2024 | 0.779011 | +0.001168 | +0.002592 | 0.737426 | 0.420443 | 0.176772 | 0.042244 | `results/pure_cdm_default_promotion/seed2024_exp95_dim80out004_prob_average_eval_script.json` |
| 2025 | 0.778059 | +0.001618 | +0.002578 | 0.736969 | 0.420585 | 0.176892 | 0.039327 | `results/pure_cdm_default_promotion/seed2025_exp95_dim80out004_prob_average_eval_script.json` |
| 2026 | 0.778969 | +0.004300 | +0.000869 | 0.738111 | 0.419737 | 0.176179 | 0.035738 | `results/pure_cdm_default_promotion/seed2026_exp95_dim80out004_prob_average_eval_script.json` |
| 2027 | 0.779448 | +0.003285 | +0.001326 | 0.739025 | 0.420379 | 0.176718 | 0.040308 | `results/pure_cdm_default_promotion/seed2027_exp95_dim80out004_prob_average_eval_script.json` |

Mean AUC:

- probability average: `0.778872`
- delta vs experiment 95 mean: `+0.002593`
- delta vs experiment 100 dim80 + output-alignment mean: `+0.001841`

Logit average was also positive but slightly lower:

- logit-average mean AUC: `0.778852`
- delta vs experiment 95 mean: `+0.002573`
- delta vs experiment 100 dim80 + output-alignment mean: `+0.001822`

## Decision

- Keep experiment 95 as the active single-checkpoint pure CDM trial runner.
- Keep experiment 100 dim80 + output alignment as an opt-in member checkpoint, not a default single-checkpoint promotion.
- Promote `scripts/evaluate_checkpoint_average.py` as the current opt-in pure-CDM checkpoint-average evaluator candidate.
- Prefer probability averaging over logit averaging for the experiment 95 + experiment 100 member pair.
- Do not describe this as hybrid stacker evidence: it does not use valid-trained combiner weights or train-history tabular features outside the checkpoints.
- Do not update `scripts/run_assist09_baseline.sh` yet; the accepted default-training question remains separate from opt-in checkpoint-average inference.
