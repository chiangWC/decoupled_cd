# Experiment 108: lightweight single64 follow-up rejections

## Status

`lightweight_followup_rejected`

## Verdict

The next low-VRAM pure-CDM follow-ups on top of exp105 `single64` still do not
produce a credible replacement for the lightweight reference.

- Soft exercise-difficulty regularization is negative on the weak seed.
- Tightening current mainline residual triggers is either clearly negative or
  outright collapses training.
- Cognitive-alignment residual focusing also collapses.
- Delaying the train-only concept prior from epoch `135` to `170` is the only
  non-destructive new schedule probe, but it repeats the same pattern as the
  earlier warmup probe: seed2026 is only microscopically positive while
  seed2024 regresses.
- Switching single-checkpoint selection from validation `AUC` to validation
  `Brier` hurts weak-seed AUC.

Keep exp105 `single64` as the lightweight reference only. Do not continue these
families as local default-route rescues.

## Context

- Branch: `exp/lightweight-single-tower-cdm`
- Reference lightweight runner: exp105 `single64`
  - seed2024 `test_auc = 0.778379`
  - seed2026 `test_auc = 0.775065`
  - historical four-seed mean `0.776736`
  - `max_cuda_memory_allocated_gb ~= 6.33`
- Constraints:
  - pure CDM only
  - single training run
  - single selected checkpoint
  - no dual tower
  - no hybrid side channel
  - no multi-checkpoint average

## Engineering

This experiment also restored the dormant
`exercise_evidence_difficulty_regularization_*` training controls in
[`scripts/train.py`](../../scripts/train.py) so the existing engine mechanism
can be probed without one-off code edits.

Remote verification after the CLI change:

- `python3 -m py_compile scripts/train.py tests/test_train_cli.py tests/test_training_modes.py`
- `python3 -m unittest tests.test_train_cli tests.test_training_modes`

Both passed.

## Results

| variant | seed | test AUC | delta vs single64 base | max CUDA GB | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| difficulty regularization `weight=0.004 min_count=3 max_abs_logit=0.15 strength=4 cap=64` | 2026 | 0.774668 | -0.000397 | 6.330007 | reject |
| pairwise history trigger `min_count=3` | 2026 | 0.773786 | -0.001279 | 6.329924 | reject |
| high-concept trigger `min_count=3` | 2026 | 0.504296 | -0.270769 | 6.329924 | collapse |
| cognitive alignment residual focus `power=1.0 floor=0.2` | 2026 | 0.504197 | -0.270868 | 6.329924 | collapse |
| concept prior train start `170` | 2026 | 0.775084 | +0.000019 | 6.329924 | too small |
| concept prior train start `170` | 2024 | 0.778212 | -0.000167 | 6.329924 | reject |
| checkpoint selection metric `brier` | 2026 | 0.773786 | -0.001279 | 6.329924 | reject |

Result paths:

- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_diffreg_w004_mc3_logit015_str4_cap64_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_pairmin3_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_highconcept3_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_cogresidpow1_floor02_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_start170_300ep.json`
- `results/pure_cdm_default_promotion/seed2024_single64_late170to230_eprior_trainonly_start170_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_selbrier_300ep.json`

## Decision

- Do not continue `exercise_evidence_difficulty_regularization` follow-ups on
  this route unless there is a new scoping mechanism beyond a global soft pull.
- Do not continue mainline residual trigger threshold edits such as
  `pairwise-history-interaction-min-count` or `high-concept-logit-min-count`.
- Do not continue cognitive-alignment residual-focus sweeps; this mechanism is
  destructive on the lightweight base.
- Treat the train-only concept-prior timing family as low-signal only:
  `warmup30@135` and `start170` both fail to improve both anchor seeds.
- Do not continue validation `Brier` checkpoint selection on this route.

The next lightweight pure-CDM attempt should come from a new mechanism-level
idea, not from more local schedule, trigger-threshold, or checkpoint-selection
micro-tuning.
