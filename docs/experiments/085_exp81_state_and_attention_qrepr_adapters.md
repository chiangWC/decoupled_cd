# Experiment 85: exp81 state and attention qrepr adapters

- 分支: `exp/expert-output-modulation`
- base: 当前 exp81 伪主线
- 代码提交:
  - `9d406aa` concept evidence state adapter
  - `09171d3` concept evidence state min-attempt gate
  - `3aca679` target-conditioned student state adapter
  - `5ae3861` target evidence attention qrepr adapter
- 状态: rejected diagnostic, not a trial candidate

## Motivation

Experiment 84 rejected deterministic evidence-prior and expert-output micro-tuning. This experiment moved the same target-local evidence signal earlier in the model:

- state formation: rewrite per-concept TKC state from explicit student-concept evidence
- target student state: rewrite the gathered student state before cognitive matching
- q representation: use target-local evidence/TKC/UKC attention over Q concepts and apply a zero-init bounded residual to `q_repr`

Matched exp81 references:

| seed | AUC | ACC | RMSE | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 0.767478 | 0.734248 | 0.425562 | 0.181103 | 0.046972 |
| 2025 | 0.771661 | 0.732992 | 0.424611 | 0.180294 | 0.050071 |
| 2027 | 0.772682 | 0.732821 | 0.424350 | 0.180073 | 0.050954 |

## Engineering validation

- Local checks:
  - `python3 -m py_compile models/decoupled_cdm.py scripts/train.py scripts/evaluate.py scripts/analyze_prediction_slices.py tests/test_decoupled_cdm.py tests/test_cli_config_wiring.py`
  - `python3 -B -m unittest tests.test_cli_config_wiring`
  - `git diff --check`
- Remote tests:
  - after `09171d3`: `bash scripts/remote_exec.sh python -m unittest tests.test_hetero_propagation tests.test_decoupled_cdm tests.test_cli_config_wiring` -> `46 tests OK`
  - after `3aca679`: same command -> `52 tests OK`
  - after `5ae3861`: same command -> `58 tests OK`
- Remote smoke:
  - target-conditioned student state, 2 epochs, passed and recorded `target_conditioned_student_state_*`
  - target evidence attention qrepr, 2 epochs, passed and recorded `target_evidence_attention_qrepr_*`

## Results

### Concept evidence state adapter

`--concept-evidence-state-adapter --concept-evidence-state-max-scale 0.125 --concept-evidence-state-min-attempts 3.0 --cognitive-logit-scale 0.9`

| seed | AUC | delta AUC | ACC delta | RMSE delta | Brier delta | ECE delta | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 | 0.770814 | +0.003335 | -0.001199 | -0.001367 | -0.001162 | -0.002928 | positive first seed |
| 2025 | 0.766028 | -0.005633 | +0.000038 | +0.002370 | +0.002019 | +0.002837 | rejected tail |

Conclusion: min-attempt gating did not fix the seed tail. Directly editing per-concept state from evidence can make one seed look strong, but the next seed loses substantially.

### Target-conditioned student state

| config | seed | AUC | delta AUC | ACC | RMSE | Brier | ECE | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scale0.25 all counts | 2024 | 0.742849 | -0.024630 | 0.720851 | 0.447259 | 0.200041 | 0.110694 | rejected |
| scale0.125 count=1 | 2024 | 0.741039 | -0.026439 | 0.721536 | 0.439204 | 0.192900 | 0.076872 | rejected |

Conclusion: rewriting `student_state` before cognitive matching is too destructive. Narrowing to single-concept targets and lowering scale does not recover ranking.

### Target evidence attention qrepr

| config | seed | AUC | delta AUC | ACC delta | RMSE delta | Brier delta | ECE delta | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| min_count=2 scale0.25 | 2024 | 0.763475 | -0.004003 | -0.000971 | +0.002041 | +0.001741 | +0.004684 | rejected broad trigger |
| exact3 scale0.125 | 2024 | 0.767584 | +0.000106 | +0.000381 | -0.000438 | -0.000373 | -0.003205 | weak positive |
| exact3 scale0.125 | 2025 | 0.771068 | -0.000593 | -0.001579 | +0.000406 | +0.000345 | -0.000228 | weak negative |
| exact3 scale0.125 | 2027 | 0.772020 | -0.000662 | -0.002854 | +0.001724 | +0.001467 | +0.006138 | rejected |

Three-seed exact3 mean relative to matched exp81:

| AUC delta | ACC delta | RMSE delta | Brier delta | ECE delta |
| ---: | ---: | ---: | ---: | ---: |
| -0.000383 | -0.001351 | +0.000564 | +0.000480 | +0.000902 |

Conclusion: target-local attention over Q concepts is less destructive than state rewrite, but exact3 only gives a weak seed2024 signal and fails seed2025/2027. It is not a trial candidate.

## Decision

- Do not merge any experiment 85 adapter into `exp/trellis-trial`.
- Do not continue parameter sweeps around `concept_evidence_state_adapter`, `target_conditioned_student_state_adapter`, or this exact `target_evidence_attention_qrepr_adapter` form.
- The useful diagnostic is action placement: final-logit priors and state rewrites are both unstable; if revisiting target-local evidence, keep it as a tightly scoped readout auxiliary or change the training objective, not the main student state.
