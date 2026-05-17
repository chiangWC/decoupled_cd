# Experiment 84: exp81 expert output modulation and evidence-prior rescues

- 分支: `exp/expert-output-modulation`
- base: 当前 exp81 伪主线
- 代码提交:
  - `fa97282` bound readout expert residual
  - `64af856` scope readout expert bound by concept count
  - `b3ee866` scale readout expert by uncertainty
  - `5200e29` add cognitive logit scale
  - `828901d` replay q-local interaction after expert
  - `4e3b1cf` apply evidence prior on final odds
  - `938ce86` scope evidence prior by concept count
  - `629d087` allow eval-only evidence prior
- 状态: rejected / diagnostic, not a trial candidate

## Motivation

Experiment 83 showed that q-local representation signal can recover in a no-expert matched family, but the current full-trigger interpretable readout expert suppresses it on exp81. This experiment tested whether changing the expert output form/position, then replaying deterministic evidence prior variants around the expert, can recover a stable ranking signal.

Matched exp81 references:

| seed | AUC | ACC | RMSE | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 0.767478 | 0.734248 | 0.425562 | 0.181103 | 0.046972 |
| 2025 | 0.771661 | 0.732992 | 0.424611 | 0.180294 | 0.050071 |
| 2026 | 0.771263 | 0.732897 | 0.424666 | 0.180341 | 0.046958 |
| 2027 | 0.772682 | 0.732821 | 0.424350 | 0.180073 | 0.050954 |

## Engineering validation

- Local syntax/static checks:
  - `python3 -m py_compile models/decoupled_cdm.py scripts/train.py scripts/evaluate.py scripts/analyze_prediction_slices.py tests/test_decoupled_cdm.py`
  - `git diff --check`
- Remote focused tests:
  - `bash scripts/remote_exec.sh python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - after `828901d`: `44 tests OK`
  - after `4e3b1cf`: `45 tests OK`
  - after `938ce86` / `629d087`: `46 tests OK`
- Remote smoke covered:
  - post-expert q-local apply mode
  - final-logit evidence prior apply mode
  - single-only evidence prior max-count
  - eval-only evidence prior train mode

## Expert-output variants

| config | seed | AUC delta | ACC delta | secondary result | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| expert residual max_logit=1.0 | 2024 | +0.001545 | -0.000514 | RMSE/Brier/ECE improved | single-seed signal |
| expert residual max_logit=1.0 | 2025 | -0.000416 | -0.000495 | RMSE/Brier/ECE slightly improved | not stable |
| expert bound max_count=2 | 2024 | +0.001908 | n/a | promising first seed | failed expansion |
| expert bound max_count=2 | 2025 | -0.000536 | n/a | secondary metrics worse | rejected |
| uncertainty-scaled expert | 2024 | -0.001585 | n/a | negative | rejected |

Conclusion: bounding the full-trigger expert can improve one seed, but it does not generalize; uncertainty scaling directly hurts ranking.

## Post-expert q-local replay

The target-concept interaction was replayed as a cognitive-logit delta after expert/evidence residuals to bypass expert suppression.

| config | seed | AUC | delta AUC | ACC | delta ACC | RMSE | Brier | ECE | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact3 post_expert_delta scale0.25 | 2024 | 0.767991 | +0.000513 | 0.733848 | -0.000400 | 0.425077 | 0.180691 | 0.042913 | weak |
| exact3 post_expert_delta scale0.5 | 2024 | 0.768436 | +0.000958 | 0.734324 | +0.000076 | 0.424957 | 0.180588 | 0.044826 | best seed2024 |
| exact3 post_expert_delta scale0.75 | 2024 | 0.768551 | +0.001073 | 0.732745 | -0.001503 | 0.425222 | 0.180814 | 0.042239 | ACC side effect |
| exact3 post_expert_delta scale0.5 | 2025 | 0.770989 | -0.000672 | 0.724562 | -0.008430 | 0.428579 | 0.183680 | 0.065262 | rejected |
| exact3 post_expert_delta scale0.25 | 2025 | 0.771123 | -0.000538 | 0.730747 | -0.002245 | 0.425221 | 0.180813 | 0.049714 | rejected |

Conclusion: moving q-local after the expert reduces the original negative effect on seed2024, but seed2025 remains negative. Expert suppression was not the only failure mode.

## Deterministic evidence-prior rescues

Replaying deterministic `concept_evidence_prior_residual` on exp81 gave strong seed2024 ranking, but repeated attempts failed to stabilize seed2026.

| config | seed | AUC | delta AUC | ACC delta | RMSE delta | Brier delta | ECE delta | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cognitive max0.25 | 2024 | 0.773261 | +0.005783 | -0.002835 | -0.000924 | -0.000785 | +0.005129 | strong seed |
| cognitive max0.25 | 2025 | 0.773105 | +0.001444 | +0.001770 | -0.001357 | -0.001150 | -0.002322 | strong seed |
| cognitive max0.25 | 2026 | 0.767959 | -0.003304 | +0.000304 | +0.000416 | +0.000354 | -0.003269 | reject |
| cognitive max0.25 | 2027 | 0.761157 | -0.011524 | -0.003349 | +0.004759 | +0.004062 | -0.001790 | rejected tail |
| cognitive max0.125 | 2024 | 0.772614 | +0.005136 | -0.002835 | -0.000275 | -0.000234 | +0.007115 | still biased |
| cognitive max0.125 | 2025 | 0.770778 | -0.000883 | +0.000894 | -0.000276 | -0.000234 | -0.003946 | not stable |
| cognitive max0.125 | 2026 | 0.770272 | -0.000991 | +0.000076 | -0.000059 | -0.000050 | -0.003790 | mostly neutral |
| cognitive max0.125 | 2027 | 0.771382 | -0.001300 | +0.001751 | +0.000581 | +0.000493 | +0.002921 | weak mixed |
| cognitive max0.1875 | 2027 | 0.767639 | -0.005042 | -0.001351 | +0.001647 | +0.001401 | -0.005849 | rejected tail |
| final_logit max0.75 | 2024 | 0.770656 | +0.003177 | -0.003007 | -0.000374 | -0.000318 | +0.002756 | strong but biased |
| final_logit max0.75 | 2025 | 0.770322 | -0.001338 | -0.002188 | +0.000959 | +0.000816 | +0.003278 | rejected |
| single-only max0.25 | 2024 | 0.773168 | +0.005690 | -0.003026 | -0.000549 | -0.000467 | +0.006963 | strong but biased |
| single-only max0.25 | 2025 | 0.770126 | -0.001535 | +0.001085 | +0.000007 | +0.000006 | -0.002555 | rejected |
| eval-only max0.25 | 2024 | 0.765730 | -0.001748 | -0.001579 | +0.000372 | +0.000317 | -0.005038 | rejected |
| max0.25 + cognitive_scale0.9 | 2024 | 0.773135 | +0.005657 | -0.001694 | -0.001373 | -0.001167 | +0.002559 | best seed2024 balance |
| max0.25 + cognitive_scale0.9 | 2025 | 0.772346 | +0.000685 | +0.001465 | -0.001161 | -0.000984 | -0.003830 | positive but smaller |
| max0.25 + cognitive_scale0.9 | 2026 | 0.767036 | -0.004227 | +0.000247 | +0.000509 | +0.000433 | -0.005142 | rejected |
| max0.25 + cognitive_scale0.9 | 2027 | 0.760422 | -0.012260 | -0.002912 | +0.004596 | +0.003921 | -0.006312 | rejected tail |
| cognitive_scale0.9 only | 2024 | 0.767066 | -0.000412 | -0.001085 | -0.000397 | -0.000337 | -0.009406 | calibration only |
| cognitive_scale0.9 only | 2026 | 0.770409 | -0.000854 | -0.000247 | -0.000389 | -0.000330 | -0.006772 | calibration only |

Result paths live under `results/expert_output_modulation/`, using the config names in the table.

## Conclusion

- Reject this branch as both a default promotion and a trial candidate.
- Deterministic evidence prior still contains real ranking signal: seed2024 repeatedly reaches about `AUC +0.0056`, and seed2025 can be positive under full-scope max0.25.
- The signal is not robust: seed2026 rejects train-time prior variants, and seed2027 exposes a much larger negative tail for `max_logit=0.25` (`AUC -0.011524`) and for `max_logit=0.25 + cognitive_scale0.9` (`AUC -0.012260`).
- The usable prior cap boundary is below `0.1875`: seed2027 at `max_logit=0.1875` still loses `AUC -0.005042`, while `max_logit=0.125` is only weak/mixed and not a promotion signal.
- The likely failure is not just multi-concept scope, final-position placement, or overconfidence. Single-only, final-logit, eval-only, and global cognitive scaling each fail at least one critical seed.
- Do not continue micro-tuning `concept_evidence_prior_*` on exp81 unless a new mechanism explains seed2026 specifically.
- Follow-up note: experiment 85 tried the immediate state/qrepr formation variants from this conclusion; state rewrite and target evidence attention qrepr were also rejected. Future work should use a materially different expert architecture or training objective, not the same target-local evidence signal moved to another residual surface.
