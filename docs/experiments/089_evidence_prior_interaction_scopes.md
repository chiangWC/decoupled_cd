# Experiment 89: evidence prior interaction and direction scopes

## Verdict

Agreement gating, train-only application, and one-sided prior direction scopes do not improve on experiment 88's best stabilized gate. The only new seed2024 `+0.005`-level row, agreement margin `1.0`, worsens ACC/RMSE/Brier/ECE and does not improve seed2027. Train-only and positive/negative-only scopes fail the seed2024 admission threshold.

## Context

Experiment 88 found the best stabilized deterministic prior so far:

- `min_confidence=0.75`
- `min_abs_mastery=0.5`
- `max_logit=0.25`
- all-mode, both prior directions

Reference rows:

- Seed2024 exp81 pseudo-mainline: `test_auc = 0.767478`.
- Seed2027 exp81 pseudo-mainline inferred from experiment 84 raw row: about `0.772681`.
- Experiment 88 best stabilized seed2024: `test_auc = 0.772527`, delta `+0.005049`.
- Experiment 88 best stabilized seed2027: `test_auc = 0.771792`, delta about `-0.000889`.

## Branch

- Branch: `exp/evidence-prior-stability-gate`
- Code commits:
  - `220b2f0 feat: gate evidence prior by model agreement`
  - `7c9967e feat: add train-only evidence prior mode`
  - `7880daf feat: scope evidence prior by direction`

## Engineering Validation

- After agreement gate: remote `python -m unittest tests.test_decoupled_cdm tests.test_training_modes tests.test_history_visibility` passed, 28 tests.
- After train-only mode: same remote suite passed, 29 tests.
- After direction scope: same remote suite passed, 30 tests.
- Smoke for agreement gate passed with 1 epoch and 2000-row split cap.

## Results

All rows use the experiment 88 stabilized base unless noted:

```bash
--concept-evidence-prior-residual
--concept-evidence-prior-min-count 1
--concept-evidence-prior-min-seen-ratio 1.0
--concept-evidence-prior-max-logit 0.25
--concept-evidence-prior-min-confidence 0.75
--concept-evidence-prior-min-abs-mastery 0.5
```

| variant | seed | result path | test AUC | delta AUC | test ACC | RMSE | Brier | ECE | verdict |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| agreement margin0.5 | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_agree_margin05_300ep.json` | 0.772134 | +0.004656 | 0.732402 | 0.424857 | 0.180503 | 0.052267 | misses seed threshold |
| agreement margin1.0 | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_agree_margin10_300ep.json` | 0.773044 | +0.005566 | 0.728006 | 0.427045 | 0.182368 | 0.062676 | AUC only, error/calibration bad |
| agreement margin1.0 | 2027 | `results/evidence_prior_stability_gate/assist09_seed2027_agree_margin10_300ep.json` | 0.771672 | -0.001009 | 0.729966 | 0.425748 | 0.181262 | 0.055085 | worse than experiment 88 best gate |
| train_only | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_trainonly_conf075_abs05_max025_300ep.json` | 0.771886 | +0.004408 | 0.731203 | 0.424755 | 0.180417 | 0.050230 | misses seed threshold |
| positive direction only | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_positiveonly_conf075_abs05_max025_300ep.json` | 0.771687 | +0.004209 | 0.731508 | 0.425202 | 0.180797 | 0.052150 | misses seed threshold |
| negative direction only | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_negativeonly_conf075_abs05_max025_300ep.json` | 0.771693 | +0.004215 | 0.732745 | 0.424852 | 0.180499 | 0.050728 | misses seed threshold |

## Decision

- Do not promote agreement gating, train-only prior mode, or direction-only prior scopes.
- Agreement margin `1.0` reaches seed2024 AUC threshold but is a worse tradeoff than experiment 88: seed2027 does not improve, and seed2024 error/calibration regress sharply.
- Train-only confirms that the deterministic prior's seed2024 lift mostly requires inference-time residual application; using it only as a training perturbation is insufficient.
- Positive-only and negative-only both miss seed2024 threshold, so the signal is not carried cleanly by a single prior direction.
- Further work should stop adding deterministic prior masks and instead change the representation or objective that consumes student-concept evidence.
