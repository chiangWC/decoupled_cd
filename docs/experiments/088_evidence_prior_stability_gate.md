# Experiment 88: evidence prior stability gates

## Verdict

High-confidence, high-mastery gates stabilize the deterministic student-concept evidence prior enough to preserve the seed2024 `+0.005`-level AUC signal and largely remove the known seed2027 raw negative tail. They do not create a clean multi-seed promotion candidate: seed2026/seed2027 remain slightly below matched exp81 references, and the three-seed mean AUC delta is only about `+0.000826`.

## Context

Experiment 87 reconfirmed the current strongest overall growth signal:

- Base: current `exp/trellis-trial` pseudo-mainline, experiment 81.
- Raw prior seed2024: `test_auc = 0.773261`, matched delta `+0.005783`.
- Known failure: experiment 84 already showed the same prior family has seed2026/seed2027 negative tails, including seed2027 `AUC -0.011524` at `max_logit=0.25`.

This experiment tested a materially different stabilization mechanism instead of continuing raw `max_logit` micro-tuning.

## Branch

- Branch: `exp/evidence-prior-stability-gate`
- Code commits:
  - `2f7b977 feat: gate concept evidence prior stability`
  - `5f3b7a6 feat: add eval-only evidence prior mode`
  - `7a83ee3 feat: scope evidence prior by concept count`

## Implementation

Added optional controls for `concept_evidence_prior_residual`:

- `concept_evidence_prior_min_confidence`: require enough target-local train-history evidence mass.
- `concept_evidence_prior_min_abs_mastery`: require a strong signed mastery margin.
- `concept_evidence_prior_apply_mode`: `all` or `eval_only`.
- `concept_evidence_prior_max_count`: optional upper bound for concept-count scope; `0` disables the upper bound.

Best stabilized command shape:

```bash
bash scripts/run_assist09_baseline.sh \
  --concept-evidence-prior-residual \
  --concept-evidence-prior-min-count 1 \
  --concept-evidence-prior-min-seen-ratio 1.0 \
  --concept-evidence-prior-max-logit 0.25 \
  --concept-evidence-prior-min-confidence 0.75 \
  --concept-evidence-prior-min-abs-mastery 0.5 \
  --output results/evidence_prior_stability_gate/assist09_seed2024_conf075_abs05_max025_300ep.json
```

## Engineering Validation

- Remote unit tests:
  - `python -m unittest tests.test_decoupled_cdm tests.test_training_modes tests.test_history_visibility`
  - Result: passed, 27 tests after final code change.
- Remote smoke:
  - `bash scripts/run_assist09_baseline.sh --epochs 1 --max-rows 2000 --concept-evidence-prior-residual --concept-evidence-prior-min-count 1 --concept-evidence-prior-min-seen-ratio 1.0 --concept-evidence-prior-max-logit 0.25 --concept-evidence-prior-min-confidence 0.25 --concept-evidence-prior-min-abs-mastery 0.25`
  - Result: passed.

## Results

Matched references:

- Seed2024 exp81 pseudo-mainline: `test_auc = 0.767478`.
- Seed2026 exp81 pseudo-mainline inferred from experiment 84 raw row: about `0.771263`.
- Seed2027 exp81 pseudo-mainline inferred from experiment 84 raw row: about `0.772681`.

| config | seed | result path | test AUC | delta AUC | verdict |
| --- | ---: | --- | ---: | ---: | --- |
| raw prior max0.25 | 2024 | `results/evidence_prior_cognitive_max025_seed2024.json` | 0.773261 | +0.005783 | seed signal confirmed |
| raw prior max0.25 | 2026 | experiment 84 | 0.767959 | -0.003304 | negative tail |
| raw prior max0.25 | 2027 | experiment 84 | 0.761157 | -0.011524 | unacceptable tail |
| conf0.25 abs0.25 max0.25 | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_conf025_abs025_max025_300ep.json` | 0.773478 | +0.006000 | preserves seed signal |
| conf0.25 abs0.25 max0.25 | 2027 | `results/evidence_prior_stability_gate/assist09_seed2027_conf025_abs025_max025_300ep.json` | 0.762499 | -0.010182 | too weak |
| conf0.50 abs0.50 max0.25 | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_conf05_abs05_max025_300ep.json` | 0.772934 | +0.005456 | preserves seed signal |
| conf0.50 abs0.50 max0.25 | 2027 | `results/evidence_prior_stability_gate/assist09_seed2027_conf05_abs05_max025_300ep.json` | 0.769286 | -0.003395 | tail reduced but still material |
| conf0.75 abs0.50 max0.25 | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_conf075_abs05_max025_300ep.json` | 0.772527 | +0.005049 | best stabilized seed signal |
| conf0.75 abs0.50 max0.25 | 2026 | `results/evidence_prior_stability_gate/assist09_seed2026_conf075_abs05_max025_300ep.json` | 0.769580 | -0.001683 | small negative |
| conf0.75 abs0.50 max0.25 | 2027 | `results/evidence_prior_stability_gate/assist09_seed2027_conf075_abs05_max025_300ep.json` | 0.771792 | -0.000889 | near neutral |
| conf0.75 abs0.50 max0.25 eval_only | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_evalonly_conf075_abs05_max025_300ep.json` | 0.766520 | -0.000958 | reject; training-time prior needed |
| conf0.75 abs0.50 max0.25 max_count1 | 2024 | `results/evidence_prior_stability_gate/assist09_seed2024_single_conf075_abs05_max025_300ep.json` | 0.772510 | +0.005032 | no improvement over full gate |
| conf0.75 abs0.50 max0.25 max_count1 | 2027 | `results/evidence_prior_stability_gate/assist09_seed2027_single_conf075_abs05_max025_300ep.json` | 0.771766 | -0.000915 | no improvement over full gate |

Best stabilized row details:

| seed | test ACC | RMSE | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| 2024 | 0.732421 | 0.424527 | 0.180223 | 0.051386 |
| 2026 | 0.732250 | 0.424567 | 0.180257 | 0.041923 |
| 2027 | 0.734115 | 0.424527 | 0.180223 | 0.051623 |

Best stabilized three-seed AUC mean delta:

```text
(+0.005049 - 0.001683 - 0.000889) / 3 = +0.000826
```

## Decision

- Do not promote this branch into `exp/trellis-trial`.
- Keep `conf0.75 abs0.50 max0.25 all-mode` as the strongest stabilized diagnostic for future reference.
- Do not continue nearby threshold or `max_logit` micro-sweeps; the best gate already shows the shape of the tradeoff.
- The next useful evidence-prior attempt needs a new interaction with training or reliability estimation, not another deterministic residual scope tweak. Eval-only failed, and single-concept scoping did not improve the stabilized gate.
