# Experiment 87: growth signal exploration

## Context

User requested free exploration until an overall `0.005`-level growth signal appears. The active base is current `exp/trellis-trial` pseudo-mainline: experiment 70 plus experiment 81 single-only concept-evidence readout.

Reference seed 2024 pseudo-mainline:

- `test_auc = 0.767478`
- `test_acc = 0.734248`
- `test_rmse = 0.425562`
- `test_brier = 0.181103`
- `test_ece = 0.046972`

## Branch

- Branch: `exp/hybrid-id-residual-growth`
- Code commit: `33e7da9 feat: add hybrid id residual probe`

## Engineering validation

- Remote unit tests:
  - `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - Result: passed, 34 tests.
- Remote smoke:
  - `bash scripts/run_assist09_baseline.sh --epochs 1 --max-rows 2000 --hybrid-id-residual --hybrid-id-residual-dim 16 --hybrid-id-residual-max-logit 1.0 --output results/hybrid_id_residual_smoke_seed2024.json`
  - Result: passed.

## Hybrid ID residual probe

Mechanism: optional zero-init transductive student/exercise ID residual on the cognitive readout. This is explicitly a hybrid side-channel probe, not a pure CDM component.

| config | result path | test AUC | delta AUC | test ACC | delta ACC | RMSE | Brier | ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dim32 max_logit1.0 | `results/hybrid_id_residual_dim32_logit1_seed2024.json` | 0.763414 | -0.004064 | 0.730461 | -0.003787 | 0.428240 | 0.183389 | 0.054255 | overfit / reject |
| dim8 max_logit0.25 | `results/hybrid_id_residual_dim8_logit025_seed2024.json` | 0.767696 | +0.000218 | 0.733468 | -0.000780 | 0.424966 | 0.180596 | 0.041460 | low-signal calibration cleanup |
| dim8 max_logit0.5 | `results/hybrid_id_residual_dim8_logit05_seed2024.json` | 0.766918 | -0.000560 | 0.733335 | -0.000913 | 0.425517 | 0.181065 | 0.044170 | reject |

Verdict: bounded hybrid ID capacity can clean up RMSE/Brier/ECE slightly, but it does not produce the requested ranking signal. Do not promote this probe.

## Deterministic evidence-prior confirmation

Mechanism: existing deterministic target-local student-concept evidence prior on the cognitive readout.

Command:

```bash
bash scripts/run_assist09_baseline.sh \
  --concept-evidence-prior-residual \
  --concept-evidence-prior-min-count 1 \
  --concept-evidence-prior-min-seen-ratio 1.0 \
  --concept-evidence-prior-max-logit 0.25 \
  --concept-evidence-prior-strength 2.0 \
  --concept-evidence-prior-confidence-cap 20.0 \
  --output results/evidence_prior_cognitive_max025_seed2024.json
```

Result:

| config | result path | test AUC | delta AUC | test ACC | delta ACC | RMSE | Brier | ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cognitive max0.25 | `results/evidence_prior_cognitive_max025_seed2024.json` | 0.773261 | +0.005783 | 0.731413 | -0.002835 | 0.424638 | 0.180318 | 0.052101 | target-level seed signal confirmed |

Slice report:

- `results/evidence_prior_cognitive_max025_seed2024_slices.json`
- `concept_count=1`: `AUC 0.777905`, `ACC 0.734521`, `RMSE 0.421490`, `ECE 0.047086`
- `concept_count=2`: `AUC 0.751718`, `ACC 0.719525`, `RMSE 0.436230`, `ECE 0.073929`
- `concept_count=3`: `AUC 0.717908`, `ACC 0.698782`, `RMSE 0.463113`, `ECE 0.105753`
- `concept_count=4+`: `AUC 0.749878`, `ACC 0.683196`, `RMSE 0.458829`, `ECE 0.086257`
- `all_seen`: `AUC 0.771397`, `ACC 0.728605`, `RMSE 0.426563`, `ECE 0.052229`
- `none_seen`: `AUC 0.806742`, `ACC 0.818456`, `RMSE 0.361630`, `ECE 0.055959`

Verdict: this confirms the requested `0.005`-level overall seed-2024 ranking signal on the current pseudo-mainline. It is not a promotion candidate by itself because experiment 84 already found unacceptable seed2026/seed2027 negative tails for the same family. Treat this as an admission signal: the student-concept evidence prior contains real ranking information, but the next useful work is stabilizing or re-parameterizing it, not merging the raw prior.

## Decision

- Reject hybrid ID residual as a mainline candidate.
- Keep deterministic evidence prior max0.25 as the strongest confirmed seed-2024 growth signal.
- Do not promote raw `concept_evidence_prior_residual` without a mechanism that addresses the known seed2026/seed2027 tail.
- Next attempt should explicitly target stability of the evidence-prior signal, for example by learning a conservative gate or uncertainty condition around the prior rather than only changing `max_logit`.
