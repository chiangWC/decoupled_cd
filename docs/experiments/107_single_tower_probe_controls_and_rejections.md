# Experiment 107: Single-tower probe controls and rejections

## Status

`single_tower_followup_rejected`, keep the exp105 single64 route as lightweight reference only.

## Verdict

Restoring the single-tower alignment probe CLI made it possible to re-test two low-cost
follow-up families on the exp105 single64 lightweight base without returning to dual
tower or multi-checkpoint routes.

Neither family produced a credible replacement path:

- `concept_dim=64 + history_evidence_output_alignment_weight=0.002` is negative on the weakest checked seed.
- `checkpoint_selection_window=3` is nearly flat on the weak seed and slightly negative on the strong seed.

Keep the exp105 single64 route as the lightweight pure-CDM reference:

- seed2024 `test_auc = 0.778379`
- four-seed historical mean `0.776736`
- `max_cuda_memory_allocated_gb ~= 6.33`

Do not continue dim64 output-alignment micro-sweeps or checkpoint-selection smoothing as
the next single-tower default-route candidate.

## Base

- Branch: `exp/lightweight-single-tower-cdm`
- Lightweight reference: experiment 105 pre-dual single64 base
- Reference metrics:
  - seed2024 `test_auc = 0.778379`, `max_cuda_memory_allocated_gb = 6.329924`
  - seed2026 `test_auc = 0.775065`
- Upper pure-CDM comparison target: experiment 106 dual-tower mean `0.778579`
- Constraints preserved:
  - pure CDM only
  - one training run and one validation-selected checkpoint per result
  - no `--dual-cdm-ensemble`
  - no valid-trained combiner
  - no hybrid tabular side-channel
  - no multi-checkpoint/model average

## Code Path Change

Restored the existing trainer-side probe controls through `scripts/train.py` so
single-tower experiments can again exercise:

- cognitive alignment confidence weighting
- cognitive alignment residual-focused weighting
- cognitive rank alignment
- output alignment

These mechanisms already existed in `trainers/engine.py`; this experiment only re-exposed
them through the CLI and summary payloads, then validated the path remotely.

## Validation Path

- Remote `py_compile` passed for:
  - `scripts/train.py`
  - `tests/test_train_cli.py`
  - `tests/test_training_modes.py`
- Remote `python3 -m unittest tests.test_train_cli tests.test_training_modes` passed.
- Remote smoke passed for:
  - `bash scripts/run_assist09_single_tower_history_alignment_trial.sh --epochs 1 --max-rows 2000 --history-evidence-output-alignment-weight 0.002`

## Results

| variant | seed | AUC | ACC | RMSE | Brier | ECE | best epoch | max CUDA GB | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| single64 base reference | 2026 | 0.775065 | n/a | n/a | n/a | n/a | n/a | 6.329924 | experiment 105 baseline |
| single64 + output alignment `0.002` | 2026 | 0.774925 | 0.736475 | 0.421705 | 0.177835 | 0.038327 | 201 | 6.332100 | rejected |
| single64 base reference | 2024 | 0.778379 | n/a | n/a | n/a | n/a | n/a | 6.329924 | experiment 105 baseline |
| single64 + checkpoint selection window `3` | 2024 | 0.778300 | 0.737902 | 0.420194 | 0.176563 | 0.038384 | 214 | 6.329924 | rejected |
| single64 + checkpoint selection window `3` | 2026 | 0.775082 | 0.736303 | 0.421650 | 0.177788 | 0.038801 | 204 | 6.329924 | too small to expand |

Result files:

- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_outalign0002_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single64_late170to230_eprior_trainonly_selwindow3_300ep.json`
- `results/pure_cdm_default_promotion/seed2024_single64_late170to230_eprior_trainonly_selwindow3_300ep.json`

Smoke artifact:

- `results/pure_cdm_default_promotion/smoke_single64_outalign002_seed2024.json`

## Comparison

| probe | checked seeds | delta vs matched single64 base | conclusion |
|---|---|---:|---|
| output alignment `0.002` | 2026 | `-0.000140` AUC | dim64 output alignment did not rescue the weak-seed tail |
| checkpoint selection window `3` | 2026 | `+0.000017` AUC | effectively flat on the weak seed |
| checkpoint selection window `3` | 2024 | `-0.000079` AUC | small regression on the strong seed |

## Decision

- Keep `scripts/run_assist09_single_tower_history_alignment_trial.sh` as the explicit non-dual single-tower command path.
- Keep the exp105 single64 route as the lightweight fallback/reference only.
- Do not expand dim64 `output_alignment=0.002`.
- Do not expand `checkpoint_selection_window=3`.
- Future single-tower work should move to a new mechanism-level idea, not more local smoothing or tiny output-alignment weight sweeps.
