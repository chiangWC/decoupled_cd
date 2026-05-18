# Experiment 98: Pure CDM Runner Integration

## Status

`pure_cdm_runner_integrated`, no new promotion beyond experiment 95.

## Verdict

The pure-CDM integration work is complete, but the new follow-up mechanisms do not improve on the experiment 95 `cogonly loss_only` runner.

The only seed2027 point above experiment 95 was reliability-weighted alignment with `confidence_power=0.5`, `confidence_floor=0.2`, and `alignment=0.05`, at `test_auc = 0.776264`. That is only `+0.000101` over experiment 95 seed2027 `0.776163` and remains below the corrected high-water threshold `0.776682`. When expanded to four seeds, the same reliability setting regressed seeds 2024 and 2025 and collapsed seed2026 to `test_auc = 0.504202`.

Keep experiment 95 as the active pure-CDM trial candidate. Do not promote the reliability-weighted, cog-only fusion, or rank-alignment variants.

## Base And References

Branch: `exp/pure-cdm-runner-integration`

Pure-CDM reference:

- experiment 95 runner: `scripts/run_assist09_history_alignment_trial.sh`
- seed2027 reference: `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_w005_300ep.json`
- seed2027 reference AUC: `0.776163`
- four-seed mean AUC: `0.776279`

Corrected high-water stop threshold:

- high-water seed2027: `0.7726816125195809`
- `+0.004` threshold: `0.776682`

Parent details:

- [095 history alignment CF-risk ablation](./095_history_alignment_cf_risk_ablation.md)
- [097 autonomous growth signal exploration](./097_autonomous_growth_signal_exploration.md)

## Implementation

Added or migrated three pure-CDM follow-up paths:

- Cog-only trainable history-evidence fusion readout:
  - `--history-evidence-fusion-feature-set cogonly`
  - runner: `scripts/run_assist09_history_cogonly_fusion_trial.sh`
- Pairwise rank-alignment loss over the same loss-only history evidence target:
  - `--history-evidence-cognitive-rank-alignment-weight`
  - runner: `scripts/run_assist09_history_rank_alignment_trial.sh`
- Reliability-weighted standardized MSE alignment:
  - `--history-evidence-cognitive-alignment-confidence-power`
  - `--history-evidence-cognitive-alignment-confidence-cap`
  - `--history-evidence-cognitive-alignment-confidence-floor`
  - runner: `scripts/run_assist09_history_alignment_reliability_trial.sh`

Remote verification:

```bash
python -m py_compile models/decoupled_cdm.py trainers/engine.py scripts/train.py
python -m unittest tests.test_training_modes tests.test_decoupled_cdm
bash scripts/run_assist09_history_cogonly_fusion_trial.sh --epochs 1 --max-rows 2000 --output results/pure_cdm_runner/smoke_history_cogonly_fusion_seed2027.json
bash scripts/run_assist09_history_rank_alignment_trial.sh --epochs 1 --max-rows 2000 --output results/pure_cdm_runner/smoke_history_rank_alignment_seed2027.json
bash scripts/run_assist09_history_alignment_reliability_trial.sh --epochs 1 --max-rows 2000 --output results/pure_cdm_runner/smoke_history_alignment_reliability_seed2027.json
```

## Seed2027 Probes

All runs use the experiment 95 `cogonly loss_only` target family unless noted.

| variant | output | test_auc | vs exp95 seed2027 | ACC | RMSE | Brier | ECE | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cog-only fusion, max logit `0.35` | `results/pure_cdm_runner/seed2027_history_alignment_cogonly_fusion_300ep.json` | 0.772717 | -0.003446 | 0.736303 | 0.423578 | 0.179418 | 0.052337 | rejected |
| cog-only fusion, max logit `0.10` | `results/pure_cdm_runner/seed2027_history_alignment_cogonly_fusion_max010_300ep.json` | 0.775467 | -0.000696 | 0.735847 | 0.422724 | 0.178695 | 0.048269 | below exp95 |
| cog-only fusion, max logit `0.05` | `results/pure_cdm_runner/seed2027_history_alignment_cogonly_fusion_max005_300ep.json` | 0.775913 | -0.000250 | 0.736417 | 0.422397 | 0.178419 | 0.047084 | best fusion, still below exp95 |
| cog-only fusion, max logit `0.025` | `results/pure_cdm_runner/seed2027_history_alignment_cogonly_fusion_max0025_300ep.json` | 0.775817 | -0.000346 | 0.735428 | 0.423293 | 0.179177 | 0.050985 | below exp95 |
| rank alignment `w=0.02`, gap `0.05` | `results/pure_cdm_runner/seed2027_history_rank_alignment_w002_gap005_300ep.json` | 0.765055 | -0.011108 | 0.729985 | 0.428266 | 0.183412 | 0.050135 | rejected |
| rank alignment `w=0.002`, gap `0.05` | `results/pure_cdm_runner/seed2027_history_rank_alignment_w0002_gap005_300ep.json` | 0.772090 | -0.004073 | 0.732707 | 0.425024 | 0.180645 | 0.053101 | rejected |
| reliability `power=0.25`, floor `0.2`, `w=0.05` | `results/pure_cdm_runner/seed2027_history_alignment_confpow025_floor02_300ep.json` | 0.776128 | -0.000035 | 0.737464 | 0.422015 | 0.178097 | 0.045270 | near exp95, below |
| reliability `power=0.5`, floor `0.2`, `w=0.05` | `results/pure_cdm_runner/seed2027_history_alignment_confpow05_floor02_300ep.json` | 0.776264 | +0.000101 | 0.736703 | 0.421584 | 0.177733 | 0.042198 | tiny seed2027 positive, below threshold |
| reliability `power=0.5`, floor `0.2`, `w=0.035` | `results/pure_cdm_runner/seed2027_history_alignment_confpow05_floor02_w0035_300ep.json` | 0.775402 | -0.000761 | 0.736151 | 0.422867 | 0.178816 | 0.048319 | rejected |
| reliability `power=0.5`, floor `0.2`, `w=0.065` | `results/pure_cdm_runner/seed2027_history_alignment_confpow05_floor02_w0065_300ep.json` | 0.776018 | -0.000145 | 0.736760 | 0.421684 | 0.177817 | 0.041692 | below exp95 |
| reliability `power=0.5`, floor `0.1`, `w=0.05` | `results/pure_cdm_runner/seed2027_history_alignment_confpow05_floor01_w005_300ep.json` | 0.775927 | -0.000236 | 0.736151 | 0.422593 | 0.178585 | 0.047076 | below exp95 |
| reliability `power=0.5`, floor `0.3`, `w=0.05` | `results/pure_cdm_runner/seed2027_history_alignment_confpow05_floor03_w005_300ep.json` | 0.776016 | -0.000147 | 0.736322 | 0.422305 | 0.178342 | 0.046442 | below exp95 |

## Reliability Multi-Seed Check

The only seed2027-positive reliability setting was expanded to seeds 2024-2026.

| seed | exp95 cogonly auc | reliability auc | auc delta | ACC | RMSE | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.777843 | 0.777415 | -0.000428 | 0.735961 | 0.421489 | 0.177653 | 0.044151 |
| 2025 | 0.776440 | 0.776375 | -0.000065 | 0.736836 | 0.421604 | 0.177750 | 0.045418 |
| 2026 | 0.774669 | 0.504202 | -0.270467 | 0.523226 | 0.514755 | 0.264972 | 0.161146 |
| 2027 | 0.776163 | 0.776264 | +0.000101 | 0.736703 | 0.421584 | 0.177733 | 0.042198 |

Current-branch sanity check:

- `scripts/run_assist09_history_alignment_trial.sh` re-run on seed2026 produced `test_auc = 0.774833` at `results/pure_cdm_runner/seed2026_history_alignment_cogonly_w005_recheck_300ep.json`.
- This confirms the seed2026 collapse is caused by reliability weighting, not by the pure-CDM runner migration.

## Decision

- Keep `scripts/run_assist09_history_alignment_trial.sh` as the active pure-CDM runner.
- Keep the new CLI options and focused tests because they preserve useful negative evidence and allow exact reproduction.
- Do not promote cog-only fusion, rank alignment, or reliability weighting.
- Do not continue local sweeps around reliability power/floor/weight without a new mechanism-level reason; the seed2026 collapse makes the risk clear.
