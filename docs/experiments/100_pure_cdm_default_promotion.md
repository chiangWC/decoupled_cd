# Experiment 100: Pure CDM Default Promotion Probes

## Status

`pure_cdm_growth_signal_no_default_promotion`

## Verdict

The pure-CDM runner path now has a real `0.778+` local signal, but it is not stable enough to promote as the default CDM runner.

The best single run is `concept_dim=80` plus weak history-evidence output alignment `w=0.004`, with seed2027 `test_auc = 0.778122`. The same setting also reaches seed2026 `test_auc = 0.778100`, and its four-seed mean is `0.777030`, above experiment 95's `0.776279`.

Do not change `scripts/run_assist09_baseline.sh` or treat this as default promotion. Seeds 2024 and 2025 regress versus experiment 95 for the dim80 output-alignment settings, and the mean gain is only about `+0.00075`. Keep experiment 95 as the active pure-CDM trial runner and keep the new output-alignment runner as an opt-in reproduction/probe path.

## Base And References

Branch: `exp/pure-cdm-default-promotion`

Pure-CDM reference:

- experiment 95 runner: `scripts/run_assist09_history_alignment_trial.sh`
- experiment 95 four-seed AUCs: `0.777843/0.776440/0.774669/0.776163`
- experiment 95 four-seed mean AUC: `0.776279`
- experiment 95 seed2027 summary: `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_w005_300ep.json`

Target:

- User corrected the practical target from `0.780` to `0.778`.
- Hybrid stacker evidence from experiment 99 is excluded from this acceptance path.

Parent details:

- [095 history alignment CF-risk ablation](./095_history_alignment_cf_risk_ablation.md)
- [098 pure CDM runner integration](./098_pure_cdm_runner_integration.md)
- [099 hybrid stacker multi-seed 0.778 validation](./099_hybrid_stacker_multiseed_078.md)

## Implementation

Added a pure-CDM train-time output-alignment objective against the same loss-only train-history evidence prior used by experiment 95:

- trainer flag: `history_evidence_output_alignment_weight`
- CLI flag: `--history-evidence-output-alignment-weight`
- runner: `scripts/run_assist09_history_output_alignment_trial.sh`
- tests: focused validation and loss-behavior coverage in `tests/test_training_modes.py`

Remote verification:

```bash
python -m py_compile trainers/engine.py scripts/train.py
python -m unittest tests.test_training_modes
SEED=2027 OUTPUT=results/pure_cdm_default_promotion/smoke_history_output_alignment_seed2027.json \
  bash scripts/run_assist09_history_output_alignment_trial.sh --epochs 1 --max-rows 2000
```

Result: focused tests passed remotely and the smoke run completed.

## Seed2027 Probe Summary

All rows are pure CDM: `history_evidence_logit_prior_location=loss_only`, no validation-trained combiner, no hybrid stacker.

| variant | output | test_auc | ACC | RMSE | Brier | ECE | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| exp95 reference | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_w005_300ep.json` | 0.776163 | 0.737902 | 0.421902 | 0.178001 | 0.044389 | reference |
| dim80 | `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim80_300ep.json` | 0.777759 | 0.737064 | 0.420866 | 0.177128 | 0.039439 | local signal |
| dim80 + output align `w=0.004` | `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim80_outalign0004_300ep.json` | 0.778122 | 0.735713 | 0.421772 | 0.177891 | 0.043800 | best single seed |
| dim80 + output align `w=0.0075` | `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim80_outalign00075_300ep.json` | 0.778018 | 0.736056 | 0.421380 | 0.177561 | 0.042061 | crosses target, less mean-stable |
| dim72 + output align `w=0.01` | `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim72_outalign001_300ep.json` | 0.777771 | 0.736855 | 0.421018 | 0.177256 | 0.041333 | below target |
| dim96 | `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim96_300ep.json` | 0.777069 | 0.732878 | 0.423344 | 0.179221 | 0.052261 | AUC-only, secondary poor |

## Multi-Seed Checks

### Dim80 + Output Alignment `w=0.004`

This is the strongest pure-CDM four-seed result from this task, but still not a default promotion.

Output pattern: `results/pure_cdm_default_promotion/seed{seed}_history_alignment_cogonly_dim80_outalign0004_300ep.json`

Expanded files: `results/pure_cdm_default_promotion/seed2024_history_alignment_cogonly_dim80_outalign0004_300ep.json`, `results/pure_cdm_default_promotion/seed2025_history_alignment_cogonly_dim80_outalign0004_300ep.json`, `results/pure_cdm_default_promotion/seed2026_history_alignment_cogonly_dim80_outalign0004_300ep.json`, `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim80_outalign0004_300ep.json`.

| seed | exp95 auc | candidate auc | auc delta | ACC | RMSE | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.777843 | 0.776419 | -0.001424 | 0.737673 | 0.421586 | 0.177734 | 0.045014 |
| 2025 | 0.776440 | 0.775481 | -0.000959 | 0.736379 | 0.421691 | 0.177823 | 0.039622 |
| 2026 | 0.774669 | 0.778100 | +0.003431 | 0.736817 | 0.420278 | 0.176633 | 0.035880 |
| 2027 | 0.776163 | 0.778122 | +0.001959 | 0.735713 | 0.421772 | 0.177891 | 0.043800 |
| mean | 0.776279 | 0.777030 | +0.000751 | 0.736646 | 0.421332 | 0.177521 | 0.041079 |

### Dim80 + Output Alignment `w=0.0075`

The slightly stronger-looking seed2027 setting does not fix the 2024/2025 tail.

Output pattern: `results/pure_cdm_default_promotion/seed{seed}_history_alignment_cogonly_dim80_outalign00075_300ep.json`

Expanded files: `results/pure_cdm_default_promotion/seed2024_history_alignment_cogonly_dim80_outalign00075_300ep.json`, `results/pure_cdm_default_promotion/seed2025_history_alignment_cogonly_dim80_outalign00075_300ep.json`, `results/pure_cdm_default_promotion/seed2026_history_alignment_cogonly_dim80_outalign00075_300ep.json`, `results/pure_cdm_default_promotion/seed2027_history_alignment_cogonly_dim80_outalign00075_300ep.json`.

| seed | exp95 auc | candidate auc | auc delta | ACC | RMSE | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.777843 | 0.776385 | -0.001458 | 0.738511 | 0.421553 | 0.177707 | 0.045732 |
| 2025 | 0.776440 | 0.775815 | -0.000625 | 0.736284 | 0.421907 | 0.178006 | 0.042660 |
| 2026 | 0.774669 | 0.777852 | +0.003183 | 0.737312 | 0.420394 | 0.176731 | 0.038639 |
| 2027 | 0.776163 | 0.778018 | +0.001855 | 0.736056 | 0.421380 | 0.177561 | 0.042061 |
| mean | 0.776279 | 0.777017 | +0.000739 | 0.737041 | 0.421308 | 0.177501 | 0.042273 |

### Other Four-Seed Checks

| variant | mean_auc | mean ACC | mean RMSE | mean Brier | mean ECE | verdict |
|---|---:|---:|---:|---:|---:|---|
| dim80, no output alignment | 0.776911 | 0.735704 | 0.421993 | 0.178079 | 0.044973 | weak AUC mean, secondary worse |
| dim72 + output align `w=0.01` | 0.776380 | 0.736370 | 0.421641 | 0.177782 | 0.042966 | near exp95, not meaningful growth |
| dim80 + output align `w=0.004` | 0.777030 | 0.736646 | 0.421332 | 0.177521 | 0.041079 | best mean, mixed seeds |
| dim80 + output align `w=0.0075` | 0.777017 | 0.737041 | 0.421308 | 0.177501 | 0.042273 | mixed seeds |

## Decision

- Keep `scripts/run_assist09_history_alignment_trial.sh` as the active pure-CDM trial runner.
- Keep `scripts/run_assist09_history_output_alignment_trial.sh` as an opt-in probe/reproduction runner.
- Do not change `scripts/run_assist09_baseline.sh`.
- Do not promote `concept_dim=80` or output alignment as the default CDM runner until a later mechanism removes the 2024/2025 regression.
- Treat the useful evidence as directional: increasing concept capacity to 80 plus a very weak output-alignment objective exposes 2026/2027 headroom near or above `0.778`, but this is not yet a stable default.
- Avoid continuing nearby output-alignment weight sweeps without a new mechanism-level reason; the four-seed checks show the main issue is seed stability, not exact local weight choice.
