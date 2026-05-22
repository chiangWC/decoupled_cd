# Experiment 106: Dual CDM branch BCE refinement

## Status

`single_checkpoint_branch_bce_refinement_promoted`, current best pure-CDM single-run/single-checkpoint default-training candidate.

## Verdict

Refining experiment 104's dual-tower branch BCE from `0.10` to `0.18` improves the four-seed AUC mean while preserving the accepted pure-CDM/default-training constraints.

Best configuration:

```bash
bash scripts/run_assist09_history_alignment_trial.sh \
  --dual-cdm-branch-bce-weight 0.18
```

After this experiment, `scripts/run_assist09_history_alignment_trial.sh` encodes `0.18` by default.

Four-seed AUC is `0.778890/0.778552/0.778256/0.778618`, mean `0.778579`, population stdev `0.000226`, and mean ECE `0.034913`. This is higher and more stable than experiment 104's branch BCE `0.10` mean `0.778370` and stdev `0.000306`.

No non-pure-CDM strategy, validation-trained combiner, hybrid feature path, or multi-checkpoint average was used. Each row is one training run and one validation-selected checkpoint.

## Base

- Branch: `exp/lightweight-single-tower-cdm`
- Runner: `scripts/run_assist09_history_alignment_trial.sh`
- Base candidate: experiment 104 dual CDM ensemble, primary `concept_dim=64`, secondary `concept_dim=80`, branch BCE `0.10`
- Reference four-seed AUC for experiment 104: `0.778773/0.778250/0.778508/0.777948`, mean `0.778370`
- Lightweight reference from experiment 105: exp104 pre-dual single64 mean `0.776736`, seed2024 max CUDA `6.33GB`

## Results

| branch BCE | seeds | AUCs | mean AUC | stdev | min | max | verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 0.05 | 2024/2027 | `0.778412/0.777724` | 0.778068 | 0.000344 | 0.777724 | 0.778412 | rejected, below exp104 on matched seeds |
| 0.15 | 2024/2025/2026/2027 | `0.779046/0.778553/0.778348/0.778106` | 0.778514 | 0.000346 | 0.778106 | 0.779046 | highest peak, mean positive |
| 0.18 | 2024/2025/2026/2027 | `0.778890/0.778552/0.778256/0.778618` | 0.778579 | 0.000226 | 0.778256 | 0.778890 | promoted default |
| 0.20 | 2024/2025/2026/2027 | `0.778955/0.778645/0.778055/0.778590` | 0.778561 | 0.000324 | 0.778055 | 0.778955 | second-best mean, weaker tail |

Detailed promoted rows:

| seed | AUC | ACC | RMSE | Brier | ECE | best epoch | max CUDA GB | result file |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2024 | 0.778890 | 0.736589 | 0.419767 | 0.176205 | 0.033601 | 156 | 11.756714 | `results/pure_cdm_default_promotion/seed2024_exp106_dual64x80_branchbce018_late170to230_eprior_trainonly_300ep.json` |
| 2025 | 0.778552 | 0.737369 | 0.420074 | 0.176462 | 0.036112 | 165 | 11.756714 | `results/pure_cdm_default_promotion/seed2025_exp106_dual64x80_branchbce018_late170to230_eprior_trainonly_300ep.json` |
| 2026 | 0.778256 | 0.737521 | 0.420099 | 0.176483 | 0.035918 | 158 | 11.756714 | `results/pure_cdm_default_promotion/seed2026_exp106_dual64x80_branchbce018_late170to230_eprior_trainonly_300ep.json` |
| 2027 | 0.778618 | 0.736779 | 0.419934 | 0.176345 | 0.034021 | 147 | 11.756714 | `results/pure_cdm_default_promotion/seed2027_exp106_dual64x80_branchbce018_late170to230_eprior_trainonly_300ep.json` |

Aggregate comparison:

| route | mean AUC | stdev | min AUC | max AUC | mean ACC | mean RMSE | mean Brier | mean ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp104 branch BCE 0.10 | 0.778370 | 0.000306 | 0.777948 | 0.778773 | n/a | n/a | n/a | 0.035412 |
| exp106 branch BCE 0.15 | 0.778514 | 0.000346 | 0.778106 | 0.779046 | 0.736907 | 0.419935 | 0.176345 | 0.034759 |
| exp106 branch BCE 0.18 | 0.778579 | 0.000226 | 0.778256 | 0.778890 | 0.737064 | 0.419969 | 0.176374 | 0.034913 |
| exp106 branch BCE 0.20 | 0.778561 | 0.000324 | 0.778055 | 0.778955 | 0.736689 | 0.419990 | 0.176392 | 0.034222 |

## Decision

Promote `dual_cdm_branch_bce_weight=0.18` as the default for `scripts/run_assist09_history_alignment_trial.sh`.

Use `0.15` only as a high-peak reference if a future search explicitly optimizes peak AUC. Use `0.20` as a nearby mean-positive reference, but not as default because it weakens the seed2026 tail.

This does not solve the user's earlier memory concern: the promoted route remains a dual tower and records `11.756714GB` CUDA peak allocation. Experiment 105's single64 route remains the lightweight fallback at `6.33GB`, but it is not competitive on four-seed mean.

## Validation Path

- Remote four-seed sweeps ran through `bash scripts/remote_exec.sh`.
- Result JSON files were written under `results/pure_cdm_default_promotion/`.
- Local runner syntax and focused validation are tracked with the promotion commit.
