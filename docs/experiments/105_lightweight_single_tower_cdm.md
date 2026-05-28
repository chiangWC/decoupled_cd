# Experiment 105: Lightweight single-tower CDM probes

## Status

`lightweight_single_tower_probe_rejected`, keep single64 as lightweight reference only.

## Verdict

The exp104 pre-dual single-tower base is already the best lightweight reference: seed2024 reaches `test_auc 0.778379` with `max_cuda_memory_allocated_gb 6.33`, and the historical four-seed AUC is `0.778379/0.776930/0.775065/0.776569`, mean `0.776736`.

That satisfies the user's VRAM target, but it does not match exp104's four-seed mean `0.778370`, and it is weaker than the later exp110 default route. The new shared-branch single-tower probes and the `concept_dim=72` capacity probe both stay under `7GB`, but both reduce AUC. They should not replace exp110 as the default route.

No non-pure-CDM strategy, validation-trained combiner, hybrid feature stacker, or multi-checkpoint average was used in this experiment.

## Base

- Branch: `exp/lightweight-single-tower-cdm`
- Reference route: experiment 104 pre-dual single-tower base
- Baseline protocol: experiment 103 late-window cognitive alignment plus train-only concept evidence prior
- Main comparison target: experiment 104 dual CDM ensemble, four-seed mean `0.778370`
- Memory metric: `max_cuda_memory_allocated_gb` recorded from CUDA peak allocated memory inside `scripts/train.py`

## Model And Training Changes

- Added optional `shared_branch_ensemble` to `DecoupledCDM`.
- Shared-branch mode keeps one backbone and one propagation path, then adds a secondary cognitive/guess/slip readout branch.
- Added `shared_branch_secondary_weight` so inference can use the primary branch alone, equal branch averaging, or a small secondary contribution without retraining.
- Added focused runner `scripts/run_assist09_lightweight_single_tower_trial.sh`.
- Added CUDA peak memory tracking to train output JSON and CSV summaries.

The intended hypothesis was that a shared-backbone auxiliary branch might retain some of exp104's two-view supervision benefit without paying the full two-tower memory cost.

## Results

| variant | seed | AUC | ACC | RMSE | Brier | ECE | best epoch | max CUDA GB | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| single64 exp104 pre-dual base | 2024 | 0.778379 | 0.735485 | 0.421299 | 0.177493 | 0.042829 | 212 | 6.329924 | best lightweight reference |
| shared branch, secondary weight 0.5, branch BCE 0.10 | 2024 | 0.776132 | 0.735466 | 0.421793 | 0.177909 | 0.041239 | 198 | 6.614818 | rejected |
| shared branch, secondary weight 0.0, branch BCE 0.10 | 2024 | 0.776504 | 0.735314 | 0.421847 | 0.177955 | 0.044008 | 206 | 6.614818 | rejected |
| shared branch, secondary weight 0.1, branch BCE 0.10 | 2024 | 0.776189 | 0.736303 | 0.421996 | 0.178080 | 0.043918 | 209 | 6.614818 | rejected |
| single72 exp104 pre-dual protocol | 2024 | 0.776216 | 0.735885 | 0.421713 | 0.177842 | 0.042131 | 189 | 6.833642 | rejected |
| single72 exp104 pre-dual protocol | 2026 | 0.775486 | 0.734610 | 0.422382 | 0.178407 | 0.043502 | 206 | 6.833642 | rejected |

Result files:

- `results/pure_cdm_default_promotion/seed2024_single64_late170to230_eprior_trainonly_memtrack_300ep.json`
- `results/pure_cdm_default_promotion/seed2024_sharedbranch64_branchbce010_late170to230_eprior_trainonly_300ep.json`
- `results/pure_cdm_default_promotion/seed2024_sharedbranch64_w000_branchbce010_late170to230_eprior_trainonly_300ep.json`
- `results/pure_cdm_default_promotion/seed2024_sharedbranch64_w010_branchbce010_late170to230_eprior_trainonly_300ep.json`
- `results/pure_cdm_default_promotion/seed2024_single72_late170to230_eprior_trainonly_memtrack_300ep.json`
- `results/pure_cdm_default_promotion/seed2026_single72_late170to230_eprior_trainonly_memtrack_300ep.json`

## Comparison To Heavy And Low-Memory References

| route | seeds | AUCs | mean AUC | max CUDA GB observed | decision |
|---|---|---:|---:|---:|---|
| exp104 dual CDM ensemble | 2024/2025/2026/2027 | `0.778773/0.778250/0.778508/0.777948` | 0.778370 | not remeasured here | historical heavy trial |
| exp110 recompute-minibatch dual CDM | 2024/2025/2026/2027 | `0.778252/0.778263/0.777449/0.779321` | 0.778321 | 6.190541 | current active trial |
| exp104 pre-dual single64 base | 2024/2025/2026/2027 | `0.778379/0.776930/0.775065/0.776569` | 0.776736 | 6.329924 on seed2024 | lightweight reference only |
| shared-branch best single seed | 2024 | `0.776504` | n/a | 6.614818 | rejected |
| single72 sampled seeds | 2024/2026 | `0.776216/0.775486` | 0.775851 | 6.833642 | rejected |

The shared-branch mode did not recover the exp104 dual-tower gain. Even when inference ignores the secondary branch (`secondary_weight=0.0`), the auxiliary branch training path lowers the seed2024 AUC versus the clean single64 base.

The `concept_dim=72` probe confirms that slightly larger single-tower capacity remains inside the memory budget, but it weakens both tested seeds. This is consistent with earlier experiment 103 capacity sweeps where dim72/80 did not form a better default candidate.

## Validation Path

- Remote smoke ran `py_compile`, focused `unittest`, and one epoch of `scripts/run_assist09_lightweight_single_tower_trial.sh`.
- Remote focused tests passed after adding `shared_branch_secondary_weight`.
- Full probes were run remotely on CUDA and read from result JSON files.

## Decision

Do not replace the current default route with shared-branch single-tower or dim72 single-tower.

Keep the exp104 pre-dual single64 base as a lightweight reference: it is clean, pure CDM, single checkpoint, and measured at `6.33GB` on seed2024. However, for the current `0.778+` default-training target, experiment 110 is stronger because it keeps similar memory while restoring the four-seed mean above `0.778`.
