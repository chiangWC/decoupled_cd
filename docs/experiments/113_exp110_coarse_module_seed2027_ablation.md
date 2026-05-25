# Experiment 113: exp110 coarse module seed2027 ablation

## Status

`coarse_module_single_seed_ablation_completed`

## Verdict

This run groups the many exp110 implementation switches into three conceptual
modules and performs a seed2027 one-factor removal against the exp110 baseline.
It is intended for paper-style reporting, not for hyperparameter search.

The strongest coarse contributor on seed2027 is `Evidence-aware readout`.
Removing it drops AUC by `-0.010232`, ACC by `-0.006508`, and raises RMSE by
`+0.005677`.

The other two coarse modules are also material:

- `Cognitive alignment objective`: AUC `-0.005407`
- `Dual-branch ensemble`: AUC `-0.004202`

No checkpoint inference average, hybrid stacker, validation-trained combiner,
or train-history tabular inference side channel is used.

## Base

- Branch: `exp/trellis-trial`
- Seed: `2027`
- Baseline result: `results/pure_cdm_exp110_ablation/seed2027_baseline_reproduce.json`
- New result directory: `results/pure_cdm_exp110_coarse_ablation_seed2027/`
- Reused reference: `results/pure_cdm_exp110_ablation/seed2027_a_no_dual_tower.json`

## Module Definitions

### Evidence-aware readout

Removed together:

- high-concept logit adapter
- pairwise history interaction adapter
- interpretable readout expert adapter
- student-conditioned UKC readout residual
- concept evidence readout residual

### Cognitive alignment objective

Removed together:

- history evidence logit prior residual and related loss-only prior weights
- history evidence cognitive alignment weight and late schedule
- concept evidence prior residual and related train-only prior flags

### Dual-branch ensemble

Removed together:

- dual CDM ensemble
- secondary tower
- branch BCE

This row reuses the existing experiment 111 seed2027 `a_no_dual_tower` output,
which is exactly this grouped removal.

## Seed2027 Result

`delta` columns are paired against the same exp110 `seed=2027` baseline.

| variant | AUC | delta AUC | ACC | delta ACC | RMSE | delta RMSE | best epoch | peak CUDA GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline_exp110` | 0.7793212012 | +0.0000000000 | 0.7379969172 | +0.0000000000 | 0.4188889911 | +0.0000000000 | 149 | 6.190541 |
| `without_dual_branch_ensemble` | 0.7751189507 | -0.0042022505 | 0.7389864698 | +0.0009895526 | 0.4206905895 | +0.0018015984 | 200 | 3.346300 |
| `without_cognitive_alignment_objective` | 0.7739138046 | -0.0054073966 | 0.7375782603 | -0.0004186569 | 0.4211626983 | +0.0022737072 | 150 | 6.190754 |
| `without_evidence_aware_readout` | 0.7690890516 | -0.0102321496 | 0.7314887058 | -0.0065082114 | 0.4245657518 | +0.0056767607 | 167 | 4.123320 |

## Result Paths

- `baseline_exp110`: `results/pure_cdm_exp110_ablation/seed2027_baseline_reproduce.json`
- `without_evidence_aware_readout`: `results/pure_cdm_exp110_coarse_ablation_seed2027/seed2027_without_evidence_aware_readout.json`
- `without_cognitive_alignment_objective`: `results/pure_cdm_exp110_coarse_ablation_seed2027/seed2027_without_cognitive_alignment_objective.json`
- `without_dual_branch_ensemble`: `results/pure_cdm_exp110_ablation/seed2027_a_no_dual_tower.json`

## Decision

- For coarse paper-style ablation, use the three-module framing:
  `Evidence-aware readout`, `Cognitive alignment objective`, and
  `Dual-branch ensemble`.
- The fine-grained experiment 112 rows should be treated as subcomponent
  attribution under these modules.
- This is still single-seed evidence. Do not replace multi-seed conclusions with
  this table without running the same coarse groups across additional seeds.
