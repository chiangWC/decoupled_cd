# Experiment 112: exp110 paper-style seed2027 component ablation

## Status

`single_seed_component_ablation_completed`

## Verdict

This is a paper-style one-factor component ablation for the experiment 110 pure-CDM
route, fixed to `seed=2027`. It excludes protocol sweeps, learning-rate sweeps,
capacity sweeps, checkpoint averaging, hybrid stackers, and validation-trained
combiners.

The largest seed2027 AUC drops come from removing:

- `concept_evidence_readout_residual`: `-0.005623`
- `history_evidence_cognitive_alignment`: `-0.005296`
- `student_conditioned_ukc_readout_residual`: `-0.004715`
- `dual_tower`: `-0.004202`
- `branch_bce`: `-0.002322`

Two controls are not clean positive contributors on this seed:

- `concept_prior` is AUC-neutral/slightly higher when removed, but ACC/RMSE worsen.
- `gs_difficulty_adapter` removal improves seed2027 AUC by `+0.000198`; treat this
  as a single-seed diagnostic, not a promotion signal.

## Base

- Branch: `exp/trellis-trial`
- Baseline: experiment 110 reproduce, `seed=2027`
- Baseline result: `results/pure_cdm_exp110_ablation/seed2027_baseline_reproduce.json`
- New result directory: `results/pure_cdm_exp110_paper_ablation_seed2027/`
- Reused references: experiment 111 `a_` outputs for exp110 incremental ablations

## Seed2027 Table

`delta` columns are paired against the same `seed=2027` baseline.

| variant | AUC | delta AUC | ACC | delta ACC | RMSE | delta RMSE | best epoch | peak CUDA GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_gs_difficulty_adapter` | 0.7795196951 | +0.0001984939 | 0.7383394546 | +0.0003425374 | 0.4187659663 | -0.0001230248 | 149 | 6.140486 |
| `no_concept_prior` | 0.7793381479 | +0.0000169467 | 0.7378066186 | -0.0001902986 | 0.4190944180 | +0.0002054269 | 149 | 6.190541 |
| `baseline_reproduce` | 0.7793212012 | +0.0000000000 | 0.7379969172 | +0.0000000000 | 0.4188889911 | +0.0000000000 | 149 | 6.190541 |
| `no_late_cog_align_anneal` | 0.7793212012 | +0.0000000000 | 0.7379969172 | +0.0000000000 | 0.4188889911 | +0.0000000000 | 149 | 6.190541 |
| `no_high_concept_logit_adapter` | 0.7793026670 | -0.0000185342 | 0.7375021409 | -0.0004947763 | 0.4189601949 | +0.0000712038 | 149 | 5.941499 |
| `no_interpretable_readout_expert_adapter` | 0.7787980045 | -0.0005231967 | 0.7374640811 | -0.0005328360 | 0.4195137090 | +0.0006247179 | 160 | 5.864804 |
| `no_pairwise_history_interaction_adapter` | 0.7779421947 | -0.0013790065 | 0.7380730366 | +0.0000761194 | 0.4196428355 | +0.0007538444 | 181 | 5.323529 |
| `no_branch_bce` | 0.7769995930 | -0.0023216082 | 0.7353707968 | -0.0026261204 | 0.4217000352 | +0.0028110441 | 160 | 6.190541 |
| `no_dual_tower` | 0.7751189507 | -0.0042022505 | 0.7389864698 | +0.0009895526 | 0.4206905895 | +0.0018015984 | 200 | 3.346300 |
| `no_student_conditioned_ukc_readout_residual` | 0.7746057959 | -0.0047154053 | 0.7378827380 | -0.0001141791 | 0.4206816530 | +0.0017926619 | 152 | 5.621492 |
| `no_cognitive_alignment` | 0.7740247342 | -0.0052964670 | 0.7372547527 | -0.0007421645 | 0.4212557468 | +0.0023667557 | 150 | 6.190754 |
| `no_concept_evidence_readout_residual` | 0.7736978077 | -0.0056233935 | 0.7337342290 | -0.0042626882 | 0.4219876995 | +0.0030987083 | 137 | 6.169054 |

## Result Paths

New outputs:

- `results/pure_cdm_exp110_paper_ablation_seed2027/seed2027_paper_no_high_concept_logit_adapter.json`
- `results/pure_cdm_exp110_paper_ablation_seed2027/seed2027_paper_no_pairwise_history_interaction_adapter.json`
- `results/pure_cdm_exp110_paper_ablation_seed2027/seed2027_paper_no_gs_difficulty_adapter.json`
- `results/pure_cdm_exp110_paper_ablation_seed2027/seed2027_paper_no_interpretable_readout_expert_adapter.json`
- `results/pure_cdm_exp110_paper_ablation_seed2027/seed2027_paper_no_student_conditioned_ukc_readout_residual.json`
- `results/pure_cdm_exp110_paper_ablation_seed2027/seed2027_paper_no_concept_evidence_readout_residual.json`

Reused experiment 111 outputs:

- `results/pure_cdm_exp110_ablation/seed2027_baseline_reproduce.json`
- `results/pure_cdm_exp110_ablation/seed2027_a_constant_cog_align.json`
- `results/pure_cdm_exp110_ablation/seed2027_a_no_concept_prior.json`
- `results/pure_cdm_exp110_ablation/seed2027_a_no_branch_bce.json`
- `results/pure_cdm_exp110_ablation/seed2027_a_no_dual_tower.json`
- `results/pure_cdm_exp110_ablation/seed2027_a_no_cog_align.json`

## Decision

- For paper-style seed2027 reporting, group `concept_prior`, `late anneal`, and
  `gs_difficulty_adapter` as controls/diagnostics rather than core positive
  contributors.
- The strongest contribution story is concept evidence readout, cognitive
  alignment, student-conditioned UKC readout, dual tower, and branch BCE.
- Do not promote removal of `gs_difficulty_adapter` from one seed alone; it
  should only be considered if a future matched multi-seed follow-up confirms it.
