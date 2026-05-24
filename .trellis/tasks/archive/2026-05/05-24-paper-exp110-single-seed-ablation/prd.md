# paper style exp110 single seed ablation

## Goal

Run a cleaner paper-style component ablation for the exp110 pure-CDM route using
a single seed comparison. The goal is to produce a readable table that explains
which model components matter, without mixing in protocol sweeps or hyperparameter
search.

## What I already know

* User wants a more complete, paper-style ablation.
* User explicitly wants single-seed comparison.
* Recent discussion focused on seed2027, so this task uses `seed=2027` unless
  the user later asks for a different seed.
* Prior experiment 111 already contains seed2027 results for the exp110 baseline
  and the exp110 incremental `a_` ablations.
* Missing pieces are older/base runner components that were not ablated in 111:
  high-concept logit adapter, pairwise history interaction adapter, GS difficulty
  adapter, interpretable readout expert, student-conditioned UKC residual, and
  concept evidence readout residual.

## Assumptions

* "Paper-style" means one-factor-at-a-time component removal versus the same
  exp110 seed2027 baseline.
* Only `AUC`, `ACC`, and `RMSE` are required in the final user-facing table.
* Existing matching seed2027 outputs from experiment 111 may be reused instead
  of rerun.

## Requirements

* Preserve pure-CDM constraints:
  * no checkpoint inference average
  * no hybrid stacker
  * no validation-trained combiner
  * no train-history tabular inference side channel
* Use exp110 seed2027 as the baseline.
* Report paired deltas for AUC, ACC, and RMSE.
* Include both recent exp110 modules and older base-runner modules.
* Keep protocol sweeps out of the final table.

## Planned Component Rows

Reference rows from experiment 111:

* `baseline_reproduce`
* `a_constant_cog_align`
* `a_no_concept_prior`
* `a_no_branch_bce`
* `a_no_dual_tower`
* `a_no_cog_align`

New seed2027 rows to run:

* `paper_no_high_concept_logit_adapter`
* `paper_no_pairwise_history_interaction_adapter`
* `paper_no_gs_difficulty_adapter`
* `paper_no_interpretable_readout_expert_adapter`
* `paper_no_student_conditioned_ukc_readout_residual`
* `paper_no_concept_evidence_readout_residual`

## Acceptance Criteria

* [x] New missing seed2027 component ablations are run or explicitly skipped
  with a reason.
* [x] Final table includes baseline, component rows, AUC/ACC/RMSE, and paired
  deltas.
* [x] If results affect route judgment, update experiment documentation.

## Definition of Done

* Remote runs complete without concurrent `remote_exec` use.
* Results are traceable to JSON outputs.
* No default runner behavior is changed.

## Result

* Detail doc: `docs/experiments/112_exp110_paper_style_seed2027_ablation.md`
* New result directory: `results/pure_cdm_exp110_paper_ablation_seed2027/`
* Main finding: on seed2027, largest AUC drops come from removing
  `concept_evidence_readout_residual`, `cognitive_alignment`,
  `student_conditioned_ukc_readout_residual`, `dual_tower`, and `branch_bce`.

## Out of Scope

* Multi-seed expansion.
* Hyperparameter/protocol sweeps.
* Checkpoint average and hybrid evaluator routes.
