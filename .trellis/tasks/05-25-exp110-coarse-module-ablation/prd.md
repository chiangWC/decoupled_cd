# exp110 coarse module ablation

## Goal

Run coarse, paper-style module ablations for the exp110 pure-CDM route using a single seed comparison. This follows the prior seed2027 component ablation and groups the many implementation switches into a small number of conceptual modules.

## What I already know

* User asked to run the 2-3 coarse module combination versions discussed previously.
* Prior discussion focused on single-seed, seed2027 comparisons.
* Use exp110 seed2027 baseline as the paired baseline.
* No checkpoint inference average, hybrid stacker, validation-trained combiner, or tabular inference side channel.

## Assumptions

* Use `seed=2027`.
* Run three coarse modules:
  * `without_evidence_aware_readout`
  * `without_cognitive_alignment_objective`
  * `without_dual_branch_ensemble`
* Reuse the exact existing `a_no_dual_tower` seed2027 result for `without_dual_branch_ensemble`, because it removes dual tower and branch BCE together.

## Requirements

* Preserve exp110 baseline settings except for the grouped module removals.
* Report AUC, ACC, RMSE and deltas against exp110 seed2027.
* Record result JSON paths.
* Do not change default runner behavior.

## Planned Ablations

### without_evidence_aware_readout

Remove these readout/evidence adapters together:

* high-concept logit adapter
* pairwise history interaction adapter
* interpretable readout expert adapter
* student-conditioned UKC readout residual
* concept evidence readout residual

### without_cognitive_alignment_objective

Remove the training-only evidence alignment objective group:

* history evidence logit prior residual and related weights
* history evidence cognitive alignment weights/schedule
* concept evidence prior residual and related train-only prior flags

### without_dual_branch_ensemble

Remove the dual-branch architecture group:

* dual CDM ensemble
* secondary tower dimension
* branch BCE

Use existing exp111 seed2027 `a_no_dual_tower` output if compatible.

## Acceptance Criteria

* [x] New runs complete or are explicitly reused from existing matching outputs.
* [x] Final table has AUC/ACC/RMSE and paired deltas.
* [x] Experiment documentation updated if useful for future paper-style reporting.

## Out of Scope

* Multi-seed expansion.
* Hyperparameter sweeps.
* Checkpoint averaging and hybrid evaluators.

## Result

* Detail doc: `docs/experiments/113_exp110_coarse_module_seed2027_ablation.md`
* New result directory: `results/pure_cdm_exp110_coarse_ablation_seed2027/`
* Reused result: `results/pure_cdm_exp110_ablation/seed2027_a_no_dual_tower.json`
* Main finding: removing `Evidence-aware readout` drops seed2027 AUC by
  `-0.010232`, removing `Cognitive alignment objective` drops `-0.005407`,
  and removing `Dual-branch ensemble` drops `-0.004202`.
