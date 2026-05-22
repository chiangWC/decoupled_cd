# cleanup non-trial cdm internals

## Goal

Shrink the active CDM model surface after the runner and `train.py` CLI cleanup. The current HEAD should expose and test the baseline plus experiment 104 model paths, not constructor options and tests for rejected probes that no longer have an active training CLI or runner.

## What I already know

* User asked whether the large CDM-side "CLI" should be cleaned, then said to continue.
* The previous cleanup left only `scripts/run_assist09_baseline.sh` and `scripts/run_assist09_history_alignment_trial.sh`.
* `scripts/train.py` now passes only baseline and exp104 kwargs into `DecoupledCDM` / `DecoupledCDMEnsemble`.
* `DecoupledCDM.__init__` still exposes probe-only knobs with no active `train.py` entry:
  * `cognitive_readout_head_count`
  * `evidence_calibrated_behavior_gate` and behavior-gate tuning
  * `student_evidence_ability_prior_*`
  * `student_evidence_gs_prior_*`
  * `concept_evidence_calibrated_readout_*`
  * `history_evidence_fusion_readout_*`
  * `history_evidence_linear_readout_*`
  * `history_evidence_output_calibration_*`
  * `exercise_evidence_prior_*`
  * `exercise_evidence_difficulty_adapter_*`
* Tests still cover those historical model routes and create maintenance noise.

## Assumptions

* Keep baseline model internals: graph propagation, high-concept adapter, pairwise interaction adapter, GS difficulty adapter, interpretable readout expert, student-conditioned UKC readout residual, concept evidence readout residual.
* Keep exp104 internals: dual CDM ensemble, concept evidence prior residual, history evidence logit prior residual with `loss_only` mode, cognitive alignment training objective, branch BCE training objective.
* Keep shared evidence helpers when still used by retained paths.
* Do not try to preserve current HEAD as a runnable reproduction surface for every old experiment; historical reproduction can use docs and git history.

## Requirements

* Remove non-trial `DecoupledCDM` constructor parameters and their validation/state.
* Remove corresponding forward-time branches, residual heads, helper methods, and tests.
* Remove non-trial propagation behavior-gate plumbing if it is only reachable through the removed CDM path.
* Keep `scripts/train.py` compatible with baseline and exp104 runners.
* Keep `DecoupledCDMEnsemble` compatible with exp104.
* Update docs/spec if this cleanup establishes a new active model-surface contract.

## Acceptance Criteria

* Active runner scripts still parse and smoke remotely.
* `scripts/run_assist09_history_alignment_trial.sh` still reports exp104 fields in output.
* Focused unit tests for retained CDM paths pass remotely.
* Removed constructor names no longer appear in active model code or active tests except archived docs/history.

## Out of Scope

* Rewriting all archived experiment detail docs.
* Removing retained baseline adapters.
* Removing trainer internals not directly tied to CDM constructor surface unless required by tests or imports.
* Rerunning the full 300-epoch exp104 training unless the user asks after cleanup; smoke is enough for this structural cleanup.

## Technical Notes

* Main files inspected: `models/decoupled_cdm.py`, `models/ensemble_cdm.py`, `models/hetero_propagation.py`, `scripts/train.py`, `trainers/engine.py`, `tests/test_decoupled_cdm.py`, `tests/test_hetero_propagation.py`.
* Relevant spec: `.trellis/spec/backend/experiment-protocol.md`, especially the current pure-CDM trial runner contract.
