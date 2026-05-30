# TKC/UKC Gate Diagnostic

## Goal

Add and run an evaluation-only diagnostic for the learned student TKC/UKC fusion
gate. The experiment should show whether the model assigns higher TKC weight
to students with richer observed concept histories, and whether low target
coverage samples tend to rely more on UKC/graph-side state.

## What I Already Know

* The propagation module computes:
  * `coverage = student_tkc_mask.mean(dim=1, keepdim=True)`
  * `tkc_weight = sigmoid(student_fusion_gate([coverage, tkc_mean, ukc_mean]))`
  * `student_state = tkc_weight * tkc_mean + (1 - tkc_weight) * ukc_mean`
* `tkc_weight` is currently local to `models/hetero_propagation.py` and is not
  returned in `PropagationOutput`.
* The gate input coverage is student-level global train-history concept
  coverage, not target-exercise coverage.
* Existing evaluation scripts already provide summary loading, path override,
  bundle preparation, and target coverage helpers:
  * `scripts/evaluate_coverage_slice.py`
  * `scripts/evaluate_history_hiding_stress.py`
* For dual-tower checkpoints, each tower has its own propagation module and
  learned student fusion gate.

## Assumptions

* Returning `tkc_weight` as diagnostic metadata does not change training or
  prediction behavior.
* The primary paper mechanism claim should be made from student global coverage
  vs average `tkc_weight`; target coverage vs `tkc_weight` should be treated as
  a sample-level association, not direct gate control.
* The first run should target ASSIST09 ordered and student-concept holdout
  Exp110 checkpoints. Cross-dataset gate diagnostics are optional follow-up.

## Requirements

* Add diagnostic access to `tkc_weight` without changing model predictions.
* Add a script that loads one or more checkpoint summaries and writes:
  * JSON report
  * binned CSV rows for plotting
  * compact summary CSV
* The script must support:
  * repeated `--summary` and matching `--model-name`
  * `--split valid|test`
  * train/valid/test/Q/graph path overrides
  * single-tower and dual-tower checkpoints
* Report at least:
  * student-global-coverage bin vs average `tkc_weight`
  * target-coverage bucket vs average `tkc_weight`
  * correlation between student global coverage and `tkc_weight`
  * correlation between target coverage and `tkc_weight`
  * counts per bin/bucket
* For dual towers, report primary, secondary, and mean tower rows.
* Add focused tests for diagnostic helper behavior.
* Run remote verification and at least one remote diagnostic command.
* Update experiment docs and experiment index with the result and caveats.

## Acceptance Criteria

* [x] `PropagationOutput` and `DecoupledForwardOutput` expose `tkc_weight`.
* [x] Existing model output consumers remain compatible.
* [x] Gate diagnostic script produces JSON and CSV artifacts remotely.
* [x] The report distinguishes direct student-coverage gate input from indirect
  target-coverage association.
* [x] Remote tests pass for the new script and touched model output behavior.
* [x] Experiment documentation records artifact paths and the main conclusion.

## Definition Of Done

* No GitHub push.
* Remote execution uses `bash scripts/remote_exec.sh`.
* Docs and `docs/experiment_index.jsonl` are updated with traceable artifacts.
* Work is committed to the normal `origin` remote only if remote verification
  requires it; do not push to `github`.

## Out Of Scope

* Training new models.
* Changing the fusion gate architecture, loss, or model scoring.
* Making publication-quality figures in this task; CSV/JSON plot data is
  sufficient.
* Claiming target coverage directly controls the gate.

## Technical Notes

* Prefer a diagnostic return over forward hooks because it is easier to test and
  handles dual-tower models explicitly.
* The evaluator can reuse `load_model`, `normalize_summary_for_current_loader`,
  `prepare_experiment_split_bundles`, and `add_target_coverage`.
* For memory safety, evaluate tower propagation on the unique students appearing
  in the target split.
