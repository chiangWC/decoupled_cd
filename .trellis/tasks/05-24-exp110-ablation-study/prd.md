# exp110 ablation study

## Goal

Run a focused ablation study using experiment 110's strongest pure-CDM training route as the baseline, excluding checkpoint inference averaging and hybrid evaluators. The goal is to identify which pieces are carrying the AUC peak/mean signal and which pieces are mainly memory or calibration tradeoffs.

## What I already know

* User wants ablations based on experiment 110's strongest route.
* Experiment 110 baseline is pure CDM, single training run, single selected checkpoint per seed.
* Experiment 110 best route:
  * `dual_cdm_secondary_concept_dim=80`
  * `dual_cdm_branch_bce_weight=0.18`
  * `training_mode=recompute_minibatch`
  * `batch_size=65536`
  * `learning_rate=0.0003`
  * `epochs=300`
  * runner-level late cognitive alignment schedule from experiment 103/104/106
  * train-only concept evidence prior starting at epoch 135
* Experiment 110 four-seed AUC: `0.778252/0.778263/0.777449/0.779321`, mean `0.7783213434`, max `0.7793212012`, peak CUDA `6.190541GB`.
* Experiment 106 full-batch `dual64x80 branchBCE=0.18` remains stronger on mean AUC: `0.778579`, but uses about `11.76GB`.
* Experiment 110 already tested and rejected or deprioritized:
  * `batch=32768` near this route, weaker seed2026 at tested lr.
  * `batch=65536 lr=1e-3`, weaker than `lr=3e-4`.
  * brier checkpoint selection, better calibration but weaker AUC.
  * `dual64x32 batch=65536 lr=1e-3`, weaker AUC.
  * activation checkpointing as the primary memory mechanism, too little full-batch memory reduction.

## Assumptions

* "Ablation" means controlled train-time ablations around exp110, not inference-time checkpoint averaging.
* AUC is the primary metric; ACC/RMSE/Brier/ECE and peak CUDA are secondary.
* Seed2026 is the most useful smoke seed because it is exp110's weakest seed.
* Seed2027 is useful when testing peak-AUC behavior because it produced exp110's current max.
* Full four-seed expansion should be reserved for ablations that pass a two-anchor smoke check.

## Open Questions

* None. User asked to design the full ablation and run all of it.

## Requirements

* Preserve pure-CDM constraints:
  * no hybrid stacker
  * no validation-trained combiner
  * no checkpoint inference average
  * no train-history tabular inference side channel
* Use exp110 as the named baseline and compare all ablations against its matching seed results.
* Record exact command lines, output JSON paths, AUC/ACC/RMSE/Brier/ECE, best epoch, and peak CUDA memory.
* Run the full selected matrix rather than stopping after smoke anchors.
* Reuse already documented four-seed results when they exactly match a requested ablation, and record them as references instead of rerunning.

## Full Ablation Matrix

All new runs use seeds `2024/2025/2026/2027` unless explicitly marked as an existing reference. Every run should be compared against the matching exp110 baseline seed:

* 2024: `0.7782518971`
* 2025: `0.7782634194`
* 2026: `0.7774488561`
* 2027: `0.7793212012`

### Track A: Core Mechanism Attribution

* Remove branch BCE: `--dual-cdm-branch-bce-weight 0.0`
  * Tests whether branch auxiliary supervision is still needed under recompute minibatch.
* Remove train-only concept prior:
  * Use the same direct `scripts/train.py` command as exp110 but omit `--concept-evidence-prior-residual`.
  * Tests whether the concept prior is essential or mainly a late training stabilizer.
* Remove late anneal, keep constant cognitive alignment:
  * `--history-evidence-cognitive-alignment-final-weight 0.05`
  * `--history-evidence-cognitive-alignment-anneal-start-epoch 170`
  * `--history-evidence-cognitive-alignment-anneal-end-epoch 230`
  * Tests whether late strengthening is still necessary in exp110.
* Remove cognitive alignment:
  * `--history-evidence-cognitive-alignment-weight 0.0`
  * `--history-evidence-cognitive-alignment-final-weight 0.0`
  * Tests whether exp110 is still driven by the experiment 95/103 cognitive objective.
* Remove dual tower:
  * Omit `--dual-cdm-ensemble`, keeping the rest of the protocol and recompute minibatch.
  * Tests whether exp110's gain is mainly dual-tower capacity/averaging or the training protocol.

### Track B: Training Semantics and Optimizer Ablation

* Full-batch training at exp110 lr:
  * `--training-mode full_batch`
  * `--learning-rate 0.0003`
  * omit `--batch-size`
  * Isolates recompute minibatch semantics from the lower learning rate.
* Recompute minibatch at exp106/default lr:
  * `--training-mode recompute_minibatch`
  * `--batch-size 65536`
  * `--learning-rate 0.001`
  * Completes the partial seed2026 triage from exp110 across all seeds.
* Batch size down:
  * `--batch-size 32768`
  * `--learning-rate 0.0003`
  * Tests whether smaller memory improves or harms AUC once lr is controlled.
* Batch size up:
  * `--batch-size 131072`
  * `--learning-rate 0.0003`
  * Tests whether closer full-batch semantics preserve AUC while staying near the memory target.
* Brier checkpoint selection:
  * `--checkpoint-selection-metric brier`
  * Completes the exp110 seed2026 calibration-oriented triage across all seeds.

### Track C: Local AUC Refinement Around exp110

* Learning rate local search:
  * `lr=0.0004`
  * `lr=0.0005`
* Branch BCE local search under recompute:
  * `--dual-cdm-branch-bce-weight 0.10`
  * `--dual-cdm-branch-bce-weight 0.15`
  * `--dual-cdm-branch-bce-weight 0.20`
* Concept prior timing:
  * immediate train prior: `--concept-evidence-prior-train-start-epoch 1`
  * delayed train prior: `--concept-evidence-prior-train-start-epoch 170`

### Track D: Capacity and Memory Boundary

* Secondary tower dimension:
  * `--dual-cdm-secondary-concept-dim 32`
  * `--dual-cdm-secondary-concept-dim 64`
  * `--dual-cdm-secondary-concept-dim 96`
* Existing reference, not rerun unless needed:
  * exp106 full-batch `dual64x80 branchBCE=0.18`, mean `0.778579`, peak CUDA about `11.76GB`
  * exp109 full-batch `dual64x32 branchBCE=0.18`, mean `0.777674`, peak CUDA about `8.85GB`

### Run Count

New four-seed runs:

* Track A: 5 variants x 4 seeds = 20 runs
* Track B: 5 variants x 4 seeds = 20 runs
* Track C: 7 variants x 4 seeds = 28 runs
* Track D: 3 variants x 4 seeds = 12 runs
* Total: 80 new runs, plus existing references from exp106/109/110.

## Acceptance Criteria

* [x] A task result note exists under `docs/experiments/` or task research notes with a clear ablation table.
* [x] Each completed run has a result JSON path and exact command recorded.
* [x] The full Track A-D matrix is either run or explicitly marked as an exact existing reference.
* [x] Final conclusion names:
  * strongest single-seed AUC variant
  * strongest mean-AUC candidate if expanded
  * best low-memory candidate
  * rejected mechanisms

## Definition of Done

* Results are reproducible from recorded commands.
* Documentation distinguishes pure-CDM training runs from checkpoint average and hybrid routes.
* Any code changes, if needed, pass focused tests.
* No default runner is changed unless explicitly requested after results are known.

## Out of Scope

* Checkpoint inference averaging.
* Hybrid stackers or validation-trained combiners.
* Broad architecture search unrelated to exp110.
* Changing `scripts/run_assist09_baseline.sh` default behavior in this task.

## Technical Notes

* Baseline doc: `docs/experiments/110_recompute_minibatch_low_memory_dual_cdm.md`
* Heavy comparison doc: `docs/experiments/106_dual_cdm_branch_bce_refinement.md`
* Runner: `scripts/run_assist09_history_alignment_trial.sh`
* Completed result directory: `results/pure_cdm_exp110_ablation/`
* Completed result note: `docs/experiments/111_exp110_ablation_study.md`
