# exp114 cross-dataset runs

## Goal

Record the remote cross-dataset runs performed from the exp110-style runner family, so ASSIST17, NIPS34, and Junyi results are discoverable in the experiment ledger.

## What I already know

* User asked to record this session's dataset runs as experiment 114.
* Runs were executed remotely on `xph-pc` under `/home/xph/jwc/research/decoupled_cd`.
* ASSIST17 and NIPS34 ran with the exp110-style `dual64x80`, `branch_bce=0.18`, `recompute_minibatch`, `batch_size=65536`, `lr=3e-4` protocol.
* Junyi cannot run the original exp110 `dual64x80` protocol on a 24GB 4090 due to dense `students x concepts x dim` propagation tensors.
* Junyi ran successfully with an exp110-near low-dimensional `dual16x32` variant: the command attempted `--concept-dim 32`, but `--dataset junyi` applied dataset defaults and the result JSON records `concept_dim=16`, `dual_cdm_secondary_concept_dim=32`.

## Assumptions

* This task is documentation-only.
* Experiment 114 should include both successful metrics and the failed Junyi original-protocol memory finding.
* No new remote training should be launched for this documentation pass.

## Requirements

* Add a new `docs/experiments/114_*.md` experiment note.
* Include dataset scale, exact command/protocol summaries, result paths, metrics, and memory notes.
* Update `docs/experiment_index.jsonl` with experiment 114.
* Keep source code unchanged.

## Acceptance Criteria

* [ ] New experiment 114 doc exists and follows the existing experiment note style.
* [ ] The doc records ASSIST17, NIPS34, and Junyi results from remote JSON/log artifacts.
* [ ] The experiment index contains an entry pointing to the new doc.
* [ ] Git diff is limited to documentation/task bookkeeping.

## Out of Scope

* New model memory optimizations for Junyi original exp110.
* Additional seeds or additional dataset runs.
* Changing dataset defaults or runner scripts.

## Technical Notes

* Relevant existing docs: `docs/experiments/110_recompute_minibatch_low_memory_dual_cdm.md`, `docs/experiments/111_exp110_ablation_study.md`.
* Remote result locations are under `results/exp110_cross_dataset/` and `results/junyi_memory_trials/`.
