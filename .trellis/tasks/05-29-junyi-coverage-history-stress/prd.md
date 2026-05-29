# Junyi Coverage And History Stress Extension

## Goal

Add Junyi evidence for the experiment 116 coverage-slice diagnostic and experiment 117 history-hiding stress test, while keeping the protocol caveat clear: Junyi cannot run original Exp110 `dual64x80` on the available 24GB GPU, so these results are a reduced-capacity Junyi-family supplement, not a direct ASSIST17/NIPS34-equivalent transfer.

## What I Already Know

* Existing Junyi Exp110-near result:
  * `results/junyi_memory_trials/junyi_seed2024_student_recompute2048_dual16x32_300ep.json`
  * `concept_dim=16`, `dual_cdm_secondary_concept_dim=32`
  * `training_mode=student_recompute_minibatch`, `student_batch_size=2048`
  * test AUC `0.824503`
* Original Exp110 `dual64x80` OOMs on Junyi due dense `10000 x 706` student-concept propagation.
* I did not find a matching Junyi Exp81 baseline summary on the remote result tree.
* Junyi data exists at `../ConceptSkillCDM/data/junyi/{train,valid,test}.csv`; it does not include a checked-in `Q_matrix.csv`, so training/evaluation derives Q from split `exer_id,cpt_seq` pairs when needed.
* The reusable evaluators already exist:
  * `scripts/evaluate_coverage_slice.py`
  * `scripts/evaluate_history_hiding_stress.py`

## Assumptions

* For Junyi, the fair comparison is "Exp81-style baseline with Junyi default capacity and student-subset training" versus "Exp110-near dual16x32 with student-subset training."
* Original-split history hiding should compare the newly trained Junyi baseline against the existing optimized Junyi Exp110-near checkpoint.
* Exp116 coverage-slice extension should generate a Junyi student-concept holdout split and train both baseline and Exp110-near on that split.
* All Junyi tables must be labeled as reduced-capacity / student-recompute, not original Exp110.

## Requirements

* Generate a Junyi student-concept holdout split outside the repo worktree.
* Train seed2024 Junyi baseline on the original split with student-recompute mode.
* Train seed2024 Junyi baseline and Exp110-near on the Junyi holdout split.
* Run coverage-slice evaluation on the Junyi holdout split.
* Run history-hiding stress on Junyi original and holdout splits with hide ratios `0.2,0.4,0.6,0.8` and mask seeds `11,13,17`.
* Update experiment docs and index with caveated Junyi results.

## Acceptance Criteria

* [x] Remote result artifacts exist under `results/cross_dataset_exp116_117/` for Junyi baseline, holdout training, coverage report, and history-hiding reports.
* [x] Coverage report includes low/full support, AUC, coverage gap, low Brier, and low ECE.
* [x] History-hiding reports include original AUC, hidden AUC, delta AUC, hidden Brier, and hidden ECE.
* [x] Docs explicitly state Junyi is not original Exp110 `dual64x80`.

## Definition Of Done

* Remote commands complete without editing remote code directly.
* Docs and experiment index are updated.
* Changes are committed, pushed, task archived, and session recorded.

## Out Of Scope

* Running original Junyi `dual64x80`, which is already recorded as OOM.
* Multi-seed Junyi sweeps.
* Junyi component ablations.
* Hyperparameter optimization based on Junyi slice/stress metrics.

## Technical Notes

* Use `student_recompute_minibatch --student-batch-size 2048` for Junyi full runs.
* Derive Q-matrix from the full train/valid/test data when preparing holdout source artifacts.
* Keep scratch data under `/home/xph/jwc/research/local_data/cross_dataset_exp116_117/`.
