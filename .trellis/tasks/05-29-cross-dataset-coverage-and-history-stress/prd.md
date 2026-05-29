# Cross Dataset Coverage And History Stress

## Goal

Extend experiment 116 and experiment 117 beyond ASSIST09 so the coverage-bias and hidden-history robustness claims are not single-dataset observations. The first comparable batch should cover ASSIST17 and NIPS34, using the same Exp81 baseline versus Exp110 full comparison shape wherever checkpoints exist or can be trained.

## What I Already Know

* Exp116 currently reports ASSIST09 student-concept-holdout coverage slices across four seeds for Exp81 and Exp110 full, with seed2027 ablations.
* Exp117 currently reports ASSIST09 history-hiding stress tests on both the student-concept holdout split and the ordered split.
* Existing cross-dataset Exp110 seed2024 summaries exist for ASSIST17 and NIPS34:
  * `results/exp110_cross_dataset/assist_17_seed2024_exp110.json`
  * `results/exp110_cross_dataset/nips34_seed2024_exp110.json`
* I did not find matching ASSIST17/NIPS34 Exp81 baseline summaries on the remote result tree, so model-comparison stress tests require baseline training.
* ASSIST17 data is available through `../ConceptSkillCDM/data/assist_17/`; its exp110 summary uses a copied Q matrix at `results/exp110_cross_dataset/assist_17_q_matrix.csv`.
* NIPS34 data was moved outside the repo to `/home/xph/jwc/research/datasets/NIPS/NIPS34/process_data/`.
* Junyi has only a reduced-capacity / different-training-mode exp110-near result. It should not be mixed into the main Exp81-vs-Exp110 original-protocol table unless a matching baseline and a clearly labeled reduced-capacity protocol are added.

## Assumptions

* "另外几个数据集" should prioritize ASSIST17 and NIPS34 because they have comparable Exp110 original-protocol checkpoints and tractable training cost.
* Exp116 cross-dataset extension should use student-concept holdout splits generated with `scripts/split_student_concept_holdout.py`, then train Exp81 and Exp110 full on those splits.
* Exp117 cross-dataset extension should evaluate both the original chronological/random split checkpoints and the new student-concept-holdout checkpoints when matching Exp81/Exp110 summaries are available.
* One seed, `2024`, is the first batch for cross-dataset coverage. Additional seeds can be added after the first-batch result confirms runtime and support sizes.

## Requirements

* Add or reuse a coverage-slice evaluator that computes test target coverage from train history only:
  * `coverage = 0`
  * `0 < coverage < 0.5`
  * `0.5 <= coverage < 1`
  * `coverage = 1`
* Report overall AUC, low-coverage AUC, full-coverage AUC, coverage gap, low-coverage ECE, and low-coverage Brier per model.
* Preserve the Exp117 history-hiding contract: evaluation-only perturbation, fixed exercise evidence by default, `delta_auc = original_auc - hidden_auc`.
* Keep result artifacts under dataset-specific result directories and update experiment docs with commands, paths, support sizes, and interpretation caveats.

## Acceptance Criteria

* [ ] ASSIST17 and NIPS34 have generated student-concept holdout splits with recorded split summaries.
* [ ] ASSIST17 and NIPS34 have seed2024 Exp81 baseline and Exp110 full summaries on those holdout splits, or a documented reason a dataset could not complete.
* [ ] ASSIST17 and NIPS34 have seed2024 Exp81 baseline summaries on their existing original splits, or a documented reason a dataset could not complete.
* [ ] Exp116 cross-dataset reports include coverage bucket support and target metrics for Exp81 and Exp110.
* [ ] Exp117 cross-dataset reports include hide ratios `0.2,0.4,0.6,0.8` and mask seeds `11,13,17` for Exp81 and Exp110.
* [ ] `docs/experiments/116_student_concept_holdout_coverage_slice.md` and `docs/experiments/117_history_hiding_stress_test.md` are updated with the new cross-dataset evidence.

## Definition Of Done

* Focused scripts compile on the remote host.
* New unit tests are added or existing focused tests are run when evaluator code changes.
* Training/evaluation commands and result artifact paths are recorded in docs.
* The task is committed and pushed before remote experiment execution.

## Out Of Scope

* Adding new model ideas or tuning hyperparameters based on coverage/stress evaluation.
* Treating Junyi reduced-capacity results as directly comparable to ASSIST17/NIPS34 original Exp110.
* Full multi-seed cross-dataset sweeps unless the first seed finishes cleanly and runtime is acceptable.

## Technical Notes

* Existing history hiding evaluator: `scripts/evaluate_history_hiding_stress.py`.
* Existing holdout splitter: `scripts/split_student_concept_holdout.py`.
* Experiment protocol source: `.trellis/spec/backend/experiment-protocol.md`.
* Cross-dataset prior record: `docs/experiments/114_cross_dataset_exp110_runs.md`.
