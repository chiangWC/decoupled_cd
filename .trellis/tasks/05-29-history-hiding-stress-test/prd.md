# History hiding stress test

## Goal

Run an evaluation-only history hiding stress test to measure whether Exp110 is
more robust than Exp81 and key Exp110 ablations when student histories are
incomplete. The experiment should be run on both the ASSIST09 student-concept
holdout split and the default ASSIST09 ordered split.

## What I Already Know

- The user wants both datasets:
  - Primary: student-concept holdout split generated for experiment 116 at
    `/tmp/assist09_holdout_seed2024` on the remote host.
  - Secondary / appendix: default `data/assist_09_ordered`.
- The experiment is evaluation-only; it must not retrain models.
- Perturbation ratios are `hide_ratio in {0.2, 0.4, 0.6, 0.8}`.
- Compare:
  - Exp81 baseline.
  - Exp110 full.
  - Exp110 without cognitive alignment.
  - Exp110 without dual tower.
- Metrics should report hidden AUC and delta AUC:
  `delta_auc = original_auc - hidden_auc`.
- The cleanest perturbation is student-side history hiding: rebuild the
  student history tensors from a masked train-history frame while keeping
  exercise global evidence tied to the original train frame.
- Use multiple mask seeds per hide ratio to avoid one random mask determining
  the conclusion.
- Existing model loading and prediction helpers are in
  `scripts/analyze_prediction_slices.py`, but its `predict_bundle` path does
  not pass `exercise_evidence`; the stress evaluator should use the same
  tensor contract as `trainers.engine.evaluate_model`.
- Remote project path is `/home/xph/jwc/research/decoupled_cd`; use
  `bash scripts/remote_exec.sh <command>` for execution.

## Requirements

- Add a reusable evaluation script for history hiding stress tests.
- The script must accept one or more training summary JSONs and evaluate a
  selected split, defaulting to test.
- The script must support:
  - `--hide-ratios`, default `0.2,0.4,0.6,0.8`.
  - `--mask-seeds`, default at least three seeds.
  - student-side interaction hiding.
  - output JSON report.
  - output per-run CSV.
  - output summary CSV aggregated by dataset, model, and hide ratio.
- For each summary, compute an unperturbed original row and perturbed rows for
  every `hide_ratio x mask_seed`.
- Report at least:
  - original AUC.
  - hidden AUC.
  - delta AUC.
  - hidden ACC.
  - hidden Brier.
  - hidden ECE.
  - hidden train-history row count.
- Run the script remotely on:
  - holdout seed2027 summaries under `results/coverage_holdout_exp116/`.
  - default ordered seed2027 summaries under
    `results/expert_output_modulation/` and
    `results/pure_cdm_exp110_ablation/`.
- Preserve remote artifacts under `results/history_hiding_exp117/`.
- Record the conclusion in a new experiment detail doc and
  `docs/experiment_index.jsonl`. Update `docs/model_improvement_plan.md` if
  the conclusion changes the paper story or next-step priority.

## Acceptance Criteria

- [ ] The evaluator script can run from summaries without retraining.
- [ ] The evaluator masks only student-side train history by default; exercise
  evidence remains based on the original train split.
- [ ] Both holdout and default ordered datasets are evaluated for seed2027.
- [ ] Results include Exp81 baseline, Exp110 full, w/o cognitive alignment, and
  w/o dual tower for both datasets where checkpoints exist.
- [ ] Each hide ratio uses multiple mask seeds and reports mean/std delta AUC.
- [ ] The report states whether Exp110 has smaller delta AUC and/or higher
  hidden AUC than Exp81 and ablations.
- [ ] Remote artifacts are recorded with paths.
- [ ] Local experiment ledger docs are updated.

## Out Of Scope

- Training new checkpoints.
- Tuning model hyperparameters for hidden histories.
- Changing runner defaults or model architecture.
- Promoting a new default model.
- Concept-level hiding unless interaction hiding shows a clear need for a
  second perturbation type.

## Technical Notes

- Existing helpers:
  - `scripts/analyze_prediction_slices.py::load_model`.
  - `data.prepare_experiment_split_bundles`.
  - `data.pipeline.build_history_tensors`.
  - `data.pipeline.build_exercise_evidence_tensor`.
  - `trainers.engine._bundle_tensors`.
  - `utils.compute_metrics`.
- Relevant docs:
  - `.trellis/spec/backend/experiment-protocol.md`.
  - `.trellis/spec/backend/database-guidelines.md`.
  - `.trellis/spec/backend/quality-guidelines.md`.
  - `docs/experiments/116_student_concept_holdout_coverage_slice.md`.
