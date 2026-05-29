# Coverage slice experiment

## Goal

Recompute test-set metrics by target knowledge coverage for the existing ASSIST09 experiment 81 baseline, experiment 110 full model, and key experiment 110 ablations. The experiment should show whether overall AUC hides a coverage-bias problem and whether experiment 110 improves low-coverage / UKC-heavy cases more clearly than full-coverage cases.

## What I already know

- The user wants the test interaction coverage defined as `|Q_e intersect Seen_s| / |Q_e|`, where `Seen_s` is built from train history only.
- Required comparison set:
  - Experiment 81 baseline.
  - Experiment 110 full model.
  - Experiment 110 without dual tower.
  - Experiment 110 without cognitive alignment.
  - Experiment 110 without branch BCE.
- Required reporting metrics:
  - overall AUC.
  - low-coverage AUC.
  - full-coverage AUC.
  - coverage gap = full-coverage AUC minus low-coverage AUC.
  - low-coverage ECE.
  - low-coverage Brier.
- Keep the original coverage buckets for diagnostics:
  - `coverage = 0`.
  - `0 < coverage < 0.5`.
  - `0.5 <= coverage < 1`.
  - `coverage = 1`.
- Local repository does not contain ASSIST09 data or `results/`; these live on the remote execution host.
- Existing `scripts/analyze_prediction_slices.py` already reconstructs predictions from training summaries and computes train-history student concept overlap. It can be reused for a remote one-off coverage analysis without changing model code.

## Assumptions

- `AUC_low_coverage`, `ECE_low_coverage`, and `Brier_low_coverage` should use the aggregate `target_coverage < 0.5`, combining `coverage = 0` and `0 < coverage < 0.5`.
- `AUC_full_coverage` should use `target_coverage = 1`.
- Use matched seeds where checkpoints are available. Experiment 110 and ablations have four seeds `2024-2027`; experiment 81 has three documented promotion seeds plus the available seed2027 exp81 baseline checkpoint.
- No new training is required.

## Requirements

- Run coverage slice evaluation on the remote host using existing summaries and checkpoints.
- Use train history only when deriving `Seen_s`.
- Compute per-run and mean-by-model results.
- Preserve result artifacts on the remote host under `results/coverage_slice_exp115/`.
- Report the result table and the conclusion in the chat.
- Update a local experiment note if the result is substantive.
- After discovering that the default ASSIST09 random interaction split has too little low-coverage support, continue with `scripts/split_student_concept_holdout.py`.
- Generate a student-concept holdout split and first run a seed2027 staged comparison for:
  - Exp81 baseline.
  - Exp110 full.
  - Exp110 without dual tower.
  - Exp110 without cognitive alignment.
  - Exp110 without branch BCE.
- If the seed2027 staged comparison supports the story, expand to more seeds as a follow-up.

## Acceptance Criteria

- [x] Every requested model has coverage-slice metrics for available seeds.
- [x] The report includes overall AUC, low-coverage AUC, full-coverage AUC, coverage gap, low-coverage ECE, and low-coverage Brier.
- [x] The report includes enough bucket counts to verify the low-coverage slice has support.
- [x] The conclusion explicitly states whether the result supports the coverage-bias story.
- [x] The student-concept holdout split summary is recorded with overlap counts.
- [x] The seed2027 staged holdout results are recorded separately from the original random-split diagnostic.
- [x] The primary Exp81-vs-Exp110 holdout comparison is expanded to multiple training seeds if the staged result supports the story.

## Out of Scope

- Training new checkpoints.
- Changing model architecture, runner defaults, or metric definitions.
- Promoting a new default model.

## Technical Notes

- Remote project path: `/home/xph/jwc/research/decoupled_cd`.
- Remote execution command: `bash scripts/remote_exec.sh <command>`.
- Relevant code:
  - `scripts/analyze_prediction_slices.py`.
  - `utils.compute_metrics`.
  - `data.q_matrix.normalize_concept_sequence`.
- Relevant docs:
  - `.trellis/spec/backend/experiment-protocol.md`.
  - `docs/experiments/081_exp70_single_only_evidence_readout_rebase.md`.
  - `docs/experiments/110_recompute_minibatch_low_memory_dual_cdm.md`.
  - `docs/experiments/111_exp110_ablation_study.md`.
