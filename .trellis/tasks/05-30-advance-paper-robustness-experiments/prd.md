# Advance Paper Robustness Experiments

## Goal

Complete the remaining paper-support experiments for the Exp110 story using a staged seed2027-first gate: advance stress-split component ablations, history-hiding robustness, and capacity-control checks first on seed2027/existing seed2027 evidence, then expand to more seeds only when the story is weak, too extreme, or needed for paper-level claims.

## What I Already Know

* The user wants all three experiment groups advanced, while preserving the seed2027-first, expand-if-needed strategy.
* Experiment 116 already has ASSIST09 student-concept holdout Exp81 vs Exp110 full for seeds 2024, 2025, 2026, and 2027.
* Experiment 116 currently has stress-split component ablations only for seed2027: w/o dual tower, w/o cognitive alignment, and w/o branch BCE.
* Experiment 117 already has evaluation-only history hiding for seed2027 on ASSIST09 ordered and student-concept holdout splits.
* The priority order for ablations is cognitive alignment, dual tower, then branch BCE.
* Existing scripts cover training, coverage slicing, and history hiding; this task should prefer experiment execution and ledger updates over code changes.

## Requirements

* Stage A: consolidate ASSIST09 student-concept holdout seed2027 ablation evidence first:
  * Exp110 w/o dual tower
  * Exp110 w/o cognitive alignment
  * Exp110 w/o branch BCE
* Expand Stage A to seeds 2024, 2025, 2026, and 2027 only if:
  * the paper needs strong component claims on the stress split
  * branch BCE remains central despite its extreme seed2027 collapse
  * seed2027 component evidence is not enough for the planned story.
* Stage B: consolidate ASSIST09 history-hiding evaluation-only robustness for Exp81 vs Exp110 full on seed2027 first:
  * ordered split
  * student-concept holdout split
  * hide ratios 0.4, 0.6, 0.8
  * mask seeds 11, 13, and 17
* Expand Stage B to seeds 2024, 2025, 2026, and 2027 only if the seed2027 table is too weak for the robustness story or the main paper table needs multi-seed robustness.
* Stage C: add at least one seed2027 capacity-control single-tower baseline on ASSIST09 holdout:
  * start with single96
  * add single128 if single96 is inconclusive or too close to Exp110
  * use the same training protocol and cognitive alignment as Exp110, removing only the dual-tower ensemble/secondary tower and branch BCE.
* Summarize results in experiment ledger docs and index entries.
* Keep claims precise:
  * component claims require multi-seed evidence from the stress split
  * robustness claims should distinguish hidden AUC from delta AUC
  * capacity claims should compare low-coverage and history-hiding behavior, not only overall AUC.

## Acceptance Criteria

* [ ] Remote experiment artifacts exist for all stage-1 seed2027 A/B/C runs or any skipped run is explicitly documented with reason.
* [ ] Expansion decisions for multi-seed A/B are documented with evidence-based reasons.
* [ ] Coverage-slice summary tables include seed2027 stress-split ablations and multi-seed means only if expansion is triggered.
* [ ] History-hiding hide80 summary table includes original AUC, hidden AUC, delta AUC, hidden Brier, and hidden ECE for Exp81 vs Exp110 full; multi-seed rows are added only if expansion is triggered.
* [ ] Capacity-control results compare Exp110 dual64x80, w/o dual tower, and single96/single128 on stress split; history-hiding comparison is added when checkpoints are available.
* [ ] `docs/experiments/116_student_concept_holdout_coverage_slice.md`, `docs/experiments/117_history_hiding_stress_test.md`, and `docs/experiment_index.jsonl` are updated.
* [ ] Remote verification commands for touched evaluators or generated scripts are recorded.

## Out Of Scope

* Changing accepted `master` model-mainline defaults.
* Promoting capacity-control variants as a new default runner.
* Adding new model architecture code unless existing CLI flags cannot express the capacity-control baseline.

## Technical Notes

* Remote execution must use `bash scripts/remote_exec.sh <command>` after local commits are pushed.
* Remote project path is `/home/xph/jwc/research/decoupled_cd`; conda env is `decoupled_cd`.
* Experiment 116 result root: `results/coverage_holdout_exp116/`.
* Experiment 117 result root: `results/history_hiding_exp117/`.
* Existing history-hiding evaluator: `scripts/evaluate_history_hiding_stress.py`.
* Existing coverage-slice evaluator: `scripts/evaluate_coverage_slice.py`.
* Existing Exp110 ablation command details are documented in `docs/experiments/111_exp110_ablation_study.md`.
