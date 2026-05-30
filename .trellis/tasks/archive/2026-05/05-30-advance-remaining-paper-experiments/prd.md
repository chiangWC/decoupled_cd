# Advance Remaining Paper Experiments

## Goal

Complete the remaining non-capacity paper-support experiments for Exp110:
multi-seed component ablations on the ASSIST09 student-concept holdout stress
split, and multi-seed evaluation-only history hiding for Exp81 vs Exp110 full.

## What I Already Know

* The previous capacity-control task only handled the seed2027 `single96`
  control and is not enough.
* Experiment 116 already has ASSIST09 holdout Exp81 and Exp110 full checkpoints
  for seeds 2024, 2025, 2026, and 2027.
* Experiment 116 currently has the three component ablations only for seed2027.
* Experiment 117 currently has history hiding mainly for seed2027.
* The user wants the other experiments advanced, not held at seed2027.

## Requirements

* Stress-split component ablations on `/tmp/assist09_holdout_seed2024`:
  * w/o cognitive alignment
  * w/o dual tower
  * w/o branch BCE
  * seeds 2024, 2025, 2026, 2027
  * reuse existing seed2027 artifacts where present; train missing seeds.
* Run coverage-slice evaluation over Exp110 full plus the three ablations after
  missing ablation checkpoints exist.
* Multi-seed history hiding, evaluation-only:
  * split: ASSIST09 student-concept holdout
  * models: Exp81 baseline vs Exp110 full
  * seeds 2024, 2025, 2026, 2027
  * hide ratios: 0.4, 0.6, 0.8
  * mask seeds: 11, 13, 17
* Update experiment ledger docs with result tables and expansion decision.

## Acceptance Criteria

* [ ] Remote artifacts exist for all missing ablation runs.
* [ ] Multi-seed ablation coverage table records mean and per-seed behavior.
* [ ] Multi-seed history-hiding hide80 table records original AUC, hidden AUC,
  delta AUC, hidden Brier, and hidden ECE.
* [ ] `docs/experiments/116_student_concept_holdout_coverage_slice.md`,
  `docs/experiments/117_history_hiding_stress_test.md`, and
  `docs/experiment_index.jsonl` are updated.
* [ ] Any remaining caveats are explicit, especially branch BCE collapse and
  whether component claims are safe.

## Out Of Scope

* Additional capacity controls (`single128`, multi-seed single96).
* Cross-dataset expansion.
* Code changes unless existing scripts cannot express the required evaluations.

## Technical Notes

* Use remote execution through `bash scripts/remote_exec.sh`.
* Remote result roots already used:
  * `results/coverage_holdout_exp116/`
  * `results/history_hiding_exp117/`
* New follow-up artifacts can use `results/paper_robustness_followup/`.
* Keep exercise evidence fixed in history hiding by using evaluator defaults.
