# Update Experiment Default Reporting Baseline

## Goal

Clarify and persist the project's experiment "default口径" so future sessions and agents do not confuse the current active runner, low-memory paper reporting baseline, and single-seed ablation baseline.

## What I Already Know

* The current active pure-CDM trial runner is experiment 104 via `scripts/run_assist09_history_alignment_trial.sh`.
* Experiment 104 uses dual tower `64x80`, `branchBCE=0.10`, single-run / single-checkpoint pure-CDM semantics, and four-seed mean AUC `0.778370`.
* Experiment 106 is a pure-CDM refinement with `branchBCE=0.18`, four-seed mean AUC `0.778579`, but it is explicitly not promoted as the current runner default.
* Experiment 110 inherits the dual tower `64x80 + branchBCE=0.18` direction and changes training to `recompute_minibatch`, `batch_size=65536`, `lr=3e-4`, producing four-seed mean AUC `0.778321` with peak CUDA about `6.19GB`.
* Experiments 111-113 use experiment 110 as the low-memory pure-CDM ablation/reporting baseline.
* For paper-style seed2027 ablations, `exp110 seed2027 baseline_reproduce` has AUC `0.7793212012`.
* The `c_prior_start001` row only changes `--concept-evidence-prior-train-start-epoch` from `135` to `1`; its four-seed mean delta is about `-0.000004`, so it should not be promoted.
* Hybrid stacker and checkpoint-average results remain evaluator/diagnostic routes, not default training runner routes.

## Assumptions

* The desired change is about durable documentation and agent-facing defaults, not only this chat's temporary wording.
* The likely target files are `docs/handoff.md`, `docs/model_improvement_plan.md`, `docs/experiment_index.jsonl`, and possibly `.trellis/spec/backend/experiment-protocol.md`.
* Code or runner changes may be out of scope unless the user wants to promote the actual active runner.

## Open Questions

* None.

## Requirements

* Promote experiment 110 as the project default experiment口径, not only as a paper/reporting baseline.
* Preserve the distinction between:
  * project active runner default,
  * low-memory pure-CDM candidate/reporting baseline,
  * paper-style seed2027 ablation baseline,
  * hybrid/checkpoint-average evaluator routes.
* Do not present `c_prior_start001` as a promoted default.
* Make the chosen default口径 explicit enough for future AI sessions to follow without re-litigating exp104/106/110/111.
* Update durable docs/spec wording so future sessions treat exp110 as the current default pure-CDM training route unless explicitly instructed otherwise.
* Update the active trial runner so `scripts/run_assist09_history_alignment_trial.sh` encodes exp110 by default.
* Preserve `scripts/run_assist09_baseline.sh` as the experiment 81 baseline runner.

## Acceptance Criteria

* [x] The chosen default口径 is stated clearly in the handoff/current snapshot documentation.
* [x] Experiment 110 and experiments 111-113 are described with consistent baseline semantics.
* [x] The docs explicitly say experiment 110 is promoted as the active/default experiment口径.
* [x] The docs explicitly say `c_prior_start001` is not the default despite the seed2027 single-seed peak.
* [x] `scripts/run_assist09_history_alignment_trial.sh` defaults to exp110 settings: `branchBCE=0.18`, `recompute_minibatch`, `batch_size=65536`, `learning_rate=0.0003`, 300 epoch default through the baseline runner.
* [x] Validation at least covers shell syntax for active runner scripts and Python compilation for touched CLI path if needed.

## Definition of Done

* Relevant docs/spec entries are updated consistently.
* If docs only change, no model tests are required; verify with targeted text search.
* If runner code changes, run the relevant shell syntax checks and project validation from the backend experiment protocol.
* Commit plan is prepared after verification.

## Out of Scope

* Re-running experiments.
* Promoting hybrid stacker or checkpoint-average evaluator results as default training routes.
* Treating `c_prior_start001` as a stable improvement.

## Technical Notes

* `docs/handoff.md` currently says the active pure-CDM trial runner is experiment 104 and experiment 110 is the best `~7GB` candidate.
* `.trellis/spec/backend/experiment-protocol.md` currently says `scripts/run_assist09_history_alignment_trial.sh` encodes experiment 104 by default and experiment 106 is not promoted unless explicitly changed.
* `docs/model_improvement_plan.md` current snapshot already records experiment 110 as the current low-memory pure-CDM training candidate and experiment 111 as exp110-based ablation.
* `docs/experiments/111_exp110_ablation_study.md` defines `c_prior_start001` as replacing concept prior train start epoch with `1`.

## Decision (ADR-lite)

**Context**: The previous durable default separated the active project runner (experiment 104) from the later low-memory exp110 line. User now wants the project default口径 changed, not only temporary chat wording.

**Decision**: Promote experiment 110 as the project default experiment口径.

**Consequences**: Durable docs/spec and the active trial runner must stop describing exp104 as the current default. The implementation must still keep historical distinctions clear: exp104 remains a predecessor, exp106 remains an intermediate branch-BCE refinement before exp110, exp111-113 remain exp110-based ablations, and `c_prior_start001` remains a non-promoted near-tie timing probe.

## Technical Approach

* Change `scripts/run_assist09_history_alignment_trial.sh` defaults from exp104 to exp110:
  * output path name uses exp110/recompute wording,
  * branch BCE changes from `0.10` to `0.18`,
  * add `--training-mode recompute_minibatch`,
  * add `--batch-size 65536`,
  * add `--learning-rate 0.0003`.
* Update current-state docs and backend experiment protocol to call exp110 the active trial/default route.
* Update experiment 110 ledger wording from "do not promote yet" to "promoted as default口径 by this task", while preserving the exact metric caveat versus exp104/106.
* Keep hybrid stacker, checkpoint average, and `c_prior_start001` explicitly non-default.
