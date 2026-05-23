# Clean current trial handoff

## Goal

Clean the current trial handoff so a new session can identify the active trial line without reading a long mixed ledger. Reconcile the runner, Trellis spec, and handoff around the latest accepted pure-CDM trial candidate.

## What I Already Know

- `docs/handoff.md` currently mixes accepted `master`, `exp/trellis-trial` pseudo-mainline, pure-CDM trial runner, checkpoint-average evaluator, hybrid evaluator, rejected branches, and future rules in the same "current mainline" flow.
- `docs/handoff.md` and `docs/model_improvement_plan.md` say experiment 106 is the current pure-CDM single-run/single-checkpoint trial candidate.
- `scripts/run_assist09_history_alignment_trial.sh` still defaults to experiment 104 branch BCE `0.10`.
- `.trellis/spec/backend/experiment-protocol.md` still describes the current pure-CDM trial runner as experiment 104 with branch BCE `0.10`.
- User clarified that experiments after 104 exist because the dual-tower route is too memory-heavy for the desired next direction. The desired follow-up is to find more stable or higher-peak pure-CDM signals around the previous `~7GB` memory level, not to blindly promote heavier dual-tower refinements.

## Requirements

- Keep `scripts/run_assist09_history_alignment_trial.sh` aligned with the current code contract: experiment 104 by default.
- Update Trellis backend experiment protocol so it clearly distinguishes the active experiment 104 runner from experiment 106's unpromoted branch-BCE refinement candidate.
- Rewrite `docs/handoff.md` as a concise handoff, separating:
  - accepted `master` reference,
  - `exp/trellis-trial` pseudo-mainline,
  - active pure-CDM trial runner,
  - the low-VRAM follow-up intent after experiment 104,
  - non-default evaluator signals,
  - rejected / paused routes.
- Keep detailed experiment history in `docs/experiment_index.jsonl`, `docs/model_improvement_plan.md`, and `docs/experiments/*.md`; do not duplicate long result ledgers in handoff.

## Acceptance Criteria

- [x] A search for current trial defaults clearly shows exp104/`0.10` as the active runner contract, not an accidental stale value.
- [x] Experiment 106 is documented as a candidate/refinement, not silently substituted as the runner default.
- [x] The handoff top section answers "what is the current trial mainline?" in a few bullets.
- [x] Shell syntax check passes for the updated runner.

## Out Of Scope

- Re-running remote full training.
- Changing `scripts/run_assist09_baseline.sh`.
- Promoting hybrid stacker or checkpoint-average evaluator to default training.
- Editing historical experiment result facts except where current-runner wording is stale.

## Technical Notes

- Current branch: `exp/trellis-trial`.
- Relevant files:
  - `scripts/run_assist09_history_alignment_trial.sh`
  - `.trellis/spec/backend/experiment-protocol.md`
  - `docs/handoff.md`
  - `docs/experiments/104_single_checkpoint_dual_cdm_ensemble.md`
  - `docs/experiments/106_dual_cdm_branch_bce_refinement.md`
