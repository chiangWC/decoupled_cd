# Pure CDM AUC 0.778 Sprint

## Goal

Continue pure-CDM-only exploration toward `test_auc >= 0.778`, with `0.780` as desirable headroom. Prefer stable single-run or default-runner-promotable signals over checkpoint-average, hybrid stacker, or validation-trained combiner paths.

## Context

- The current branch is `exp/pure-cdm-default-promotion`, a descendant Trellis-enabled experiment branch.
- The prior accepted reference remains experiment 95: `scripts/run_assist09_history_alignment_trial.sh`, `loss_only cogonly`, four-seed mean `test_auc = 0.776279`.
- Experiment 100 showed dim80 + weak output alignment can hit `0.778+` on seeds 2026/2027 but regressed seeds 2024/2025.
- Experiment 101 rejected nearby output-alignment/confidence/weight-decay/capacity/lr/readout/difficulty-initialization follow-ups as insufficient to fix the tail.
- Experiment 102 checkpoint average reached a strong pure-CDM evaluator mean, but the user rejected it as the default-training answer.
- The user explicitly permits free exploration, parameter validation, and aggressive changes until a valid growth or stability signal appears.

## Requirements

1. Preserve pure CDM acceptance semantics: no hybrid stacker, no validation-trained combiner, no train-history tabular inference side-channel.
2. Prefer single-run/single-checkpoint training candidates or model/training changes that could plausibly become default runner behavior.
3. Search before changing flags, defaults, metric names, or repeated run-script patterns.
4. Add focused unit tests for new flags, target construction, validation behavior, or model plumbing.
5. Commit and push before remote tests, smoke runs, or training through `bash scripts/remote_exec.sh`.
6. Run meaningful remote probes after implementation.
7. Record experiment conclusions in `docs/experiment_index.jsonl` and detail/model-plan docs when results change route decisions.

## Acceptance Criteria

- A pure-CDM-only candidate or validation probe is implemented or configured.
- Focused tests pass remotely for touched code paths.
- At least one meaningful remote pure-CDM probe runs and is compared against experiment 95/high-water results.
- Growth signals, null results, or stability signals are documented with commands and artifact paths.

## Out Of Scope

- Promoting `scripts/evaluate_ensemble.py`, hybrid feature stackers, or valid-trained stackers.
- Treating experiment 102 checkpoint averaging as the single-run default-training solution.
- Reintroducing valid/test target rows into propagation history.
- Broad unrelated refactors that do not help isolate the next AUC signal.
