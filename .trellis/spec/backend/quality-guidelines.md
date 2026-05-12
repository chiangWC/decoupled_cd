# Quality Guidelines

> Code quality, experiment protocol, and review standards for this Python project.

---

## Overview

The highest-risk bugs in this repository are protocol drift, hidden data leakage, mismatched defaults, and experiment conclusions that cannot be traced later. Optimize changes for controlled comparison and durable evidence, not broad refactors.

---

## Required Patterns

- Use `from __future__ import annotations` in new Python modules unless matching an existing test file style.
- Prefer dataclasses for structured return values shared across modules. Examples: `StepDataBundle`, `TrainResult`, `PropagationOutput`.
- Prefer keyword-only public functions for multi-argument data/model orchestration.
- Keep tensor dtypes explicit where data enters PyTorch; labels and masks are generally `torch.float32`, IDs are `torch.long`.
- Keep adapters and residual heads zero-initialized when they are intended to preserve an existing baseline at initialization.
- When changing defaults or flags, search the repo first and update every relevant layer: CLI parser, `configs/defaults.py`, run scripts, JSON payload, CSV summary, tests, and docs.
- Preserve the current mainline protocol in [Experiment Protocol](./experiment-protocol.md) unless the active task explicitly changes it.

---

## Forbidden Patterns

- Do not run project tests, training, evaluation, or smoke tests locally by default. Commit and push first, then use `bash scripts/remote_exec.sh <command>`.
- Do not directly edit remote code on `xph-pc`; the remote is the execution environment, not the source-editing environment.
- Do not put exploratory experiment changes directly on `master`.
- Do not continue piling small final-logit residuals or sidecars as the default path toward the current AUC sprint target; docs mark representation-level changes as the default priority.
- Do not reintroduce valid/test target interactions into propagation history.
- Do not treat `dual graph` as current mainline; it is a legacy ablation path.
- Do not add broad refactors while testing a single experimental hypothesis.
- Do not leave experiment judgments only in chat or logs.

---

## Testing Requirements

This repo uses `unittest`, with tests under `tests/`.

Add or update tests when changing:

- Data schema validation or mapping behavior.
- Split/history visibility semantics.
- Training-mode validation or checkpoint behavior.
- Graph-mode validation.
- Model adapters, residual gating, zero-initialization, or detach behavior.
- Metric names or summary payload fields.

Examples:

- `tests/test_history_visibility.py` guards train-history reuse for valid/test.
- `tests/test_training_modes.py` guards `full_batch` and `recompute_minibatch` semantics.
- `tests/test_decoupled_cdm.py` guards zero-init and residual trigger behavior.
- `tests/test_hetero_propagation.py` guards graph-mode validation and concept aggregation behavior.

---

## Remote Verification

Default workflow:

1. Keep changes local until ready.
2. Commit the current branch.
3. Push the branch to `origin`.
4. Run verification remotely through `scripts/remote_exec.sh`.

Remote execution requires the remote branch to match local `HEAD` and the remote worktree to be clean.

---

## Experiment Documentation

After an experiment has a conclusion:

- Update `docs/experiment_index.jsonl` at minimum.
- Update `docs/model_improvement_plan.md` when the conclusion changes current route, candidate priority, or repeat-risk.
- Add or update `docs/experiments/<id>_*.md` when seed metrics, slice diagnostics, commands, or result paths matter.
- Update `docs/handoff.md` only when the conclusion changes current mainline state, default priority, or recommended branch.

---

## Code Review Checklist

- Does the change preserve the intended experimental control variable?
- Are defaults and CLI flags consistent across train/evaluate/run scripts?
- Are output JSON and summary CSV fields updated?
- Are history visibility and graph-mode invariants preserved?
- Are tests focused on the behavior that could regress?
- Are docs updated only at the appropriate ledger level without duplicating long metric tables?
