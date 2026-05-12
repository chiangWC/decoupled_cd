# Error Handling

> Fail fast on invalid experiment configuration and data contracts.

---

## Overview

This project has no API error response layer. Errors are raised directly from Python scripts, data loaders, trainer code, and model modules. Prefer early validation with clear `ValueError` messages over downstream tensor shape failures or silent fallback behavior.

---

## Error Types

- Use `ValueError` for invalid CLI combinations, unsupported modes, missing required inputs, and violated data-flow invariants.
- Use standard library exceptions naturally for missing files and malformed CSVs unless a clearer project-specific message is needed.
- Do not add custom exception hierarchies unless multiple callers need to branch on the error type.

Examples:

- `scripts/train.py::validate_graph_args` rejects dual-graph inputs in single-graph mode.
- `models/hetero_propagation.py` validates `graph_mode` and dual-graph tensor requirements.
- `trainers/engine.py::train_model` rejects unsupported `training_mode`, invalid `batch_size`, and incompatible `full_batch`/`batch_size` combinations.
- `trainers/engine.py::_validate_history_visibility` rejects target leakage in evaluation history.

---

## Error Handling Patterns

- Validate arguments at the boundary, before running training or model forward passes.
- Keep error messages specific enough to tell the user which flag, mode, or invariant is wrong.
- Do not catch broad exceptions around training, evaluation, or preprocessing; failed experiments should fail visibly.
- Narrow fallback handling is acceptable only for non-critical convenience paths. Example: `utils/io.py::append_summary_csv` recovers from a malformed existing summary CSV by starting a new row frame.

---

## CLI Validation

Scripts should reject contradictory or legacy argument combinations:

- `single` graph mode must not accept `--prerequisite-graph` or `--similarity-graph`.
- `dual` graph mode requires both legacy dual-graph inputs.
- `full_batch` training must not consume `--batch-size`.
- `recompute_minibatch` training requires `--batch-size`.
- Deprecated aliases such as `--alpha` and `--beta` may be preserved, but conflicting new and legacy values must raise.

---

## Common Mistakes

- Do not silently switch graph modes or training modes to make a run proceed.
- Do not hide remote execution failures; `scripts/remote_exec.sh` is expected to fail when the local branch has not been pushed or the remote worktree is dirty.
- Do not return default metrics for invalid labels or predictions except the existing `auc=0.5` fallback for degenerate ROC-AUC computation.
