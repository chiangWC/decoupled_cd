# PRD: Record exp114 Junyi Validation Result

## Goal

Update the exp114 experiment note with the completed Junyi
`student_recompute_minibatch` full-run validation result.

## Scope

- Edit `docs/experiments/114_cross_dataset_exp110_runs.md`.
- Preserve the existing exp114 old-run record.
- Add the new runtime, memory, and metric comparison for the optimized Junyi
  training mode.
- Make clear that `student_recompute_minibatch` is a training-mode/runtime
  optimization, not the original exp110 `recompute_minibatch` protocol.

## Acceptance

- The exp114 document includes the new result JSON path, runtime, peak memory,
  core metrics, best epoch, and comparison against the old Junyi record.
- The document states whether the new mode affects the other dataset records.
- No code changes are made.
