# Trusted Unified Validation Launch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task with TDD.

**Goal:** Make only controller-launched outer campaigns capable of producing
validation progress, with crash-safe capabilities and single-open immutable
snapshots.

**Architecture:** `unified_validation_controller.py` registers validation data
and artifact roots, journals issuance/launch, constructs the two fixed outer
commands, directly runs them, and snapshots the unique new attempts before
advancing. `run-split` is an internal capability consumer usable only during an
active launch.

**Tech stack:** Python standard library, existing remote campaign runner,
PyTorch/unittest fixtures.

## Constraints

- No real experiments, test-data reads, pushes, or caller proof paths.
- Any launched-pair failure/interruption permanently blocks the iteration.
- Every JSON/data proof hash and parse derives from one stable descriptor.
- All controller writes are fsynced, atomic, journaled, and flocked.

### Task 1: Stable snapshot and exact registration

**Files:** `scripts/unified_validation_controller.py`,
`tests/test_unified_validation_controller.py`, `scripts/run_unified_validation.py`.

- [ ] Add failing TOCTOU tests that replace summary/status between old hash and
  parse, malformed/duplicate dataset manifest tests, and baseline test-token
  rejection.
- [ ] Implement `snapshot_json(path)` and `snapshot_file(path)` using one fd and
  stable before/after `fstat`; route all proof validation through snapshots.
- [ ] Extend `controller-init` with required validation data/artifact roots and
  copy exact no-test data fingerprints into state.
- [ ] Verify focused GREEN.

### Task 2: Crash-safe capability lifecycle

**Files:** `scripts/unified_validation_controller.py`,
`tests/test_unified_validation_controller.py`.

- [ ] Observe failure-injection RED for consumed publish, issued cleanup, and
  issuance interruptions.
- [ ] Publish a complete consumed record atomically before issued cleanup;
  consumed existence wins recovery.
- [ ] Add pending issuance journal/token payload and idempotent recovery/re-emit
  for every interruption boundary.
- [ ] Verify forged/reuse/stale/route and crash tests GREEN.

### Task 3: Controller-owned run-pair provenance

**Files:** `scripts/unified_validation_controller.py`,
`scripts/run_unified_validation.py`, `tests/test_unified_validation_controller.py`.

- [ ] Build a controlled temporary outer-runner fixture that really creates a
  new attempt, running/final status, summary, and hashes via `Popen`.
- [ ] Observe RED for manual run-split consumption/progress, caller status,
  nonunique attempts, wrong command, child nonzero, and controller interruption.
- [ ] Add `run-pair` CLI with no caller paths. Journal the exact fixed commands,
  Popen both splits, require exit zero and one new registered attempt, then
  snapshot/freeze proof in the same controller process.
- [ ] Make failure/interruption permanently block; prohibit later reconstruction.
- [ ] Verify provenance tests GREEN.

### Task 4: Proof schema and full gates

**Files:** `scripts/unified_validation_controller.py`,
`tests/test_unified_validation_controller.py`, `docs/unified_v2_validation_log.md`.

- [ ] Add RED tests for mastery shape, finite loss, positive mastery weight,
  valid-only routing, every independent five-gate failure/equality boundary,
  and positive/negative final `0.001` cases.
- [ ] Implement strict summary/status schema and full joint/final gates.
- [ ] Remove any caller-driven progress path and update ledger/report wording.
- [ ] Verify focused tests GREEN.

### Task 5: Verification and delivery

- [ ] Run compileall and `git diff --check`.
- [ ] Run focused controller/cohort/runner tests.
- [ ] Run the full unittest suite with zero failures.
- [ ] Update `.superpowers/sdd/task-8-fix-report.md` with RED/GREEN evidence.
- [ ] Commit as chiangWC, verify exact HEAD/clean tree, rerun focused tests, and
  report hashes without push/experiment.
