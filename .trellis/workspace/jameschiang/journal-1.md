# Journal - jameschiang (Part 1)

> AI development session journal
> Started: 2026-05-12

---



## Session 1: Finalize Trellis initialization

**Date**: 2026-05-12
**Task**: Finalize Trellis initialization
**Branch**: `exp/trellis-trial`

### Summary

Initialized project-local Trellis constraints, moved execution workflow into Trellis specs, slimmed legacy docs into compatibility indexes, and archived the bootstrap guidelines task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `15d035a` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 2: Promote Exp70 Single-Only Evidence Readout

**Date**: 2026-05-16
**Task**: Promote Exp70 Single-Only Evidence Readout
**Branch**: `exp/trellis-trial`

### Summary

Added concept_evidence_readout_max_count, validated the exp70 single-only evidence readout with remote unit tests, smoke, three-seed training, and slice analysis, promoted experiment 81 to exp/trellis-trial, and synced the pseudo-mainline contract plus ledger docs.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `765bc68` | (see git log) |
| `cf56641` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: Sync experiment ledger back to trial

**Date**: 2026-05-17
**Task**: Sync experiment ledger back to trial
**Branch**: `exp/trellis-trial`

### Summary

Committed experiment 82 and 83 ledger docs, synced the doc-only ledger commit back onto exp/trellis-trial, and restored trial as the durable bootstrap source for future experiment branches.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `059276c` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 4: Pure CDM history alignment trial

**Date**: 2026-05-18
**Task**: Pure CDM history alignment trial
**Branch**: `exp/trellis-trial`

### Summary

Merged the loss-only history-evidence cognitive alignment trial path into exp/trellis-trial, fixed exercise evidence bundle support, validated the cogonly CF-risk ablation, set the active pure-CDM trial runner to cogonly, and cleaned the experiment ledger/handoff so experiments 93, 94, and 95 have separate docs and concise routing.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `7e91b0b` | (see git log) |
| `ea1e23e` | (see git log) |
| `a198cb4` | (see git log) |
| `7bccb4d` | (see git log) |
| `5e96486` | (see git log) |
| `ebe6cef` | (see git log) |
| `69a9bde` | (see git log) |
| `3301515` | (see git log) |
| `b97cacc` | (see git log) |
| `29c23a6` | (see git log) |
| `223ecf0` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 5: Pure CDM dual ensemble promotion

**Date**: 2026-05-20
**Task**: Pure CDM dual ensemble promotion
**Branch**: `exp/trellis-trial`

### Summary

Recorded experiment 104, merged the single-checkpoint dual CDM ensemble with branch BCE into exp/trellis-trial, verified focused tests remotely, and updated handoff/model improvement ledger to make experiment 104 the active pure-CDM default-training candidate.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `4400e6c` | (see git log) |
| `305c1dd` | (see git log) |
| `fb1542b` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 6: Clean model evidence helpers

**Date**: 2026-05-21
**Task**: Clean model evidence helpers
**Branch**: `exp/trellis-trial`

### Summary

Refactored DecoupledCDM evidence residual helpers to share target evidence statistics, mastery/confidence summaries, trigger masks, and zero-init logic. Verified py_compile, diff check, and remote tests/test_decoupled_cdm.py.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `9660809` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 7: Cleanup active ASSIST09 trial CLI

**Date**: 2026-05-22
**Task**: Cleanup active ASSIST09 trial CLI
**Branch**: `exp/trellis-trial`

### Summary

Removed obsolete ASSIST09 runner wrappers, promoted exp104 as the active trial runner default, cleaned non-trial train.py CLI flags, updated trial docs/specs, and verified baseline/exp104 remote smoke plus full exp104 GPU rerun.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `70c1f7f` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 8: Cleanup non-trial CDM internals

**Date**: 2026-05-22
**Task**: Cleanup non-trial CDM internals
**Branch**: `exp/trellis-trial`

### Summary

Trimmed retired DecoupledCDM constructor/model branches after CLI cleanup, kept baseline and exp104 paths, updated exp104/model-surface docs, and verified focused tests plus baseline/exp104 remote smoke.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `4e5d24c` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
