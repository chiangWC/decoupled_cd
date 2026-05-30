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


## Session 9: Clarify current trial handoff

**Date**: 2026-05-23
**Task**: Clarify current trial handoff
**Branch**: `exp/trellis-trial`

### Summary

Cleaned the handoff and experiment ledger wording so current active trial remains exp104, exp106 is documented as an unpromoted branch-BCE refinement, and post-104 work is framed as low-VRAM signal exploration rather than a default-promotion chain.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `1c33471` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 10: exp110 ablation study

**Date**: 2026-05-24
**Task**: exp110 ablation study
**Branch**: `exp/trellis-trial`

### Summary

Designed and ran the full exp110 pure-CDM ablation matrix excluding checkpoint averaging; recorded experiment 111 with aggregate results, conclusions, and ledger updates. Added CUDA peak-memory tracking to train summaries and fixed CUDA initialization.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `849a1c7` | (see git log) |
| `71e9898` | (see git log) |
| `ba63730` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 11: exp110 seed2027 component ablation

**Date**: 2026-05-24
**Task**: exp110 seed2027 component ablation
**Branch**: `exp/trellis-trial`

### Summary

Ran and recorded a paper-style seed2027 one-factor component ablation for exp110, reusing existing exp111 rows and adding missing base-runner component removals.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `76815ec` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 12: exp110 coarse module ablation

**Date**: 2026-05-25
**Task**: exp110 coarse module ablation
**Branch**: `exp/trellis-trial`

### Summary

Ran and recorded seed2027 coarse paper-style module ablations for exp110: evidence-aware readout, cognitive alignment objective, and dual-branch ensemble.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `09a1823` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 13: Record exp114 cross-dataset runs

**Date**: 2026-05-25
**Task**: Record exp114 cross-dataset runs
**Branch**: `exp/trellis-trial`

### Summary

Recorded exp114 cross-dataset ASSIST17, NIPS34, and Junyi runs, including Junyi memory and runtime findings.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `e900a94` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 14: Optimize Junyi propagation runtime

**Date**: 2026-05-28
**Task**: Optimize Junyi propagation runtime
**Branch**: `exp/trellis-trial`

### Summary

Added student-subset propagation and student_recompute_minibatch training mode for Junyi-scale runtime optimization; validated remote compile, unit tests, and Junyi smoke.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `816fc8e` | (see git log) |
| `4dcd318` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 15: Record exp114 Junyi memcheck result

**Date**: 2026-05-28
**Task**: Record exp114 Junyi memcheck result
**Branch**: `exp/trellis-trial`

### Summary

Updated exp114 with the Junyi student-subset full-run validation, including nvidia-smi process-memory peak from the memcheck rerun and removing the misleading 5.69GB memory wording.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `a6c5864` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 16: Promote exp110 default reporting baseline

**Date**: 2026-05-28
**Task**: Promote exp110 default reporting baseline
**Branch**: `exp/trellis-trial`

### Summary

Promoted exp110 as the default pure-CDM experiment reporting baseline, updated runner defaults and experiment ledger docs, verified shell syntax, Python compilation, and experiment index JSONL, then synced a sanitized GitHub copy to chiangWC/decoupled_cd.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `1d4d129` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 17: Coverage slice holdout validation

**Date**: 2026-05-29
**Task**: Coverage slice holdout validation
**Branch**: `exp/trellis-trial`

### Summary

Ran ASSIST09 coverage-slice diagnostics, expanded the student-concept holdout comparison to four seeds for Exp81 vs Exp110 full, and recorded experiment 115/116 ledger updates.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `7077c34` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 18: History hiding stress evaluation

**Date**: 2026-05-29
**Task**: History hiding stress evaluation
**Branch**: `exp/trellis-trial`

### Summary

Added a history-hiding stress evaluator, ran seed2027 evaluation perturbations on holdout and ordered ASSIST09 splits, and recorded experiment 117 plus evaluator contract.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `2f2e2f7` | (see git log) |
| `fb438a1` | (see git log) |
| `e2df711` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 19: Cross-dataset coverage and history stress

**Date**: 2026-05-29
**Task**: Cross-dataset coverage and history stress
**Branch**: `exp/trellis-trial`

### Summary

Extended Exp116/Exp117 to ASSIST17 and NIPS34 with reusable coverage evaluator, remote seed2024 training/evaluation artifacts, docs, and evaluator spec contract.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `08c3aeb` | (see git log) |
| `fccac3d` | (see git log) |
| `8cbd1e8` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 20: Junyi coverage and history stress supplement

**Date**: 2026-05-29
**Task**: Junyi coverage and history stress supplement
**Branch**: `exp/trellis-trial`

### Summary

Ran and recorded Junyi reduced-capacity Exp116/Exp117 supplements: generated holdout split, trained missing baseline/holdout models remotely, evaluated coverage slice and history hiding, and updated experiment docs/index with reduced-capacity caveats.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `d466ee1` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 21: TKC UKC gate diagnostic

**Date**: 2026-05-30
**Task**: TKC UKC gate diagnostic
**Branch**: `exp/trellis-trial`

### Summary

Added a gate diagnostic evaluator exposing tkc_weight, ran ASSIST09 ordered and holdout diagnostics, documented mixed mechanism evidence, and recorded the evaluator contract.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `b40e6ff` | (see git log) |
| `205e131` | (see git log) |
| `25c7c64` | (see git log) |
| `2fa0e4b` | (see git log) |
| `c5468c5` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 22: Remote R gate diagnostic plotting

**Date**: 2026-05-30
**Task**: Remote R gate diagnostic plotting
**Branch**: `exp/trellis-trial`

### Summary

Verified remote Rscript availability and generated PNG/PDF gate diagnostic figures from experiment 118 artifacts.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `bc25708` | (see git log) |
| `e9da59e` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 23: Exp110 capacity-control robustness follow-up

**Date**: 2026-05-30
**Task**: Exp110 capacity-control robustness follow-up
**Branch**: `exp/trellis-trial`

### Summary

Planned staged paper robustness follow-up, ran seed2027 single96 capacity control on ASSIST09 holdout, evaluated coverage slices and history hiding, recorded experiment 119 and updated experiment ledgers.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `c8381fa` | (see git log) |
| `12b9547` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
