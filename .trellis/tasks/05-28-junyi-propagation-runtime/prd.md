# Optimize Junyi propagation runtime

## Goal

Make exp110-style Junyi experiments practical by reducing the repeated dense
`students x concepts x dim` propagation cost during minibatch training and
evaluation, while preserving the current model semantics for existing runs.

## What I already know

- The user asked to execute the optimization and fully validate it.
- Experiment 114 showed Junyi's successful reduced-capacity run still took
  about 1h29m, even after lowering dimensions to fit memory.
- The documented bottleneck is dense student-concept propagation, not the
  interaction row count: Junyi has `10000 x 706` student-concept cells.
- Current `recompute_minibatch` training calls model forward once per
  interaction minibatch, and each forward currently computes propagation for
  all students.
- `models/hetero_propagation.py` builds full `tkc_states` and `ukc_states` with
  shape `(num_students, num_concepts, dim)`.
- `models/decoupled_cdm.py` only needs states for the target students for
  prediction, while some auxiliary/residual paths also inspect target-student
  concept states.

## Assumptions

- The highest-value optimization is a model-side student-subset/chunked
  propagation path, not just a runner shortcut.
- Existing default full propagation behavior should remain unchanged when no
  subset is requested.
- The optimized path may be used by minibatch training and evaluation; full
  batch training can continue using the existing full propagation path.

## Requirements

- Add a student-subset propagation path that computes TKC/UKC state tensors only
  for the unique target students needed by a forward pass.
- Preserve predictions and losses for target interactions compared with the
  full propagation path, within normal floating-point tolerance.
- Wire recompute-minibatch training to use the subset path.
- Wire evaluation to use chunked/subset prediction so Junyi-sized datasets avoid
  unnecessary all-student propagation per chunk.
- Keep current result fields and CLI behavior compatible.
- Avoid changing the accepted exp110/exp114 modeling contract unless explicitly
  documented.

## Acceptance Criteria

- [ ] Unit tests prove subset propagation predictions match full propagation
      for the same target interactions.
- [ ] Unit tests cover dual-tower ensemble subset forwarding if applicable.
- [ ] Existing training-mode and CLI tests pass.
- [ ] A small smoke command runs successfully after the change.
- [ ] Documentation or experiment notes are updated if the runtime path changes
      user-visible behavior.

## Definition of Done

- Tests added or updated for the optimized path.
- Focused Python compile checks pass.
- Focused unit tests pass.
- Broader relevant test suite passes where practical.
- `git status` reviewed and only task-related files changed.

## Out of Scope

- Re-running full Junyi 300-epoch experiments in this local session.
- Changing model defaults or promoting a new experiment result.
- Adding hybrid side channels, checkpoint averaging, or validation-trained
  combiners.

## Technical Notes

- Likely files: `models/hetero_propagation.py`, `models/decoupled_cdm.py`,
  `models/ensemble_cdm.py`, `trainers/engine.py`, `scripts/train.py`, tests.
- Existing exp114 doc: `docs/experiments/114_cross_dataset_exp110_runs.md`.
- Existing propagation code: `models/hetero_propagation.py`.
- Existing minibatch training loop: `trainers/engine.py`.
