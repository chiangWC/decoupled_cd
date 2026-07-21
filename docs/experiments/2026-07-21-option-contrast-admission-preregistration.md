# Option-Contrast Admission and Signal-Gate Preregistration

## Decision

The next route tests whether categorical response identity supplies information that
binary correctness and Q-matrix membership discard. A wrong answer is not treated as
an interchangeable zero: different selected options may identify different
misconception states. This route is admitted only if that additional train-history
signal is available on multiple datasets and improves held-out prediction before a
neural framework module is implemented.

This is a prospective gate for implementation, not a claim of blind discovery. The
repository and raw fields were inspected before this document was frozen. All data
admission thresholds, predictive thresholds, controls, and stop rules below are fixed
before running the formal audit and signal results.

## Scope and provenance

Three datasets are in the formal option-aware pool:

- NIPS34: use the existing recent KnoField standard and `nips34_chold_v2`
  protocols. Recover selected choice and per-interaction correct choice from the
  original Task 3/4 interaction file, and require an exact raw-to-protocol row
  multiset match.
- EdNet: use the frozen 10,000-user KT1 snapshot and official EdNet KT1 files. Select
  5,000 eligible students by a stable SHA-256 ranking of
  `(split_seed=2024, dataset, raw_user_id)`; keep each student's first
  student-item attempt. Every selected snapshot user's model-relevant fields
  (timestamp, question, and selected answer) must match the corresponding official
  KT1 file, and each extracted file must match the CRC and size recorded in the
  hash-verified official archive.
- ENEM: use the frozen `enem_dep.zip` source from the DP-MTL release, retain
  students with at least 30 unique item interactions, and use each item's answer
  choice as the concept. The source archive is used for internal research only;
  redistribution and paper-release terms must be resolved separately because the
  source repository does not state an explicit license.

ASSIST09 is excluded from the formal option pool. Its native `answer_id` coverage is
too sparse and its `answer_text` is largely constructed response, so coercing it
into distractor identities would change the research question.

Every source file, source repository revision where applicable, generated protocol,
Q-matrix, split, holdout assignment, dense-ID map, option sidecar, and raw-to-dense
row map is recorded with a cryptographic hash. Generated `source_row_id` values
depend only on student and item identity, never on label, correctness, or selected
option. Formal generation must run from a clean committed worktree, and the manifest
records both the commit and preparation-script hash. The preparation seed is fixed
at 2024; the model seed remains 42.

## Leakage boundary

Option identity is a history-only input:

- model-accessible option sidecars contain training rows only;
- validation/test selected options are never exposed to the model or feature
  constructor;
- concept-holdout target membership is derived only from training history;
- all interactions for a student-item pair are atomic, and preprocessing keeps one
  deterministic first attempt for new protocols;
- no free student-ID embedding, dataset-ID routing, target label, target selected
  option, or test statistic may enter the model.

A separate full-row option file may exist solely under an audit-source directory.
Its manifest entry must be marked `training_access=false`, and training code must
not accept that path.

## Data-admission gate

Each formal dataset must satisfy all of the following:

- selected-option coverage at least 0.995;
- all available option values are integer and in range; agreement between
  `selected_option == correct_option` and binary label, conditional on valid option
  rows, is at least 0.999;
- in both standard and holdout training splits, at least 90% of rows belong to
  item-option cells observed for at least five different students;
- in both training splits, at least 100 items have at least two distinct wrong
  options, each selected by at least five different students;
- target slice has at least 500 rows, 100 students, 100 positive rows, and 100
  negative rows;
- at least 90% of target students have at least ten valid option-bearing
  training-history rows, and at least 80% have at least three incorrect
  option-bearing history rows;
- standard and holdout protocols have identical underlying interaction rows and
  compatible ID/Q mappings;
- raw-to-generated provenance, hashes, and cross-split integrity all pass.

The target is low coverage for NIPS34 and exact zero for EdNet and ENEM. Validation
target composition determines admission; test composition is recorded descriptively
but does not activate or reject the route. Test results therefore remain
validation-driven confirmation rather than blind evaluation. The route continues
only if all three datasets pass. A failed dataset is not replaced after seeing the
audit result unless the failure is a protocol-integrity bug rather than a threshold
failure.

## Train-only predictive signal gate

The formal signal gate uses only the concept-holdout protocol's training
interactions. For each student, let `k=max(3, floor(0.20*n))`; the `k` student-item
groups with the lowest SHA-256 key over
`(split_seed, dataset, stu_id, exer_id)` become pseudo-targets, provided at least ten
history rows remain. The key excludes label, correctness, and option identity.
Coverage is recomputed from the remaining history: NIPS uses low coverage and
EdNet/ENEM use exact zero. All hidden rows form the pseudo-overall set, while rows in
that coverage scope form pseudo-T.

Five student-disjoint cross-fit folds are assigned by a stable student-only hash;
this is not a multi-seed experiment. Global statistics and estimator parameters for
a held fold are fitted on the other four folds only. A held student's pseudo-target
label and selected option are never inputs.

Three fixed estimators share the same folds and output interface:

- Full: uses the student's real history option identities and train-fitted
  option-to-concept misconception signatures.
- Direct: uses only binary correctness and Q-conditioned history summaries.
- Shuffle Capacity: has access to the same option fields and matched feature
  capacity, but deterministically resamples option identity from
  `P(option | item, correctness)`, breaking student-specific option identity while
  preserving item/correctness marginals.

For each dataset, the ablation baseline is the complete control with higher
pseudo-T AUC, not a per-metric mixture. Delta-T is Full minus that fixed control on
pseudo-T; pseudo-overall AUC uses all hidden rows, and Brier is measured on pseudo-T.
Ties select Direct. Student-clustered paired bootstrap confidence intervals use the
fixed predictions and do not constitute multiple model seeds.

The option signal passes only if EdNet and at least one of NIPS34 or ENEM satisfy the
following joint evidence:

- Full improves pseudo-T AUC by at least 0.005 on two datasets;
- at least one of those improvements is at least 0.010;
- at least one student-clustered paired-bootstrap 95% confidence-interval lower
  bound is greater than zero;
- pseudo-overall AUC does not regress by more than 0.001 on any passing dataset;
- Brier score does not regress by more than 0.0002 on any passing dataset.

Failure is recorded unchanged. Thresholds, folds, pseudo-target masks, or controls
must not be relaxed to rescue the route.

## Module admitted only after the signal gate

Passing the gate activates one complete Categorical-Response State Completion
Module. Its input is train-only item/Q/selected-option/correctness history plus option
alternatives. Its output is the sole student-specific `framework_state`, complete
mastery, reliability, and diagnostics consumed by a fixed standard Diagnosis.
Selected-versus-unchosen contrast and concept-specific misconception composition are
internal mechanisms, not separate paper modules.

The clean module comparison is fixed in advance:

- Full: categorical-response contrast and complete state construction;
- Direct: binary-correctness state construction;
- Capacity Control: the same option information and parameter budget, but a flat
  option encoder/MLP without selected-versus-unchosen contrast or concept-specific
  state composition;
- objective_off: the Full data flow with any module-specific auxiliary objective
  disabled.

No legacy state, residual prediction path, second head, or student-ID shortcut is
allowed. External comparisons for new EdNet/ENEM protocols must be rerun on identical
rows. Binary CD baselines and option-aware baselines such as CRKT or Option Tracing
are external competitors; their authors' implementations are not the source of our
module.
