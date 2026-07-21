# Eedi Tasks 1&2 option-signal extension preregistration

## Decision scope

This experiment does not revise or rescue the rejected `option_contrast_signal_v1`
gate. It prospectively adds a previously unused option-rich dataset and asks whether
the positive ENEM result replicates. The route may be useful on datasets that expose
categorical answer choices even though such fields are not universal across CD
benchmarks. Lack of option metadata elsewhere is an applicability limitation, not a
failure condition.

The route is activated for a neural module only when Eedi Tasks 1&2 passes the fixed
dataset-level signal gate below. The already frozen ENEM result is used without
refitting or threshold changes. EdNet remains positive but sub-threshold supporting
evidence; NIPS34 remains infeasible for its predeclared pseudo-target slice.

## Source and deterministic cohort

Use the official Eedi NeurIPS 2020 Tasks 1&2 files already present on the experiment
host:

- `train_task_1_2.csv`, SHA-256
  `721ebae1c5ddb3f8a4c85a437893216bbba1d8b2ca950ee0681d1f3e98ebdc0e`;
- `question_metadata_task_1_2.csv`, SHA-256
  `673aabe79e8dba2cf82e4bf87221796f2672c1ad264c9e08e2a99bf235704a12`;
- `subject_metadata.csv`, SHA-256
  `d576a6eccc171d8eb82a284a9586c27b2bd9941f39f3fcd6ae1110470766f202`.

The source contains 15,867,850 unique student-question rows, 118,971 students,
27,613 questions, four choices per question, complete option fields, and exact
agreement between `AnswerValue == CorrectAnswer` and `IsCorrect`. These descriptive
checks were completed before this document was frozen.

To keep the cross-fit diagnostic comparable with the frozen 5,000-student EdNet
snapshot, retain students with at least 30 rows and select exactly 5,000 by ascending
SHA-256 of `(2024, "eedi12-student", raw_user_id)`. This key excludes labels,
correctness, question identity, and selected option. All rows for selected students
are retained; the source already has one row per student-question pair. Raw IDs are
mapped to sorted dense IDs. Every question uses all official `SubjectId` values as
its Q concepts; subject hierarchy or answer metadata is not used by the estimator.

The generated cohort is an internal train-only signal protocol, not yet a formal
external benchmark. It does not establish an external win and must not be used to
select a test result.

## Frozen signal experiment

Run the unchanged estimator, controls, pseudo-target construction, five
student-disjoint folds, and seeds from `audit_option_contrast_signal.py`:

- model seed `42`, pseudo split seed `2024`;
- hide `max(3, floor(0.20*n))` rows per eligible student by a label-free stable key;
- keep at least ten history rows;
- target scope `low_coverage`, meaning train-history concept coverage below `0.5`;
- Full uses train-history wrong-option identity;
- Direct uses correctness/Q history only;
- Shuffle has the same option feature interface but breaks student-specific option
  identity while preserving item/correctness marginals;
- choose the stronger complete control once by out-of-fold target AUC.

Eedi passes only if Full relative to that fixed stronger control has:

- target AUC improvement at least `0.005`;
- overall AUC regression no worse than `0.001`;
- target Brier regression no worse than `0.0002`;
- a 2,000-replicate student-clustered paired-bootstrap 95% CI lower bound above
  zero.

The frozen ENEM summary is
`results/option_contrast_signal_v1/enem_summary.json`, SHA-256
`0f255d1383696f0c96eb7ad5caf6ceab90f798bd84a3b6c66f05bb15ada8d1aa`.
Its fixed Full-minus-Shuffle deltas are `+0.010376` target/overall AUC and
`-0.003108` target Brier.

## Consequence

If Eedi passes, ENEM and Eedi jointly activate literature review and implementation
of one complete categorical-response state module. This does not fix the final
number of paper modules: later mechanisms may be retained if each independently
improves external performance and has a clean, material ablation. If Eedi fails,
the option route remains an ENEM-specific diagnostic and no neural module is built
from it in this round. Thresholds, cohort size, Q construction, target scope, or
controls are not changed after observing the result.
