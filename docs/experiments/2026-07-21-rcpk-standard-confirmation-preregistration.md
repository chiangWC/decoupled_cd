# RCPK standard-only confirmation preregistration

## Scope change and evidential status

The original RCPK screen declared concept-holdout stability as part of the
module responsibility and rejected the candidate after the holdout effect
collapsed. That decision remains valid and is not overwritten by this round.

After inspecting the valid results, the research scope is explicitly changed
to standard, in-distribution cognitive diagnosis. The already observed
ASSIST09 and NIPS34 standard results are discovery evidence, not a fresh blind
confirmation. This round asks the narrower question of whether verified static
relation semantics, rather than extra graph capacity, produce a reproducible
standard-validation gain and enough external wins for a standard-only paper
route.

RCPK is therefore a usable candidate component, not yet a qualified paper
module. No claim about TKC/UKC completion, cold start, or concept-holdout
robustness is made in this scope.

## Frozen model and data flow

The implementation is frozen at commit `18fcae13`:

- model seed 42 and split seed 2024;
- four path steps and item-ease shrinkage 20;
- 20% train-history context-target masking;
- response BCE only;
- factorized item/Q target requirement;
- target-conditioned diagnosis;
- no student-ID embedding, residual prediction head, learned relation gate,
  auxiliary objective, or dataset router.

The module consumes the student's unordered train-only responses, Q incidence,
item ease and a typed static-relation graph. It outputs the unique
target-specific student state consumed by Diagnosis.

## Confirmation controls

For ASSIST09 and NIPS34, compare the frozen real-relation Full model with:

1. Q-only Direct;
2. three deterministic relation-type and exact-degree-preserving rewires.

All five variants have identical trainable parameters, optimizer, data order,
mask, initialization hash and dataset recipe. Graph transitions are
non-parameter buffers. The strongest control is the control with the highest
standard-validation AUC; it is selected once per dataset. No test artifact is
opened.

The real-relation semantics gate passes only if:

- Full strictly exceeds all four controls on both discovery datasets;
- `Full - strongest_control >= 0.005` on at least one dataset;
- the other dataset has `Full - strongest_control >= 0.002`;
- Full does not regress in Brier by more than `0.0002` against the strongest
  control on either dataset.

These thresholds are frozen before any neural rewire run. The existing
train-only logistic signal screen is not substituted for the neural controls.

## Pool expansion

If the semantics gate passes, export provenance-locked real and rewired graphs
for XES3G5M and Junyi and run the same standard-validation comparison. A dataset
enters the standard-only winning set when Full:

- strictly exceeds its strongest complete control by at least `0.002` AUC;
- has a student-clustered paired-bootstrap 95% CI lower bound above zero;
- strictly exceeds the current row-aligned external standard-validation AUC;
- does not regress ACC, RMSE and Brier simultaneously.

RCPK qualifies as a standard-only paper module only if one architecture obtains
at least three such external standard wins, with at least one module effect of
`0.005` or larger. Existing holdout results remain reported as a limitation and
cannot be presented as positive evidence.

If only two external standard wins are reached, RCPK remains a performance
component and may be combined with a second independently motivated component
only when the second component targets the identified absolute-performance
deficit and has its own clean ablation. If fewer than two wins are reached, the
RCPK route is stopped without tuning or renaming.

## Execution and integrity

- Experiments run only on the xph GPU workspace in the `decoupled_cd`
  environment.
- Standard results use validation rows as both validation and evaluation input;
  `test_metrics` must remain null.
- Target responses are removed from all history features before prediction.
- Save summary JSON, checkpoint, history, row-aligned predictions and hashes.
- Run unit tests, `compileall` and `git diff --check` before formal execution.
- OOM is an explicit failure and cannot silently change batch size or recipe.
