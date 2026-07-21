# History Set Representation standard-only audit preregistration

## Motivation and scope

The frozen RCPK model already contains an upstream History Set Representation
box, but it has not been qualified under the new standard-only objective. Its
earlier concept-holdout gate remains rejected and is not reopened. This audit
asks whether the existing component contributes independently to standard
prediction when RCPK is fixed.

No new architecture or hyperparameter is introduced. This is a factorial
audit of an existing framework component before designing another module.

## Module contract

Inputs are the student's unordered train-only attempted items, binary outcomes,
train-only item correctness and history confidence. The module outputs one
student history state consumed by state construction and the frozen RCPK
composer.

The four frozen variants are:

| Variant | Attempted-item identity | Difficulty residual | Capacity |
|---|:---:|:---:|:---:|
| Full `calibrated_history` | yes | yes | matched |
| Identity control `identity_raw_control` | yes | no | matched |
| Summary control `calibrated_summary_control` | no | yes | matched |
| Direct control `raw_summary_control` | no | no | matched |

All encoders have the same shape. RCPK uses the real admitted relation graph in
every variant; target requirement, Diagnosis, context-target masking, optimizer,
data order and seed remain fixed. The strongest of the three controls is chosen
once per dataset by standard-validation AUC.

## Execution and gate

Reuse the current Full results. Train the three controls on ASSIST09, NIPS34,
XES3G5M and Junyi standard validation only. No holdout or test is opened.

The History Set Representation becomes a second paper-module candidate only if:

- Full exceeds the strongest control by at least `0.002` on at least two of the
  three external-winning datasets ASSIST09, XES3G5M and Junyi;
- at least one winning dataset has a delta of `0.005` or larger;
- at least one winning dataset has a student-clustered paired-bootstrap 95% CI
  lower bound above zero;
- none of the three external standard wins is lost relative to the strongest
  control because of the module;
- Full beats both the identity-preserving and calibration-preserving controls,
  preventing either item identity or difficulty calibration alone from being
  credited as the whole mechanism.

NIPS34 is retained as a rescue diagnostic: a positive module delta is useful,
but it cannot replace the two-winning-dataset requirement because Full remains
below the external NIPS34 line.

Failure leaves RCPK as the only retained paper-module candidate. The History
component then remains ordinary model plumbing and cannot be renamed, tuned or
claimed from a weaker control.

## Integrity

- Fixed model seed 42 and existing dataset recipes.
- RCPK implementation, real graphs and four-hop statistic are unchanged.
- Full/control active parameter counts and initialization hashes must match.
- `evaluation_stage=validation`; all `test_metrics` must remain null.
- Save summaries, checkpoints, histories and row predictions for any bootstrap
  comparison.
- Run `compileall` and `git diff --check`; OOM cannot change a recipe.
