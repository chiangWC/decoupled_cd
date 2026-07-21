# RCPK standard test confirmation preregistration

## Frozen claim

RCPK has four standard-validation external wins. Real relations have a clean
effect above `0.002` on ASSIST09, NIPS34, and Junyi; XES3G5M is retained as an
honestly reported architecture win but not as RCPK attribution evidence.

The standard-only architecture, per-dataset recipes, validation-selected
checkpoints, controls, graph files, seed 42, and external registry are now
frozen. Concept holdout and TKC/UKC completion are outside this claim.

## Confirmation set

Evaluate each already selected validation checkpoint once on the untouched
standard test rows. No retraining or checkpoint reselection is allowed.

| Dataset | Full checkpoint | Frozen control | External test line |
|---|---|---|---:|
| ASSIST09 | `rcpk_masked_stage1/a09_std_full` | Q-only | SVGCD 0.778200 |
| NIPS34 | `nips_standard_recovery/lr2e3` | Q-only | ORCDF 0.789300 |
| XES3G5M | `rcpk_standard_expansion/xes_full` | rewire 0 | ORCDF 0.792500 |
| Junyi | `rcpk_standard_expansion/junyi_full` | rewire 2 | ORCDF 0.824587 |

The control is selected only from validation AUC and cannot change after test
evaluation. Full and control use identical test rows and train-only history.

## Reporting rule

- Report AUC, ACC, RMSE, Brier, and ECE for every Full and control checkpoint.
- A test external win requires Full AUC to exceed the frozen external test AUC.
- Report the Full-minus-control test delta without imposing a new selection
  gate; validation supplied the module qualification decision.
- Count ASSIST09, NIPS34, and Junyi as confirmation of the module story only if
  Full both beats the external line and remains above its frozen control.
- Report XES3G5M regardless of direction and do not relabel a base win as an
  RCPK effect.
- Save row-level predictions and hashes. No new tuning follows from these test
  outcomes.

