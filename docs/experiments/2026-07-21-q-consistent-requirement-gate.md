# Q-Consistent Requirement Gate

## Why the gate was rerun

The original ASSIST17 target comparison mixed three different definitions of
student concept coverage.  All counts below refer to holdout validation rows:

| Artifact | History concept source | Target concept source | exact-zero rows |
|---|---|---|---:|
| frozen Full slice | interaction-row `cpt_seq` | interaction-row `cpt_seq` | 21,670 |
| registered external slice | interaction-row `cpt_seq` | exercise Q union | 16,638 |
| current protocol | exercise Q union | exercise Q union | 6,851 |

ASSIST17 exercises can have multiple Q-matrix concepts while an interaction
row carries only one `cpt_seq` value.  Consequently, the old Full target AUC
`0.796573` and external target AUC `0.781337` were not measured on the same
rows.  Their target margin and the old Requirement ablation cannot be treated
as aligned evidence.

The repaired protocol treats the Q-matrix as authoritative on both sides:
every history exercise contributes the union of all its Q concepts, and every
target exercise is evaluated against the same union.  The evaluator rebuilds
this mask directly from `train.csv`, `valid.csv`, and `Q_matrix.csv`; it never
trusts a coverage column stored in an old prediction file.

## Paired rerun

The Full model was rerun from the current implementation with the same
dataset recipe, model seed 42, data order, initialization contract and
validation checkpoint rule as `factorized_item_control`.  Both paths use the
same item identity, item representation, Q view, item difficulty and
downstream diagnosis.  The control removes only joint hidden-unit nonlinear
Q-item composition and replaces it with an exactly parameter-matched additive
factorization.

All Full, control and external predictions were aligned to the original
validation rows using `stu_id`, `exer_id`, and `label`; Junyi also used
`source_row_id` and `split_row_index`.  No test file or test prediction was
opened.  Student-clustered paired bootstrap uses 2,000 replicates and seed
2024.

## Q-consistent validation result

| Dataset | Full S/H/T | Control S/H/T | External S/H/T | Full external result |
|---|---|---|---|:---:|
| ASSIST17 | .801303 / .800336 / .780853 | .801657 / .799879 / .781492 | .784373 / .783852 / .772290 | strict win |
| MOOCRadar | .930873 / .926996 / .936415 | .930343 / .926716 / .935352 | .929553 / .924178 / .933108 | strict win |
| XES3G5M | .791930 / .786833 / .785588 | .792322 / .787696 / .785818 | .786396 / .781620 / .778528 | strict win |
| Junyi | .825245 / .825002 / .825002 | .825856 / .824025 / .824025 | .820434 / .819989 / .819989 | strict win |

Thus the current Full topology remains a strong performance path: it has four
ordinary and four strict validation wins under the corrected protocol.  This
does not establish that Requirement Query is responsible for those wins.

| Dataset | Delta S | Delta H | Delta T | student-clustered Delta T 95% CI |
|---|---:|---:|---:|---:|
| ASSIST17 | -0.000354 | +0.000457 | -0.000639 | [-0.003034, +0.001680] |
| MOOCRadar | +0.000531 | +0.000280 | +0.001063 | [+0.000101, +0.002099] |
| XES3G5M | -0.000393 | -0.000863 | -0.000230 | [-0.001974, +0.001441] |
| Junyi | -0.000611 | +0.000978 | +0.000978 | [+0.000051, +0.001952] |

The preregistered Requirement gate required at least two Full-winning
datasets with `Delta T >= 0.002`, including one with `Delta T >= 0.003`, at
least one CI lower bound above zero, and no winning S/H/T regression beyond
`0.001`.  The CI and regression checks pass, but zero datasets reach either
target-gain threshold.  The conjunctive gate therefore fails.

## Decision

Joint nonlinear Q-item composition is rejected as a paper module.  The 14
remaining History-by-Requirement factorial jobs will not run.  The four-win
Full model may remain a performance reference, but its old Q-only/prototype
Requirement ablations and the mismatched ASSIST17 target margin must not be
used as module attribution evidence.

ASSIST17 external validation T is corrected to `0.7722901914958902`.  Its
stored test T was produced with the legacy mixed mask and cannot be
Q-consistently reconstructed from the currently retained row-level artifact.
The number is preserved for provenance but is pending Q-consistent
confirmation and is not a formal test opponent for a new model.

## Artifacts

- evaluator commit: `908b55a8e34f631bae960a7fe7fddc7fd12e6e65`
- result: `results/goal_two_module/factorized_requirement_factorial_v10/q_consistent_requirement_gate.json`
- result SHA-256: `b9f28914f448d8dbf42255f70f787003e6a8ffc84fef604ce05dbebce06daf03`
