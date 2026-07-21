# NIPS34 standard recipe recovery preregistration

## Motivation

The frozen RCPK architecture has three external standard-validation wins, but
NIPS34 remains `0.003649` below the row-aligned ORCDF line despite a clean
`0.007096` RCPK module effect. The second History Set component audit failed,
so this round does not add another structure. It tests whether the NIPS deficit
is a bounded recipe/capacity issue.

## Frozen architecture and candidates

RCPK, real relations, four path steps, 20% context-target masking, factorized
item/Q requirement, target-conditioned Diagnosis and seed 42 remain unchanged.
The current NIPS anchor uses concept dimension 32, learning rate `1e-3` and
early-stop patience 1.

Run exactly three Full candidates with concept dimension 64 and early-stop
patience 5:

| Candidate | Learning rate |
|---|---:|
| `dim64_lr5e4` | `5e-4` |
| `dim64_lr1e3` | `1e-3` |
| `dim64_lr2e3` | `2e-3` |

Dimension 64 matches the other three external-winning datasets and therefore
reduces rather than increases cross-dataset architectural variation. All other
NIPS settings remain fixed: at most 80 epochs, student batch 64, zero weight
decay, response BCE and validation-only evaluation.

## Decision rule

The current external validation line is `0.788477589`. A candidate enters
control confirmation only if:

- validation AUC is at least `0.788777589`, including a `0.0003` safety margin;
- Brier is no worse than the current Full value `0.186955243` by more than
  `0.0002`;
- test metrics remain null.

Rank passing candidates by AUC, then Brier, then smaller learning-rate change.
For the selected candidate, rerun Q-only and all three relation rewires under
the identical recipe. NIPS becomes a fourth RCPK external win only if Full
still exceeds every control by at least `0.002` and exceeds the external line.

If no candidate crosses the safety line, stop NIPS recipe tuning. Do not add a
module, enlarge the search, change the seed or inspect test results.
