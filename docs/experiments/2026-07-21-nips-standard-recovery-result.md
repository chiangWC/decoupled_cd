# NIPS34 standard recipe recovery result

## Decision

The bounded recipe recovery passes. With the frozen RCPK mechanism, increasing
the shared concept dimension from 32 to 64, using learning rate `2e-3`, and
allowing early-stop patience 5 raises standard-validation AUC above the
row-aligned ORCDF line while preserving a large real-relation effect.

NIPS34 is therefore the fourth RCPK standard-validation external win. This is a
dataset recipe change, not a new paper component. The architecture fingerprint
is `f0151739cd1cbfaa`, shared with the existing dimension-64 ASSIST09,
XES3G5M, and Junyi candidates.

## Bounded recipe screen

All candidates use seed 42, the real NIPS34 relation graph, four RCPK steps,
20% context-target masking, factorized item/Q requirement, response BCE, and
validation-only evaluation.

| Candidate | Validation AUC | Brier | Best epoch | Safety gate |
|---|---:|---:|---:|:---:|
| dim64, lr `5e-4` | 0.787511 | 0.186299 | 29 | fail |
| dim64, lr `1e-3` | 0.788470 | 0.185870 | 29 | fail |
| dim64, lr `2e-3` | **0.789270** | **0.184915** | 56 | pass |

The preregistered AUC safety line was `0.788778`; the selected candidate is
`0.000492` above it. Test metrics are null in all three summaries.

## Attribution confirmation

The selected recipe was rerun with Q-only and three exact degree/type-preserving
relation rewires. All five variants have initialization hash
`2db9b5f859aa4e8cc6f5901729899cbdc04a36fc95ae4fffe92331bf5f11661c`.

| Variant | Validation AUC | Brier | Full - variant AUC |
|---|---:|---:|---:|
| Full real relations | **0.789270** | **0.184915** | -- |
| Q-only | 0.779979 | 0.189107 | +0.009290 |
| Rewire 0 | 0.779082 | 0.189498 | +0.010188 |
| Rewire 1 | 0.779858 | 0.189211 | +0.009411 |
| Rewire 2 | 0.779028 | 0.189491 | +0.010242 |

Full exceeds every complete control by more than the preregistered `0.002`
minimum. Its external margin over ORCDF `0.788477589` is `+0.000792112`.

## Integrity and next use

- No test prediction or test metric was opened during recipe selection.
- The data, real/rewired graphs, seed, optimizer family, objective, masking, and
  downstream model were unchanged.
- The search stops at the three preregistered learning rates; no further NIPS
  tuning is permitted.
- The selected Full and validation-selected strongest control (Q-only) enter
  the frozen standard-test confirmation together.

