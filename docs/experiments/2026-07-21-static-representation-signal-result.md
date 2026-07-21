# Static-relation representation signal: formal result

Date: 2026-07-21

## Decision

The curriculum/static-relation representation route is **activated for model-level
experimentation**. This is a mechanism-screening result, not yet evidence that a
neural component qualifies as a paper contribution.

The preregistered primary responsibility was overall prediction rather than
exact-zero completion. Full relation semantics were compared with Q-only and
three exact degree/type-preserving relation rewires under the same fixed
four-hop feature block, scaler, logistic regression and student-disjoint OOF
protocol.

## Reproducibility

- implementation commit: `0f24a74`
- result SHA-256:
  `2d6ca244860ade96dc5e07c14e6309b79bd7a193ec44cc0c9b06502083cb09b8`
- NIPS34 prediction SHA-256:
  `fc6751063f57f3e1da3df9eb8f7bd9fdf5f609c5d5583ceb56e4991e10453782`
- XES3G5M prediction SHA-256:
  `efd540da5c6e60d0a46c90ad65d48e4f7404d671844daf92e48732f7cf69ddbb`
- formal result:
  `results/static_representation_signal_v1/result.json` (generated asset)

ASSIST09 and Junyi reuse the frozen, hash-checked predictions from the preceding
static-metadata completion screen. NIPS34 and XES3G5M were fit only after this
screen was preregistered.

## Primary results

| Dataset | Full overall AUC | Strongest control | Delta | Brier delta | Student-clustered 95% CI | Dataset pass |
|---|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | 0.766843 | 0.748616 | +0.018228 | -0.005991 | [0.015676, 0.020730] | yes |
| NIPS34 | 0.782736 | 0.775080 | +0.007656 | -0.003084 | [0.006833, 0.008432] | yes |
| XES3G5M | 0.782741 | 0.780613 | +0.002129 | -0.000408 | not required | no |
| Junyi | 0.821572 | 0.819620 | +0.001952 | -0.000911 | not required | no |

The aggregate gate passed: two datasets exceeded the predeclared 0.005 effect
threshold, one exceeded 0.01, a newly admitted opportunity (NIPS34) passed, and
both passing datasets had bootstrap lower bounds above zero.

## Target-slice safety

Target-slice results are diagnostic in this screen because the declared
responsibility is overall representation. Against the control selected by the
primary overall criterion, ASSIST09 improved exact-zero T by 0.009212 and
XES3G5M improved exact-zero T by 0.003007. NIPS34 had only 34 eligible
low-coverage pseudo-target rows and therefore produced no target decision.
Junyi's one-to-one item/concept construction makes its target identical to the
screened overall set.

These numbers must not be conflated with the earlier completion screen, which
selected the strongest control by target AUC and found only +0.003610 on
ASSIST09.

## Interpretation and next use

The result supports a specific claim: on ASSIST09 and NIPS34, the semantics of
verified static relations contain predictive information beyond Q incidence,
node type, degree and connectivity alone. Positive but sub-threshold XES3G5M and
Junyi results do not support the same effect-size claim.

It does not establish an external win and it does not qualify a paper module.
The next candidate must be a complete Curriculum-Relation Representation
component with an independent input/output boundary. It will replace, rather
than bypass, the corresponding item/concept representation path and will be
tested against both a Q-only direct baseline and a parameter-matched relation
control. Its declared primary responsibility remains overall prediction, while
S/H/T safety and the final external-win criteria remain mandatory.

ASSIST09 and NIPS34 are now evidence-backed rescue opportunities because the
screened signal is comparable to or larger than their historical overall
deficits. That is only a prioritization fact; no rescue or publication claim is
made until the mechanism is trained inside the actual CD architecture.
