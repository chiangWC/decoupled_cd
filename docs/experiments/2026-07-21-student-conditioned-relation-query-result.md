# Student-Conditioned Relation Query: first-screen result

Date: 2026-07-21

## Decision

Student-Conditioned Relation Query (SCRQ) v1 is rejected. It fails the frozen
first-screen effect and external-margin gates, so it is not a paper module and
does not advance to rewired controls, target evaluation, bootstrap or test.

This rejects the specific student-conditioned four-hop static-relation query.
It does not imply that all auxiliary metadata is unusable, nor that a route
must be abandoned merely because metadata is available on only some datasets.
Here the reason is empirical: the model-level gain is too small and does not
transfer from standard to concept holdout.

## Frozen execution

- preregistration commit: 253f3b3
- implementation commit: b6fc8ea
- model seed: 42
- selection stage: validation only; every completed result has null
  test_metrics
- Full and Direct use the same trainable topology, parameter count, optimizer,
  data order and initialization hash
- Full uses admitted real relations; Direct leaves metadata slots empty and
  retains Q
- response BCE is the only objective

The original two-stage fingerprint remains 099906acdba8c3b4 when SCRQ is
disabled. Full and Direct share SCRQ architecture fingerprint
3c70059df0f854ae.

## Diagnostic convergence checks

| Dataset / run | Full AUC | Direct AUC | Delta AUC | Full Brier | Direct Brier |
|---|---:|---:|---:|---:|---:|
| ASSIST09 one epoch | 0.733401 | 0.723252 | +0.010150 | 0.191673 | 0.195808 |
| ASSIST09 five epochs | 0.769175 | 0.764431 | +0.004744 | 0.180351 | 0.181844 |
| NIPS34 five epochs | 0.778493 | 0.778409 | +0.000085 | 0.190392 | 0.190192 |

The one-epoch ASSIST09 difference justified continuing convergence checks, but
it was not treated as a formal effect. NIPS34 stayed effectively tied and its
Brier direction was slightly worse.

## Formal ASSIST09 validation screen

| Split | Full AUC | Direct AUC | Delta AUC | Full Brier | Direct Brier | Delta Brier |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0.769044 | 0.764549 | +0.004495 | 0.180288 | 0.181821 | -0.001533 |
| concept holdout | 0.757064 | 0.756166 | +0.000898 | 0.185628 | 0.184702 | +0.000925 |

The preregistered component effect is:

delta_overall = min(delta standard, delta holdout) = +0.000898.

This is below the required +0.005. The holdout Brier regression also exceeds
the +0.0002 safety tolerance.

Against the current row-aligned external lines, Full remains noncompetitive:

| Split | Full AUC | External AUC | Margin |
|---|---:|---:|---:|
| standard | 0.769044 | 0.776432 | -0.007388 |
| concept holdout | 0.757064 | 0.768947 | -0.011883 |

The worst external margin improves over Direct by only +0.000898, below the
required +0.005, and neither split reaches the ordinary-win tolerance.

## Gate consequence

ASSIST09 already makes the conjunctive first-screen gate impossible. NIPS34's
five-epoch standard contrast is only +0.000085 and provides no contrary
evidence. Therefore:

- no NIPS34 full-recipe holdout jobs are run;
- no degree/type-preserving rewires are run, because a stronger control cannot
  repair failure against Direct;
- no target slice or paired bootstrap is opened;
- no test confirmation is opened;
- SCRQ is not tuned, renamed, gated, or attached as a residual prediction head.

## Failure interpretation

The earlier fixed signal screen used student-disjoint pseudo targets and a
logistic estimator with hand-normalized graph statistics plus common summary
features. That screen established predictive information in static relation
semantics, not that a neural relation-query component would generalize under
the actual standard and concept-holdout protocols.

SCRQ transferred some of that signal to ASSIST09 standard but almost none to
holdout. On NIPS34, the small concept-hierarchy relation set produced no
material model contrast. The failure is thus a protocol-transfer and
cross-dataset stability failure, not lack of optimization or lack of a clean
control.

The next module search should not continue static-relation transport by
default. It should return to the four-win 099906acdba8c3b4 performance path,
identify a train-only signal that is material on at least two already winning
datasets or on two newly qualified datasets, and only then map that signal to a
new complete component.
