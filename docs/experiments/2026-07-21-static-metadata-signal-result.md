# Static metadata predictive-signal result

## Frozen execution

- Formal code commit:
  `27d72ca50ac9d5f98ef6680247e024a7d1607135`.
- Result payload SHA-256:
  `7829e3fdf31597ebcf469f0c0714fe6933b8158982a015c956a2f93305527406`.
- ASSIST09 prediction SHA-256:
  `035b43808c9b3987bc896e3ad2007fad9e04dc02170cd59668b13307d8f4f0a8`.
- Junyi prediction SHA-256:
  `b3bc9a9add8fc7303a6f60b9d6f24b1842d814ca783e73a72824adbc112db5b0`.
- Model seed 42, split seed 2024, five student-disjoint OOF folds.
- Only holdout-training labels were used. Protocol validation and test files
  were not opened.
- Full, Q-only, and three degree/type-matched rewires used the same 14 feature
  columns, shared scaling rule, and fixed logistic-regression estimator.

## Result

The preregistered Typed Curriculum State Completion signal gate did not
activate. No dataset met all deterministic conditions, so the aggregate gate
stopped before bootstrap.

| Dataset | pseudo-T support | Stronger control | Control T AUC | Full T AUC | Delta T AUC | Delta overall AUC | Delta T Brier | Pass |
|---|---:|---|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | 1,160 rows / 622 students | Q-only | 0.759695 | 0.763305 | +0.003610 | +0.018898 | +0.000489 | no |
| NIPS34 | 34 rows / 30 students | ineligible | — | — | — | — | — | no decision |
| Junyi | 42,100 rows / 7,812 students | shuffle 1 | 0.819620 | 0.821572 | +0.001952 | +0.001952 | −0.000911 | no |

NIPS34 had only 16 negative and 18 positive low-coverage rows, below the
predeclared 500 rows, 100 students, and 100 per-label minima. Its prediction
models were therefore not fit.

The complete eligible-dataset comparison is:

| Dataset | Variant | Overall AUC | T AUC | T Brier |
|---|---|---:|---:|---:|
| ASSIST09 | Q-only | 0.747946 | 0.759695 | 0.134058 |
| ASSIST09 | Full real relations | 0.766843 | 0.763305 | 0.134548 |
| ASSIST09 | shuffle 0 | 0.748616 | 0.754093 | 0.134564 |
| ASSIST09 | shuffle 1 | 0.748173 | 0.757343 | 0.134584 |
| ASSIST09 | shuffle 2 | 0.748480 | 0.751867 | 0.135442 |
| Junyi | Q-only | 0.818428 | 0.818428 | 0.160855 |
| Junyi | Full real relations | 0.821572 | 0.821572 | 0.159369 |
| Junyi | shuffle 0 | 0.819251 | 0.819251 | 0.160435 |
| Junyi | shuffle 1 | 0.819620 | 0.819620 | 0.160281 |
| Junyi | shuffle 2 | 0.819475 | 0.819475 | 0.160337 |

Junyi has item-concept identity Q, so every pseudo-target row is exact-zero
and its overall and T scopes coincide.

## Interpretation

The result rejects only the fixed claim that four-hop real-relation transport
is already a strong missing-state completion mechanism:

- ASSIST09 T gain was positive but below +0.005 and worsened T Brier beyond
  the frozen tolerance.
- Junyi beat every control but the gain over the strongest degree-matched
  rewire was too small.
- NIPS34 could not test the registered target slice.

It does not support the earlier broad inference that static metadata is
useless. ASSIST09 real relations improved pseudo-overall AUC by +0.018898 over
Q-only and by more than +0.018 over every degree-matched rewire. This is a
large relation-semantic signal, but it is concentrated outside exact-zero.

Accordingly:

1. Do not implement or rename this fixed transport mechanism as a
   TKC-to-UKC completion contribution.
2. Preserve ASSIST09's overall result as evidence for a distinct,
   predeclared item/curriculum representation responsibility.
3. Before implementing that component, audit whether XES3G5M, MOOCRadar,
   EdNet, or another qualified public dataset exposes identity-verifiable
   static hierarchy/relationship metadata.
4. A representation candidate must obtain material overall gains on at least
   two Full-winning datasets, one at least +0.010, without materially
   regressing S/H/T; the final architecture must still achieve at least three
   external wins.
5. If no coherent metadata-rich pool exists, retain this as a mechanism
   observation rather than forcing a dataset-specific branch.

No hop count, relation weight, target scope, rewire, classifier, or threshold
was tuned after reading the formal output.
