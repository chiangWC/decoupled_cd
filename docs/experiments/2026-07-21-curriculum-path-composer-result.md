# Curriculum Path Composer: first-screen result

Date: 2026-07-21

## Decision

Curriculum Path Composer (CPC) v1 is rejected. It does not pass the frozen
model-level gate, does not produce an external win, and is not a paper module.
It must not be renamed, patched with a residual/gate, or rerun as the same
mechanism.

This decision rejects the specific implementation that first compresses the
typed static graph into global exercise/concept representations. It does not
reject static metadata as a data source: the preceding student-conditioned
signal screen remains positive, and a different mechanism may use that signal
if it has a distinct input/output responsibility and a new preregistration.

## Frozen execution

- implementation commit: `4a4a8c6`
- model seed: 42
- selection stage: validation only; `test_metrics` is null in every completed
  result
- Full: Q plus verified real static relations
- Direct: Q only with identical CPC parameters and auxiliary-node capacity
- standard and holdout share the same dataset recipe and graph variant
- all Full/Direct pairs have identical initialization hashes

The first screen used ASSIST09 and NIPS34, the two datasets that passed the
preregistered static-representation signal gate. No hyperparameter was chosen
from these model-level results.

## Completed full-recipe results

| Dataset / split | Full AUC | Direct AUC | Delta AUC | Full Brier | Direct Brier | Delta Brier |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST09 standard | 0.752000 | 0.750720 | +0.001279 | 0.192570 | 0.193853 | -0.001282 |
| ASSIST09 holdout | 0.744056 | 0.738989 | +0.005067 | 0.193684 | 0.195549 | -0.001864 |
| NIPS34 standard | 0.779941 | 0.779737 | +0.000204 | 0.189265 | 0.189165 | +0.000101 |

ASSIST09's preregistered primary component effect is
`min(delta standard, delta holdout) = +0.001279`, below the required +0.005.
NIPS34 standard alone bounds its same primary effect above by +0.000204, so it
cannot pass regardless of its holdout value.

The Full architecture is also not externally competitive on the completed
axes:

| Dataset / split | Full AUC | Row-aligned external AUC | Margin |
|---|---:|---:|---:|
| ASSIST09 standard | 0.752000 | 0.776432 | -0.024432 |
| ASSIST09 holdout | 0.744056 | 0.768947 | -0.024891 |
| NIPS34 standard | 0.779941 | 0.788478 | -0.008537 |

NIPS34 holdout Full/Direct jobs were stopped once the standard result made the
frozen `min(delta S, delta H) >= 0.005` gate logically impossible. Rewired
capacity controls were not run: failure against Direct already prevents Full
from beating the stronger of Direct and the rewires by the required amount.
Target evaluation and paired bootstrap were likewise not opened because the
deterministic first-screen gate failed.

## Diagnostic runs

One-epoch smoke and five-epoch staged runs were used only to verify training
and convergence behavior. They were not formal selection results. The early
ASSIST09 advantage shrank as training proceeded, while NIPS34 stayed near zero;
the full-recipe results above therefore rule out an early-checkpoint artifact.

## Failure diagnosis and next responsibility

The successful fixed signal audit propagated each student's signed response
and item-difficulty residual from that student's observed items toward a target
item. CPC v1 instead propagated only global exercise/concept embeddings and
train-population item statistics before student history was encoded. The
model therefore did not implement the student-conditioned relational signal
that opened the route.

The next eligible mechanism should preserve individual history interactions
until they are conditioned on a target concept/item. A suitable responsibility
is student-conditioned relational retrieval: input a student's unordered
train-only response set, a target query and verified relation paths; output a
target-conditioned state consumed by diagnosis. Q-only relations,
degree/type-preserving rewires and a matched-capacity history aggregator remain
mandatory controls. It requires a separate literature mapping and
preregistration before implementation.
