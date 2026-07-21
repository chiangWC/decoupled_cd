# Response-Conditioned Path Kernel preregistration

## Failure-driven decision

The train-only static-representation screen established a large semantic-relation
signal on ASSIST09 (`+0.018228` overall AUC) and NIPS34 (`+0.007656`) against
degree/type-preserving rewires. Two neural realizations then failed for different
reasons: Curriculum Path Composer summarized the graph before seeing a student's
responses, while Student-Conditioned Relation Query learned separate relation/hop
attention and did not preserve the fixed screen's low-variance aggregate on
holdout validation.

This round does not tune, rename, or rerun either rejected model. It tests one
new failure-driven hypothesis: preserve the exact response-conditioned path
statistic that passed the signal screen and learn only the map from that statistic
to the target student state.

## Literature mechanism and CD mapping

The module independently adapts two graph-reasoning ideas:

- Neural Bellman-Ford Networks formulate a source-target representation as an
  aggregate over paths computed by generalized Bellman-Ford updates:
  https://papers.nips.cc/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html
- the finite-state automaton graph layer represents relations by paths accepted
  by an automaton:
  https://proceedings.neurips.cc/paper/2020/hash/1fdc0ee9d95c71d73df82ac8f0721459-Abstract.html

No author model or code is copied. In the CD mapping, the source boundary is a
student's unordered train-only response history, the graph is the typed
item/concept/curriculum relation graph, and the target is the exercise currently
being diagnosed. A two-state path automaton distinguishes paths using only Q
incidence from paths that have used at least one admitted metadata edge.

The component is named **Response-Conditioned Path Kernel (RCPK)**. It is an
overall representation component, not a TKC-to-UKC completion claim and not a
knowledge-tracing model: timestamps and next-step order are not used.

## Framework boundary

RCPK inputs are:

- an upstream Q-pooled student concept state;
- the target exercise requirement;
- the unordered train-only exercise/response mask;
- train-only item attempts and correctness used to estimate shrunk item ease;
- Q incidence and an admitted static-relation graph.

It outputs one `target_student_state [num_targets, dim]`, a path reliability,
and path diagnostics. The fixed target-conditioned Diagnosis consumes this
state as its only student-specific target representation; it cannot also read
the pre-kernel Q-pooled state through a parallel prediction head.

The complete paper framework may contain other visible components. RCPK is the
only component under contribution testing in this round, and the final number
of claimed modules is not fixed.

## Frozen path statistic

Each attempted exercise initializes four channels: attempt mass, correct mass,
response minus empirical-Bayes item ease, and absolute residual. Item ease uses
training interactions only and shrinkage strength 20. Four generalized
Bellman-Ford steps maintain two states:

```text
Q-only_next = Q-only @ Q-transition
used-relation_next = used-relation @ all-transition
                   + Q-only @ metadata-transition
```

At the target exercise, mass/correct/residual/absolute-residual values are summed
over four hops and converted to the same six fixed statistics that passed the
signal screen: response mean, residual mean, absolute residual mean, log path
mass, inverse first relation hop, and mass reliability. A fixed-width encoder and
state composer map these statistics, the upstream state and target requirement
to the unique `target_student_state`.

There is no learned relation/hop attention, relation embedding, auxiliary loss,
student-ID embedding, second prediction head, logit residual, or dataset router.
The first screen fixes four hops, shrinkage 20, response BCE, seed 42 and each
dataset's existing training recipe.

## Clean comparisons

All variants instantiate identical trainable tensors in identical order and share
the same data order, history mask, optimizer, recipe, initialization hash and
Diagnosis:

| Variant | Graph used by the path automaton |
|---|---|
| Full | Q plus verified real static relations |
| Direct | Q only; metadata transition is empty |
| Rewire 0/1/2 | Q plus exact degree/type-preserving relation rewires |

The stronger complete control is selected once per dataset by the minimum of
standard and holdout overall validation AUC and reused for all S/H/T deltas.
Graph tensors are non-persistent, non-parameter buffers, so parameter counts are
identical. Full-versus-Direct runs first; rewires run only if Full can still pass
the deterministic Direct bound.

## First-screen gate

Run ASSIST09 and NIPS34 standard/holdout validation. No test artifact is opened.
For each dataset define:

```text
delta_overall = min(Full S - control S, Full H - control H)
```

RCPK advances only if:

- Full creates a new S/H/T Pareto point;
- it rescues at least one of ASSIST09/NIPS34 to an ordinary external win or
  improves that dataset's worst external margin by at least `0.005`;
- at least one screen dataset has `delta_overall >= 0.005`;
- the other has no S/H/T regression worse than `0.001`;
- neither target AUC regresses by more than `0.001`.

## Module qualification

A passing first screen is expanded to all currently valid datasets. RCPK becomes
a paper module only if the same architecture obtains at least three ordinary
external wins and two strict wins, and relative to the stronger complete control:

- at least two Full-winning datasets have `delta_overall >= 0.003`;
- at least one has `delta_overall >= 0.005`;
- at least one student-clustered paired-bootstrap 95% CI lower bound is above
  zero;
- no Full-winning S/H/T axis regresses more than `0.001`;
- the module adds an external win or improves the worst target margin by at
  least `0.001`.

Bootstrap clusters by student and is not a model seed. Test confirmation remains
closed until architecture, recipes and comparisons are frozen. Failure is recorded
without adding learned attention, a residual head, an alternate target slice or a
renamed retry.

## Protocol amendment before a valid first screen

The first end-to-end execution exposed a training-boundary mismatch before any
candidate comparison was accepted. With legacy `context_target_frac=0`, the
training response being supervised remains in the student's response matrix. RCPK
can return to the same exercise through a Q/metadata cycle and therefore consume
that response during training, while validation correctly excludes the target.
The observed falling training loss and collapsed validation AUC are leakage-style
overfitting, not a valid module estimate.

Those runs are marked `invalid_protocol` and cannot support either acceptance or
rejection. Before rerunning, the implementation must require
`context_target_frac=0.20` for Full and every control. This value is inherited from
the already frozen train-only static-representation screen rather than selected
from model results. The existing context-target builder removes hidden exercises
from the response mask and concept evidence and supervises only those hidden rows.
All other architecture, graph, optimization, seed and decision thresholds remain
unchanged. No test result was opened.
