# Student-Conditioned Relation Query: preregistration

Date: 2026-07-21

## Decision question

The static-representation signal screen showed that verified curriculum
relations are predictive on ASSIST09 and NIPS34 only when each student's
train-only response pattern is transported toward the target. Curriculum Path
Composer v1 then failed because it compressed the graph into global
exercise/concept representations before conditioning on a student.

This candidate asks a narrower question:

> Does preserving response-level relational context until target query time
> improve cognitive diagnosis beyond the same history/state model with Q-only
> or structurally rewired relations?

The mechanism is called **Student-Conditioned Relation Query (SCRQ)**. It is a
candidate component, not yet a paper module. Static metadata need not exist in
every dataset; datasets without admitted relations execute the same topology
with empty metadata slots.

## Literature mechanism and CD mapping

SCRQ independently combines mechanisms from three primary sources:

- Neighborhood Interaction/KNI identifies an early-summarization failure in
  graph recommendation and retains distinct user-side/item-side neighborhood
  interactions: https://arxiv.org/abs/1908.04032
- KGNN-LS makes relation importance user-specific before neighborhood
  aggregation: https://arxiv.org/abs/1905.04413
- Set Transformer establishes attention-based permutation-invariant processing
  for unordered sets: https://proceedings.mlr.press/v97/lee19d.html

No author model or code is copied. In the CD mapping, a recommender user is a
student, consumed items are train-only attempted exercises, feedback is the
binary response plus its residual from train-only item ease, and the query item
is the target exercise/concept requirement. Unlike the recommendation papers,
SCRQ has no free student-ID embedding: all student conditioning is reconstructed
from the current train-only response set.

The work remains cognitive diagnosis rather than knowledge tracing. History is
treated as an unordered set, timestamps are not inputs, and the objective is
response diagnosis under partial concept coverage rather than next-step
temporal prediction.

## Framework boundary

The framework contains the following visible boxes:

1. Q semantic alignment;
2. unordered history representation and concept-state construction;
3. SCRQ, the only new candidate in this round;
4. fixed target-conditioned response diagnosis.

Only SCRQ is currently under contribution testing. Other boxes may later
become claimed components only if their own clean ablations pass; the final
number of claimed modules is not fixed.

SCRQ inputs are:

- the upstream completed student concept state;
- the student's unordered train-only exercise/response set;
- train-only item attempt/ease statistics;
- target exercise and its Q requirement;
- admitted typed curriculum relations.

SCRQ outputs are:

- target_student_state [num_targets, dim];
- relation_reliability [num_targets];
- relation/hop diagnostics.

Diagnosis receives student-specific information only through
target_student_state; it cannot read the pre-query student state through a
parallel prediction head.

## Frozen mechanism

For each minibatch, observed exercise nodes receive four response-memory
channels: attempted mass, correct mass, response-minus-item-ease residual, and
absolute residual. Item ease uses the frozen empirical-Bayes shrinkage strength
20 and only training interactions.

Four propagation steps retain relation type and hop identity. Q-only paths and
paths that use at least one admitted metadata edge remain distinguishable.
For each target exercise, SCRQ reads the transported channels at the target and
normalizes signed response statistics by transported mass. A history-derived
student summary scores relation/hop channels; a fixed-width query composer maps
the upstream Q-pooled state, target requirement and retrieved response context
to the unique target_student_state.

The implementation uses concept_dim=64, four hops, four response channels and
response BCE only. There is no module-specific auxiliary loss in the first
screen. The same per-dataset recipe is used for standard and holdout. No
post-result hop, channel, shrinkage or loss tuning is allowed before the first
decision.

## Clean controls

Every variant instantiates identical trainable parameters in the same order and
shares data order, mask, optimizer, recipe, initialization hash and target
history.

| Variant | Relational input |
|---|---|
| Full | Q plus verified real static relations |
| Direct | Q only; metadata relation slots are empty |
| Rewire 0/1/2 | Q plus deterministic exact degree/type-preserving rewires |

The capacity control is the strongest validation result among the three
rewires. A dataset's ablation baseline is the stronger of Direct and that
capacity control, selected by the primary overall metric and then reused for
S/H/T. Graph tensors are non-parameter buffers, so Full and controls have
identical parameter count.

A diagnostic history_global control may use the same response-memory width
but makes every history item equally reachable from the query. It may strengthen
the control but can never weaken the predeclared Direct/rewire comparison.

## First screen and gate

The first screen is ASSIST09 and NIPS34 because both passed the frozen
student-conditioned static-representation signal screen, both currently lose
external overall AUC, and their admitted relation schemas differ.

For each dataset define the component effect as:

delta_overall = min(delta standard overall AUC, delta holdout overall AUC).

Full advances only if all conditions hold:

- Full forms a new validation S/H/T Pareto point;
- Full rescues at least one screen dataset to an ordinary external win, or
  improves its worst external margin by at least 0.005;
- at least one screen dataset has delta_overall >= 0.005;
- the other screen dataset has no S/H/T regression beyond 0.001;
- target AUC does not regress beyond 0.001.

The first run compares Full and Direct. Rewires are run only if Full clears the
deterministic Full-versus-Direct bound. If Full already fails against Direct,
a stronger control cannot reverse that failure.

## Final qualification

A passing first-screen candidate is expanded to the existing winning datasets
and every admitted pool dataset. Architecture topology is shared; only relation
edge content differs.

SCRQ becomes a paper module only if:

- the Full architecture obtains at least three ordinary external wins and at
  least two strict wins;
- on at least two Full-winning datasets, delta_overall >= 0.005;
- at least one Full-winning dataset has delta_overall >= 0.010;
- at least one student-clustered paired-bootstrap 95% interval has lower bound
  above zero;
- no Full-winning dataset regresses more than 0.001 on S/H/T;
- the module adds an external win or improves the worst target margin by at
  least 0.001.

Bootstrap is paired by student and is not a model seed. Model seed remains 42;
no multi-seed run is allowed. Test confirmation remains closed until
architecture, recipes and controls are frozen.

If SCRQ fails, it is recorded and retired without adding a gate, residual
prediction head, alternate target scope or renamed rerun. The next candidate
must follow the newly observed failure responsibility rather than preserve this
mechanism by default.

## Engineering checks before formal execution

- target interactions never enter response memory or item statistics;
- permutation of a student's history leaves outputs unchanged;
- changing real relations while holding Q fixed changes Full outputs;
- disconnecting target_student_state changes predictions and there is no
  student-specific diagnosis bypass;
- Full/control parameters and initialization hashes are identical;
- graph variants preserve ID mapping and exact degree/type signatures;
- standard/holdout use one recipe per dataset;
- finite-gradient check, one-epoch smoke, compileall, manual unit tests and
  git diff --check pass on the remote xph workspace.
