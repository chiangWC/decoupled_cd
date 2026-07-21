# Static metadata predictive-signal gate

## Correction to the earlier admission decision

The earlier outcome-blind audit answered a narrow topology question: whether
static metadata made target concepts reachable that were not already reachable
through Q within four hops. Its 1/4 result rejects incremental reachability as
the admission criterion for a common module. It does **not** establish that
real curriculum relations lack predictive value when they overlap Q paths.

This second audit therefore tests relation quality directly. In particular,
NIPS34 is informative even though its incremental reachability was zero:
real hierarchy paths may transport student history more usefully than Q-only
paths or fake hierarchy paths while reaching the same nodes.

The active research goal is also corrected: the final framework may contain
any coherent number of functional components. Only components with material,
fair ablations will be claimed as contributions. A useful static-information
route is not rejected merely because the original pool does not expose the
same field everywhere; the dynamic pool may be expanded if a coherent
three-dataset result is obtained.

## Frozen question and data

Using only holdout-training interactions, Q, and previously identity-proven
public metadata, does the real static relation structure improve pseudo-target
prediction over both:

1. Q-only, with the metadata feature block fixed to zero; and
2. three relation-type and node-degree preserving metadata rewires?

The first screen comprises ASSIST09, NIPS34, and Junyi. ASSIST17 is excluded
from this screen because the earlier identity audit found invariant
problem-type relations incident to only 102/3,162 items. This is a mechanism
opportunity decision, not a negative performance result.

Metadata is exactly the previously proven source:

- ASSIST09 invariant template and assistment memberships, subject to the same
  2-item minimum and 10%-of-items maximum group size;
- NIPS34 subject parent hierarchy;
- Junyi official prerequisite and similarity graphs frozen to RCD commit
  `178e6a9d1485265cbc5f2f5dc93ecbe5f3c28ef0`.

The executable verifies the previous audit payload, source, protocol
`data/train/Q` hashes, and Junyi Git blobs. It may read `data.csv` ID/Q
columns for identity reconstruction but never opens protocol validation or
test files. Pseudo-target labels come only from `train.csv`.

## Pseudo protocol and fixed estimator

For every student with sufficient history, 20% of complete student-item groups
are hidden by stable hash using split seed 2024, leaving at least ten support
rows. Students are assigned to five disjoint OOF folds. The target scope is
exact-zero for ASSIST09 and Junyi and low-coverage for NIPS34. A dataset is
ineligible if this scope has fewer than 500 rows, 100 students, or 100 examples
of either label. No replacement scope is chosen after seeing this result.

All variants use the same eight common features:

- history length and response balance;
- observed-concept fraction;
- target Q size, seen fraction, and support attempts;
- reference-student item count and Beta-20-smoothed item ease.

The six-dimensional relation block is computed by four fixed heterogeneous
random-walk steps from the student's support items:

- path-weighted response mean;
- item-ease-calibrated residual mean;
- absolute residual mean;
- log effective path mass;
- inverse first metadata-path hop;
- bounded path reliability.

Only paths containing at least one metadata edge enter this block. Q edges are
bidirectional. Directed curriculum edges use weight 1 in their declared
direction and 0.5 for the inverse path; undirected metadata uses weight 1 in
both directions. Transition rows are normalized over the full outgoing graph.
At each step, a two-state automaton separately tracks paths that have and have
not yet used metadata. Q-only has the same six columns fixed to zero.

The three ASSIST bipartite/category, directed hierarchy/prerequisite, and
undirected similarity rewires use deterministic double-edge swaps. Each
relation type preserves its exact degree sequence, edge count, endpoint
domains, and directionality, rejects self-loops/duplicates, and completes at
least ten accepted swaps per original edge. Edge hashes and degree hashes are
recorded. The strongest of the three complete shuffled OOF predictors is the
shuffle control, selected once by target AUC.

Within each fold, a single StandardScaler is fit to the vertical concatenation
of all reference-variant matrices. Every variant then fits the same
`L2 LogisticRegression(C=1, liblinear, max_iter=1000, random_state=42)`.
The stronger control is selected once per dataset as the higher target-AUC
model among Q-only and strongest shuffle. Predictors are deliberately small:
passing authorizes a neural module experiment; it does not itself claim a
paper module.

## Frozen gate

A dataset passes deterministically only when Full minus its stronger control
satisfies:

- target AUC at least +0.005;
- pseudo-overall AUC no worse than -0.001;
- target Brier increase no greater than 0.0002;
- Full target AUC strictly exceeds both Q-only and every shuffle.

The route activates only if:

- at least two of the three datasets pass;
- at least one passing dataset is ASSIST09 or NIPS34, so Junyi alone cannot
  establish generality;
- one passing dataset has target-AUC improvement at least +0.010;
- on at least one passing dataset, a 2,000-replicate student-clustered paired
  bootstrap has a 95% interval lower bound above zero.

The bootstrap resamples students, not model seeds. Model seed remains 42 and
no multi-seed experiment is performed.

Passing authorizes implementation of a complete Typed Curriculum State
Completion component, followed by Full/Q-only/degree-matched capacity
ablations and external-win evaluation. Failing rejects this particular
relation-transport definition; it does not imply that all metadata-rich
datasets or all static-information mechanisms are useless. Hop count,
rewiring, estimator, data scope, and thresholds will not be tuned after the
formal output is read.
