# Curriculum-relation representation signal gate

## Evidence-driven question

The preceding completion-oriented signal audit was negative on its exact-zero
primary metric, but revealed a large ASSIST09 pseudo-overall AUC gain
(+0.018898 over Q-only). This follow-up is explicitly evidence-driven rather
than blind. It asks a different component question:

> Do real static curriculum relations improve general item/concept
> representation, rather than specifically completing exact-zero UKC state?

The claimed responsibility is pseudo-overall response discrimination. Target
slice behavior remains a safety diagnostic. This distinction is frozen before
opening NIPS34 overall or newly admitted XES3G5M predictions.

## Data and controls

The screen uses ASSIST09, NIPS34, Junyi, and XES3G5M. XES uses its official KC
tree mapped exactly by the separately frozen admission audit. The other three
use the same identity-proven metadata as the previous signal audit.

For each dataset, holdout-training interactions are split into the same 20%
stable-hash pseudo-target and five student-disjoint OOF folds. No protocol
validation or test labels are opened. Model seed is 42 and split seed is 2024.

The fixed predictors remain:

- Full: Q plus real curriculum relations;
- Q-only: identical six-column relation block fixed to zero;
- three relation-type and exact-degree-preserving rewires;
- the same eight common features, four-step metadata-path block, shared
  StandardScaler, and fixed L2 logistic regression.

ASSIST09 and Junyi predictions are reused byte-for-byte from the preceding
frozen execution because feature generation and fitting do not depend on which
metric is later selected. Their prediction hashes are verified. NIPS34 is now
fit because its overall pseudo-target is sufficiently large even though its
low-coverage diagnostic slice was not. XES3G5M is fit for the first time.

XES parent-to-child paths are directed. A path may traverse the inverse
direction with the already frozen weight 0.5. Its 182 internal KCs remain
auxiliary graph nodes, while Q and target outputs use only the current 241
concept IDs.

## Frozen primary gate

For each dataset, the strongest shuffle is selected once by pseudo-overall
AUC. The stronger control is then the higher-overall-AUC model between Q-only
and that shuffle. A dataset passes when:

- Full minus stronger-control pseudo-overall AUC is at least +0.005;
- pseudo-overall Brier does not increase by more than 0.0002;
- Full pseudo-overall AUC strictly exceeds Q-only and all three rewires.

The route activates when:

- at least two datasets pass;
- at least one newly unopened opportunity, XES3G5M or NIPS34, passes;
- one passing dataset improves pseudo-overall AUC by at least +0.010;
- at least one passing dataset has a student-clustered paired-bootstrap
  95% interval lower bound above zero.

The bootstrap uses 2,000 student resamples and is not a model seed.

The registered target-slice safety diagnostic uses the existing exact-zero or
low-coverage scope when it contains at least 500 rows, 100 students, and both
labels. Any target AUC regression beyond 0.001 is flagged and blocks neural
module implementation even if the overall gate passes. A smaller diagnostic
scope is reported as insufficient rather than changed.

## Consequence

Passing authorizes one complete **Curriculum-Relation Representation**
component:

- input: Q-connected item/concept identities and verified static relations;
- output: the unique item/concept representation consumed by downstream state
  inference and diagnosis;
- no legacy embedding or second prediction-head bypass;
- Direct control: Q-only representation;
- Capacity control: parameter-matched degree/type-preserving relation graph.

The component must later produce material overall S/H benefits on at least two
datasets actually won by the Full model, retain the external target lead, and
survive student-clustered paired ablation. A signal pass is not itself module
qualification.

Failing this gate rejects the fixed four-step relation representation as the
next component. It does not erase the descriptive ASSIST09 finding or forbid a
future metadata-rich pool with a substantively different mechanism.
