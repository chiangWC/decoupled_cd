# Target-Local Pairing Activation Audit: Preregistration

## Decision question

This is a validation-only activation audit, not a framework module or paper
contribution. It asks:

> After retaining a strong outcome-partitioned history summary and the true
> target in every path, does placing the target inside each historical
> item-response encoding, before set aggregation, add reproducible predictive
> signal beyond a late-fusion encoder with the same raw inputs and exactly
> matched active capacity?

The audit starts from negative evidence. Outcome-Partitioned Multi-Set improved
its control by only `+0.000391` on ASSIST17 and `+0.000741` on MOOCRadar.
For binary responses,

```text
sum_i x_i tensor [r_i, 1-r_i]
  = [sum_{r_i=1} x_i, sum_{r_i=0} x_i].
```

A response-item tensor that only performs this partition is therefore a
reparameterization of the rejected correct/incorrect pools. It must not be
implemented, renamed, or rerun. This audit instead tests whether the target must
enter a nonlinear interaction encoder before history compression.

Passing activates a new module-design search; it does not qualify the audit
probe as a paper module. Failure closes the current history/pairing branch.

## Literature source and novelty boundary

The primary implementation analogy is the deterministic target-to-context path
in [Attentive Neural Processes](https://arxiv.org/abs/1901.05761): each target
input attends to observed context input-output pairs before its prediction.
The associated
[Google DeepMind Neural Processes repository](https://github.com/google-deepmind/neural-processes)
contains ANP notebooks under Apache-2.0. We use the paper-level data-flow
analogy and an independent PyTorch implementation; no notebook code is copied.

[DIN](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction)
conditions attention over a user's behavior on the candidate item, while
[TAGNN](https://arxiv.org/abs/2005.02844) makes its session representation vary
with the candidate target. They are adjacent recommendation precedents for a
target-dependent history summary, not cognitive-diagnosis methods or
implementation sources.

The novelty boundary is deliberately narrow.
[SAKT](https://arxiv.org/abs/1907.06837) already queries past
exercise-response interactions with the current exercise, and
[AKT](https://arxiv.org/abs/2007.12324) uses the current question in its
context-aware, monotonic attention over past interactions. Both are direct
knowledge-tracing precedents for the target-conditioned-history motif.
Consequently, this project must not claim the first target-aware history
aggregation, the first question-conditioned student representation, or a new
attention primitive. The present probe uses an independent target/item MLP and
masked mean; it is not a renamed implementation of any of these attention
architectures.

The audit tests only whether, under a **student-local inductive cognitive
diagnosis protocol** with no free student-ID state, response-calibrated support
interactions should generate a target-specific diagnostic state. Any eventual
claim would have to come from a CD-specific state interface, clean controls,
external wins, and a qualified ablation. The audit alone establishes none of
those claims.

## Completely test-independent protocol

The four datasets are ASSIST17, MOOCRadar, XES3G5M, and Junyi. Model seed is 42
and split seed is 2024. The loader and CLI may name and open exactly:

```text
train.csv
valid.csv
Q_matrix.csv
```

There is no test-path argument. The audit must not parse an existing
student-disjoint manifest: those manifests were generated using source-test
membership and contain test metadata. It must not call `train.py`,
`resolve_verified_protocol_manifest`, or any loader requiring test support or
query. Tests must fail if it attempts to open `test.csv`,
`test_support.csv`, `test_query.csv`, or a legacy manifest.

Q is the union of all concept rows for an exercise in `Q_matrix.csv`.
Interaction-level concept strings never override this union. All
`(student,item)` groups are atomic.

### Student roles and rows

Only students present in both source train and source valid are eligible. The
order-invariant assignment

```text
stable_fraction(2024, "target-local-validation-role", student_id)
```

places values below `0.8` in the optimizer role and the remainder in the
validation role.

For optimizer students, source train is divided by complete
`(student,item)` groups into 70% support and 30% query using namespace
`"target-local-optimizer-inner-query"`. A student must retain at least 10
support groups and 3 query groups. Five leave-student-out folds use namespace
`"target-local-optimizer-fold"`.

The provisional item union from all optimizer-role students is not the final
known-item set. After the inner 10/3 retention rule, the final known set is
exactly the support-plus-query item union of retained optimizer profiles.
Validation support and query are filtered against that final union, then the
validation 10/3 student rule, row counts, student hashes, and row hashes are
recomputed. An item appearing only for an optimizer student rejected by the
inner rule is therefore unknown and cannot retain a validation target.

For validation students:

- support consists only of their source-train rows;
- query consists only of their source-valid rows;
- rows containing items absent from retained optimizer support-plus-query are
  removed;
- a `(student,item)` group occurring in support is removed from query;
- at least 10 support groups and 3 query groups must remain.

Student retention, donor feasibility, scalers, item statistics, and features
are label-blind with respect to validation query. The complete validation file
receives a provenance byte hash, but that hash never enters splitting, mapping,
features, or model input. The feature projection has its own label-free hash.
Validation-query labels are loaded separately only after probabilities have
been produced. The outcome-free row ID hashes canonical `stu_id`, canonical
`exer_id`, and `split_row_index` when present, otherwise immutable source-file
row position. The label loader re-reads `stu_id`, `exer_id`, optional
`split_row_index`, and `label`, reconstructs the same IDs, and checks the full
`valid.csv` SHA-256 both before and after the read against the frozen source
audit. Duplicate IDs and missing selected IDs are fatal. Predictions and
selected labels must have exactly equal ID sets and join one-to-one; the
prediction-order SHA-256 must be identical before and after the join. Equal row
counts alone never establish alignment.

### Foldwise train-only statistics

For optimizer fold `f`, item ease, attempt count, conditional-response
signature, ease-tercile boundaries, and their learned preprocessing use only
optimizer students outside `f`. Validation item quantities are rebuilt from
all retained optimizer students. No held student's response contributes to its
own item statistics. A zero-count item uses one shared unknown-item
representation, never an untrained item-specific embedding.

Numeric scalers are fit once from the assembled optimizer OOF feature set:
held-fold support-row descriptors plus their true query-target-row
descriptors. Donor rows never enter as extra scaler-fit samples and therefore
cannot reweight an item by duplication. After fitting, a donor operation only
replaces target features that have been transformed with that same frozen item
numeric scaler; validation rows never enter a scaler fit either.

## Frozen donor-target maps

Each map changes only the target supplied to the local pre-pool branch. It does
not shuffle support responses.

For each student, unique query target-item groups are partitioned by

```text
(target Q-cardinality bucket, train-only target-item ease tercile)
```

where cardinality buckets are `1`, `2`, and `3+`. A cell with at least two
distinct target items is movable. Items are ordered using namespace
`"target-local-donor-order"`; a non-zero cyclic shift selected with namespace
`"target-local-donor-shift"` creates a derangement. Hash inputs include split,
fold where applicable, replicate, student, stratum, and target item.

Thus:

- no eligible target maps to itself;
- repeated rows of one `(student,target item)` group share one donor;
- donor and target belong to the same student and frozen stratum;
- there is no cross-stratum fallback, including for singleton cells;
- support item, Q, ease, response, and order remain unchanged;
- the common true-target path always retains the real target.

Replicate 0 is frozen for the independently trained PermPair control on both
optimizer and validation records; it is not resampled per epoch. Replicates
0--199 are frozen donor-sensitivity maps. After model fitting is complete, the
maps are applied to one fixed trained RealPair model without refitting,
checkpoint choice, threshold choice, or model selection. They report how its
predictions and AUC respond to admissible donor targets; they are not a
permutation null distribution and produce no p-value. Every run records each
cell's distinct-item count, eligible and changed row/student counts,
fixed-point count, row-level true-target-to-donor mapping, and mapping SHA-256.
A dataset is `not_identified` if fewer than 20% of validation query rows change
or fewer than 100 validation students have a movable cell.

A conditional randomization p-value would require exchangeability of the
observed true local target and donor assignments under a specified null. That
condition is unavailable here: the true target is structurally distinguished,
donors are deterministic functions of student, item, train-only ease stratum,
and replicate, and the fixed model was fit on the true-target RealPair path.
The 200 constrained maps therefore support a fixed-model sensitivity
diagnostic only.

## Label-blind feasibility precheck

Before preregistration, the independent train+valid-only split and
cardinality-by-ease donor rule were checked using only student/item IDs, Q, and
train-derived item ease. No validation outcome or model prediction was read.

| Dataset | Validation students | Query rows | Movable-row fraction | Movable students |
|---|---:|---:|---:|---:|
| ASSIST17 | 348 | 6,491 | 0.936 | 340 |
| MOOCRadar | 419 | 8,133 | 0.924 | 419 |
| XES3G5M | 419 | 4,308 | 0.957 | 419 |
| Junyi | 670 | 3,078 | 0.839 | 645 |

These values establish feasibility only. They are not effect estimates and do
not satisfy any activation condition. Earlier exact-Q-by-ease and
response-mobility calculations concern different permutations and must not be
substituted for this donor-target audit.

### Completed final implementation audit

The retained-optimizer-finalized, four-dataset label-blind protocol audit at
commit `e207b00` completed without test access. Its artifacts are under
`results/goal_two_module/target_local_pairing_v15_protocol_e207b00/`.
Concrete replicate-0 values read from the final audit JSON files are:

| Dataset | Retained validation students | Query rows | Changed rows | Changed-row fraction | Changed students | Fixed points | Protocol status | Mapping SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ASSIST17 | 348 | 6,491 | 6,080 | 0.9366815591 | 340 | 0 | `identified` | `515a4e1d7b0bb10fe9e695bca8d8ed7ad1c834f9b3e66d826321626f056e9efb` |
| MOOCRadar | 419 | 8,133 | 7,492 | 0.9211852945 | 419 | 0 | `identified` | `ad0c5c5ea5561e01efa6dbb0c37f47b78b9f1003d0615e7e64726f452c39a85d` |
| XES3G5M | 419 | 4,308 | 4,168 | 0.9675023213 | 419 | 0 | `identified` | `c96f649299fa4f02af3fabb55f59deb2cb5b6bde406cc6104494f1a8403e026c` |
| Junyi | 670 | 3,078 | 2,528 | 0.8213125406 | 639 | 0 | `identified` | `4ace89db56d204337f2b4cd992538f232dada877ebc119383db36ebc1020ff5b` |

The retained-profile item-union correction removed zero additional validation
query rows and zero additional validation students on all four datasets; this
equality was checked rather than assumed. The individual final `summary.json`
SHA-256 values are:

- ASSIST17: `3e884141c00d70e23df2f18da67193968774727d0182e6819dcb5a78e5d17621`;
- MOOCRadar: `77658461daf6a2771cb150620b7fad59aa89f03c807b456529fe9b246c340f4b`;
- XES3G5M: `959cc4e4e15874cf952a2724d05011fe220c0f4ae9d19a06db05b92c430af7c8`;
- Junyi: `b394005fab87df8d19ad8592c72a9dc8e540ca774c94b122d3c11e906509b4e5`.

The superseded `9d95259` aggregate remains provenance only and must not be
used to lock formal training inputs.

## Shared inputs

Every probe receives, for every individual support row, its item identity,
union-Q concepts, and foldwise-train-only ease/count/signature. The three
row-level response columns are frozen as:

- `response`: that row's binary outcome;
- `response_minus_foldwise_item_ease`: that row's response minus its foldwise
  train-only item ease;
- `group_attempts_over_group_attempts_plus_one`: if the support contains
  `n_ui` rows for that student's item, `n_ui / (n_ui + 1)`, repeated on each of
  those rows as group-attempt confidence.

The separate six-column support summary, in exact tensor order, is
`theta_logit`, `raw_accuracy`, `log1p_support_rows`,
`log1p_unique_support_items`, `log1p_correct_rows`, and
`log1p_incorrect_rows`. Every probe also receives:

- the true target identity, union-Q concepts, and train-only statistics;
- true correct/incorrect item-state means, contrast, counts, and confidences.

Repeated support attempts remain individual encoder records although their
`(student,item)` group is split atomically. Pooling is a masked mean, never a
sum, so history length cannot scale the target injection. The common
diagnosis/head explicitly consumes the true target state in every variant.
PermPair therefore corrupts only the local interaction, not the predicted item.

## Frozen probes

Let `h_ui` encode a complete support interaction, `t_j` be the true target,
and `B(u,j)` contain the true Outcome-Partitioned Multi-Set summary plus
common student and true-target features.

### Strong-OPMS control

```text
logit = H_opms(B(u,j)).
```

This is the direct strong-history control corresponding to the prior negative
evidence. It is not called same-raw-input or capacity matched because its
irreversible outcome pooling precedes the placement probe.

### RealPair

```text
z_real = masked_mean_i F(h_ui, t_j)
logit  = H(B(u,j), t_j, z_real).
```

### PermPair control

```text
z_perm = masked_mean_i F(h_ui, t_donor(u,j,replicate=0))
logit  = H(B(u,j), t_j, z_perm).
```

PermPair is independently trained with the same frozen donor rule used at
validation. Only its local target changes.

### LateFusion same-raw-input, exact-capacity control

```text
s_late = masked_mean_i G(h_ui)
z_late = F_late(s_late, t_j)
logit  = H(B(u,j), t_j, z_late).
```

One shared `TargetPlacementProbe` class must implement the three paired paths.
RealPair and LateFusion use the same item, Q, response, target, placement, and
head modules; only target placement relative to masked-mean pooling changes.
PermPair uses the same pre-pool path with its donor. All three must have
identical parameter names, shapes, active-parameter counts, initialization
hash, minibatches, and optimizer settings: parameter difference is exactly 0%.
Every module must receive a finite non-zero gradient in a synthetic check.

A target-intervention test holds support and common true target fixed while
changing only the local donor. It must change the Real/Perm pre-pool
representation, leave Strong-OPMS/common inputs unchanged, and leave
LateFusion's pre-pool support representation unchanged. Permuting support order
must leave every prediction invariant.

## Fixed optimization

- frozen item-numeric descriptor width: 16; item-ID and concept embedding
  dimensions: 16 each;
- fused item/state dimension: 32; every probe MLP hidden dimension: 64;
- no dropout, auxiliary loss, scheduler, dataset override, or special loss;
- binary cross-entropy with logits;
- AdamW, learning rate `0.001`, weight decay `0.0001`;
- batch size 128, exactly 20 epochs;
- seed 42 and identical deterministic epoch batches;
- no validation early stopping, checkpoint selection, hyperparameter search,
  or post-hoc control selection.

All variants start from one serialized initialization. PermPair is a separate
training job, not an inference-only corruption. The 200 further donor maps are
fixed-model sensitivity diagnostics only and never trigger refitting or model
choice.

## Metrics and activation gate

All variants predict the same validation query row IDs in the same order. The
primary metric is overall AUC; Brier is the calibration guard. ACC and RMSE are
descriptive. Inference uses 2,000 paired student-cluster bootstrap replicates
with seed 2024. Each replicate draws one validation-student multiset and applies
that exact draw jointly to RealPair and all three controls. It reports the
three pairwise AUC-difference intervals and, within the same replicate,

```text
Delta_AUC_joint_min = min_c (AUC(RealPair) - AUC(c)).
```

The joint-min 95% interval is computed from the 2,000 replicate-wise minima,
not by taking the minimum of three separately sampled intervals.

No control is selected after observing validation. For every control
`c` in `{Strong-OPMS, PermPair, LateFusion}`, report:

```text
Delta_AUC_c   = AUC(RealPair) - AUC(c)
Delta_Brier_c = Brier(RealPair) - Brier(c).
```

The dataset effect is `min_c Delta_AUC_c`: RealPair must beat every control.
The 200 fixed-model donor maps report sensitivity curves and quantiles with the
common true-target path unchanged, but no inferential p-value.

Further module work activates only if all conditions hold:

1. at least two identified datasets have dataset effect `>=0.002`;
2. at least one identified dataset has dataset effect `>=0.003`;
3. at least one identified dataset has a joint-min paired-bootstrap 95% CI
   lower bound above zero;
4. RealPair Brier regression is at most `0.0002` against every control on
   every dataset counted in condition 1;
5. on no identified dataset is RealPair AUC more than `0.001` below any
   control.

The gate is conjunctive. No threshold, control, metric, or dataset may be
changed after validation. A `not_identified` dataset is reported but supplies
neither support nor a regression failure.

Three target-item robustness summaries are mandatory but non-gating. First,
define the top 1% most frequent target items using optimizer-train-only item
frequency before reading validation labels or predictions, remove their
validation rows, and recompute all effects. Second, run a target-item-cluster
bootstrap using one shared item draw for all four variants. Third, report
leave-one-validation-target-item-out influence on the three pairwise effects
and their joint minimum, retaining only removals whose remaining rows contain
both outcome classes. The primary student-cluster bootstrap already addresses
student dependence. These item diagnostics may qualify interpretation but
cannot activate or veto module work.

## Required outputs

The implementation records:

- hashes/counts for source train, source valid, and Q only;
- role, inner support/query, fold, and row-alignment hashes;
- student and `(student,item)` overlap counts;
- Q-union conflicts and unknown-item filtering;
- foldwise reference-student/row hashes and ease-tercile edges;
- donor cell sizes, mapping hashes, changed fractions/students, and fixed
  points for replicate 0 and all 200 validation donor-sensitivity maps;
- architecture, input, initialization, minibatch, active-parameter,
  prediction, and gradient-check audits;
- AUC, Brier, ACC, RMSE, all pairwise deltas, three pairwise bootstrap reports,
  and the joint-min bootstrap report;
- for every fixed-model donor map, mapping/input/prediction/order SHA-256 plus
  the AUC-sensitivity distribution and its frozen quantiles;
- non-gating optimizer-frequency top-1%-target-item exclusion,
  target-item-cluster bootstrap, and validation-target-item-LOO sensitivity
  summaries.

Generated mappings and predictions remain unversioned under `results/`.
