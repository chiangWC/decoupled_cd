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

The primary implementation analogy is the deterministic target-to-context read
in [Attentive Neural Processes](https://arxiv.org/abs/1901.05761): a target
location reads a set of observed context input-output pairs before prediction.
The associated
[Google DeepMind Neural Processes repository](https://github.com/google-deepmind/neural-processes)
contains ANP notebooks under Apache-2.0. We use the paper-level data-flow
analogy and an independent PyTorch implementation; no notebook code is copied.

[DIN](https://www.kdd.org/kdd2018/accepted-papers/view/deep-interest-network-for-click-through-rate-prediction)
and [TAGNN](https://arxiv.org/abs/2005.02844) are adjacent recommendation
precedents: both construct target-dependent summaries of historical behavior
instead of one fixed user/session vector. They motivate the placement question
but are not cognitive-diagnosis methods and are not implementation sources.

The novelty boundary is deliberately narrow.
[SAKT](https://arxiv.org/abs/1907.06837) and
[AKT](https://arxiv.org/abs/2007.12324) already use a current question to
select or weight past question-response interactions in knowledge tracing.
Consequently, this project must not claim the first target-aware history
aggregation, the first question-conditioned student representation, or a new
attention primitive.

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

For validation students:

- support consists only of their source-train rows;
- query consists only of their source-valid rows;
- rows containing items unknown to the optimizer role are removed;
- a `(student,item)` group occurring in support is removed from query;
- at least 10 support groups and 3 query groups must remain.

Student retention, donor feasibility, scalers, item statistics, and features
are label-blind with respect to validation query. The complete validation file
receives a provenance byte hash, but that hash never enters splitting, mapping,
features, or model input. The feature projection has its own label-free hash.
Validation-query labels are loaded separately by stable row ID only after
aligned probabilities have been produced.

### Foldwise train-only statistics

For optimizer fold `f`, item ease, attempt count, conditional-response
signature, ease-tercile boundaries, scalers, and learned preprocessing use only
optimizer students outside `f`. Validation uses quantities rebuilt from all
optimizer students. No held student's response contributes to its own item
statistics. A zero-count item uses one shared unknown-item representation,
never an untrained item-specific embedding.

## Frozen donor-target permutation

This permutation changes only the target supplied to the local pre-pool
branch. It does not shuffle support responses.

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
0--199 form the validation permutation reference distribution. Every run
records each cell's distinct-item count, eligible and changed row/student
counts, fixed-point count, row-level true-target-to-donor mapping, and mapping
SHA-256. A dataset is `not_identified` if fewer than 20% of validation query
rows change or fewer than 100 validation students have a movable cell.

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

## Shared inputs

Every probe receives:

- every support interaction's item identity, union-Q concepts, train-only
  ease/count/signature, response, response residual, and confidence;
- support-derived student accuracy, size, and confidence;
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

- item and concept embedding dimensions: 16 each;
- item/state dimension: 32; placement hidden dimension: 64;
- no dropout, auxiliary loss, scheduler, dataset override, or special loss;
- binary cross-entropy with logits;
- AdamW, learning rate `0.001`, weight decay `0.0001`;
- batch size 128, exactly 20 epochs;
- seed 42 and identical deterministic epoch batches;
- no validation early stopping, checkpoint selection, hyperparameter search,
  or post-hoc control selection.

All variants start from one serialized initialization. PermPair is a separate
training job, not an inference-only corruption. The 200 further donor maps are
evaluation-only and never trigger refitting or model choice.

## Metrics and activation gate

All variants predict the same validation query row IDs in the same order. The
primary metric is overall AUC; Brier is the calibration guard. ACC and RMSE are
descriptive. Inference uses 2,000 paired student-cluster bootstrap replicates
with seed 2024.

No control is selected after observing validation. For every control
`c` in `{Strong-OPMS, PermPair, LateFusion}`, report:

```text
Delta_AUC_c   = AUC(RealPair) - AUC(c)
Delta_Brier_c = Brier(RealPair) - Brier(c).
```

The dataset effect is `min_c Delta_AUC_c`: RealPair must beat every control.
Using one fixed trained RealPair, the conditional-permutation p-value is

```text
p = (1 + number of permuted-local-target AUC >= RealPair AUC) / 201,
```

with the common true-target path unchanged.

Further module work activates only if all conditions hold:

1. at least two identified datasets have dataset effect `>=0.002`;
2. at least one identified dataset has dataset effect `>=0.003`;
3. at least one dataset has paired-bootstrap 95% CI lower bounds above zero
   against all three controls;
4. permutation `p<=0.05` on every dataset counted in condition 1;
5. RealPair Brier regression is at most `0.0002` against every control on
   every dataset counted in condition 1;
6. on no identified dataset is RealPair AUC more than `0.001` below any
   control.

The gate is conjunctive. No threshold, control, metric, or dataset may be
changed after validation. A `not_identified` dataset is reported but supplies
neither support nor a regression failure.

## Required outputs

The implementation records:

- hashes/counts for source train, source valid, and Q only;
- role, inner support/query, fold, and row-alignment hashes;
- student and `(student,item)` overlap counts;
- Q-union conflicts and unknown-item filtering;
- foldwise reference-student/row hashes and ease-tercile edges;
- donor cell sizes, mapping hashes, changed fractions/students, and fixed
  points for replicate 0 and all 200 validation replicates;
- architecture, input, initialization, minibatch, active-parameter,
  prediction, and gradient-check audits;
- AUC, Brier, ACC, RMSE, all pairwise deltas, three bootstrap reports, and the
  permutation p-value.

Generated mappings and predictions remain unversioned under `results/`.
