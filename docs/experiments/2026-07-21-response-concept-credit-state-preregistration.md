# Response-to-Concept Credit Routing and State Completion: Preregistration

## Decision question

This is a validation-only activation audit for one complete candidate module:
**Response-to-Concept Credit Routing and State Completion**. It asks:

> When one response is attached to multiple Q-matrix concepts, does a learned,
> Q-constrained allocation of that response to concepts produce a better full
> student-concept state than copying the complete response to every concept?

The candidate is not generic Slot Attention and is not a slice classifier. Its
input is the student's train-only support interactions and their Q sets. It
first assigns each support response competitively among that interaction's Q
concepts, then completes a state for every concept from the routed evidence.
Its only predictive output path is the resulting state. A fixed
Q-conditioned diagnosis consumes that state and has no access to raw history,
student identity, routing weights, or a legacy state.

Passing this audit only activates integration and external-win evaluation. It
does not make the candidate a paper module. Failure closes this implementation
without tuning, auxiliary objectives, residual paths, or renamed reruns.

## Problem source and novelty boundary

Pelánek describes items associated with multiple knowledge components as a
credit-assignment problem in the "Knowledge Component Combinations" discussion
of [Adaptive Learning is Hard](https://link.springer.com/article/10.1007/s40593-024-00400-6).
The existing evidence constructor supplies one concrete null assumption: a
multi-concept response is copied in full to every concept in the item's Q set.
The audit tests that assumption; it does not assume in advance that it is
wrong.

The competitive aggregation mechanism is inspired by
[Slot Attention](https://proceedings.neurips.cc/paper/2020/file/8511df98c02ab60aea1b2356c013bc0f-Paper.pdf):
several concept slots compete to explain an input and receive normalized
updates. The authors' reference implementation is available in the
[Google Research repository](https://github.com/google-research/google-research/tree/master/slot_attention)
under Apache-2.0, but this experiment uses an independent PyTorch
implementation from the paper-level data-flow idea. No source code, trained
weights, or complete model is copied.

Neither ingredient is claimed as new. Multi-concept credit assignment,
conjunctive cognitive-diagnosis models, response-conditioned attention, and
competitive slot aggregation all predate this work. A possible contribution
would have the narrower boundary of a student-local, response-conditioned,
Q-constrained responsibility router coupled to complete concept-state
generation under an inductive CD protocol. That claim is eligible only if the
mechanism passes the controls below, helps external wins, and later satisfies
the repository's terminal module-qualification rule.

## Frozen framework boundary

The experiment has two framework boxes:

1. **Response-to-Concept Credit Routing and State Completion** is the only
   candidate contribution. It maps train-only support interactions to a full
   student-concept state.
2. **Q-conditioned Diagnosis** is fixed infrastructure. It maps the state and
   target item Q vector to a response probability and is not presented as a
   contribution.

For student `u`, support interaction `j` supplies one token containing:

- item/Q semantics;
- its binary response;
- response minus train-only item ease;
- attempt confidence; and
- the binary Q mask defining its eligible concepts.

The item-ease statistic and every normalization are estimated from optimizer
training rows only. There is no free student-ID embedding. For every predicted
`(student, item)` group, every row of that group is removed from its support
history before any statistic, routing, or state construction. Validation
targets never enter support history.

Within the module, each interaction communicates only with concepts in its Q
set. Learned response/item queries and concept keys produce masked logits. A
softmax **over that Q set** gives normalized responsibilities; the consumed
credit weight is `|Q| * softmax(logits)`, so its mass sums to `|Q|`. Equal
logits therefore assign weight one to every eligible concept, exactly matching
the total response mass of full Q-copy rather than weakening multi-concept
items by normalization. The same frozen number of competitive update
iterations is used for every dataset. Routed interaction values are aggregated
per student-concept pair, after which one shared completion network generates
observed and unobserved concept states. Completion is part of this single
module, not a second claimed module.

The module returns at least:

```text
framework_state [student, concept, dim]
mastery [student, concept]
routing_responsibility and routing_credit [support interaction, eligible Q concept]
routing_entropy and routed-mass diagnostics
architecture_fingerprint
```

`framework_state` is the sole student-specific input to Diagnosis. Cutting or
permuting it must alter `probs`; no response-statistic, student-ID, prior-state,
residual, auxiliary prediction head, or dataset-specific route may bypass it.

## Preregistered variants

All three variants use the same EvidenceBundle, shared state-completion
network, fixed Diagnosis, optimizer recipe, row order, masks, and target-group
exclusion. Module objects are instantiated in the same order from independently
named RNG streams. Shared tensors have identical initialization hashes.

| Variant | Frozen data flow |
|---|---|
| **Full** | Learned masked logits condition on the student's actual response token and compete over each interaction's Q set; `|Q| * softmax` credit enters the shared completion network. |
| **Direct** | The current null: credit weight one copies the complete response to every concept in Q, followed by the same evidence aggregation and shared completion network. |
| **Capacity Control** | Exactly the Full iterative skeleton, tensor dimensions, parameters, initialization, aggregation, and completion network, but its learned routing is static item-concept routing: the scorer receives item/Q/concept features while actual response and student-condition channels are replaced by their frozen neutral values. |

Capacity Control can learn that a particular item assigns different static
credit to its Q concepts, but cannot change that allocation with the student's
actual response or response-conditioned state. It therefore rules out the
explanation that item difficulty or static Q refinement alone caused a gain.
If Capacity Control matches or beats Full, the response-conditioned routing
claim fails. Direct is the practical current-data-flow control; Capacity
Control is the mechanism and capacity control. There is no module-specific
loss in this audit: all variants optimize response BCE only.

The routing-output layer in both Full and Capacity Control is zero-initialized.
Both therefore begin with credit weight one on every eligible concept and are
identical to Direct at initialization on their consumed evidence path. For an
interaction with `|Q| = 1`, all three variants remain strictly equivalent:
softmax is one, credit weight is one, and routed evidence, completed state,
logits, and probabilities must match under shared parameters. This identity is
covered by a deterministic unit test. Consequently, single-concept datasets
cannot establish a positive routing ablation.

## Completely test-independent protocol

The activation screen uses only MOOCRadar and NIPS34, chosen before model
implementation because they contain materially different multi-concept
structures: MOOCRadar mixes single- and multi-concept items, while NIPS34 is
dominated by multi-concept items. This is a mechanism screen, not a decision to
make either dataset a paper dataset.

Optimization reuses the existing student-disjoint support-query protocol. For
each source split, optimizer and validation students are disjoint. Each
optimizer student's atomic `(student,item)` groups receive one deterministic
70% support / 30% query assignment; its query responses supervise the model
while only its support rows construct state. Validation students use their
frozen train-only support and validation query. The assignment is reused for
all 20 epochs: there is no epoch-wise support resampling, fold rotation, or
target migration between support and query.

Execution is sequential. Stage 1 trains Full, Direct, and Capacity Control on
MOOCRadar and NIPS34 `*_chold_v2` only: six jobs. Only if all preregistered
holdout C, T, H-overall, uncertainty, and Brier conditions pass does Stage 2
train the same three variants on the two current KnoField standard splits:
six additional jobs. Stage 2 checks the S-overall and standard-C guards. The
maximum is therefore 12 jobs; failure in Stage 1 terminates the candidate and
forbids the standard jobs.

Model seed is 42 and split seed is 2024. Every job uses exactly 20 epochs,
batch size 128, AdamW with learning rate `1e-3` and weight decay `1e-4`, no
scheduler, no early stopping, no checkpoint-window selection, and no auxiliary
objective. Epoch 20 is the only evaluated checkpoint. Standard and holdout use
the same model and optimizer recipe for a dataset.

The implementation may name and open training, validation, Q-matrix, mapping,
and train-derived graph/statistic files only. It has no test-path argument and
must fail if it attempts to open a test interaction or test prediction.
Validation outcomes are loaded only after all Full, Direct, and Capacity
Control prediction manifests for that dataset/split have been written.
Outcome-free row IDs, source-file hashes, selected-row hashes, and prediction
order hashes must join one-to-one before a metric is computed.

Q is the union of all concept rows for an item in `Q_matrix.csv`.
Interaction-level concept strings cannot override it. Duplicate
`(student,item)` rows are atomic for support exclusion and clustered together
for row identity. All state inputs, item ease, support coverage, slice
membership, and diagnostics are built only from training support.

## Label-blind feasibility amendment after `48bdf1e`

Commit `48bdf1e` preregistered a stricter C slice. After that commit, but before
any candidate implementation, model fitting, prediction, validation-label
read, or test-file access, a target-label-blind feasibility precheck used
student/item identities, union-Q, train-only support membership and, only to
apply the already frozen strict definition, train-only support responses.
Under the frozen student-disjoint holdout protocol, the strict slice contained
only 156 rows from 35 students on MOOCRadar, versus 8,717 rows from 507 students
on NIPS34. It therefore failed the already registered minimum of 500 rows and
100 students on MOOCRadar before a model could be evaluated.

Allowing this known structural failure to stand would make the candidate fail
mechanically rather than test its routing mechanism. This amendment is thus a
pre-implementation feasibility correction, not a response to an observed
effect. No validation outcome, test row, model probability, checkpoint, or
metric was inspected to choose the replacement. All model, training, C-effect,
T-effect, overall, bootstrap, and Brier thresholds remain unchanged.

The replacement broad C definition gives these exact outcome-free counts:

| Protocol | Dataset | C rows | C students |
|---|---|---:|---:|
| holdout | MOOCRadar | 1,581 | 295 |
| holdout | NIPS34 | 28,051 | 1,013 |
| standard | MOOCRadar | 2,557 | 406 |
| standard | NIPS34 | 29,079 | 1,015 |

The superseded strict definition is retained below as `C_strict`, along with
its adverse MOOCRadar feasibility result, rather than being erased.

## Clean-ablation gate tightening after `8d4dc37`

After the feasibility amendment, and still before candidate implementation,
model fitting, prediction, validation-label read, or test-file access, a clean-
ablation review found a loophole in the control rule. Selecting one control by
C AUC and then reusing only that control for T, overall, and Brier could ignore
the other control when it was stronger on one of those axes. That would not
establish that Full survives both reasonable explanations.

This pre-implementation amendment closes that loophole. Every AUC effect and
guard below compares Full with the higher AUC of Direct and Capacity Control
for the metric being tested. Every Brier guard compares Full with the lower
Brier of the two controls. Which control has higher C AUC is still reported
descriptively, but it has no control-selection or gating role. The numerical
thresholds, datasets, stages, training recipe, C/T definitions, and final paper
qualification rule are unchanged.

## Primary credit-sensitive slice C

The gating mechanism slice `C` is frozen without reading a validation target
label or prediction. A validation query row belongs to `C` exactly when:

1. its target item has at least two concepts in union-Q;
2. after excluding the target `(student,item)` group, the student's train-only
   support contains at least three **distinct multi-concept items** whose
   union-Q intersects the target Q set; and
3. at least two distinct concepts in the target Q set are covered by those
   eligible multi-concept support items.

Support items, not repeated interaction rows, determine the count in condition
2. This broad, outcome-free slice identifies targets for which responsibility
allocation has multiple ambiguous support items and at least two plausible
target-concept destinations. It does not use support correctness to select
rows. Slice row IDs are frozen before validation labels are opened. For a
dataset to be identified, holdout-validation `C` must contain at least 500
rows, at least 100 students, and, once labels are opened after prediction,
both response classes; otherwise the candidate cannot pass this two-dataset
activation screen.

Holdout-validation `C` AUC is the primary mechanism metric. Standard-validation
`C` AUC, ACC, RMSE, Brier, and ECE are reported as robustness diagnostics and
must obey the regression guard below. No threshold in the definition of `C`
may be changed after inspecting a validation prediction.

### Descriptive strict slice C_strict

`C_strict` retains the original `48bdf1e` definition. It requires a multi-Q
target, at least three target-group-excluded train-only support rows for every
target concept, and a range of at least 0.10 among the concepts' Beta-smoothed
support response rates `(correct_c + 1) / (attempts_c + 2)`. A support row is
counted once for `c` when `c` is in its union-Q.

On student-disjoint holdout validation, its target-label-blind membership is:

| Dataset | C_strict rows | C_strict students |
|---|---:|---:|
| MOOCRadar | 156 | 35 |
| NIPS34 | 8,717 | 507 |

`C_strict` is reported only as a descriptive mechanism diagnostic. It cannot
rank controls, choose a checkpoint, activate or reject the candidate, or
replace C in any threshold or confidence interval.

## Secondary target slice T and why C is separate

`T` retains the live-registry definition: exact-zero for MOOCRadar and
low-coverage for NIPS34, built solely from the corresponding training history.
It is evaluated on holdout validation.

`C` and `T` need not overlap. Exact-zero `T` asks for a target concept with no
direct student history, whereas `C` requires multi-concept support items that
cover at least two concepts in the target Q set. `C` therefore tests the
module's declared local routing mechanism; `T` tests whether better allocation
and shared completion transfer to the missing or sparse state that matters to
the project-level external-win objective.

Improving `C` alone would support a narrow diagnostic finding but would not
justify continuing a state-completion module for this project. The secondary
`T` condition below prevents activation of a specialist that cannot improve
the intended missing-state setting.

## Frozen activation rule

For dataset `d` and AUC metric `m`, define the conservative two-control
contrast

```text
delta_auc[m, d] = AUC_m(Full)
                  - max(AUC_m(Direct), AUC_m(Capacity Control))

delta_C[d] = delta_auc[holdout C, d]
delta_T[d] = delta_auc[holdout T, d]
```

Thus Full must simultaneously survive both controls on each evaluated axis;
the maximizing control may differ by metric, but no control result is hidden.
For Brier metric `m`, the corresponding conservative regression is

```text
delta_brier[m, d] = Brier_m(Full)
                    - min(Brier_m(Direct), Brier_m(Capacity Control)).
```

Stage 1 passes only if conditions 1--3 hold and the holdout parts of conditions
4--5 hold. Stage 2 is then allowed and the candidate activates only if the
remaining standard parts also hold:

1. `delta_C >= 0.002` on both MOOCRadar and NIPS34, and at least one dataset
   has `delta_C >= 0.003`;
2. a student-clustered paired bootstrap over holdout-validation C has a 95%
   confidence interval above zero on at least one dataset for the **joint
   two-control contrast**: each replicate computes Full-minus-Direct and
   Full-minus-Capacity AUC and records their minimum;
3. `delta_T >= -0.001` on both datasets and `delta_T >= 0.001` on at least one;
4. `delta_auc[holdout overall, d] >= -0.001` on both datasets; after Stage 1
   passes, `delta_auc[standard overall, d] >= -0.001` and
   `delta_auc[standard C, d] >= -0.001` likewise hold on both datasets;
5. during Stage 1, `delta_brier[m, d] <= 0.0002` for holdout overall,
   holdout C, and holdout T wherever defined; after Stage 1 passes, the same
   bound holds for standard overall and standard C.

The paired bootstrap resamples students, not rows, for 2,000 deterministic
replicates using split seed 2024 and a dedicated namespace. In every replicate,
the joint C contrast is
`min(AUC_C(Full)-AUC_C(Direct), AUC_C(Full)-AUC_C(Capacity))`.
A replicate is valid only when all three resampled C AUC values are defined.
Fewer than 1,800 valid replicates out of 2,000 is an automatic failure. The
artifact saves every invalid replicate index, every replicate's sampled-
student multiset hash, and an aggregate sampling-manifest hash; the confidence
interval is computed from valid replicates only. This is a paired uncertainty
analysis, not a model seed. Undefined required AUC, insufficient C support,
row/hash mismatch, single-class T, non-finite output, or a failed leakage check
is an automatic failure. There is no partial pass, tuning round, or test run
after failure.

Passing this gate activates integration into the common architecture and an
external-win validation audit. It does **not** qualify the candidate as a
paper module. Final qualification remains governed by `docs/research_goal.md`:
the frozen Full architecture must obtain at least three ordinary external
wins, and this module must beat the stronger reasonable control on Full-winning
datasets by at least `0.005` T AUC on two datasets, at least `0.01` on one,
with a student-clustered 95% CI lower bound above zero on at least one and no
material S/H/T regression. Test confirmation is permitted only after the
architecture, recipe, and ablations are frozen.

## Required implementation checks and artifacts

Before formal validation runs, the implementation must pass:

- no-target-in-history and no-test-file access tests;
- Q-union, multi-row group exclusion, and train-only statistic tests;
- responsibilities and credit are zero outside Q and non-negative inside Q;
  normalized responsibilities sum to one and consumed credit sums to `|Q|`;
- zero routing-output initialization makes Full and Capacity match Direct's
  full-copy evidence before optimization;
- exact Full/Capacity/Direct equivalence for every single-concept interaction;
- common parameter/init hashes and identical shared completion/Diagnosis
  tensors across variants before optimization;
- parameter-count disclosure and Capacity Control equality with Full;
- a state-only prediction-path intervention test;
- finite-gradient checks and one-epoch CUDA smoke for all three variants on
  both datasets;
- `compileall`, targeted unit tests, and `git diff --check`.

Formal artifacts include the complete configuration, code commit, environment,
data/Q/mapping hashes, architecture fingerprint, initialization hashes,
checkpoint hash, row-aligned probabilities, both controls' per-metric results,
the descriptive C-control ranking, slice IDs and prevalence, routing
entropy/mass diagnostics, S/H/C/T metrics, all two-control deltas, clustered-
bootstrap replicates, invalid-replicate indices, sampled-student hashes,
interval, and one machine-readable activation decision. Failed runs and
negative results are retained as provenance but are not promoted to a module
contribution.
