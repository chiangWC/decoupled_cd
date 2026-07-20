# Cross-fitted Conditional Response Signature Audit

## Purpose

This is a CPU opportunity audit, not a model contribution. It asks whether an
item's response curve conditional on a student's support-derived ability adds
predictive information beyond a strong item-ID 2PL-like control. A failed gate
rejects this mechanism before any neural module is implemented.

## Frozen protocol

The audit uses only the ASSIST17 and XES3G5M student-disjoint seed-2024
manifests. Optimizer students are assigned to five deterministic folds. Within
each optimizer student, complete `(student, item)` groups are assigned 70% to
ability support and 30% to audit query; students with fewer than 10 support or
3 query groups are excluded. For an OOF example in fold `f`, theta-bin edges,
item statistics, and Q prototypes use only students outside `f`.

Validation theta is constructed only from `valid_support`; `valid_query` labels
are used only after prediction. The script verifies manifest file hashes,
student separation, group separation, OOF student separation, and row-ID
separation and records the corresponding hashes.

Fixed choices are:

- model seed 42 and split seed 2024;
- `theta = clip(logit((correct+1)/(attempts+2)), -4, 4)`;
- five stable-rank student-theta quintiles, including deterministic tie handling;
- beta strength 20;
- L2 logistic regression with `C=1`;
- one train-OOF `StandardScaler` shared by all common dense features and one
  label-free scaler fit symmetrically to pooled Full/Capacity signatures;
- raw 0/1 Q/item indicators and sparse `theta × item-ID` interactions are not scaled;
- 2,000 student-clustered paired-bootstrap replicates.

No bin count, smoothing strength, classifier regularization, or slice is tuned.

## Conditional signature and controls

For item `e`, bin `b`, and its exact-Q group `g`, the global bin rate is
`p_b=(C_b+1)/(N_b+2)`. The leave-item-out prototype and shrunken item curve are

```text
g_eb = (C_gb-C_eb + 20 p_b) / (N_gb-N_eb + 20)
p_eb = (C_eb + 20 g_eb) / (N_eb + 20)
w_eb = N_eb / (N_eb + 20)
r_eb = w_eb [logit(p_eb) - logit(g_eb)].
```

The signature contains five residuals, five confidences, the residual and
confidence at the target student's discrete theta bin, and the local residual
slope: exactly 13 dimensions. Exact-Q singleton items receive 13 zeros in both
Full and Capacity.

The four fixed predictors are:

- Direct: theta, raw support accuracy, log support size, target-Q multi-hot,
  Q-cardinality, train-only item ease, and log item count.
- Strong2PL-like: Direct plus item-ID intercepts and unconstrained
  `theta × item-ID` slopes.
- Full: Strong2PL-like plus the target item's 13-dimensional signature.
- Capacity: Strong2PL-like plus 13 dimensions from another item selected by a
  deterministic no-fixed-point rotation within the same exact-Q group.

Full and Capacity therefore have identical width. All matrices remain sparse.
Each dataset selects one stronger control by validation overall AUC from
Strong2PL-like and Capacity; the selection is not repeated per metric or slice.

## Fixed rejection gate

Full must satisfy every condition against that dataset's single stronger
control:

1. overall AUC gain at least 0.003 on both datasets;
2. at least one overall AUC gain at least 0.005;
3. overall Brier regression at most 0.0001 on both datasets;
4. at least one overall student-cluster bootstrap 95% CI lower bound above 0;
5. XES exact-zero AUC gain at least 0.005 and Brier regression at most 0.0002.

ASSIST17 exact-zero has only 81 validation rows and is diagnostic only. Failure
of any condition rejects the mechanism without tuning or substituting MOO.

## Outputs

Each dataset writes aligned row-level predictions with coverage and Q-cardinality
membership, an audit JSON, and all overall/coverage/Q-cardinality metrics. The
aggregate output contains the conjunctive gate, prediction hashes, row hashes,
fold hashes, data hashes, leakage audit, and paired bootstrap results.

## Result and decision

The fixed run is stored under
`results/student_local_inductive/conditional_response_signature/` (generated,
not versioned). Full failed the gate:

| Dataset | Direct | Strong2PL | Capacity | Full | Full-control AUC | Full-control Brier | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST17 | 0.754247 | 0.756600 | **0.756901** | 0.756407 | -0.000494 | +0.000107 | [-0.002014, 0.000972] |
| XES3G5M | 0.770431 | 0.776116 | **0.776577** | 0.775289 | -0.001288 | +0.000165 | [-0.003652, 0.001062] |

Capacity was the single selected control on both datasets. On XES exact-zero,
Full-control AUC was -0.001294 and Brier was +0.000133. Only the XES
exact-zero Brier condition passed; both overall AUC conditions, both CI
conditions, the joint overall Brier condition, and the XES exact-zero AUC
condition failed. Conditional Response Signature is therefore rejected and
must not be implemented as a framework module.

An earlier diagnostic mistakenly scaled the complete sparse matrix, including
one-hot columns. It is archived at
`results/student_local_inductive/conditional_response_signature_diagnostic_all_sparse_scaled/`
and is excluded from the decision. The table above is the one complete rerun
using the intended frozen preprocessing; no C, bins, smoothing, gate, or control
selection changed after looking at the diagnostic run.
