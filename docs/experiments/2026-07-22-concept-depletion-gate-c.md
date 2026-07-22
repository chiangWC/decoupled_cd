# Gate C: cross-model concept-depletion consistency

Date: 2026-07-22
Branch: `codex/concept-depletion-gate-c`
Parent: `f2070b87`

## Motivation

Gate B was an ORCDF-NCD screen, not a model-general conclusion. It supported
ASSIST09 and XES3G5M only. Gate C is a user-requested robustness extension that
does not revise Gate B or its stopping rule. It asks whether the same frozen
intervention produces a consistent effect across graph and non-graph CD
families.

## Frozen paired protocols

The exact Gate B protocols and selected audit rows are reused without
regeneration:

- ASSIST09: 1,650 paired students;
- ASSIST17: 1,609;
- MOOCRadar: 1,905;
- XES3G5M: 1,898;
- EdNet: 1,769.

The concept-depleted and quantity-matched random-depleted training arms,
early-stopping rows, audit rows, Q matrices, removal counts and all file hashes
remain fixed. Source standard test files are never opened. NIPS34 remains
unqualified because the frozen construction found no eligible paired row.

## Model families

Three fixed model implementations are compared:

1. **ORCDF-NCD**, already completed in Gate B. It uses correct/wrong
   student-exercise graphs, exercise-concept Q edges, three graph-convolution
   layers, response-flip contrastive learning and the NCD interaction function.
2. **SVGCD**, run through the existing row-aligned external benchmark package
   with its registered seed-42 recipe: 128-dimensional embeddings, two GNN
   layers, variational/contrastive objectives, 100-epoch ceiling and patience
   10.
3. **KaNCD-style MF**, the repository's non-graph in-harness implementation.
   It uses 64-dimensional student/item/concept factors and a monotonic NCD
   interaction network. Training is fixed to minibatch 4096, learning rate
   0.001, 50-epoch ceiling and patience 10. A code audit after the first paired
   result found that this is the paper's MF-style factorization, but not the
   EduCDM package's default GMF network and nonlinearities.

The implementation discrepancy activates a stricter sensitivity analysis:
**EduCDM KaNCD-GMF** directly instantiates
`EduCDM.KaNCD.KaNCD.Net` from EduCDM 1.0.1. Its source SHA-256 is
`ac0e3ee126596a83c3cb315f418a4c078aa70db85192e1d7a834895316feca30`.
The shared harness fixes dimension 64, GMF, learning rate 0.002, batch 1024,
30-epoch ceiling and patience 10. This cross-check is deliberately stricter
than the preregistered internal screen; it cannot rescue a failed result.

No model receives concept/static metadata beyond the shared Q matrix and the
training responses in its arm. Each pair uses identical initialization seed 42,
checkpoint-selection rows and recipe.

## Estimand

For every model, dataset and selected audit row:

```text
concept-specific damage = log_loss(concept_depleted)
                        - log_loss(random_depleted)
```

The primary uncertainty estimate is the same 2,000-replicate
student-clustered bootstrap with seed 2024. Brier and AUC are secondary.
A model supports a dataset only when there are at least 500 selected students,
both labels, positive mean primary damage and a strictly positive 95% CI lower
bound.

## Final admission rule

The broad TKC/UKC problem is admitted only if **the same at least three
datasets** satisfy both:

- KaNCD supports the dataset; and
- at least one graph model, ORCDF-NCD or SVGCD, supports the dataset.

This requires agreement between a non-graph ID-factorized CD model and a graph
CD model. Agreement between the two graph models alone is insufficient.

If fewer than three common datasets pass, student-local target-concept
incompleteness is rejected as this project's main cross-dataset research
problem. Positive isolated datasets may be reported only as cases; no dataset
removal, metric substitution, threshold relaxation or third protocol revision
is allowed.

## Results

The table reports the primary mean paired log-loss damage. Bold values have a
strictly positive student-bootstrap 95% CI lower bound.

| Dataset | ORCDF-NCD | SVGCD | KaNCD-style MF | EduCDM KaNCD-GMF |
|---|---:|---:|---:|---:|
| ASSIST09 | **+0.024381** | **+0.027601** | **+0.003528** | +0.016548 |
| ASSIST17 | +0.004296 | **+0.020483** | **+0.001880** | +0.003836 |
| MOOCRadar | +0.001321 | **+0.010423** | +0.012199 | +0.006950 |
| XES3G5M | **+0.013706** | **+0.009381** | **+0.000466** | **+0.018742** |
| EdNet | +0.002607 | -0.001268 | +0.010098 | -0.000665 |

The user-requested SVGCD checks are positive on both named datasets:

- ASSIST09: AUC damage `+0.013505`, log-loss damage `+0.027601`, 95% CI
  `[+0.012392, +0.042306]`;
- MOOCRadar: AUC damage `+0.008423`, log-loss damage `+0.010423`, 95% CI
  `[+0.003498, +0.017080]`.

Under the preregistered internal KaNCD-style screen, ASSIST09, ASSIST17 and
XES3G5M form three graph/non-graph common datasets, so that mechanical gate is
admitted. This admission is not publication-safe: replacing the internal
variant by the official EduCDM network leaves only XES3G5M as a significant
graph/non-graph common dataset. The official-network log-loss intervals are:

| Dataset | Damage | 95% CI | Support |
|---|---:|---:|:---:|
| ASSIST09 | +0.016548 | [-0.002223, +0.034265] | No |
| ASSIST17 | +0.003836 | [-0.007497, +0.014897] | No |
| MOOCRadar | +0.006950 | [-0.000543, +0.014582] | No |
| XES3G5M | +0.018742 | [+0.001331, +0.036236] | Yes |
| EdNet | -0.000665 | [-0.008591, +0.007444] | No |

## Decision

The robust cross-family count is `1/5`, below the fixed requirement of three.
Student-local target-concept incompleteness is therefore rejected as this
project's main **cross-dataset** research problem. XES3G5M remains a reliable
case, and SVGCD's four positive datasets suggest a graph-family sensitivity,
but neither fact establishes a model-general CD failure. H/T slices may remain
evaluation axes; they cannot supply the paper's universal problem statement.

Point-estimate signs are positive for the official network on four datasets,
including ASSIST09/ASSIST17/MOOCRadar, but sign agreement without a positive CI
lower bound is recorded only as a trend. The bootstrap quantifies student
sampling uncertainty, not training-seed uncertainty; no multi-seed runs were
introduced.

## Integrity and verification

- all generated-arm hashes must match Gate B manifests;
- prediction rows must align exactly on `audit_row_id/stu_id/exer_id/label`;
- model seed is 42; bootstrap seed is 2024; no multi-seed training;
- each model uses one recipe across both arms and all datasets;
- no OOM fallback may change batch size, epochs or architecture;
- run unit tests, `compileall` and `git diff --check` before formal jobs.

Formal outputs:

- `results/problem_gate_b/orcdf_screen/gate_b_summary.json`;
- `results/problem_gate_c/svgcd_screen/gate_b_summary.json`;
- `results/problem_gate_c/kancd_screen/gate_b_summary.json`;
- `results/problem_gate_c/educdm_kancd_screen/gate_b_summary.json`;
- `results/problem_gate_c/final/gate_c_support_matrix.csv`;
- `results/problem_gate_c/final/gate_c_summary.json`.
