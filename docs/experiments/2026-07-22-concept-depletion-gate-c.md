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
3. **KaNCD**, the repository's non-graph in-harness implementation. It uses
   64-dimensional student/item/concept factors and a monotonic NCD interaction
   network. Training is fixed to minibatch 4096, learning rate 0.001,
   50-epoch ceiling and patience 10.

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

## Integrity and verification

- all generated-arm hashes must match Gate B manifests;
- prediction rows must align exactly on `audit_row_id/stu_id/exer_id/label`;
- model seed is 42; bootstrap seed is 2024; no multi-seed training;
- each model uses one recipe across both arms and all datasets;
- no OOM fallback may change batch size, epochs or architecture;
- run unit tests, `compileall` and `git diff --check` before formal jobs.
