# Gate D: graph-family concept-depletion verification

Date: 2026-07-22
Branch: `codex/concept-depletion-gate-d-graph-family`
Parent: `f4a4e90`

## Question

Gate C rejected target-concept incompleteness as a model-general CD problem,
but SVGCD supported four of five datasets. Gate D tests the narrower, frozen
hypothesis that graph CD models are systematically more sensitive to removing
student-local target-concept responses than a non-graph CD model.

This is a problem-verification experiment, not a model-selection or SOTA run.
No result from Gate D changes the external-win registry.

## Frozen intervention and data boundary

Reuse the exact Gate B/Gate C protocols without regeneration:

- ASSIST09: 1,650 selected students;
- ASSIST17: 1,609;
- MOOCRadar: 1,905;
- XES3G5M: 1,898;
- EdNet: 1,769.

For each student, the concept-depleted arm deletes all training responses
touching the target row's Q concepts. The random-depleted arm deletes the same
number of that student's target-disjoint responses. Validation tuning rows,
audit rows, Q matrices, removal counts and generated-file hashes stay fixed.
The source standard test split is never opened.

## Added graph families

Gate D adds two independently implemented PyEdmine models:

1. **RCD**: directed/undirected concept relations plus exercise-concept and
   student-exercise relation graphs. Its concept dependency graph and all
   bipartite edges are built from the current arm's training rows only.
2. **HyperCD**: user, exercise and concept hypergraph convolution. Its user
   hyperedges are clustered from the current arm's training response matrix;
   exercise/concept hypergraphs use the shared Q matrix.

These are intentionally mechanism-distinct from existing ORCDF-NCD and SVGCD.
The tracked PyEdmine source is imported read-only. Each job uses an isolated
generated runtime root, so no files are written into the external repository.
Relevant source commit and file hashes are recorded.

Fixed recipes use the authors' example defaults with only seed and epoch
ceiling made explicit:

- common: model seed 42, 50-epoch ceiling, early stopping patience 5, best
  validation AUC, Adam, no scheduler;
- RCD: batch 1,024, learning rate 1e-4, weight decay 0;
- HyperCD: batch 256, learning rate 1e-4, weight decay 5e-4, 3 layers,
  feature dimension 512, embedding dimension 16, leaky coefficient 0.8.

There is no per-dataset tuning and no multi-seed training. An OOM is a recorded
failure; batch size or model configuration must not be changed silently.

## Estimands

For each model and audit row, the within-model primary damage is

```text
D_model = log_loss(concept-depleted) - log_loss(random-depleted).
```

The direct graph-vs-non-graph interaction is

```text
I_graph = D_graph - D_official_KaNCD.
```

Both use 2,000 student-clustered bootstrap replicates with seed 2024. A model
supports a dataset only when the point estimate is positive and the primary
95% CI lower bound is strictly above zero. A graph model exceeds KaNCD only
when the interaction point estimate and its 95% CI lower bound are positive.
AUC damage and AUC interaction are secondary and cannot replace failed
log-loss evidence.

## Frozen decision rule

The four graph models are ORCDF-NCD, SVGCD, RCD and HyperCD. Official EduCDM
KaNCD-GMF is the non-graph reference.

A dataset supports the graph-family hypothesis only if:

- at least three of the four graph models support it; and
- at least two graph models have a significantly positive interaction against
  official KaNCD-GMF.

The graph-family problem is admitted only if the same rule holds on at least
three datasets. Otherwise it remains a dataset/model-specific phenomenon and
cannot be used as the paper's broad research-problem claim. No dataset may be
removed and no threshold, metric or intervention may be revised after seeing
the results.

## Verification

- assert every arm hash matches its Gate B manifest;
- assert prediction order and labels match the frozen audit CSV exactly;
- assert graph construction uses only the arm's training rows;
- record source commit, dirty-path audit and relevant source hashes;
- run an ASSIST09 smoke pair before the formal queue;
- run unit tests, `compileall` and `git diff --check` before formal jobs;
- schedule against real-time GPU memory/utilization, excluding insufficient
  devices but allowing safe sharing; never alter a recipe after OOM.
