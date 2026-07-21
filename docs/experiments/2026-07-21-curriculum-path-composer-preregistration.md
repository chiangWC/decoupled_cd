# Curriculum Path Composer: model-level preregistration

Date: 2026-07-21

## Why this candidate is opened

The frozen static-representation screen passed on the declared overall metric:
real relation semantics beat Q-only and exact degree/type-preserving rewires by
0.018228 AUC on ASSIST09 and 0.007656 on NIPS34, with both
student-clustered 95% intervals above zero. XES3G5M and Junyi were positive but
below the effect threshold.

This evidence opens a model-level candidate. It does not qualify a module and
does not require static metadata to exist in every dataset. The final pool and
story remain results-driven.

## Literature mechanism and CD mapping

The component borrows one mechanism from Graph Transformer Networks/FastGTN:
learned soft composition of typed one-hop relations into useful multi-hop
paths. GTN learns task-relevant meta-path graphs rather than requiring manually
specified meta paths:

- paper: https://arxiv.org/abs/2106.06218
- authors' code: https://github.com/seongjunyun/Graph_Transformer_Networks

No author code is copied. The implementation is an independent, compact PyTorch
translation for the CD graph schema.

The heterogeneous graph contains:

- exercise nodes;
- assessed concept nodes;
- auxiliary curriculum/group nodes when present;
- Q incidence in both directions;
- verified hierarchy/prerequisite edges in both directions;
- verified similarity edges;
- verified item-group membership in both directions.

Train-only item attempt, correctness and confidence statistics initialize the
exercise-side signal. No validation/test response enters the graph.

## Complete component boundary

The candidate is called **Curriculum Path Composer (CPC)**.

Inputs:

- learned exercise and concept identities;
- train-only exercise statistics;
- Q incidence;
- verified typed static relations and auxiliary nodes.

Outputs:

- `exercise_nodes [num_exercises, dim]`;
- `concept_nodes [num_concepts, dim]`;
- learned per-hop/per-channel relation weights and path diagnostics.

The output replaces the existing Q semantic-alignment output. Evidence
construction, concept prior, state construction, target requirement and
diagnosis may consume only CPC's output; they have no raw-embedding bypass.
CPC is one framework box. Its relation selection, message propagation and
channel aggregation are internal mechanisms, not separately claimed modules.

The first implementation uses four implicit path-composition steps and four
channels. Relation slots and topology are identical across datasets; missing
relation types are represented by empty edge sets, never dataset-ID routing.

## Frozen variants

All variants instantiate the identical CPC parameters in the same order and
share the same model initialization, optimizer, data order and training recipe.

| Variant | Graph supplied to CPC |
|---|---|
| Full | Q plus verified real static relations |
| Direct | Q only; static relation slots are empty |
| Rewire 0/1/2 | Q plus deterministic relation-type and exact-degree-preserving rewires |

The capacity control is the strongest validation result among the three
rewires. The ablation control for a dataset is the stronger of Direct and that
capacity control. Controls are selected by the predeclared primary overall
criterion, not separately for S, H and T.

The same static graph variant is used by standard and holdout. CPC has no
component-specific auxiliary objective in this round.

## Primary responsibility and gate

CPC's declared responsibility is overall response discrimination. For each
dataset define its ablation effect as:

`delta_overall = min(delta_standard_overall_auc, delta_holdout_overall_auc)`.

Target AUC is a mandatory safety and external-win axis, not CPC's primary
metric.

The first screen uses ASSIST09 and NIPS34 because both passed the frozen signal
gate and both are current external-loss opportunities with different relation
schemas (item groups versus concept hierarchy).

The candidate advances beyond the first screen only if:

- Full forms a new S/H/T validation Pareto point;
- Full rescues at least one of ASSIST09 or NIPS34 to an ordinary external win,
  or improves that dataset's worst external margin by at least 0.005;
- at least one dataset has `delta_overall >= 0.005`;
- the other dataset has no S/H/T regression beyond 0.001.

If it advances, Full is run on the existing three winning datasets and the
other admitted metadata datasets. Datasets without verified static relations
receive Q-only edges through the same CPC topology.

Final CPC qualification requires:

- the Full architecture wins at least three datasets;
- on at least two Full-winning datasets, `delta_overall >= 0.005`;
- at least one Full-winning dataset has `delta_overall >= 0.010`;
- at least one student-clustered paired-bootstrap 95% interval has lower bound
  above zero;
- no Full-winning dataset has a material S/H/T regression;
- Full preserves target-slice external leadership wherever it claims a win.

The bootstrap is paired by student and is not a model seed.

## Framework and narrative decision

A passing CPC would become one claimed component inside a multi-box CD
framework; standard evidence/state/diagnosis boxes remain visible even if they
are not claimed contributions. The number of ultimately claimed components is
not fixed. Existing or later components are claimed only when their own clean
ablations are material.

If CPC fails, its result is recorded without renaming or residual/gate repair.
The next candidate is selected from the observed failure responsibility. The
route is not rejected merely because some current datasets lack metadata, but
neither is a metadata-rich pool declared successful before external wins and
clean model ablations exist.

## Engineering checks

- Canonical relation files are generated only from the hash-verified admission
  sources and are external generated assets.
- Standard/holdout use the same graph file and training recipe per dataset.
- Q/ID mapping, auxiliary-node mapping and edge direction are audited.
- Target interactions never enter history-derived item statistics.
- Downstream predictions change when CPC output is perturbed or disconnected.
- Full/controls have identical parameter counts and initialization hashes.
- Compileall, unit tests, one-epoch smoke, gradient checks and
  `git diff --check` precede formal runs.
