# Target-Local Pairing Activation Audit: Validation Result and Rejection

## Frozen decision

Target-Local Pairing **did not pass** its preregistered activation gate. The
current history/pairing branch is closed. RealPair is an activation probe, not
a paper module, and these results do not justify building or naming a
Target-Local Pairing framework component.

This is a validation-only decision. The run used model seed 42, split seed
2024, exactly 20 epochs, and the frozen optimizer and architecture described in
the preregistration. There was no hyperparameter search, early stopping,
checkpoint selection, or multi-seed experiment. No test file, test manifest,
or test prediction was read.

The operational consequence is fixed:

- do not tune RealPair, change its capacity, add an auxiliary objective, or
  rescue it with a residual, gate, adapter, or second head;
- do not rename or rerun the same target-before-pooling mechanism;
- retain the artifacts as negative evidence and move to a mechanism with a
  different responsibility and a new preregistered control;
- do not count RealPair, Strong-OPMS, PermPair, or LateFusion as a qualified
  paper module.

## Overall validation metrics

All four variants predict exactly the same validation-query row IDs. AUC is the
primary metric and lower Brier is better. Bold marks the best value within a
dataset only; it is not a model-selection operation.

| Dataset | Variant | AUC | Brier |
|---|---|---:|---:|
| ASSIST17 | RealPair | **0.766218** | **0.197454** |
| ASSIST17 | Strong-OPMS | 0.765091 | 0.198016 |
| ASSIST17 | PermPair | 0.759584 | 0.200681 |
| ASSIST17 | LateFusion | 0.765759 | 0.197493 |
| MOOCRadar | RealPair | **0.913174** | **0.064812** |
| MOOCRadar | Strong-OPMS | 0.912140 | 0.065532 |
| MOOCRadar | PermPair | 0.911789 | 0.066320 |
| MOOCRadar | LateFusion | 0.912197 | 0.065118 |
| XES3G5M | RealPair | 0.749942 | 0.138843 |
| XES3G5M | Strong-OPMS | 0.744929 | 0.140951 |
| XES3G5M | PermPair | **0.752051** | **0.136507** |
| XES3G5M | LateFusion | 0.750151 | 0.138983 |
| Junyi | RealPair | 0.825858 | 0.157657 |
| Junyi | Strong-OPMS | **0.827624** | 0.157383 |
| Junyi | PermPair | 0.824093 | 0.158524 |
| Junyi | LateFusion | 0.827237 | **0.157108** |

## Attribution against every frozen control

For control `c`, the reported AUC delta is
`AUC(RealPair) - AUC(c)`. The dataset effect is the minimum of the three
deltas, because RealPair had to beat every frozen control. The 95% interval is
the preregistered joint-min interval from 2,000 shared student-cluster bootstrap
draws; this bootstrap is not a multi-seed experiment.

| Dataset | vs Strong-OPMS | vs PermPair | vs LateFusion | Min-control effect | Joint-min 95% CI |
|---|---:|---:|---:|---:|---:|
| ASSIST17 | +0.001127 | +0.006635 | +0.000460 | +0.000460 | [-0.003306, +0.001111] |
| MOOCRadar | +0.001033 | +0.001385 | +0.000976 | +0.000976 | [-0.001296, +0.001612] |
| XES3G5M | +0.005013 | -0.002109 | -0.000209 | -0.002109 | [-0.009094, +0.000556] |
| Junyi | -0.001766 | +0.001765 | -0.001379 | -0.001766 | [-0.005145, +0.000027] |

The controls identify why the branch is rejected:

- On ASSIST17, the true local target is much better than the independently
  trained PermPair donor target (`+0.006635`), but moving the same true target
  from before to after pooling costs only `+0.000460`. The data support target
  identity, not a substantial target-before-aggregation placement effect.
- MOOCRadar is directionally consistent against all controls, but its limiting
  gain is only `+0.000976`, below the `0.002` activation threshold, and the
  joint interval crosses zero.
- On XES3G5M, PermPair beats RealPair by `0.002109`; LateFusion also beats it
  slightly. The result is incompatible with a stable benefit from real
  target-local pairing even though RealPair beats the structurally different
  Strong-OPMS control.
- On Junyi, Strong-OPMS and LateFusion beat RealPair by `0.001766` and
  `0.001379`, respectively. Beating only PermPair does not establish the
  claimed placement mechanism.

## Preregistered five-condition gate

The aggregate decision is conjunctive. All four datasets passed the donor-map
identification precondition, but the model gate failed four of five checks.

| Gate condition | Result | Evidence |
|---|:---:|---|
| At least two identified datasets have min-control effect `>= 0.002` | **Fail** | Zero datasets qualify. |
| At least one identified dataset has min-control effect `>= 0.003` | **Fail** | The largest min-control effect is MOOCRadar `+0.000976`. |
| At least one joint student-bootstrap 95% CI has lower bound `> 0` | **Fail** | Every joint interval crosses zero. |
| Counted datasets have maximum Brier regression `<= 0.0002` | Pass | Vacuously true because no dataset qualified for the `0.002` effect set. |
| No identified dataset has min-control effect below `-0.001` | **Fail** | XES3G5M is `-0.002109`; Junyi is `-0.001766`. |

Accordingly, `activated=false` is not a marginal judgment or a post-hoc
threshold choice. The mechanism misses both the magnitude requirement and the
cross-dataset non-regression requirement.

## Fixed-model donor sensitivity is diagnostic only

The additional 200 donor maps were applied to one fixed trained RealPair model
without refitting. They are not an exchangeable randomization null, are not a
permutation test, produce no p-value, and are non-gating. Their
`donor_minus_real` AUC summaries are retained only to describe prediction
sensitivity:

| Dataset | Mean | 2.5% quantile | Median | 97.5% quantile |
|---|---:|---:|---:|---:|
| ASSIST17 | -0.003794 | -0.005351 | -0.003844 | -0.002217 |
| MOOCRadar | -0.000264 | -0.001015 | -0.000284 | +0.000694 |
| XES3G5M | +0.000900 | +0.000147 | +0.000923 | +0.001600 |
| Junyi | -0.001669 | -0.002540 | -0.001696 | -0.000631 |

These values cannot override the trained-control comparisons. In particular,
the XES3G5M donor maps tend to improve the fixed model, which reinforces rather
than repairs the failure of the real-target placement hypothesis.

## Provenance and immutable artifacts

The unversioned result root is:

```text
results/goal_two_module/target_local_pairing_activation_6989290/
```

Training and evaluation ran from commit
`6989290c8d57b2643204e16a8d6aa31f2dd8528c`; that commit was present on
`origin/codex/student-local-inductive`, and the recorded worktree was clean.
The architecture-family SHA-256 is
`9bea88325fa11dc393fc14de57797932f23be795c3bb6e25fde34f99ca4f5a3f`.

| Artifact | SHA-256 |
|---|---|
| `aggregate/activation_decision.json` | `eb87a986bd4e2574eeb2ff8c92dfbd8d445d3375b7bae6d91428da01f21c8c18` |
| `evaluate/ASSIST17/evaluation.json` | `f6cdacfba0fdc18720574a021a739e710856c80cf655f11118d7de00b4e84dd0` |
| `evaluate/MOOCRadar/evaluation.json` | `0edaddee10ea8da1f989260629065404b5ba7c92019068937cc63076f877e5d7` |
| `evaluate/XES3G5M/evaluation.json` | `39a06ffc162ca8244230c9e245361b1160a4006cab162a62b034cf0395c1bd9b` |
| `evaluate/Junyi/evaluation.json` | `548bfc4289acbe811ed2f1ca5ceaa7447502694d9da18568acf2bfd4077aedcf` |

The per-dataset evaluation records also preserve row-order, prediction
manifest, aligned-prediction, protocol, source-data, feature, initialization,
batch-plan, checkpoint, and donor-map hashes. Generated checkpoints,
predictions, aligned label tables, and result JSON files remain under
`results/` and are not committed.

The source of truth for the decision is the aggregate JSON above. This document
is its human-readable rejection record. It must not be cited as a test result,
an external-model comparison, or evidence that a paper module was obtained.
