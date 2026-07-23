# RCPK target-conditioning mechanism gate

Date: 2026-07-23

## Question and evidential boundary

RCPK already passes the frozen standard-only external and module-effect gates.
Its existing Direct/rewire controls establish that the admitted real relations
are useful, but they do not isolate the mechanism named by the model: selecting
relation paths for the current target exercise.

This validation asks one narrower question:

> Given the same student responses, the same real relation graph, the same
> trainable capacity and the same target-aware Diagnosis, does target-specific
> relation aggregation outperform a target-independent relation summary?

The candidate explanation is **target-irrelevant relational dilution**: a
single global relation summary can mix response history unrelated to the
current target. This is not assumed to be an established CD problem. It is
retained only if the natural diagnostic and the clean neural control below
support it. The rejected cross-dataset TKC/UKC and graph-family depletion claims
remain rejected. Difficulty calibration and runtime efficiency are not claims
in this gate.

Only standard validation data are used for selection. Existing test results are
not consulted to change the implementation, recipe or thresholds.

## Frozen Full and clean control

The frozen Full model remains byte-compatible with architecture fingerprint
`f0151739cd1cbfaa`. Its four-hop two-state path automaton gathers, at the current
target item, signals that have traversed at least one admitted metadata edge.

The new `student_global` control changes only the gather operation. At each hop
it sums the same transported response channels over all exercise nodes for the
student, producing one target-independent relation summary that is repeated for
that student's targets. The rest of the model still receives the ordinary
target requirement and Q-pooled state, so this is not a target-blind diagnosis
model. It isolates target conditioning inside the relation module.

Full and control therefore share:

- real graph, Q matrix, response history and item statistics;
- four propagation steps and the same response channels;
- every trainable tensor, parameter count and initialization order;
- target requirement, upstream state, Diagnosis and response BCE;
- seed 42, data order, 20% context-target masking, optimizer and dataset recipe.

No parameter, gate, auxiliary loss or dataset-specific route is added. Existing
Full checkpoints and validation predictions may be reused after a default-mode
prediction-hash compatibility check. Only the control is trained from scratch.

## Natural relevance-share diagnostic

For each standard-validation target row, use train history only. A history item
is target-reachable when the frozen directed RCPK graph contains a path of at
most four edges from that item to the target and the path uses at least one
metadata edge. Define

```text
relevance_share = number of distinct attempted history items target-reachable
                  / number of distinct attempted history items.
```

Rows with fewer than five distinct history items or zero reachable items are
reported separately and are excluded from the dilution trend. Positive-share
rows are divided into three equal-frequency bins per dataset using only the
share values, not labels or predictions.

For Full and `student_global`, report row count, AUC, log loss and Brier in each
bin. The paired primary mechanism quantity is

```text
global_damage = log_loss(student_global) - log_loss(Full).
```

The low-share versus high-share difference in global damage uses 2,000
student-clustered bootstrap replicates with seed 2024. This is an explanatory
diagnostic rather than a replacement for the overall-AUC gate. Same-item history
incidence and history length are reported so that repeated-target exposure is
not silently mistaken for relation-path relevance.

## Frozen execution and decision rule

Run ASSIST09, NIPS34 and Junyi because the frozen standard test confirmed both
an external win and a positive complete-control module effect on these three
datasets. XES3G5M is excluded from mechanism qualification because its frozen
real-relation module delta is negative; it remains a disclosed architecture-only
win.

The target-conditioning explanation is retained only if all conditions hold:

1. Full exceeds `student_global` in standard-validation AUC by at least `0.005`
   on at least two datasets and by at least `0.010` on at least one dataset.
2. A 2,000-replicate student-clustered paired bootstrap gives a strictly
   positive 95% AUC-delta lower bound on at least one dataset.
3. Full does not regress against `student_global` in Brier by more than
   `0.0002` on any counted dataset.
4. At least two datasets have larger mean `global_damage` in the low positive
   relevance-share bin than in the high bin; at least one such low-minus-high
   95% interval must have a strictly positive lower bound.

If the neural gate passes but the relevance trend fails, target-specific
