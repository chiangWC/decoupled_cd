# RCPK target-conditioning mechanism gate result

Date: 2026-07-23
Branch: `codex/rcpk-target-conditioning-gate`
Preregistration: `2026-07-23-rcpk-target-conditioning-gate.md`

## Decision

The clean neural mechanism gate passes, but the proposed natural problem
explanation fails.

Holding the real relation graph, response history, propagation steps, trainable
parameters, target-aware Diagnosis, initialization and dataset recipe fixed,
RCPK's target-conditioned gather materially outperforms a single student-global
relation summary on ASSIST09 and NIPS34. This establishes that the target gather
is a functional part of the qualified standard-only module rather than an
unnecessary implementation detail.

The stronger explanatory claim is rejected. The preregistered hypothesis that a
global relation summary is especially damaged when the target-reachable share
of a student's history is low is not supported consistently: ASSIST09 has the
opposite point-estimate ordering, Junyi is positive but statistically
inconclusive, and NIPS34 has no relevance-share variation to identify the
contrast. **Target-irrelevant relational dilution must not be used as the paper's
empirically established problem statement.**

## Clean neural control

`student_global` propagates the same four response channels through the same
real graph, but sums transported signals over all exercise nodes once per
student instead of gathering them at the current target. The ordinary
Q/item-conditioned route remains target-aware. Thus the comparison removes only
target conditioning inside the relation aggregation, not target information
from the whole model.

All values below are frozen standard-validation metrics.

| Dataset | Full AUC | Student-global AUC | AUC delta | Paired 95% CI | Full Brier | Global Brier | External line | Full external margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSIST09 | **0.795594** | 0.761334 | **+0.034259** | [+0.029786, +0.038679] | **0.172425** | 0.183209 | 0.776432 | +0.019162 |
| NIPS34 | **0.789270** | 0.779150 | **+0.010120** | [+0.009065, +0.011145] | **0.184915** | 0.189618 | 0.788478 | +0.000792 |
| Junyi | **0.831026** | 0.828677 | +0.002348 | [+0.000949, +0.003714] | **0.151368** | 0.152633 | 0.820434 | +0.010592 |

The gate required two datasets at `+0.005`, one at `+0.010`, one strictly
positive paired interval, and no Brier regression above `0.0002`. ASSIST09 and
NIPS34 satisfy the magnitude conditions; both have strictly positive intervals;
Full improves Brier on all three datasets. The neural mechanism gate therefore
passes.

Secondary Full-minus-global changes are:

| Dataset | ACC | RMSE | Brier | ECE |
|---|---:|---:|---:|---:|
| ASSIST09 | +0.021732 | -0.012788 | -0.010784 | +0.020568 |
| NIPS34 | +0.008917 | -0.005435 | -0.004704 | -0.005559 |
| Junyi | +0.002428 | -0.001622 | -0.001265 | -0.000871 |

Lower RMSE/Brier/ECE is better. ASSIST09 ECE worsens even though AUC, ACC, RMSE
and Brier improve; this limitation is retained.

## Natural relevance-share diagnostic

A train-history item is counted as target-reachable only when it can reach the
current target within four frozen RCPK edges and the path uses at least one
metadata edge. Same-item train-history incidence is zero in every reported bin,
so the result is not driven by repeated exposure to the exact target exercise.

| Dataset | Positive-share rows | Low mean share | High mean share | Low global damage | High global damage | Low - high | 95% CI | Decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| ASSIST09 | 19,860 | 0.02155 | 0.34566 | +0.009527 | +0.012703 | -0.003176 | [-0.017442, +0.012168] | wrong direction, inconclusive |
| Junyi | 22,259 | 0.06102 | 0.34097 | +0.002987 | +0.001277 | +0.001710 | [-0.002825, +0.006274] | right direction, inconclusive |
| NIPS34 | 138,273 | 1.00000 | 1.00000 | -- | -- | -- | -- | unidentifiable; one unique share |

Positive global damage means the student-global control has worse row log loss
than Full. ASSIST09's middle bin, not its low bin, has the largest damage
(`+0.026510`). NIPS34's relation topology makes every history item reachable
within four steps, so any forced equal-frequency split would merely partition
identical values by row order; the analysis code explicitly rejects that
invalid contrast.

The natural gate required the low bin to exceed the high bin on two datasets and
at least one strictly positive interval. It fails. No alternate binning,
reachability radius, metric or dataset subset was tried.

## Integrity and corrected NIPS recipe

- Training implementation commit: `66e9d4cb`.
- Final analysis implementation commit: `e6a008ea`.
- Model seed is 42; bootstrap seed is 2024; no multi-seed training was run.
- Full/control initialization hashes match on every valid comparison.
- Architecture fingerprint is `f0151739cd1cbfaa` for all three datasets.
- ASSIST09 uses the frozen 20-epoch, dimension-64 recipe; Junyi uses the frozen
  40-epoch, dimension-64 recipe.
- The first NIPS control accidentally used the old `lr=1e-3`, 14-epoch discovery
  budget and was excluded before any gate decision. The valid comparison uses
  the frozen recovered recipe: dimension 64, `lr=2e-3`, 80-epoch ceiling,
  patience 5. Its initialization hash matches Full and it stopped after 18
  epochs with best epoch 13.
- Default `target_conditioned` mode retains the frozen state dict and
  architecture fingerprint. Checkpoint re-evaluation reproduces AUC; the
  maximum row-probability difference from the retained ASSIST09 CSV is
  `8.7e-7`, so the regenerated CSV is numerically but not byte-hash identical.
- Validation targets do not enter train history. Prediction rows align exactly
  on student, exercise and label.
- Focused tests, `compileall` and `git diff --check` pass. The remote environment
  lacks the optional `pytest` package, so the focused test functions were also
  executed directly.

Key generated artifacts remain ignored under
`results/rcpk_target_conditioning_gate/`. Final summary SHA-256 values are:

| Artifact | SHA-256 |
|---|---|
| ASSIST09 global summary | `f1ca7c73cfb4915710a247f2b234db59a72da7bda99adbb1817e53c59e5e466c` |
| NIPS34 global summary | `7c3962a3ca472b1d7d921eb061cbcd3073593af37c4be9cdbe16c997a2c419c2` |
| Junyi global summary | `250c4aad0963d6bc0964d312276fc0209486c91c93856a91303b681f0cf89d99` |
| ASSIST09 relevance summary | `7ded949591d482445eb98f6fac55bd700ca22e21cd75f83b836288feeaac5bfb` |
| NIPS34 relevance summary | `8300281cf9a59142c0fc7fdaebeb96f5a110a1a72916c4aec6c85fb3219d67a1` |
| Junyi relevance summary | `665567b9ae4a8774a556dbc2d943dc5d7f852a13b98311d7b47d42cd48ff0b42` |

## Research consequence

RCPK now has a stronger clean ablation than the previous Q-only/rewire evidence:
real relations matter, and their target-conditioned aggregation matters. This is
sufficient to retain the mechanism as a contribution candidate.

It is not sufficient to freeze a paper story. The next gate is a focused
literature/novelty audit of query-conditioned relational aggregation in CD,
followed by frozen test evaluation of the new student-global control only if the
mechanism remains distinguishable from existing interaction-aware graph CD.
Efficiency, item-ease calibration, TKC/UKC incompleteness and low-share
relational dilution are not eligible fallback narratives.
