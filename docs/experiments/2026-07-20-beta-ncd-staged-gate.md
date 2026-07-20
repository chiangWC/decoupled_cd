# BETA-NCD Paper-Aligned Staged Gate

## Scope

This validation-only screen asks whether BETA-CD with the NCD backbone is a
credible external baseline under the student-disjoint full-support protocol.
It is not a proposed module. Model seed is 42, protocol split seed is 2024,
and no test interactions were parsed.

## Invalid v1 run and formula repair

The implementation at commit `d1a980a` incorrectly averaged the inner
response BCE over both Monte Carlo samples and support responses. On the
ASSIST17 one-epoch checkpoint, adaptation therefore had essentially no
behavioral effect:

- mean absolute posterior-mean movement: `6.26e-6`;
- mean absolute prior-to-adapted probability movement: `4.16e-8`;
- mean absolute original-to-flipped-support probability movement: `8.19e-8`.

Those runs are invalid and generated artifacts are archived under
`results/student_local_inductive/beta_ncd_invalid_mean_reduction/`.

Commit `0b2928d` restores the paper equations:

```text
inner = mean_over_MC(sum_over_support(response NLL))
        + eta * KL(posterior || prior)
outer = -log(mean_over_MC(joint query likelihood))
```

There are exactly three complete-support inner updates, with one KL term per
update, and the outer loss is not divided by query length. The repaired
architecture fingerprint is
`20245c18791fd379b03c1119358e63138b82c9e14979293726929471d95467ce`.
All v1 checkpoints are incompatible performance artifacts.

## Pre-registered five-epoch gate

Every condition below was fixed before inspecting the five-epoch result.
Passing would authorize only an ASSIST17 20-epoch screen, not a three-dataset
or 100-epoch run.

### A. Learning curve

1. Best validation AUC through epoch 5 is at least `0.510`.
2. Best validation AUC improves over epoch 1 by at least `0.010`.
3. Epoch-5 train meta-loss is at most 97% of epoch-1 loss.

### B. Support use

At the best checkpoint:

1. adapted AUC minus prior-only AUC is at least `0.005`;
2. adapted AUC minus all-support-labels-flipped AUC is at least `0.010`;
3. `mean|p_adapt-p_prior| / std(p_adapt)` is at least `0.05`;
4. `mean|p_adapt-p_flip| / std(p_adapt)` is at least `0.10`.

### C. Length dependence

1. In every student support-length quartile, adapted-minus-prior AUC is at
   least `-0.005`.
2. After separately regressing support accuracy and each student's signed
   mean query shift on `log1p(support_length)`, their residual Pearson
   correlation is at least `0.30`.

### D. Numerical stability

All losses, parameters, posteriors, and predictions must be finite. Every
inner learning rate must remain in `[0.02, 0.5]`; adapted log standard
deviations must remain in `[-8, 2]`; studentwise mean absolute posterior
movement must have p99 at most `0.5` and maximum at most `2.0`; rowwise
absolute prior-to-adapted probability movement must have p99 at most `0.25`;
and no epoch loss may exceed 1.5 times the epoch-1 loss.

## Result

The repaired one-epoch smoke was stable and increased mean posterior movement
to `0.001489`, confirming that the formula bug was removed. The fixed
five-epoch run produced:

| Epoch | Train meta-loss | Validation AUC |
|---:|---:|---:|
| 1 | 25.887475 | 0.497046 |
| 2 | 22.973122 | 0.494670 |
| 3 | 22.837361 | 0.495028 |
| 4 | 22.738145 | 0.497510 |
| 5 | 22.655792 | **0.505358** |

Section A failed: best AUC was below `0.510`, and the gain over epoch 1 was
`0.008312`, below `0.010`. Loss fell by 12.48%, so A3 passed.

The counterfactual support diagnostics passed Sections B and C. They were
recomputed from the checkpoint on CPU; minor AUC differences from the stored
GPU prediction file can occur when nearly tied probabilities change order.

- adapted/prior/flipped AUC:
  `0.505771 / 0.500443 / 0.494320`;
- adapted-minus-prior and adapted-minus-flipped AUC:
  `+0.005328 / +0.011452`;
- standardized probability effects:
  `0.0723 / 0.1753`;
- length-controlled residual correlation: `0.4505`;
- quartile adapted-minus-prior AUC:
  `+0.00339 / +0.00778 / +0.00265 / +0.00684`.

Section D also passed: inner learning rates were approximately `0.10386`,
adapted log standard deviations stayed in `[-4.0013, -3.0002]`, posterior
movement p99/max were `0.01435 / 0.01683`, rowwise probability-movement p99
was `0.04033`, and all audited values were finite.

## Decision and protocol boundary

The repaired baseline genuinely consumes support and is numerically stable,
but it failed the pre-registered learning-speed gate. It is therefore recorded
as `rejected_after_5ep_staged_gate`; no 20/100-epoch or cross-dataset run is
authorized. Corrected generated artifacts are stored, but not versioned, under
`results/student_local_inductive/beta_ncd_rejected_after_5ep_gate/`.

If BETA-NCD is revisited as a same-protocol external baseline, it must retain
the full support available to every compared model. The paper's 3/5/10-shot
setting changes the model input, coverage buckets, and target-slice semantics;
it can only be a separately fingerprinted few-shot auxiliary protocol with
fixed sampling and recomputed coverage. It cannot replace the full-support
result or define an external win line.
