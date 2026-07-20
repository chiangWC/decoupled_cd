# Student-local support composition / difficulty-equating opportunity audit

Date: 2026-07-20
Status: pre-implementation protocol
Scope: ASSIST17 and XES3G5M, split seed 2024, CPU only

## Question

The rejected cross-concept completion audit showed that support history carries strong
overall-ability signal, but concept-to-concept features did not improve over a stronger
ability control. This audit asks a narrower question:

> After target item/concept difficulty and the student's raw support accuracy are
> controlled, does the difficulty composition of the support set improve prediction
> for a disjoint student's future responses?

A positive result would motivate a support-composition or difficulty-equating component.
It would not, by itself, establish a TKC/UKC completion contribution.

## Data protocol

1. Build the existing student-disjoint protocol from each standard split:
   optimizer training students, validation support students, and validation query
   students are mutually disjoint.
2. Fit audit predictors only from optimizer-training students. Deterministically split
   each optimizer student's interactions by atomic student-exercise group into support
   and query rows. Require at least 10 support rows and 3 query rows.
3. Evaluate once on the true validation-support to validation-query transition. No
   validation student or validation response is used to fit a predictor, item statistic,
   scaler, or difficulty bin boundary.
4. Interpret Q-matrix membership as the union of every long-form row for an exercise.
   Exact-zero means that none of the target exercise's unioned concepts occurs in support.
5. Any source row touching a query group is absent from that student's support.

## Train-only item statistics

Item ease is a beta-smoothed correctness rate. For optimizer-student fitting examples,
all item and global counts contributed by that student are subtracted before computing
ease, including the target query response. For validation students, statistics use the
complete optimizer-train pool and no validation rows. Item frequency/confidence and
difficulty-bin boundaries follow the same rule.

This prevents a student's query label from leaking through target-item or support-item
population statistics.

## Compared predictors

All predictors receive identical target-side information:

- target item population ease and population count;
- unioned target-Q multi-hot vector and Q cardinality;
- support raw correctness rate, support length, and concept coverage.

The strong naive controls are:

- raw_logistic: regularized logistic regression on only the common features;
- raw_extra_trees: an ExtraTrees classifier on only the common features.

The corresponding candidates use exactly the same estimator and hyperparameters but add:

- support-item ease mean, dispersion and quantiles;
- response-minus-item-ease residual distribution;
- five train-defined difficulty-bin accuracies, counts and confidences;
- target-relative support difficulty, harder/easier proportions, and a
  target-difficulty-local residual estimate.

Thus equated_logistic is attributable against raw_logistic, and
equated_extra_trees against raw_extra_trees. The formal comparison additionally uses
the strongest naive control by validation overall AUC, so estimator capacity cannot
create the claimed gain.

## Capacity safeguard

ExtraTrees samples only a subset of features at each split. Adding many continuous
equating summaries can therefore improve the chance of seeing any student-side feature,
even if their semantics are irrelevant. A capacity control repeats the six common
numeric features deterministically until its numeric width equals the equated input.
It has the same target item/Q features, estimator and hyperparameters. The formal
stronger control is selected from raw and capacity-control predictors by overall AUC.

## Metrics and gate

Report row-level AUC and Brier overall, plus the exact-zero slice when both labels exist.
ACC, RMSE, sample count and positive rate are recorded for interpretation.

The opportunity gate compares the best equated candidate with the stronger naive
control:

- both datasets must improve overall AUC by at least 0.003;
- at least one dataset must improve by at least 0.005;
- response-weighted Brier must not regress by more than 0.0001 on either dataset.

Exact-zero is diagnostic only because ASSIST17 may have few examples. A failed gate is
recorded as evidence against this component and is not repaired by lowering thresholds
or relabeling the rejected cross-completion mechanism.
