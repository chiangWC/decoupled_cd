# Gate B: paired concept-depletion intervention

Date: 2026-07-21
Branch: `codex/concept-depletion-gate-b`
Parent: `8790633a`

## Question

Gate A found natural student-local target-concept coverage, but only ASSIST17
showed a positive two-way-fixed-effect association between missing coverage and
external-model log loss. Gate B asks the stronger intervention question:

> Holding the evaluated student/item row and the number of removed training
> interactions fixed, does removing the target concept evidence damage a strong
> CD model more than removing the same amount of non-target history?

This is a problem-existence audit, not model selection. It uses only standard
train and validation files. Repository test files are neither opened nor
copied. No new TKC/UKC model is implemented unless this gate passes.

## Paired data construction

For each dataset, the original validation rows are split by a stable hash into
an early-stopping split and an audit split. For each student, at most one audit
row is selected without reading its label. A row is eligible when:

- every target Q concept occurs in that student's standard training history;
- deleting every student-history row touching any target concept leaves at
  least five history rows;
- there are enough non-target history rows to remove the same count in the
  quantity-matched arm.

The two arms are:

- `concept_depleted`: remove every selected student's training row whose Q set
  intersects the selected target row's Q set;
- `random_depleted`: remove the same number of rows from the same student,
  sampled by a stable hash only among rows disjoint from the target Q set.

The two training files therefore have identical row counts globally and per
selected student. In the concept arm, selected audit rows have exact-zero
target coverage; in the random arm, they retain full target coverage. Both
arms share the same Q matrix, early-stopping rows, audit rows, seed and external
model recipe.

## Estimand and gate

ORCDF is the common first-screen model because it runs under one implementation
and one topology across the current pool. For each selected audit row, define

```text
concept-specific damage = loss(concept_depleted) - loss(random_depleted)
```

The primary loss is row-wise log loss. Brier and AUC are secondary. A 2,000
replicate student-cluster bootstrap with seed 2024 estimates the 95% interval;
at most one target row is selected per student, but clustering is retained in
the implementation contract.

Gate B passes the ORCDF screen only when:

- at least three datasets contain at least 500 eligible audit students and both
  labels;
- at least three datasets have mean log-loss damage above zero with a 95%
  interval lower bound above zero.

Only if that screen passes will SVGCD repeat the same frozen arms on the three
supporting datasets. The TKC/UKC problem is admitted only if both strong models
support at least three common datasets. Otherwise the coverage-incompleteness
story remains rejected; no metric substitution, post-hoc dataset removal, or
threshold relaxation is allowed.

ASSIST09, ASSIST17, MOOCRadar, NIPS34, XES3G5M and EdNet are screened for
construction feasibility. Junyi is excluded because its one-exercise/
one-concept protocol provides no natural full-coverage audit row.

## Integrity checks

- selection and removal never read audit labels;
- target interactions never enter either training history;
- concept and random arms have identical selected students, audit rows,
  per-student removal counts and total row counts;
- every selected row is exact-zero in the concept arm and full-coverage in the
  random arm;
- prediction alignment includes a stable `audit_row_id`;
- standard dataset and generated-arm SHA-256 hashes are recorded;
- seed is fixed to 42 for model training and 2024 for deterministic protocol
  construction/bootstrap.
