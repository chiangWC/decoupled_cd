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

## Result

All five feasible dataset pairs were trained with the same ORCDF implementation,
seed 42, 30-epoch ceiling, validation-AUC checkpoint selection and patience 5.
The source standard test files were not opened. NIPS34 produced no eligible
paired audit row under the frozen construction and is reported as unqualified.

| Dataset | Paired students | Rows removed/student (mean) | Concept AUC | Random AUC | Log-loss damage | Student-bootstrap 95% CI | Supports |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | 1,650 | 11.34 | 0.750463 | 0.762728 | +0.024381 | [+0.004937, +0.042894] | yes |
| ASSIST17 | 1,609 | 14.47 | 0.786011 | 0.785341 | +0.004296 | [-0.005727, +0.014814] | no |
| MOOCRadar | 1,905 | 7.94 | 0.926925 | 0.927057 | +0.001321 | [-0.003146, +0.005895] | no |
| XES3G5M | 1,898 | 2.09 | 0.775046 | 0.788854 | +0.013706 | [+0.005121, +0.022878] | yes |
| EdNet | 1,769 | 46.27 | 0.744507 | 0.746998 | +0.002607 | [-0.005916, +0.010644] | no |

Positive damage means that deleting target-concept history is worse than
deleting the same number of target-disjoint history rows. ASSIST09 and XES3G5M
support this effect. ASSIST17, MOOCRadar and EdNet have positive log-loss point
estimates but intervals crossing zero; ASSIST17's paired AUC point estimate
slightly favors the concept-depleted arm.

## Decision

**Gate B fails: 2/5 qualified datasets support the primary effect, below the
pre-registered requirement of 3.** SVGCD confirmation is therefore not run.

Together with Gate A, the evidence says that target-concept evidence can matter
materially on particular datasets, but not that student-local concept
incompleteness is a stable cross-dataset failure mode of strong CD models.
Gate A supported only ASSIST17, whereas the stronger paired intervention
supports ASSIST09 and XES3G5M. The non-overlap is itself evidence against a
single broad TKC/UKC problem claim and shows that natural coverage association
is not a reliable proxy for intervention damage.

Accordingly, no new TKC/UKC completion module is justified by this audit. The
existing standard-only RCPK route remains the active qualified result. The
negative Gate A/B results must be retained as limitations rather than rescued
by changing the outcome, dropping non-supporting datasets or weakening the
gate.

Machine artifacts remain under `results/problem_gate_b/` on xph:

- paired protocols and hashes: `protocols/<dataset>/manifest.json`;
- ORCDF checkpoints and row-aligned predictions: `orcdf/<dataset>/<arm>/`;
- combined table and bootstrap summary: `orcdf_screen/gate_b_results.csv`
  and `orcdf_screen/gate_b_summary.json`.
