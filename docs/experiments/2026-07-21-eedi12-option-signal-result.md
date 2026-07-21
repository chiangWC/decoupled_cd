# Eedi Tasks 1&2 option-signal extension result

## Decision

The Eedi extension stopped at the preregistered pseudo-target feasibility
check. No estimator, control comparison, bootstrap, neural module, validation
result, or test result was run. The categorical-response route therefore still
has one strongly positive dataset (ENEM) and is not activated as a cross-dataset
paper module in this round.

This is a protocol-scope failure, not a negative option-effect estimate. The
all-subject Eedi Q representation includes broad hierarchy ancestors that are
shared across many questions; after retaining training history, too few hidden
rows have target concept coverage below `0.5`.

## Frozen execution

- implementation commit: `445d6373`;
- model seed `42`, pseudo split seed `2024`;
- source rows: 15,867,850;
- deterministic eligible cohort: 5,000 students and 662,705 rows;
- cohort questions/concepts: 27,112 / 386;
- option coverage and option-label agreement: `1.0 / 1.0`;
- duplicate student-question rows: `0`;
- no validation, test, answer-metadata, target-label, or target-option field was
  opened by the feature construction.

The label-free pseudo split produced 532,193 support rows and 130,512 hidden
rows. Only 242 hidden rows from 207 students were in the predeclared
`low_coverage` scope, with 88 negative and 154 positive labels. The fixed
minimums were 500 rows, 100 students and 100 examples of each label. The
negative-label minimum therefore failed.

## Reproducibility

- selected raw-user hash:
  `c98fca4994a2761efc04cde1b5a2fe19f50c67de640b09c3a88efcb3600a89ca`;
- pseudo-target selection hash:
  `2e91839f3e90b294cff36057afc00e279ac655e4d33aa3a5a4333fe72866a862`;
- aggregate result SHA-256:
  `bd00edf4120aca42f37adc52e3a7b28e36503ef0b305858d6f081387960492a4`;
- Eedi summary SHA-256:
  `e66d96a4052907c919fa8db673b852a75369239403cce0665d3aaaef00eb0df8`.

Generated artifacts are under
`results/eedi12_option_signal_extension_v1/` on the experiment host.

## Consequence

Do not change Eedi to leaf-only Q, overall-only targets, a larger cohort, or a
different coverage threshold to rescue this preregistered run. ENEM's
`+0.010376` option signal remains valid and may motivate a future route if an
independently admitted second option-rich dataset supports it. The present
search returns to mechanisms already supported on at least two datasets.
