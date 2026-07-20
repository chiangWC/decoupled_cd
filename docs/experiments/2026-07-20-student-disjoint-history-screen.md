# Student-disjoint History Screen

Date: 2026-07-20

## Scope and protocol

This screen asks whether the current CalibratedHistory path remains attributable
when validation students are absent from optimizer training. The protocol is
`student_disjoint_support_query`: optimizer-train, validation-support and
validation-query students are disjoint; a validation student's prediction may
use only that student's support rows plus population quantities learned from
optimizer training. Support/query `(student, exercise)` overlap is zero.

All three paths were trained independently with model `seed=42`, split
`seed=2024`, architecture fingerprint `099906acdba8c3b4`, common
dataset-specific initialization, data order and optimizer recipe:

- Full: attempted-exercise semantic pool plus calibrated history statistics.
- CalibratedSummary: removes the attempted-exercise semantic pool but retains
  calibrated aggregate statistics.
- RawSummary: removes both the semantic pool and difficulty residual.

The stronger control is selected once per dataset by validation overall AUC and
then reused for both overall and zero-coverage comparisons. It is
CalibratedSummary on all three datasets.

This is a validation-only screen. No test evaluation or test prediction was
run or inspected; every training summary records `test_metrics=null`.

## Validation metrics

Overall ACC, RMSE, Brier and ECE use all validation-query rows. Zero AUC and
Brier use `bucket:zero`, defined only from the corresponding validation
support history.

| Dataset | Path | Overall AUC | ACC | RMSE | Brier | ECE | Zero AUC | Zero Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST17 | Full | 0.766650 | 0.696512 | 0.441304 | 0.194750 | 0.018277 | 0.705160 | 0.228423 |
| ASSIST17 | CalibratedSummary | 0.766115 | 0.701878 | 0.442516 | 0.195820 | 0.033608 | 0.705160 | 0.231749 |
| ASSIST17 | RawSummary | 0.764529 | 0.691818 | 0.442319 | 0.195646 | 0.026276 | 0.700246 | 0.233006 |
| XES3G5M | Full | 0.788589 | 0.845794 | 0.343043 | 0.117678 | 0.010344 | 0.780943 | 0.113704 |
| XES3G5M | CalibratedSummary | 0.787365 | 0.840187 | 0.344787 | 0.118878 | 0.030933 | 0.773706 | 0.117134 |
| XES3G5M | RawSummary | 0.784578 | 0.841121 | 0.343937 | 0.118292 | 0.023594 | 0.774893 | 0.116206 |
| MOOCRadar | Full | 0.927791 | 0.913688 | 0.249779 | 0.062390 | 0.009496 | 0.949838 | 0.035147 |
| MOOCRadar | CalibratedSummary | 0.927750 | 0.914418 | 0.249474 | 0.062237 | 0.010783 | 0.951513 | 0.034579 |
| MOOCRadar | RawSummary | 0.926937 | 0.912473 | 0.250235 | 0.062618 | 0.012851 | 0.947513 | 0.034385 |

Validation-query sample sizes are 2,982 rows from 174 students on ASSIST17,
2,140 rows from 212 students on XES3G5M, and 4,113 rows from 212 students on
MOOCRadar. Their zero buckets contain 81, 946 and 1,082 rows, respectively.

## Paired differences

Intervals use 2,000 student-clustered bootstrap replicates with bootstrap
`seed=2024`. Positive values favor Full.

| Dataset | Overall delta AUC | Overall 95% CI | Zero delta AUC | Zero 95% CI |
|---|---:|---:|---:|---:|
| ASSIST17 | +0.000535 | [-0.006046, +0.007154] | +0.000000 | [-0.041845, +0.041448] |
| XES3G5M | +0.001224 | [-0.006407, +0.008830] | +0.007237 | [-0.006604, +0.020287] |
| MOOCRadar | +0.000041 | [-0.001746, +0.001878] | -0.001676 | [-0.007047, +0.002343] |

## Data identity

The manifests report zero train/validation/test student overlap, zero
support/query group overlap, zero Q-mapping conflicts and zero Q-interaction
mismatches.

- ASSIST17
  - manifest: `assist_17_student_disjoint_seed2024/manifest.json`
  - manifest SHA-256: `4d0598e139e7818f0e06ad4102dc70b63c2272e717f5cffe83bdb8440ce70447`
  - train: `36cd7833175346be3d28ebaf00a12a007f5c30898e5d7b28ebf5c932be8209cd`
  - valid support: `2bb87c03e617877da9200aac7d2dbf433cce1c7db1a412f3e02a39fcecddd01c`
  - valid query: `61cb6793b3d8ac1521c886765df92bde369213c2d7e67a7367831830ecf4d09d`
  - Q matrix: `23a59ec57c3b454d2d3fece3760aa91ec65a297e357766265466591b5bb0f9e3`
- XES3G5M
  - manifest: `xes3g5m_student_disjoint_seed2024/manifest.json`
  - manifest SHA-256: `b9a306169838d1d4faefea80b9a1d10cd423dc1ae6db972710a8c8be5029ed94`
  - train: `a7c5c147157fc9eba929d929d726144051fcf634dd8f65a483109e4cc42ecadb`
  - valid support: `0fa6e248e8c19a006a6bb27c514dd9b8b47e92102f19a1751a9d6fecfc21ebbb`
  - valid query: `dbd3a832f8c2df281cb0649cdcc9faae8af4b2a140af21e02f4d59ddca3ccddb`
  - Q matrix: `13965c21cc2728281df235877805fcbf137bf851a4f9e232de6ec14620546df7`
- MOOCRadar
  - manifest: `moocradar_student_disjoint_seed2024/manifest.json`
  - manifest SHA-256: `59e7ff7a96a1ad1c7ff9675fe0c1db5904516822b9b544ec677aad37757b5823`
  - train: `31d7b1e4ba2226597c9653da33c488521ed4036aeba63a6de0f5d94dc1944641`
  - valid support: `06dc1352f542f4d1988251b07527163e90aad649bcd7cea1299aece26c658e28`
  - valid query: `707c054bc12c5b5dec716fc50255ef91714f79fb4e8a30e8e093f187b8130980`
  - Q matrix: `b2526274cf733d0170028037b727c682eb81366804dc0f72703699fb90aaf8af`

## Decision

CalibratedHistory does not pass the student-disjoint module gate. Relative to
the stronger control, no dataset reaches an overall gain of 0.002. On the zero
bucket only XES3G5M exceeds 0.002; ASSIST17 is tied and MOOCRadar is worse.
Every paired 95% interval crosses zero. The requirement of gains of at least
0.002 on two datasets and a positive CI lower bound on at least one dataset is
therefore unmet. MOOCRadar does not rescue the mechanism.

This result is scoped to the student-disjoint support-query protocol. It does
not rewrite earlier same-student-split measurements, but it prevents using
those measurements alone to claim that CalibratedHistory is an inductive
new-student module.
