# Experiment 114: Cross-dataset exp110-style runs

## Status

`cross_dataset_seed2024_recorded`

## Verdict

The exp110-style low-memory pure-CDM protocol transferred cleanly to ASSIST17
and NIPS34 on seed2024. Both datasets completed with the original exp110
`dual64x80`, `branch_bce=0.18`, `recompute_minibatch`, `batch_size=65536`,
`lr=3e-4`, single-checkpoint protocol.

Junyi exposed the expected dense student-concept memory limit. The original
exp110 `dual64x80` protocol OOMed on a 24GB RTX 4090 because Junyi has
`10000 x 706` student-concept cells. Lower-memory CLI-only probes found that
`dual64x32` and `single64` still OOM, while the successful full run used the
Junyi dataset-default primary tower (`concept_dim=16`) plus a `secondary_dim=32`
dual tower. Earlier chat shorthand called this run `dual32x32`, but the JSON
summary shows the exact executed configuration is `dual16x32`.

No hybrid stacker, validation-trained combiner, checkpoint average, test-label
fitting, or train-history tabular inference side channel was used.

## Base

- Branch: `exp/trellis-trial`
- Remote host: `xph-pc`
- Remote project path: `/home/xph/jwc/research/decoupled_cd`
- Seed: `2024`
- Protocol family: experiment 110 single-checkpoint pure-CDM runner
- Result directories:
  - `results/exp110_cross_dataset/`
  - `results/junyi_memory_trials/`

## Dataset Scale

| dataset | students | exercises | concepts in run summary | train rows | valid rows | test rows |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST17 | 1702 | 3162 | 102 | 282380 | 30541 | 77360 |
| NIPS34 | 4918 | 948 | 57 | 999467 | 108670 | 274590 |
| Junyi | 10000 | 706 | 706 | 262473 | 24299 | 67063 |

NIPS34 was uploaded as `data/NIPS.tar.gz` for the run, then moved out of the
repo to keep the remote worktree clean:

```text
/home/xph/jwc/research/datasets/NIPS/NIPS.tar.gz
/home/xph/jwc/research/datasets/NIPS/NIPS34/process_data/
```

The NIPS34 result JSON still records the in-repo paths used at run time. Future
re-runs should pass the moved absolute paths explicitly.

## Command Template

ASSIST17 and NIPS34 used the original exp110 training shape:

```bash
python scripts/train.py \
  --train-interactions <train.csv> \
  --valid-interactions <valid.csv> \
  --test-interactions <test.csv> \
  --q-matrix <Q_matrix.csv> \
  --graph-mode single \
  --concept-dim 64 \
  --gs-mode conditional \
  --high-concept-logit-adapter --high-concept-logit-min-count 2 \
  --pairwise-history-interaction-adapter --pairwise-history-interaction-min-count 2 \
  --gs-difficulty-adapter \
  --interpretable-readout-expert-adapter --interpretable-readout-expert-count 3 \
  --student-conditioned-ukc-readout-residual \
  --concept-evidence-readout-residual \
  --concept-evidence-readout-min-count 1 \
  --concept-evidence-readout-max-count 1 \
  --concept-evidence-readout-min-seen-ratio 1.0 \
  --concept-evidence-readout-max-logit 0.5 \
  --concept-evidence-prior-residual \
  --concept-evidence-prior-min-count 1 \
  --concept-evidence-prior-min-seen-ratio 1.0 \
  --concept-evidence-prior-max-logit 0.3 \
  --concept-evidence-prior-min-confidence 0.75 \
  --concept-evidence-prior-min-abs-mastery 0.5 \
  --concept-evidence-prior-apply-mode train_only \
  --concept-evidence-prior-train-start-epoch 135 \
  --history-evidence-logit-prior-residual \
  --history-evidence-logit-prior-location loss_only \
  --history-evidence-logit-prior-min-count 1 \
  --history-evidence-logit-prior-min-seen-ratio 0.0 \
  --history-evidence-logit-prior-max-logit 4.0 \
  --history-evidence-logit-prior-component-cap 4.0 \
  --history-evidence-logit-prior-weight-student 0.0 \
  --history-evidence-logit-prior-weight-exercise 0.0 \
  --history-evidence-logit-prior-weight-target-concept 0.44 \
  --history-evidence-logit-prior-weight-concept 0.22 \
  --history-evidence-logit-prior-weight-mastery 0.22 \
  --history-evidence-cognitive-alignment-weight 0.05 \
  --history-evidence-cognitive-alignment-final-weight 0.0881 \
  --history-evidence-cognitive-alignment-anneal-start-epoch 170 \
  --history-evidence-cognitive-alignment-anneal-end-epoch 230 \
  --dual-cdm-ensemble \
  --dual-cdm-secondary-concept-dim 80 \
  --dual-cdm-branch-bce-weight 0.18 \
  --training-mode recompute_minibatch \
  --batch-size 65536 \
  --learning-rate 0.0003 \
  --epochs 300 \
  --checkpoint-selection-metric auc \
  --seed 2024
```

Junyi's successful run kept the same protocol except:

```bash
--dataset junyi
--concept-dim 32                 # overwritten by dataset defaults to 16
--dual-cdm-secondary-concept-dim 32
```

Because `--dataset junyi` applies dataset defaults after parsing, the explicit
`--concept-dim 32` matched the parser default and was rewritten to Junyi's
dataset default `16`. The run summary is the source of truth:
`concept_dim=16`, `dual_cdm_secondary_concept_dim=32`.

## Seed2024 Results

| dataset | variant | test AUC | test ACC | RMSE | Brier | ECE | best val AUC | best epoch | run-summary CUDA GB | result JSON |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ASSIST17 | exp110 `dual64x80` | 0.7799261764 | 0.7132497415 | 0.4345909466 | 0.1888692909 | 0.0091661134 | 0.7792912392 | 58 | 3.867387 | `results/exp110_cross_dataset/assist_17_seed2024_exp110.json` |
| NIPS34 | exp110 `dual64x80` | 0.7847075505 | 0.7149568448 | 0.4329188996 | 0.1874187736 | 0.0172906271 | 0.7819392372 | 61 | 3.853576 | `results/exp110_cross_dataset/nips34_seed2024_exp110.json` |
| Junyi | exp110-near `dual16x32` | 0.8213176536 | 0.7615823927 | 0.4043068619 | 0.1634640386 | 0.0312580187 | 0.8144277655 | 168 | 16.198913 | `results/junyi_memory_trials/junyi_seed2024_exp110_dual32x32_300ep.json` |

## Junyi Student-Subset Validation

After adding target-student subset propagation, Junyi was re-run with the same
exp110-style feature stack and `dual16x32` capacity, but with
`--training-mode student_recompute_minibatch --student-batch-size 2048`.
This is not the original exp110 `recompute_minibatch` protocol; it is a
runtime/training-mode optimization for datasets where dense
`students x concepts x dim` propagation dominates.

The optimized full Junyi run completed on `cuda:2` in `1084s` (`18m04s`),
stopping at epoch `136` after best epoch `131`. A follow-up same-command
memory-check rerun with per-second `nvidia-smi --query-compute-apps` sampling
completed in `1152s` (`19m12s`) and reproduced the same metrics and best
epoch. Each epoch used five student-batched optimizer steps. The run summary
records
`num_students=10000`, `num_exercises=706`, `num_concepts=706`,
`concept_dim=16`, and `dual_cdm_secondary_concept_dim=32`.

| dataset | variant | test AUC | test ACC | RMSE | Brier | ECE | best val AUC | best epoch | peak `nvidia-smi` process GiB | wall time | result JSON |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Junyi | `dual16x32` + `student_recompute_minibatch`, student batch 2048 | 0.8245031558 | 0.7655935464 | 0.4011424020 | 0.1609152267 | 0.0145228535 | 0.8181139402 | 131 | 13.076172 | 18m04s original, 19m12s memcheck | `results/junyi_memory_trials/junyi_seed2024_student_recompute2048_dual16x32_300ep.json`; memcheck: `results/junyi_memory_trials/junyi_seed2024_student_recompute2048_dual16x32_300ep_memcheck.json` |

Compared with the old Junyi `recompute_minibatch` record, wall-clock improved
from about `5342s` (`1h29m`) to `1084s`, or about `4.9x` faster end-to-end.
Rough per-epoch time improved from about `30.9s` to `8.0s`, about `3.9x`
faster. The memcheck rerun measured a `nvidia-smi` process-memory peak of
`13390 MiB` (`13.08 GiB`) for the optimized run. The old Junyi run did not
collect this `nvidia-smi` peak, so the old run's process-memory peak is unknown
unless the old mode is rerun with the same sampler. The optimized run also
improved the recorded test metrics:
test AUC `0.824503` versus `0.821318`, ACC `0.765594` versus `0.761582`,
RMSE `0.401142` versus `0.404307`, Brier `0.160915` versus `0.163464`, and
ECE `0.014523` versus `0.031258`.

This follow-up does not change the ASSIST17 or NIPS34 exp114 records. Those
datasets still report the original exp110 `dual64x80` + `recompute_minibatch`
protocol. The student-subset mode is only recorded here as a practical Junyi
path for future full-dataset experiments.

## Junyi Runtime Note

The successful Junyi low-memory run was not just memory-heavy; it was also much
slower than the other cross-dataset runs. The remote log starts at
`2026-05-25 14:13:57` and finishes at `2026-05-25 15:42:59`, so wall-clock was
about `1h29m`. The run stopped at epoch `173` with best epoch `168`, matching
the default early-stop patience of five non-improving epochs; it did not run
all 300 configured epochs.

Each epoch used only five recompute-minibatch optimizer steps. That means the
runtime is dominated by Junyi's dense `10000 x 706` student-concept propagation
and evaluation path rather than by the interaction-batch count. Treat Junyi
full-run checks as long-running jobs even after dimensions are reduced enough
to fit in memory.

Follow-up optimization note:

- Use `--training-mode student_recompute_minibatch` with
  `--student-batch-size <N>` for future Junyi probes when the goal is faster
  iteration. This batches optimizer steps by student IDs and uses target-student
  subset propagation, avoiding repeated all-student propagation inside
  interaction minibatches.
- The default `full_batch` and `recompute_minibatch` modes remain available for
  protocol comparisons. Treat `student_recompute_minibatch` as a runtime
  optimization/training-mode change and record it explicitly in any Junyi result
  summary. The validated full-run setting was `--student-batch-size 2048`,
  which reduced Junyi wall-clock from about `1h29m` to `18m04s`. A
  same-command memcheck rerun measured `13.08 GiB` peak `nvidia-smi` process
  memory.

## Junyi Memory Findings

Junyi's original exp110 `dual64x80` run failed on dense propagation tensors:

- `results/exp110_cross_dataset/junyi_seed2024_exp110.log`
  - OOM while allocating `(num_students, num_concepts, dim)` in
    `_aggregate_exercise_messages_by_concept`
  - attempted allocation: `2.11 GiB`
  - process memory in use: about `21.62 GiB`
- `results/exp110_cross_dataset/junyi_seed2024_exp110_retry_expandable.log`
  - retry on an otherwise free GPU with
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - OOM while concatenating correct and incorrect TKC components
  - attempted allocation: `4.21 GiB`
  - process memory in use: about `21.51 GiB`

Additional one-epoch CLI-only probes:

| variant | result | note |
|---|---|---|
| `dual64x32` | OOM | failed while concatenating dense propagation components, attempted `1.68 GiB` |
| `single64` | OOM | reached backward before failing, attempted `1.68 GiB` |
| executed `dual16x32` | passes | one-epoch smoke peak was about `16.21GB`; full run peak `16.20GB` |

The controlling scale is not interaction rows; it is dense
`students x concepts x dim` state:

| dataset | student-concept cells |
|---|---:|
| ASSIST09 ordered | 306639 |
| NIPS34 | about 305k in raw scale |
| Junyi | 7060000 |

Junyi has about `23x` the ASSIST09 student-concept cell count. This explains
why exp110 on ASSIST09/NIPS34 fits in the `~4-6GB` band while Junyi needs a
much smaller tower configuration without model-level propagation chunking.

## Decision

- Record ASSIST17 and NIPS34 as successful seed2024 cross-dataset transfers of
  the exp110 original `dual64x80` protocol.
- Record Junyi original exp110 as not runnable on a 24GB RTX 4090 without code
  changes or much lower dimensions.
- Treat the successful Junyi `dual16x32` result as a pragmatic low-memory
  cross-dataset probe, not as an exp110-equivalent result.
- Treat Junyi full-run checks as long-running jobs: the successful reduced
  run took about `1h29m` on `cuda:1` and stopped at epoch `173` after best
  epoch `168`.
- For future Junyi experiments, prefer the validated
  `student_recompute_minibatch` mode when the question does not require an
  old-mode runtime comparison. The seed2024 `dual16x32` follow-up ran in
  `18m04s`; a same-command memcheck rerun measured `13.08 GiB` peak
  `nvidia-smi` process memory; the run produced test AUC `0.824503`.
