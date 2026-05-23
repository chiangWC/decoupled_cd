# Experiment 110: Recompute minibatch low-memory dual CDM

## Status

`low_memory_pure_cdm_signal_confirmed`

## Verdict

`recompute_minibatch` training with a large batch keeps the experiment 106
heavy dual-CDM structure in the `~6GB` memory band while preserving almost all
of the experiment 104/106 AUC.

Best candidate:

```bash
SEED=<seed> OUTPUT=results/pure_cdm_distillation/seed${SEED}_dual64x80_branchbce018_recompute65536_lr3e4_300ep.json \
  bash scripts/run_assist09_history_alignment_trial.sh \
    --dual-cdm-secondary-concept-dim 80 \
    --dual-cdm-branch-bce-weight 0.18 \
    --training-mode recompute_minibatch \
    --batch-size 65536 \
    --learning-rate 0.0003 \
    --epochs 300
```

Four-seed AUC is `0.778252/0.778263/0.777449/0.779321`, mean
`0.778321`, population stdev `0.000665`, and peak CUDA memory `6.19GB`.

This is the clearest low-memory pure-CDM signal found so far:

- vs experiment 105 `single64` mean `0.776736`: `+0.001585` AUC
- vs experiment 109 `dual64x32` mean `0.777674`: `+0.000647` AUC
- vs experiment 104 full-batch `dual64x80 branchBCE=0.10` mean `0.778370`:
  `-0.000049` AUC
- vs experiment 106 full-batch `dual64x80 branchBCE=0.18` mean `0.778579`:
  `-0.000258` AUC
- peak CUDA drops from about `11.76GB` for full-batch `64x80` to `6.19GB`.

No hybrid stacker, validation-trained combiner, fixed checkpoint average,
test-label fitting, or train-history tabular inference side channel was used.
This is still a single training run and single selected checkpoint per seed.

## Base

- Branch: `exp/pure-cdm-distillation`
- Starting point: experiment 106 `dual64x80 + branchBCE=0.18`
- Prior low-memory boundary: experiment 109 `dual64x32 + branchBCE=0.18`
- Dataset and protocol: ASSIST09 ordered split, single graph, train-history
  propagation for valid/test.

## Engineering

Added an optional secondary-tower activation checkpoint flag:

```bash
--dual-cdm-secondary-checkpoint
```

This flag is honest recomputation, not freezing or detaching. It preserves
secondary-tower gradients and only trades compute for activation memory.
However, it was not the winning path in this experiment: full-batch
`dual64x32` only dropped from about `8.85GB` to `8.64GB` with checkpointing.
The real memory win came from existing `recompute_minibatch` training.

Remote verification after the code change:

```bash
python -m py_compile models/ensemble_cdm.py scripts/train.py tests/test_decoupled_cdm.py
python -m unittest tests.test_decoupled_cdm tests.test_training_modes tests.test_train_cli
```

Passed remotely: 63 tests OK.

## Search Notes

### Activation checkpoint smoke

| variant | seed | epochs | AUC | peak CUDA GB | verdict |
|---|---:|---:|---:|---:|---|
| `dual64x32 branchBCE=0.18 + secondary_checkpoint` | 2026 | 1 | 0.498965 | 8.640988 | memory drop too small |

### Recompute minibatch smoke

| variant | seed | epochs | optimizer steps/epoch | peak CUDA GB | verdict |
|---|---:|---:|---:|---:|---|
| `dual64x32 batch=32768 + secondary_checkpoint` | 2026 | 1 | 6 | 3.433066 | memory solved |
| `dual64x80 batch=32768 + secondary_checkpoint` | 2026 | 1 | 6 | 4.719258 | memory solved |
| `dual64x80 batch=32768` | 2026 | 1 | 6 | 4.870220 | checkpoint not needed |
| `dual64x80 batch=65536` | 2026 | 1 | 3 | 6.186983 | closest to full-batch while under 7GB |

### Triage runs

| variant | seed | AUC | ACC | RMSE | Brier | ECE | peak CUDA GB | best epoch | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dual64x80 batch=32768 lr=1e-3` | 2026 | 0.776671 | 0.737274 | 0.421018 | 0.177256 | 0.039551 | 4.870220 | 39 | underfits/oversteps vs exp109 |
| `dual64x80 batch=65536 lr=1e-3` | 2026 | 0.777330 | 0.739253 | 0.419953 | 0.176361 | 0.035585 | 6.190541 | 62 | better but still weak |
| `dual64x80 batch=65536 lr=1e-3 brier-select` | 2026 | 0.777157 | 0.739519 | 0.419539 | 0.176013 | 0.029804 | 6.190541 | 56 | better calibration, weaker AUC |
| `dual64x32 batch=65536 lr=1e-3` | 2026 | 0.776850 | 0.737407 | 0.420330 | 0.176677 | 0.035941 | 4.527024 | 83 | lower memory, weaker AUC |
| `dual64x80 batch=65536 lr=3e-4` | 2026 | 0.777449 | 0.738435 | 0.419515 | 0.175993 | 0.028955 | 6.190541 | 140 | two-anchor candidate |

Decision from triage: use the heavier `64x80` secondary tower, keep
`batch=65536`, and lower learning rate to `3e-4`. Smaller batch oversteps in
epoch terms; smaller secondary tower loses too much AUC.

## Four-Seed Result

All rows use:

- `dual_cdm_secondary_concept_dim=80`
- `dual_cdm_branch_bce_weight=0.18`
- `training_mode=recompute_minibatch`
- `batch_size=65536`
- `learning_rate=0.0003`

| seed | AUC | ACC | RMSE | Brier | ECE | peak CUDA GB | best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.7782518971 | 0.7402043807 | 0.4189197958 | 0.1754937953 | 0.0278152301 | 6.190541 | 151 |
| 2025 | 0.7782634194 | 0.7379588574 | 0.4194837142 | 0.1759665865 | 0.0305863060 | 6.190541 | 155 |
| 2026 | 0.7774488561 | 0.7384346039 | 0.4195146333 | 0.1759925275 | 0.0289546935 | 6.190541 | 140 |
| 2027 | 0.7793212012 | 0.7379969172 | 0.4188889911 | 0.1754679869 | 0.0282077546 | 6.190541 | 149 |

### Aggregate

| metric | value |
|---|---:|
| AUC mean | 0.7783213434 |
| AUC population stdev | 0.0006650428 |
| min AUC | 0.7774488561 |
| max AUC | 0.7793212012 |
| mean ACC | 0.7386486898 |
| mean RMSE | 0.4192017836 |
| mean Brier | 0.1757302240 |
| mean ECE | 0.0288909961 |
| peak CUDA GB observed | 6.190541 |

## Decision

- Record this as the current best `~7GB` pure-CDM training candidate.
- Do not promote it as default yet: it nearly matches experiment 104 and saves
  substantial memory, but it is still slightly below experiment 106 on mean AUC.
- Prefer this route over experiment 109 when memory is the priority. Experiment
  109 is useful as a full-batch low-cost dual-tower boundary, but experiment 110
  has better mean AUC and lower peak memory.
- If continuing, the next high-value probes are:
  - `batch=131072` smoke/full seed2026 to check whether even closer full-batch
    semantics stay near the 7GB target
  - `lr=4e-4/5e-4` around `batch=65536` if chasing AUC without losing the
    low-memory band
  - optional runner wrapper only after deciding whether experiment 110 should
    become a named low-memory runner
