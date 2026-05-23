# Experiment 109: Low-cost dual-tower boundary

## Status

`low_cost_dual_tower_signal_confirmed_not_7gb`

## Verdict

Reducing the experiment 106 dual-tower secondary dimension from `80` to `32`
keeps a stable pure-CDM signal while lowering peak CUDA memory from about
`11.76GB` to about `8.85GB`.

Best low-cost candidate found in this round:

```bash
bash scripts/run_assist09_history_alignment_trial.sh \
  --dual-cdm-secondary-concept-dim 32 \
  --dual-cdm-branch-bce-weight 0.18
```

Four-seed AUC is `0.779241/0.776995/0.777735/0.776727`, mean
`0.777674`, population stdev `0.000977`, and mean ECE `0.032693`.

This is a real low-cost pure-CDM signal because every checked seed improves
over the matched experiment 105 `single64` lightweight reference, and memory is
about `2.9GB` lower than experiment 106. It is not a replacement for experiment
104/106 on AUC: the mean remains below exp104 `0.778370` and exp106 `0.778579`.
It also does not satisfy a strict `~7GB` target; the closest non-collapsing
boundary tested here is `64x16`, which uses about `7.89GB` but does not preserve
stable AUC across checked anchors.

The teacher-distillation route was also tested and rejected as the primary
answer. It stayed near `6.33GB`, but the best seed2026 gain was only
`+0.000140` AUC and worsened secondary metrics.

No hybrid stacker, validation-trained combiner, fixed checkpoint average,
test-label fitting, or train-history tabular inference side channel was used.

## Base

- Branch: `exp/pure-cdm-distillation`
- Low-memory reference: experiment 105 `single64`
  - AUCs: `0.778379/0.776930/0.775065/0.776569`
  - mean AUC: `0.776736`
  - peak CUDA: about `6.33GB`
- Heavy comparison: experiment 106 `dual64x80`, branch BCE `0.18`
  - AUCs: `0.778890/0.778552/0.778256/0.778618`
  - mean AUC: `0.778579`
  - peak CUDA: about `11.76GB`

## Engineering

- Added `scripts/run_assist09_single64_distillation_trial.sh` for pure-CDM
  teacher-distillation probes.
- Exposed `checkpoint_distillation_*` trainer controls in `scripts/train.py`.
- Added train-split teacher target generation from a pure-CDM teacher summary.
- Added `max_cuda_memory_allocated_gb` reporting to `scripts/train.py`.
- Fixed `scripts/analyze_prediction_slices.predict_bundle` to pass
  `exercise_evidence` into model forward.

Remote verification:

```bash
python -m py_compile scripts/train.py scripts/analyze_prediction_slices.py tests/test_train_cli.py
python -m unittest tests.test_train_cli tests.test_training_modes
```

Both passed remotely.

## Distillation Probes

Teacher: exp106 `seed2026` pure-CDM dual64x80 branch-BCE `0.18` summary.

Reference: experiment 105 `single64` seed2026 AUC `0.775065`.

| variant | seed | AUC | delta vs single64 | RMSE | Brier | ECE | peak CUDA GB | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BCE distill `w=0.03` | 2026 | 0.774794 | -0.000271 | 0.422439 | 0.178455 | 0.040968 | 6.330964 | rejected |
| BCE distill `w=0.05` | 2026 | 0.774726 | -0.000339 | 0.422952 | 0.178889 | 0.043344 | 6.329971 | rejected |
| BCE distill `w=0.10` | 2026 | 0.775115 | +0.000050 | 0.421684 | 0.177818 | 0.039047 | 6.330964 | noise-level |
| BCE distill `w=0.20` | 2026 | 0.774639 | -0.000426 | 0.422695 | 0.178671 | 0.041718 | 6.330964 | rejected |
| BCE distill `w=0.10`, start epoch 135 | 2026 | 0.775073 | +0.000008 | 0.421752 | 0.177875 | 0.038139 | 6.330964 | noise-level |
| standardized logit MSE `w=0.10` | 2026 | 0.775205 | +0.000140 | 0.422251 | 0.178296 | 0.042141 | 6.330964 | too small |

Decision: teacher distillation did not form a clear low-memory AUC signal. Do
not expand this family without a new distillation mechanism, such as selective
slice distillation or a stronger purity-preserving target.

## Dual-Tower Boundary Probes

All rows use `dual_cdm_branch_bce_weight=0.18` and the experiment 104/106 late
alignment plus train-only concept-prior protocol.

| secondary dim | seed | AUC | delta vs single64 | ACC | RMSE | Brier | ECE | peak CUDA GB | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 2026 | 0.768610 | -0.006455 | 0.732859 | 0.423190 | 0.179090 | 0.024624 | 7.394128 | collapsed ranking |
| 16 | 2026 | 0.776500 | +0.001435 | 0.737369 | 0.419784 | 0.176219 | 0.024779 | 7.887326 | weak-seed signal |
| 16 | 2024 | 0.776929 | -0.001450 | 0.738187 | 0.419470 | 0.175955 | 0.025316 | 7.887326 | strong-anchor AUC regression |
| 24 | 2026 | 0.777029 | +0.001964 | 0.735866 | 0.420356 | 0.176699 | 0.031137 | 8.363357 | boundary signal |
| 24 | 2024 | 0.777802 | -0.000577 | 0.736703 | 0.420082 | 0.176469 | 0.033120 | 8.363357 | still hurts strong anchor |
| 32 | 2024 | 0.779241 | +0.000862 | 0.738777 | 0.419564 | 0.176034 | 0.035674 | 8.852084 | candidate |
| 32 | 2025 | 0.776995 | +0.000065 | 0.737198 | 0.419846 | 0.176271 | 0.029630 | 8.849196 | candidate |
| 32 | 2026 | 0.777735 | +0.002670 | 0.737845 | 0.420219 | 0.176584 | 0.035567 | 8.852084 | candidate |
| 32 | 2027 | 0.776727 | +0.000158 | 0.738549 | 0.419633 | 0.176092 | 0.029903 | 8.849196 | candidate |

### `64x32` Aggregate

| metric | value |
|---|---:|
| AUC mean | 0.7776744230 |
| AUC population stdev | 0.0009768782 |
| min AUC | 0.7767269292 |
| max AUC | 0.7792408013 |
| mean ACC | 0.7380920665 |
| mean RMSE | 0.4198155127 |
| mean Brier | 0.1762451298 |
| mean ECE | 0.0326934310 |
| peak CUDA GB observed | 8.852084 |
| delta vs single64 mean | +0.000938 |
| delta vs exp104 mean | -0.000695 |
| delta vs exp106 mean | -0.000905 |

## Decision

- Keep experiment 106 `dual64x80 branchBCE=0.18` as the stronger heavy pure-CDM
  candidate.
- Record `dual64x32 branchBCE=0.18` as a stable lower-memory candidate: it is
  below exp104/106 AUC, but it gives four-seed positive deltas over the
  `single64` lightweight reference while saving about `2.9GB` CUDA peak memory
  versus exp106.
- Do not use `dual64x8`; it collapses ranking despite lower memory.
- Do not promote `dual64x16` as an AUC route; it is closer to `~7GB` and has
  good ECE, but seed2024 regresses enough that two-anchor mean is flat versus
  single64.
- Treat `dual64x24` as a boundary ablation, not a four-seed candidate unless a
  future task prioritizes memory over AUC.
