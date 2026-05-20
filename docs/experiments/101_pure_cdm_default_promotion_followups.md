# Experiment 101: Pure CDM Default Promotion Follow-Ups

## Status

`pure_cdm_followups_rejected`

## Verdict

Follow-up probes after experiment 100 did not produce a stronger pure-CDM default-promotion candidate.

Experiment 100 remains the best local pure-CDM signal: `concept_dim=80 + output_alignment=0.004` reaches seed2027 `test_auc = 0.778122` and seed2026 `0.778100`, but its four-seed mean is only `0.777030` and seeds 2024/2025 regress versus experiment 95.

The new follow-ups tried confidence weighting, lower/higher cognitive-alignment weights, optimizer weight decay, training protocol changes, capacity interpolation, cog-only linear readout, and exercise-evidence difficulty initialization. None removed the seed2024/2025 tail. Do not promote dim80, output alignment, weight decay, linear readout, exercise difficulty initialization, or the tested training-protocol variants as the default CDM runner.

## Base

Branch: `exp/pure-cdm-default-promotion`

Reference:

- experiment 95 cog-only runner four-seed AUCs: `0.777843/0.776440/0.774669/0.776163`
- experiment 100 best four-seed candidate: dim80 + output alignment `w=0.004`, mean AUC `0.777030`
- experiment 100 best single seed: seed2027 dim80 + output alignment `w=0.004`, AUC `0.778122`

## Follow-Up Results

### Confidence-Weighted Output Alignment

These variants applied target-concept train-history confidence weighting to the output-alignment loss on seed2027. They improve some calibration/error metrics but do not beat the unweighted `w=0.004` AUC.

| variant | test_auc | ACC | RMSE | Brier | ECE | verdict |
|---|---:|---:|---:|---:|---:|---|
| unweighted output align `w=0.004` | 0.778122 | 0.735713 | 0.421772 | 0.177891 | 0.043800 | experiment 100 reference |
| power `0.5`, floor `0.0` | 0.777698 | 0.733144 | 0.422838 | 0.178792 | 0.047909 | below reference |
| power `0.5`, floor `0.2` | 0.777865 | 0.736341 | 0.421507 | 0.177668 | 0.042891 | lower AUC |
| power `1.0`, floor `0.0` | 0.777921 | 0.735561 | 0.421789 | 0.177906 | 0.045414 | lower AUC |
| power `1.0`, floor `0.2` | 0.777962 | 0.737635 | 0.420844 | 0.177110 | 0.042618 | best secondary metrics, lower AUC |

Decision: do not expand multi-seed; confidence weighting is an error/calibration tradeoff, not a `0.778` AUC improvement.

### Lower Cognitive Alignment Weights

Lowering dim80 cog-only alignment weight below `0.05` was negative on the weak seeds.

| seed | weight | test_auc | delta vs exp95 | ACC | RMSE | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.035 | 0.775755 | -0.002088 | 0.737788 | 0.421950 | 0.178041 | 0.045893 |
| 2024 | 0.040 | 0.776150 | -0.001693 | 0.738644 | 0.421680 | 0.177814 | 0.045475 |
| 2024 | 0.045 | 0.776197 | -0.001646 | 0.739462 | 0.421675 | 0.177810 | 0.045728 |
| 2025 | 0.035 | 0.775017 | -0.001424 | 0.736494 | 0.422699 | 0.178674 | 0.045245 |
| 2025 | 0.040 | 0.775242 | -0.001198 | 0.736170 | 0.422506 | 0.178512 | 0.044941 |
| 2025 | 0.045 | 0.775352 | -0.001089 | 0.736779 | 0.422207 | 0.178259 | 0.044129 |

Increasing to `0.060` on seed2024 also failed:

| seed | weight | test_auc | delta vs exp95 | delta vs dim80 base | ACC | RMSE | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.060 | 0.776327 | -0.001516 | -0.000138 | 0.739538 | 0.421438 | 0.177610 | 0.045081 |

Decision: dim80 instability is not fixed by nearby cognitive-alignment weight tuning.

### Adam Weight Decay

Added an opt-in `--weight-decay` training knob and ran a weak-seed sweep. All tested values were strongly negative.

| seed | weight_decay | test_auc | delta vs exp95 | delta vs dim80 base | ACC | RMSE | Brier | ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.00001 | 0.767025 | -0.010818 | -0.009440 | 0.732745 | 0.424829 | 0.180479 | 0.035275 |
| 2024 | 0.00003 | 0.754669 | -0.023174 | -0.021796 | 0.727207 | 0.428390 | 0.183518 | 0.019714 |
| 2024 | 0.00010 | 0.745308 | -0.032535 | -0.031157 | 0.722678 | 0.431052 | 0.185806 | 0.008120 |
| 2025 | 0.00001 | 0.764448 | -0.011993 | -0.010983 | 0.732973 | 0.425592 | 0.181128 | 0.039536 |
| 2025 | 0.00003 | 0.758540 | -0.017900 | -0.016891 | 0.728729 | 0.427344 | 0.182623 | 0.031238 |
| 2025 | 0.00010 | 0.751055 | -0.025386 | -0.024376 | 0.721441 | 0.431590 | 0.186270 | 0.039111 |

Decision: do not use Adam L2 weight decay for this runner. It regularizes away ranking capacity.

### Capacity Interpolation

The attempted dim76 interpolation was rejected immediately because seed2024 collapsed:

| variant | seed | test_auc | ACC | RMSE | Brier | ECE | best_epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| dim76 + output align `w=0.004` | 2024 | 0.503790 | 0.584388 | 0.500346 | 0.250347 | 0.120850 | 9 |

Decision: do not continue non-standard intermediate dimensions on this path. The useful dimension probes remain the stable multiples already recorded in experiment 100.

### Training Protocol

The historical mainline training-protocol candidates did not rescue seed2024 when applied to dim80 + output alignment `w=0.004`.

| variant | seed | test_auc | delta vs exp95 | delta vs dim80+out004 | ACC | RMSE | Brier | ECE | best_epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lr=7e-4`, early stop `20`, scheduler patience `5` | 2024 | 0.776193 | -0.001650 | -0.000226 | 0.738187 | 0.421394 | 0.177573 | 0.044033 | 227 |
| `lr=1.5e-3`, early stop `20`, scheduler patience `5` | 2024 | 0.776484 | -0.001359 | +0.000065 | 0.739424 | 0.421524 | 0.177682 | 0.046262 | 136 |

Decision: do not expand these protocol variants; the small seed2024 delta is insufficient and remains below experiment 95.

### Cog-Only Linear Readout

The cog-only linear history-evidence readout did not improve dim80 seed2024.

| variant | seed | test_auc | delta vs exp95 | delta vs dim80 base | ACC | RMSE | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dim80 cog-only linear readout, max logit `0.25` | 2024 | 0.776358 | -0.001485 | -0.000107 | 0.738225 | 0.421484 | 0.177649 | 0.045155 |

Decision: do not promote linear readout, and do not expand this exact setting.

### Exercise Evidence Difficulty Initialization

Initializing exercise difficulty from train-history exercise evidence is a model-side structural probe, but it strongly damaged the dim80 output-alignment candidate on seed2024.

| variant | seed | test_auc | delta vs exp95 | delta vs dim80+out004 | ACC | RMSE | Brier | ECE | initialized exercises |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dim80 + output align `w=0.004` + difficulty init max_abs_logit `0.10` | 2024 | 0.762815 | -0.015028 | -0.013604 | 0.730708 | 0.432508 | 0.187063 | 0.063495 | 17169 |

Decision: do not use exercise-evidence difficulty initialization as a rescue for this default-promotion path. It over-injects exercise-level train-history signal and worsens both ranking and calibration.

## Decision

- Keep experiment 95 as the active pure-CDM trial runner.
- Keep experiment 100's dim80 + output-alignment runner as an opt-in local-signal probe only.
- Do not change `scripts/run_assist09_baseline.sh`.
- Do not continue nearby sweeps over output-alignment confidence weighting, cognitive-alignment weight, Adam weight decay, dim76/dim78 interpolation, exercise difficulty initialization, or the tested learning-rate/patience variants.
- Future pure-CDM default-promotion work needs a structural mechanism that specifically fixes seed2024/2025 without sacrificing the seed2026/2027 headroom, not more local weight/protocol tuning around dim80.
