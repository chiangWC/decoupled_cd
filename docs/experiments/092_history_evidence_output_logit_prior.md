# Experiment 92: History Evidence Output Logit Prior

## Status

`diagnostic_signal_confirmed`, not promoted.

## Question

Can the experiment 91 train-history signal be expressed without a valid-trained hybrid stacker, using a fixed and interpretable CDM readout prior?

## Mechanism

Branch: `exp/evidence-prior-calibrated-readout`

New default-off model option:

- `--history-evidence-logit-prior-residual`
- `--history-evidence-logit-prior-location output`

The residual is deterministic and has no learned head. It adds a bounded output-logit prior from train-history evidence:

- student global ability log-odds delta
- exercise ease log-odds delta
- target student-concept mastery log-odds delta
- global concept ease log-odds delta
- target mastery confidence prior

It uses only train-history tensors and the existing valid/test history visibility contract. It is not a sklearn combiner and does not fit on validation labels.

## Corrected Reference

Current exp81 high-water reference:

- source: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`
- `test_auc = 0.7726816125195809`
- `test_acc = 0.7328207958286551`
- `test_rmse = 0.4243497503536669`
- `test_brier = 0.18007271062521943`
- `test_ece = 0.050954005791600435`

Stopping threshold: `0.7766816125195809`.

## Results

Primary fixed equal-weight result:

- output: `results/pure_cdm_hybrid_signal/seed2027_output_logit_prior_equal022_eval.json`
- base checkpoint: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`
- fixed weight for all five evidence terms: `0.22`
- max output prior logit: `4.0`
- component cap: `4.0`
- `test_auc = 0.776813484867231`
- AUC delta vs high-water: `+0.004131872347650112`
- `test_acc = 0.7325353479609508`
- `test_rmse = 0.42848995373217164`
- `test_brier = 0.1836036404493986`
- `test_ece = 0.0800283091196373`

Equal-weight sweep on seed2027 high-water checkpoint:

| weight | test_auc | delta vs 0.772682 | acc | rmse | brier | ece |
|---:|---:|---:|---:|---:|---:|---:|
| 0.16 | 0.776142 | +0.003461 | 0.732783 | 0.426930 | 0.182270 | 0.072301 |
| 0.18 | 0.776392 | +0.003710 | 0.733030 | 0.427423 | 0.182691 | 0.075139 |
| 0.20 | 0.776617 | +0.003935 | 0.732688 | 0.427944 | 0.183136 | 0.077384 |
| 0.22 | 0.776813 | +0.004132 | 0.732535 | 0.428490 | 0.183604 | 0.080028 |
| 0.24 | 0.776983 | +0.004302 | 0.732383 | 0.429059 | 0.184092 | 0.082386 |
| 0.25 | 0.777061 | +0.004379 | 0.732554 | 0.429352 | 0.184343 | 0.083540 |
| 0.28 | 0.777264 | +0.004583 | 0.732345 | 0.430258 | 0.185122 | 0.087059 |
| 0.30 | 0.777379 | +0.004697 | 0.732212 | 0.430884 | 0.185661 | 0.089479 |

More aggressive fixed non-equal probes also crossed the threshold, but with larger calibration damage:

- `raw75`: `test_auc = 0.777818474646415`, `ece = 0.11918894173935373`
- `localbest`: `test_auc = 0.7778148077921421`, `ece = 0.12850083693518574`
- `equal05`: `test_auc = 0.7777939039815148`, `ece = 0.11060232532748893`

## Interpretation

This satisfies the corrected high-water `+0.004` AUC signal using fixed train-history evidence inside the model readout. The cleanest over-threshold point is `equal022`, because it uses one shared coefficient and avoids valid-trained weighting.

It should not be promoted as a pure cognitive-logit CDM structure yet:

- Applying the same prior inside the cognitive logit strongly degraded AUC.
- Training from scratch with the fixed cognitive prior also degraded AUC.
- The successful version acts on the final output logit, so it is best described as an interpretable deterministic readout prior.
- RMSE/Brier/ECE regress materially versus the seed2027 high-water reference.

Next step: if continuing toward pure CDM, move this evidence into a calibrated objective or reliability-aware readout that preserves ranking without the observed calibration damage. Do not report it as a valid-trained hybrid stacker or as a promoted default run-script component.
