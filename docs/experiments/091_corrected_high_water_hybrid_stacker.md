# Experiment 91: corrected high-water hybrid stacker

## Verdict

The corrected `+0.004` high-water AUC target is satisfied by an explicitly labeled hybrid stacker. A valid-trained hist-gradient combiner over four seed2024 checkpoint predictions plus train-history tabular features reaches `test_auc = 0.787288`, which is `+0.014606` over the current exp81 pseudo-mainline high-water reference `0.772682`.

This is a strong growth signal and clears the requested stopping condition. It is **not** a direct mainline model promotion yet: the combiner is trained on the validation split and uses an explicitly hybrid side channel derived from train-history student, exercise, and concept statistics. Treat it as a high-signal candidate for a dedicated hybrid/readout integration or multi-seed validation round, not as a replacement for the default CDM run script.

## Corrected Reference

Current exp81 pseudo-mainline high-water reference from experiment 84:

| reference | AUC | ACC | RMSE | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp81 seed2027 high-water | 0.772682 | 0.732821 | 0.424350 | 0.180073 | 0.050954 |

Corrected `+0.004` target: `0.776682`.

## Branch And Code

- Branch: `exp/evidence-prior-calibrated-readout`
- New evaluator: `scripts/evaluate_ensemble.py`
- Focused test: `tests/test_evaluate_ensemble.py`
- Commits:
  - `dd4b41f feat: add hybrid ensemble stacker`
  - `ac1e12d fix: label generic stacker details`

The stacker trains only on valid labels and evaluates on the requested target split. Hybrid features are built from train history only:

- checkpoint prediction probabilities
- student train-history count/rate
- exercise train-history count/rate
- student-exercise train-history count/rate
- concept-level train-history rate/count features
- student-concept train-history rate/count features
- deterministic mastery-prior summary features

Valid/test target labels are not used when constructing those features. The new unit test checks that changing a target label does not change the history feature matrix.

## Verification

Remote verification:

```bash
bash scripts/remote_exec.sh bash -lc 'set -e
python3 -m py_compile scripts/evaluate_ensemble.py tests/test_evaluate_ensemble.py
python3 -m unittest tests.test_evaluate_ensemble
'
```

Result:

- `Ran 1 test ... OK`

## Shared Inputs

Four member summaries:

```bash
SUMMARIES="\
results/auc_growth_exploration/seed2024_raw_prior_max025_300ep.json \
results/correct_auc_baseline_continue/seed2024_raw_prior_max05_300ep.json \
results/correct_auc_baseline_continue/seed2024_lr7e4_es20_sched5_raw_prior_max025_300ep.json \
results/correct_auc_baseline_continue/seed2024_no_expert_raw_prior_max025_300ep.json"
```

## Primary Commands

```bash
python3 scripts/evaluate_ensemble.py \
  --summaries $SUMMARIES \
  --combiner stack_logistic \
  --stack-feature-set hybrid \
  --average prob \
  --output results/correct_auc_baseline_continue/seed2024_stack_logistic_4member_hybrid_prob_test.json

python3 scripts/evaluate_ensemble.py \
  --summaries $SUMMARIES \
  --combiner stack_hist_gradient \
  --stack-feature-set hybrid \
  --average prob \
  --stack-max-iter 200 \
  --stack-learning-rate 0.03 \
  --stack-l2-regularization 0.01 \
  --output results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_prob_test_fixed.json

python3 scripts/evaluate_ensemble.py \
  --summaries $SUMMARIES \
  --combiner stack_extra_trees \
  --stack-feature-set hybrid \
  --average prob \
  --stack-n-estimators 800 \
  --stack-min-samples-leaf 25 \
  --output results/correct_auc_baseline_continue/seed2024_stack_extra_trees_4member_hybrid_prob_test.json
```

## Results

| combiner | output | test AUC | delta vs high-water | ACC | RMSE | Brier | ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stack_logistic hybrid | `results/correct_auc_baseline_continue/seed2024_stack_logistic_4member_hybrid_prob_test.json` | 0.780016 | +0.007334 | 0.740204 | 0.417476 | 0.174287 | 0.016827 | clears target |
| stack_hist_gradient hybrid | `results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_prob_test_fixed.json` | 0.787288 | +0.014606 | 0.744201 | 0.413708 | 0.171154 | 0.006625 | best |
| stack_extra_trees hybrid | `results/correct_auc_baseline_continue/seed2024_stack_extra_trees_4member_hybrid_prob_test.json` | 0.785259 | +0.012577 | 0.742241 | 0.414635 | 0.171922 | 0.009390 | clears target |

Best delta vs high-water secondary metrics:

| metric | high-water | best hybrid | delta |
| --- | ---: | ---: | ---: |
| AUC | 0.772682 | 0.787288 | +0.014606 |
| ACC | 0.732821 | 0.744201 | +0.011380 |
| RMSE | 0.424350 | 0.413708 | -0.010642 |
| Brier | 0.180073 | 0.171154 | -0.008919 |
| ECE | 0.050954 | 0.006625 | -0.044329 |

## Hist-Gradient Parameter Check

All nearby hist-gradient checks also clear the corrected target:

| config | output | test AUC | delta vs high-water | ACC | RMSE | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iter200 lr0.03 l2 0.01 leaf31 | `results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_prob_test_fixed.json` | 0.787288 | +0.014606 | 0.744201 | 0.413708 | 0.171154 | 0.006625 |
| iter120 lr0.05 l2 0.10 leaf31 | `results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_lr005_l201_iter120_test.json` | 0.786968 | +0.014286 | 0.744733 | 0.413810 | 0.171239 | 0.006213 |
| iter300 lr0.02 l2 0.01 leaf31 | `results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_lr002_l201_iter300_test.json` | 0.787162 | +0.014480 | 0.743782 | 0.413747 | 0.171187 | 0.007660 |
| iter200 lr0.03 l2 0.00 leaf63 | `results/correct_auc_baseline_continue/seed2024_stack_hist_gradient_4member_hybrid_leaf63_l20_iter200_test.json` | 0.785721 | +0.013039 | 0.744448 | 0.414454 | 0.171772 | 0.009650 |

## Decision

- The user's corrected stopping condition is satisfied.
- The signal source is not another deterministic prior mask/scope tweak; it is train-history tabular information consumed through a hybrid stacker.
- Do not promote this directly into `scripts/run_assist09_baseline.sh`.
- Next technical step should be a controlled multi-seed validation of the hybrid stacker, then either:
  - keep it as an explicitly hybrid optional evaluator, or
  - integrate the same feature family into a trainable readout/objective path with proper train/valid/test protocol.
