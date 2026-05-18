# Pure CDM Default Promotion

## Goal

Continue from `exp/trellis-trial` toward a result that can plausibly become a pure CDM runner/default CDM promotion. The experiment 99 hybrid stacker is useful diagnostic evidence, but it does not satisfy the target because it uses a validation-trained combiner and train-history tabular side-channel outside the default CDM model path.

## What I Already Know

- User rejected treating the hybrid stacker as acceptable progress: the route must move toward pure CDM runner/default CDM promotion.
- Current practical AUC target remains `test_auc >= 0.778`; `0.780` is desirable headroom.
- Experiment 95 is the active pure-CDM candidate: `scripts/run_assist09_history_alignment_trial.sh`, `loss_only cogonly`, four-seed mean `test_auc = 0.776279`.
- Experiment 98 rejected reliability weighting, cog-only fusion readout, and rank alignment as promotions; reliability had a seed2026 collapse.
- Experiment 99 shows train-history features contain strong signal, but the implementation path must not be a valid-trained stacker or output-only hybrid evaluator.
- Experiment 100 produced the best pure-CDM local signal so far: dim80 + weak output alignment `w=0.004` reaches seed2027 `test_auc = 0.778122` and seed2026 `0.778100`, but four-seed mean is only `0.777030` and seeds 2024/2025 regress, so it is not a default promotion.
- Experiment 101 rejected follow-ups around experiment 100: confidence weighting, nearby cognitive-alignment weights, Adam weight decay, dim76 capacity interpolation, lr/patience rescue, cog-only linear readout, and exercise difficulty initialization did not fix the seed2024/2025 tail.
- Experiment 102 found a pure checkpoint-average path: prediction-only probability average of experiment 95 and experiment 100 member checkpoints reaches four-seed AUC `0.779011/0.778059/0.778969/0.779448`, mean `0.778872`, without a validation-trained combiner or hybrid tabular side-channel.
- Remote execution must run only through `bash scripts/remote_exec.sh` after local commits are pushed.

## Assumptions

- "Pure CDM runner/default CDM promotion" means the final candidate must be reproducible through a training/run script and must not fit a combiner on validation labels.
- Prediction-only checkpoint averaging may be considered a pure runner/evaluator candidate when every member is a pure-CDM checkpoint and weights are fixed, not learned from validation labels.
- Train-history evidence may be used only as deterministic training target, model-side cognitive/readout structure, or initialization/regularization using train history; it must preserve valid/test history visibility.
- The current useful step is to harden and document the checkpoint-average evaluator, while keeping the single-checkpoint default-training question separate.

## Requirements

1. Work on a descendant of `exp/trellis-trial`.
2. Keep hybrid stacker results explicitly out of the acceptance path.
3. Preserve the current experiment 95 runner as a baseline/reference unless evidence supports a default change.
4. Add or restore pure-CDM-only code paths needed for the next candidate.
5. Include focused unit tests for any new flags, target construction, or validation behavior.
6. Commit and push before remote test/training execution.
7. Run remote verification and at least one meaningful pure-CDM experiment probe.
8. Update experiment ledger with clear promotion/rejection criteria and result paths.

## Acceptance Criteria

- [x] A pure-CDM runner candidate is implemented or restored on this branch.
- [x] Focused tests pass remotely.
- [x] At least one full or diagnostic remote run evaluates a pure-CDM candidate against experiment 95/high-water.
- [x] The result is documented without using hybrid stacker metrics as acceptance evidence.
- [x] If a candidate looks promotable, docs say exactly what default runner changes are justified; if not, docs say what was rejected and why.

## Out Of Scope

- Do not promote `scripts/evaluate_ensemble.py` or any valid-trained stacker.
- Do not report experiment 99 as default CDM performance.
- Do not use validation/test labels to train an inference-time combiner.
- Do not reintroduce direct student/exercise history shortcut terms unless a new controlled experiment proves they are clean.
- Do not conflate the experiment 102 two-checkpoint inference runner with a single-checkpoint `scripts/run_assist09_baseline.sh` default-training promotion.

## Technical Notes

- Specs read: backend experiment protocol, quality, directory, data artifact, error handling, logging, and shared guides.
- Key parent docs:
  - `docs/experiments/095_history_alignment_cf_risk_ablation.md`
  - `docs/experiments/098_pure_cdm_runner_integration.md`
  - `docs/experiments/099_hybrid_stacker_multiseed_078.md`
  - `docs/experiments/102_pure_cdm_checkpoint_average_runner.md`
- Likely code areas:
  - `scripts/evaluate_checkpoint_average.py`
  - `scripts/train.py`
  - `scripts/run_assist09_history_alignment_trial.sh`
  - `models/decoupled_cdm.py`
  - `trainers/engine.py`
  - `tests/`
