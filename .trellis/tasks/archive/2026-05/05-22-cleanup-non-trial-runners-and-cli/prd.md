# cleanup non-trial runners and cli

## Goal

Clean the repository's ASSIST09 experiment surface so users can run the accepted baseline and the current strongest exp104 trial without wading through rejected historical runner scripts or CLI flags.

## What I already know

* User asked to remove scripts and CLI for experiments not merged into the current trial, then run the strongest exp104 configuration.
* `scripts/run_assist09_baseline.sh` is the official exp81 baseline runner.
* `scripts/run_assist09_history_alignment_trial.sh` currently encodes an older exp95-style base and needs extra flags to reproduce exp104.
* Experiment 104 is the current strongest single-run/single-checkpoint pure-CDM default-training candidate.
* Exp104 requires dual CDM ensemble, branch BCE, late-window cognitive alignment annealing, and train-only concept evidence prior flags.
* Historical runner wrappers include concept evidence probes, evidence behavior gate, reliability/fusion/linear/output/rank alignment probes, and student-conditioned UKC wrapper.

## Assumptions

* Keep general utility scripts such as preprocessing, graph building, remote execution, evaluation, and multiseed helpers unless they directly encode rejected trial variants.
* Keep model internals and tests for now unless needed to remove CLI wiring safely; the user specifically asked for script and CLI cleanup.
* Promote exp104 into `scripts/run_assist09_history_alignment_trial.sh` rather than adding another runner name.

## Requirements

* Remove obsolete `scripts/run_assist09_*.sh` wrappers that only represent rejected or superseded experiments.
* Preserve `scripts/run_assist09_baseline.sh`.
* Make `scripts/run_assist09_history_alignment_trial.sh` run exp104 by default.
* Keep CLI arguments required by baseline and exp104.
* Remove CLI arguments and training-summary fields only tied to non-trial experiment routes when doing so does not break exp104/baseline execution.
* Update tests or docs references affected by removed runner scripts/CLI flags.
* Run static checks and focused tests.
* Run exp104 remotely on GPU after cleanup.

## Acceptance Criteria

* `bash scripts/run_assist09_baseline.sh --epochs 1 --max-rows 2000 --device cpu --output ...` still runs.
* `bash scripts/run_assist09_history_alignment_trial.sh --epochs 1 --max-rows 2000 --device cpu --output ...` uses exp104 flags and runs.
* Full relevant unittest set passes or known pre-existing test isolation issue is clearly documented.
* Full exp104 seed2024 remote run completes on GPU and writes a result JSON.

## Out of Scope

* Removing model implementation for old experimental mechanisms unless required by CLI cleanup.
* Rewriting experiment history docs.
* Promoting checkpoint-average or hybrid stacker routes.

## Technical Notes

* Exp104 command is documented in `docs/experiments/104_single_checkpoint_dual_cdm_ensemble.md`.
* CLI implementation is concentrated in `scripts/train.py` and training objective plumbing in `trainers/engine.py`.
