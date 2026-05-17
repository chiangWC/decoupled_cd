# Experiment 86: contrastive readout expert

- 分支: `exp/contrastive-readout-expert`
- base: 当前 exp81 伪主线
- 代码提交:
  - `835d825` add optional `contrastive_readout_expert_adapter`
- 状态: rejected diagnostic, not a trial candidate

## Motivation

Experiment 83 showed that the experiment 51 full-trigger readout expert can suppress cleaner representation signals, especially around `q-local`. Experiment 84 tried output bounding, concept-count scoping, uncertainty scaling, post-expert replay, and deterministic prior rescues; those did not stabilize across seeds. Experiment 85 then moved target-local evidence into state / qrepr surfaces and also failed.

This experiment tested a materially different expert parameterization: keep the current interpretable expert gate, but center expert scores per sample before gate mixing. The intent was to remove common-mode additive residual capacity from the expert and force it to express only contrastive differences between experts:

```text
expert_scores := expert_scores - mean(expert_scores)
residual := sum(gate_probs * expert_scores)
```

The flag is default-off and requires `--interpretable-readout-expert-adapter`:

```bash
--contrastive-readout-expert-adapter
```

## Engineering validation

- Local:
  - `python3 -m py_compile models/decoupled_cdm.py scripts/train.py scripts/evaluate.py scripts/analyze_prediction_slices.py tests/test_decoupled_cdm.py`
  - `git diff --check`
- Local unittest was blocked because this Trellis worktree's `/usr/bin/python3` does not have `torch`; `scripts/enter_env.sh` is intentionally limited to `/home/jameschiang/work/decoupled_cd`.
- Remote tests:
  - `bash scripts/remote_exec.sh python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - Result: `33 tests`, `OK`
- Remote smoke:
  - `bash scripts/run_assist09_baseline.sh --contrastive-readout-expert-adapter --epochs 1 --max-rows 2000 --device cpu --output results/contrastive_readout_expert/assist_09_contrastive_expert_smoke_1ep.json`
  - Passed; summary recorded `contrastive_readout_expert_adapter=true`

## Results

Matched exp81 references:

| seed | AUC | ACC | RMSE | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 0.767478 | 0.734248 | 0.425562 | 0.181103 | 0.046972 |
| 2025 | 0.771661 | 0.732992 | 0.424611 | 0.180294 | 0.050071 |
| 2026 | 0.771263 | 0.732897 | 0.424666 | 0.180341 | 0.046958 |

Candidate result paths:

- `results/contrastive_readout_expert/assist_09_seed2024_contrastive_expert_300ep.json`
- `results/contrastive_readout_expert/assist_09_seed2025_contrastive_expert_300ep.json`
- `results/contrastive_readout_expert/assist_09_seed2026_contrastive_expert_300ep.json`

| seed | AUC | delta AUC | ACC delta | RMSE delta | Brier delta | ECE delta | verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 | 0.769478 | +0.002000 | -0.000552 | -0.000794 | -0.000675 | -0.001843 | positive first seed |
| 2025 | 0.770949 | -0.000712 | -0.001637 | +0.000278 | +0.000236 | -0.000354 | negative expansion |
| 2026 | 0.770742 | -0.000521 | -0.000267 | +0.000204 | +0.000174 | +0.002897 | negative expansion |

Three-seed mean relative to matched exp81:

| AUC delta | ACC delta | RMSE delta | Brier delta | ECE delta |
| ---: | ---: | ---: | ---: | ---: |
| +0.000256 | -0.000818 | -0.000104 | -0.000088 | +0.000233 |

## Decision

- Do not promote `contrastive_readout_expert_adapter`.
- Do not continue near-neighbor sweeps around per-sample centering / common-mode removal unless a stronger mechanism or slice diagnosis appears.
- The diagnostic is useful but weak: constraining expert common-mode capacity can create a clean seed2024 AUC/RMSE/Brier/ECE improvement, but it does not survive expansion and introduces ACC/ECE tradeoffs.
- This reinforces the broader experiment 83-85 conclusion: experiment 51 expert interaction remains a real design concern, but small changes to expert output form are not enough to move the current exp81 pseudo-mainline toward the `0.78` sprint target.
