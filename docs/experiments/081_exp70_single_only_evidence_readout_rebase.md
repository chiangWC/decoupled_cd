# Experiment 81: exp70 single-only concept evidence readout rebase

- 分支: `exp/exp70-evidence-readout-scope`
- base: `exp/trellis-trial` rollbacked experiment 70 pseudo-mainline `3869b6d`
- 对照: 当前 experiment 70 official three-seed baseline，见 [070_student_conditioned_ukc_readout_sidecar.md](./070_student_conditioned_ukc_readout_sidecar.md)

## 动机

实验 79 证明 `concept_evidence_readout_max_count=1` 在实验 78 底座上是稳定但偏弱的低幅信号；实验 80 又说明它和 `exact-3 target interaction` 组合后能形成更强候选。但实验 76/78 已整体从当前 `exp/trellis-trial` 默认口径回退，所以首先需要回答一个更基本的问题:

- 只把 trainable concept-evidence readout residual 限制在 `concept_count=1`
- 不再叠加 deterministic prior
- 直接回到 experiment 70 当前 trial 底座

这条结构本身能否在当前主线语义上形成 clean multi-seed gain。

## 实现

- 新增 `--concept-evidence-readout-max-count`
  - `0` 表示不设上限
  - `1` 表示 residual 只在 `concept_count=1` 时触发
- train/evaluate/slice loader/单测同步支持该字段
- 默认值仍是 `0`，因此不改变当前 baseline 行为

本轮候选配置:

```bash
bash scripts/run_assist09_baseline.sh \
  --concept-evidence-readout-residual \
  --concept-evidence-readout-min-count 1 \
  --concept-evidence-readout-max-count 1 \
  --concept-evidence-readout-min-seen-ratio 1.0 \
  --concept-evidence-readout-max-logit 0.5
```

## 工程验证

- 本地:
  - `python3 -m py_compile models/decoupled_cdm.py scripts/train.py scripts/evaluate.py scripts/analyze_prediction_slices.py tests/test_decoupled_cdm.py`
  - `git diff --check`
- 远端单测:
  - `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - 结果: `31 tests`, `OK`
- 远端 smoke:
  - `bash scripts/run_assist09_baseline.sh --epochs 1 --max-rows 2000 --device cpu ...`
  - summary 已记录 `concept_evidence_readout_max_count=1`

## 正式训练

exp70 baseline 不需要复跑，直接使用 experiment 70 official three-seed reference:

- `seed2024`: `AUC 0.765368`, `ACC 0.728558`, `RMSE 0.427880`, `Brier 0.183082`, `ECE 0.052010`
- `seed2025`: `AUC 0.766528`, `ACC 0.729605`, `RMSE 0.426968`, `Brier 0.182302`, `ECE 0.049967`
- `seed2026`: `AUC 0.764655`, `ACC 0.729148`, `RMSE 0.427201`, `Brier 0.182501`, `ECE 0.045154`

Candidate result paths:

- `results/assist09_exp70_single_only_readout_seed2024.json`
- `results/assist09_exp70_single_only_readout_seed2025.json`
- `results/assist09_exp70_single_only_readout_seed2026.json`

| seed | baseline AUC | candidate AUC | delta AUC | baseline ACC | candidate ACC | delta ACC | baseline RMSE | candidate RMSE | delta RMSE | baseline Brier | candidate Brier | delta Brier | baseline ECE | candidate ECE | delta ECE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 0.765368 | 0.767478 | +0.002110 | 0.728558 | 0.734248 | +0.005690 | 0.427880 | 0.425562 | -0.002318 | 0.183082 | 0.181103 | -0.001979 | 0.052010 | 0.046972 | -0.005038 |
| 2025 | 0.766528 | 0.771661 | +0.005133 | 0.729605 | 0.732992 | +0.003387 | 0.426968 | 0.424611 | -0.002357 | 0.182302 | 0.180294 | -0.002008 | 0.049967 | 0.050071 | +0.000104 |
| 2026 | 0.764655 | 0.771263 | +0.006608 | 0.729148 | 0.732897 | +0.003749 | 0.427201 | 0.424666 | -0.002535 | 0.182501 | 0.180341 | -0.002160 | 0.045154 | 0.046958 | +0.001804 |

Three-seed mean relative to experiment 70:

- `AUC +0.004617`
- `ACC +0.004275`
- `RMSE -0.002404`
- `Brier -0.002049`
- `ECE -0.001043`

这是当前 experiment 70 底座上第一条完成 multi-seed 验证后仍保持明显正向的可解释 evidence readout 候选。

## Slice 观察

Seed=2024 slice output:

- `results/assist09_exp70_single_only_readout_seed2024_slices.json`

与结构意图一致的关键切片:

- `concept_count=1`: `AUC 0.770828`, `ACC 0.737147`, `RMSE 0.423174`, `ECE 0.042957`
- `concept_count=2`: `AUC 0.751309`, `ACC 0.724460`, `RMSE 0.433814`, `ECE 0.067711`
- `none_seen`: `AUC 0.803505`, `ACC 0.811821`, `RMSE 0.364644`, `ECE 0.058400`
- `all_seen`: `AUC 0.765823`, `ACC 0.731651`, `RMSE 0.427479`, `ECE 0.047533`

这说明收益至少没有来自“误触发到多知识点题”的全局污染，更符合“把 targeted evidence readout 只收束到单知识点题”的原始假设。

## 结论

- 这条路线已经在当前 experiment 70 底座上形成 clear multi-seed signal，不再只是实验 79/80 那种建立在回退底座上的历史旁证。
- strongest evidence:
  - `seed2024` 单 seed 已过 `AUC +0.002` 扩线门槛
  - `2025/2026` 两颗补 seed 都继续放大 AUC 正向
  - three-seed mean 上 `AUC/ACC/RMSE/Brier/ECE` 全同向
- 当前判断:
  - 这条候选现已 promote 到 `exp/trellis-trial` 默认口径，作为 experiment 70 之后新的 pseudo-mainline follow-up
  - promote 理由是当前证据已经直接来自 exp70 底座上的 three-seed clean gain，而不是来自已回退的 exp76/78 底座
  - `B49 seed=2024` 交叉复验仍然值得补，但当前改作为 promote 后的 follow-up，而不是阻塞条件
