# Experiment 75: history-conditioned Q representation

- 分支: `exp/history-conditioned-q-repr`
- 提交: `9fa8368`
- 代码状态: 探索性实现已保留在实验分支；不合入当前 `master` 主线
- 结果路径:
  - 正式单次: `results/history_conditioned_q_repr/assist_09_seed2024_300ep.json`
  - slice: `results/history_conditioned_q_repr/assist_09_seed2024_300ep_slices.json`
  - 主线参考 slice: `results/history_conditioned_q_repr/mainline_seed2024_reference_slices.json`

## 动机

- 当前冲刺目标是 `test_auc ~= 0.780`，常规 final-logit residual / sidecar 的边际收益已经不足。
- 实验 72 的 `target-conditioned student context` 直接写回学生状态后明显伤排序；本轮改为更受约束的表示级改动:
  - 不改 `student_state` 主状态
  - 在 `q_repr` 生成后、主 cognitive readout 前加入一个零初始化 residual
  - residual 只读取目标题 Q 概念上的学生历史统计，默认 `concept_count >= 2` 触发

## 做法

- 新增 `--history-conditioned-q-representation-adapter`
- 新增 `--history-conditioned-q-representation-min-count`，默认 `2`
- 模块位置:
  - `q_repr = q_repr + history_conditioned_q_repr_residual(...)`
  - 后续 cognitive match、high-concept adapter、interpretable readout expert、UKC sidecar、conditional g/s 都读取更新后的 `q_repr`
- 输入信号:
  - 目标题 Q 概念 embedding
  - 目标题 exercise embedding
  - 学生在目标 Q 概念上的历史 `accuracy / seen / log_attempts`
  - `concept_count / coverage / difficulty / concept dispersion`
- 训练时显式从历史统计中扣掉当前 target exercise，避免这个新表示分支直接读到当前标签。
- residual 末层零初始化；显式历史统计和概念统计分支不把梯度打回主概念 embedding，避免 `sqrt(0)` dispersion 反传产生 NaN。

## 工程验证

- 远端单测:
  - `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - 结果: 通过，`22 tests`
- 远端 smoke:
  - `OUTPUT=results/history_conditioned_q_repr/smoke_seed2024_1ep_2000.json bash scripts/run_assist09_history_conditioned_q_repr.sh --epochs 1 --max-rows 2000 --device cpu`
  - 结果: 通过

## 结果

对比基线使用同一代码口径下的当前主线 `results/assist_09_mainline_300ep.json`，`seed=2024`。

| metric | mainline | exp75 | delta |
| --- | ---: | ---: | ---: |
| test_auc | 0.765444 | 0.763824 | -0.001620 |
| test_acc | 0.730404 | 0.730138 | -0.000266 |
| test_rmse | 0.427481 | 0.427965 | +0.000484 |
| test_brier | 0.182740 | 0.183154 | +0.000414 |
| test_ece | 0.051686 | 0.049865 | -0.001822 |

valid 同步回撤:

| metric | delta |
| --- | ---: |
| valid_auc | -0.001920 |
| valid_acc | -0.001128 |
| valid_rmse | +0.000632 |
| valid_brier | +0.000539 |
| valid_ece | -0.002267 |

## Slice

相对当前主线参考 slice:

- `concept_count=2`: `AUC +0.000766`, `RMSE -0.001957`, `ECE -0.009501`
- `concept_count=4+`: `AUC +0.000397`, `ACC +0.002755`, `RMSE -0.000310`, `ECE -0.011914`
- `concept_count=1`: `AUC -0.002175`, `RMSE +0.000979`
- `none_seen`: `AUC -0.004392`, `ACC -0.001206`, `ECE +0.001603`

## 结论

- 不扩 seed，不合入主线。
- 这个 representation-level `q_repr` 改动确实让多知识点 slice 的误差/校准略好，`4+` 也有极小 AUC 正向，但量级远低于冲 `0.78` 所需的单 seed `AUC +0.002` 准入信号。
- 整体 AUC/ACC/RMSE/Brier 均回撤，且即使 adapter 默认只触发 `concept_count>=2`，训练后的共享表示仍拖累 `concept_count=1` 与 `none_seen` 排序。
- 若以后复访这条思想，优先不要再做共享 `q_repr` 写回；更合理的方向是把历史条件化限制在多知识点专用、与单知识点和 none_seen 解耦的局部专家，或直接转向更强的学生状态形成机制。
