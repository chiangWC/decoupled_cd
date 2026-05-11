# Experiment 6: 超参数扫描

改动:

- 扫描范围:
  - `learning_rate in {1e-3, 3e-4, 1e-4}`
  - `concept_dim in {16, 32, 64}`
- `concept_dim = 128` 在当前 full-batch GPU 路径下稳定 OOM。

实验结论:

- 最优组合是 `learning_rate = 1e-3`, `concept_dim = 64`。
- 当时最好文件:
  - `results/hparam_sweeps/assist09_transition_lr_1e-3_dim_64.json`

结论:

- 之前一些“结构改动无效”的判断，部分受不合理超参数影响。
- 后续结构比较必须固定在更合理的超参数上进行。
