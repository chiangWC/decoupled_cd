# Reuse Plan

基于 [README_spec.md](/home/xph/jwc/research/decoupled_cd/README_spec.md) 与参考项目 [../ConceptSkillCDM](/home/xph/jwc/research/ConceptSkillCDM/README.md) 的结构分析，这里只判断“哪些工程部分适合迁移到新项目”，不涉及直接拷贝核心模型结构。

## 当前状态

本项目已经完成的最小工程复用包括:

- 已从参考项目工程思路中吸收并落地:
  - 指标计算与评估导出
  - `train/valid/test` 分离评估
  - `best val AUC` 选模
  - 训练历史 CSV 落盘
  - 实验 summary CSV 追加
  - GPU 设备选择工具
  - best checkpoint 自动保存
  - learning-rate scheduler 接入
- 已实现的主线文件:
  - [scripts/train.py](/home/xph/jwc/research/decoupled_cd/scripts/train.py)
  - [trainers/engine.py](/home/xph/jwc/research/decoupled_cd/trainers/engine.py)
  - [utils/metrics.py](/home/xph/jwc/research/decoupled_cd/utils/metrics.py)
  - [utils/io.py](/home/xph/jwc/research/decoupled_cd/utils/io.py)
  - [utils/device.py](/home/xph/jwc/research/decoupled_cd/utils/device.py)

## 目标约束

- 新项目当前目标:
  - 完善工程层复用
  - 保持 Step 1-4 全流程可运行
  - 在此基础上逐步优化模型效果
- Step 2 已明确:
  - `TKC` 可以接收习题信息、行为信号和知识邻接传播
  - `UKC` 只能接收知识结构传播，不能混入习题和行为信号
- 这些符号的正式定义以 [README_spec.md](/home/xph/jwc/research/decoupled_cd/README_spec.md) 为准:
  - `E_u`: 学生 `u` 有作答记录的习题集合
  - `TKC_u`: 学生 `u` 已测试知识点集合
  - `UKC_u`: `K \\ TKC_u`
- 参考项目只参考数据处理、训练框架、评估代码。
- 明确不复用参考项目核心模型结构。
- 当前项目已有最小数据骨架:
  - [data/readers.py](/home/xph/jwc/research/decoupled_cd/data/readers.py)
  - [data/mappings.py](/home/xph/jwc/research/decoupled_cd/data/mappings.py)
  - [data/pipeline.py](/home/xph/jwc/research/decoupled_cd/data/pipeline.py)

结论先行: 新项目应该复用参考项目的“工程壳层”，而不是复用它的“模型表达方式”。你自己的 `data/` 已经比参考项目更贴近当前任务，因此不建议把参考项目的数据模块整文件搬进来。

## 已经完成的复用

- 已改造复用 [../ConceptSkillCDM/src/experiment_utils.py](/home/xph/jwc/research/ConceptSkillCDM/src/experiment_utils.py) 的思路:
  - 在 [utils/metrics.py](/home/xph/jwc/research/decoupled_cd/utils/metrics.py) 中实现 `compute_metrics`
  - 在 [utils/io.py](/home/xph/jwc/research/decoupled_cd/utils/io.py) 中实现训练历史与 summary CSV 落盘

- 已改造复用 [../ConceptSkillCDM/src/trainer.py](/home/xph/jwc/research/ConceptSkillCDM/src/trainer.py) 的思路:
  - 在 [trainers/engine.py](/home/xph/jwc/research/decoupled_cd/trainers/engine.py) 中实现 `train / validate / test` 评估骨架
  - 已具备 `best val AUC` 选模、best checkpoint 保存、early-stop 和 scheduler 结构

- 已改造复用 [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py) 的思路:
  - 在 [utils/device.py](/home/xph/jwc/research/decoupled_cd/utils/device.py) 中实现 GPU 候选解析、显存查询与自动设备选择

## 可直接复用的文件/函数

- 文件: [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py)
  - 函数: `parse_int_csv`
  - 理由: 纯命令行整数列表解析，与模型无关。

- 文件: [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py)
  - 函数: `parse_gpu_ids`
  - 理由: 纯 GPU 参数解析，可直接保留。

- 文件: [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py)
  - 函数: `calc_effective_max_concurrent`
  - 理由: 只处理并发上限计算，属于通用实验调度逻辑。

- 文件: [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py)
  - 函数: `pick_gpus_for_job`
  - 理由: 资源调度函数，不依赖模型语义。

- 文件: [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py)
  - 函数: `configure_main_process_gpus`
  - 理由: 设置 `CUDA_VISIBLE_DEVICES` 的逻辑可直接复用。

- 文件: [../ConceptSkillCDM/src/experiment_utils.py](/home/xph/jwc/research/ConceptSkillCDM/src/experiment_utils.py)
  - 函数: `setup_logging`
  - 理由: 日志初始化完全是工程基础设施。

- 文件: [../ConceptSkillCDM/src/experiment_utils.py](/home/xph/jwc/research/ConceptSkillCDM/src/experiment_utils.py)
  - 函数: `compute_metrics`
  - 理由: AUC / ACC / RMSE 的计算与模型结构无关。

- 文件: [../ConceptSkillCDM/src/experiment_utils.py](/home/xph/jwc/research/ConceptSkillCDM/src/experiment_utils.py)
  - 函数: `save_epoch_history_csv`
  - 理由: 训练历史按 epoch 落盘的方式可直接服务新项目。

- 文件: [../ConceptSkillCDM/run_ablation.py](/home/xph/jwc/research/ConceptSkillCDM/run_ablation.py)
  - 函数: `parse_csv_tokens`
  - 理由: 纯字符串解析工具，可复用到后续实验脚本。

- 文件: [../ConceptSkillCDM/run_ablation.py](/home/xph/jwc/research/ConceptSkillCDM/run_ablation.py)
  - 函数: `append_arg`
  - 理由: 命令构造辅助函数，和具体模型无关。

- 文件: [../ConceptSkillCDM/run_ablation.py](/home/xph/jwc/research/ConceptSkillCDM/run_ablation.py)
  - 函数: `_write_csv`
  - 理由: 小型实验汇总写盘工具，可直接借用。

## 需要改造后复用的部分

- 文件: [../ConceptSkillCDM/src/config.py](/home/xph/jwc/research/ConceptSkillCDM/src/config.py)
  - 部分: `DATASET_DEFAULTS`, `apply_dataset_defaults`
  - 理由: “按数据集提供默认配置”的机制值得保留，但参数项必须改成面向 `E_u / TKC_u / UKC_u`，删掉 graph / hyperlens / residual / module ablation 参数。

- 文件: [../ConceptSkillCDM/main.py](/home/xph/jwc/research/ConceptSkillCDM/main.py)
  - 部分: 入口脚本组织方式
  - 理由: `parse_args -> 准备数据 -> 构建模型 -> 训练/评估 -> 保存结果` 的流程可保留，但参数定义、模型初始化、分析输出需要完全重写。

- 文件: [../ConceptSkillCDM/src/dataset.py](/home/xph/jwc/research/ConceptSkillCDM/src/dataset.py)
  - 部分: `build_id_mappings`, `build_q_matrix`, `create_dataloaders`, `CognitiveDiagnosisDataset`
  - 理由: 其中的清洗、Q 矩阵构造、DataLoader 思路可参考，但不建议整文件迁移。
  - 补充: 你当前 [data/mappings.py](/home/xph/jwc/research/decoupled_cd/data/mappings.py) 和 [data/pipeline.py](/home/xph/jwc/research/decoupled_cd/data/pipeline.py) 已覆盖最核心的数据入口，应以你自己的模块为主，再按需吸收参考项目里的过滤和 loader 组织。

- 文件: [../ConceptSkillCDM/src/trainer.py](/home/xph/jwc/research/ConceptSkillCDM/src/trainer.py)
  - 部分: `train_epoch`, `validate`, `train_one_experiment`, `run_inference`
  - 理由: 这些函数的“训练循环骨架”可参考，但当前实现和 `src/model.py` 强耦合，只能保留流程层，不能直接搬代码。

- 文件: [../ConceptSkillCDM/src/experiment_utils.py](/home/xph/jwc/research/ConceptSkillCDM/src/experiment_utils.py)
  - 部分: `append_summary_csv`
  - 理由: 统一汇总实验结果到 CSV 的机制有价值，但列字段现在绑定旧项目的 `module1 / module2 / module3` 运行时状态，必须删减重构。

- 文件: [../ConceptSkillCDM/gpu_utils.py](/home/xph/jwc/research/ConceptSkillCDM/gpu_utils.py)
  - 部分: `get_best_gpu`, `get_best_gpus`, `pick_gpu_with_slot_round_robin`
  - 理由: 可以保留做批量实验资源调度，但这些函数依赖 `nvidia-smi` 的运行环境，建议封装到新项目 `utils/device.py` 或 `scripts/launcher.py` 中，而不是原样散落。

- 文件: [../ConceptSkillCDM/run_all_datasets.py](/home/xph/jwc/research/ConceptSkillCDM/run_all_datasets.py)
  - 部分: 多数据集批量启动框架
  - 理由: 将来做 benchmark 时可复用其调度思想，但当前 Step 1 先不需要多 GPU 批量实验，建议后置。

- 文件: [../ConceptSkillCDM/src/analysis.py](/home/xph/jwc/research/ConceptSkillCDM/src/analysis.py)
  - 部分: 分析模块的文件位置和“训练后导出分析图”的职责划分
  - 理由: 组织方式可保留，但内容应改为导出 `E_u / TKC_u / UKC_u` 的统计、覆盖率、样例检查图，而不是旧模型图结构热力图。

## 必须重写的部分

- 文件: [../ConceptSkillCDM/src/model.py](/home/xph/jwc/research/ConceptSkillCDM/src/model.py)
  - 部分: 全部
  - 理由: 这是参考项目核心模型结构，明确禁止复用。
  - 补充: 新项目 Step 2 明确要求 `TKC` 与 `UKC` 分离传播，而参考项目没有这一结构边界。

- 文件: [../ConceptSkillCDM/src/personalization.py](/home/xph/jwc/research/ConceptSkillCDM/src/personalization.py)
  - 部分: 全部
  - 理由: `HyperLensConceptAdapter` 是旧模型的局部证据适配器，属于核心建模逻辑，不适合迁入新项目。

- 文件: [../ConceptSkillCDM/src/module_activity.py](/home/xph/jwc/research/ConceptSkillCDM/src/module_activity.py)
  - 部分: 全部
  - 理由: 该文件的诊断指标完全围绕旧模型的 graph / adapter / residual 模块定义。

- 文件: [../ConceptSkillCDM/src/trainer.py](/home/xph/jwc/research/ConceptSkillCDM/src/trainer.py)
  - 部分: 所有与 `CognitiveDiagnosisModel` 内部结构强耦合的逻辑
  - 理由: 包括模块开关校验、图熵诊断、残差分支诊断、旧 checkpoint 兼容，都不应延续到新模型。

- 文件: [../ConceptSkillCDM/best_configs.py](/home/xph/jwc/research/ConceptSkillCDM/best_configs.py)
  - 部分: 全部
  - 理由: 超参数完全服务旧模型，不应成为新项目配置来源。

- 文件: [../ConceptSkillCDM/run_ablation.py](/home/xph/jwc/research/ConceptSkillCDM/run_ablation.py)
  - 部分: 变体定义、诊断提取、结果解释
  - 理由: 这些实验脚本围绕旧模块消融语义，新项目当前阶段没有对应对象。

- 文件: [../ConceptSkillCDM/src/analysis.py](/home/xph/jwc/research/ConceptSkillCDM/src/analysis.py)
  - 部分: `run_structure_heatmap`, `run_hyperlens_analysis`
  - 理由: 可视化对象属于旧模型内部状态，不是新项目 Step 1 需要分析的内容。

- 目录: [../ConceptSkillCDM/docs](/home/xph/jwc/research/ConceptSkillCDM/docs)
  - 部分: 全部
  - 理由: 这些文档在讲旧模型故事线，只能当写作参考，不能作为新设计依据。

- 目录: [../ConceptSkillCDM/results](/home/xph/jwc/research/ConceptSkillCDM/results)
  - 部分: 全部
  - 理由: 历史结果不应作为新项目结构设计的输入。

- 目录: [../ConceptSkillCDM/logs](/home/xph/jwc/research/ConceptSkillCDM/logs)
  - 部分: 全部
  - 理由: 运行日志只对应旧实验上下文，没有迁移价值。

## 下一步继续复用

- [../ConceptSkillCDM/src/experiment_utils.py](/home/xph/jwc/research/ConceptSkillCDM/src/experiment_utils.py)
  - `setup_logging`
  - 适合继续迁入并改造成项目日志模块

- [../ConceptSkillCDM/src/config.py](/home/xph/jwc/research/ConceptSkillCDM/src/config.py)
  - `DATASET_DEFAULTS`
  - 适合继续改造成新项目的 `configs/defaults.py`

- [../ConceptSkillCDM/run_all_datasets.py](/home/xph/jwc/research/ConceptSkillCDM/run_all_datasets.py)
  - 多数据集调度框架
  - 适合在当前单任务训练稳定后再复用

## 当前已稳定下来的训练基线

- 数据:
  - `data/assist_09_ordered/train.csv`
  - `data/assist_09_ordered/valid.csv`
  - `data/assist_09_ordered/test.csv`
- 知识图:
  - `data/assist_09_ordered/transition_graph/propagation_graph.csv`
- 默认超参数:
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `gs_mode = conditional`
- 当前最好正式结果:
  - `300 epoch`
  - `best_val_auc = 0.716939`
  - `test_auc = 0.709411`
  - best checkpoint:
    - `results/assist_09_current_worktree_300ep_best.pt`

## 推荐的新项目目录结构

建议在当前骨架基础上收敛到下面这个结构:

```text
decoupled_cd/
├── README_spec.md
├── configs/
│   └── defaults.py
├── data/
│   ├── readers.py
│   ├── mappings.py
│   ├── pipeline.py
│   ├── q_matrix.py
│   └── datasets.py
├── models/
│   ├── encoders.py
│   ├── knowledge_sets.py
│   └── decoupled_cdm.py
├── trainers/
│   ├── engine.py
│   ├── evaluator.py
│   └── losses.py
├── utils/
│   ├── logging.py
│   ├── metrics.py
│   ├── device.py
│   └── seed.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── docs/
│   └── reuse_plan.md
├── results/
└── logs/
```

## 目录迁移建议

- 保留现有:
  - [data/readers.py](/home/xph/jwc/research/decoupled_cd/data/readers.py)
  - [data/mappings.py](/home/xph/jwc/research/decoupled_cd/data/mappings.py)
  - [data/pipeline.py](/home/xph/jwc/research/decoupled_cd/data/pipeline.py)
  - [scripts/prepare_data.py](/home/xph/jwc/research/decoupled_cd/scripts/prepare_data.py)

- 优先新增:
  - `data/q_matrix.py`: 统一 `Q` 矩阵转 tensor、`TKC_u` 相关辅助函数
  - `data/datasets.py`: 新项目训练样本定义
  - `models/encoders.py`: 从学生作答记录得到 `E_u`
  - `models/knowledge_sets.py`: 由 `Q` 和 `K` 构造 `TKC_u / UKC_u`
  - `trainers/engine.py`: 最小训练循环
  - `utils/metrics.py`: 从参考项目迁入指标计算

- 后置新增:
  - `scripts/evaluate.py`
  - `configs/defaults.py`

## 总结

推荐迁移边界如下:

- 迁工程层:
  - 配置管理
  - 日志
  - 训练循环骨架
  - 指标计算
  - GPU 调度
  - 结果汇总

- 不迁模型层:
  - `src/model.py`
  - `src/personalization.py`
  - `src/module_activity.py`
  - 与 graph / hyperlens / residual calibration / IRT 强绑定的逻辑

- Step 2 的直接工程含义:
  - 可以借鉴“图传播代码的工程组织方式”
  - 不能借鉴“旧模型把不同信息源混合进同一主干表示”的具体实现

最重要的一点是: 新项目应把参考项目当作“工程模板来源”，而不是“建模模板来源”。
