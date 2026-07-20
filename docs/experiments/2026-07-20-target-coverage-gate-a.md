# Gate A：目标局部概念覆盖问题存在性检验

日期：2026-07-20
分支：`codex/target-coverage-gate-a`
基线提交：`d1a980a1e0d717b0e0c555aacdffc73128503858`

## 目的与边界

本实验只回答一个问题：在已知学生、随机交互划分的认知诊断设置中，当测试题所需概念未被该学生的训练历史充分覆盖时，强外部模型是否出现可重复的预测损失上升。

本实验不是模型选择或调参实验，不访问任何 test 文件。所有变量均由 standard train history、standard validation 行、冻结的 Full validation 预测和逐行对齐的外部 validation 预测构造。

## 数据与定义

使用 ASSIST17、MOOCRadar、XES3G5M 和 Junyi。对学生 (u) 和 validation 题目 (e)，定义：

[
c(u,e)=\frac{|O_u \cap R_e|}{|R_e|},
]

其中 (O_u) 是学生在 train history 中接触过的概念集合，(R_e) 是题目 Q 向量对应的需求概念集合。主回归量为未覆盖比例 (1-c(u,e))。

Junyi 的当前协议是一题一概念，且学生—题目划分使 validation 行全部为 exact-zero。它可以用于报告该现象的发生率，但若 (c) 没有变异，双向固定效应会将其明确标记为 `not_identified`，不参与 Gate A 的支持数据集计数。

## 主估计量

主结果使用外部模型逐行 log-loss：

[
L_{ue}=\beta(1-c(u,e))+\alpha_u+\gamma_e+\epsilon_{ue}.
]

- 学生固定效应 (alpha_u) 控制学生能力、历史长度、历史正确率和全局概念覆盖等所有学生层面的不变因素。
- 题目固定效应 (gamma_e) 控制题目难度、训练频率、Q 概念数和题目身份等所有题目层面的不变因素。
- 因而 (eta) 由“同一学生面对不同覆盖题目”与“同一题目被不同覆盖学生作答”的交叉变化识别。
- (eta>0) 表示目标局部覆盖越缺失，外部模型预测损失越高。

同时报告 external Brier、Full log-loss 和 Full 相对 external 的 log-loss advantage，但它们不是 Gate A 的主判据。

不把不同全局覆盖程度的学生直接混在一起解释：除学生固定效应外，还按学生全局概念覆盖五分位分层重复 external log-loss 检验。

## 不确定性与门槛

- 固定效应点估计使用交替去均值求解。
- 95% CI 使用学生为聚类单位的 pairs bootstrap；固定 2,000 次，`seed=2024`。
- Gate A 预先规定：至少 3 个数据集的 external log-loss 效应 95% CI 下界严格大于 0 才通过。
- 无可识别覆盖变异的数据集记为 N/A，不视为支持，也不人为补造对照。
- bootstrap 不是多 seed；模型预测、划分和所有训练产物均保持冻结。

## 资产

| 数据集 | train/valid | Full validation | 外部 validation |
|---|---|---|---|
| ASSIST17 | `knofield_data/assist_17` | `final_standard_v9/a17_predictions.csv` | ORCDF standard validation |
| MOOCRadar | `knofield_data/moocradar` | `final_standard_v9/moo_predictions.csv` | SVGCD standard validation |
| XES3G5M | `knofield_data/xes3g5m` | `final_standard_v9/xes_predictions.csv` | ORCDF standard validation |
| Junyi | `knofield_data/junyi` | `final_standard_v9/junyi_predictions.csv` | ORCDF standard validation |

脚本保存输入 SHA-256、逐行丰富表、覆盖发生率、全局覆盖分层指标、固定效应结果和机器可读总结。任何预测行的学生、题目或标签顺序不一致都会直接报错。

## 复现

```bash
/home/xph/anaconda3/envs/decoupled_cd/bin/python \
  scripts/analyze_target_coverage_gate.py \
  --dataset-spec ASSIST17 \
    /home/xph/jwc/research/knofield_data/assist_17 \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/a17_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/legacy/orcdf_a17_standard/valid_predictions.csv \
  --dataset-spec MOOCRadar \
    /home/xph/jwc/research/knofield_data/moocradar \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/moo_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/legacy/svgcd/moocradar/standard/valid_predictions.csv \
  --dataset-spec XES3G5M \
    /home/xph/jwc/research/knofield_data/xes3g5m \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/xes_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/legacy/orcdf_xes_standard/valid_predictions.csv \
  --dataset-spec Junyi \
    /home/xph/jwc/research/knofield_data/junyi \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/junyi_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/orcdf/junyi/standard/valid_predictions.csv \
  --bootstrap 2000 \
  --seed 2024 \
  --output-dir results/goal_two_module/gate_a_target_coverage
```

正式运行后只追加结果与判决，不改变定义、估计量或门槛。
