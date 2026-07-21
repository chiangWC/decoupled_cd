# Bottleneck-Aware Monotone Requirement Surface：结果与拒绝记录

## 冻结结论

Bottleneck-Aware Monotone Requirement Surface 未通过预注册的 Stage 1
机会门，正式拒绝。它不能进入 Claude v2 端到端集成，不能计为论文模块，
也不触发 standard、holdout、test 或外部胜局确认。

本结论来自唯一正式目录：

```text
results/goal_two_module/monotone_requirement_surface_efb4ce6/
```

模型 seed 固定为 42；四个 head 均使用 full-batch Adam、learning rate
`5e-3`、weight decay 0 和固定 2,000 steps。没有调参、early stopping、
checkpoint 选择或多 seed。两数据集的 prediction 先分别写入外部 barrier，
再生成全局 seal；只有 seal 固定后才加载 audit-query labels。
本轮没有打开 valid/test，结果中 `test_or_validation_opened=false`。

## 候选与对照

- Full：读取目标概念 readiness 的最小值 `b` 和均值 `g`，用 16 参数
  二维硬单调 HLL 学习联合曲面。
- Pooled：16 参数一维 HLL，只读取 `g`。
- Capacity：16 参数、同输入的两个一维 HLL，在 logit 尺度加性组合，
  不含二维联合曲面。
- Standard NCD：读取完整 Q-masked readiness 的正权
  `K -> 256 -> 128 -> 1` 对照；容量不匹配，只作保守性能保险。

Full 与 Capacity 初始函数及活动参数量匹配。每个数据集的正式 effect 为
Full AUC 减去三个对照中最高的 AUC，不能挑选较弱对照。

## Q2 主结果

| 数据集 | 变体 | AUC | ACC | RMSE | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| ASSIST17 | Full | 0.749906 | 0.692139 | 0.445615 | 0.198573 | 0.017190 |
| ASSIST17 | Pooled | **0.750094** | 0.692870 | **0.445498** | **0.198469** | **0.010590** |
| ASSIST17 | Capacity | 0.749958 | **0.693601** | 0.445553 | 0.198518 | 0.011423 |
| ASSIST17 | Standard NCD | 0.740022 | 0.682267 | 0.449674 | 0.202207 | 0.017395 |
| MOOCRadar | Full | **0.829958** | **0.816794** | **0.365501** | **0.133591** | 0.036772 |
| MOOCRadar | Pooled | 0.828863 | 0.811705 | 0.365773 | 0.133790 | 0.037689 |
| MOOCRadar | Capacity | 0.827467 | 0.815522 | 0.366893 | 0.134610 | 0.029309 |
| MOOCRadar | Standard NCD | 0.809936 | 0.783715 | 0.378041 | 0.142915 | **0.025419** |

ASSIST17 Q2 包含 2,735 行、246 名学生；MOOCRadar 包含 786 行、
205 名学生。保守 effect 与 student-clustered 95% CI 为：

| 数据集 | Full − Pooled | Full − Capacity | Full − NCD | 保守 effect | 95% CI |
|---|---:|---:|---:|---:|---:|
| ASSIST17 | -0.000188 | -0.000053 | +0.009884 | **-0.000188** | [-0.002282, +0.001621] |
| MOOCRadar | +0.001095 | +0.002491 | +0.020022 | **+0.001095** | [-0.006344, +0.004211] |

bootstrap 固定 2,000 次并按学生聚类；它不是多 seed。

## 预注册门

| 条件 | 结果 | 证据 |
|---|:---:|---|
| 两数据集 Q2 effect 均 `>=0.005` | **失败** | -0.000188 / +0.001095 |
| 至少一处 Q2 effect `>=0.010` | **失败** | 最大仅 +0.001095 |
| 至少一处 student CI 下界 `>0` | **失败** | 两处均跨 0 |
| 两处 Q2 Brier 回归 `<=0.001` | 通过 | +0.000105 / -0.000199 |
| 两处 eligible-pooled effect 非负 | **失败** | ASSIST17 -0.000396 |
| 去 top-5 item 后两处 effect 为正 | 通过 | +0.000115 / +0.000387 |
| 去 top-5 Q-pair 后两处 effect 为正 | **失败** | -0.000147 / -0.000262 |
| 两处 Full surface noncollapsed | 通过 | 最大 mixed difference 0.013473 / 0.005744 |

聚合决策为 `passed=false`。它同时错失幅度、跨数据集稳定性、CI 和
Q-pair 稳健性，不属于可以靠四舍五入解释的边缘失败。

## 责任归因

两处曲面均明确偏离加性，且没有 clamp 命中，说明优化器确实学到了二维
交互；但该交互没有转化为可泛化收益。ASSIST17 上 Pooled 与 Capacity 都
略高于 Full；MOOCRadar 的小幅正收益在移除 top-5 Q-pair 后反向。
因此失败责任不是“模块塌缩”或“没有训练到”，而是当前 support-only
readiness 下，最弱概念与均值的联合 Diagnosis 不是足够强的预测机会。

Standard NCD 在第 1,000 到 2,000 步仍有 5.9%/7.8% loss 改善，可能尚未
完全收敛；这不影响拒绝，因为两数据集真正限制 Full effect 的都是
Pooled，而不是 NCD。不得删除强 Pooled 对照来重写结论。

后续不得调 lattice、增加 residual/gate/loss、改变阈值或换名重跑。
若继续探索，必须转向职责或数据流实质不同的新模块。

## Provenance 与不可变工件

正式代码 commit：

```text
efb4ce6eb30d6196888ec97ea225de77c0a58fd8
```

architecture fingerprint：

```text
72b6a87b1930b67951876e4368c3110bffa390348f57458469b5a17af4ddc312
```

| 工件 | SHA-256 |
|---|---|
| `stage1_decision.json` | `4d9831e1f1d49b703e0b7aaa3e85c6cd3f8a7097d4ebbe015c00e4c2c905f56e` |
| `global_prediction_barrier.json` | `1fd6a3a22f17335065646c0654ba2eaf9c29ebd1e93da10ac648dffb3eaf3e40` |
| `ASSIST17/evaluation.json` | `93c43d97436a2b743d206fdac5f37f753d5964b053e622b6a8654a035cf57d88` |
| `MOOCRadar/evaluation.json` | `1a62a59477332571b6bd3e85dc3316060da9018b2aff05f74fa2184d88a38eb0` |
| `ASSIST17/prediction_barrier.json` | `b51dda786d0c0a3e4aaf934eb0fbf39924b9023d5ef02e40b7a25cfef8615241` |
| `MOOCRadar/prediction_barrier.json` | `3cb956113231a4f67adc80c22a0665e6708028267127f233c87bbde460e0b02c` |

全局 barrier 的嵌入式外部锚为
`64ffb1fd435a11636453c6e5ef4f6de72cbd901587c10bb34dd2f7d1e3fd5a3f`。
生成的 checkpoints、predictions、surface、bootstrap 明细和带标签对齐表
保留在 `results/`，不提交到 git。本文档是人类可读拒绝记录，不能作为
test 结果、外部模型比较或已获得论文模块的证据。
