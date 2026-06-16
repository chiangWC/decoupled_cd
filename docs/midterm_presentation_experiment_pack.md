# 中期答辩实验展示包

这份文档按论文实验设计整理 PPT 可展示表格。它不是新的实验记录，而是从现有实验文档和远端结果 JSON 中抽取展示口径。

核心叙事：

> DecoupledCDM 面向 coverage bias 和不完整历史观测下的认知诊断。常规 split 上模型具备竞争力；在 low-coverage 和 hidden-history 诊断压力场景中，相比 NCD 等 published baseline 有明显改善；RCD/SCD/SVGCD 仍然是很强的 published baselines，不能声称全面超过。

## 展示总览

| 实验 | 使用 split | 回答问题 | PPT 中的表 |
|---|---|---|---|
| Standard baseline comparison | Standard | 常规 CD 性能是否有竞争力 | 表 1：standard split published baseline comparison |
| Coverage-bias diagnosis | Holdout | low coverage 是否更难 | 表 2：holdout coverage slice diagnostic |
| Holdout baseline comparison | Holdout | published baselines 能否处理 low coverage | 表 3：holdout split baseline comparison；表 4：low-coverage comparison |
| TKC-only ablation | Holdout | UKC branch 是否必要 | 表 5：TKC-only overall；表 6：TKC-only coverage slice |
| Coarse module ablation | Holdout | 主要模块在多数据集上是否稳定有效 | 表 7：multi-dataset coarse ablation；表 8：coverage-slice coarse ablation |
| History hiding | Holdout | 隐藏历史时是否可靠 | 表 9：Ours multi-ratio；表 10：RCD multi-ratio |
| Capacity control | Holdout | 是否只是参数更多 | 表 11：capacity-control backup |

说明：

- baseline 主表统一展示 `AUC / ACC / RMSE`，这是 CD 论文中最常见的三项整体指标。
- coverage slice 表的核心是 `Low AUC / Full AUC / Gap / Low Brier / Low ECE`，因为它回答的是低覆盖切片问题，不是整体分类准确率问题。
- Junyi 不建议放主 PPT。`junyi_sample` 可作为备查，但主线建议聚焦 ASSIST09、ASSIST17、NIPS34。

## 1. Standard Baseline Comparison

来源：`docs/experiments/120_published_cd_baseline_comparison.md`，并从远端 JSON/log 补齐 `ACC/RMSE`。

**表 1：Standard split published baseline comparison.**

| Dataset | Method | AUC | ACC | RMSE |
|---|---|---:|---:|---:|
| assist09_standard | Ours | 0.7793 | 0.7380 | 0.4189 |
| assist09_standard | DINA | 0.7008 | 0.6243 | 0.4682 |
| assist09_standard | IRT | 0.7021 | 0.6594 | 0.4745 |
| assist09_standard | MIRT | 0.6964 | 0.7020 | 0.4455 |
| assist09_standard | NCD | 0.7568 | 0.7276 | 0.4305 |
| assist09_standard | RCD | 0.7730 | 0.7341 | 0.4207 |
| assist09_standard | SVGCD | 0.7830 | 0.7443 | 0.4161 |
| assist09_standard | SCD | 0.7727 | 0.7337 | 0.4214 |
| assist17_standard | Ours | 0.7823 | 0.7139 | 0.4337 |
| assist17_standard | DINA | 0.6767 | 0.6080 | 0.4880 |
| assist17_standard | IRT | 0.6905 | 0.4413 | 0.5276 |
| assist17_standard | MIRT | 0.7520 | 0.6904 | 0.4471 |
| assist17_standard | NCD | 0.7606 | 0.6957 | 0.4448 |
| assist17_standard | RCD | 0.7840 | 0.7172 | 0.4325 |
| assist17_standard | SVGCD | 0.7867 | 0.7174 | 0.4317 |
| assist17_standard | SCD | 0.7875 | 0.7184 | 0.4306 |
| nips34_standard | Ours | 0.7862 | 0.7176 | 0.4319 |
| nips34_standard | DINA | 0.7058 | 0.6190 | 0.4769 |
| nips34_standard | IRT | 0.6714 | 0.5375 | 0.4983 |
| nips34_standard | MIRT | 0.7634 | 0.6954 | 0.4446 |
| nips34_standard | NCD | 0.7718 | 0.7046 | 0.4392 |
| nips34_standard | RCD | 0.7789 | 0.7095 | 0.4371 |
| nips34_standard | SVGCD | 0.7829 | 0.7140 | 0.4348 |
| nips34_standard | SCD | 0.7787 | 0.7077 | 0.4371 |

PPT 讲法：

- Standard split 上，Ours 在 ASSIST09/NIPS34 上接近或超过强 baseline；ASSIST17 上 SCD/SVGCD/RCD 更强。
- 这张表回答“方法不是只在 stress split 上有效”，但不能讲全面最优。
- 如果 PPT 太挤，保留 `Ours / NCD / RCD / SVGCD / SCD`，把 DINA/IRT/MIRT 放备份页。

## 2. Coverage-Bias Diagnosis

来源：`docs/experiments/120_published_cd_baseline_comparison.md` 的 holdout coverage slice。

定义：

```text
target_coverage(s, e) = |Q_e intersect Seen_s(train)| / |Q_e|
low_coverage = target_coverage < 0.5, including zero coverage
coverage_gap = full_coverage_auc - low_coverage_auc
```

**表 2：Coverage-bias diagnostic on holdout splits.**

| Dataset | Method | Overall AUC | Low AUC | Full AUC | Gap | Low Brier | Low ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| assist09_holdout | NCD | 0.7227 | 0.6673 | 0.7608 | 0.0934 | 0.2036 | 0.0585 |
| assist09_holdout | RCD | 0.7634 | 0.7495 | 0.7739 | 0.0244 | 0.1790 | 0.0339 |
| assist09_holdout | Ours | 0.7551 | 0.7339 | 0.7768 | 0.0429 | 0.1909 | 0.0813 |
| assist17_holdout | NCD | 0.7363 | 0.7157 | 0.7620 | 0.0463 | 0.2128 | 0.0271 |
| assist17_holdout | RCD | 0.7830 | 0.7819 | 0.7921 | 0.0102 | 0.1880 | 0.0125 |
| assist17_holdout | Ours | 0.7558 | 0.7432 | 0.7806 | 0.0374 | 0.2132 | 0.0977 |
| nips34_holdout | NCD | 0.7251 | 0.6725 | 0.7702 | 0.0977 | 0.2274 | 0.0248 |
| nips34_holdout | RCD | 0.7732 | 0.7733 | 0.7734 | 0.0001 | 0.1937 | 0.0365 |
| nips34_holdout | Ours | 0.7726 | 0.7669 | 0.7801 | 0.0132 | 0.1953 | 0.0258 |

PPT 讲法：

- NCD 在三个 holdout split 上都有明显 `Low AUC < Full AUC`，coverage-induced bias 是可观测问题。
- Ours 相比 NCD 大幅提高 Low AUC，并缩小 NCD 的 coverage gap。
- RCD 是强图模型，coverage gap 很小；这说明图结构建模本身对 low coverage 很有帮助。

## 3. Holdout Baseline Comparison

来源：`docs/experiments/120_published_cd_baseline_comparison.md`，并从远端 JSON/log 补齐 `ACC/RMSE`。第 1 节只展示 standard；holdout baseline 放在这里。

**表 3：Holdout split published baseline comparison.**

| Dataset | Method | AUC | ACC | RMSE |
|---|---|---:|---:|---:|
| assist09_holdout | Ours | 0.7587 | 0.7280 | 0.4267 |
| assist09_holdout | DINA | 0.6393 | 0.5668 | 0.4920 |
| assist09_holdout | IRT | 0.6938 | 0.6620 | 0.4738 |
| assist09_holdout | MIRT | 0.6856 | 0.6991 | 0.4486 |
| assist09_holdout | NCD | 0.7228 | 0.7046 | 0.4424 |
| assist09_holdout | RCD | 0.7634 | 0.7307 | 0.4239 |
| assist09_holdout | SVGCD | 0.7684 | 0.7340 | 0.4219 |
| assist09_holdout | SCD | 0.7631 | 0.7270 | 0.4254 |
| assist17_holdout | Ours | 0.7637 | 0.6991 | 0.4425 |
| assist17_holdout | DINA | 0.6407 | 0.5787 | 0.5060 |
| assist17_holdout | IRT | 0.6872 | 0.4369 | 0.5285 |
| assist17_holdout | MIRT | 0.7526 | 0.6942 | 0.4466 |
| assist17_holdout | NCD | 0.7363 | 0.6785 | 0.4533 |
| assist17_holdout | RCD | 0.7830 | 0.7178 | 0.4325 |
| assist17_holdout | SVGCD | 0.7836 | 0.7158 | 0.4326 |
| assist17_holdout | SCD | 0.7866 | 0.7193 | 0.4308 |
| nips34_holdout | Ours | 0.7728 | 0.7041 | 0.4391 |
| nips34_holdout | DINA | 0.6408 | 0.5585 | 0.5041 |
| nips34_holdout | IRT | 0.6689 | 0.5337 | 0.4998 |
| nips34_holdout | MIRT | 0.7584 | 0.6915 | 0.4466 |
| nips34_holdout | NCD | 0.7251 | 0.6623 | 0.4594 |
| nips34_holdout | RCD | 0.7732 | 0.7043 | 0.4400 |
| nips34_holdout | SVGCD | 0.7735 | 0.7047 | 0.4396 |
| nips34_holdout | SCD bs128 partial | 0.7732 | 0.7046 | 0.4397 |

注：`nips34_holdout` 的 SCD 是 `batch_size=128` memory-rescue partial epoch3，不是默认 `batch_size=256` 的完整 5 epoch；主表可放脚注，答辩时不要主动强调为完整 SCD 复现实验。

**表 4：Holdout low-coverage comparison.**

| Dataset | NCD Low AUC | Ours Low AUC | RCD Low AUC | NCD Gap | Ours Gap | RCD Gap |
|---|---:|---:|---:|---:|---:|---:|
| assist09_holdout | 0.6673 | 0.7339 | 0.7495 | 0.0934 | 0.0429 | 0.0244 |
| assist17_holdout | 0.7157 | 0.7432 | 0.7819 | 0.0463 | 0.0374 | 0.0102 |
| nips34_holdout | 0.6725 | 0.7669 | 0.7733 | 0.0977 | 0.0132 | 0.0001 |

PPT 讲法：

- 表 3 说明 holdout split 下 Ours 仍然有竞争力，但 RCD/SVGCD/SCD 很强。
- 表 4 说明 Ours 明显优于 NCD 的 low-coverage diagnosis。
- 不要说 Ours 在 holdout published baselines 上全面第一。

## 4. TKC-Only Direct Ablation

来源：`docs/experiments/123_tkc_only_direct_ablation.md` 和对应远端 result JSON。

目的：直接验证 UKC branch 是否必要。

```text
Full adaptive: Z_s = w_s z_s^TKC + (1 - w_s) z_s^UKC
TKC-only:      Z_s = z_s^TKC
```

**表 5：TKC-only overall metrics.**

| Dataset | Model | AUC | ACC | RMSE |
|---|---|---:|---:|---:|
| ASSIST09 holdout | Full adaptive | 0.7539 | 0.7255 | 0.4295 |
| ASSIST09 holdout | TKC-only | 0.7530 | 0.7255 | 0.4300 |
| ASSIST17 holdout | Full adaptive | 0.7558 | 0.6883 | 0.4482 |
| ASSIST17 holdout | TKC-only | 0.7548 | 0.6873 | 0.4483 |
| NIPS34 holdout | Full adaptive | 0.7726 | 0.7055 | 0.4388 |
| NIPS34 holdout | TKC-only | 0.7724 | 0.7043 | 0.4391 |

**表 6：TKC-only coverage slice.**

| Dataset | Model | Low AUC | Full AUC | Gap | Low Brier | Low ECE |
|---|---|---:|---:|---:|---:|---:|
| ASSIST09 holdout | Full adaptive | 0.7331 | 0.7766 | 0.0434 | 0.1891 | 0.0790 |
| ASSIST09 holdout | TKC-only | 0.7318 | 0.7766 | 0.0448 | 0.1894 | 0.0782 |
| ASSIST17 holdout | Full adaptive | 0.7432 | 0.7806 | 0.0374 | 0.2132 | 0.0977 |
| ASSIST17 holdout | TKC-only | 0.7404 | 0.7810 | 0.0406 | 0.2140 | 0.0974 |
| NIPS34 holdout | Full adaptive | 0.7669 | 0.7801 | 0.0132 | 0.1953 | 0.0258 |
| NIPS34 holdout | TKC-only | 0.7669 | 0.7796 | 0.0127 | 0.1955 | 0.0313 |

PPT 讲法：

- overall 指标差异不大，但方向一致：TKC-only 的 AUC/ACC/RMSE 基本都略差。
- coverage slice 上 ASSIST09/ASSIST17 更贴合主线：移除 UKC 后 low AUC 下降、gap 扩大。
- NIPS34 是 mixed，不要说三套数据集完全一致。

## 5. Multi-Dataset Coarse Ablations

来源：

- `docs/experiments/124_multi_dataset_coarse_module_ablation.md`
- `docs/experiments/116_student_concept_holdout_coverage_slice.md`

实验口径：ASSIST17 使用 seed2028 exp110 holdout 口径；NIPS34 使用 seed2024 `dim80x96/lr2e-4` holdout 口径。Junyi 不纳入该消融。

**表 7：Multi-dataset coarse ablation overall metrics.**

| Dataset | Variant | AUC | Delta AUC | ACC | RMSE | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| assist17_holdout | Full model | 0.7637 | +0.0000 | 0.6991 | 0.4425 | 0.1958 | 0.0289 |
| assist17_holdout | w/o evidence-aware readout | 0.7890 | +0.0253 | 0.7206 | 0.4299 | 0.1849 | 0.0206 |
| assist17_holdout | w/o cognitive alignment objective | 0.7516 | -0.0121 | 0.6794 | 0.4517 | 0.2040 | 0.0641 |
| assist17_holdout | w/o dual-branch ensemble | 0.7523 | -0.0114 | 0.6904 | 0.4462 | 0.1991 | 0.0087 |
| nips34_holdout | Full model | 0.7728 | +0.0000 | 0.7041 | 0.4391 | 0.1928 | 0.0272 |
| nips34_holdout | w/o evidence-aware readout | 0.7755 | +0.0027 | 0.7073 | 0.4373 | 0.1912 | 0.0167 |
| nips34_holdout | w/o cognitive alignment objective | 0.7736 | +0.0008 | 0.7058 | 0.4379 | 0.1917 | 0.0086 |
| nips34_holdout | w/o dual-branch ensemble | 0.7714 | -0.0014 | 0.7046 | 0.4394 | 0.1930 | 0.0197 |

**表 8：Multi-dataset coarse ablation coverage-slice metrics.**

| Dataset | Variant | Low AUC | Full AUC | Gap | Low Brier | Low ECE |
|---|---|---:|---:|---:|---:|---:|
| assist17_holdout | Full model | 0.7524 | 0.7815 | 0.0292 | 0.2018 | 0.0429 |
| assist17_holdout | w/o evidence-aware readout | 0.7851 | 0.7949 | 0.0097 | 0.1873 | 0.0287 |
| assist17_holdout | w/o cognitive alignment objective | 0.7392 | 0.7822 | 0.0430 | 0.2214 | 0.1272 |
| assist17_holdout | w/o dual-branch ensemble | 0.7322 | 0.7772 | 0.0450 | 0.2075 | 0.0309 |
| nips34_holdout | Full model | 0.7677 | 0.7799 | 0.0122 | 0.1960 | 0.0408 |
| nips34_holdout | w/o evidence-aware readout | 0.7756 | 0.7758 | 0.0002 | 0.1913 | 0.0181 |
| nips34_holdout | w/o cognitive alignment objective | 0.7663 | 0.7823 | 0.0160 | 0.1948 | 0.0134 |
| nips34_holdout | w/o dual-branch ensemble | 0.7675 | 0.7774 | 0.0099 | 0.1953 | 0.0302 |

PPT 讲法：

- 粗粒度消融已经补到多数据集，但结果不是所有模块都单调正贡献。
- Cognitive alignment 在 ASSIST17 上证据最清楚：移除后 overall/low AUC 下降，gap 和 low ECE 明显变差；NIPS34 上 low AUC 略降、gap 变大，但 overall/calibration 是 mixed。
- Dual-branch ensemble 整体更像稳定化/性能组件：两个数据集 overall AUC 都下降，ASSIST17 low AUC 降得明显。
- Evidence-aware readout 不能作为核心创新点讲；这次跨数据集单 seed 里去掉它反而提升，说明它更像数据集相关的辅助/tuning 组件。
- 如果 PPT 时间不够，表 7/8 只展示 `Full / w/o cognitive alignment / w/o dual-branch`，把 evidence-aware readout 放备查页。

## 6. History Hiding Robustness

来源：

- `docs/experiments/117_history_hiding_stress_test.md`
- `docs/experiments/120_published_cd_baseline_comparison.md`

实验口径：evaluation-only 地隐藏训练历史证据，checkpoint 权重固定。对内部模型，重新构建 masked train interactions 对应的 student-side history tensors；对 RCD，重新构建 masked train file 对应的 train-history graph。

**表 9：Ours multi-ratio history hiding on holdout splits.**

| Dataset | Model | Hide | Original AUC | Hidden AUC | Hidden ACC | Delta AUC | Hidden Brier | Hidden ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assist09_holdout | Ours | 0.2 | 0.7539 | 0.7493 | 0.7229 | 0.0046 | 0.1864 | 0.0491 |
| assist09_holdout | Ours | 0.4 | 0.7539 | 0.7429 | 0.7191 | 0.0110 | 0.1889 | 0.0521 |
| assist09_holdout | Ours | 0.6 | 0.7539 | 0.7331 | 0.7135 | 0.0208 | 0.1926 | 0.0556 |
| assist09_holdout | Ours | 0.8 | 0.7539 | 0.7177 | 0.7030 | 0.0362 | 0.1984 | 0.0619 |
| assist17_holdout | Ours | 0.2 | 0.7558 | 0.7517 | 0.6848 | 0.0040 | 0.2025 | 0.0537 |
| assist17_holdout | Ours | 0.4 | 0.7558 | 0.7448 | 0.6781 | 0.0109 | 0.2052 | 0.0560 |
| assist17_holdout | Ours | 0.6 | 0.7558 | 0.7349 | 0.6699 | 0.0209 | 0.2091 | 0.0598 |
| assist17_holdout | Ours | 0.8 | 0.7558 | 0.7156 | 0.6547 | 0.0401 | 0.2152 | 0.0583 |
| nips34_holdout | Ours | 0.2 | 0.7726 | 0.7697 | 0.7026 | 0.0029 | 0.1937 | 0.0178 |
| nips34_holdout | Ours | 0.4 | 0.7726 | 0.7655 | 0.6989 | 0.0071 | 0.1954 | 0.0177 |
| nips34_holdout | Ours | 0.6 | 0.7726 | 0.7581 | 0.6924 | 0.0145 | 0.1985 | 0.0180 |
| nips34_holdout | Ours | 0.8 | 0.7726 | 0.7395 | 0.6768 | 0.0331 | 0.2059 | 0.0219 |

**表 10：RCD multi-ratio history hiding.**

| Dataset | Method | Hide | Original AUC | Hidden AUC | Hidden ACC | Delta AUC | Hidden Brier | Hidden ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| assist09_holdout | RCD | 0.2 | 0.7634 | 0.7614 | 0.7292 | 0.0020 | 0.1806 | 0.0250 |
| assist09_holdout | RCD | 0.4 | 0.7634 | 0.7580 | 0.7262 | 0.0054 | 0.1822 | 0.0302 |
| assist09_holdout | RCD | 0.6 | 0.7634 | 0.7517 | 0.7227 | 0.0117 | 0.1853 | 0.0396 |
| assist09_holdout | RCD | 0.8 | 0.7634 | 0.7361 | 0.7113 | 0.0273 | 0.1933 | 0.0604 |
| assist17_holdout | RCD | 0.2 | 0.7830 | 0.7823 | 0.7175 | 0.0007 | 0.1874 | 0.0060 |
| assist17_holdout | RCD | 0.4 | 0.7830 | 0.7809 | 0.7155 | 0.0021 | 0.1880 | 0.0069 |
| assist17_holdout | RCD | 0.6 | 0.7830 | 0.7778 | 0.7130 | 0.0052 | 0.1893 | 0.0100 |
| assist17_holdout | RCD | 0.8 | 0.7830 | 0.7704 | 0.7075 | 0.0126 | 0.1928 | 0.0230 |
| nips34_holdout | RCD | 0.2 | 0.7732 | 0.7731 | 0.7041 | 0.0001 | 0.1937 | 0.0353 |
| nips34_holdout | RCD | 0.4 | 0.7732 | 0.7730 | 0.7039 | 0.0002 | 0.1937 | 0.0351 |
| nips34_holdout | RCD | 0.6 | 0.7732 | 0.7728 | 0.7035 | 0.0005 | 0.1938 | 0.0352 |
| nips34_holdout | RCD | 0.8 | 0.7732 | 0.7720 | 0.7028 | 0.0012 | 0.1942 | 0.0347 |

PPT 讲法：

- Ours 和 RCD 都可以做 multi-ratio hidden-history stress test。
- ASSIST09/ASSIST17 上，RCD 随隐藏比例增大单调退化，说明 stress test 对图模型也有意义。
- NIPS34 上 RCD 几乎不变，因此必须承认数据集差异。

## 7. Capacity Control Backup

来源：`docs/experiments/119_exp110_capacity_control_holdout.md`。只有在被问“是不是只是参数更多”时使用。

**表 11：Capacity-control backup.**

| Model | Overall AUC | ACC | RMSE | Low AUC | Full AUC | Gap |
|---|---:|---:|---:|---:|---:|---:|
| Full model | 0.7539 | 0.7255 | 0.4295 | 0.7331 | 0.7766 | 0.0434 |
| w/o dual tower | 0.7542 | 0.7261 | 0.4286 | 0.7286 | 0.7740 | 0.0453 |
| single96 capacity | 0.7533 | 0.7254 | 0.4304 | 0.7226 | 0.7755 | 0.0528 |

History hiding backup：

| Model | Original AUC | Hidden AUC | Hidden ACC | Delta AUC | Hidden Brier | Hidden ECE |
|---|---:|---:|---:|---:|---:|---:|
| Full model | 0.7539 | 0.7177 | 0.7030 | 0.0362 | 0.1984 | 0.0619 |
| w/o dual tower | 0.7542 | 0.7030 | 0.6919 | 0.0511 | 0.2082 | 0.0787 |
| single96 capacity | 0.7533 | 0.7057 | 0.6976 | 0.0475 | 0.2078 | 0.0888 |

PPT 讲法：

- 更宽的 single tower 可以接近 overall AUC/ACC/RMSE，但不能恢复 low-coverage 或 hidden-history 鲁棒性。
- 这可以支持“不是简单参数量解释”，但它是 single-seed backup 结果。

## 推荐 Slide 顺序

1. 问题：CD 评估中的 coverage bias。
2. 方法：tested / untested concept states 解耦 + cognitive alignment。
3. 表 1：standard baseline comparison。
4. 表 2：coverage-bias diagnosis。
5. 表 3/4：holdout baseline and low-coverage comparison。
6. 表 5/6：TKC-only ablation。
7. 表 7：multi-dataset coarse ablation overall。
8. 表 8：coverage-slice coarse ablation。
9. 表 9/10：history hiding robustness。
9. 局限与未来工作：更强的 graph baselines、NIPS34 RCD invariance、Junyi concept-definition ambiguity。

## 不要说

- 不要说模型全面超过所有 published baselines。
- 不要说 RCD 在 hidden history 下失败；RCD 仍然很强，而且 NIPS34 几乎不变。
- 不要说 history hiding 证明了广义公平性；它只是 incomplete observation-history stress test。
- 不要把项目历史模型放成答辩主 baseline；展示时尽量使用 published baselines。
- 不要混用 706-concept Junyi 和 39-concept `junyi_sample`。
- 不要把 capacity control、dual tower 或 branch BCE 当成主线创新。

## 简短答辩话术

如果被问：为什么 published-baseline 表不是全面领先？

> 本文的核心贡献不是声称一个结构在普通 split 上全面超过所有 CD 模型，而是关注 coverage-biased diagnosis。也就是说，当目标题对应概念在该学生历史中缺少观测时，模型通过区分 tested 和 untested concept states，并引入 cognitive evidence alignment，使诊断更稳定。实验显示，在这些 stress diagnostics 上，相比 NCD 等 published baseline 有明显收益；同时 RCD 这类强图模型依然很有竞争力，这也说明后续优化需要进一步吸收图结构建模能力。

如果被问：history hiding 是否合理？

> valid/test 的目标行不会用于构建历史状态。我们只扰动 train-history evidence：对自己的模型是重建 masked student-history tensors，对 RCD 是重建 masked train-history graph；checkpoint 权重保持不变，然后观察历史观测不完整时预测如何变化。因此它是一个 robustness diagnostic，不是重新训练后的主表比较。
