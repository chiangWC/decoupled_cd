# Experiment 79: single-concept scoped evidence readout

- 分支: `exp/concept-evidence-readout-next`
- base: `exp/trellis-trial` 伪主线 `d981ae8`
- 代码状态: rejected low-signal branch，不合入 `exp/trellis-trial`
- 对照: 实验 78 伪主线默认 `concept_evidence_prior + concept_evidence_readout(min_count=1, max_count=0, seen_ratio=1.0, max_logit=0.5)`

## 动机

实验 78 的 readout residual 正收益主要来自 `concept_count=1` 与 `all_seen`，但 `none_seen` / `partial_seen` 以及部分多知识点切片回撤。实验 79 增加可解释触发上限 `--concept-evidence-readout-max-count`，首测只允许 residual 在单知识点题触发，尝试保留收益并降低其它切片副作用。

该结构仍只使用 Q 矩阵和 train-history student-concept evidence，不读学生-题目 pair memory，不是 CF。

## 实现

- 新增 `DecoupledCDM.concept_evidence_readout_max_count`。
- `0` 表示不启用上限，保持实验 78 行为兼容。
- `--concept-evidence-readout-max-count 1` 表示 readout residual 只在 `concept_count=1` 且满足 seen_ratio 条件时触发。
- 训练、评估、切片分析 summary loader 均已支持该字段。

Best command:

```bash
OUTPUT=results/concept_evidence_readout_next/assist_09_seed2024_exp79_readout_single_only_300ep.json \
  bash scripts/run_assist09_baseline.sh \
  --concept-evidence-readout-max-count 1
```

## 单 seed 结果

Matched baseline 是实验 78 伪主线 seed=2024。

| config | result path | test_auc | delta AUC | test_acc | delta ACC | test_rmse | delta RMSE | test_brier | delta Brier | test_ece | delta ECE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exp78 pseudo-mainline | `results/concept_evidence_prior_next/assist_09_seed2024_prior_default_plus_readout_min1_max05_300ep.json` | 0.772562 | 0.000000 | 0.730823 | 0.000000 | 0.424811 | 0.000000 | 0.180464 | 0.000000 | 0.052361 | 0.000000 | reference |
| readout max_logit=0.75 | `results/concept_evidence_readout_next/assist_09_seed2024_exp78_readout_max075_300ep.json` | 0.770940 | -0.001621 | 0.730480 | -0.000343 | 0.425753 | +0.000942 | 0.181266 | +0.000801 | 0.054811 | +0.002449 | rejected |
| single-only readout | `results/concept_evidence_readout_next/assist_09_seed2024_exp79_readout_single_only_300ep.json` | 0.772865 | +0.000303 | 0.734343 | +0.003521 | 0.423911 | -0.000900 | 0.179701 | -0.000764 | 0.050570 | -0.001791 | multi-metric signal |

## Multi-seed 验证

Matched baseline 是实验 78 伪主线；candidate 是 `--concept-evidence-readout-max-count 1`。

| seed | baseline AUC | candidate AUC | delta AUC | baseline ACC | candidate ACC | delta ACC | baseline RMSE | candidate RMSE | delta RMSE | baseline ECE | candidate ECE | delta ECE | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2024 | 0.772562 | 0.772865 | +0.000303 | 0.730823 | 0.734343 | +0.003521 | 0.424811 | 0.423911 | -0.000900 | 0.052361 | 0.050570 | -0.001791 | normal |
| 2025 | 0.769170 | 0.769706 | +0.000536 | 0.734476 | 0.735504 | +0.001028 | 0.425773 | 0.424985 | -0.000788 | 0.054576 | 0.050221 | -0.004355 | normal |
| 2026 | 0.502804 | 0.502923 | +0.000119 | 0.511446 | 0.517251 | +0.005804 | 0.519218 | 0.517176 | -0.002042 | 0.173121 | 0.168051 | -0.005070 | both runs degenerate |
| 2027 | 0.771996 | 0.773462 | +0.001466 | 0.733068 | 0.730880 | -0.002188 | 0.424792 | 0.424642 | -0.000150 | 0.053035 | 0.052333 | -0.000702 | normal |

All seeds `2024/2025/2026/2027`:

- mean `AUC +0.000606`
- mean `ACC +0.002041`
- mean `RMSE -0.000970`
- mean `Brier -0.000920`
- mean `ECE -0.002979`

Normal-learning seeds `2024/2025/2027`:

- mean `AUC +0.000769`
- mean `ACC +0.000787`
- mean `RMSE -0.000613`
- mean `Brier -0.000521`
- mean `ECE -0.002283`

## Slice 对照

Seed=2024 相对实验 78:

- `concept_count=1`: `AUC -0.000156`, `ACC +0.003403`, `RMSE -0.000691`, `Brier -0.000582`, `ECE -0.000881`。
- `concept_count=2`: `AUC +0.001626`, `ACC +0.003868`, `RMSE -0.001968`, `Brier -0.001723`, `ECE -0.005700`。
- `concept_count=3`: `AUC +0.001781`, `ACC +0.004430`, `RMSE -0.001653`, `Brier -0.001536`, `ECE -0.010861`。
- `concept_count=4+`: `AUC +0.003819`, `ACC +0.008264`, `RMSE -0.001098`, `Brier -0.001004`, `ECE +0.002806`。
- `all_seen`: `AUC +0.000225`, `ACC +0.003441`, `RMSE -0.000796`, `Brier -0.000679`, `ECE -0.001048`。
- `none_seen`: `AUC -0.000034`, `ACC +0.004222`, `RMSE -0.002714`, `Brier -0.001959`, `ECE -0.003635`。
- `partial_seen`: `AUC +0.005071`, `ACC +0.012121`, `RMSE -0.009123`, `Brier -0.007772`, `ECE -0.031928`。

Slice result paths:

- `results/concept_evidence_readout_next/assist_09_seed2024_exp78_baseline_slices_for_exp79_compare.json`
- `results/concept_evidence_readout_next/assist_09_seed2024_exp79_readout_single_only_slices.json`

## 结论

- `--concept-evidence-readout-max-count 1` 有可解释正向迹象：多 seed AUC 全为正向，RMSE/Brier/ECE 也全为正向。
- 但增量幅度不够明显：正常学习 seeds 的 mean `AUC +0.000769`，弱于当前阶段扩主线需要的信号强度。
- 相比直接放大 readout residual（`max_logit=0.75` 明显回撤），收窄触发范围更有效。
- 决策: 不合入 `exp/trellis-trial`；保留为低幅稳定化诊断，后续除非有更强结构能放大该方向，否则不继续围绕同一 readout scope 微调扩线。
- 风险: seed2027 的 ACC 回撤，且 seed2026 退化仍未解决。
