# Experiment 83: exp81 q-local expert interaction diagnosis

- 分支: `exp/exp81-target-concept-interaction-rebase`
- base: 当前 `exp/trellis-trial` exp81 伪主线语义
- 代码状态: 已完成诊断与 coverage-gating follow-up；结论影响后续搜索方向，但不构成新的默认主线
- 对照: 当前 exp81 伪主线 `experiment 70 + single-only concept-evidence readout + experiment 51 full-trigger readout expert + exp70 none_seen sidecar`

## 动机

在 experiment 81 promote 之后，继续把 `q-local` 叠回当前伪主线没有形成 clean 增益。但这还不能直接说明 `q-local` 本身失效，因为更早的诊断已经提示 experiment 51 的 full-trigger readout expert 可能压制部分更干净的表示层结构。

这一轮要回答的是三个更具体的问题:

- `q-local` 在当前 exp81 伪主线上究竟是整体无效，还是被 `expert` 吸收方式压制
- 如果拿掉 `expert`，`q-local` 能否在 matched family 里恢复
- 能否通过 coverage-based gate 保留 `expert` 的同时释放 `q-local`

## 实验设计

### 1. 直接叠回当前 exp81 伪主线

在当前默认伪主线之上直接开启 `q-conditioned local mastery` readout。

### 2. no-expert matched family

保留 `sidecar + single-only readout`，只去掉 `expert`，然后在这个 matched family 中比较:

- no-expert baseline
- no-expert + `q-local`

目的是隔离 “`q-local` 是否必须依赖去掉 `expert` 才能恢复”。

### 3. coverage-gating follow-up

在 `expert + q-local` 的组合上继续测试三种 gate:

- inverse coverage gate
- partial coverage gate
- soft coverage gate

目的是判断问题是否主要绑定在 `all_seen` 区域上的 `expert` 作用。

## 工程验证

- coverage gate 相关代码提交:
  - `10b8e53` `feat: gate readout expert by inverse target coverage`
  - `d96e385` `feat: add partial coverage gate for readout expert`
  - `578cdd5` `feat: add soft coverage gate for readout expert`
  - `40ecac6` `fix: simplify expert coverage gate validation`
- 远端回归:
  - `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes`
  - 结果: `55 tests`, `OK`

## 结果目录

- `results/q_local_single_only_combo/`
- `results/q_local_single_only_no_expert/`
- `results/expert_q_local_inverse_coverage/`
- `results/expert_q_local_partial_coverage/`
- `results/expert_q_local_soft_coverage/`

## 正式结果

### 1. `q-local` 直接叠回当前 exp81 伪主线

| seed | exp81 pseudo-mainline AUC | `exp81 + q-local` AUC | delta AUC |
| ---: | ---: | ---: | ---: |
| 2024 | 0.767478 | 0.769124 | +0.001646 |
| 2025 | 0.771661 | 0.769131 | -0.002530 |
| 2026 | 0.771263 | 0.767620 | -0.003643 |

Three-seed mean relative to current exp81 pseudo-mainline:

- `AUC -0.001509`

这条线应直接判负，不能作为当前 trial 的 promote 候选。

### 2. no-expert matched family

| seed | no-expert baseline AUC | no-expert + `q-local` AUC | delta AUC |
| ---: | ---: | ---: | ---: |
| 2024 | 0.768669 | 0.770341 | +0.001673 |
| 2025 | 0.770623 | 0.770536 | -0.000087 |
| 2026 | 0.768806 | 0.770799 | +0.001993 |

Matched-family three-seed mean deltas:

- `AUC +0.001193`
- `ACC +0.000013`
- `RMSE -0.000279`
- `Brier -0.000237`

再把这条 no-expert candidate 与当前 exp81 pseudo-mainline 直接比较:

- `seed=2024`: `0.767478 -> 0.770341`, `AUC +0.002863`
- `seed=2025`: `0.771661 -> 0.770536`, `AUC -0.001125`
- `seed=2026`: `0.771263 -> 0.770799`, `AUC -0.000464`

Three-seed mean relative to current exp81 pseudo-mainline:

- `AUC +0.000425`
- `ACC -0.003356`
- `RMSE +0.001396`
- `Brier +0.001189`
- `ECE +0.000060`

因此，去掉 `expert` 后 `q-local` 的确会恢复，但这还不是一个足够 clean 的主线替代版本。

### 3. coverage-gating follow-up

#### inverse coverage gate

在 `expert + q-local` 上加入 inverse coverage gate 后，三 seed AUC 精确退化成 no-expert + `q-local` 轨迹:

- `seed=2024`: `0.770341`
- `seed=2025`: `0.770536`
- `seed=2026`: `0.770799`

这说明只要把 `all-seen` 区域上的 `expert` 硬关掉，模型就基本回到 no-expert family。

#### partial coverage gate

`seed=2024` 结果与 inverse gate 相同:

- `AUC 0.770341`

说明问题不在 `none_seen` 和 `partial_seen` 的更细分界线，而主要绑定在 `all-seen` 区域上的 `expert` 作用本身。

#### soft coverage gate

保留 `all-seen` 区域上的非零 `expert` floor 后，`seed=2024` 立刻掉回去:

- `AUC 0.766486`
- `ACC 0.732231`
- `RMSE 0.426164`
- `Brier 0.181616`
- `ECE 0.047113`

它不仅弱于 no-expert trajectory，也弱于 ungated 的 `exp81 + q-local`。

## 结论

- `q-local` 本身并没有被彻底证伪；它在 no-expert matched family 中能稳定恢复。
- 当前真正被诊断出来的问题是 experiment 51 的 `expert` 吸收方式，尤其是 `all-seen` 区域上的作用，会压制 `q-local`。
- `coverage-gating` family 不构成 promote 路线:
  - hard gate 只会退化成 no-expert + `q-local`
  - soft gate 会把负面影响带回来
- 所以这轮的有效产出不是新的默认主线，而是后续结构探索的约束:
  - 不要再把 `q-local` 直接叠回当前带 `expert` 的 trial
  - 也不要继续调 coverage gate 公式
  - 如果继续这条线，应改 `expert` 的作用形式或作用位置，而不是继续改 coverage mask
