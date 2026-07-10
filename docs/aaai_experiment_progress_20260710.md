# AAAI 双路线实验进度（2026-07-10）

## 当前结论

seed 42 campaign 已按停止规则结束。插件路线在 ASSIST17、XES3G5M、ASSIST09 三个数据集达到“最高 holdout DOA，且同骨干 AUC 下降不超过 0.002”，计数为 3/3；因此没有启动完整模型 XES 长训。完整模型路线已在 ASSIST17、MOOCRadar 达到严格 zero-slice AUC 目标，计数为 2/3。当前更适合作为 AAAI 主线的是“近乎无损预测性能的可插拔诊断增强”，完整模型作为严格未覆盖知识预测的补充证据。

统一设置为 `train_seed=42`、`doa_seed=42`、`split_seed=2024`、`min_responses=3`。配置只由 validation 决定；冻结后，每个最终配置只评估 test 一次。正式产物位于 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-20260710/`，不进入 Git。

## 插件路线：3/3 达标

| 数据集 | 冻结配置 | test AUC | 同骨干基线 | AUC 差值 | test holdout DOA | 对照/门槛 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| ASSIST17 | ORCDF, λ=0.5, epoch 6 | 0.784994 | 0.783660 | +0.001334 | **0.711672** | 0.694853 / 历史 aux 0.708857 | 胜 |
| XES3G5M | ORCDF, λ=0.5, epoch 6 | 0.784351 | 0.784877 | -0.000526 | **0.664573** | 0.644924 / 历史 aux 0.664573 | 胜 |
| ASSIST09 | 修正后 SVGCD, λ=0.5, epoch 4 | 0.768438 | 0.768647 | -0.000209 | **0.677355** | 修正基线 0.665268 / 历史 aux 0.670806 | 胜 |

ASSIST09 的 validation 选择同样通过双守门：AUC 从 0.766863 到 0.767156，weighted holdout DOA 从 0.674535 到 0.687123，非加权 holdout DOA 从 0.701466 到 0.723644。三组 test 中最大 AUC 损失仅 0.000526。该结果支持的主张是：共享 mastery monotonic auxiliary 能跨 ORCDF、SVGCD，在三个数据集一致提高严格留出诊断排序，同时保持预测 AUC。

## 完整模型路线：2/3 达标

| 数据集 | 冻结配置 | test overall AUC | test zero AUC | 目标 | 结论 |
|---|---|---:|---:|---:|---|
| ASSIST17 | v2 base, full batch, dim 64, 300 epochs | 0.786333 | **0.783969** | >0.7808 | 胜 |
| MOOCRadar | hybrid+mono+UKC, student minibatch 64, dim 64, 30 epochs | 0.929329 | **0.946124** | >0.9454 | 胜；overall 距 0.9300 约 0.000671 |

这两项可作为“严格未覆盖知识切片仍具竞争力”的补充，但当前不能声称完整模型已在三个数据集胜出。XES3G5M base 长训因插件路线先达到 3/3 而未启动；corrected-support 目前只有单元测试与 smoke 证据，不作为性能结论。

## 数据集取舍与负面记录

- NIPS34 是两条路线的备用集；触发全局停止后未做新一轮正式复现，不能把历史结果混入本轮计数。
- MOOCRadar 不进入插件主搜索：其概念与题目接近一一对应，历史 ORCDF auxiliary 出现优化爆炸；但它在完整模型路线成功。
- ASSIST09 不进入完整模型追加搜索：它是稀疏边界集，且插件路线已成功。
- 所有失败 attempt 均保留。ASSIST09 validation DOA 的 `attempt-001` 仅因 shell 多行参数错误失败，正确结果在 `attempt-002`；没有读取 test。

## 已确认的实现问题

- r18 的 `plugin-decouple` 新增 `ukc_gate`，但历史 optimizer 没有包含该参数；`decouple=base`、`both=aux` 的完全相同结果是 no-op，不应作为机制有效性证据。
- r22 所谓“密度”来自归一化图行和；该特征恒定为 `log(2)`。历史 `--v2-dual-graph-adaptive` 只保留兼容，不再称密度门；新 support-adaptive 使用去自环后的 UKC 可达比例，并保证系数为 0 时数值等价于 Mo1。
- 历史 SVGCD 三阶段训练存在阶段间残留梯度，OneCycleLR 也没有按真实 optimizer step 更新。本轮 ASSIST09 的 λ=0 与 λ=0.5 均在修正后重跑，未混用旧 SVGCD 指标。
- 历史 ORCDF 将固定 scale 表示为不优化的 Parameter，且插件 gate 可能漏入 optimizer；本轮改为固定 buffer，并显式注册需要学习的插件参数。

## 协议边界与复现说明

插件 test 由冻结 selection、数据/holdout 哈希和不可覆盖 claim 绑定，DOA 仅从一次评估生成的 cache 计算。XES3G5M、ASSIST09 以及完整模型两项的外层 runner 均未预先读取 test。ASSIST17 插件实验存在一项已知偏差：外层 runner 在 claim 前将 `test.csv` 列为哈希输入；模型 test 推理仍只执行一次，后续 DOA 也未重读 test，但该项不算完美的 test-once 审计样本。

GPU 策略后经用户授权改为优先空闲卡、显存低于 50% 时可共享。ASSIST09 使用当时最空闲的 GPU 3，任务峰值显存 2856 MiB，无 OOM。vendor 的 ORCDF/SVGCD 快照缺少许可证，仅限远端内部实验，不得 push、公开或再分发。
