# 标准–留出联合无回退插件实验设计

## 背景与当前判定

现有正式结果全部来自 student–concept holdout 数据。按旧规则，插件允许相对同骨干 λ=0 的 test AUC 下降不超过 0.002，因此 ASSIST17、XES3G5M、ASSIST09 计为 3/3。新目标更严格：同一数据集的一套插件超参数必须在标准划分和留出划分都不降低 AUC，同时严格留出 holdout DOA 保持正增益。

当前留出 test 状态为：ASSIST17 AUC 差值 +0.001334、holdout DOA 0.711672，已经满足新目标；XES3G5M AUC 差值 -0.000526、holdout DOA 0.664573；ASSIST09 AUC 差值 -0.000209、holdout DOA 0.677355。后两者需要重新选择插件配方。标准划分尚未运行当前插件，不能预判升降。

## 目标与非目标

每个数据集允许选择不同配方，但同一数据集在 standard 与 holdout 上必须使用相同的 `aux_weight`、difficulty detach、warm-up、初始化方式和学习率倍率。两个划分分别训练模型权重，因为其训练数据不同。

正式成功必须同时满足：

1. standard test AUC 不低于修正后同骨干 λ=0；
2. holdout test AUC 不低于修正后同骨干 λ=0；
3. holdout weighted DOA 不低于同骨干 λ=0；
4. holdout DOA 严格提升，并保持论文主表门槛：ASSIST17 >0.708857、XES3G5M ≥0.664573、ASSIST09 >0.670806。

外部模型仅作横向参考，不设零回退硬门。标准随机划分的普通 DOA 可记录，但不进入成功判定。完整模型路线不属于本轮插件无回退搜索；插件路线完成后只补 ASSIST17、MOOCRadar 标准结果，不据外部最强模型反复调参。

## 联合 validation 选择

每个配方同时生成 standard 与 holdout 两套 validation 记录。修正后的 λ=0 必须在两套数据上按同一代码重新复现；SVGCD 历史标准结果因三阶段梯度与 OneCycleLR 问题只作参考。

配方只有同时满足以下条件才可冻结：

- `standard_valid_auc_delta >= 0`；
- `holdout_valid_auc_delta >= 0`；
- `holdout_valid_weighted_doa_delta >= 0`；
- `holdout_valid_doa_delta > 0`。

可行配方先按 `min(standard_valid_auc_delta, holdout_valid_auc_delta)` 降序，再按 holdout DOA 差值、较早 epoch 排序。只有最小 AUC 差值达到 +0.001，才直接进入 test；这 0.001 是防止 test 轻微反转的 validation 安全余量，不是最终论文门槛。未达到安全余量时，不提前打开 test，而是继续预注册的优化或切换保证型适配器。

标准划分没有 holdout assignment；新增 joint selector 通过稳定 `recipe_id` 合并两份 manifest、标准 validation AUC、留出 validation AUC/DOA。现有 holdout selector 保持兼容，不改变已归档结果。

## 搜索与停止顺序

首批数据集和骨干固定为 ASSIST17/ORCDF、XES3G5M/ORCDF、ASSIST09/修正 SVGCD。全部使用 seed 42；数据原有 split 不改变。

1. 先验证当前 λ=0.5 配方的两套 validation；不打开新 test。
2. 失败数据集搜索 `aux_weight ∈ {0.05, 0.1, 0.25, 0.5}`。
3. 对联合 validation 最好的两个权重分别测试 legacy 与 `detach_item_difficulty`。
4. 仍无安全余量时加入前 20% epoch 线性 warm-up。
5. 再从对应 λ=0 最佳 checkpoint 微调，学习率倍率为 `{0.25, 1.0}`。
6. 若共享骨干配方仍不可行，停止该族搜索并进入 prediction-invariant adapter；不扩展更多无解释网格。

每个数据集一旦冻结配方，只允许 standard test 和 holdout test 各评估一次。若最终 test 仍出现 AUC 负差值，该数据集记失败，不利用 test 数值继续选择同一 campaign 的其他配方；后续结构修订必须作为新的、显式披露已见 test 的 campaign。

## 保证型 prediction-invariant adapter

保证型方案从匹配的 λ=0 checkpoint 出发，冻结骨干、预测头及所有影响预测概率的参数，新增独立 mastery adapter。adapter 仅接收冻结的学生表示与题目/概念信息，并只承载 mastery monotonic auxiliary；预测 API 始终走原 λ=0 分支，DOA API 使用 adapter mastery。

该边界产生可测试的不变量：在相同 checkpoint、输入和 eval 模式下，启用 adapter 前后的 prediction tensor 必须逐元素相同，因而标准与留出 AUC/ACC/RMSE 数值等价。adapter 是否成功只由 holdout validation DOA 决定；若无法提升 DOA，则该数据集失败，不声称无回退增强。

共享骨干调参仍是首选，因为它已经展示较强 DOA；adapter 是无法取得联合 AUC 余量时的结构保证，不与共享骨干结果静默混表。

## 数据隔离与 test-once

新增标准协议类型 `standard`，其 selection/claim 不要求 holdout assignment 哈希；留出协议继续要求该哈希。claim identity 必须包含数据集、协议类型、配方、checkpoint、id map、数据哈希和 route commit，避免 standard 与 holdout 冲突。

训练和 validation runner 不得读取真实 test。冻结 joint selection 后，standard 与 holdout evaluator 分别先原子创建不可覆盖 claim，再读取对应 test；DOA 从 holdout test 的一次评估 cache 计算。ASSIST17 旧结果存在 claim 前预哈希 test 的偏差，新配置若重新评估必须使用修正协议，旧 claim 和偏差记录不得删除。

## 验证、资源和产物

标准库 `unittest` 至少覆盖：recipe 跨划分绑定、AUC/DOA 四重守门、+0.001 安全余量、协议类型隔离、标准 claim 不需要 holdout 哈希、留出 claim 仍强制该哈希、prediction-invariant 数值等价、冻结骨干无梯度、adapter 参数进入 optimizer，以及 validation/test 分发和 test-once。

每个骨干执行 1-epoch synthetic CPU smoke；修改训练路径时再执行一次 GPU smoke。GPU 优先使用空闲卡，无空闲卡时可使用显存低于 50% 的卡；每张卡独立 `flock`，OOM 只判当前 attempt 失败，不静默修改 batch size。

新资产放在 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-joint-splits-20260710/`。代码、设计和中文结果摘要提交到远端分支并生成 Git bundle；数据、日志、checkpoint、预测、mastery 和 vendor 代码不得 push。最终摘要必须分别列出共享骨干配方、保证型 adapter、失败数据集和已经见过 test 的历史边界。
