# Unified Mastery A0 里程碑（2026-07-12）

## 结论

本轮仅使用 seed 42、split seed 2024 和验证集，冻结主 cohort 为
MOOCRadar / ASSIST17 / XES3G5M。三者 A0 均未超过审计后的最强外部
comparator，因此本里程碑是可复现的 A0 起点，不是性能达标声明。未解析任何
test 标签；数据审计只记录 test 文件内容哈希。

campaign 根目录为
'/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712'
（下文记作 $ROOT）。A0 训练证明绑定执行 route
'8e213c01c62802448494ce958faf98786daa0552'，架构 fingerprint 为
'ef5d6e82e32d095248a362ab01b6a768bbf47785fe34196dee5b6ddd0ece2131'。
受信 proof replay/finalization 修复提交为 '228b819'；它没有重跑 GPU，而是
校验七个不可变 proof 的连续 counter、controller、route、baseline、recipe 与
原始 SHA-256 后，原子重算最终 recipe。

## 冻结指标与 comparator 差距

| 数据集 | 最终 recipe | A0 standard AUC | A0 holdout AUC | A0 exact-zero AUC | 最强 comparator | standard gap | holdout gap | zero gap |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| ASSIST17 | 1 | 0.7750000504027643 | 0.7767601074356755 | 0.7773152631590659 | ORCDF | -0.009420161013223516 | -0.007357438623234347 | -0.004733039968172759 |
| MOOCRadar | 2 | 0.9240430764333312 | 0.922193551048275 | 0.9326811465612022 | SVGCD | -0.005598735406510014 | -0.0027485772022497468 | -0.0005960778315650117 |
| XES3G5M | 0 | 0.7786990641562951 | 0.7702064491415873 | 0.7692902450501526 | ORCDF | -0.01075577682068618 | -0.010791141320400599 | -0.007524454042345585 |

A0 的 standard ordinary/weighted DOA 分别为：ASSIST17
'0.6790287156434344 / 0.6782052477686286'，MOOCRadar
'0.49350790531341393 / 0.6650838940465037'，XES3G5M
'0.45584415584415583 / 0.7452830188679245'。

最强 comparator 的精确值与审计来源如下：

| 数据集 | split/metric | 模型 | 值 | 来源 |
|---|---|---|---:|---|
| ASSIST17 | standard/auc | ORCDF | 0.7844202114159878 | $ROOT/audit/comparator-jobs/ORCDF/ASSIST17/standard/attempt-001/baseline-rows.json |
| ASSIST17 | holdout/auc | ORCDF | 0.7841175460589098 | $ROOT/audit/comparator-jobs/ORCDF/ASSIST17/holdout/attempt-001/baseline-rows.json |
| ASSIST17 | holdout/zero_auc | ORCDF | 0.7820483031272386 | $ROOT/audit/comparator-jobs/ORCDF/ASSIST17/holdout/attempt-001/baseline-rows.json |
| MOOCRadar | standard/auc | SVGCD | 0.9296418118398412 | $ROOT/audit/comparator-jobs/SVGCD/MOOCRadar/standard/attempt-001/baseline-rows.json |
| MOOCRadar | holdout/auc | SVGCD | 0.9249421282505248 | $ROOT/audit/comparator-jobs/SVGCD/MOOCRadar/holdout/attempt-001/baseline-rows.json |
| MOOCRadar | holdout/zero_auc | SVGCD | 0.9332772243927672 | $ROOT/audit/comparator-jobs/SVGCD/MOOCRadar/holdout/attempt-001/baseline-rows.json |
| XES3G5M | standard/auc | ORCDF | 0.7894548409769813 | $ROOT/audit/comparator-jobs/ORCDF/XES3G5M/standard/attempt-001/baseline-rows.json |
| XES3G5M | holdout/auc | ORCDF | 0.7809975904619879 | $ROOT/audit/comparator-jobs/ORCDF/XES3G5M/holdout/attempt-001/baseline-rows.json |
| XES3G5M | holdout/zero_auc | ORCDF | 0.7768146990924982 | $ROOT/audit/comparator-jobs/ORCDF/XES3G5M/holdout/attempt-001/baseline-rows.json |

## Recipe 与不可变证明

- ASSIST17 recipe 1：concept_dim=64, epochs=80, learning_rate=0.002,
  mastery_loss_weight=0.1, patience=5, student_batch_size=128,
  training_mode=student_recompute_minibatch, weight_decay=0.0；recipe SHA-256
  '938db4c268aee03e3ef89473fb70176de47d111817c79a18ee8b3e2071084425'；
  proof counter 2，proof SHA-256
  'd56bbb415d52f7182ebbb57616fb03c32ca72d079fee5ae91b077638ba418095'。
- MOOCRadar recipe 2：concept_dim=256, epochs=30, learning_rate=0.001,
  mastery_loss_weight=0.1, patience=5, student_batch_size=64,
  training_mode=student_recompute_minibatch, weight_decay=0.0；recipe SHA-256
  '3deb9840be1f5031e92575876f382a89cefe525d5464c259d8575d7ed9bd97aa'；
  proof counter 5，proof SHA-256
  '3cd57fe3477ab7b1ea2dc88b20b366dc42fc927966204e62df1fbbfed531d319'。
- XES3G5M recipe 0：concept_dim=64, epochs=30, learning_rate=0.001,
  mastery_loss_weight=0.1, patience=5, student_batch_size=64,
  training_mode=student_recompute_minibatch, weight_decay=0.0；recipe SHA-256
  '1b340edcb1d7d6ba50f9a9e06f8c1c164120c470ea2038110875cccbfe9abcdc'；
  proof counter 6，proof SHA-256
  'f18639f0058a7f7adfa5870f252f7ef6ee2ac8c35dd7c2d4a8647609519ca0c0'。

七个尝试均未通过外部 overall guard；下表完整保留失败结果。星号表示虽未达
comparator、但在该数据集内部按 min(standard gap, holdout gap)、zero gap、
earlier counter 排名入选。

| counter | 数据集/recipe | standard AUC (gap) | holdout AUC (gap) | zero AUC (gap) |
|---:|---|---:|---:|---:|
| 1 | ASSIST17/r0 | 0.7742460315939887 (-0.010174179821999085) | 0.776434841372969 (-0.007682704685940789) | 0.7769512845109965 (-0.005097018616242122) |
| 2 | ASSIST17/r1* | 0.7750000504027643 (-0.009420161013223516) | 0.7767601074356755 (-0.007357438623234347) | 0.7773152631590659 (-0.004733039968172759) |
| 3 | MOOCRadar/r0 | 0.9134173958220229 (-0.01622441601781832) | 0.9126648350594619 (-0.012277293191062855) | 0.9192002442906617 (-0.014076980102105452) |
| 4 | MOOCRadar/r1 | 0.922367207693493 (-0.007274604146348174) | 0.9208522343321219 (-0.004089893918402865) | 0.9303758150090654 (-0.0029014093837017585) |
| 5 | MOOCRadar/r2* | 0.9240430764333312 (-0.005598735406510014) | 0.922193551048275 (-0.0027485772022497468) | 0.9326811465612022 (-0.0005960778315650117) |
| 6 | XES3G5M/r0* | 0.7786990641562951 (-0.01075577682068618) | 0.7702064491415873 (-0.010791141320400599) | 0.7692902450501526 (-0.007524454042345585) |
| 7 | XES3G5M/r1 | 0.5808558228059484 (-0.20859901817103288) | 0.5430764016959158 (-0.23792118876607216) | 0.5029000498411811 (-0.27391464925131714) |

finalized proof registry SHA-256 为
'0f0375b62a99494f00f0fb125e2042946ea54dd2a61a82463032842c54e06627'。
XES r1 的 full-batch fallback（epochs=3000, patience=50）负结果保留于
counter 7，没有被用于 A1 继承。

## 数据集审计、comparator 审计与 smoke

数据集 audit SHA-256 为
'bbc0ba03ad0d8182b111b91c8ddce50694ab23abf9a840337923fbfb79894192'。
排除项：

- ASSIST09：standard exact-zero 仅 583，低于 1000 门槛。
- NIPS34：standard 与 holdout exact-zero 均为 0，只能作为 partial-only。
- Junyi、EdNet-ICDM：缺失所需 Q/holdout 资产，状态为 provisional。

comparator audit SHA-256 为
'071b5df25d8641fdc249a9a56175961025b3deeba36a4a471640563ef5d0b181'；
KaNCD、ORCDF、SVGCD 在三个 eligible 数据集的 standard/holdout 共 18 个
预声明 job 均一次完成，接受 36 个 metric rows、拒绝 0 个本轮 rows，无 OOM、
无 retry、无 batch 改写。ORCDF 的确定性内存 gate 全部判定可行；CPU job 的
实际 peak RSS 未由 outer runner 返回，因此只能报告解析/图张量下界，不能声称
已测得 CPU 峰值。

唯一 A0 架构 smoke 位于 $ROOT/smoke/a0/attempt-001，GPU UUID 为
'GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0'，peak GPU memory 为
'0.019381046295166016 GiB'，mastery shape 为 [3, 3]，最终 BCE 为
'0.7542915344238281'，参数量 133756。条件 simplex 构造保证
guess + slip < 1。没有 A1 smoke，也没有第二次 A0 smoke。

## 冻结哈希

- provisional registry SHA-256：
  '12e5b991856909b6c214a1310c909ffa9008f10f1170f31bf34e23cce96b20dc'
- cohort canonical SHA-256：
  '6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5'
- $ROOT/cohort.json 文件 SHA-256：
  '60644ed4b7d4acfb47b328935b1add600e17fa0060c4df08e66e65ca75eb9590'

freeze 与带完整 A0/baseline/dataset 输入的 verify 均通过；冻结排名为
MOOCRadar、ASSIST17、XES3G5M。
