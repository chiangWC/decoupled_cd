# Experiment 54: Q-conditioned local mastery readout revisit

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 54: Q-conditioned local mastery readout revisit
  - 分支: `exp/q-conditioned-local-mastery-readout`
  - 提交: `aeb42b7`
  - 做法:
    - 不再只读全局 `student_state`
    - 对每个目标交互，只 gather 该题 Q mask 命中的概念
    - 对每个目标概念，基于 `tkc_state / ukc_state / concept_embedding / seen_flag / difficulty` 共享打分
    - 用几何均值式的 concept aggregation 形成题目级 local mastery logit
    - 在原有 `cognitive_logits` 外加 zero-init gate: `old_logit + gate * local_mastery_logit`
  - 工程验证:
    - 初版按 “交互数 x 全概念数” 展开局部状态导致远端正式训练 OOM
    - 改成只对命中概念做 gather 后，单测与 smoke 恢复正常
  - 结果:
    - smoke:
      - `max_rows=2000`, `epoch=1` 能正常训练并产出 summary
    - `seed=2024`:
      - `AUC 0.761978`
      - `ACC 0.726674`
      - `RMSE 0.428573`
      - `Brier 0.183675`
      - `ECE 0.046842`
  - 相对实验 51 `seed=2024`:
    - `AUC -0.002073`
    - `ACC -0.001998`
    - `RMSE +0.000278`
    - `Brier +0.000238`
    - `ECE -0.003823`
  - 切片:
    - `concept_count=2`: `AUC 0.749767`, `ACC 0.717925`, `ECE 0.060700`
    - `concept_count=3`: `AUC 0.698639`, `ACC 0.682171`, `ECE 0.082560`
    - `concept_count=4+`: `AUC 0.731486`, `ACC 0.680441`, `ECE 0.125002`
    - `none_seen`: `AUC 0.808718`, `ACC 0.800362`, `ECE 0.112099`
  - 结论:
    - 这次复访已经不是实验 35 那种“局部概念状态均值 residual”，而是更接近主 readout 的逐概念打分再聚合版本；即便如此，overall `AUC/ACC` 仍明显不成立，切片上也没有出现足够强的多知识点 clean win
    - 不继续沿这条 local mastery main-readout 设计扩线
    - 若后续再访，必须带着更强的概念交互假设或更明确的聚合归纳偏置，而不是再重复“逐概念打分 + 简单聚合”框架
