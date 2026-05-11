# Experiment 59: parallel local context readout adapter

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 59: parallel local context readout adapter
  - 分支: `exp/parallel-local-context-readout`
  - 做法:
    - 保留实验 51 当前主线全部配置
    - 不再把 local context 回写主 `student_state`
    - 改为从目标题相关概念的 `TKC/UKC` 局部状态构造 target-conditioned attention，再走一条并联 readout residual 分支，直接加到 `cognitive_logits`
  - 工程验证:
    - 远端 `python -m unittest tests.test_decoupled_cdm` 通过
    - 但 `2 epoch + max_rows=5000` smoke 在验证阶段触发 CUDA device-side assert；根因表现为 BCE 输入超出 `[0, 1]`
    - 修过一轮输入维度与 `nan_to_num`/无效行归零后，smoke 仍然不稳定
  - 结论:
    - 这不是“指标略差但可继续调参”的情况，而是当前实现本身在数值上就不稳，尚未达到可比较 overall 指标的最小门槛
    - 不继续沿这版 parallel local context readout 实现扩线，也不进入 rescue sweep
    - 若以后还要 revisit 更大一级 local-context 模块，优先先加显式幅度约束或更保守的 mixture 结构，再决定是否值得进入正式比较
