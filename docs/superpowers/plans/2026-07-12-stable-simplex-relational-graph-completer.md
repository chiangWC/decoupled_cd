# Stable Simplex Relational Graph Completer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立数值可信的 A0v4，并在同一统一架构下实现、验证 A2 Evidence-Relation Graph Completer，只有通过冻结 validation 门后才打开外部比较与 test。

**Architecture:** version 4 用固定 `2^-20` cognitive mass 下界替换 v3 普通 softmax；A2 只替换 missing-mastery completer，以 train-only 学生–概念证据二部图、正负关系消息传递和共享双线性解码器补全未观测 mastery。A0v4 与 A2 共用 evidence estimator、NeuralCDM monotone decoder、行为通道、划分协议及冻结 cohort。

**Tech Stack:** Python、PyTorch、标准库 `unittest`、现有不可覆盖 campaign runner/controller、远端 Conda `decoupled_cd`、CUDA GPU。Python/Torch/CUDA 精确版本由 campaign runner 记录。

## Global Constraints

- 远端唯一可写工作树是 `/home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model`；不得修改 `decoupled_cd_v2`，不得 push。
- Git 身份固定为 `chiangWC <215551297+chiangWC@users.noreply.github.com>`；每个里程碑提交并在 campaign root 生成 Git bundle。
- campaign root 固定为 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712/`；数据、日志、checkpoint、预测和 mastery 不进 Git。
- 仅使用 seed `42` 与既有 split seed `2024`；优先空闲 GPU，也可使用显存占用低于一半的卡，同卡 `flock` 排他；OOM 不静默改 batch size。
- primary cohort 固定为 `MOOCRadar / ASSIST17 / XES3G5M`，cohort SHA-256 为 `6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5`。
- A0v4 recipe 固定为 MOOCRadar r2、ASSIST17 r1、XES3G5M r0；不得重新选择 recipe。
- version 4 manifest 固定 `behavior_model=conditional-simplex-floor-2m20`；v3 manifest/checkpoint 必须拒绝，历史结果不得重写。
- A2 固定两层传播、关系阈值 `0.5`、reliability cap `20`、每 epoch 确定性遮蔽 `20%` observed edges；不得增加 ID decoder bypass、残差 completer、MoE、第二 mastery/head 或数据集分支。
- A2 相对 A0v4 门：三个数据集 standard overall 均不降、holdout overall 均不降、zero AUC 至少 `2/3` 严格提升且至少一个 delta `>=0.001`；DOA 为软指标。
- 相对门失败则 test 保持关闭并登记失效机制；相对门通过后才重建外部门，三个数据集外部门均通过后才允许一次 test。

---

### Task 1: Version-4 架构身份与稳定行为通道

**Files:**
- Modify: `models/unified_v2_spec.py`
- Modify: `models/unified_v2_components.py`
- Modify: `tests/test_unified_v2_spec.py`
- Modify: `tests/test_unified_v2_components.py`

**Interfaces:**
- Produces: `UnifiedArchitectureSpec(completion: Literal["prior", "evidence-relational-graph"])` 的 canonical v4 manifest/fingerprint。
- Produces: `ConditionalSimplexBehaviorModel.COGNITIVE_FLOOR = 2.0 ** -20`，返回现有 `BehaviorState`。

- [ ] **Step 1: 写 manifest、旧版本拒绝与极端数值测试**

```python
def test_v4_manifest_and_v3_rejection(self):
    prior = UnifiedArchitectureSpec(completion="prior")
    graph = UnifiedArchitectureSpec(completion="evidence-relational-graph")
    self.assertEqual(prior.version, 4)
    self.assertEqual(prior.behavior_model, "conditional-simplex-floor-2m20")
    self.assertEqual(graph.manifest()["modules"], "m1-evidence-relational-graph-m3-m4")
    old = dict(prior.manifest(), version=3, behavior_model="conditional-simplex")
    with self.assertRaisesRegex(ValueError, "version 4"):
        UnifiedArchitectureSpec.from_manifest(old)

def test_stable_simplex_has_positive_extreme_derivative(self):
    for dtype in (torch.float32, torch.float64):
        model = ConditionalSimplexBehaviorModel(2, 2, 2).to(dtype=dtype)
        with torch.no_grad():
            model.out[-1].weight.zero_()
            model.out[-1].bias.copy_(torch.tensor([1e4, 1e4, -1e4], dtype=dtype))
        cognitive = torch.tensor([0.25], dtype=dtype, requires_grad=True)
        state = model(cognitive, torch.tensor([0]), torch.tensor([0]))
        derivative, = torch.autograd.grad(state.probs.sum(), cognitive)
        self.assertGreaterEqual(float(state.cognitive_weight), 2.0 ** -20)
        self.assertLess(float(state.guess_probs + state.slip_probs), 1.0)
        torch.testing.assert_close(derivative, state.cognitive_weight)
```

- [ ] **Step 2: 运行测试确认当前 v3 失败**

Run: `python -m unittest tests.test_unified_v2_spec tests.test_unified_v2_components -v`

Expected: FAIL，因为 manifest 是 v3，float32 极端 softmax 的 cognitive weight 为零。

- [ ] **Step 3: 实现 canonical v4 spec 和稳定 simplex**

```python
@dataclass(frozen=True)
class UnifiedArchitectureSpec:
    mastery_estimator: Literal["evidence-parameter"] = "evidence-parameter"
    completion: Literal["prior", "evidence-relational-graph"] = "prior"
    cognitive_decoder: Literal["neuralcdm-monotonic"] = "neuralcdm-monotonic"
    behavior_model: Literal["conditional-simplex-floor-2m20"] = "conditional-simplex-floor-2m20"
    mastery_output: Literal["student-concept"] = "student-concept"
    version: int = 4
```

```python
logits = self.out(features)
raw_cognitive = logits.softmax(dim=-1)[..., 2]
cognitive_weight = self.COGNITIVE_FLOOR + (1.0 - self.COGNITIVE_FLOOR) * raw_cognitive
behavior_mass = 1.0 - cognitive_weight
guess = behavior_mass * torch.sigmoid(logits[..., 0] - logits[..., 1])
slip = behavior_mass - guess
probs = guess + cognitive_weight * cognitive_probs
```

同步更新 `from_manifest()` 精确 schema、version 错误文本和 `modules` 映射；删除 `lowrank` 作为可构造 v4 completion，旧 checkpoint 因 manifest/fingerprint 不匹配而拒绝。

- [ ] **Step 4: 回归并提交**

Run: `python -m unittest tests.test_unified_v2_spec tests.test_unified_v2_components -v`

Expected: PASS；两种 dtype 均满足正导数和 `guess+slip<1`。

```bash
git add models/unified_v2_spec.py models/unified_v2_components.py tests/test_unified_v2_spec.py tests/test_unified_v2_components.py
git commit -m "fix: stabilize unified behavior simplex"
```

### Task 2: 证据关系图的数据语义与确定性遮蔽

**Files:**
- Create: `models/evidence_relation_graph.py`
- Create: `tests/test_evidence_relation_graph.py`

**Interfaces:**
- Produces: `RelationGraphBatch(positive_weight, negative_weight, reconstruction_mask, target)`，shape 均为 `[S,K]`。
- Produces: `build_relation_graph(evidence, *, epoch, training, mask_fraction=0.2, reliability_cap=20.0)`。
- Produces: `node_summary_features(evidence, q_matrix) -> (Tensor[S,3], Tensor[K,4])`。

- [ ] **Step 1: 写 relation、可靠性、特征与无泄漏测试**

```python
def test_relation_graph_and_epoch_mask(self):
    evidence = torch.zeros(10, 5, 2)
    evidence[..., 0] = 4
    evidence[..., 1] = torch.arange(50).reshape(10, 5).remainder(5)
    left = build_relation_graph(evidence, epoch=7, training=True)
    right = build_relation_graph(evidence, epoch=7, training=True)
    torch.testing.assert_close(left.reconstruction_mask, right.reconstruction_mask)
    self.assertEqual(int(left.reconstruction_mask.sum()), 10)
    self.assertFalse(bool(left.positive_weight[left.reconstruction_mask].any()))
    self.assertFalse(bool(left.negative_weight[left.reconstruction_mask].any()))
    expected = (evidence[..., 1] + 1.0) / (evidence[..., 0] + 2.0)
    torch.testing.assert_close(left.target, expected)

def test_node_features_have_fixed_non_id_shapes(self):
    students, concepts = node_summary_features(evidence, q_matrix)
    self.assertEqual(tuple(students.shape), (10, 3))
    self.assertEqual(tuple(concepts.shape), (5, 4))
```

- [ ] **Step 2: 运行测试确认模块缺失**

Run: `python -m unittest tests.test_evidence_relation_graph -v`

Expected: FAIL with `ModuleNotFoundError`。

- [ ] **Step 3: 实现图 batch、train-only 特征与确定性 hash mask**

```python
@dataclass(frozen=True)
class RelationGraphBatch:
    positive_weight: torch.Tensor
    negative_weight: torch.Tensor
    reconstruction_mask: torch.Tensor
    target: torch.Tensor

def _deterministic_mask(observed, epoch, fraction):
    flat_ids = torch.arange(observed.numel(), device=observed.device,
                            dtype=torch.int64).reshape_as(observed)
    hashed = (flat_ids * 1103515245 + (epoch + 42) * 12345) & 0x7fffffff
    rank = hashed.masked_fill(~observed, torch.iinfo(torch.int64).max).flatten().argsort()
    count = int(observed.sum().item() * fraction)
    result = torch.zeros_like(observed)
    if count:
        result.flatten()[rank[:count]] = True
    return result
```

`build_relation_graph()` 从 `evidence[..., :2]` 计算平滑 target、observed、`min(attempts,20)/20`；target `>=0.5` 为 positive。训练时从正负消息权重清除 reconstruction mask；推理要求 `epoch=None` 并返回空 mask。特征只聚合 coverage、平滑 accuracy、log attempts 与 Q item frequency，不创建 ID embedding。

- [ ] **Step 4: 回归并提交**

Run: `python -m unittest tests.test_evidence_relation_graph -v`

Expected: PASS；50 条边精确遮蔽 10 条且不存在 target-edge leakage。

```bash
git add models/evidence_relation_graph.py tests/test_evidence_relation_graph.py
git commit -m "feat: build deterministic evidence relation graphs"
```

### Task 3: 两层关系消息传递与共享图解码器

**Files:**
- Modify: `models/evidence_relation_graph.py`
- Modify: `tests/test_evidence_relation_graph.py`

**Interfaces:**
- Produces: `EvidenceRelationGraphCompleter(num_students, num_concepts, hidden_dim)`。
- Consumes: full train-only `evidence: Tensor[S,K,*]`、`q_matrix: Tensor[E,K]`、可选 `student_ids`、`epoch`、`training`。
- Produces: `GraphCompletionState(mastery, reconstruction_mask, target, student_state, concept_state)`。

- [ ] **Step 1: 写 shape、梯度、subset 和 no-ID-bypass 测试**

```python
def test_completer_supports_noncontiguous_subset_without_id_embedding(self):
    model = EvidenceRelationGraphCompleter(4, 3, hidden_dim=5)
    self.assertFalse(any(isinstance(m, torch.nn.Embedding) for m in model.modules()))
    state = model(evidence, q_matrix, student_ids=torch.tensor([3, 1]),
                  epoch=0, training=True)
    self.assertEqual(tuple(state.mastery.shape), (2, 3))
    self.assertEqual(tuple(state.student_state.shape), (2, 5))
    state.mastery.sum().backward()
    self.assertTrue(all(p.grad is not None for p in model.parameters()))

def test_positive_and_negative_relations_change_predictions(self):
    model = EvidenceRelationGraphCompleter(3, 2, hidden_dim=4)
    positive_evidence = torch.tensor([[[4., 4.], [4., 4.]],
                                      [[4., 4.], [4., 4.]],
                                      [[4., 4.], [4., 4.]]])
    negative_evidence = torch.tensor([[[4., 0.], [4., 0.]],
                                      [[4., 0.], [4., 0.]],
                                      [[4., 0.], [4., 0.]]])
    q = torch.eye(2)
    positive = model(positive_evidence, q, None, None, False).mastery
    negative = model(negative_evidence, q, None, None, False).mastery
    self.assertFalse(torch.equal(positive, negative))
```

- [ ] **Step 2: 运行测试确认 completer 缺失**

Run: `python -m unittest tests.test_evidence_relation_graph -v`

Expected: FAIL，因为 `EvidenceRelationGraphCompleter` 尚未定义。

- [ ] **Step 3: 实现 relation-specific 聚合和 bilinear decoder**

```python
class RelationMessageLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.positive = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.negative = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, adjacency, source):
        positive, negative = adjacency
        degree = (positive + negative).sum(-1, keepdim=True).clamp_min(1.0)
        message = (positive @ self.positive(source)
                   + negative @ self.negative(source)) / degree
        return self.norm(F.relu(message))
```

```python
class EvidenceRelationGraphCompleter(nn.Module):
    def __init__(self, num_students, num_concepts, hidden_dim):
        super().__init__()
        self.student_encoder = nn.Linear(3, hidden_dim)
        self.concept_encoder = nn.Linear(4, hidden_dim)
        self.c2s = nn.ModuleList(RelationMessageLayer(hidden_dim) for _ in range(2))
        self.s2c = nn.ModuleList(RelationMessageLayer(hidden_dim) for _ in range(2))
        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.student_bias = nn.Linear(3, 1, bias=False)
        self.concept_bias = nn.Linear(4, 1, bias=False)

    def forward(self, evidence, q_matrix, student_ids=None, epoch=None, training=False):
        graph = build_relation_graph(evidence, epoch=epoch, training=training)
        sf, cf = node_summary_features(evidence, q_matrix)
        students, concepts = self.student_encoder(sf), self.concept_encoder(cf)
        for c2s, s2c in zip(self.c2s, self.s2c):
            students = c2s((graph.positive_weight, graph.negative_weight), concepts)
            concepts = s2c((graph.positive_weight.T, graph.negative_weight.T), students)
        logits = students @ self.bilinear @ concepts.T
        logits = logits + self.student_bias(sf) + self.concept_bias(cf).T
        selected = slice(None) if student_ids is None else student_ids
        return GraphCompletionState(torch.sigmoid(logits[selected]),
                                    graph.reconstruction_mask[selected],
                                    graph.target[selected], students[selected], concepts)
```

同文件定义明确的返回类型：

```python
@dataclass(frozen=True)
class GraphCompletionState:
    mastery: torch.Tensor
    reconstruction_mask: torch.Tensor
    target: torch.Tensor
    student_state: torch.Tensor
    concept_state: torch.Tensor
```

bilinear 用 Xavier uniform 初始化。实现可用等价 sparse op，但不得构造 `[S,K,K]`；关系方向与 degree normalization 必须由测试覆盖。

- [ ] **Step 4: 回归并提交**

Run: `python -m unittest tests.test_evidence_relation_graph -v`

Expected: PASS；参数梯度有限、subset 顺序保持 `[3,1]`、模型没有 `nn.Embedding`。

```bash
git add models/evidence_relation_graph.py tests/test_evidence_relation_graph.py
git commit -m "feat: add relational graph mastery completer"
```

### Task 4: A0v4/A2 模型、loss 与 checkpoint 集成

**Files:**
- Modify: `models/decoupled_cdm.py`
- Modify: `models/unified_decoupled_cdm.py`
- Modify: `models/__init__.py`
- Modify: `trainers/engine.py`
- Modify: `scripts/train.py`
- Modify: `tests/test_unified_v2_training.py`

**Interfaces:**
- Adds `DecoupledForwardOutput.completion_target_mask`、`completion_targets`、`completion_student_state`、`completion_concept_state`。
- Produces: `masked_graph_reconstruction_loss(output)`，只供 A2。
- Adds model forward keyword `completion_epoch: int | None = None`；evaluation 传 `None`，training 传 zero-based epoch。

- [ ] **Step 1: 写 hard assembly、masked loss、旧 checkpoint 和 CLI 测试**

```python
def test_a2_uses_graph_only_for_missing_cells(self):
    a2 = self.model(completion="evidence-relational-graph")
    output = self.forward(a2, self.tensors(), completion_epoch=0)
    mask = output.mastery_observed_mask
    torch.testing.assert_close(output.mastery[mask], output.observed_mastery[mask])
    torch.testing.assert_close(output.mastery[~mask], output.completion_predictions[~mask])
    masked_graph_reconstruction_loss(output).backward()
    self.assertTrue(any(p.grad is not None for p in a2.completer.parameters()))

def test_v3_checkpoint_is_rejected(self):
    model = self.model(completion="prior")
    state = copy.deepcopy(model.state_dict())
    state["_checkpoint_architecture_manifest"] = model._text_tensor('{"version":3}')
    with self.assertRaisesRegex(ValueError, "architecture manifest mismatch"):
        model.load_state_dict(state)

def test_cli_requires_positive_graph_reconstruction_weight(self):
    args = self.parse_and_validate("--model", "unified_v2", "--unified-completion",
        "evidence-relational-graph", "--unified-completion-loss-weight", "1")
    self.assertEqual(args.unified_completion, "evidence-relational-graph")
```

- [ ] **Step 2: 运行集成测试确认失败**

Run: `python -m unittest tests.test_unified_v2_training -v`

Expected: FAIL，因为 graph completion、epoch 和 target fields 尚未接入。

- [ ] **Step 3: 接入模型与专用 reconstruction loss**

```python
def masked_graph_reconstruction_loss(output):
    mask, target = output.completion_target_mask, output.completion_targets
    if output.completion_predictions is None or mask is None or target is None:
        raise ValueError("graph reconstruction outputs are required")
    if not bool(mask.any()):
        raise ValueError("at least one removed graph edge is required")
    return F.binary_cross_entropy(output.completion_predictions[mask], target[mask])
```

`UnifiedDecoupledCDM` 在 `prior` 时保持 `GlobalConceptPriorCompleter`，在 `evidence-relational-graph` 时构造新 completer。A2 forward 先用完整 train-only graph 计算，再按 `student_ids` 取行；observed estimator 与 `assemble_mastery()` 不变。A0v4 graph fields 为 `None`；A2 evaluation 用全部 observed edges 且不产生 reconstruction target。

- [ ] **Step 4: 将 epoch 传入两种训练模式并绑定 checkpoint**

```python
def _train_full_batch_epoch(*, epoch_index: int, **kwargs):
    output = _forward_model(
        model=kwargs["model"], q_matrix=kwargs["tensors"]["q_matrix"],
        concept_graph=kwargs["tensors"]["concept_graph"],
        prerequisite_graph=kwargs["tensors"]["prerequisite_graph"],
        similarity_graph=kwargs["tensors"]["similarity_graph"],
        student_exercise_mask=kwargs["tensors"]["student_exercise_mask"],
        response_matrix=kwargs["tensors"]["response_matrix"],
        student_tkc_mask=kwargs["tensors"]["student_tkc_mask"],
        student_ukc_mask=kwargs["tensors"]["student_ukc_mask"],
        student_concept_evidence=kwargs["tensors"]["student_concept_evidence"],
        exercise_evidence=kwargs["tensors"]["exercise_evidence"],
        target_student_ids=kwargs["tensors"]["interaction_student_ids"],
        target_exercise_ids=kwargs["tensors"]["interaction_exercise_ids"],
        completion_epoch=epoch_index,
    )

def _forward_student_batch(*, model, tensors, batch_indices, epoch_index):
    return _forward_model(
        model=model, q_matrix=tensors["q_matrix"],
        concept_graph=tensors["concept_graph"],
        prerequisite_graph=tensors["prerequisite_graph"],
        similarity_graph=tensors["similarity_graph"],
        student_exercise_mask=tensors["student_exercise_mask"],
        response_matrix=tensors["response_matrix"],
        student_tkc_mask=tensors["student_tkc_mask"],
        student_ukc_mask=tensors["student_ukc_mask"],
        student_concept_evidence=tensors["student_concept_evidence"],
        exercise_evidence=tensors["exercise_evidence"],
        target_student_ids=tensors["interaction_student_ids"][batch_indices],
        target_exercise_ids=tensors["interaction_exercise_ids"][batch_indices],
        use_student_subset=True, completion_epoch=epoch_index,
    )
```

checkpoint 绑定 manifest、fingerprint、completion、graph hidden dim 与两个 loss weights；缺字段、v3 或不匹配均 raise。`scripts/train.py` summary 输出 manifest/fingerprint、`completion_mask_fraction=0.2`、graph hidden dim。

- [ ] **Step 5: 回归并提交**

Run: `python -m unittest tests.test_unified_v2_training tests.test_unified_v2_components tests.test_unified_v2_spec tests.test_evidence_relation_graph -v`

Expected: PASS；full-batch 与 student-recompute 均产生 graph 参数梯度，A0 无 graph 参数。

```bash
git add models/decoupled_cdm.py models/unified_decoupled_cdm.py models/__init__.py trainers/engine.py scripts/train.py tests/test_unified_v2_training.py
git commit -m "feat: integrate graph completion into unified v4"
```

### Task 5: 新 campaign controller、相对门与 test-closed 证明

**Files:**
- Create: `scripts/stable_graph_validation_controller.py`
- Create: `tests/test_stable_graph_validation_controller.py`
- Modify: `scripts/unified_campaign.py`

**Interfaces:**
- Produces CLI: `smoke`、`run-validation`、`replay`、`freeze-a0v4`、`relative-gate`、`external-gate`、`run-test-once`、`status`。
- Consumes现有 verified cohort/audit 和 runner proof helpers；只写 `unified-ergc-r5-20260712`。

- [ ] **Step 1: 写 campaign identity、recipe、五项门和 test-closed 测试**

```python
def test_identity_recipes_and_test_closed(self):
    self.assertEqual(controller.CAMPAIGN_ID, "unified-ergc-r5-20260712")
    self.assertEqual(controller.FROZEN_RECIPES,
                     {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0})
    with self.assertRaisesRegex(ValueError, "test remains closed"):
        controller.issue_attempt(architecture="a2", dataset="ASSIST17", split="test")

def test_relative_gate(self):
    deltas = {
      "ASSIST17": {"standard": 0., "holdout": 0., "zero": .001},
      "MOOCRadar": {"standard": .0001, "holdout": .0001, "zero": .0001},
      "XES3G5M": {"standard": 0., "holdout": 0., "zero": -.0001}}
    self.assertTrue(controller.relative_gate(deltas)["passed"])
    deltas["XES3G5M"]["holdout"] = -1e-12
    self.assertFalse(controller.relative_gate(deltas)["passed"])
```

还要测试 cohort SHA、fingerprint、route commit、proof counter、artifact SHA 任一不匹配均拒绝；relative fail 时 external/test 均拒绝。

- [ ] **Step 2: 运行测试确认 controller 缺失**

Run: `python -m unittest tests.test_stable_graph_validation_controller -v`

Expected: FAIL with `ImportError`。

- [ ] **Step 3: 实现 controller 并复用受信 proof 验证**

```python
CAMPAIGN_ID = "unified-ergc-r5-20260712"
FROZEN_COHORT_SHA256 = "6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5"
FROZEN_RECIPES = {"MOOCRadar": 2, "ASSIST17": 1, "XES3G5M": 0}
ARCHITECTURES = {"a0v4": ("prior", 0.0),
                 "a2": ("evidence-relational-graph", 1.0)}

def relative_gate(deltas):
    overall = all(row["standard"] >= 0 and row["holdout"] >= 0
                  for row in deltas.values())
    zero_wins = sum(row["zero"] > 0 for row in deltas.values())
    margin = any(row["zero"] >= 0.001 for row in deltas.values())
    return {"passed": overall and zero_wins >= 2 and margin,
            "overall_nonregression": overall, "zero_wins": zero_wins,
            "zero_delta_at_least_0.001": margin}
```

从现有 controller 导入 canonical hash、route cleanliness、proof replay、atomic write helpers，或机械提取到 `scripts/unified_campaign.py` 并让旧/new controller 同测；不得新写宽松 shell 解析。

- [ ] **Step 4: controller 回归并提交**

Run: `python -m unittest tests.test_stable_graph_validation_controller tests.test_unified_validation_controller tests.test_unified_campaign -v`

Expected: PASS；没有 relative-pass artifact 时不存在 test issuance path。

```bash
git add scripts/stable_graph_validation_controller.py scripts/unified_campaign.py tests/test_stable_graph_validation_controller.py
git commit -m "feat: gate stable graph validation campaign"
```

### Task 6: 全量回归、一次 GPU smoke 与 A0v4 冻结验证

**Files:**
- Create: `docs/experiments/stable_graph_a0v4_20260712.md`
- Modify only on a demonstrated regression: Tasks 1–5 implementation and matching tests。

**Interfaces:**
- Produces exactly one A0v4 smoke and six validation metrics (3 datasets × standard/holdout)。

- [ ] **Step 1: 跑全量 CPU suite 与 cleanliness gate**

```bash
cd /home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model
conda run -n decoupled_cd python -m unittest discover -s tests -v
git status --short
```

CPU gate 仅允许以下两种结果之一：

1. full discovery 全绿；或
2. full discovery 的唯一失败精确为 `test_sigterm_to_runner_forwards_to_nested_child_process_group`，并且同一 checkout 上立即隔离复跑该测试通过。

任何其他失败、多个失败或隔离复跑失败都阻断 GPU。该窄 flaky policy 是 Task 6 实际采用的 gate；不得把第二种结果表述为 full suite 全绿。通过 gate 后仍要求 worktree clean。

- [ ] **Step 2: 动态选卡并只运行一次 A0v4 smoke**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py smoke --architecture a0v4 --seed 42 --epochs 1 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
```

Expected: `smoke/a0v4/attempt-001` 记录 GPU UUID、峰值显存、route、manifest/fingerprint、mastery shape、有限 loss；再次调用必须拒绝覆盖。

- [ ] **Step 3: 运行三个冻结 A0v4 validation job**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-validation --architecture a0v4 --dataset MOOCRadar --seed 42 --split-seed 2024 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-validation --architecture a0v4 --dataset ASSIST17 --seed 42 --split-seed 2024 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-validation --architecture a0v4 --dataset XES3G5M --seed 42 --split-seed 2024 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
```

Controller 从 `FROZEN_RECIPES` 注入 recipe，不接受 CLI 改 epochs/dim/LR/batch。可让不同 GPU 并行不同数据集，但同 dataset 的 standard/holdout 是不可分割 proof，同卡以 flock 排他。

- [ ] **Step 4: replay、冻结、写报告并归档**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py replay --architecture a0v4
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py freeze-a0v4
git add docs/experiments/stable_graph_a0v4_20260712.md
git commit -m "docs: freeze stable unified v4 baseline"
head=$(git rev-parse --short=12 HEAD)
git bundle create "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712/bundles/a0v4-$head.bundle" HEAD
```

Expected: 三数据集各有唯一 standard/holdout 指标、fingerprint 相同、未访问 test。报告列出 A0v4 vs v3 A0 的 AUC 差值，但 v3 只作历史诊断。

### Task 7: A2 一次 smoke、冻结 validation 与相对门

**Files:**
- Create: `docs/experiments/stable_graph_a2_20260712.md`
- Modify on A2 fail: `docs/experiments/unified_mastery_failure_registry.md`

**Interfaces:**
- Consumes frozen A0v4 proof and identical recipes/cohort。
- Produces exactly one A2 smoke、六个 validation metrics 和 `relative-gate.json`。

- [ ] **Step 1: 只运行一次 A2 smoke**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py smoke --architecture a2 --seed 42 --epochs 1 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
```

Expected: `smoke/a2/attempt-001`；masked edge count > 0，graph 梯度有限，observed mastery hard assembly 相等，未 OOM。

- [ ] **Step 2: 运行三个冻结 A2 validation job**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-validation --architecture a2 --dataset MOOCRadar --seed 42 --split-seed 2024 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-validation --architecture a2 --dataset ASSIST17 --seed 42 --split-seed 2024 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-validation --architecture a2 --dataset XES3G5M --seed 42 --split-seed 2024 --campaign-root /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712
```

首轮 completion loss weight 固定 `1.0`，graph hidden dim 等于 recipe concept dim。不得看完 validation 后改变模块语义或加入 dataset-specific 数值。

- [ ] **Step 3: replay 并执行相对门**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py replay --architecture a2
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py relative-gate
```

Expected: 每个数据集输出 standard/holdout/zero/ordinary DOA/weighted DOA delta 和五项门。任何 overall delta < 0 都 fail，不能用较弱外部基线掩盖同架构回退。

- [ ] **Step 4: 根据门结果登记并提交**

若 pass，报告写 `relative gate PASS; external gate remains closed until rebuilt`。若 fail，failure registry 写 fingerprint、三数据集全部 delta、`test 未打开` 和下一独立机制 `MNAR / exposure-aware mastery completion`；不得对 A2 做残差小补丁。

```bash
git add docs/experiments/stable_graph_a2_20260712.md docs/experiments/unified_mastery_failure_registry.md
git commit -m "docs: record relational graph validation"
head=$(git rev-parse --short=12 HEAD)
git bundle create "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712/bundles/a2-$head.bundle" HEAD
```

### Task 8: 外部门、test-once 或下一机制移交

**Files:**
- Modify: `docs/experiments/stable_graph_a2_20260712.md`
- Create only on A2 pass: `docs/experiments/stable_graph_a2_external_gate_20260712.md`

**Interfaces:**
- Consumes relative artifact and comparator audit SHA `071b5df25d8641fdc249a9a56175961025b3deeba36a4a471640563ef5d0b181`。
- Produces either external-gate/test-once proof or concrete MNAR design input，不能同时产生。

- [ ] **Step 1: 仅在 relative pass 后动态重建外部门**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py external-gate --comparator-audit /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712/audit/comparator-audit.json
```

Expected: 动态选择同 protocol strongest comparator；三个数据集 zero AUC 均胜出且 standard/holdout overall 守门。relative fail 时命令拒绝。

- [ ] **Step 2: 外部门通过才执行唯一 test-once**

```bash
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py run-test-once --architecture a2
```

Expected: external pass nonce、完整 replay、clean route 和未使用 test nonce 同时成立才运行；完成后原子消费 nonce，再调用必须拒绝。

- [ ] **Step 3: 最终验证与安全归档**

```bash
conda run -n decoupled_cd python -m unittest discover -s tests -v
conda run -n decoupled_cd python scripts/stable_graph_validation_controller.py status
git status --short
head=$(git rev-parse --short=12 HEAD)
git bundle create "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712/bundles/final-$head.bundle" HEAD
sha256sum "/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712/bundles/final-$head.bundle"
```

Expected: tests PASS、controller 与报告一致、worktree clean、无 test 泄漏、无模型产物入 Git。

- [ ] **Step 4: A2 未达到外部门时立即进入 MNAR 设计**

失败移交从 artifact 明确判断 standard 回退、holdout 回退、zero 无提升，或 reconstruction 与 response objective 冲突。下一设计只改变 missingness/exposure completer，保留 v4 behavior、同一 mastery/head、cohort 与 test-closed 协议；不得把 A2 失败描述为整体完成。

最终汇报明确区分 A0v4、A2 相对结果、外部 comparator、三类 AUC、DOA、test 是否打开，以及失败时注册的下一机制。
