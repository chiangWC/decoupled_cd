# Joint Standard–Holdout No-Regression Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ASSIST17、XES3G5M、ASSIST09 上找到同一数据集跨 standard/holdout 共用的插件配方，使两种 test AUC 均不低于修正 λ=0，同时保持严格 holdout DOA 增益。

**Architecture:** 保留现有 holdout 协议与历史 frozen ID，给新 campaign 增加可选的 `data_protocol`、`dataset_name` 和稳定 `recipe_id`。新的 joint selector 只读取两套 validation manifest 与 holdout DOA，按四重守门和 +0.001 安全余量冻结两个 child selection。先搜索共享骨干辅助损失；若没有安全配方，再启用冻结预测分支、仅训练独立 mastery adapter 的 prediction-invariant 后备路线。

**Tech Stack:** Python 3.12、PyTorch 2.5、标准库 `unittest`、现有 ORCDF/SVGCD vendor、现有 `scripts/run_remote_campaign.py`、Git worktree/bundle。

## Global Constraints

- 远端代码根目录为 `/home/xph/jwc/research/decoupled_cd_codex`；参考目录 `/home/xph/jwc/research/decoupled_cd_v2` 只读。
- 从插件设计提交 `0be0f013e6937ab38e3c70d18321a1cb34eb0b6f` 创建 `codex/joint-split-no-regression-20260710` 和隔离 worktree `decoupled_cd_codex_worktrees/joint_splits`。
- Git 身份固定为 `chiangWC <215551297+chiangWC@users.noreply.github.com>`；不 push GitHub。
- 所有训练与 DOA 使用 seed 42；原有数据 split 不重建；`doa_seed=42`、`split_seed=2024`、`min_responses=3`。
- 同一数据集的 standard/holdout 使用相同插件 recipe；两个 split 分别训练权重，epoch 可分别选择。
- 共享骨干配方只有 validation 最小 AUC 余量达到 +0.001 才允许打开 test；prediction-invariant adapter 以逐元素 prediction 等价证明替代该余量。final hard gate 始终是两种 AUC 差值均 `>=0`。
- holdout weighted DOA 不下降，holdout DOA 严格提升，并保持：ASSIST17 `>0.708857`、XES3G5M `>=0.664573`、ASSIST09 `>0.670806`。
- final 配方的 standard/holdout test 各读取一次；失败后不得使用 test 数值继续选择本 campaign 的其他配方。
- GPU 优先空闲卡；无空闲卡时允许显存占用低于 50% 的卡。每张卡使用独立 `flock`；OOM 不允许静默改 batch size。
- 资产根目录为 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-joint-splits-20260710/`，不提交数据、日志、checkpoint、预测、mastery 或 vendor 资产。

---

### Task 1: 创建隔离实现分支并扩展候选协议

**Files:**
- Modify: `scripts/plugin_campaign.py`
- Modify: `external/ORCDF/main_plugin.py`
- Modify: `external/SVGCD/main_plugin.py`
- Test: `tests/test_plugin_campaign.py`
- Test: `tests/test_orcdf_plugin.py`
- Test: `tests/test_svgcd_plugin.py`

**Interfaces:**
- Produces: `compute_recipe_id(recipe_config, backbone_config) -> str`
- Produces: `protocol_data_kind(protocol) -> "standard" | "holdout"`
- Produces: 新 campaign protocol 可选字段 `data_protocol: "standard" | "holdout"`、`dataset_name: str`
- Preserves: 缺少两个新字段的历史 protocol 按原字节语义计算 frozen ID

- [ ] **Step 1: 创建 worktree 并确认基线**

```bash
git -C /home/xph/jwc/research/decoupled_cd_codex worktree add \
  -b codex/joint-split-no-regression-20260710 \
  /home/xph/jwc/research/decoupled_cd_codex_worktrees/joint_splits \
  0be0f013e6937ab38e3c70d18321a1cb34eb0b6f
git -C /home/xph/jwc/research/decoupled_cd_codex_worktrees/joint_splits config user.name chiangWC
git -C /home/xph/jwc/research/decoupled_cd_codex_worktrees/joint_splits config user.email 215551297+chiangWC@users.noreply.github.com
source /home/xph/anaconda3/etc/profile.d/conda.sh
conda activate decoupled_cd
cd /home/xph/jwc/research/decoupled_cd_codex_worktrees/joint_splits
python3 -m unittest discover -s tests -v
```

Expected: 86 tests pass; worktree clean.

- [ ] **Step 2: 写 protocol/recipe 的失败测试**

在 `tests/test_plugin_campaign.py` 增加：

```python
def test_new_protocol_binds_dataset_split_and_stable_recipe_id(self):
    plugin = {"aux_weight": 0.1, "aux_detach_item_difficulty": False,
              "aux_warmup_fraction": 0.0}
    backbone = {"latent_dim": 32, "gcn_layers": 3}
    assert compute_recipe_id(plugin, backbone) == compute_recipe_id(
        dict(reversed(list(plugin.items()))), dict(reversed(list(backbone.items())))
    )
    standard = self.protocol | {
        "data_protocol": "standard", "dataset_name": "assist17"
    }
    holdout = self.protocol | {
        "data_protocol": "holdout", "dataset_name": "assist17"
    }
    assert compute_frozen_config_id(
        checkpoint_sha256="a" * 64, id_maps_sha256="b" * 64,
        plugin_config=plugin, backbone_config=backbone, protocol=standard,
    ) != compute_frozen_config_id(
        checkpoint_sha256="a" * 64, id_maps_sha256="b" * 64,
        plugin_config=plugin, backbone_config=backbone, protocol=holdout,
    )
```

同时断言只提供一个新字段、空 dataset 或未知 protocol 会失败；旧 fixture 不增加字段时 frozen ID 保持原期望值。

- [ ] **Step 3: 运行测试确认失败**

```bash
python3 -m unittest tests.test_plugin_campaign -v
```

Expected: FAIL because `compute_recipe_id` and optional protocol validation do not exist.

- [ ] **Step 4: 实现兼容协议与 recipe ID**

在 `scripts/plugin_campaign.py` 增加：

```python
_DATA_PROTOCOLS = {"standard", "holdout"}

def compute_recipe_id(
    recipe_config: Mapping[str, Any],
    backbone_config: Mapping[str, Any],
) -> str:
    payload = {
        "recipe_config": _json_copy(dict(recipe_config)),
        "backbone_config": _json_copy(dict(backbone_config)),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
```

在 `_validated_protocol` 中仅当任一新字段出现时要求二者同时存在，验证 `data_protocol` 和非空 `dataset_name`，但不要给历史 protocol 自动插入默认值。`CandidateStore.save()` 的 record 增加：

```python
has_kind = "data_protocol" in normalized
has_dataset = "dataset_name" in normalized
if has_kind != has_dataset:
    raise ValueError("data_protocol and dataset_name must be provided together")
if has_kind:
    if normalized["data_protocol"] not in _DATA_PROTOCOLS:
        raise ValueError("data_protocol must be 'standard' or 'holdout'")
    if not isinstance(normalized["dataset_name"], str) or not normalized[
        "dataset_name"
    ].strip():
        raise ValueError("dataset_name must be a non-empty string")

def protocol_data_kind(protocol: Mapping[str, Any]) -> str:
    # Historical selections predate data_protocol and are all holdout runs.
    return str(protocol.get("data_protocol", "holdout"))

"recipe_config": self.recipe_config,
"recipe_id": compute_recipe_id(self.recipe_config, self.backbone_config),
```

`CandidateStore.__init__()` 新增 `recipe_config`；两个 runner 增加 `recipe_config(args)`，只包含跨 split 必须相同的超参数。它不得包含 checkpoint 路径、checkpoint SHA、Q/data hash 或 epoch。当前首轮字段为 `aux_weight`、`aux_detach_item_difficulty`、`aux_warmup_fraction` 与 ORCDF 的 `decouple`；后续任务再加入 init mode、LR multiplier 和 adapter 开关。

- [ ] **Step 5: 给两个 vendor runner 增加显式新 campaign 参数**

在 ORCDF/SVGCD 的 pre-parser 增加：

```python
pre.add_argument("--plugin-data-protocol", choices=("standard", "holdout"))
pre.add_argument("--plugin-dataset-name")
```

`campaign_protocol()` 只在两个参数非空时写入两个字段；仅有一个时直接报错。新正式命令必须显式传参，旧命令保持兼容。

```python
if bool(args.plugin_data_protocol) != bool(args.plugin_dataset_name):
    raise ValueError(
        "plugin data protocol and dataset name must be provided together"
    )
if args.plugin_data_protocol:
    protocol.update({
        "data_protocol": args.plugin_data_protocol,
        "dataset_name": args.plugin_dataset_name,
    })
```

- [ ] **Step 6: 运行定向与全量测试**

```bash
python3 -m unittest tests.test_plugin_campaign tests.test_orcdf_plugin tests.test_svgcd_plugin -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts models external/ORCDF/main_plugin.py external/SVGCD/main_plugin.py
git diff --check
```

Expected: all tests pass; compileall and diff check exit 0.

- [ ] **Step 7: 提交协议里程碑**

```bash
git add scripts/plugin_campaign.py external/ORCDF/main_plugin.py \
  external/SVGCD/main_plugin.py tests/test_plugin_campaign.py \
  tests/test_orcdf_plugin.py tests/test_svgcd_plugin.py
git commit -m "feat: bind plugin recipes across data protocols"
```

---

### Task 2: 增加 standard baseline selector 与 joint recipe selector

**Files:**
- Create: `scripts/select_standard_checkpoint.py`
- Create: `scripts/select_joint_plugin_recipe.py`
- Create: `tests/test_select_standard_checkpoint.py`
- Create: `tests/test_select_joint_plugin_recipe.py`
- Modify: `scripts/select_plugin_checkpoint.py`

**Interfaces:**
- Produces: `select_standard_checkpoint(...) -> dict[str, Any]`
- Produces: `select_joint_recipe(...) -> dict[str, Any]`
- Produces files: `joint_selection.json`, `standard_selection.json`, `holdout_selection.json`
- Consumes: 两份 candidate manifest、holdout validation DOA CSV、两份 λ=0 baseline selection

- [ ] **Step 1: 写 standard selector 失败测试**

```python
def test_standard_baseline_uses_auc_and_never_requires_doa(self):
    selected = select_standard_checkpoint(
        mode="baseline", manifest_path=self.standard_manifest,
        output_path=self.output,
    )
    assert selected["protocol"]["data_protocol"] == "standard"
    assert selected["validation"]["auc"] == 0.80
    assert "holdout_assignments_sha256" not in selected

def test_standard_plugin_uses_zero_auc_tolerance(self):
    with self.assertRaisesRegex(SelectionError, "no standard plugin candidate"):
        select_standard_checkpoint(
            mode="plugin", manifest_path=self.plugin_manifest,
            baseline_selection_path=self.baseline, output_path=self.output,
        )
```

- [ ] **Step 2: 写 joint selector 四重守门测试**

```python
def test_joint_selector_requires_same_recipe_and_safety_margin(self):
    result = select_joint_recipe(
        standard_manifest_path=self.standard_manifest,
        holdout_manifest_path=self.holdout_manifest,
        holdout_doa_csv_path=self.holdout_doa,
        standard_baseline_path=self.standard_baseline,
        holdout_baseline_path=self.holdout_baseline,
        output_dir=self.output_dir,
        auc_safety_margin=0.001,
    )
    assert result["decision"] == "shared"
    assert result["min_validation_auc_delta"] >= 0.001
    assert result["holdout_validation_doa_delta"] > 0
```

分别构造 standard AUC 负差、holdout AUC 负差、weighted DOA 负差、非加权 DOA 无增益、recipe 不同、只有 `0 <= min_delta < 0.001` 的 fixture；最后一种必须返回 `decision="needs_adapter"` 且不得生成 child selection。

- [ ] **Step 3: 运行测试确认失败**

```bash
python3 -m unittest tests.test_select_standard_checkpoint \
  tests.test_select_joint_plugin_recipe -v
```

Expected: import failures for the two new modules.

- [ ] **Step 4: 实现 standard selector**

`select_standard_checkpoint()` 必须验证 manifest 全部为同一 `dataset_name`、`data_protocol="standard"`、seed 42。baseline 模式在 `aux_weight==0` 中按 AUC、较早 epoch 排序；plugin 模式只接受 `validation.auc >= baseline.validation.auc`。selection payload 包含 checkpoint/id-map hashes、recipe ID、dataset、protocol、validation 和 canonical frozen ID，不包含 holdout 字段。

```python
if mode == "baseline":
    eligible = [
        row for row in candidates
        if float(row["plugin_config"]["aux_weight"]) == 0.0
    ]
    selected = min(
        eligible, key=lambda row: (-row["validation"]["auc"], row["epoch"])
    )
else:
    baseline_auc = float(baseline["validation"]["auc"])
    eligible = [
        row for row in candidates
        if row["validation"]["auc"] >= baseline_auc
    ]
    if not eligible:
        raise SelectionError("no standard plugin candidate satisfies baseline AUC")
    selected = min(
        eligible, key=lambda row: (-row["validation"]["auc"], row["epoch"])
    )
```

CLI：

```text
python scripts/select_standard_checkpoint.py \
  --mode baseline|plugin --manifest MANIFEST \
  [--baseline-selection BASELINE] --output-json OUTPUT
```

- [ ] **Step 5: 实现 joint selector**

核心候选结构固定为：

```python
@dataclass(frozen=True)
class JointCandidate:
    recipe_id: str
    standard: dict[str, Any]
    holdout: dict[str, Any]
    holdout_doa: dict[str, Any]
    standard_auc_delta: float
    holdout_auc_delta: float
    holdout_weighted_doa_delta: float
    holdout_doa_delta: float

    @property
    def min_auc_delta(self) -> float:
        return min(self.standard_auc_delta, self.holdout_auc_delta)
```

对相同 recipe 的 standard/holdout epoch 做笛卡尔组合，先应用四重 `>=0/>0` 硬门，再过滤 `min_auc_delta >= auc_safety_margin`。排序键固定为：

```python
hard_feasible = [
    candidate for candidate in joint_candidates
    if candidate.standard_auc_delta >= 0.0
    and candidate.holdout_auc_delta >= 0.0
    and candidate.holdout_weighted_doa_delta >= 0.0
    and candidate.holdout_doa_delta > 0.0
]
safe = [
    candidate for candidate in hard_feasible
    if candidate.min_auc_delta >= auc_safety_margin
]
if not safe:
    return write_needs_adapter_diagnostics(hard_feasible, output_dir)
selected = min(
    safe,
    key=lambda candidate: (
        -candidate.min_auc_delta,
        -candidate.holdout_doa_delta,
        candidate.standard["epoch"],
        candidate.holdout["epoch"],
    ),
)
```

有安全候选时原子写三份 JSON；没有时只写 `joint_diagnostics.json`，返回 `needs_adapter`，CLI exit code 3。所有输出都绑定输入文件 SHA256。

- [ ] **Step 6: 让旧 selector 可复用公开解析函数**

把 manifest/protocol/artifact hash 校验提升为无下划线的公共函数，旧 `select_plugin_checkpoint.py` 调用这些函数，保持现有 0.002 holdout 历史模式与 86 个回归测试不变。禁止复制两套不一致的 protocol 校验。

- [ ] **Step 7: 运行全部 selector 测试并提交**

```bash
python3 -m unittest tests.test_select_plugin_checkpoint \
  tests.test_select_standard_checkpoint tests.test_select_joint_plugin_recipe -v
python3 -m unittest discover -s tests -v
git diff --check
git add scripts/select_plugin_checkpoint.py scripts/select_standard_checkpoint.py \
  scripts/select_joint_plugin_recipe.py tests/test_select_standard_checkpoint.py \
  tests/test_select_joint_plugin_recipe.py
git commit -m "feat: select no-regression recipes across splits"
```

---

### Task 3: 扩展 standard test-once claim，保持 holdout 强绑定

**Files:**
- Modify: `scripts/plugin_campaign.py`
- Modify: `scripts/doa_cached.py`
- Test: `tests/test_plugin_campaign.py`
- Test: `tests/test_doa_cached.py`
- Test: `tests/test_plugin_data_isolation.py`

**Interfaces:**
- Standard selection/claim: 不包含 `holdout_assignments_sha256`
- Holdout selection/claim/cache: 继续强制 `holdout_assignments_sha256`
- Claim file name: frozen config ID；其 protocol 已区分 standard/holdout

- [ ] **Step 1: 写协议隔离失败测试**

增加以下断言：standard claim 在 selection 无 holdout hash 时成功；holdout 同样输入必须失败；standard/holdout frozen ID 与 claim 路径不同；两类 test 都在 resource loader 第一次读取 test 前创建 claim。

```python
def test_standard_claim_omits_holdout_but_holdout_still_requires_it(self):
    standard = self.make_selection(data_protocol="standard", holdout_sha=None)
    claim = claim_test_evaluation(**standard)
    assert "holdout_assignments_sha256" not in json.loads(claim.read_text())
    with self.assertRaises(FrozenConfigMismatchError):
        claim_test_evaluation(**self.make_selection(
            data_protocol="holdout", holdout_sha=None
        ))
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python3 -m unittest tests.test_plugin_campaign \
  tests.test_plugin_data_isolation tests.test_doa_cached -v
```

Expected: standard selection rejected for missing holdout hash.

- [ ] **Step 3: 实现条件式 holdout 绑定**

`_validated_selection()` 返回 `str | None` 的 holdout SHA。`data_protocol == "holdout"` 时沿用严格 SHA 校验；`standard` 时 selection 出现该字段反而报错，避免含义混乱。claim record 与 evaluation cache manifest 仅在 holdout 时写该字段。

```python
kind = protocol_data_kind(normalized_protocol)
raw_holdout_sha = selection.get("holdout_assignments_sha256")
if kind == "holdout":
    holdout_sha = _validated_sha256(
        raw_holdout_sha, "selection holdout_assignments_sha256"
    )
elif raw_holdout_sha is not None:
    raise FrozenConfigMismatchError(
        "standard selection must not bind holdout assignments"
    )
else:
    holdout_sha = None
```

`doa_cached.py` 通过 `protocol_data_kind(protocol)` 明确拒绝非 holdout cache；缺少新字段的历史 cache 按 legacy holdout 处理，从而既防止误读 standard，又保持旧结果可审计。

- [ ] **Step 4: 运行全套测试与 CPU smoke**

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q scripts external/ORCDF/main_plugin.py external/SVGCD/main_plugin.py
SMOKE_DIR=$(mktemp -d)
CUDA_VISIBLE_DEVICES="" python3 external/ORCDF/main_plugin.py --plugin-mode train \
  --plugin-data-protocol standard --plugin-dataset-name synthetic \
  --plugin-aux-weight 0 \
  --data_dir /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/synthetic/plugin_fixture \
  --train_file train.csv --valid_file valid.csv --test_file test.csv \
  --plugin-q-matrix-file Q_matrix.csv --log_dir "$SMOKE_DIR" \
  --latent_dim 4 --gcn_layers 1 --prednet_len1 8 --prednet_len2 4 \
  --batch_size 4 --epochs 1 --patience 1 --seed 42
test -s "$SMOKE_DIR/candidates/manifest.jsonl"
rm -rf "$SMOKE_DIR"
```

Expected: 生成 1 个 candidate checkpoint；validation AUC 有限；正式数据 test 不被读取。

- [ ] **Step 5: 提交 test-once 里程碑并生成 bundle**

```bash
git add scripts/plugin_campaign.py scripts/doa_cached.py \
  tests/test_plugin_campaign.py tests/test_doa_cached.py \
  tests/test_plugin_data_isolation.py
git commit -m "feat: guard standard and holdout test claims"
git bundle create \
  /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/bundles/joint-protocol-$(git rev-parse --short HEAD).bundle \
  codex/joint-split-no-regression-20260710
```

---

### Task 4: 运行修正 λ=0 与当前 λ=0.5 的联合 validation 首轮

**Files:**
- No tracked file changes
- Artifacts: `formal-aaai-joint-splits-20260710/shared/{dataset}/{protocol}/{recipe}/attempt-NNN/`

**Interfaces:**
- ORCDF datasets: ASSIST17、XES3G5M
- SVGCD dataset: ASSIST09
- Produces: 每数据集两套 baseline selection、λ=0.5 manifests、holdout validation DOA、joint diagnostics/selection

- [ ] **Step 1: 启动前执行环境与数据审计**

```bash
git status --porcelain
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,uuid --format=csv,noheader
sha256sum /home/xph/jwc/research/knofield_data/{assist_17,xes3g5m,assist_09}/{train,valid,Q_matrix}.csv
sha256sum /home/xph/jwc/research/knofield_data/{assist_17_chold_v2,xes3g5m_chold_v2,assist_09_chold_v2}/{train,valid,Q_matrix,student_concept_holdout_assignments}.csv
```

Expected: clean tree；所有文件存在；runner status 将绑定相同 hashes。

- [ ] **Step 2: 为每个 split 运行 λ=0 baseline**

所有命令都通过 `scripts/run_remote_campaign.py`，外层 `--dataset-file` 只列 train/valid/Q；不得列 test。显式传：

```text
--plugin-data-protocol standard|holdout
--plugin-dataset-name assist17|xes3g5m|assist09
--plugin-aux-weight 0 --seed 42
```

配置表：

| 数据集 | 骨干 | standard 目录 | holdout 目录 | epochs/patience |
|---|---|---|---|---|
| ASSIST17 | ORCDF | `assist_17` | `assist_17_chold_v2` | 30/5 |
| XES3G5M | ORCDF | `xes3g5m` | `xes3g5m_chold_v2` | 30/5 |
| ASSIST09 | SVGCD | `assist_09` | `assist_09_chold_v2` | 60/5 |

ORCDF 沿用 latent32、gcn3、keep0.9、NCD、lr0.004、batch1024；SVGCD 沿用 emb128、256/128、dropout0.5、gnn2、lr0.001、batch1024。

- [ ] **Step 3: 只在 validation 生成 baseline selection 与 holdout DOA**

standard 使用 `select_standard_checkpoint.py --mode baseline`；holdout 使用现有 `doa_external.py --split valid` 和 `select_plugin_checkpoint.py --mode baseline`。验证每个 status 为 completed、route commit 正确、无 test hash。

- [ ] **Step 4: 为两种 split 运行当前 λ=0.5**

与 baseline 唯一差异是 `--plugin-aux-weight 0.5` 和 model name；同一数据集两套 manifest 的 `recipe_id` 必须相同。

- [ ] **Step 5: 运行 joint selector**

```bash
python3 scripts/select_joint_plugin_recipe.py \
  --standard-manifest "$STD_MANIFEST" \
  --holdout-manifest "$HOLDOUT_MANIFEST" \
  --holdout-valid-doa-csv "$HOLDOUT_DOA" \
  --standard-baseline-selection "$STD_BASE" \
  --holdout-baseline-selection "$HOLDOUT_BASE" \
  --auc-safety-margin 0.001 --output-dir "$ATTEMPT_DIR"
```

Expected: 只依据 validation 输出 `shared` 或 `needs_adapter`；不执行 test。

---

### Task 5: 对失败数据集执行预注册共享骨干搜索

**Files:**
- Conditionally modify: `scripts/plugin_campaign.py`
- Conditionally modify: `external/ORCDF/main_plugin.py`
- Conditionally modify: `external/SVGCD/main_plugin.py`
- Conditionally test: `tests/test_plugin_campaign.py`
- Conditionally test: `tests/test_orcdf_plugin.py`
- Conditionally test: `tests/test_svgcd_plugin.py`
- Artifacts: `formal-aaai-joint-splits-20260710/shared/...`

**Interfaces:**
- Search recipe: `(aux_weight, detach, warmup_fraction, init_mode, lr_multiplier)`
- Conditional helper: `validate_training_initialization(...) -> None`
- Stop: 每数据集第一次出现 +0.001 safe joint selection 后停止该数据集新训练

- [ ] **Step 1: 依次运行权重波次**

仅对 Task 4 未通过的数据集，在 standard/holdout 同时运行 `0.05, 0.1, 0.25`；λ=0.5 不重跑。每个完整 recipe 波次结束后计算 holdout valid DOA 并运行 joint selector，禁止根据单一 split 提前挑选。

- [ ] **Step 2: 测试最佳两个权重的 detach**

按 hard-feasible 候选的 `min_auc_delta` 选两个权重，分别添加 `--plugin-aux-detach-item-difficulty`；仍使用相同 seed 和 epochs。

- [ ] **Step 3: 仅对无安全余量数据集测试 warm-up**

最佳 recipe 增加 `--plugin-aux-warmup-fraction 0.2`。如果 safe，立即冻结并停止该数据集共享搜索。

- [ ] **Step 4: 如需 fine-tune，先写失败测试和最小 CLI**

增加 train-only 参数：

```text
--plugin-init-checkpoint PATH
--plugin-init-mode baseline-finetune
--plugin-lr-multiplier 0.25|1.0
```

测试必须证明 checkpoint 在创建 optimizer 前加载、id map/Q/schema 一致、plugin config/recipe ID 绑定 init mode 和 multiplier、evaluate 模式不能伪装成 fine-tune。实现后从匹配 λ=0 epoch 启动，实际 optimizer LR 为原 LR 乘 multiplier。

两个 runner 共用以下初始化顺序；`snapshot_evaluation_artifacts` 提供 checkpoint/id-map/Q 的冻结字节，selection 校验必须在构造 optimizer 前完成：

```python
if args.plugin_init_mode == "baseline-finetune":
    if args.plugin_init_checkpoint is None:
        raise ValueError("baseline-finetune requires --plugin-init-checkpoint")
    init_snapshot = snapshot_evaluation_artifacts(
        args.plugin_init_checkpoint, q_matrix_path
    )
    validate_training_initialization(
        checkpoint_path=args.plugin_init_checkpoint,
        snapshot=init_snapshot,
        q_matrix_bytes=q_matrix_bytes,
        processor_id_maps=processor_id_maps(proc),
    )
    load_evaluation_checkpoint(model, init_snapshot.checkpoint_bytes, device)
effective_lr = args.lr * args.plugin_lr_multiplier
trainer_args = argparse.Namespace(**vars(args))
trainer_args.lr = effective_lr
trainer = Trainer(model, loaders, proc, trainer_args, logger)
```

`plugin_config()` 同时写入 `init_mode`、baseline checkpoint SHA、base LR 与 multiplier；`recipe_config()` 只写 init mode、base LR 与 multiplier，不写 split-specific checkpoint SHA。不得在 optimizer 创建后修改 LR。

- [ ] **Step 5: 运行 fine-tune 的两个倍率并最终判定**

只有 standard/holdout 两套 validation 都完成后运行 joint selector。若仍无 +0.001 safe recipe，将该数据集标为 `shared_exhausted`，进入 Task 6；不追加权重或学习率网格。

- [ ] **Step 6: 提交仅在本任务产生的 CLI 代码**

```bash
python3 -m unittest discover -s tests -v
git diff --check
git add scripts/plugin_campaign.py external/ORCDF/main_plugin.py \
  external/SVGCD/main_plugin.py tests/test_plugin_campaign.py \
  tests/test_orcdf_plugin.py tests/test_svgcd_plugin.py
git commit -m "feat: support baseline-initialized plugin tuning"
```

如果未触发 fine-tune 代码变更，本步骤不创建空提交。

---

### Task 6: 条件式实现 prediction-invariant mastery adapter

**Condition:** 仅当至少一个数据集在 Task 5 标记为 `shared_exhausted` 时执行。

**Files:**
- Create: `models/diagnostic_mastery_adapter.py`
- Modify: `external/ORCDF/ORCDF/plugin.py`
- Modify: `external/ORCDF/main_plugin.py`
- Modify: `external/SVGCD/main_plugin.py`
- Test: `tests/test_diagnostic_mastery_adapter.py`
- Test: `tests/test_orcdf_plugin.py`
- Test: `tests/test_svgcd_plugin.py`

**Interfaces:**
- Produces: `PredictionInvariantMasteryAdapter(knowledge_n: int)`
- Produces: `forward(base_mastery_logits: Tensor) -> Tensor`
- CLI: `--plugin-diagnostic-adapter`，必须配合 baseline init checkpoint
- Prediction path invariant；`mastery_matrix()` 返回 adapter mastery

- [ ] **Step 1: 写 adapter 单元失败测试**

```python
def test_adapter_is_zero_residual_and_only_adapter_receives_gradient():
    adapter = PredictionInvariantMasteryAdapter(knowledge_n=3)
    base = torch.randn(4, 3, requires_grad=True)
    adapted = adapter(base)
    torch.testing.assert_close(adapted, base.detach(), rtol=0, atol=0)
    adapted.sum().backward()
    assert base.grad is None
    assert all(p.grad is not None for p in adapter.parameters())
```

- [ ] **Step 2: 写两个骨干的 prediction invariance 失败测试**

加载同一 λ=0 state，分别构造 adapter off/on 模型。对固定 batch 断言：

```python
torch.testing.assert_close(pred_adapter, pred_base, rtol=0, atol=0)
assert optimizer_parameter_ids == {id(p) for p in model.diagnostic_adapter.parameters()}
assert all(not p.requires_grad for name, p in model.named_parameters()
           if not name.startswith("diagnostic_adapter."))
```

同时断言训练一步后 prediction 仍逐元素一致，但 adapter mastery 发生变化。

- [ ] **Step 3: 运行测试确认失败**

```bash
python3 -m unittest tests.test_diagnostic_mastery_adapter \
  tests.test_orcdf_plugin tests.test_svgcd_plugin -v
```

- [ ] **Step 4: 实现零残差 adapter**

```python
class PredictionInvariantMasteryAdapter(nn.Module):
    def __init__(self, knowledge_n: int) -> None:
        super().__init__()
        self.residual = nn.Linear(knowledge_n, knowledge_n, bias=True)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, base_mastery_logits: torch.Tensor) -> torch.Tensor:
        frozen = base_mastery_logits.detach()
        return frozen + self.residual(frozen)
```

ORCDF prediction 继续使用 extractor 的原 `student_ts`；auxiliary 与 `mastery_matrix()` 使用 adapter 输出。SVGCD prediction 继续使用原 `prednet_stu`；auxiliary 与 mastery 使用 adapter 输出。启用 adapter 时冻结其他参数并重建仅含 adapter 参数的 optimizer。

从 λ=0 初始化 adapter 模型时只允许缺少 adapter 自身参数：

```python
incompatible = model.load_state_dict(base_state_dict, strict=False)
expected_missing = {
    name for name, _ in model.named_parameters()
    if name.startswith("diagnostic_adapter.")
}
if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
    raise ValueError("baseline checkpoint is incompatible with adapter model")
```

adapter checkpoint 保存后，validation/test evaluation 必须恢复为 `strict=True`。`recipe_config()` 增加 `diagnostic_adapter=True`，但不包含 split-specific base checkpoint SHA。

- [ ] **Step 5: 运行全量测试与 CPU/GPU smoke**

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q models scripts external/ORCDF external/SVGCD
git diff --check
```

GPU smoke 使用 synthetic 1 epoch，记录 adapter 前后 prediction artifact SHA；两者必须相同。

- [ ] **Step 6: 提交 adapter 并生成 bundle**

```bash
git add models/diagnostic_mastery_adapter.py external/ORCDF/ORCDF/plugin.py \
  external/ORCDF/main_plugin.py external/SVGCD/main_plugin.py \
  tests/test_diagnostic_mastery_adapter.py tests/test_orcdf_plugin.py \
  tests/test_svgcd_plugin.py
git commit -m "feat: add prediction-invariant mastery adapter"
git bundle create \
  /home/xph/jwc/research/local_data/decoupled_cd_codex_routes/bundles/joint-adapter-$(git rev-parse --short HEAD).bundle \
  codex/joint-split-no-regression-20260710
```

- [ ] **Step 7: 只用 validation 训练并选择 adapter**

对 `shared_exhausted` 数据集，从两种 split 的 λ=0 selection 初始化；standard 只验证 prediction 等价，holdout 以 DOA 增益选择 adapter epoch。若 holdout valid DOA 未提升，数据集记失败，不打开 test。

adapter 因结构上 prediction-invariant，不要求 +0.001 validation AUC 余量；进入 test 的前提改为两种 validation prediction CSV 的 probability SHA 与各自 λ=0 完全一致，以及 holdout validation DOA 通过增益门。

---

### Task 7: 冻结唯一 final 配方并执行双 test-once

**Files:**
- No tracked code changes
- Artifacts: `formal-aaai-joint-splits-20260710/final/{dataset}/...`

**Interfaces:**
- Consumes: joint child selections 或 adapter selections
- Produces: fresh λ=0 standard/holdout test、plugin standard/holdout test、holdout cached DOA

- [ ] **Step 1: 做最终 preflight**

验证 branch clean、selection/input SHA 未变、claim 不存在、runner 外层输入不含真实 test、GPU/CPU 环境可导入。冻结文件记录 recipe ID、两份 child frozen ID、route commit 和全部 validation deltas。

- [ ] **Step 2: 对 fresh λ=0 baseline 各打开一次 test**

standard 与 holdout baseline 分别使用各自 selection/ledger；外层 runner 不列 test。保存 AUC/ACC/RMSE、cache 和 claim。holdout baseline 从 cache 计算 DOA，建立精确同代码比较。

- [ ] **Step 3: 对唯一 plugin 配方各打开一次 test**

先原子 claim、再读 test。standard 不计算 holdout DOA；holdout 只从 evaluation cache 计算 DOA，不重读 test。

- [ ] **Step 4: 运行硬门断言**

```python
assert plugin_standard_auc - base_standard_auc >= 0.0
assert plugin_holdout_auc - base_holdout_auc >= 0.0
assert plugin_holdout_weighted_doa - base_holdout_weighted_doa >= 0.0
assert plugin_holdout_doa - base_holdout_doa > 0.0
assert plugin_holdout_doa >= dataset_threshold
```

ASSIST17 threshold 采用严格 `>0.708857`，XES3G5M `>=0.664573`，ASSIST09 `>0.670806`。任何断言失败即记录该数据集失败，不继续调参。

- [ ] **Step 5: 审计 test-once 与停止状态**

扫描所有 final status：无 running、无 missing output、无 dirty tree、seed/split 全匹配；每个 frozen ID 恰有一个 claim。特别检查新 ASSIST17 runner 未将 test 列为 `--dataset-file`，同时保留旧偏差记录。

---

### Task 8: 验证、中文总结、提交和最终 bundle

**Files:**
- Create: `docs/joint_split_no_regression_results_20260710.md`
- Modify: `docs/aaai_experiment_progress_20260710.md`

**Interfaces:**
- Produces: 分数据集 standard/holdout AUC delta、holdout DOA delta、共享/adapter 路线标识、失败/排除记录

- [ ] **Step 1: 运行最终软件验证**

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q models scripts external/ORCDF external/SVGCD
git diff --check
```

Expected: all tests pass and worktree only包含待提交的两份文档。

- [ ] **Step 2: 写中文结果摘要**

摘要表必须同时展示 base/plugin 的 standard AUC、holdout AUC、两项差值、base/plugin holdout DOA 和差值。共享骨干与 prediction-invariant adapter 分表；不得把未通过双 AUC 硬门的结果标为胜出。

明确记录 seed 42、validation +0.001 安全余量、test-once、既有 XES/ASSIST09 test 已见事实和 ASSIST17 旧预哈希偏差。完整模型标准补测另立后续计划，不混入插件成功计数。

- [ ] **Step 3: 提交文档**

```bash
git add docs/joint_split_no_regression_results_20260710.md \
  docs/aaai_experiment_progress_20260710.md
git commit -m "docs: report joint split no-regression results"
```

- [ ] **Step 4: 创建并验证最终 bundle**

```bash
B=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/bundles
F=$B/joint-split-final-$(git rev-parse --short HEAD).bundle
git bundle create "$F" codex/joint-split-no-regression-20260710
git bundle verify "$F"
sha256sum "$F"
```

- [ ] **Step 5: 最终只读审计**

确认所有 worktree clean、无 upstream、origin 无新分支、`decoupled_cd_v2` 最新 mtime 仍早于 campaign、无本方 GPU 进程、所有 runner 输出 SHA256 可复算。最终交付报告列出 commits、bundle SHA、测试数、三数据集成功数和任何协议例外。
