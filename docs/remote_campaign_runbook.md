# 远端实验 Campaign 运行手册

## 目录与约束

统一 artifact 根目录为：

```text
/home/xph/jwc/research/local_data/decoupled_cd_codex_routes
```

`scripts/run_remote_campaign.py` 每次调用都在该根目录下分配新的
`attempt-NNN/`。编号分配使用独占文件锁；已有 attempt 不会被复用或覆盖。
每个 attempt 至少包含 `status.json` 与 `command.log`。请勿在运行结束后修改
attempt 内容；需要重跑时创建下一个 attempt。

执行前必须满足：

- 位于预期 Git worktree，且已提交当前代码；tracked/untracked 文件均不得残留。
- `--cwd` 必须位于 `--repo-root` 指向的同一 worktree，不能借用另一仓库
  的 clean HEAD；vendor commit 还必须存在于该 route 的提交历史中。
- 显式列出本次使用的所有 `--dataset-file`，用于记录 SHA-256。
- 插件路线用 `--vendor-commit NAME=COMMIT` 逐项记录 40 位 vendor
  提交；名称不得重复。当前 worktree HEAD 自动作为 route commit。
- 输出写入环境变量 `CAMPAIGN_ATTEMPT_DIR` 指向的目录，并通过
  `--output-file` 声明需要校验的文件。
- `--capture-env` 只填写非敏感变量名，不要记录 token、密码或凭据。

## 先做 dry-run

在远端激活环境并进入仓库 worktree：

```bash
source /home/xph/anaconda3/etc/profile.d/conda.sh
conda activate decoupled_cd

WORKTREE=/path/to/clean-worktree
SPLIT_DIR=/path/to/preprocessed-split
MASTERY_DIR=/path/to/mastery
ARTIFACT_ROOT=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes
cd "${WORKTREE}"

python3 scripts/run_remote_campaign.py \
  --artifact-root "${ARTIFACT_ROOT}" \
  --repo-root "${PWD}" \
  --cwd "${PWD}" \
  --dataset-file "${SPLIT_DIR}/test.csv" \
  --dataset-file "${MASTERY_DIR}/mastery.npy" \
  --dataset-file "${MASTERY_DIR}/id_maps.json" \
  --vendor-commit "orcdf=${ORCDF_VENDOR_COMMIT}" \
  --output-file doa.csv \
  --seed 42 \
  --doa-seed 42 \
  --min-responses 3 \
  --split-seed 2024 \
  --dry-run \
  -- bash -lc 'python3 scripts/doa_external.py \
      --dataset-name "$1" \
      --split-dir "$2" \
      --split test \
      --mastery-dir "$3" \
      --model-name "$4" \
      --min-responses "$CAMPAIGN_MIN_RESPONSES" \
      --doa-seed "$CAMPAIGN_DOA_SEED" \
      --output-csv "$CAMPAIGN_ATTEMPT_DIR/doa.csv"' \
    campaign assist_09 "${SPLIT_DIR}" "${MASTERY_DIR}" model-name
```

dry-run 不执行末尾命令，但仍会创建一个状态为 `dry_run` 的 attempt，便于审计
完整 argv、默认参数、Git HEAD、数据哈希与运行环境。确认其 `status.json` 后，
删除 `--dry-run` 重新调用；正式运行会分配新的 attempt。

四个可复现参数的默认值即为：`seed=42`、`doa_seed=42`、
`min_responses=3`、`split_seed=2024`。runner 同时向子进程提供：

- `CAMPAIGN_ATTEMPT_DIR`
- `CAMPAIGN_SEED`
- `CAMPAIGN_DOA_SEED`
- `CAMPAIGN_MIN_RESPONSES`
- `CAMPAIGN_SPLIT_SEED`

如需在验证集计算外部 mastery 的 DOA，传入 `--split valid`；不传 `--split`
仍读取 `test.csv`，也可显式传入 `--split test`。

## 状态检查

运行结束后检查 runner 输出的 attempt 路径：

```bash
python3 -m json.tool "${ARTIFACT_ROOT}/attempt-NNN/status.json"
sha256sum "${ARTIFACT_ROOT}/attempt-NNN/command.log"
```

`status.json` 会原子替换为终态，并记录开始/结束 UTC、退出码、argv/cwd、选定
环境变量、Git HEAD/clean 状态、数据文件哈希、Python/Torch/CUDA/GPU 元数据
（环境可提供时）、route/vendor commits、子进程及其可观察后代按 GPU UUID
汇总的运行期峰值显存，以及声明输出和命令日志的哈希。峰值显存每秒通过
`nvidia-smi` 采样；无 NVIDIA 工具时字段保留并标为不可用，不影响 CPU 作业。
采样器异常会降级为审计错误；中断 runner 时会终止并回收整个命令进程组。
产物在哈希期间消失或不可读时，错误写入对应 output 条目，attempt 仍会获得终态。
`completed` 表示退出码为 0；
`failed` 保留非零退出码；`dry_run` 的退出码为 `null`。

## 插件路线：validation 选型与单次 test

ORCDF 和 SVGCD 的插件入口都要求显式传入 `--plugin-mode train` 或
`--plugin-mode evaluate`。train 模式每个实际训练 epoch 都写入新的
`candidates/epoch-NNN/`，并追加 `candidates/manifest.jsonl`；它只在
validation loader 上 early-stop，不迭代 test loader，也不输出 test 指标。

下面给出 ORCDF 的完整命令骨架；使用 SVGCD 时将 `MODEL_RUNNER` 改为
`external/SVGCD/main_plugin.py`、将 `VENDOR_NAME` 改为 `svgcd`，并删除
ORCDF 专属的 `--plugin-decouple`。
下面每个阶段都通过 `scripts/run_remote_campaign.py` 分配独立 attempt；先给
`campaign_run` 临时加上 `--dry-run` 审核状态，再删除该参数正式执行。数据 split 必须预先由
`split_seed=2024` 生成；选型协议固定为 `seed=42`、`doa_seed=42`、
`min_responses=3`、`split_seed=2024`，不得使用 `doa_external.py` 的默认值。

```bash
WORKTREE=/home/xph/jwc/research/decoupled_cd_codex_worktrees/plugin
MODEL_RUNNER=external/ORCDF/main_plugin.py
SPLIT_DIR=/path/to/split-seed-2024
HOLDOUT_ASSIGNMENTS=/path/to/holdout_assignments.csv
ARTIFACT_ROOT=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes
CAMPAIGN_ROOT="${ARTIFACT_ROOT}/plugin-selector-YYYYMMDD"
TEST_LEDGER=/path/to/append-only-test-ledger
VENDOR_NAME=orcdf
VENDOR_COMMIT=0fb7f252f5535943c0ccc579fd21ed31c57e7138
LAMBDA=0.1

cd "${WORKTREE}"
export MODEL_RUNNER SPLIT_DIR HOLDOUT_ASSIGNMENTS LAMBDA

campaign_run() {
  local stage=$1
  shift
  python3 scripts/run_remote_campaign.py \
    --artifact-root "${CAMPAIGN_ROOT}/${stage}" \
    --repo-root "${WORKTREE}" \
    --cwd "${WORKTREE}" \
    --vendor-commit "${VENDOR_NAME}=${VENDOR_COMMIT}" \
    --seed 42 \
    --doa-seed 42 \
    --min-responses 3 \
    --split-seed 2024 \
    "$@"
}
```

先训练 λ=0 baseline。`--plugin-model-name` 是 DOA CSV 与 manifest 的唯一连接键；
同一 manifest 内不能重复。

```bash
BASE_ATTEMPT=$(campaign_run baseline-train \
  --dataset-file "${SPLIT_DIR}/train.csv" \
  --dataset-file "${SPLIT_DIR}/valid.csv" \
  --output-file model/candidates/manifest.jsonl \
  -- bash -lc '
    python3 "${MODEL_RUNNER}" \
      --plugin-mode train \
      --plugin-model-name orcdf-baseline \
      --plugin-aux-weight 0 \
      --data_dir "${SPLIT_DIR}" \
      --train_file train.csv \
      --valid_file valid.csv \
      --test_file test.csv \
      --log_dir "${CAMPAIGN_ATTEMPT_DIR}/model" \
      --seed 42 \
      --plugin-doa-seed 42 \
      --plugin-min-responses 3 \
      --plugin-max-pairs-per-concept 100000 \
      --plugin-split-seed 2024')
BASE_RUN="${BASE_ATTEMPT}/model"
```

只用 `valid.csv` 和每个 epoch 的 mastery 计算 holdout DOA。下面的 helper 从
manifest 构造 `doa_external.py` 所需的重复参数，避免手工写错 model 名称。

```bash
build_doa_args() {
  local manifest=$1
  local run_dir=$2
  DOA_ARGS=()
  CANDIDATE_DATASET_ARGS=()
  while IFS=$'\t' read -r model_name mastery_path; do
    local mastery_dir="${run_dir}/$(dirname "${mastery_path}")"
    DOA_ARGS+=(
      --mastery-dir "${mastery_dir}"
      --model-name "${model_name}"
    )
    CANDIDATE_DATASET_ARGS+=(
      --dataset-file "${mastery_dir}/checkpoint.pth"
      --dataset-file "${mastery_dir}/mastery.npy"
      --dataset-file "${mastery_dir}/id_maps.json"
    )
  done < <(python3 - "${manifest}" <<'PY'
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    row = json.loads(line)
    print(f'{row["model_name"]}\t{row["mastery_path"]}')
PY
  )
}

build_doa_args "${BASE_RUN}/candidates/manifest.jsonl" "${BASE_RUN}"
BASE_DOA_ATTEMPT=$(campaign_run baseline-valid-doa \
  --dataset-file "${SPLIT_DIR}/valid.csv" \
  --dataset-file "${HOLDOUT_ASSIGNMENTS}" \
  --dataset-file "${BASE_RUN}/candidates/manifest.jsonl" \
  "${CANDIDATE_DATASET_ARGS[@]}" \
  --output-file valid_doa.csv \
  -- bash -lc '
    python3 scripts/doa_external.py \
      --dataset-name assist_09 \
      --split-dir "${SPLIT_DIR}" \
      --split valid \
      --holdout-assignments "${HOLDOUT_ASSIGNMENTS}" \
      --min-responses 3 \
      --max-pairs-per-concept 100000 \
      --doa-seed 42 \
      "$@" \
      --output-csv "${CAMPAIGN_ATTEMPT_DIR}/valid_doa.csv"' \
    doa "${DOA_ARGS[@]}")
BASE_VALID_DOA="${BASE_DOA_ATTEMPT}/valid_doa.csv"

BASE_SELECT_ATTEMPT=$(campaign_run baseline-select \
  --dataset-file "${BASE_RUN}/candidates/manifest.jsonl" \
  --dataset-file "${BASE_VALID_DOA}" \
  "${CANDIDATE_DATASET_ARGS[@]}" \
  --output-file selection.json \
  -- bash -lc '
    python3 scripts/select_plugin_checkpoint.py \
      --mode baseline \
      --manifest "$1" \
      --valid-doa-csv "$2" \
      --output-json "${CAMPAIGN_ATTEMPT_DIR}/selection.json"' \
    select "${BASE_RUN}/candidates/manifest.jsonl" "${BASE_VALID_DOA}")
BASE_SELECTION="${BASE_SELECT_ATTEMPT}/selection.json"
```

随后训练插件配置并重复 valid DOA。selector 只接受
`AUC >= baseline_AUC - 0.002` 且
`holdout_doa_weighted >= baseline_holdout_doa_weighted` 的候选；可行候选中按
holdout DOA、AUC、较早 epoch 的顺序选定。无可行候选会非零退出，不能转去
查看 test 后再挑配置。

```bash
PLUGIN_ATTEMPT=$(campaign_run plugin-train \
  --dataset-file "${SPLIT_DIR}/train.csv" \
  --dataset-file "${SPLIT_DIR}/valid.csv" \
  --output-file model/candidates/manifest.jsonl \
  -- bash -lc '
    python3 "${MODEL_RUNNER}" \
      --plugin-mode train \
      --plugin-model-name orcdf-plugin \
      --plugin-decouple \
      --plugin-aux-weight "${LAMBDA}" \
      --data_dir "${SPLIT_DIR}" \
      --train_file train.csv \
      --valid_file valid.csv \
      --test_file test.csv \
      --log_dir "${CAMPAIGN_ATTEMPT_DIR}/model" \
      --seed 42 \
      --plugin-doa-seed 42 \
      --plugin-min-responses 3 \
      --plugin-max-pairs-per-concept 100000 \
      --plugin-split-seed 2024')
PLUGIN_RUN="${PLUGIN_ATTEMPT}/model"

build_doa_args "${PLUGIN_RUN}/candidates/manifest.jsonl" "${PLUGIN_RUN}"
PLUGIN_DOA_ATTEMPT=$(campaign_run plugin-valid-doa \
  --dataset-file "${SPLIT_DIR}/valid.csv" \
  --dataset-file "${HOLDOUT_ASSIGNMENTS}" \
  --dataset-file "${PLUGIN_RUN}/candidates/manifest.jsonl" \
  "${CANDIDATE_DATASET_ARGS[@]}" \
  --output-file valid_doa.csv \
  -- bash -lc '
    python3 scripts/doa_external.py \
      --dataset-name assist_09 \
      --split-dir "${SPLIT_DIR}" \
      --split valid \
      --holdout-assignments "${HOLDOUT_ASSIGNMENTS}" \
      --min-responses 3 \
      --max-pairs-per-concept 100000 \
      --doa-seed 42 \
      "$@" \
      --output-csv "${CAMPAIGN_ATTEMPT_DIR}/valid_doa.csv"' \
    doa "${DOA_ARGS[@]}")
PLUGIN_VALID_DOA="${PLUGIN_DOA_ATTEMPT}/valid_doa.csv"

PLUGIN_SELECT_ATTEMPT=$(campaign_run plugin-select \
  --dataset-file "${PLUGIN_RUN}/candidates/manifest.jsonl" \
  --dataset-file "${PLUGIN_VALID_DOA}" \
  --dataset-file "${BASE_SELECTION}" \
  "${CANDIDATE_DATASET_ARGS[@]}" \
  --output-file selection.json \
  -- bash -lc '
    python3 scripts/select_plugin_checkpoint.py \
      --mode plugin \
      --manifest "$1" \
      --valid-doa-csv "$2" \
      --baseline-selection "$3" \
      --output-json "${CAMPAIGN_ATTEMPT_DIR}/selection.json"' \
    select "${PLUGIN_RUN}/candidates/manifest.jsonl" \
      "${PLUGIN_VALID_DOA}" "${BASE_SELECTION}")
PLUGIN_SELECTION="${PLUGIN_SELECT_ATTEMPT}/selection.json"
```

最后从冻结 selection 读取 checkpoint 与 config ID，只执行一次 test。runner 会用
实际 checkpoint、CLI plugin config 和锁定 protocol 重算 config ID，并要求与
`--plugin-selection-json` 完全一致。test claim
使用 `O_EXCL` 在读取或迭代 test loader 前创建；同一 frozen config ID 的重复或并发
调用都会拒绝。若命令失败，也不要删除 claim 后重试；应保留审计记录并人工判定。

```bash
read -r FROZEN_CONFIG_ID CHECKPOINT_REL < <(
  python3 - "${PLUGIN_SELECTION}" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
print(selection["frozen_config_id"], selection["checkpoint_path"])
PY
)

CHECKPOINT="${PLUGIN_RUN}/${CHECKPOINT_REL}"
ID_MAPS="$(dirname "${CHECKPOINT}")/id_maps.json"
TEST_ATTEMPT=$(campaign_run test-once \
  --dataset-file "${SPLIT_DIR}/train.csv" \
  --dataset-file "${SPLIT_DIR}/valid.csv" \
  --dataset-file "${SPLIT_DIR}/test.csv" \
  --dataset-file "${CHECKPOINT}" \
  --dataset-file "${ID_MAPS}" \
  --dataset-file "${PLUGIN_SELECTION}" \
  --output-file model/metrics.json \
  --output-file model/predictions.csv \
  --output-file model/mastery.npy \
  --output-file model/id_maps.json \
  --output-file "${TEST_LEDGER}/${FROZEN_CONFIG_ID}.json" \
  -- bash -lc '
    python3 "${MODEL_RUNNER}" \
      --plugin-mode evaluate \
      --plugin-eval-split test \
      --plugin-checkpoint "$1" \
      --plugin-selection-json "$2" \
      --plugin-test-ledger-dir "$3" \
      --plugin-decouple \
      --plugin-aux-weight "${LAMBDA}" \
      --data_dir "${SPLIT_DIR}" \
      --train_file train.csv \
      --valid_file valid.csv \
      --test_file test.csv \
      --log_dir "${CAMPAIGN_ATTEMPT_DIR}/model" \
      --seed 42 \
      --plugin-doa-seed 42 \
      --plugin-min-responses 3 \
      --plugin-max-pairs-per-concept 100000 \
      --plugin-split-seed 2024' \
    test "${CHECKPOINT}" "${PLUGIN_SELECTION}" "${TEST_LEDGER}")
```

evaluate 模式输出 `metrics.json`、`predictions.csv`、`mastery.npy` 与
`id_maps.json`。`--plugin-eval-split valid` 可以重复执行且不需要 ledger；
`--plugin-eval-split test` 则必须同时提供 ledger 和冻结 selection JSON。
