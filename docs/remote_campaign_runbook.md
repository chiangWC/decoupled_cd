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
- 显式列出本次使用的所有 `--dataset-file`，用于记录 SHA-256。
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
（环境可提供时）以及声明输出和命令日志的哈希。`completed` 表示退出码为 0；
`failed` 保留非零退出码；`dry_run` 的退出码为 `null`。
