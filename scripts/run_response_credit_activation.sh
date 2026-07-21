#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {stage1|stage2} OUTPUT_ROOT" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi
STAGE="$1"
if [[ "$STAGE" != "stage1" && "$STAGE" != "stage2" ]]; then
  usage
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ "${CONDA_DEFAULT_ENV:-}" != "decoupled_cd" ]]; then
  echo "Activate the decoupled_cd Conda environment before running." >&2
  exit 2
fi
PYTHON_BIN="$(command -v python)"
if [[ "$PYTHON_BIN" != "${CONDA_PREFIX:-}/bin/python" ]]; then
  echo "python does not come from the active decoupled_cd environment." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Formal activation requires a clean worktree." >&2
  exit 2
fi

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "Formal activation requires a named current branch." >&2
  exit 2
fi
HEAD_COMMIT="$(git rev-parse HEAD)"
git fetch --quiet origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
FETCHED_ORIGIN="$(git rev-parse "refs/remotes/origin/$BRANCH")"
LIVE_ORIGIN="$(
  git ls-remote --heads origin "refs/heads/$BRANCH" | awk 'NR == 1 {print $1}'
)"
if [[ -z "$LIVE_ORIGIN" || "$HEAD_COMMIT" != "$FETCHED_ORIGIN" || "$HEAD_COMMIT" != "$LIVE_ORIGIN" ]]; then
  echo "HEAD, fetched origin, and live origin branch must be identical." >&2
  exit 2
fi

OUTPUT_ROOT="$(realpath -m "$2")"
case "$OUTPUT_ROOT/" in
  "$REPO_ROOT"/results/*) ;;
  *)
    echo "OUTPUT_ROOT must be below the repository results directory." >&2
    exit 2
    ;;
esac
RELATIVE_ROOT="${OUTPUT_ROOT#"$REPO_ROOT"/}"
if ! git check-ignore -q -- "$RELATIVE_ROOT/provenance.json"; then
  echo "Formal result root must be covered by .gitignore." >&2
  exit 2
fi

declare -A HOLDOUT_SOURCES=(
  [MOOCRadar]="/home/xph/jwc/research/knofield_data/moocradar_chold_v2"
  [NIPS34]="/home/xph/jwc/research/knofield_data/nips34_chold_v2"
)
declare -A STANDARD_SOURCES=(
  [MOOCRadar]="/home/xph/jwc/research/knofield_data/moocradar"
  [NIPS34]="/home/xph/jwc/research/knofield_data/nips34_clean"
)
DATASETS=(MOOCRadar NIPS34)
VARIANTS=(full direct capacity)

if [[ "$STAGE" == "stage1" ]]; then
  SPLIT_KIND="holdout"
  STAGE_DIR="$OUTPUT_ROOT/stage1"
  STAGE1_JSON=""
  if [[ -e "$OUTPUT_ROOT" ]]; then
    if [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]]; then
      echo "Stage 1 output root already contains artifacts: $OUTPUT_ROOT" >&2
      exit 2
    fi
  fi
else
  SPLIT_KIND="standard"
  STAGE_DIR="$OUTPUT_ROOT/stage2"
  STAGE1_JSON="$OUTPUT_ROOT/stage1/aggregate/activation_decision.json"
  if [[ ! -f "$OUTPUT_ROOT/stage1/COMPLETE" || ! -f "$STAGE1_JSON" ]]; then
    echo "Stage 2 requires a completed Stage 1 in the same stable root." >&2
    exit 2
  fi
  "$PYTHON_BIN" - "$STAGE1_JSON" "$HEAD_COMMIT" <<'PY'
from pathlib import Path
import sys

from scripts.run_response_credit_activation import validate_stage1_decision_artifact

validate_stage1_decision_artifact(Path(sys.argv[1]), expected_commit=sys.argv[2])
PY
fi
if [[ -e "$STAGE_DIR" ]]; then
  echo "Stage output already exists; refusing to overwrite: $STAGE_DIR" >&2
  exit 2
fi

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/.staging"
RUNNING_PIDS=()
RUN_COMPLETED=0
preserve_failure() {
  local status="$1"
  trap - EXIT INT TERM
  for pid in "${RUNNING_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${RUNNING_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if (( RUN_COMPLETED == 0 )); then
    touch "$STAGE_DIR/FAILED"
    echo "Incomplete artifacts and logs preserved under $STAGE_DIR" >&2
  fi
  exit "$status"
}
trap 'preserve_failure $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

assert_frozen_repository() {
  local current_branch current_head live
  current_branch="$(git branch --show-current)"
  current_head="$(git rev-parse HEAD)"
  if [[ "$current_branch" != "$BRANCH" || "$current_head" != "$HEAD_COMMIT" ]]; then
    echo "Repository branch or HEAD changed during the formal run." >&2
    return 1
  fi
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "Worktree changed during the formal run." >&2
    return 1
  fi
  live="$(git ls-remote --heads origin "refs/heads/$BRANCH" | awk 'NR == 1 {print $1}')"
  if [[ "$live" != "$HEAD_COMMIT" ]]; then
    echo "Live origin branch changed during the formal run." >&2
    return 1
  fi
}

source_for_dataset() {
  local dataset="$1"
  if [[ "$SPLIT_KIND" == "holdout" ]]; then
    echo "${HOLDOUT_SOURCES[$dataset]}"
  else
    echo "${STANDARD_SOURCES[$dataset]}"
  fi
}

run_prediction_task() {
  local dataset="$1"
  local variant="$2"
  local gpu="$3"
  local source_dir final_dir staging_dir log_path
  source_dir="$(source_for_dataset "$dataset")"
  final_dir="$STAGE_DIR/predict/$dataset/$variant"
  staging_dir="$STAGE_DIR/.staging/predict/${dataset}-${variant}"
  log_path="$STAGE_DIR/logs/${dataset}_${variant}_predict.log"
  if [[ -e "$final_dir" || -e "$staging_dir" ]]; then
    echo "Prediction path already exists for $dataset/$variant." >&2
    return 2
  fi
  mkdir -p "$(dirname "$final_dir")" "$(dirname "$staging_dir")"
  command=(
    "$PYTHON_BIN" scripts/run_response_credit_activation.py predict
    --dataset "$dataset"
    --source-dir "$source_dir"
    --split-kind "$SPLIT_KIND"
    --variant "$variant"
    --output-dir "$staging_dir"
    --device cuda:0
    --epochs 20
    --expected-commit "$HEAD_COMMIT"
  )
  if [[ "$STAGE" == "stage2" ]]; then
    command+=(--stage1-json "$STAGE1_JSON")
  fi
  echo "Starting $dataset/$variant on physical GPU $gpu" >>"$STAGE_DIR/logs/scheduler.log"
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" >"$log_path" 2>&1
  touch "$staging_dir/COMPLETE"
  mv "$staging_dir" "$final_dir"
  echo "Completed $dataset/$variant on physical GPU $gpu" >>"$STAGE_DIR/logs/scheduler.log"
}

# The one-epoch smoke measured a 14.1 GiB worst-case peak. Formal jobs launch
# only with at least 18 GiB currently free, and this scheduler assigns at most
# one task to each physical GPU. GPU utilization affects preference, not
# eligibility: an occupied card is allowed when it still has enough memory.
MIN_FREE_MIB=18432
POLL_SECONDS=15
TASK_DATASETS=(NIPS34 NIPS34 NIPS34 MOOCRadar MOOCRadar MOOCRadar)
TASK_VARIANTS=(full direct capacity full direct capacity)
NEXT_TASK=0
declare -A PID_GPU=()
declare -A PID_NAME=()

while (( NEXT_TASK < ${#TASK_DATASETS[@]} || ${#RUNNING_PIDS[@]} > 0 )); do
  remaining_pids=()
  for pid in "${RUNNING_PIDS[@]}"; do
    state="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -n "$state" && "$state" != Z* ]]; then
      remaining_pids+=("$pid")
      continue
    fi
    if ! wait "$pid"; then
      echo "Prediction failed (${PID_NAME[$pid]}); stopping without retry." >&2
      exit 1
    fi
    unset "PID_GPU[$pid]" "PID_NAME[$pid]"
  done
  RUNNING_PIDS=("${remaining_pids[@]}")

  declare -A BUSY_GPUS=()
  for pid in "${RUNNING_PIDS[@]}"; do
    BUSY_GPUS["${PID_GPU[$pid]}"]=1
  done
  launched=0
  GPU_ROWS=()
  while IFS=, read -r raw_index raw_free raw_util; do
    index="${raw_index//[[:space:]]/}"
    free="${raw_free//[[:space:]]/}"
    util="${raw_util//[[:space:]]/}"
    if [[ -z "$index" || -z "$free" || -z "$util" || -n "${BUSY_GPUS[$index]:-}" ]]; then
      continue
    fi
    if (( free < MIN_FREE_MIB )); then
      continue
    fi
    score=$((free - 32 * util))
    GPU_ROWS+=("$score:$index:$free:$util")
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  if (( ${#GPU_ROWS[@]} > 0 )); then
    mapfile -t GPU_ROWS < <(printf '%s\n' "${GPU_ROWS[@]}" | sort -t: -k1,1nr)
  fi
  for row in "${GPU_ROWS[@]}"; do
    if (( NEXT_TASK >= ${#TASK_DATASETS[@]} )); then
      break
    fi
    IFS=: read -r _score gpu free util <<<"$row"
    assert_frozen_repository
    dataset="${TASK_DATASETS[$NEXT_TASK]}"
    variant="${TASK_VARIANTS[$NEXT_TASK]}"
    echo "Dispatching $dataset/$variant: gpu=$gpu free=${free}MiB util=${util}%" \
      >>"$STAGE_DIR/logs/scheduler.log"
    run_prediction_task "$dataset" "$variant" "$gpu" &
    pid=$!
    RUNNING_PIDS+=("$pid")
    PID_GPU["$pid"]="$gpu"
    PID_NAME["$pid"]="$dataset/$variant gpu=$gpu"
    NEXT_TASK=$((NEXT_TASK + 1))
    launched=1
  done
  if (( launched == 0 )); then
    if (( ${#RUNNING_PIDS[@]} == 0 && NEXT_TASK < ${#TASK_DATASETS[@]} )); then
      echo "Waiting: no GPU currently has ${MIN_FREE_MIB} MiB free." \
        >>"$STAGE_DIR/logs/scheduler.log"
    fi
    sleep "$POLL_SECONDS"
  fi
done
assert_frozen_repository
PREDICTION_ARGS=()
for dataset in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    prediction_dir="$STAGE_DIR/predict/$dataset/$variant"
    if [[ ! -f "$prediction_dir/COMPLETE" ]]; then
      echo "Missing completed prediction: $dataset/$variant" >&2
      exit 1
    fi
    PREDICTION_ARGS+=(--prediction-dir "$prediction_dir")
  done
done
BARRIER_STAGING="$STAGE_DIR/.staging/prediction-barrier"
BARRIER_FINAL="$STAGE_DIR/prediction-barrier"
"$PYTHON_BIN" scripts/run_response_credit_activation.py prediction-barrier \
  "${PREDICTION_ARGS[@]}" \
  --split-kind "$SPLIT_KIND" \
  --output-dir "$BARRIER_STAGING" \
  >"$STAGE_DIR/logs/prediction_barrier.log" 2>&1
touch "$BARRIER_STAGING/COMPLETE"
mv "$BARRIER_STAGING" "$BARRIER_FINAL"

# This is the first point at which validation outcomes may be opened.
for dataset in "${DATASETS[@]}"; do
  assert_frozen_repository
  evaluation_staging="$STAGE_DIR/.staging/evaluate-$dataset"
  evaluation_final="$STAGE_DIR/evaluate/$dataset"
  mkdir -p "$(dirname "$evaluation_final")"
  "$PYTHON_BIN" scripts/run_response_credit_activation.py evaluate \
    --full-dir "$STAGE_DIR/predict/$dataset/full" \
    --direct-dir "$STAGE_DIR/predict/$dataset/direct" \
    --capacity-dir "$STAGE_DIR/predict/$dataset/capacity" \
    --source-dir "$(source_for_dataset "$dataset")" \
    --output-dir "$evaluation_staging" \
    --bootstrap-replicates 2000 \
    >"$STAGE_DIR/logs/${dataset}_evaluate.log" 2>&1
  touch "$evaluation_staging/COMPLETE"
  mv "$evaluation_staging" "$evaluation_final"
done

assert_frozen_repository
AGGREGATE_STAGING="$STAGE_DIR/.staging/aggregate"
AGGREGATE_FINAL="$STAGE_DIR/aggregate"
aggregate_command=(
  "$PYTHON_BIN" scripts/run_response_credit_activation.py aggregate
  --stage "$STAGE"
  --evaluation-json "$STAGE_DIR/evaluate/MOOCRadar/evaluation.json"
  --evaluation-json "$STAGE_DIR/evaluate/NIPS34/evaluation.json"
  --output-dir "$AGGREGATE_STAGING"
)
if [[ "$STAGE" == "stage2" ]]; then
  aggregate_command+=(--stage1-json "$STAGE1_JSON")
fi
"${aggregate_command[@]}" >"$STAGE_DIR/logs/aggregate.log" 2>&1
touch "$AGGREGATE_STAGING/COMPLETE"
mv "$AGGREGATE_STAGING" "$AGGREGATE_FINAL"
touch "$STAGE_DIR/COMPLETE"
RUN_COMPLETED=1
trap - EXIT INT TERM

"$PYTHON_BIN" - "$AGGREGATE_FINAL/activation_decision.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({
    "gate": payload["gate"],
    "passed": payload.get("stage1_passed", payload.get("activated")),
    "checks": payload["checks"],
}, indent=2))
PY
