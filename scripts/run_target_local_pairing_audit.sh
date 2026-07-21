#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "decoupled_cd" ]]; then
  echo "Activate the decoupled_cd Conda environment before running." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Formal audit requires a clean worktree." >&2
  exit 2
fi

HEAD_COMMIT="$(git rev-parse HEAD)"
git fetch origin
if ! git branch -r --contains "$HEAD_COMMIT" | grep -qE "^[[:space:]]+origin/"; then
  echo "HEAD $HEAD_COMMIT is not contained in a fetched origin branch." >&2
  exit 2
fi

OUTPUT_ROOT="$1"
if ! git check-ignore -q "$OUTPUT_ROOT"; then
  echo "Output root must be covered by .gitignore so evaluation stays clean." >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  if [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Output root is not empty: $OUTPUT_ROOT" >&2
    exit 2
  fi
fi
STAGING_ROOT="${OUTPUT_ROOT}.staging-${HEAD_COMMIT:0:12}-$$"
if [[ -e "$STAGING_ROOT" ]]; then
  echo "Atomic staging root already exists: $STAGING_ROOT" >&2
  exit 2
fi
if ! git check-ignore -q "$STAGING_ROOT"; then
  echo "Atomic staging root must also be covered by .gitignore." >&2
  exit 2
fi
RUN_COMPLETED=0
preserve_failed_run() {
  status=$?
  trap - EXIT
  if (( RUN_COMPLETED == 0 )) && [[ -d "$STAGING_ROOT" ]]; then
    failed_root="${OUTPUT_ROOT}.failed-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    mv "$STAGING_ROOT" "$failed_root"
    echo "Incomplete artifacts preserved at $failed_root" >&2
  fi
  exit "$status"
}
trap preserve_failed_run EXIT
mkdir -p "$STAGING_ROOT/logs"

PROTOCOL_ROOT="results/goal_two_module/target_local_pairing_v15_protocol_e207b00"
declare -A SOURCE_DIRS=(
  [ASSIST17]="/home/xph/jwc/research/knofield_data/assist_17"
  [MOOCRadar]="/home/xph/jwc/research/knofield_data/moocradar"
  [XES3G5M]="/home/xph/jwc/research/knofield_data/xes3g5m"
  [Junyi]="/home/xph/jwc/research/knofield_data/pool_v2/junyi_standard_v2"
)
declare -A PROTOCOL_SHA256=(
  [ASSIST17]="3e884141c00d70e23df2f18da67193968774727d0182e6819dcb5a78e5d17621"
  [MOOCRadar]="77658461daf6a2771cb150620b7fad59aa89f03c807b456529fe9b246c340f4b"
  [XES3G5M]="959cc4e4e15874cf952a2724d05011fe220c0f4ae9d19a06db05b92c430af7c8"
  [Junyi]="b394005fab87df8d19ad8592c72a9dc8e540ca774c94b122d3c11e906509b4e5"
)
DATASETS=(ASSIST17 MOOCRadar XES3G5M Junyi)
MAX_PARALLEL=3

TASK_PEAK_MIB="${TARGET_LOCAL_TASK_PEAK_MIB:-4096}"
GPU_RESERVE_MIB="${TARGET_LOCAL_GPU_RESERVE_MIB:-1024}"
if ! [[ "$TASK_PEAK_MIB" =~ ^[0-9]+$ && "$GPU_RESERVE_MIB" =~ ^[0-9]+$ ]]; then
  echo "GPU memory estimates must be integer MiB values." >&2
  exit 2
fi

GPU_ROWS=()
while IFS=, read -r raw_index raw_free raw_util; do
  index="${raw_index//[[:space:]]/}"
  free="${raw_free//[[:space:]]/}"
  util="${raw_util//[[:space:]]/}"
  if [[ -z "$index" || -z "$free" || -z "$util" ]]; then
    continue
  fi
  available=$((free - GPU_RESERVE_MIB))
  if (( available < TASK_PEAK_MIB )); then
    continue
  fi
  score=$((free - 20 * util))
  GPU_ROWS+=("$score:$index:$free:$util")
done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

if (( ${#GPU_ROWS[@]} == 0 )); then
  echo "No GPU has the configured task peak plus reserve; refusing to alter the recipe." >&2
  exit 1
fi
mapfile -t GPU_ROWS < <(printf "%s\n" "${GPU_ROWS[@]}" | sort -t: -k1,1nr)
GPU_SLOTS=()
for row in "${GPU_ROWS[@]}"; do
  IFS=: read -r _score index _free _util <<< "$row"
  GPU_SLOTS+=("$index")
done
if (( ${#GPU_SLOTS[@]} > MAX_PARALLEL )); then
  GPU_SLOTS=("${GPU_SLOTS[@]:0:MAX_PARALLEL}")
fi

next_dataset=0
while (( next_dataset < ${#DATASETS[@]} )); do
  PIDS=()
  NAMES=()
  for gpu in "${GPU_SLOTS[@]}"; do
    if (( next_dataset >= ${#DATASETS[@]} )); then
      break
    fi
    dataset="${DATASETS[$next_dataset]}"
    prediction_dir="$STAGING_ROOT/predict/$dataset"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/target_local_pairing_runner.py predict \
      --dataset "$dataset" \
      --source-dir "${SOURCE_DIRS[$dataset]}" \
      --protocol-json "$PROTOCOL_ROOT/$dataset/summary.json" \
      --expected-protocol-sha256 "${PROTOCOL_SHA256[$dataset]}" \
      --expected-commit "$HEAD_COMMIT" \
      --output-dir "$prediction_dir" \
      --device cuda:0 \
      >"$STAGING_ROOT/logs/${dataset}_predict.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$dataset")
    next_dataset=$((next_dataset + 1))
  done
  failed=0
  for index in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$index]}"; then
      echo "Prediction failed for ${NAMES[$index]}; recipe was not changed." >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    exit 1
  fi
done

for dataset in "${DATASETS[@]}"; do
  python scripts/target_local_pairing_runner.py evaluate \
    --prediction-dir "$STAGING_ROOT/predict/$dataset" \
    --source-dir "${SOURCE_DIRS[$dataset]}" \
    --output-dir "$STAGING_ROOT/evaluate/$dataset" \
    >"$STAGING_ROOT/logs/${dataset}_evaluate.log" 2>&1
done

python scripts/target_local_pairing_runner.py aggregate \
  --evaluation-json "$STAGING_ROOT/evaluate/ASSIST17/evaluation.json" \
  --evaluation-json "$STAGING_ROOT/evaluate/MOOCRadar/evaluation.json" \
  --evaluation-json "$STAGING_ROOT/evaluate/XES3G5M/evaluation.json" \
  --evaluation-json "$STAGING_ROOT/evaluate/Junyi/evaluation.json" \
  --output-dir "$STAGING_ROOT/aggregate" \
  >"$STAGING_ROOT/logs/aggregate.log" 2>&1

if [[ -d "$OUTPUT_ROOT" ]]; then
  rmdir "$OUTPUT_ROOT"
fi
mv "$STAGING_ROOT" "$OUTPUT_ROOT"
RUN_COMPLETED=1
trap - EXIT

python - "$OUTPUT_ROOT/aggregate/activation_decision.json" <<"PY"
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({"activated": payload["activated"], "checks": payload["checks"]}, indent=2))
PY
