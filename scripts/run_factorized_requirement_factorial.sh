#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Plan or execute the missing validation-only jobs in the 4-dataset 2x2
History x Requirement factorial.

Usage:
  bash scripts/run_factorized_requirement_factorial.sh \
    --data-root /path/to/knofield_data \
    --legacy-result-root /path/to/results/goal_two_module \
    [--output-root results/goal_two_module/factorized_requirement_factorial_v10] \
    [--stage requirement_gate|paired_full|history_gate] \
    [--devices cuda:0,cuda:2,cuda:3] [--max-parallel 3] \
    [--expected-commit <sha>] [--execute]
  bash scripts/run_factorized_requirement_factorial.sh \
    --devices self_gpu0,self_gpu1 --max-parallel 9 \
    --scheduler-self-test

The default requirement_gate stage contains only eight w/o-Requirement jobs.
The paired_full stage reruns the eight Full cells on the current code and data
pipeline so that a gate never compares against legacy checkpoints produced by
different preprocessing semantics.
The history_gate stage contains only four holdout w/o-both jobs. Its matched
History-Full anchors are the existing <dataset>_holdout_wo_requirement
artifacts, so both sides use factorized_item_control and differ only in History
representation. The rejected Requirement full-factorial stage is deliberately
unavailable. Without --execute the script only prints its selected plan.
Validation uses valid.csv as the train.py test placeholder, so no test file is
opened and --evaluation-stage validation never computes test metrics.
EOF
}

DATA_ROOT="${KNOFIELD_DATA_ROOT:-}"
LEGACY_RESULT_ROOT=""
OUTPUT_ROOT="results/goal_two_module/factorized_requirement_factorial_v10"
DEVICES="cpu"
MAX_PARALLEL=1
EXPECTED_COMMIT=""
STAGE="requirement_gate"
SCHEDULER_SELF_TEST=0
EXECUTE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --legacy-result-root)
            LEGACY_RESULT_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --expected-commit)
            EXPECTED_COMMIT="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --scheduler-self-test)
            SCHEDULER_SELF_TEST=1
            shift
            ;;
        --execute)
            EXECUTE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$SCHEDULER_SELF_TEST" -eq 0 && -z "$DATA_ROOT" ]]; then
    echo "--data-root or KNOFIELD_DATA_ROOT is required." >&2
    exit 2
fi
if [[ ! "$MAX_PARALLEL" =~ ^[0-9]+$ || "$MAX_PARALLEL" -lt 1 ]]; then
    echo "--max-parallel must be positive." >&2
    exit 2
fi

IFS=',' read -r -a RAW_DEVICE_LIST <<< "$DEVICES"
declare -A SEEN_DEVICES=()
DEVICE_LIST=()
for device in "${RAW_DEVICE_LIST[@]}"; do
    if [[ -z "$device" ]]; then
        echo "--devices cannot contain an empty device." >&2
        exit 2
    fi
    if [[ -z "${SEEN_DEVICES[$device]+x}" ]]; then
        SEEN_DEVICES["$device"]=1
        DEVICE_LIST+=("$device")
    fi
done
if [[ "${#DEVICE_LIST[@]}" -lt 1 ]]; then
    echo "--devices must contain at least one device." >&2
    exit 2
fi
REQUESTED_MAX_PARALLEL="$MAX_PARALLEL"
if [[ "$MAX_PARALLEL" -gt "${#DEVICE_LIST[@]}" ]]; then
    MAX_PARALLEL="${#DEVICE_LIST[@]}"
fi
DEVICE_SLOTS=("${DEVICE_LIST[@]:0:MAX_PARALLEL}")

# Stage 1 asks the life-or-death Requirement question before paying for the
# entire 2x2. Legacy target-ID-removing controls are intentionally not reused.
REQUIREMENT_GATE_TASKS=(
    "assist17|standard|wo_requirement"
    "assist17|holdout|wo_requirement"
    "moocradar|standard|wo_requirement"
    "moocradar|holdout|wo_requirement"
    "xes3g5m|standard|wo_requirement"
    "xes3g5m|holdout|wo_requirement"
    "junyi|standard|wo_requirement"
    "junyi|holdout|wo_requirement"
)

# A data-pipeline change can preserve raw file hashes while changing the
# tensors consumed by a model (for example, reconstructing all exercise
# concepts from Q). These jobs provide a commit-matched Full for a clean gate.
PAIRED_FULL_TASKS=(
    "assist17|standard|full_current"
    "assist17|holdout|full_current"
    "moocradar|standard|full_current"
    "moocradar|holdout|full_current"
    "xes3g5m|standard|full_current"
    "xes3g5m|holdout|full_current"
    "junyi|standard|full_current"
    "junyi|holdout|full_current"
)

# Requirement failed its Q-consistent gate. The only remaining clean question
# is whether calibrated History beats calibrated summary when both paths use
# the same factorized Requirement control. Existing *_holdout_wo_requirement
# predictions are the matched History-Full anchors.
HISTORY_GATE_TASKS=(
    "assist17|holdout|wo_both"
    "moocradar|holdout|wo_both"
    "xes3g5m|holdout|wo_both"
    "junyi|holdout|wo_both"
)

case "$STAGE" in
    requirement_gate)
        TASKS=("${REQUIREMENT_GATE_TASKS[@]}")
        ;;
    paired_full)
        TASKS=("${PAIRED_FULL_TASKS[@]}")
        ;;
    history_gate)
        TASKS=("${HISTORY_GATE_TASKS[@]}")
        ;;
    *)
        echo "--stage must be requirement_gate, paired_full, or history_gate." >&2
        exit 2
        ;;
esac

dataset_recipe() {
    local dataset="$1"
    local split="$2"
    case "$dataset:$split" in
        assist17:standard)
            DATASET_DIR="$DATA_ROOT/assist_17"
            EPOCHS=80
            STUDENT_BATCH=128
            LEARNING_RATE=0.002
            EARLY_STOP=15
            SCHEDULER_PATIENCE=5
            ;;
        assist17:holdout)
            DATASET_DIR="$DATA_ROOT/assist_17_chold_v2"
            EPOCHS=80
            STUDENT_BATCH=128
            LEARNING_RATE=0.002
            EARLY_STOP=15
            SCHEDULER_PATIENCE=5
            ;;
        moocradar:standard)
            DATASET_DIR="$DATA_ROOT/moocradar"
            EPOCHS=40
            STUDENT_BATCH=64
            LEARNING_RATE=0.001
            EARLY_STOP=10
            SCHEDULER_PATIENCE=10
            ;;
        moocradar:holdout)
            DATASET_DIR="$DATA_ROOT/moocradar_chold_v2"
            EPOCHS=40
            STUDENT_BATCH=64
            LEARNING_RATE=0.001
            EARLY_STOP=10
            SCHEDULER_PATIENCE=10
            ;;
        xes3g5m:standard)
            DATASET_DIR="$DATA_ROOT/xes3g5m"
            EPOCHS=40
            STUDENT_BATCH=64
            LEARNING_RATE=0.001
            EARLY_STOP=10
            SCHEDULER_PATIENCE=10
            ;;
        xes3g5m:holdout)
            DATASET_DIR="$DATA_ROOT/xes3g5m_chold_v2"
            EPOCHS=40
            STUDENT_BATCH=64
            LEARNING_RATE=0.001
            EARLY_STOP=10
            SCHEDULER_PATIENCE=10
            ;;
        junyi:standard)
            DATASET_DIR="$DATA_ROOT/pool_v2/junyi_standard_v2"
            EPOCHS=40
            STUDENT_BATCH=128
            LEARNING_RATE=0.001
            EARLY_STOP=10
            SCHEDULER_PATIENCE=10
            ;;
        junyi:holdout)
            DATASET_DIR="$DATA_ROOT/pool_v2/junyi_chold_v2"
            EPOCHS=40
            STUDENT_BATCH=128
            LEARNING_RATE=0.001
            EARLY_STOP=10
            SCHEDULER_PATIENCE=10
            ;;
        *)
            echo "Unsupported task: $dataset $split" >&2
            return 2
            ;;
    esac
}

variant_modes() {
    local variant="$1"
    case "$variant" in
        full_current)
            EVIDENCE_MODE="calibrated_history"
            REQUIREMENT_MODE="exercise_specific"
            ;;
        wo_history)
            EVIDENCE_MODE="calibrated_summary_control"
            REQUIREMENT_MODE="exercise_specific"
            ;;
        wo_requirement)
            EVIDENCE_MODE="calibrated_history"
            REQUIREMENT_MODE="factorized_item_control"
            ;;
        wo_both)
            EVIDENCE_MODE="calibrated_summary_control"
            REQUIREMENT_MODE="factorized_item_control"
            ;;
        *)
            echo "Unsupported variant: $variant" >&2
            return 2
            ;;
    esac
}

build_train_command() {
    local dataset="$1"
    local split="$2"
    local variant="$3"
    local device="$4"
    dataset_recipe "$dataset" "$split"
    variant_modes "$variant"
    local stem="$OUTPUT_ROOT/${dataset}_${split}_${variant}"
    TRAIN_COMMAND=(
        python scripts/train.py
        --model two_stage_tkc_ukc
        --train-interactions "$DATASET_DIR/train.csv"
        --valid-interactions "$DATASET_DIR/valid.csv"
        --test-interactions "$DATASET_DIR/valid.csv"
        --q-matrix "$DATASET_DIR/Q_matrix.csv"
        --concept-dim 64
        --epochs "$EPOCHS"
        --student-batch-size "$STUDENT_BATCH"
        --learning-rate "$LEARNING_RATE"
        --weight-decay 0
        --training-mode student_recompute_minibatch
        --early-stop-patience "$EARLY_STOP"
        --lr-scheduler-patience "$SCHEDULER_PATIENCE"
        --lr-scheduler-factor 0.5
        --min-learning-rate 1e-5
        --checkpoint-selection-metric auc
        --checkpoint-selection-start-epoch 1
        --checkpoint-selection-window 1
        --semantic-node-mode bidirectional_q
        --evidence-representation-mode "$EVIDENCE_MODE"
        --evidence-refinement-mode identity_passthrough
        --target-requirement-mode "$REQUIREMENT_MODE"
        --concept-prior-mode population_q
        --state-completion-mode personalized_interaction
        --diagnosis-mode target_conditioned
        --completion-evidence-cap 20
        --seed 42
        --evaluation-stage validation
        --device "$device"
        --log-dir "$OUTPUT_ROOT/logs"
        --output "${stem}.json"
    )
    ANALYZE_COMMAND=(
        python scripts/analyze_prediction_slices.py
        --summary "${stem}.json"
        --split valid
        --device "$device"
        --output "${stem}_slices.json"
        --csv-output "${stem}_slices.csv"
        --prediction-output "${stem}_predictions.csv"
    )
}

run_task() {
    local task="$1"
    local device="$2"
    IFS='|' read -r dataset split variant <<< "$task"
    build_train_command "$dataset" "$split" "$variant" "$device"
    local stem="$OUTPUT_ROOT/${dataset}_${split}_${variant}"
    if [[ -e "${stem}.json" || -e "${stem}_best.pt" ]]; then
        echo "Refusing to overwrite existing artifact: $stem" >&2
        return 3
    fi
    mkdir -p "$OUTPUT_ROOT/logs"
    {
        printf 'Running %s on %s\n' "$task" "$device"
        "${TRAIN_COMMAND[@]}"
        "${ANALYZE_COMMAND[@]}"
    } > "${stem}.runner.log" 2>&1
}

declare -A ACTIVE_DEVICE_BY_PID=()
declare -A ACTIVE_LABEL_BY_PID=()
AVAILABLE_DEVICES=()
ACTIVE_COUNT=0
SCHEDULER_FAILURES=0
TASK_RUNNER_FUNCTION="run_task"
SCHEDULER_TRACE=""

wait_for_completed_slot() {
    local completed_pid=""
    local status=0
    local device
    local label
    if wait -n -p completed_pid "${!ACTIVE_DEVICE_BY_PID[@]}"; then
        status=0
    else
        status=$?
    fi
    if [[ -z "$completed_pid" ]]; then
        echo "Scheduler could not identify the completed job." >&2
        return 6
    fi
    if [[ -z "${ACTIVE_DEVICE_BY_PID[$completed_pid]+x}" ]]; then
        echo "Scheduler could not identify the completed job." >&2
        return 6
    fi
    device="${ACTIVE_DEVICE_BY_PID[$completed_pid]}"
    label="${ACTIVE_LABEL_BY_PID[$completed_pid]}"
    unset 'ACTIVE_DEVICE_BY_PID[$completed_pid]'
    unset 'ACTIVE_LABEL_BY_PID[$completed_pid]'
    ACTIVE_COUNT=$((ACTIVE_COUNT - 1))
    AVAILABLE_DEVICES+=("$device")
    if [[ "$status" -ne 0 ]]; then
        echo "FAILED: $label on $device (status=$status)" >&2
        SCHEDULER_FAILURES=$((SCHEDULER_FAILURES + 1))
    fi
}

run_scheduled_tasks() {
    local task
    local device
    local pid
    ACTIVE_DEVICE_BY_PID=()
    ACTIVE_LABEL_BY_PID=()
    AVAILABLE_DEVICES=("${DEVICE_SLOTS[@]}")
    ACTIVE_COUNT=0
    SCHEDULER_FAILURES=0

    for task in "${TASKS[@]}"; do
        if [[ "$ACTIVE_COUNT" -ge "$MAX_PARALLEL" ]]; then
            wait_for_completed_slot
        fi
        if [[ "${#AVAILABLE_DEVICES[@]}" -lt 1 ]]; then
            echo "Scheduler has no free device slot." >&2
            return 6
        fi
        device="${AVAILABLE_DEVICES[0]}"
        AVAILABLE_DEVICES=("${AVAILABLE_DEVICES[@]:1}")
        if [[ -n "$SCHEDULER_TRACE" ]]; then
            printf '%s|%s\n' "$task" "$device" >> "$SCHEDULER_TRACE"
        fi
        "$TASK_RUNNER_FUNCTION" "$task" "$device" &
        pid=$!
        ACTIVE_DEVICE_BY_PID["$pid"]="$device"
        ACTIVE_LABEL_BY_PID["$pid"]="$task"
        ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
    done

    while [[ "$ACTIVE_COUNT" -gt 0 ]]; do
        wait_for_completed_slot
    done
    if [[ "$SCHEDULER_FAILURES" -ne 0 ]]; then
        echo "$SCHEDULER_FAILURES scheduled task(s) failed." >&2
        return 5
    fi
}

self_test_task() {
    local task="$1"
    local device="$2"
    local lock_dir="$SELF_TEST_ROOT/locks/$device"
    if ! mkdir "$lock_dir"; then
        printf '%s|%s\n' "$task" "$device" >> "$SELF_TEST_ROOT/collisions"
        return 9
    fi
    case "$task" in
        first_short)
            sleep 0.05
            ;;
        first_long)
            sleep 0.30
            ;;
        *)
            sleep 0.02
            ;;
    esac
    rmdir "$lock_dir"
}

run_scheduler_self_test() {
    local third_assignment
    SELF_TEST_ROOT="$(mktemp -d)"
    mkdir "$SELF_TEST_ROOT/locks"
    SCHEDULER_TRACE="$SELF_TEST_ROOT/assignments"
    TASK_RUNNER_FUNCTION="self_test_task"
    TASKS=(
        "first_short"
        "first_long"
        "follow_short"
        "follow_final"
    )
    DEVICE_SLOTS=("self_gpu0" "self_gpu1")
    MAX_PARALLEL=2
    if ! run_scheduled_tasks; then
        echo "scheduler_self_test=FAIL" >&2
        return 1
    fi
    if [[ -e "$SELF_TEST_ROOT/collisions" ]]; then
        echo "scheduler_self_test=FAIL collision_detected" >&2
        return 1
    fi
    third_assignment="$(sed -n '3p' "$SCHEDULER_TRACE")"
    if [[ "$third_assignment" != "follow_short|self_gpu0" ]]; then
        echo "scheduler_self_test=FAIL actual=$third_assignment" >&2
        return 1
    fi
    rm "$SCHEDULER_TRACE"
    rmdir "$SELF_TEST_ROOT/locks"
    rmdir "$SELF_TEST_ROOT"
    echo "scheduler_self_test=PASS effective_parallel=2 reused=self_gpu0"
}

if [[ "$SCHEDULER_SELF_TEST" -eq 1 ]]; then
    run_scheduler_self_test
    exit 0
fi

CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_HEAD="$(git rev-parse HEAD)"
printf 'stage=%s tasks=%d execute=%d requested_parallel=%d effective_parallel=%d branch=%s head=%s\n' \
    "$STAGE" "${#TASKS[@]}" "$EXECUTE" "$REQUESTED_MAX_PARALLEL" \
    "$MAX_PARALLEL" "$CURRENT_BRANCH" "$CURRENT_HEAD"

if [[ "$EXECUTE" -eq 0 ]]; then
    for index in "${!TASKS[@]}"; do
        task="${TASKS[$index]}"
        device="${DEVICE_SLOTS[$((index % ${#DEVICE_SLOTS[@]}))]}"
        IFS='|' read -r dataset split variant <<< "$task"
        build_train_command "$dataset" "$split" "$variant" "$device"
        printf '%02d\t%s\t%s\t%s\n' \
            "$((index + 1))" "$dataset" "$split" "$variant"
        printf '  '
        printf '%q ' "${TRAIN_COMMAND[@]}"
        printf '\n'
    done
    exit 0
fi

if [[ -n "$(git status --short)" ]]; then
    echo "Formal execution requires a clean worktree." >&2
    exit 4
fi
REMOTE_CONTAINING_HEAD="$(git branch -r --contains "$CURRENT_HEAD" 2>/dev/null || true)"
if [[ -z "$REMOTE_CONTAINING_HEAD" ]]; then
    echo "Formal execution requires HEAD to be present on a fetched remote ref." >&2
    exit 4
fi
if [[ -z "$EXPECTED_COMMIT" ]]; then
    echo "Formal execution requires --expected-commit." >&2
    exit 4
fi
if [[ "$STAGE" != "history_gate" && -z "$LEGACY_RESULT_ROOT" ]]; then
    echo "Formal execution requires --legacy-result-root." >&2
    exit 4
fi
RESOLVED_EXPECTED_COMMIT="$(
    git rev-parse --verify "${EXPECTED_COMMIT}^{commit}" 2>/dev/null || true
)"
if [[ -z "$RESOLVED_EXPECTED_COMMIT" || "$CURRENT_HEAD" != "$RESOLVED_EXPECTED_COMMIT" ]]; then
    echo "HEAD does not match --expected-commit." >&2
    echo "branch=$CURRENT_BRANCH head=$CURRENT_HEAD expected=$EXPECTED_COMMIT" >&2
    exit 4
fi
mkdir -p "$OUTPUT_ROOT"
printf 'stage=%s\nbranch=%s\nhead=%s\nexpected_commit=%s\n' \
    "$STAGE" "$CURRENT_BRANCH" "$CURRENT_HEAD" \
    "$RESOLVED_EXPECTED_COMMIT" \
    > "$OUTPUT_ROOT/runner_identity_${STAGE}.log"
if [[ "$STAGE" == "history_gate" ]]; then
    python scripts/audit_history_gate_anchors.py \
        --artifact-root "$OUTPUT_ROOT" \
        --output "$OUTPUT_ROOT/locked_history_anchor_audit.json" \
        > "$OUTPUT_ROOT/locked_history_anchor_audit.stdout.log"
else
    python scripts/audit_factorized_requirement_artifacts.py \
        --manifest configs/factorized_requirement_artifacts.json \
        --artifact-root "$LEGACY_RESULT_ROOT" \
        --output "$OUTPUT_ROOT/locked_artifact_audit_${STAGE}.json" \
        > "$OUTPUT_ROOT/locked_artifact_audit_${STAGE}.stdout.log"
fi

TASK_RUNNER_FUNCTION="run_task"
if ! run_scheduled_tasks; then
    exit 5
fi
