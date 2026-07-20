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
    [--stage requirement_gate|full_factorial] \
    [--devices cuda:0,cuda:2,cuda:3] [--max-parallel 3] \
    [--expected-commit <sha>] [--gate-approved] [--execute]

The default requirement_gate stage contains only eight w/o-Requirement jobs.
The 14-job full_factorial stage requires an explicit --gate-approved after the
pre-registered Requirement gate passes. Without --execute the script only
prints its selected plan. Validation uses valid.csv as the train.py test
placeholder, so no test file is opened and --evaluation-stage validation never
computes test metrics.
EOF
}

DATA_ROOT="${KNOFIELD_DATA_ROOT:-}"
LEGACY_RESULT_ROOT=""
OUTPUT_ROOT="results/goal_two_module/factorized_requirement_factorial_v10"
DEVICES="cpu"
MAX_PARALLEL=1
EXPECTED_COMMIT=""
STAGE="requirement_gate"
GATE_APPROVED=0
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
        --gate-approved)
            GATE_APPROVED=1
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

if [[ -z "$DATA_ROOT" ]]; then
    echo "--data-root or KNOFIELD_DATA_ROOT is required." >&2
    exit 2
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
    echo "--max-parallel must be positive." >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
if [[ "${#DEVICE_LIST[@]}" -lt 1 ]]; then
    echo "--devices must contain at least one device." >&2
    exit 2
fi

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

# Stage 2 fills only the remaining 14 cells after Stage 1 passes. Full exists
# for all eight cells; the strong History control already exists for ASSIST17
# holdout and Junyi holdout.
FULL_FACTORIAL_TASKS=(
    "assist17|standard|wo_history"
    "assist17|standard|wo_both"
    "assist17|holdout|wo_both"
    "moocradar|standard|wo_history"
    "moocradar|standard|wo_both"
    "moocradar|holdout|wo_history"
    "moocradar|holdout|wo_both"
    "xes3g5m|standard|wo_history"
    "xes3g5m|standard|wo_both"
    "xes3g5m|holdout|wo_history"
    "xes3g5m|holdout|wo_both"
    "junyi|standard|wo_history"
    "junyi|standard|wo_both"
    "junyi|holdout|wo_both"
)

case "$STAGE" in
    requirement_gate)
        TASKS=("${REQUIREMENT_GATE_TASKS[@]}")
        ;;
    full_factorial)
        TASKS=("${FULL_FACTORIAL_TASKS[@]}")
        ;;
    *)
        echo "--stage must be requirement_gate or full_factorial." >&2
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

CURRENT_BRANCH="$(git branch --show-current)"
CURRENT_HEAD="$(git rev-parse HEAD)"
printf 'stage=%s tasks=%d execute=%d max_parallel=%d branch=%s head=%s\n' \
    "$STAGE" "${#TASKS[@]}" "$EXECUTE" "$MAX_PARALLEL" \
    "$CURRENT_BRANCH" "$CURRENT_HEAD"

if [[ "$EXECUTE" -eq 0 ]]; then
    for index in "${!TASKS[@]}"; do
        task="${TASKS[$index]}"
        device="${DEVICE_LIST[$((index % ${#DEVICE_LIST[@]}))]}"
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
if [[ -z "$EXPECTED_COMMIT" ]]; then
    echo "Formal execution requires --expected-commit." >&2
    exit 4
fi
if [[ -z "$LEGACY_RESULT_ROOT" ]]; then
    echo "Formal execution requires --legacy-result-root." >&2
    exit 4
fi
if [[ "$STAGE" == "full_factorial" && "$GATE_APPROVED" -ne 1 ]]; then
    echo "full_factorial requires --gate-approved." >&2
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
python scripts/audit_factorized_requirement_artifacts.py \
    --manifest configs/factorized_requirement_artifacts.json \
    --artifact-root "$LEGACY_RESULT_ROOT" \
    --output "$OUTPUT_ROOT/locked_artifact_audit_${STAGE}.json" \
    > "$OUTPUT_ROOT/locked_artifact_audit_${STAGE}.stdout.log"

pids=()
labels=()
failures=0
for index in "${!TASKS[@]}"; do
    while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
        sleep 1
    done
    task="${TASKS[$index]}"
    device="${DEVICE_LIST[$((index % ${#DEVICE_LIST[@]}))]}"
    run_task "$task" "$device" &
    pids+=("$!")
    labels+=("$task")
done

for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        echo "FAILED: ${labels[$index]}" >&2
        failures=$((failures + 1))
    fi
done
if [[ "$failures" -ne 0 ]]; then
    echo "$failures factorial task(s) failed." >&2
    exit 5
fi
