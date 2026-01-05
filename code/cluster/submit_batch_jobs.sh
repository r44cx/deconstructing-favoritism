#!/bin/bash
set -e

# Parse command line arguments
JUDGES=""
TARGETS=""
MAX_JOBS=10
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --judges)
            JUDGES="$2"
            shift 2
            ;;
        --targets)
            TARGETS="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$JUDGES" ]; then
    echo "Error: --judges required"
    echo "Usage: $0 --judges MODEL1,MODEL2 --targets TARGET1,TARGET2 [--max-jobs N]"
    exit 1
fi

IFS=',' read -ra JUDGE_ARRAY <<< "$JUDGES"
[ -n "$TARGETS" ] && IFS=',' read -ra TARGET_ARRAY <<< "$TARGETS"

EVAL_DIR="/cluster/home/$USER/BA-HS25-zimmenoe/data/judge_eval"
[ ! -d "$EVAL_DIR" ] && echo "Error: $EVAL_DIR not found" && exit 1

EVAL_SETS=()
if [ -n "$TARGETS" ]; then
    for target in "${TARGET_ARRAY[@]}"; do
        target_norm=$(echo $target | sed 's/-/_/g' | sed 's/:/_/g')
        for file in "$EVAL_DIR"/*"$target_norm"*.json; do
            [ -f "$file" ] && EVAL_SETS+=($(basename "$file"))
        done
    done
else
    for file in "$EVAL_DIR"/*.json; do
        [ -f "$file" ] && [[ $(basename "$file") != "preparation_summary.json" ]] && EVAL_SETS+=($(basename "$file"))
    done
fi

EVAL_SETS=($(printf '%s\n' "${EVAL_SETS[@]}" | sort -u | head -$MAX_JOBS))
[ ${#EVAL_SETS[@]} -eq 0 ] && echo "Error: No evaluation sets found" && exit 1

TOTAL_JOBS=$((${#JUDGE_ARRAY[@]} * ${#EVAL_SETS[@]}))
if [ $TOTAL_JOBS -gt $MAX_JOBS ]; then
    LIMITED_SETS=$((MAX_JOBS / ${#JUDGE_ARRAY[@]}))
    EVAL_SETS=($(printf '%s\n' "${EVAL_SETS[@]}" | head -$LIMITED_SETS))
    TOTAL_JOBS=$((${#JUDGE_ARRAY[@]} * ${#EVAL_SETS[@]}))
fi

echo "Submitting $TOTAL_JOBS jobs"
[ "$DRY_RUN" = true ] && echo "DRY RUN MODE" && exit 0

SUBMITTED=0
FAILED=0
for judge in "${JUDGE_ARRAY[@]}"; do
    for eval_set in "${EVAL_SETS[@]}"; do
        JUDGE_MODEL="$judge" EVALUATION_SET="$eval_set" sbatch "$(dirname "$0")/judge_collection.submit" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            SUBMITTED=$((SUBMITTED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done
done

echo "Submitted: $SUBMITTED, Failed: $FAILED"
