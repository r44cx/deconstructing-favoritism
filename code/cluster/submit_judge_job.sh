#!/bin/bash
set -e

# Parse command line arguments
JUDGE_MODEL=""
EVALUATION_SET=""
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --evaluation-set) EVALUATION_SET="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Usage: $0 --judge-model MODEL --evaluation-set FILE [--mode pairwise|absolute|both]"; exit 1 ;;
    esac
done

[ -z "$JUDGE_MODEL" ] && echo "Error: --judge-model required" && exit 1
[ -z "$EVALUATION_SET" ] && echo "Error: --evaluation-set required" && exit 1

MODE=${MODE:-"pairwise"}

EVAL_FILE="/cluster/home/$USER/BA-HS25-zimmenoe/data/judge_eval/$EVALUATION_SET"
[ ! -f "$EVAL_FILE" ] && echo "Error: $EVAL_FILE not found" && exit 1

SUBMIT_SCRIPT="$(dirname "$0")/judge_collection.submit"
[ ! -f "$SUBMIT_SCRIPT" ] && echo "Error: $SUBMIT_SCRIPT not found" && exit 1

JUDGE_NORMALIZED=$(echo "$JUDGE_MODEL" | sed 's/:/_/g' | sed 's/-/_/g' | sed 's/\//_/g')
EVAL_NORMALIZED=$(echo "$EVALUATION_SET" | sed 's/\.json$//')
OUTPUT_FILE="/cluster/home/$USER/BA-HS25-zimmenoe/data/judge_outputs/${EVAL_NORMALIZED}_${JUDGE_NORMALIZED}_outputs.json"

echo "Judge: $JUDGE_MODEL"
echo "Eval: $EVALUATION_SET"
echo "Mode: $MODE"
echo "Output: $OUTPUT_FILE"

JOB_ID=$(JUDGE_MODEL="$JUDGE_MODEL" EVALUATION_SET="$EVALUATION_SET" MODE="$MODE" sbatch "$SUBMIT_SCRIPT" | awk '{print $4}')

echo "Job: $JOB_ID"
echo "Logs: /cluster/home/$USER/judge_logs/${JOB_ID}_*_judge_collection.{out,err}"
