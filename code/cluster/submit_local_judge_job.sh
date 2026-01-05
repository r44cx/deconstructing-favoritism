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

JUDGE_MODEL=${JUDGE_MODEL:-"tinyllama:latest"}
MODE=${MODE:-"pairwise"}
[ -z "$EVALUATION_SET" ] && echo "Error: --evaluation-set required" && exit 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_FILE="$REPO_ROOT/data/judge_eval/$EVALUATION_SET"
[ ! -f "$EVAL_FILE" ] && echo "Error: $EVAL_FILE not found" && exit 1

export JUDGE_MODEL="$JUDGE_MODEL"
export EVALUATION_SET="$EVALUATION_SET"
export MODE="$MODE"

"$SCRIPT_DIR/local_judge_job.sh"

