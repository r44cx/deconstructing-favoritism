#!/bin/bash

USER=$(whoami)


echo "[1/6] Creating directory structure..."
mkdir -p /cluster/home/$USER/dev/faviscore
mkdir -p /cluster/home/$USER/judge_logs
mkdir -p /scratch/$USER/faviscore/data/judge_eval
mkdir -p /scratch/$USER/faviscore/data/judge_outputs
mkdir -p /raid/persistent_scratch/$USER/ollama_models
mkdir -p /raid/persistent_scratch/$USER/venvs



module load python/3.10.14

unset PIP_TARGET
unset PYTHONPATH

venv_path="/raid/persistent_scratch/$USER/venvs/faviscore_env"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

cp "$BASE_DIR/collect_judge_outputs.py" /cluster/home/$USER/dev/faviscore/
cp "$BASE_DIR/prepare_judge_data.py" /cluster/home/$USER/dev/faviscore/
cp "$BASE_DIR/submit_judge_collection.py" /cluster/home/$USER/dev/faviscore/
cp "$BASE_DIR/monitor_judge_jobs.py" /cluster/home/$USER/dev/faviscore/


export OLLAMA_MODELS=/raid/persistent_scratch/$USER/ollama_models

# Start Ollama server
/scratch/ollama/bin/ollama serve > /dev/null 2>&1 &
sleep 10

# Test with small model for quick verification
if /scratch/ollama/bin/ollama pull llama3.2:1b > /dev/null 2>&1; then
    /scratch/ollama/bin/ollama list
else
fi

kill $OLLAMA_PID 2>/dev/null || true
