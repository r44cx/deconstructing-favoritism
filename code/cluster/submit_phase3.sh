#!/bin/bash
set -e

EVAL_SET="/cluster/home/$USER/BA-HS25-zimmenoe/data/judge_eval/high_density_sample_150.json"

if [ ! -f "$EVAL_SET" ]; then
    echo "Evaluation set not found: $EVAL_SET"
    ls -lh /cluster/home/$USER/BA-HS25-zimmenoe/data/judge_eval/*.json 2>/dev/null || echo "No evaluation sets found"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

if python3 -c "import vllm" 2>/dev/null; then
    echo "vLLM is installed"
else
    echo "vLLM not found. It will be installed when the job runs."
fi

if [ -f "$HOME/.cache/huggingface/token" ]; then
    echo "HuggingFace token found"
else
    echo "HuggingFace token not found"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

read -p "Submit FP16 ablation job? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

JOB_ID=$(sbatch /cluster/home/$USER/BA-HS25-zimmenoe/code/cluster/phase3_fp16_ablation.submit | awk '{print $4}')

if [ -n "$JOB_ID" ]; then
    echo "Job submitted: $JOB_ID"
    echo "Monitor: squeue -u $USER"
else
    echo "Job submission failed"
    exit 1
fi
