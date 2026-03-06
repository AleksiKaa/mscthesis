#!/bin/bash

echo "JOB START"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "PWD: $(pwd)"
echo "DATE: $(date)"

module load model-huggingface
module load scicomp-llm-env

echo "Modules loaded"

python -u src/scripts/generate_batched.py "$@"

echo "JOB END"