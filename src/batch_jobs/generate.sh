#!/bin/bash -l

echo "Job $SLURM_JOB_ID running on $(hostname)"
echo "Working dir: $(pwd)"

module purge
module load model-huggingface
module load scicomp-llm-env

echo "Modules loaded"

python -u src/scripts/generate_batched.py "$@"

echo "JOB END"
