#!/bin/bash

echo "Job $SLURM_JOB_ID running on $(hostname)"
echo "Working dir: $(pwd)"

module purge
module load model-huggingface
echo "Modules loaded"

source activate thesis
echo "Conda environment activated"

echo "pycache disabled"
export PYTHONDONTWRITEBYTECODE=1

python -u src/scripts/generate_batched.py $SLURM_JOB_ID "$@"

echo "JOB END"
