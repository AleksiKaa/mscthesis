#!/bin/bash
#SBATCH --chdir=/home/kaariaa3/mscthesis
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB # This is system memory, not GPU memory.
#SBATCH --gpus=1
#SBATCH --gres=min-vram:80g
#SBATCH --output ./outputs/jobs/batch.%j.out
#SBATCH --error ./outputs/errs/batch.%j.err

module purge
module load model-huggingface
module load scicomp-llm-env

echo "Modules loaded"

python ./src/scripts/generate_batched.py $SLURM_JOB_ID "$@"