#!/bin/bash
#SBATCH --chdir=/home/kaariaa3/mscthesis
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB # This is system memory, not GPU memory.
#SBATCH --gpus=1
#SBATCH --gres=min-vram:32g
#SBATCH --output ./outputs/jobs/batch.%J.out
#SBATCH --error ./outputs/errs/batch.%J.err

module purge
module load model-huggingface
module load scicomp-llm-env

echo "Modules loaded"

python ./src/scripts/generate_batched.py $SLURM_JOB_ID "$@"