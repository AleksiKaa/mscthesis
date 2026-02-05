#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB # This is system memory, not GPU memory.
#SBATCH --gpus=1
#SBATCH --output huggingface.%J.out
#SBATCH --error huggingface.%J.err

# By loading the model-huggingface module, models will be loaded from /scratch/shareddata/dldata/huggingface-hub-cache which is a shared scratch space.
module load model-huggingface

# Load a ready to use conda environment to use HuggingFace Transformers
module load scicomp-llm-env

python huggingface_example.py