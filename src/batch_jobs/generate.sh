#!/bin/bash

# By loading the model-huggingface module, models will be loaded from /scratch/shareddata/dldata/huggingface-hub-cache which is a shared scratch space.
module load model-huggingface

# Load a ready to use conda environment to use HuggingFace Transformers
module load scicomp-llm-env

python ../scripts/generate_batched.py $SLURM_JOB_ID "$@"