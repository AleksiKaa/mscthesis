#!/bin/bash

# This script checks the status of SLURM jobs for a specific batch of jobs for different models 
# by looking for error logs and counting the number of started, finished, and failed jobs.
# Expects the first argument to be the batch name (e.g., "v1", "default", "test", etc.)

models=(
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-32B"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
)

for model in "${models[@]}"; do

    failed_jobs=($(grep -HRls 'Traceback (most recent call last):' ./outputs/$1/$model/*/*.err))

    echo "Model: $model"
    echo "Number of started jobs: $(find . -path "./outputs/$1/$model/*/*.json" | wc -l)"
    echo "Number of finished jobs: $(find . -path "./outputs/$1/$model/*/*.csv" | wc -l)"
    echo "Number of failed jobs: ${#failed_jobs[@]}"

    if [ ${#failed_jobs[@]} -ne 0 ]; then
        echo "Failed job details:"
        for job in "${failed_jobs[@]}"; do
            echo "Job ID: $(basename $(dirname $job))"
        done
    fi
    echo "-----------------------------"

done