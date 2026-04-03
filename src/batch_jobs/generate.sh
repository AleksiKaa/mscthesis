#!/bin/bash

echo "Job $SLURM_JOB_ID running on $(hostname)"
echo "Working dir: $(pwd)"

while getopts ":s:" opt; do
    case $opt in
        s) SCRIPTFILE=$OPTARG ;;
        *) usage ;; 
    esac
done

module purge
module load model-huggingface
module load scicomp-llm-env
echo "Modules loaded"

echo "pycache disabled"
export PYTHONDONTWRITEBYTECODE=1

echo "Running script: $SCRIPTFILE"
python -u src/scripts/$SCRIPTFILE $SLURM_JOB_ID "$@"

echo "JOB END"
