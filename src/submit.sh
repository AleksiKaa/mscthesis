#!/bin/bash

# Default values
TIME="01:00:00"
CPUS=4
MEM="16GB"
GPUS=1
VRAM="32g"
DIR=/home/kaariaa3/mscthesis
OUTDIR=./outputs/jobs/batch.%J.out
ERRDIR=./outputs/errs/batch.%J.err
MODEL=Qwen/Qwen2.5-14B-Instruct
NROWS=-1
CASE="augment"
SAVE="True"
FILE="/home/kaariaa3/mscthesis/data/complete_dataset.csv"

usage() {
    echo "Usage: $0 [-t time] [-v vram] [-m model]"
    echo "Example: $0 -t 01:00:00 -v 32g -m Qwen/Qwen2.5-14B-Instruct"
    exit 1
}

while getopts "t:v:m:n:c:s:h" opt; do
    case $opt in
        t) TIME=$OPTARG ;;
        v) VRAM=$OPTARG ;;
        m) MODEL=$OPTARG ;;
        n) NROWS=$OPTARG ;;
        c) CASE=$OPTARG ;;
        s) SAVE=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

echo "Submitting job with:"
echo "Time: $TIME"
echo "CPUs: $CPUS"
echo "Memory: $MEM"
echo "GPU VRAM: $VRAM"
echo "Model: $MODEL"
echo "Mode: $CASE"
echo "Number of rows: $NROWS"

sbatch \
    --chdir="$DIR" \
    --time="$TIME" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --gpus="$GPUS" \
    --gres=min-vram:$VRAM \
    --output="$OUTDIR" \
    --error="$ERRDIR" \
    ./src/batch_jobs/generate.sh \
    -m "$MODEL" \
    -n "$NROWS" \
    -t "$CASE" \
    -s "$SAVE" \
    -f "$FILE"