#!/bin/bash

# Default values
TIME="01:00:00"
CPUS=4
MEM="16GB"
GPUS=1
VRAM="32g"
DIR=/home/kaariaa3/mscthesis
OUTDIR=./outputs/jobs/batch.%j.out
ERRDIR=./outputs/errs/batch.%j.err
MODEL=Qwen/Qwen2.5-14B-Instruct
NROWS=-1
CASE="augment"
SAVE="True"
FILE="/home/kaariaa3/mscthesis/data/complete_dataset.csv"
DEBUG=false

usage() {
    echo "Usage: $0 [-t time] [-v vram] [-m model]"
    echo "Example: $0 -t 01:00:00 -v 32g -m Qwen/Qwen2.5-14B-Instruct"
    exit 1
}

while getopts "t:v:m:n:c:s:d:h" opt; do
    case $opt in
        t) TIME=$OPTARG ;;
        v) VRAM=$OPTARG ;;
        m) MODEL=$OPTARG ;;
        n) NROWS=$OPTARG ;;
        c) CASE=$OPTARG ;;
        s) SAVE=$OPTARG ;;
        d) DEBUG=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

BATCH_JOB=./src/batch_jobs/generate.sh
if [ $DEBUG = true ]; then
  BATCH_JOB=./src/batch_jobs/debug.sh
  TIME="00:05:00"
  CPUS=1
  MEM="1GB"
  VRAM="1g"
fi

echo "Submitting job with:"
echo "Time: $TIME"
echo "CPUs: $CPUS"
echo "Memory: $MEM"
echo "GPU VRAM: $VRAM"
echo "Model: $MODEL"
echo "Mode: $CASE"
echo "Number of rows: $NROWS"
echo "Batch job: $BATCH_JOB"

sbatch \
    --chdir="$DIR" \
    --time="$TIME" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --gpus="$GPUS" \
    --gres=min-vram:$VRAM \
    --output="$OUTDIR" \
    --error="$ERRDIR" \
    $BATCH_JOB \
    -f "$FILE" \
    -m "$MODEL" \
    -c "$SAVE"  \
    -t "$CASE" \
    -n "$NROWS" \