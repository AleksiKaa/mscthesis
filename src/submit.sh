#!/bin/bash

# Default values
TIME="01:00:00"
CPUS=4
MEM="16GB"
GPUS=1
VRAM="40g"
DIR=/home/kaariaa3/mscthesis
OUTDIR=./outputs/jobs/batch.%j.out
ERRDIR=./outputs/errs/batch.%j.err
MODEL=Qwen/Qwen2.5-14B-Instruct
NROWS=-1
NDEMOS=0
CASE="detect"
SAVE=True
FILE="/home/kaariaa3/mscthesis/data/final_dataset.csv"
DEBUG=false
FIXEDDEMOS=False

usage() {
    echo "Usage: $0 [-t time] [-v vram] [-m model]"
    echo "Example: $0 -t 01:00:00 -v 32g -m Qwen/Qwen2.5-14B-Instruct"
    exit 1
}

while getopts "c:d:h:m:n:p:r:s:t:v:x:" opt; do
    case $opt in
        c) CASE=$OPTARG ;;
        d) DEBUG=$OPTARG ;;
        h) usage ;;
        m) MODEL=$OPTARG ;;
        n) NROWS=$OPTARG ;;
        p) NDEMOS=$OPTARG ;;
        r) MEM=$OPTARG ;;
        s) SAVE=$OPTARG ;;
        t) TIME=$OPTARG ;;
        v) VRAM=$OPTARG ;;
        x) FIXEDDEMOS=$OPTARG ;;
        *) usage ;; 
    esac
done

BATCH_JOB=./src/batch_jobs/generate.sh
if [ "$DEBUG" = true ]; then
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
echo "Number of demos per prompt: $NDEMOS"
echo "Batch job: $BATCH_JOB"
echo "Use fixed demos: $FIXEDDEMOS"

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
    -n $NROWS \
    -d $NDEMOS \
    --fixed_demos $FIXEDDEMOS