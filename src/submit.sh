#!/bin/bash

# Default values
TIME="01:00:00"
CPUS=4
MEM="16GB"
GPUS=1
VRAM="40g"
DIR=/home/kaariaa3/mscthesis
DEBUG=false
NOTE=""
PYTHONARGS=""
MODEL="Qwen/Qwen2.5-14B-Instruct"

usage() {
    echo "Usage: $0 [-t time] [-v vram] [-m model]"
    echo "Example: $0 -t 01:00:00 -v 32g -m Qwen/Qwen2.5-14B-Instruct"
    exit 1
}

while getopts "d:h:m:n:p:r:t:v:" opt; do
    case $opt in
        d) DEBUG=$OPTARG ;;
        h) usage ;;
        m) MODEL=$OPTARG ;;
        n) NOTE=$OPTARG ;;
        p) PYTHONARGS=$OPTARG ;;
        r) MEM=$OPTARG ;;
        t) TIME=$OPTARG ;;
        v) VRAM=$OPTARG ;;
        *) usage ;; 
    esac
done

OUTDIR=./outputs/$MODEL/jobs/batch.%j.out
ERRDIR=./outputs/$MODEL/errs/batch.%j.err

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
echo "Batch job: $BATCH_JOB"
echo "Note: $NOTE"

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
    $PYTHONARGS