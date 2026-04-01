#!/bin/bash

echo "HOST: $(hostname)"
echo "PWD: $(pwd)"

echo "Args: $@"

echo "Module command:"
type module

echo "MODULEPATH:"
echo $MODULEPATH

#echo "Available modules:"
#module avail

#echo "Load Core"
#module load Core                # optional, if required by cluster

echo "Loading modules..."
module load model-huggingface

echo "Activating conda environment..."
source activate thesis

mamba env list

echo "Loaded modules:"
module list