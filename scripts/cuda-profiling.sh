#!/bin/bash

# RUN THIS FILE FROM THE ROOT DIRECTORY

# BUILD
module purge
module load gcc cuda
make -C src/cuda clean
make -C src/cuda all

# OUTPUT FOLDER
FOLDERNAME=$(date '+%Y%m%d%H%M%S')
FOLDER=output/$FOLDERNAME
mkdir -p $FOLDER

# STRONG SCALING
sbatch -o $FOLDER/cuda-profiling.out scripts/job-cuda-profiling.sh
