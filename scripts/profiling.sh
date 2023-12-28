#!/bin/bash

# RUN THIS FILE FROM THE ROOT DIRECTORY

FOLDERNAME=$(date '+%Y%m%d%H%M%S')
FOLDER=output/$FOLDERNAME
mkdir -p $FOLDER

# RUN
module purge
module load intel
make -C src clean
make -C src profiling
sbatch --ntasks=1 -o $FOLDER/profiling-bh.out scripts/job-profiling.sh -bh $FOLDER/profiling-bh.out
sbatch --ntasks=1 -o $FOLDER/profiling-bf.out scripts/job-profiling.sh -bf $FOLDER/profiling-bf.out
