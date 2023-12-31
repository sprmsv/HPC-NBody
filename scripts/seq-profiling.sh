#!/bin/bash

# RUN THIS FILE FROM THE ROOT DIRECTORY

# OUTPUT FOLDER
FOLDERNAME=$(date '+%Y%m%d%H%M%S')
FOLDER=output/$FOLDERNAME
mkdir -p $FOLDER

# BUILD
module purge
module load intel
make -C src/seq clean
make -C src/seq profiling

# RUN
sbatch --ntasks=1 -o $FOLDER/seq-profiling-bh.out scripts/job-seq-profiling.sh -bh $FOLDER/seq-profiling-bh.out
sbatch --ntasks=1 -o $FOLDER/seq-profiling-bf.out scripts/job-seq-profiling.sh -bf $FOLDER/seq-profiling-bf.out
