#!/bin/bash

# RUN THIS FILE FROM THE ROOT DIRECTORY

# BUILD
module purge
module load gcc cuda
make -C src/cuda clean
make -C src/cuda all

REPEAT=10
counter=0
TIMENOW=$(date '+%Y%m%d%H%M%S')
while [ $counter -lt $REPEAT ]
do 
    # OUTPUT FOLDER
    FOLDERNAME=$TIMENOW-$counter
    FOLDER=output/$FOLDERNAME
    mkdir -p $FOLDER

    # STRONG SCALING
    sbatch -o $FOLDER/cuda-tpb.out scripts/job-cuda-tpb.sh

    ((counter++))
done
