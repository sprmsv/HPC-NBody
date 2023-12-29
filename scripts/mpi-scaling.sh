#!/bin/bash

# RUN THIS FILE FROM THE ROOT DIRECTORY

# BUILD
module purge
module load intel intel-oneapi-mpi
make -C src/mpi clean
make -C src/mpi all

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
    sbatch --ntasks=1 -o $FOLDER/mpi-strongscaling-ntasks01.out scripts/job-mpi-strongscaling.sh
    sbatch --ntasks=2 -o $FOLDER/mpi-strongscaling-ntasks02.out scripts/job-mpi-strongscaling.sh
    sbatch --ntasks=4 -o $FOLDER/mpi-strongscaling-ntasks04.out scripts/job-mpi-strongscaling.sh
    sbatch --ntasks=8 -o $FOLDER/mpi-strongscaling-ntasks08.out scripts/job-mpi-strongscaling.sh
    sbatch --ntasks=16 -o $FOLDER/mpi-strongscaling-ntasks16.out scripts/job-mpi-strongscaling.sh
    sbatch --ntasks=32 -o $FOLDER/mpi-strongscaling-ntasks32.out scripts/job-mpi-strongscaling.sh
    sbatch --ntasks=64 --qos=parallel -o $FOLDER/mpi-strongscaling-ntasks64.out scripts/job-mpi-strongscaling.sh

    # WEAK SCALING
    sbatch --ntasks=1 -o $FOLDER/mpi-weakscaling-ntasks01.out scripts/job-mpi-weakscaling.sh 964
    sbatch --ntasks=2 -o $FOLDER/mpi-weakscaling-ntasks02.out scripts/job-mpi-weakscaling.sh 1770
    sbatch --ntasks=4 -o $FOLDER/mpi-weakscaling-ntasks04.out scripts/job-mpi-weakscaling.sh 3274
    sbatch --ntasks=8 -o $FOLDER/mpi-weakscaling-ntasks08.out scripts/job-mpi-weakscaling.sh 6081
    sbatch --ntasks=16 -o $FOLDER/mpi-weakscaling-ntasks16.out scripts/job-mpi-weakscaling.sh 11350
    sbatch --ntasks=32 -o $FOLDER/mpi-weakscaling-ntasks32.out scripts/job-mpi-weakscaling.sh 21268
    sbatch --ntasks=64 --qos=parallel -o $FOLDER/mpi-weakscaling-ntasks64.out scripts/job-mpi-weakscaling.sh 40000

    ((counter++))
done
