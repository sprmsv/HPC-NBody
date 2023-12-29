#!/bin/bash

#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:05:00

ITERS=20
PARTICLES=$1

srun src/mpi/nbody data/galaxy.txt $ITERS $PARTICLES
