#!/bin/bash

#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:20:00

ITERS=20

srun src/mpi/nbody data/galaxy.txt $ITERS 1000
srun src/mpi/nbody data/galaxy.txt $ITERS 2000
srun src/mpi/nbody data/galaxy.txt $ITERS 5000
srun src/mpi/nbody data/galaxy.txt $ITERS 10000
srun src/mpi/nbody data/galaxy.txt $ITERS 20000
srun src/mpi/nbody data/galaxy.txt $ITERS 40000
