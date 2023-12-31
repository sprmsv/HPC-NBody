#!/bin/bash

#SBATCH --reservation=Course-math-454-final
#SBATCH --account=math-454
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_free

ITERS=20

srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 1
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 2
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 4
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 8
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 16
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 32
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 64
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 128
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 256
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 512
srun nvprof src/cuda/nbody data/galaxy.txt $ITERS 1024
