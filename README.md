# HPC-NBody
Parallelizing sequential algorithms for solving the N-Body problem using MPI and CUDA

## How to compile and run

```bash
module load gcc
make -C ./src all
src/nbody data/very-small.txt -bh
# srun -N 1 -n 1 --account=math-454 --reservation=Course-math-454-final ./nbody-code examples/very-small.txt
```
