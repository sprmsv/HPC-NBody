
# BRUTEFORCE CUDA

0. Review the lecture slides

1. An accelerated version of the particle-particle method using CUDA with hand written kernels.

2. Write report + Amdhal’s and Gustafson’s laws

*. Make sure to do these:
    - (?) Split dimensions as well? 
    - allocate memory with cudaMemcpy
    - cudaDeviceSynchronize (join)
    - constant memory: __device__ __constant__ OR cudaMemcpyToSymbol()
    - Block shared memory
        - Very fast
        - In V100, up to 128 KB per SM, but a maximum of 48 KB per block.
        - --> __shared__
        - threads have to be synchronized explicitly  -> __syncthreads();
        - Coalesced memory access by warp: contiguous, in-order, aligned
        - Avoid bank conflict
    - Keep time with events as in matrix_mul.cu

# Submission

- Report (max 4 pages)

- Slides (max 3 slides)
    1. Strong scaling with Amdhal’s prediction (MPI).
    2. Weak scaling with Gustafson’s prediction (MPI).
    3. The influence of grid and block sizes on the performance of the CUDA version.

- Codes (tarball): Two different folders
