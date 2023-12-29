# BARNESHUT MPI

0. Review the lecture slides

2. Write report + Amdhal’s and Gustafson’s laws

# BRUTEFORCE CUDA

0. Review the lecture slides

1. An accelerated version of the particle-particle method using CUDA with hand written kernels.

2. Write report + Amdhal’s and Gustafson’s laws

# Data

# Report

- Profiling
- Which parts parallelizable?
- Sequential fraction for theoretical bounds
- Amdhal’s and Gustafson’s laws

# Submission

- Report (max 4 pages)

- Slides (max 3 slides)
    1. Strong scaling with Amdhal’s prediction (MPI).
    2. Weak scaling with Gustafson’s prediction (MPI).
    3. The influence of grid and block sizes on the performance of the CUDA version.

- Codes (tarball): Two different folders


# INFO

Regarding the MPI parallelization of the n-body problem, in the final project,
it is not required to partition the bodies' tree across the different MPI process.
For achieving good scalability beyond a few processes, it is certainly needed,
but for the sake of the course it is enough to keep a local copy of the full tree
in every single process.

Then, every single MPI process will be in charge of computing
the forces on a group of particles (assigned to that process) against the full tree.

After, every process will have to update its local tree using the new positions of all
the particles.
