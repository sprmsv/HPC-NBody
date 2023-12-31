#ifndef NBODYBRUTEFORCE_H_
#define NBODYBRUTEFORCE_H_

#include "parameters.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <string>
#include <exception>
#include <algorithm>


__host__ void nbodybruteforce(particle_t* array, int nbr_particles, int nbr_iterations, int threads_per_block);
__host__ void throw_last_gpu_error();

__global__ void compute_brute_force(particle_t* array, int nbr_particles);
__global__ void update_positions(particle_t* array, int nbr_particles);


#endif /*NBODYBRUTEFORCE_H_*/
