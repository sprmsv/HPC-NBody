#include "bruteforce.h"


// Particle-particle method using an arbitrary number of GPU threads per block 
__host__ void nbodybruteforce(particle_t* array, int nbr_particles, int nbr_iterations, int threads_per_block)
{
	// Assign grid and block sizes
	dim3 bsz(threads_per_block);
	dim3 gsz(nbr_particles / threads_per_block);
	const int threads_per_block_rem = nbr_particles % threads_per_block;

	// Launch the kernels in each iteration and synchronize
	double step = 1.;
	for (int it = 0; it < nbr_iterations; it++){
		compute_brute_force<<<gsz, bsz>>>(array, nbr_particles, step);
		if (threads_per_block_rem > 0) {
			int offset = gsz.x * threads_per_block;
			compute_brute_force<<<1, threads_per_block_rem>>>(array, nbr_particles, step, offset);
		}
		cudaDeviceSynchronize();
		throw_last_gpu_error();
		update_positions<<<gsz, bsz>>>(array, step);
		if (threads_per_block_rem > 0) {
			int offset = gsz.x * threads_per_block;
			update_positions<<<1, threads_per_block_rem>>>(array, step, offset);
		}
		cudaDeviceSynchronize();
		throw_last_gpu_error();
	}
}

// Kernel for computing the forces, accelerations, and velocities of one particle
__global__ void compute_brute_force(particle_t* array, int nbr_particles, double step, int offset)
{
	// TODO: Copy or reference or pointer?
	// Get the thread particle
	particle_t* p = &array[offset + blockIdx.x * blockDim.x + threadIdx.x];

	// Declare variables
	double x_sep, y_sep, z_sep, dist_sq, grav_base;
	double F_x = 0.;
	double F_y = 0.;
	double F_z = 0.;
	double a_x = 0.;
	double a_y = 0.;
	double a_z = 0.;

	// Loop over source particles and accumulate local forces
	for (int i = 0; i < nbr_particles; i++) {
		if (array[i].id == p->id) {
			continue;
		}
		x_sep = array[i].x[0] - p->x[0];
		y_sep = array[i].x[1] - p->x[1];
		z_sep = array[i].x[2] - p->x[2];
		dist_sq = max((x_sep * x_sep) + (y_sep * y_sep) + (z_sep * z_sep), 0.01);
		grav_base = GRAV_CONSTANT * (p->m) * (array[i].m) / dist_sq / sqrt(dist_sq);
		F_x += grav_base * x_sep;
		F_y += grav_base * y_sep;
		F_z += grav_base * z_sep;
	}

	// Compute accelerations locally 
	a_x = F_x / p->m;
	a_y = F_y / p->m;
	a_z = F_z / p->m;

	// Increment particle velocities
	p->v[0] += a_x * step;
	p->v[1] += a_y * step;
	p->v[2] += a_z * step;
}

// Kernel for updating the position of one particle
__global__ void update_positions(particle_t* array, double step, int offset)
{
	// TODO: Copy or reference or pointer?
	// Get the thread particle
	particle_t* p = &array[offset + blockIdx.x * blockDim.x + threadIdx.x];

	// Increment the position
	p->x[0] += p->v[0] * step;
	p->x[1] += p->v[1] * step;
	p->x[2] += p->v[2] * step;
}

// Get the last CUDA error and throw it as a runtime error
__host__ void throw_last_gpu_error()
{
	cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
			"Error Launching Kernel: "
            + std::string(cudaGetErrorName(error)) + " - "
            + std::string(cudaGetErrorString(error)));
    }
}
