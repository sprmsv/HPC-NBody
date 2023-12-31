#include "bruteforce.h"


// Particle-particle method using an arbitrary number of GPU threads per block
__host__ void nbodybruteforce(particle_t* array, int nbr_particles, int nbr_iterations, int threads_per_block)
{
	// Assign grid and block sizes
	dim3 bsz(threads_per_block);
	int extra_block = (nbr_particles % threads_per_block > 0) ? 1 : 0;
	dim3 gsz(nbr_particles / threads_per_block + extra_block);

	// Launch the kernels in each iteration and synchronize
	for (int it = 0; it < nbr_iterations; it++){
		compute_brute_force<<<gsz, bsz>>>(array, nbr_particles);
		cudaDeviceSynchronize();
		throw_last_gpu_error();
		update_positions<<<gsz, bsz>>>(array, nbr_particles);
		cudaDeviceSynchronize();
		throw_last_gpu_error();
	}
}

// Kernel for computing the forces, accelerations, and velocities of one particle
__global__ void compute_brute_force(particle_t* array, int nbr_particles)
{
	// Return if the thread is extra (can only happen in the last block)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nbr_particles) {
		return;
	}

	// Get the thread particle and copy its attributes
	particle_t* p = &array[idx];
	const double p_m = p->m;
	const int p_id = p->id;
	const double p_x = p->x[0];
	const double p_y = p->x[1];
	const double p_z = p->x[2];

	// Declare other local variables
	double x_sep, y_sep, z_sep, dist_sq, grav_base;
	double F_x = 0.;
	double F_y = 0.;
	double F_z = 0.;
	double a_x = 0.;
	double a_y = 0.;
	double a_z = 0.;

	// Loop over source particles and accumulate local forces
	for (int i = 0; i < nbr_particles; i++) {
		if (array[i].id == p_id) {
			continue;
		}
		x_sep = array[i].x[0] - p_x;
		y_sep = array[i].x[1] - p_y;
		z_sep = array[i].x[2] - p_z;
		dist_sq = max((x_sep * x_sep) + (y_sep * y_sep) + (z_sep * z_sep), 0.01);
		grav_base = GRAV_CONSTANT * (p_m) * (array[i].m) / dist_sq / sqrt(dist_sq);
		F_x += grav_base * x_sep;
		F_y += grav_base * y_sep;
		F_z += grav_base * z_sep;
	}

	// Compute accelerations locally
	a_x = F_x / p_m;
	a_y = F_y / p_m;
	a_z = F_z / p_m;

	// Increment particle velocities
	p->v[0] += a_x * TIMESTEP;
	p->v[1] += a_y * TIMESTEP;
	p->v[2] += a_z * TIMESTEP;
}

// Kernel for updating the position of one particle
__global__ void update_positions(particle_t* array, int nbr_particles)
{
	// Return if the thread is extra (can only happen in the last block)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= nbr_particles) {
		return;
	}

	// Get the thread particle
	particle_t* p = &array[idx];

	// Increment the position
	p->x[0] += p->v[0] * TIMESTEP;
	p->x[1] += p->v[1] * TIMESTEP;
	p->x[2] += p->v[2] * TIMESTEP;
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
