// (c) 2020, Vincent Keller (Vincent.Keller@epfl.ch)
// (c) 2023, Sepehr Mousavi (sepehr.mousavi@epfl.ch)

#include "parameters.h"
#include "reader.h"
#include "bruteforce.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstring>


typedef std::chrono::high_resolution_clock clk;
typedef std::chrono::duration<double> second;

void print_usage(char* name)
{
	std::cerr << "Usage: " << name << "<input-filename> <iterations> <threads-per-block>" << std::endl;
	exit(1);
}

int main(int argc, char** argv)
{
	// Check arguments
	if (argc != 4) {
		print_usage(argv[0]);
	}

	// Get device properties
	int dev_id = 0;
	cudaDeviceProp device_prop;
	cudaGetDevice(&dev_id);
	cudaGetDeviceProperties(&device_prop, dev_id);
	if (device_prop.computeMode == cudaComputeModeProhibited) {
		std::cerr
			<< "Error: device is running in <Compute Mode Prohibited>, "
			<< "no threads can use ::cudaSetDevice()"
			<< std::endl;
		return -1;
	}
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cout
			<< "cudaGetDeviceProperties returned error code "
			<< error
			<< ", line(" << __LINE__ << ")"
			<< std::endl;
		return error;
	} else {
		std::cout
			<< "GPU Device " << dev_id << ": \"" << device_prop.name
			<< "\" with compute capability " << device_prop.major << "." << device_prop.minor
			<< std::endl;
	}

	// Read arguments and input file
	int nbr_particles;
	int nbr_iterations;
	int threads_per_block;
	nbr_particles = get_nbr_particles(argv[1]);
	nbr_iterations = std::stoi(argv[2]);
	threads_per_block = std::stoi(argv[3]);
	if (threads_per_block > nbr_particles) {
		std::cerr
			<< "Cannot have more threads than particles"
			<< std::endl;
		return -1;
	}

	// Read particles and allocate on the unified memory
	particle_t* array;
	cudaMallocManaged(&array, nbr_particles*sizeof(particle_t));
	read_test_case(argv[1], array);

	// Run the simulation
	auto start = clk::now();
	nbodybruteforce(array, nbr_particles, nbr_iterations, threads_per_block);
	second time = clk::now() - start;

	// Free unified memory
	cudaFree(array);

	// Messages
	std::cout << argv[1] << ": "
			<< "BH " << 0 << "\t"
			<< "N " << nbr_particles << "\t"
			<< "ITERS " << nbr_iterations << "\t"
			<< "DT " << TIMESTEP << "\t"
			<< "SIZE " << SIZEOFSPACE << "\t"
			<< "THETA " << THETA << "\t"
			<< "TPB " << threads_per_block << "\t"
			<< std::scientific
			<< std::showpos
			<< std::setprecision(4)
			<< "TIME " << time.count() << "\t"
			<< std::endl;

	return 0;
}
