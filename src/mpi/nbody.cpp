// (c) 2020, Vincent Keller (Vincent.Keller@epfl.ch)
// (c) 2023, Sepehr Mousavi (sepehr.mousavi@epfl.ch)

#include "parameters.h"
#include "barneshut.h"
#include "reader.h"

#include <chrono>
#include <iostream>
#include <iomanip>

#include <cstring>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef std::chrono::high_resolution_clock clk;
typedef std::chrono::duration<double> second;

void print_usage(char* name)
{
	std::cerr << "Usage: " << name << "<input-filename> <iterations> <particles>" << std::endl;
	exit(1);
}

int main(int argc, char** argv)
{
	// Initialize MPI
	int prank, psize;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);

	// Declare variables
	particle_t* array;
	int nbr_particles;
	int nbr_iterations;

	// Check arguments
	if (argc != 4) {
		print_usage(argv[0]);
	}

	// Read input file
	nbr_particles = get_nbr_particles(argv[1]);
	if (std::stoi(argv[3]) > nbr_particles) {
		print_usage(argv[0]);
	}
	else {
		nbr_particles = std::stoi(argv[3]);
	}
	array = read_test_case(argv[1], nbr_particles, psize);
	nbr_iterations = std::stoi(argv[2]);

	// Run the simulation
	auto start = clk::now();
	nbodybarneshut(array, nbr_particles, nbr_iterations, psize, prank);
	second time = clk::now() - start;

	// Free memory
	free(array);

	// Messages
	if (prank == 0) {
		std::cout << argv[1] << ": "
			<< "BH " << 1 << "\t"
			<< "N " << nbr_particles << "\t"
			<< "ITERS " << nbr_iterations << "\t"
			<< "DT " << TIMESTEP << "\t"
			<< "SIZE " << SIZEOFSPACE << "\t"
			<< "THETA " << THETA << "\t"
			<< std::scientific
			<< std::showpos
			<< std::setprecision(4)
			<< "TIME " << time.count() << "\t"
			<< std::endl;
	}

	// Finalize MPI
	MPI_Finalize();

	return 0;
}
