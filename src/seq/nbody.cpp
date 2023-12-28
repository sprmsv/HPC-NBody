// (c) 2020, Vincent Keller (Vincent.Keller@epfl.ch)
// (c) 2023, Sepehr Mousavi (sepehr.mousavi@epfl.ch)

#include "parameters.h"
#include "barneshut.h"
#include "bruteforce.h"
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
	std::cerr << "Usage: " << name << "<input-filename> <iterations> [-bh OR -bf]" << std::endl;
	exit(1);
}

int main(int argc, char** argv)
{
	particle_t* array;
	int nbr_particles;
	int nbr_iterations;
	bool BARNESHUT = false;

	// Read method
	if (argc == 3) {
		BARNESHUT = false;
	}
	else if (argc == 4) {
		if (!strcmp(argv[3], "-bh")) {
			BARNESHUT = true;
		}
		else if (!strcmp(argv[3], "-bf")) {
			BARNESHUT = false;
		}
		else {
			print_usage(argv[0]);
		}
	}
	else {
		print_usage(argv[0]);
	}

	// Read input file
	nbr_particles = get_nbr_particles(argv[1]);
	array = read_test_case(argv[1]);
	nbr_iterations = std::stoi(argv[2]);

	// Run the simulation
	auto start = clk::now();
	if (BARNESHUT) {
		nbodybarneshut(array, nbr_particles, nbr_iterations);
	}
	else {
		nbodybruteforce(array, nbr_particles, nbr_iterations);
	}
	second time = clk::now() - start;

	// Free memory
	free(array);

	// Messages
	std::cout << argv[1] << ": "
			<< "BH " << BARNESHUT << "\t"
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

	return 0;
}
