#ifndef READER_H_
#define READER_H_

#include "parameters.h"

#include <iostream>
#include <algorithm>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>


particle_t* read_test_case(const char* fn, int& nbr_particles, int psize);
int get_nbr_particles(const char* fn);
particle_t getMinMax(particle_t* array, int nbr_particles);

#endif /*READER_H_*/
