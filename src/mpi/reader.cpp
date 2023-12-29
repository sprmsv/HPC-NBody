#include "reader.h"

/*
Reads an XYZ file containing the particles and their masses.
Use : https://github.com/martinsparre/Gadget2Conversion to read an IC's file example from Gadget 2

File format for this function is :

Line 1 		: N = number of particles
Line 2 -> N	: x : y : z : vx : vy : vz : m : ID : V

where :
	(x,y,z)		: position of particle ID
	(vx,vy,vz)	: velocity of particle ID
	m		: mass of particle ID
	ID		: unique ID number
	V		: potential (not used in this simulation)
*/

particle_t* read_test_case(const char* fn, int& nbr_particles, int psize)
{
	particle_t* mat;
	int nbr_particles_file;
	float x, y, z, vx, vy, vz, m, V;
	int id;
	FILE* f;

	if ((f = fopen(fn, "r")) == NULL)
	{
		std::cerr << "ERROR: Could not open file" << std::endl;
		exit(1);
	}
	rewind(f);
	fscanf(f, "%d", &nbr_particles_file);
	if (nbr_particles == 0) {
		nbr_particles = nbr_particles_file;
	}
	mat = (particle_t*) malloc(nbr_particles * sizeof(particle_t));

	for (int i = 0; i < nbr_particles; i++) {
		fscanf(f, "%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%f", &x, &y, &z, &vx, &vy, &vz, &m, &id, &V);
		mat[i].x[0] = x;
		mat[i].x[1] = y;
		mat[i].x[2] = z;
		mat[i].v[0] = vx;
		mat[i].v[1] = vy;
		mat[i].v[2] = vz;
		if (m==0.) m = 1.;
		mat[i].m = m;
		mat[i].id = id;
		mat[i].prank = (id - 1) % psize;
		mat[i].V = V;
		mat[i].parent = NULL;
		mat[i].f[0] = 0.;
		mat[i].f[1] = 0.;
		mat[i].f[2] = 0.;
	}

	if (f !=stdin) fclose(f);

	return mat;
}

int get_nbr_particles(const char* fn)
{
	int nbr_part = 0;
	FILE* f;

	if ((f = fopen(fn, "r")) == NULL)
	{
		std::cerr << "ERROR: Could not open file" << std::endl;
		exit(1);
	}
	fscanf(f, "%d", &nbr_part);

	if (f != stdin) fclose(f);

	return nbr_part;
}

particle_t getMinMax(particle_t* array, int nbr_particles)
{
	int i;
	double minx = DBL_MAX;
	double maxx = DBL_MIN;
	double miny = DBL_MAX;
	double maxy = DBL_MIN;
	double minz = DBL_MAX;
	double maxz = DBL_MIN;
	double maxt, mint;
	particle_t particle_minmax;

	for (i = 0; i < nbr_particles; i++)
	{
		if (array[i].x[0] < minx) minx = array[i].x[0];
		if (array[i].x[0] > maxx) maxx = array[i].x[0];
		if (array[i].x[1] < miny) miny = array[i].x[1];
		if (array[i].x[1] > maxy) maxy = array[i].x[1];
		if (array[i].x[2] < minz) minz = array[i].x[2];
		if (array[i].x[2] > maxz) maxz = array[i].x[2];
	}

	maxt = std::max(maxx, maxy);
	maxt = std::max(maxt, maxz);
	mint = std::min(minx, miny);
	mint = std::min(mint, minz);

	// NOTE: This will fail if min and max have the same sign
	particle_minmax.x[0] = mint * SIZEOFSPACE;
	particle_minmax.v[0] = maxt * SIZEOFSPACE;
	particle_minmax.x[1] = mint * SIZEOFSPACE;
	particle_minmax.v[1] = maxt * SIZEOFSPACE;
	particle_minmax.x[2] = mint * SIZEOFSPACE;
	particle_minmax.v[2] = maxt * SIZEOFSPACE;

	return particle_minmax;
}
