#include "bruteforce.h"


void nbodybruteforce(particle_t* array, int nbr_particles, int nbr_iterations)
{
	double step = 1.;
	for (int n = 0; n < nbr_iterations; n++){
		for (int i = 0; i < nbr_particles; i++){
			compute_brute_force(&array[i], array, nbr_particles, step);
		}
		update_positions(array, nbr_particles, step);
	}
}

void compute_brute_force(particle_t* p, particle_t* array, int nbr_particles, double step)
{
	double x_sep, y_sep, z_sep, dist_sq, grav_base;
	double F_x=0.;
	double F_y=0.;
	double F_z=0.;
	double a_x=0.;
	double a_y=0.;
	double a_z=0.;

	for (int i = 0; i < nbr_particles; i++) {
		if (array[i].id == p->id) {
			continue;
		}
		x_sep = array[i].x - p->x;
		y_sep = array[i].y - p->y;
		z_sep = array[i].z - p->z;
		dist_sq = std::max((x_sep * x_sep) + (y_sep * y_sep) + (z_sep * z_sep), 0.01);
		grav_base = GRAV_CONSTANT * (p->m) * (array[i].m) / dist_sq / sqrt(dist_sq);
		F_x += grav_base * x_sep;
		F_y += grav_base * y_sep;
		F_z += grav_base * z_sep;
	}

	// a = F / m
	a_x = F_x / p->m;
	a_y = F_y / p->m;
	a_z = F_z / p->m;

	// v = a * step
	p->vx += a_x * step;
	p->vy += a_y * step;
	p->vz += a_z * step;
}

void update_positions(particle_t* array, int nbr_particles, double step)
{
  for (int i = 0; i  < nbr_particles; i++) {
    array[i].x += array[i].vx * step;
    array[i].y += array[i].vy * step;
    array[i].z += array[i].vz * step;
  }
}
