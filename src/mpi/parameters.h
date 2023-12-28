#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <stdbool.h>

#define GRAV_CONSTANT 0.5
#define THETA 1.0
#define TIMESTEP 1.
#define SIZEOFSPACE 10.0  // Expansion ratio

#define SW_DOWN 0
#define SE_DOWN 1
#define NW_DOWN 2
#define NE_DOWN 3
#define SW_UP 4
#define SE_UP 5
#define NW_UP 6
#define NE_UP 7


typedef struct particles particle_t;
typedef struct nodes node;

struct particles
{
	// Position
	double x;
	double y;
	double z;

	// Speed
	double vx;
	double vy;
	double vz;

	// Force
	double fx;
	double fy;
	double fz;

	// Mass
	double m;

	// Identifier
	int id;

	// Potential
	double V;

	// Parent node
	node* parent;
};

struct nodes
{
	// Particles and their number
	particle_t* particle;
	int sub_nbr_particles;

	// Node depth
	int depth;

	// Node coordinates
	double minx;
	double maxx;
	double miny;
	double maxy;
	double minz;
	double maxz;

	// Mass and Center of Mass
	double mass;
	double centerx;
	double centery;
	double centerz;

	// Parent and children
	node* parent;
	node* children;
};

#endif /*PARAMETERS_H_*/
