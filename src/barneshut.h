#ifndef NBODYBARNESHUT_H_
#define NBODYBARNESHUT_H_

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

#include "parameters.h"
#include "reader.h"
#include "math.h"


void nbodybarneshut(particle_t* array, int nbr_particles, int nbr_iterations);
void construct_bh_tree(particle_t* array, node* root, int nbr_particles);
void compute_bh_force(node* n) ;
void compute_force(particle_t* p, double xpos, double ypos, double zpos, double mass);
void compute_force_particle(node* n, particle_t* p);
void compute_force_in_node(node* n,node* root);
void move_all_particles(node* root, node* n, double step);
void move_particle(node* root, node* n, particle_t* p, double step);
bool is_particle_out_of_scope(particle_t* p, node* root);
void clean_tree(node* root);

void insert_particle(particle_t* p, node* n);
void init_tree(particle_t* ranges, node* root);
int get_octrant(particle_t* p, node* n);
void create_children(node* parent);
void init_node(node* n, node* parent, double minx, double maxx, double miny, double maxy, double minz, double maxz);

#endif /*NBODYBARNESHUT_H_*/
