#ifndef NBODYBARNESHUT_H_
#define NBODYBARNESHUT_H_

#include "parameters.h"
#include "reader.h"
#include "math.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <algorithm>


void nbodybarneshut(particle_t* array, int nbr_particles, int nbr_iterations, int psize, int prank);

void init_tree(particle_t* ranges, node* root);
void clean_tree(node* root);
void construct_bh_tree(particle_t* array, node* root, int nbr_particles);
void insert_particle(particle_t* p, node* n);
void create_children(node* parent);
void init_node(node* n, node* parent, double minx, double maxx, double miny, double maxy, double minz, double maxz);
int get_octrant(particle_t* p, node* n);

void move_all_particles(node* root, node* n, double step, int psize, int prank);
void move_particle(node* root, node* n, particle_t* p, double step, int psize, int prank);

void reassign_all_particles(node* newroot, node* n);
void reassign_particle(node* newroot, node* n, particle_t* p);
bool is_particle_out_of_scope(particle_t* p, node* root);

void compute_force_in_node(node* root, node* n, int psize, int prank);
void compute_force_particle(node* n, particle_t* p);
void compute_force(particle_t* p, double xpos, double ypos, double zpos, double mass);

void compute_bh_force(node* n);

void communicate(particle_t* array, int nbr_particles, int psize, int prank);

void print_tree(node * n);
void print_node(node * n);
void print_particle(particle_t * p);

#endif /*NBODYBARNESHUT_H_*/
