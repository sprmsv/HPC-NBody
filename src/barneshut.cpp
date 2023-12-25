#include "barneshut.h"


void nbodybarneshut(particle_t* array, int nbr_particles, int nbr_iterations)
{
	double step = TIMESTEP;
	node* root;
	node* newroot;
	node* oldroot;
	particle_t ranges;

	root = (node*) malloc(sizeof(node));
	newroot = (node*) malloc(sizeof(node));
	ranges = getMinMax(array, nbr_particles);
	init_tree(&ranges, root);
	init_tree(&ranges, newroot);

	construct_bh_tree(array, root, nbr_particles);
	for (int it = 0; it < nbr_iterations; ++it) {
		compute_force_in_node(root, root);
		compute_bh_force(root);  // CHECK: Why?
		move_all_particles(newroot, root, step);
		oldroot = root;
		root = newroot;
		newroot = oldroot;
		clean_tree(newroot);
	}

	clean_tree(newroot);
	clean_tree(newroot);
	free(root);
	free(newroot);
}

// Move all the particles from node n to new_root
void move_all_particles(node* new_root, node* n, double step)
{
	if (n->children != NULL) {
		for (int i = 0; i < 8; i++){
			move_all_particles(new_root, &n->children[i], step);
		}
	}
	else {
		particle_t* p = n->particle;
		move_particle(new_root, n, p, step);
	}
}

// Compute new position/velocity of the particle
void move_particle(node* root, node* n, particle_t* p, double step)
{
	double ax,ay,az;

	if ((p==NULL) || (n==NULL))
		return;

	ax = p->fx/p->m;
	ay = p->fy/p->m;
	az = p->fz/p->m;
	p->vx += ax*step;
	p->vy += ay*step;
	p->vz += az*step;
	p->x += p->vx * step;
	p->y += p->vy * step;
	p->z += p->vz * step;

	if (!is_particle_out_of_scope(p, root)) {
		insert_particle(p, root);
	}
	else {
		n->particle = NULL;
	}
}

// Check if a particle is out of scope (lost body in space)
bool is_particle_out_of_scope(particle_t* p, node* root)
{
	if ((p->x < root->minx) || (p->y < root->miny) || (p->z < root->minz))
		return true;
	if ((p->x > root->maxx) || (p->y > root->maxy) || (p->z > root->maxz))
		return true;
	return false;
}

// Clean tree root
void clean_tree(node* root)
{
	if (root == NULL)
		return;

	if (root->children != NULL) {
		for (int i = 0; i < 8; i++) {
			clean_tree(&root->children[i]);
		}
		free(root->children);
		root->children = NULL;
		root->sub_nbr_particles=0;
	}
}

// compute the forces on the BH tree
// CHECK: What does this do? Seems like computing force of the external nodes on their own particle!
void compute_bh_force(node* n)
{
	if (n->children != NULL) {
		for (int i = 0; i < 8; i++) {
			compute_bh_force(&n->children[i]);
		}
	}
	else {
		compute_force_particle(n, n->particle);
	}
}

// Compute force of node n on particle p
void compute_force_particle(node* n, particle_t* p)
{
	double diffx, diffy, diffz, distance;
	double size;

	// If the node is empty
	if ((n==NULL) || (n->sub_nbr_particles==0))
		return;

	// If the node is external
	if ((n->particle != NULL) && (n->children==NULL)) {
		compute_force(p, n->centerx, n->centery, n->centerz, n->mass);
	}
	// If the node is internal
	else {
		size = n->maxx - n->minx;
		diffx = n->centerx - p->x;
		diffy = n->centery - p->y;
		diffz = n->centerz - p->z;
		distance = sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
		if (size / distance < THETA) {
			// Use an approximation of the force if the node is far away
			compute_force(p, n->centerx, n->centery, n->centerz, n->mass);
		}
		else {
			// Otherwise, run the procedure recursively on each of the current node's children.
			for (int i = 0; i < 8; i++) {
				compute_force_particle(&n->children[i], p);
			}
		}
	}
}

// Compute force (accumulates on p->f)
void compute_force(particle_t* p, double xpos, double ypos, double zpos, double mass)
{
	double xsep, ysep, zsep, dist_sq, gravity;

	xsep = xpos - p->x;
	ysep = ypos - p->y;
	zsep = zpos - p->z;
	dist_sq = std::max((xsep*xsep)+ (ysep*ysep)+(zsep*zsep), 0.01);
	gravity = GRAV_CONSTANT * (p->m) * (mass) / dist_sq / sqrt(dist_sq);

	p->fx += gravity * xsep;
	p->fy += gravity * ysep;
	p->fz += gravity * zsep;
}

// Compute all forces on all the particles of a node
void compute_force_in_node(node* n, node* root)
{
	if (n==NULL) return;

	if ((n->particle != NULL) && (n->children == NULL)) {
		// Compute all forces on the particle of the node
		n->particle->fx = 0;
		n->particle->fy = 0;
		n->particle->fz = 0;
		compute_force_particle(root, n->particle);
	}
	if (n->children != NULL) {
		// Recall on all children nodes
		for(int i = 0; i < 8; i++) {
			compute_force_in_node(&n->children[i], root);
		}
	}
}

// Construction of the barnes-hut tree
void construct_bh_tree(particle_t* array, node* root, int nbr_particles)
{
	for (int i = 0; i < nbr_particles; i++){
		insert_particle(&array[i], root);
	}
}

// Add particle p in node n or one of its children
void insert_particle(particle_t* p, node* n)
{
	int octrant;
	double totalmass = 0.;
	double totalx = 0.;
	double totaly = 0.;
	double totalz = 0.;

	// there is no particle
	if ((n->sub_nbr_particles == 0) && (n->children==NULL)) {
		n->particle = p;
		n->centerx = p->x;
		n->centery = p->y;
		n->centerz = p->z;
		n->mass = p->m;
		n->sub_nbr_particles++;
		p->node = n;
	}

	// There is already a particle
	else {
		// Insert the old particle in the correct children (if necessary)
		if (n->children==NULL) {
			create_children(n);
			particle_t* particle_parent = n->particle;
			octrant = get_octrant(particle_parent, n);
			n->particle = NULL;
			insert_particle(particle_parent, &n->children[octrant]);
		}
		// Now insert the new p
		octrant = get_octrant(p, n);
		insert_particle(p, &n->children[octrant]);

		// Update mass and barycenter (sum of momentums / total mass)
		for (int i = 0; i < 8; i++) {
			totalmass += n->children[i].mass;
			totalx += n->children[i].centerx * n->children[i].mass;
			totaly += n->children[i].centery * n->children[i].mass;
			totalz += n->children[i].centerz * n->children[i].mass;
		}
		n->mass = totalmass;
		n->centerx = totalx / totalmass;
		n->centery = totaly / totalmass;
		n->centerz = totalz / totalmass;
		n->sub_nbr_particles++;
		p->node = n;
	}
}

// create 8 children from 1 node
void create_children(node* n)
{
	n->children = (node*) malloc(8 * sizeof(node));

	double x12 = n->minx+(n->maxx-n->minx)/2.;
	double y12 = n->miny+(n->maxy-n->miny)/2.;
	double z12 = n->minz+(n->maxz-n->minz)/2.;

	init_node(&n->children[SW_DOWN], n, n->minx, x12, n->miny, y12, n->minz, z12);
	init_node(&n->children[NW_DOWN], n, n->minx, x12, n->miny, y12, z12, n->maxz);
	init_node(&n->children[SE_DOWN], n, n->minx, x12, y12, n->maxy, n->minz, z12);
	init_node(&n->children[NE_DOWN], n, n->minx, x12, y12, n->maxy, z12, n->maxz);
	init_node(&n->children[SW_UP], n, x12, n->maxx, n->miny, y12, n->minz, z12);
	init_node(&n->children[NW_UP], n, x12, n->maxx, n->miny, y12, z12, n->maxz);
	init_node(&n->children[SE_UP], n, x12, n->maxx, y12, n->maxy, n->minz, z12);
	init_node(&n->children[NE_UP], n, x12, n->maxx, y12, n->maxy, z12, n->maxz);
}

// Init a node n attached to parent.
void init_node(node* n, node* parent, double minx, double maxx, double miny, double maxy, double minz, double maxz)
{
	n->parent=parent;
	n->children = NULL;
	n->minx = minx;
	n->maxx = maxx;
	n->miny = miny;
	n->maxy = maxy;
	n->minz = minz;
	n->maxz = maxz;
	n->depth = parent->depth + 1;
	n->particle = NULL;
	n->sub_nbr_particles = 0.;
	n->centerx = 0.;
	n->centery = 0.;
	n->centerz = 0.;
	n->mass = 0.;
}

// get the "octrant" where the particle resides (octrant is a generalization in 3D of a 2D quadrant)
int get_octrant(particle_t* p, node* n)
{
	int octrant = -1;

	double xmin = n->minx;
	double xmax = n->maxx;
	double x_center = xmin+(xmax-xmin)/2;

	double ymin = n->miny;
	double ymax = n->maxy;
	double y_center = ymin+(ymax-ymin)/2;

	double zmin = n->minz;
	double zmax = n->maxz;
	double z_center = zmin+(zmax-zmin)/2;

	if (n==NULL)
		std::cerr << "ERROR: Node is NULL" << std::endl;
	if (p==NULL)
		std::cerr << "ERROR: Particle is NULL" << std::endl;

	if (p->x <= x_center) {
		if (p->y <= y_center) {
			if (p->z <= z_center) {
				octrant = SW_DOWN;
			}
			else {
				octrant = NW_DOWN;
			}
		}
		else {
			if (p->z <= z_center) {
				octrant = SE_DOWN;
			}
			else {
				octrant = NE_DOWN;
			}
		}
	}
	else {
		if (p->y <= y_center) {
			if (p->z <= z_center) {
				octrant = SW_UP;
			}
			else {
				octrant = NW_UP;
			}
		}
		else {
			if (p->z <= z_center) {
				octrant = SE_UP;
			}
			else {
				octrant = NE_UP;
			}
		}
	}
	return octrant;
}

// Init the tree
// Remark: We use a particle struct to transfer min and max values from main
void init_tree(particle_t* ranges, node* root)
{
	root->minx = ranges->x;
	root->maxx = ranges->vx;
	root->miny = ranges->y;
	root->maxy = ranges->vy;
	root->minz = ranges->z;
	root->maxz = ranges->vz;
	root->particle = NULL;
	root->sub_nbr_particles = 0;
	root->parent = NULL;
	root->children = NULL;
	root->centerx = 0.;
	root->centery = 0.;
	root->centerz = 0.;
	root->mass = 0.;
	root->depth = 0;
}
