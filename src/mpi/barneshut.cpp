#include "barneshut.h"


void nbodybarneshut(particle_t* array, int nbr_particles, int nbr_iterations, int psize, int prank)
{
	// Declare variables
	double step = TIMESTEP;
	node* root;
	node* newroot;
	node* oldroot;
	particle_t ranges;

	// Initialize the nodes
	root = (node*) malloc(sizeof(node));
	newroot = (node*) malloc(sizeof(node));
	ranges = getMinMax(array, nbr_particles);
	init_tree(&ranges, root);
	init_tree(&ranges, newroot);

	// Construct a tree from the particles
	construct_bh_tree(array, root, nbr_particles);

	// Take time steps and move the particles
	for (int it = 0; it < nbr_iterations; ++it) {
		compute_force_in_node(root, root, psize, prank);
		move_all_particles(newroot, root, step, psize, prank);
		communicate(array, nbr_particles, psize, prank);
		reassign_all_particles(newroot, root);
		// Swap the root pointers
		oldroot = root;
		root = newroot;
		clean_tree(oldroot);
		newroot = oldroot;
	}

#ifdef DEBUG
	if (prank == 0) print_tree(root);
#endif

	// Deallocate memory
	clean_tree(root);
	clean_tree(newroot);
	free(root);
	free(newroot);
}

// Init the tree
// Remark: We use a particle struct to transfer min and max values from main
void init_tree(particle_t* ranges, node* root)
{
	root->minx = ranges->x[0];
	root->maxx = ranges->v[0];
	root->miny = ranges->x[1];
	root->maxy = ranges->v[1];
	root->minz = ranges->x[2];
	root->maxz = ranges->v[2];
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
		n->centerx = p->x[0];
		n->centery = p->x[1];
		n->centerz = p->x[2];
		n->mass = p->m;
		n->sub_nbr_particles++;
		p->parent = n;
	}

	// There is already a particle
	else {
		// Insert the old particle in the correct child (if necessary)
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
		p->parent = n;
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

	if (p->x[0] <= x_center) {
		if (p->x[1] <= y_center) {
			if (p->x[2] <= z_center) {
				octrant = SW_DOWN;
			}
			else {
				octrant = NW_DOWN;
			}
		}
		else {
			if (p->x[2] <= z_center) {
				octrant = SE_DOWN;
			}
			else {
				octrant = NE_DOWN;
			}
		}
	}
	else {
		if (p->x[1] <= y_center) {
			if (p->x[2] <= z_center) {
				octrant = SW_UP;
			}
			else {
				octrant = NW_UP;
			}
		}
		else {
			if (p->x[2] <= z_center) {
				octrant = SE_UP;
			}
			else {
				octrant = NE_UP;
			}
		}
	}
	return octrant;
}

// Update position/velocity of all the particles of node n
void move_all_particles(node* newroot, node* n, double step, int psize, int prank)
{
	if (n->children != NULL) {
		for (int i = 0; i < 8; i++){
			move_all_particles(newroot, &n->children[i], step, psize, prank);
		}
	}
	else {
		move_particle(newroot, n, n->particle, step, psize, prank);
	}
}

// Compute position/velocity of the particle
void move_particle(node* newroot, node* n, particle_t* p, double step, int psize, int prank)
{
	// Return if not viable
	if ((p==NULL) || (n==NULL))
		return;

	// Update velocity and position (if the particle is assigned)
	if (p->prank == prank) {
		double ax, ay, az;
		ax = p->f[0] / p->m;
		ay = p->f[1] / p->m;
		az = p->f[2] / p->m;
		p->v[0] += ax * step;
		p->v[1] += ay * step;
		p->v[2] += az * step;
		p->x[0] += p->v[0] * step;
		p->x[1] += p->v[1] * step;
		p->x[2] += p->v[2] * step;
	}
}

// Reassign all particles of the node to a new root
void reassign_all_particles(node* newroot, node* n)
{
	if (n->children != NULL) {
		for (int i = 0; i < 8; i++){
			reassign_all_particles(newroot, &n->children[i]);
		}
	}
	else {
		reassign_particle(newroot, n, n->particle);
	}
}

// Reassign a particle to a new root
void reassign_particle(node* newroot, node* n, particle_t* p)
{
	if ((p==NULL) || (n==NULL))
		return;

	// Insert in the new root if still in scope
	if (!is_particle_out_of_scope(p, newroot)) {
		insert_particle(p, newroot);
	}
	else {
		n->particle = NULL;
	}
}

// Check if a particle is out of scope (lost body in space)
bool is_particle_out_of_scope(particle_t* p, node* root)
{
	if ((p->x[0] < root->minx) || (p->x[1] < root->miny) || (p->x[2] < root->minz))
		return true;
	if ((p->x[0] > root->maxx) || (p->x[1] > root->maxy) || (p->x[2] > root->maxz))
		return true;
	return false;
}

// Compute all forces from a root on all the particles of a node
void compute_force_in_node(node* root, node* n, int psize, int prank)
{
	if (n==NULL) return;

	// If external node, compute all forces on the particle of the node
	if ((n->particle != NULL) && (n->children == NULL)) {
		// If the particle is assigned to this process
		if (n->particle->prank == prank) {
			// Calculate the forces
			n->particle->f[0] = 0;
			n->particle->f[1] = 0;
			n->particle->f[2] = 0;
			compute_force_particle(root, n->particle);
		}
	}
	// If internal node, call the function on all children
	if (n->children != NULL) {
		for (int i = 0; i < 8; i++) {
			compute_force_in_node(root, &n->children[i], psize, prank);
		}
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
	// If the node is internal, it depends on its relatve distance
	else {
		size = n->maxx - n->minx;
		diffx = n->centerx - p->x[0];
		diffy = n->centery - p->x[1];
		diffz = n->centerz - p->x[2];
		distance = sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
		// Use an approximation of the force if the node is far away
		if (size / distance < THETA) {
			compute_force(p, n->centerx, n->centery, n->centerz, n->mass);
		}
		// Otherwise, run the procedure recursively on each of the current node's children.
		else {
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

	xsep = xpos - p->x[0];
	ysep = ypos - p->x[1];
	zsep = zpos - p->x[2];
	dist_sq = std::max((xsep*xsep)+ (ysep*ysep)+(zsep*zsep), 0.01);
	gravity = GRAV_CONSTANT * (p->m) * (mass) / dist_sq / sqrt(dist_sq);

	p->f[0] += gravity * xsep;
	p->f[1] += gravity * ysep;
	p->f[2] += gravity * zsep;
}

// Communicate the computed positions and velocities to the other processes
void communicate(particle_t* array, int nbr_particles, int psize, int prank)
{
	// Return if not multiprocessing
	if (psize < 2) return;

	// Declare varialbes
	int nbr_particles_send = nbr_particles / psize + ((nbr_particles % psize) > prank);
	int nbr_particles_recv = nbr_particles - nbr_particles_send;
	int ids_send[nbr_particles_send];
	int ids_recv[nbr_particles_recv];
	double pos_send[3 * nbr_particles_send];
	double pos_recv[3 * nbr_particles_recv];
	double vel_send[3 * nbr_particles_send];
	double vel_recv[3 * nbr_particles_recv];

	// Construct the send buffers
	int i_send = 0;
	for (int i_array = 0; i_array < nbr_particles; i_array++) {
		particle_t& p = array[i_array];
		if (p.prank == prank) {
			ids_send[i_send] = i_array;
			pos_send[3 * i_send + 0] = p.x[0];
			pos_send[3 * i_send + 1] = p.x[1];
			pos_send[3 * i_send + 2] = p.x[2];
			vel_send[3 * i_send + 0] = p.v[0];
			vel_send[3 * i_send + 1] = p.v[1];
			vel_send[3 * i_send + 2] = p.v[2];
			i_send++;
		}
	}

	// Broadcast ids and forces (the order is important)
	MPI_Request reqs_ids[psize];
	MPI_Request reqs_pos[psize];
	MPI_Request reqs_vel[psize];
	int recv_displ = 0;
	for (int rank = 0; rank < psize; rank++) {
		if (rank == prank) {
			MPI_Ibcast(&ids_send[0], nbr_particles_send, MPI_INT, rank, MPI_COMM_WORLD, &reqs_ids[rank]);
			MPI_Ibcast(&pos_send[0], (3 * nbr_particles_send), MPI_DOUBLE, rank, MPI_COMM_WORLD, &reqs_pos[rank]);
			MPI_Ibcast(&vel_send[0], (3 * nbr_particles_send), MPI_DOUBLE, rank, MPI_COMM_WORLD, &reqs_vel[rank]);
		}
		else {
			int nbr_particles_rank = nbr_particles / psize + ((nbr_particles % psize) > rank);
			MPI_Ibcast(&ids_recv[recv_displ], nbr_particles_rank, MPI_INT, rank, MPI_COMM_WORLD, &reqs_ids[rank]);
			MPI_Ibcast(&pos_recv[3 * recv_displ], (3 * nbr_particles_rank), MPI_DOUBLE, rank, MPI_COMM_WORLD, &reqs_pos[rank]);
			MPI_Ibcast(&vel_recv[3 * recv_displ], (3 * nbr_particles_rank), MPI_DOUBLE, rank, MPI_COMM_WORLD, &reqs_vel[rank]);
			recv_displ += nbr_particles_rank;
		}
	}
	MPI_Waitall(psize, reqs_ids, MPI_STATUSES_IGNORE);
	MPI_Waitall(psize, reqs_pos, MPI_STATUSES_IGNORE);
	MPI_Waitall(psize, reqs_vel, MPI_STATUSES_IGNORE);

	// Unpack the received buffers
	for (int i_recv = 0; i_recv < nbr_particles_recv; i_recv++) {
		int i_array = ids_recv[i_recv];
		particle_t& p = array[i_array];
		p.x[0] = pos_recv[3 * i_recv + 0];
		p.x[1] = pos_recv[3 * i_recv + 1];
		p.x[2] = pos_recv[3 * i_recv + 2];
		p.v[0] = vel_recv[3 * i_recv + 0];
		p.v[1] = vel_recv[3 * i_recv + 1];
		p.v[2] = vel_recv[3 * i_recv + 2];
	}
}

// print a tree
void print_tree(node* root){
	node* child;
	if (root->children != NULL){
		for (int i = 0; i < 8; i++){
			child = &root->children[i];
			print_tree(child);
		}
	}
	print_node(root);
}

// print a node 
void print_node(node* n){
	int d = n->depth;
	for (int i=0; i < d; i++) {
		printf("\t");
	}
	printf("[level %d]", d);
	printf(" ([%f:%f:%f])", n->centerx, n->centery, n->centerz);
	printf(" Node ");
	printf(" M = %f", n->mass);
	printf(" has %d particles ", n->sub_nbr_particles);
	if (n->particle!=NULL) {
		particle_t* p = n->particle;
		printf(". Particle ID = %d", p->id);
		printf(" prank = %d", p->prank);
	}
	printf("\n");
}

// print a particle 
void print_particle(particle_t* p){
	printf("[Particle %d]", p->id);
	printf(" position ([%f:%f:%f])", p->x[0], p->x[1], p->x[2]);
	printf(" M = %f", p->m);
	printf("\n");
}
