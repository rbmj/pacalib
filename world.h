#ifndef WORLD_H_INC
#define WORLD_H_INC

#include <stdbool.h>
#include <stdio.h>

#include "world_config.h"

struct world_t;
typedef struct world_t* world;

/* boundary_t:
 * WRAPAROUND_BOUNDARY corresponds to a 3-D Torus world
 * ABSORBING_BOUNDARY corresponds to a finite 2-D world
 */
enum boundary_t {
    WRAPAROUND_BOUNDARY,
    ABSORBING_BOUNDARY
};

struct world_data {
    elem_t* data;
    unsigned size;
    bool includes_halo;
};

/* Enum representing the von neumann neighborhood */
enum vn_neighborhood {
    VN_LEFT = 0,
    VN_UP,
    VN_DOWN,
    VN_RIGHT,
    VN_NEIGHBORHOOD_SIZE
};

/* Enum representing the moore neighborhood */
enum moore_neighborhood {
    M_UPLEFT = 0,
    M_UP,
    M_UPRIGHT,
    M_LEFT,
    M_RIGHT,
    M_DOWNLEFT,
    M_DOWN,
    M_DOWNRIGHT,
    M_NEIGHBORHOOD_SIZE
};

struct world_data world_data_create_null();
struct world_data world_data_create(unsigned);
void world_data_destroy(struct world_data);
void world_data_realloc(struct world_data*, unsigned);

void world_data_sendto(world w, unsigned rank);
struct world_data world_data_recvfrom(unsigned rank);

/*
 * Callback to perform transformations on each cell in the
 * current world.
 *
 * world_xform_fn callbacks will be called on each cell with:
 *      elem_t val:  Current value of this cell
 *      elem_t* neighbors:  Array of the moore neighborhood
 *      unsigned x:  x location of this node (absolute)
 *      unsigned y:  y location of this node (absolute)
 *      unsigned sz: size of an edge (absolute)
 *
 * The neighbors array can be indexed using the moore_neighborhood
 * enum.
 *
 * The function should return the new value of the cell.
 */
typedef elem_t(*world_xform_fn)(elem_t val, elem_t* neighbors, unsigned x, unsigned y, unsigned sz);

/*
 * Callback to initialize a cell.
 *
 * The function should return the value of the cell
 */
typedef elem_t(*world_init_fn)();

/*
 * Callback to collect data on the world.
 *
 * world_collect_fn callbacks will be called on each cell with:
 *      elem_t val:  Current value of this cell
 *      void* data:  Whatever is passed to world_collect()
 */
typedef void (*world_collect_fn)(elem_t val, void* data);

/*
 * Create a world:
 *      unsigned worldsize:  Number of cells along each edge of the entire world
 *      enum boundary_t bound:  Type of boundary for this world
 *      world_init_fn init:  Callback to initialize the world
 *      world_xform_fn step:  Callback to transform the world one generation
 *
 * Returns a new world, which should be destroyed with world_destroy().
 *
 * Currently this implementation is limited to square worlds which evenly divide
 * into a square domain decomposition.
 */
world world_create(unsigned worldsize, enum boundary_t bound, world_init_fn init, world_xform_fn step);

/*
 * Destroy a world:
 *      world w: The world to destroy
 */
void world_destroy(world w);

/*
 * Simulate one generation of the world.
 * 
 * This calls the world_xform_fn callback passed in to world_create() on
 * each cell and then increments the age of the simulation.
 */
void world_sim(world w);

/*
 * Do some arbitrary transformation on the world.
 *
 * Calls cb for every cell in w.
 */
void world_xform(world w, world_xform_fn cb);

/* 
 * Collect some data about the world.
 *      world w:  The world to manipulate
 *      world_collect_fn cb:  Called on every cell in w
 *      void* data:  Auxiliary data passed to each call of cb
 */
void world_collect(world, world_collect_fn, void*);

/*
 * Print the world to a file.
 * Note that this prints each rank's chunk of the world separately.
 */
void world_print(world, FILE*);

/*
 * Print the world represented by world_data to a file.
 * The same caveat as world_print() applies:  print's each rank's
 * chunk separately.
 */

void world_data_print(struct world_data, FILE*);
/*
 * Print the world to a file.
 * Note that this prints each rank's chunk of the world separately.
 *
 * This version includes the rank's halo when printing.
 */
void world_print_raw(world, FILE*);

/*
 * Print the world represented by world_data to a file.
 * The same caveat as world_print() applies:  print's each rank's
 * chunk separately.
 *
 * This version includes the rank's halo when printing.
 */
void world_data_print_raw(struct world_data, FILE*);

/*
 * Get the age of this simulation.
 *
 * Will return the number of times world_sim() has been called.
 */
unsigned world_get_age(world);

/*
 * Get a non-owning reference to the data for this world.
 *
 * This may be invalidated, but does not need to be destroyed.
 */
struct world_data world_get_data(world);

/*
 * Get a pointer to element (x, y), numbered from the
 * top left.
 */
elem_t* world_get(world w, unsigned x, unsigned y);

/* 
 * Get the size of this chunk of the world
 */
unsigned world_size(world w);

/*
 * Apply a callback to the (x, y)th element in the
 * WHOLE WORLD, measured absolutely.
 */
void world_abs_apply(world w, unsigned x, unsigned y, world_xform_fn fn);

/*
 * Get an owning reference to the data in this rank.
 * The data will be reduced by the supplied factor along
 * each dimension.  For example, if the second argument is
 * 2, then the resulting data will be a quarter the size of
 * the actual data stored (half along both x and y axes).
 *
 * The number of points along each dimension for this rank
 * must be divisible by the provided factor.
 */
void world_slice_reduce(world w, unsigned factor, struct world_data* d);

void world_gather_scaled(world w, unsigned rank, unsigned factor, struct world_data* d);
#endif
