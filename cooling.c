#include "world.h"
#include "mpi_assert.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

/* Simulation parameters */
unsigned sim_size = 20000;
unsigned numrounds = 10;
unsigned gen_per_round = 200;
unsigned scalefactor = 40;

double rand_double() {
    return ((double)rand())/RAND_MAX;
}

elem_t cooling_init() {
    return 20.;
}

elem_t cooling_step(elem_t elem, elem_t* neighbors, unsigned x, unsigned y, unsigned N) {
    if (x == 0 || x == N - 1) {
        return 90.;
    }
    if (y == 0 || y == N - 1) {
        return 20.;
    }
#ifdef COOLING_POSTS
    if (cooling_post(x, y)) {
        return 10.;
    }
#endif
    return (neighbors[M_UP] + neighbors[M_LEFT] + neighbors[M_DOWN]
            + neighbors[M_RIGHT] + elem) / 5;
}

void cooling_collect(elem_t e, double* d) {
    *d += e;
}

/* return valid only in rank 0 */
double cooling_collect_all(world w, unsigned rank, unsigned numprocs) {
    double d = 0.;
    double global = 0.;
    unsigned sz = world_size(w);
    world_collect(w, (world_collect_fn)cooling_collect, &d);
    d = d / (sz*sz);
    MPI_Reduce(&d, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    global = global / numprocs;
    return global;
}

void get_random_seed() {
    FILE* f = fopen("/dev/urandom", "r");
    do_assert(f != NULL, "Cannot open /dev/urandom for reading\n");
    unsigned int seed;
    do_assert(fread(&seed, sizeof(seed), 1, f) == 1, "Cannot read from /dev/urandom\n");
    fclose(f);
    srand(seed);
}

void print_all(FILE* f, struct world_data d) {
    for (unsigned i = 0; i < d.size; ++i) {
        for (unsigned j = 0; j < d.size; ++j) {
            fprintf(f, ELEM_T_FMT " ", d.data[i*d.size+j]);
        }
        fprintf(f, "\n");
    }
}

int main() {
    MPI_Init(NULL, NULL);
    unsigned numprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
    /* determine domain decomposition parameters */
    unsigned edge_procs = sqrt(numprocs);
    do_assert(edge_procs*edge_procs == numprocs, "Number of spawned processes must be square\n");
    do_assert(sim_size % edge_procs == 0, "World size must be divisible by sqrt(num_processes)\n");
    do_assert(sim_size % (edge_procs*scalefactor) == 0, "Bad Scale Factor\n");
    get_random_seed();
    struct world_data bigbuf = world_data_create_null();
    int i,j;
#ifndef NO_DATA_OUT
    char resdir[64];
    char buf[64];
    if (rank == 0) {
        /* create results directory */
        snprintf(resdir, sizeof(resdir), "results.%06i", (int)time(NULL));
        do_assert(mkdir(resdir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0,
                "Cannot create results directory\n");

    }
    mpi_assert(MPI_Bcast(resdir, sizeof(resdir), MPI_CHAR, 0, MPI_COMM_WORLD));
    do_assert(chdir(resdir) == 0, "Cannot change directory\n");
#endif
    world w = world_create(sim_size, ABSORBING_BOUNDARY, cooling_init, cooling_step);
    for (i = 0; i < numrounds; ++i) {
        double d = cooling_collect_all(w, rank, numprocs);
        if (rank == 0) {
            printf("After %i rounds of %i generations:\n", i, gen_per_round);
            printf("\tAverage temperature is %lf.\n", d);
        }
#ifndef NO_DATA_OUT
        world_gather_scaled(w, 0, scalefactor, &bigbuf);
        if (rank == 0) {
            snprintf(buf, sizeof(buf), "data.%i", i);
            FILE* f = fopen(buf, "w");
            do_assert(f != NULL, "Cannot open results file\n");
            print_all(f, bigbuf);
            fclose(f);
        }
#endif
        for (j = 0; j < gen_per_round; ++j) {
            world_sim(w);
        }
    }
    mpi_assert(MPI_Barrier(MPI_COMM_WORLD));
    double d = cooling_collect_all(w, rank, numprocs);
#ifndef NO_DATA_OUT
    world_gather_scaled(w, 0, scalefactor, &bigbuf);
    if (rank == 0) {
        snprintf(buf, sizeof(buf), "data.%i", i);
        FILE* f = fopen(buf, "w");
        do_assert(f != NULL, "Cannot open results file\n");
        print_all(f, bigbuf);
        fclose(f);
    }
#endif
    if (rank == 0) {
        printf("After %i rounds of %i generations:\n", i, gen_per_round);
        printf("\tAverage temperature is %lf.\n", d);
    }
    world_destroy(w);
    MPI_Finalize();
    return 0;
} 
