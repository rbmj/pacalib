#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "world.h"
#include "mpi_assert.h"

struct world_t {
    elem_t* sim[2];           /* simulation memory spaces, (size+2)x(size+2) each */
    elem_t* recvbuf;          /* size+2 extra bytes for receiving */
    int valid_sim_space;      /* which sim space contains valid data (0/1) */
    unsigned size;            /* number of cells on each dimension of sim space, not including halo */
    unsigned rank;            /* rank of this process */
    unsigned edge_procs;      /* number of process chunks on each dimension of the world space */
    unsigned age;             /* number of times world_sim has been called */
    world_xform_fn simstep;   /* xform fn to do simulation */
    enum boundary_t bound;    /* boundary type */
    unsigned abs_size;        /* world size (absolute) */
};

void world_destroy(world w) {
    if (w) {
        free(w);
    }
}

void world_halo(world);

world world_create(unsigned worldsize, enum boundary_t bound, world_init_fn init, world_xform_fn step) {
    unsigned numprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, (int*)&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, (int*)&rank);
    /* determine domain decomposition parameters */
    unsigned edge_procs = (unsigned)lround(sqrt(numprocs));
    do_assert(edge_procs*edge_procs == numprocs, "Number of spawned processes must be square\n");
    do_assert(worldsize % edge_procs == 0, "World size must be divisible by sqrt(num_processes)\n");
    unsigned size = worldsize / edge_procs;
    /* do everything in one memory allocation */
    unsigned allocsz = size + 2; /* allow size for halos */
    unsigned bytes = 0;
    bytes += sizeof(struct world_t); /* space for world_t */
    bytes += allocsz*allocsz*2*sizeof(elem_t); /* space for both sim spaces */
    bytes += allocsz*sizeof(elem_t); /* space for recvbuf */
    struct world_t* w = malloc(bytes);
    do_assert(w, "Memory allocation failure\n");
    /* initialize everything */
    w->recvbuf = (elem_t*)(w+1);
    w->sim[0] = w->recvbuf + allocsz;
    w->sim[1] = w->sim[0] + allocsz*allocsz;
    w->size = size;
    w->rank = rank;
    w->edge_procs = edge_procs;
    w->valid_sim_space = 0;
    w->bound = bound;
    w->age = 0;
    w->simstep = step;
    w->abs_size = worldsize;
    /* do initialization */
    elem_t* p = w->sim[w->valid_sim_space];
    unsigned i,j;
    for (j = 0; j < size + 2; ++j) {
        for (i = 0; i < size + 2; ++i) {
            /* j then i so that we maintain semantic consistency with the layout of memory */
            if (i == 0 || i == bound - 1 || j == 0 || j == bound - 1) {
                *p = 0;
            }
            else {
                *p = init();
            }
            ++p;
        }
    }
    world_halo(w);
    w->valid_sim_space = 1 - w->valid_sim_space;
    return w;
}

struct coord {
    unsigned x;
    unsigned y;
};

struct coord to_coord(unsigned rank, unsigned edge_procs) {
    struct coord c = {rank % edge_procs, rank / edge_procs};
    return c;
}

unsigned from_coord(struct coord c, unsigned edge_procs) {
    return c.y * edge_procs + c.x;
}

struct coord get_neighbor_coord(struct coord c, unsigned edge_procs, enum vn_neighborhood pos) {
    switch (pos) {
    case VN_LEFT:
        c.x += edge_procs - 1;
        c.x %= edge_procs;
        break;
    case VN_UP:
        c.y += edge_procs - 1;
        c.y %= edge_procs;
        break;
    case VN_DOWN:
        c.y += 1;
        c.y %= edge_procs;
        break;
    case VN_RIGHT:
        c.x += 1;
        c.x %= edge_procs;
        break;
    default:
        assert(false); /* shouldn't happen!! */
    }
    return c;
}

unsigned get_neighbor(unsigned rank, unsigned edge_procs, enum vn_neighborhood pos) {
    return from_coord(get_neighbor_coord(to_coord(rank, edge_procs), edge_procs, pos), edge_procs);
}

void world_halo(world w) {
    elem_t* buf = w->recvbuf;
    elem_t* space = w->sim[w->valid_sim_space];
    elem_t* p;
    unsigned i;
    unsigned rank = w->rank;
    unsigned edge_procs = w->edge_procs;
    unsigned size = w->size;
    struct coord c = to_coord(rank, edge_procs);
    unsigned neighbors[4];
    MPI_Status s;
    for (i = 0; i < 4; ++i) {
        neighbors[i] = from_coord(get_neighbor_coord(c, edge_procs, i), edge_procs);
    }
    if (c.x & 1) {
        /* odd column */
        if (w->bound == WRAPAROUND_BOUNDARY || c.x != edge_procs - 1) {
            /* send right */
            for (p = space + size, i = 0; i < size + 2; p += size + 2, ++i) {
                buf[i] = *p;
            }
            MPI_Send(buf, size + 2, MPI_ELEM_T, neighbors[VN_RIGHT], 0, MPI_COMM_WORLD);
            /* recv right */
            MPI_Recv(buf, size + 2, MPI_ELEM_T, neighbors[VN_RIGHT], 0, MPI_COMM_WORLD, &s);
            for (p = space + size + 1, i = 0; i < size + 2; p += size + 2, ++i) {
                *p = buf[i];
            }
        }
        /* odd column can never be left side */
        /* send left */
        for (p = space + 1, i = 0; i < size + 2; p += size + 2, ++i) {
            buf[i] = *p;
        }
        MPI_Send(buf, size + 2, MPI_ELEM_T, neighbors[VN_LEFT], 0, MPI_COMM_WORLD);
        /* recv left */
        MPI_Recv(buf, size + 2, MPI_ELEM_T, neighbors[VN_LEFT], 0, MPI_COMM_WORLD, &s);
        for (p = space, i = 0; i < size + 2; p += size + 2, ++i) {
            *p = buf[i];
        }
    }
    else {
        /* even column */
        if (w->bound == WRAPAROUND_BOUNDARY || c.x != 0) {
            /* recv left */
            MPI_Recv(buf, size + 2, MPI_ELEM_T, neighbors[VN_LEFT], 0, MPI_COMM_WORLD, &s);
            for (p = space, i = 0; i < size + 2; p+= size + 2, ++i) {
                *p = buf[i];
            }
            /* send left */
            for (p = space + 1, i = 0; i < size + 2; p += size + 2, ++i) {
                buf[i] = *p;
            }
            MPI_Send(buf, size + 2, MPI_ELEM_T, neighbors[VN_LEFT], 0, MPI_COMM_WORLD);
        }
        if (w->bound == WRAPAROUND_BOUNDARY || c.x != edge_procs - 1) {
            /* recv right */
            MPI_Recv(buf, size + 2, MPI_ELEM_T, neighbors[VN_RIGHT], 0, MPI_COMM_WORLD, &s);
            for (p = space + size + 1, i = 0; i < size + 2; p += size + 2, ++i) {
                *p = buf[i];
            }
            /* send right */
            for (p = space + size, i = 0; i < size + 2; p += size + 2, ++i) {
                buf[i] = *p;
            }
            MPI_Send(buf, size + 2, MPI_ELEM_T, neighbors[VN_RIGHT], 0, MPI_COMM_WORLD);
        }
    }
    if (c.y & 1) {
        /* odd row */
        /* send top */
        MPI_Send(space + size + 2, size + 2, MPI_ELEM_T, neighbors[VN_UP], 0, MPI_COMM_WORLD);
        /* recv top */
        MPI_Recv(space, size + 2, MPI_ELEM_T, neighbors[VN_UP], 0, MPI_COMM_WORLD, &s);
        if (w->bound == WRAPAROUND_BOUNDARY || c.y != edge_procs - 1) {
            /* send bottom */
            MPI_Send(space + (size + 2) * size, size + 2, MPI_ELEM_T, neighbors[VN_DOWN], 0, MPI_COMM_WORLD);
            /* recv bottom */
            MPI_Recv(space + (size + 2) * (size + 1), size + 2, MPI_ELEM_T, neighbors[VN_DOWN], 0, MPI_COMM_WORLD, &s);
        }
    }
    else {
        /* even row */
        if (w->bound == WRAPAROUND_BOUNDARY || c.y != edge_procs - 1) {
            /* recv bottom */
            MPI_Recv(space + (size + 2) * (size + 1), size + 2, MPI_ELEM_T, neighbors[VN_DOWN], 0, MPI_COMM_WORLD, &s);
            /* send bottom */
            MPI_Send(space + (size + 2) * size, size + 2, MPI_ELEM_T, neighbors[VN_DOWN], 0, MPI_COMM_WORLD);
        }
        if (w->bound == WRAPAROUND_BOUNDARY || c.y != 0) {
            /* recv top */
            MPI_Recv(space, size + 2, MPI_ELEM_T, neighbors[VN_UP], 0, MPI_COMM_WORLD, &s);
            /* send top */
            MPI_Send(space + size + 2, size + 2, MPI_ELEM_T, neighbors[VN_UP], 0, MPI_COMM_WORLD);
        }
    }
}

void world_collect(world w, world_collect_fn cb, void* arg) {
    elem_t* p = w->sim[1-w->valid_sim_space];
    unsigned size = w->size;
    p += (size + 2) + 1;
    unsigned i, j;
    for (j = 0; j < size; ++j) {
        for (i = 0; i < size; ++i) {
            cb(*p++, arg);
        }
        p += 2;
    }
}

void world_xform(world w, world_xform_fn cb) {
    unsigned vss = w->valid_sim_space;
    unsigned size = w->size;
    unsigned abssz = w->abs_size;
    unsigned i, j;
    elem_t* prev = w->sim[1-vss] + (size + 2) + 1;
    elem_t* cur = w->sim[vss] + (size + 2) + 1;
    elem_t neighborhood[M_NEIGHBORHOOD_SIZE];
    struct coord base = to_coord(w->rank, w->edge_procs);
    base.x *= w->size;
    base.y *= w->size;
    for (j = 0; j < size; ++j) {
        for (i = 0; i < size; ++i) {
            neighborhood[M_UPLEFT] = *(prev - (size + 2) - 1);
            neighborhood[M_UP] = *(prev - (size + 2));
            neighborhood[M_UPRIGHT] = *(prev - (size + 2) + 1);
            neighborhood[M_LEFT] = *(prev - 1);
            neighborhood[M_RIGHT] = *(prev + 1);
            neighborhood[M_DOWNLEFT] = *(prev + (size + 2) - 1);
            neighborhood[M_DOWN] = *(prev + (size + 2));
            neighborhood[M_DOWNRIGHT] = *(prev + (size + 2) + 1);
            *cur++ = cb(*prev++, neighborhood, base.x + i, base.y + j, abssz);
        }
        /* currently on right halo - need to increment over to next row,
         * then to first valid cell (so += 2) */
        cur += 2;
        prev += 2;
    }
    world_halo(w);
    w->valid_sim_space = 1-vss;
}

void world_sim(world w) {
    world_xform(w, w->simstep);
    ++(w->age);
}

unsigned world_get_age(world w) {
    return w->age;
}

struct world_data world_data_create_null() {
    struct world_data ret = {0};
    return ret;
}

struct world_data world_data_create(unsigned sz) {
    struct world_data ret;
    ret.data = malloc(sz*sz*sizeof(elem_t));
    ret.size = sz;
    ret.includes_halo = false;
    return ret;
}

void world_data_destroy(struct world_data w) {
    if (w.data) {
        free(w.data);
    }
}

void world_data_realloc(struct world_data* d, unsigned newsz) {
    d->data = realloc(d->data, newsz*newsz*sizeof(elem_t));
    d->size = newsz;
}

void world_data_sendto(world w, unsigned rank) {
    unsigned sz = w->size;
    MPI_Send(&sz, 1, MPI_UNSIGNED, rank, 0, MPI_COMM_WORLD);
    sz += 2;
    MPI_Send(w->sim[1-w->valid_sim_space], sz*sz, MPI_ELEM_T, rank, 0, MPI_COMM_WORLD);
}

struct world_data world_data_recvfrom(unsigned rank) {
    MPI_Status s;
    struct world_data ret;
    unsigned num_elem;
    MPI_Recv(&(ret.size), 1, MPI_UNSIGNED, rank, 0, MPI_COMM_WORLD, &s);
    num_elem = ret.size+2;
    num_elem *= num_elem;
    ret.data = malloc(num_elem*sizeof(elem_t));
    MPI_Recv(ret.data, num_elem, MPI_ELEM_T, rank, 0, MPI_COMM_WORLD, &s);
    return ret;
}

struct world_data world_get_data(world w) {
    struct world_data d;
    d.data = w->sim[1-w->valid_sim_space];
    d.size = w->size;
    d.includes_halo = true;
    return d;
}

void world_print(world w, FILE* out) {
    world_data_print(world_get_data(w), out);
}

void world_print_raw(world w, FILE* out) {
    world_data_print_raw(world_get_data(w), out);
}

void world_data_print_raw(struct world_data d, FILE* out) {
    unsigned i, j;
    elem_t* p = d.data;
    for (j = 0; j < d.size + 2; ++j) {
        for (i = 0; i < d.size + 2; ++i) {
            fprintf(out, ELEM_T_FMT " ", *p++);
        }
        fprintf(out, "\n");
    }
}

void world_data_print(struct world_data d, FILE* out) {
    unsigned i, j;
    /* skip over top halo */
    elem_t* p = d.data + (d.size + 2);
    for (j = 1; j < d.size + 1; ++j) {
        ++p; /* skip over left halo */
        for (i = 1; i < d.size + 1; ++i) {
            fprintf(out, ELEM_T_FMT " ", *p++);
        }
        ++p; /* skip over right halo */
        fprintf(out, "\n");
    }
}

elem_t* world_get(world w, unsigned x, unsigned y) {
    return &(w->sim[1-w->valid_sim_space][(x + 1) + y*(w->size+2)]);
}

unsigned world_size(world w) {
    return w->size;
}

void world_abs_apply(world w, unsigned x, unsigned y, world_xform_fn fn) {
    unsigned size = w->size;
    unsigned rank = w->rank;
    unsigned abssz = w->abs_size;
    unsigned edge_procs = w->edge_procs;
    struct coord c = {x / size, y / size};
    unsigned reside_rank = from_coord(c, edge_procs);
    if (rank == reside_rank) {
        unsigned i = x % size;
        unsigned j = y % size;
        elem_t* p = &(w->sim[1-w->valid_sim_space][(i+1) + j*(w->size+2)]);
        elem_t neighborhood[8];
        neighborhood[M_UPLEFT] = *(p - (size + 2) - 1);
        neighborhood[M_UP] = *(p - (size + 2));
        neighborhood[M_UPRIGHT] = *(p - (size + 2) + 1);
        neighborhood[M_LEFT] = *(p - 1);
        neighborhood[M_RIGHT] = *(p + 1);
        neighborhood[M_DOWNLEFT] = *(p + (size + 2) - 1);
        neighborhood[M_DOWN] = *(p + (size + 2));
        neighborhood[M_DOWNRIGHT] = *(p + (size + 2) + 1);
        *p = fn(*p, neighborhood, x, y, abssz);
    }
}

void world_slice_reduce(world w, unsigned factor, struct world_data* data) {
    unsigned size = w->size;
    do_assert(size % factor == 0, "scale factor must divide world evenly");
    unsigned num = size / factor;
    elem_t* space = w->sim[1-w->valid_sim_space];
    if (data->size != num) {
        world_data_realloc(data, num);
    }
    for (unsigned i = 0; i < num; ++i) {
        for (unsigned j = 0; j < num; ++j) {
            elem_t sum = 0;
            for (unsigned k = 0; k < factor; ++k) {
                for (unsigned l = 0; l < factor; ++l) {
                    sum += space[(j*factor+l+1)*(size+2)+(i*factor+k+1)];
                }
            }
            data->data[j*num+i] = sum/(factor*factor);
        }
    }
}

void world_gather_scaled(world w, unsigned rank, unsigned factor, struct world_data* data) {
    unsigned abssz = w->abs_size;
    unsigned size = w->size;
    unsigned edge = w->edge_procs;
    static elem_t* buf = NULL;
    static unsigned buf_alloc = 0;
    static struct world_data slice = {0};
    /* TODO: Report errors */
    do_assert(size % factor == 0, "scale factor must divide world evenly");
    do_assert(data, "data must not be null");
    unsigned num = size / factor;
    if (w->rank == rank) {
        if (data->size != abssz / factor) {
            world_data_realloc(data, abssz / factor);
        }
        unsigned needed_size = num*edge;
        needed_size *= needed_size;
        if (buf_alloc != needed_size) {
            buf_alloc = needed_size;
            buf = realloc(buf, needed_size*sizeof(elem_t));
        }
    }
    world_slice_reduce(w, factor, &slice);
    mpi_assert(MPI_Gather(slice.data, num*num, MPI_ELEM_T,
            buf, num*num, MPI_ELEM_T, rank, MPI_COMM_WORLD));
    if (w->rank == rank) {
        for (int j = 0; j < num*edge; ++j) {
            for (int i = 0; i < edge; ++i) {
                unsigned r = i + (j/num)*edge;
                memcpy(data->data + (j*edge+i)*num, 
                        buf + (r*num + (j%num))*num, num*sizeof(elem_t));
            }
        }
    }
}
