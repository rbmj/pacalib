#ifndef MPI_ASSERT_H_INC
#define MPI_ASSERT_H_INC
#include <mpi.h>
#include <stdlib.h>

#define do_assert(sym, msg) \
    do { \
        if (!(sym)) { \
            fputs(msg, stderr); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
            exit(1); \
        } \
    } while (0)

#define mpi_assert(sym) do_assert((sym) == MPI_SUCCESS, "MPI Error\n")

#endif
