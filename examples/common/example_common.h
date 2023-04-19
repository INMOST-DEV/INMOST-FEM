#pragma once

#include "inmost.h"
#include "anifem++/fem/fem_space.h"


#if defined(USE_MPI)
#define BARRIER MPI_Barrier(MPI_COMM_WORLD);
#else
#define BARRIER
#endif

void InmostInit(int* argc, char** argv[], const std::string& solver_db, int& pRank, int& pCount);
void InmostFinalize();

/// @brief Create unite 1D space from name
/// @param name is one of "P0", "P1", "P2", "P3", "MINI" (=P1+B4), "MINI1" (=P1+B4), "MINI2" (=P2+B4), "MINI3" (=P3+B4)
/// @return corresponding space
Ani::FemSpace choose_space_from_name(const std::string& name);

void print_mesh_sizes(INMOST::Mesh* m);
void print_linear_solver_status(INMOST::Solver& s, const std::string& prob_name = "problem", bool exit_on_fail = false);