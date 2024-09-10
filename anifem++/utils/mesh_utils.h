#include "inmost.h"
#include <array>
#include <memory>

#ifndef ANIFEM_UTILS_MESH_UTILS_H
#define ANIFEM_UTILS_MESH_UTILS_H

/** Create mesh for cubic domain [0, size]^3
 *  @param nx, ny, nz are numbers of steps per axis
**/
std::unique_ptr<INMOST::Mesh> GenerateCube(INMOST_MPI_Comm comm, unsigned nx, unsigned ny, unsigned nz, double size = 1.0);
///Create mesh for domain [O[0], A[0]]x[O[1], A[1]]x[O[2], A[2]]
///@param naxis are numbers of steps per axis
std::unique_ptr<INMOST::Mesh> GenerateParallelepiped(INMOST_MPI_Comm comm, std::array<unsigned, 3> naxis, std::array<double, 3> A = {1, 1, 1}, std::array<double, 3> O = {0, 0, 0});
/// Redistribute mesh over mpi-processors 
void RepartMesh(INMOST::Mesh* m);
/// Print global mesh discretization statistics
void print_mesh_sizes(INMOST::Mesh* m);

#endif //ANIFEM_UTILS_MESH_UTILS_H