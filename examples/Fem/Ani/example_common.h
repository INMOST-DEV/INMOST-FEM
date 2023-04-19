#include "inmost.h"


#if defined(USE_MPI)
#define BARRIER MPI_Barrier(MPI_COMM_WORLD);
#else
#define BARRIER
#endif

void InmostInit(int* argc, char** argv[], const std::string& solver_db, int& pRank, int& pCount);
void InmostFinalize();