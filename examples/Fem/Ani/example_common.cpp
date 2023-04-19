#include "example_common.h"
using namespace INMOST;

void InmostInit(int* argc, char** argv[], const std::string& solver_db, int& pRank, int& pCount){
#if defined(USE_MPI)
    int is_inited = 0;
    MPI_Initialized(&is_inited);
    if (!is_inited) MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &pCount); // Get the total number of processors used
    if (pRank == 0)
        std::cout << "Running with MPI using\n";
#else
    pRank = 0, pCount = 1;
    std::cout << "Running without MPI using\n";
#endif
    Mesh::Initialize(argc, argv);
#ifdef USE_PARTITIONER
    Partitioner::Initialize(argc, argv);
#endif
    Solver::Initialize(argc, argv, solver_db.c_str());
}

void InmostFinalize(){
    Solver::Finalize();
#ifdef USE_PARTITIONER
    Partitioner::Finalize();
#endif
    Mesh::Finalize();
#if defined(USE_MPI)
    int flag = 0;
    MPI_Finalized(&flag);
    if (!flag) MPI_Finalize();
#endif
}