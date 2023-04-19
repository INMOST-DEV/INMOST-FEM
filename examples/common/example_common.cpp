#include "example_common.h"
#include "anifem++/fem/spaces/spaces.h"
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

Ani::FemSpace choose_space_from_name(const std::string& name){
    using namespace Ani;
    std::map<std::string, int> conv {
        {"P0", 0}, {"P1", 1}, {"P2", 2}, {"P3", 3},
        {"MINI", 4}, {"MINI1", 4}, {"MINI2", 5}, {"MINI3", 6},
    };
    auto it = conv.find(name);
    if (it == conv.end())
        throw std::runtime_error("Faced unknown space name = \"" + name + "\"");
    switch (it->second){
        case 0: return Ani::FemSpace{ P0Space{} };
        case 1: return Ani::FemSpace{ P1Space{} };
        case 2: return Ani::FemSpace{ P2Space{} };
        case 3: return Ani::FemSpace{ P3Space{} };
        case 4: return Ani::FemSpace{ P1Space{} } + Ani::FemSpace{ BubbleSpace{} };
        case 5: return Ani::FemSpace{ P2Space{} } + Ani::FemSpace{ BubbleSpace{} };
        case 6: return Ani::FemSpace{ P3Space{} } + Ani::FemSpace{ BubbleSpace{} };
    }
    throw std::runtime_error("Doesn't find space with specified name = \"" + name + "\"");
    return FemSpace{};   
}

void print_mesh_sizes(INMOST::Mesh* m){
    long nN = m->TotalNumberOf(NODE), nE = m->TotalNumberOf(EDGE), nF = m->TotalNumberOf(FACE), nC = m->TotalNumberOf(CELL);
    if (m->GetProcessorRank() == 0) {
        std::cout << "Mesh info:"
            << " #N " << nN << " #E " << nE << " #F " << nF << " #T " << nC << std::endl;
    }
}

void print_linear_solver_status(INMOST::Solver& s, const std::string& prob_name, bool exit_on_fail){
    int pRank = 0, pCount = 1;
    #if defined(USE_MPI)
        MPI_Comm_rank(MPI_COMM_WORLD, &pRank);  // Get the rank of the current process
        MPI_Comm_size(MPI_COMM_WORLD, &pCount); // Get the total number of processors used
    #endif
    bool success_solve = s.IsSolved();
    if(!success_solve){
        if (pRank == 0) std::cout << prob_name << ":\n\tsolution failed:\n";
        for (int p = 0; p < pCount; ++p) {
            BARRIER;
            if (pRank != p) continue;
            std::cout << "\t               : #lits " << s.Iterations() << " residual " << s.Residual() << ". "
                      << "preconding = " << s.PreconditionerTime() << "s, solving = " << s.IterationsTime() << "s" << std::endl;
            std::cout << "\tRank " << pRank << " failed to solve system. ";
            std::cout << "Reason: " << s.GetReason() << std::endl;
        }
        if (exit_on_fail)
            exit(-1);
    }
    else{
        if(pRank == 0) {
            std::cout << prob_name << ":\n";
            std::string _s_its = std::to_string(s.Iterations()), s_its = "    ";
            std::copy(_s_its.begin(), _s_its.end(), s_its.begin() + 3 - _s_its.size());

            std::cout << "\tsolved_succesful: #lits " << s_its << " residual " << s.Residual() << ", "
                      << "preconding = " << s.PreconditionerTime() << "s, solving = " << s.IterationsTime() << "s" << std::endl;
        }
    }
}