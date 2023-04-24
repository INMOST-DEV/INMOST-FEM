#pragma once

#include "inmost.h"
#include "anifem++/fem/fem_space.h"
#include "anifem++/fem/quadrature_formulas.h"


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
INMOST::Tag createFemVarTag(INMOST::Mesh* m, const Ani::DofT::BaseDofMap& dofmap, const std::string& tag_name = "var", bool use_fixed_size = true);

template<int N = 1, typename FUNC>
std::array<double, N> integrate_vector_func(INMOST::Mesh* m, const FUNC& f, uint order = 5){
    std::array<double, N> res = {0};
    auto formula = tetrahedron_quadrature_formulas(order);
    uint q = formula.GetNumPoints();
    for (auto it = m->BeginCell(); it != m->EndCell(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
        auto nds = it->getNodes();
        double vol = it->Volume();
        double XY[4][3] = {0};
        for (int ni = 0; ni < 4; ++ni)
            for (int k = 0; k < 3; ++k)
                XY[ni][k] = nds[ni].Coords()[k];
        for (uint n = 0; n < q; ++n){
            std::array<double, 3> x = {0};
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k)
                    x[k] += formula.GetPointData()[4*n+i]*XY[i][k];
            std::array<double, N> val = f(it->getAsCell(), x);
            double w = formula.GetWeightData()[n];
            for (int l = 0; l < N; ++l)
                res[l] += w*vol*val[l];        
        }        
    }
    m->Integrate(res.data(), N);
    return res;
}
template<typename FUNC>
double integrate_scalar_func(INMOST::Mesh* m, const FUNC& f, uint order = 5){
    return integrate_vector_func(m, [&f](const INMOST::Cell& c, const Ani::Coord<> &X)->std::array<double, 1>{ return {f(c, X)}; }, order)[0];
}