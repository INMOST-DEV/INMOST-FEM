#ifndef ANIFEM_UTILS_MESH_UTILS_INL
#define ANIFEM_UTILS_MESH_UTILS_INL

#include "anifem++/fem/quadrature_formulas.h"
#include "utils.h"

template<int N, typename FUNC>
std::array<double, N> integrate_vector_func(INMOST::Mesh* m, const FUNC& f, uint order){
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
double integrate_scalar_func(INMOST::Mesh* m, const FUNC& f, uint order){
    return integrate_vector_func(m, [&f](const INMOST::Cell& c, const Ani::Coord<> &X)->std::array<double, 1>{ return {f(c, X)}; }, order)[0];
}

#endif //ANIFEM_UTILS_MESH_UTILS_INL