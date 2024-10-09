#ifndef ANIFEM_UTILS_MESH_UTILS_INL
#define ANIFEM_UTILS_MESH_UTILS_INL

#include "anifem++/fem/quadrature_formulas.h"
#include "utils.h"

template<typename Traits>
void eval_op_var_at_point(double* out, INMOST::Cell c, std::array<double, 3> x_point, const Ani::ApplyOpBase& op, Ani::AssemblerT<Traits>& discr, INMOST::Tag var_tag, const int* component, unsigned int ncomp){
    using namespace Ani;
    INMOST::Mesh* m = c.GetMeshLink();
    INMOST::ElementArray<INMOST::Node> nds(m, 4);
    INMOST::ElementArray<INMOST::Edge> eds(m); INMOST::ElementArray<INMOST::Face> fcs(m);
    Ani::collectConnectivityInfo(c, nds, eds, fcs, true, false);
    double X[12]{};
    for (int n = 0; n < 4; ++n)
        for (int k = 0; k < 3; ++k)
            X[3*n + k] = nds[n].Coords()[k];
    Ani::Tetra<const double> XYZ(X+0, X+3, X+6, X+9);
    std::size_t nfa = op.Nfa();
    auto wreq = fem3DapplyX_memory_requirements(op, 1);
    wreq.dSize += nfa;
    std::vector<char> buf(wreq.enoughRawSize());
    wreq.allocateFromRaw(buf.data(), buf.size());
    DenseMatrix<> dofs(wreq.ddata, nfa, 1);
    wreq.ddata += nfa, wreq.dSize -= nfa;
    discr.GatherDataOnElement(var_tag, c, dofs.data, component, ncomp);
    DenseMatrix<> Ua(out, op.Dim(), 1);
    fem3DapplyX(XYZ, ArrayView<const double>(x_point.data(), 3), dofs, op, Ua, wreq);
    return;
}
template<std::size_t N, typename Traits>
std::array<double, N> eval_op_var_at_point(INMOST::Cell c, std::array<double, 3> x_point, const Ani::ApplyOpBase& op, Ani::AssemblerT<Traits>& discr, INMOST::Tag var_tag, const int* component, unsigned int ncomp){
    assert(N >= op.Dim() && "Not enough memory to save the result");
    std::array<double, N> res{0};
    eval_op_var_at_point(res.data(), c, x_point, op, discr, var_tag, component, ncomp);
    return res;
}
template<std::size_t N, typename Traits>
std::array<double, N> eval_op_var_at_point(INMOST::Cell c, std::array<double, 3> x_point, const Ani::ApplyOpBase& op, Ani::AssemblerT<Traits>& discr, INMOST::Tag var_tag, std::initializer_list<int> components){ 
    return eval_op_var_at_point<N>(c, x_point, op, discr, var_tag, components.begin(), components.size()); 
}

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