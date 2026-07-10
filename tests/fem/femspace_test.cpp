#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include "anifem++/fem/fem_space.h"
#include "anifem++/fem/complement_fem_space.h"
#include "anifem++/fem/quadrature_formulas.h"
#include "anifem++/fem/geometry.h"
#include "anifem++/fem/spaces/spaces.h"
#include "anifem++/fem/operations/operations.h"

namespace {

template<typename CompSpace>
int complement_edge_row(const CompSpace& comp, int vi, int vj){
    const int a = std::min(vi, vj), b = std::max(vi, vj);
    static const unsigned char I[6] = {0, 0, 0, 1, 1, 2}, J[6] = {1, 2, 3, 2, 3, 3};
    for (int k = 0; k < static_cast<int>(comp.m_dof_tags.size()); ++k){
        const auto& tag = comp.m_dof_tags[k];
        if (!(tag.etype & Ani::DofT::EDGE))
            continue;
        if (tag.nelem >= 0 && tag.nelem < 6 && I[tag.nelem] == static_cast<unsigned char>(a) && J[tag.nelem] == static_cast<unsigned char>(b))
            return k;
    }
    return -1;
}

} // namespace

struct AniInterface_FemSpace_evaluateT{
    template<typename FEMTYPE, typename FUNC>
    static std::vector<double> interp(Ani::Tetra<const double> XYZ, const FUNC& f){
        using namespace Ani;
        const int NF = Operator<IDEN, FEMTYPE>::Nfa::value;
        std::vector<double> res(NF);
        interpolateByDOFs<FEMTYPE>(XYZ, f, ArrayView<>(res.data(), NF), DofT::TetGeomSparsity().setCell(true));
        return res;
    }
    template<int OPERATOR, typename FEMTYPE>
    static std::vector<double> eval_f(Ani::Tetra<const double> XYZ, Ani::ArrayView<> udofs, const std::array<double, 3>& x){
        using namespace Ani;
        const int Dim = Operator<OPERATOR, FEMTYPE>::Dim::value;
        const int NFA = Operator<IDEN, FEMTYPE>::Nfa::value;
        std::vector<double> res(Dim);
        DenseMatrix<> u(res.data(), Dim, 1);
        DenseMatrix<> dofs(udofs.data, NFA, 1);
        auto req = fem3DapplyX_memory_requirements<Operator<OPERATOR, FEMTYPE>>(x.size()/3, XYZ.fusion);  
        std::vector<char> buf(req.enoughRawSize());
        req.allocateFromRaw(buf.data(), buf.size());
        fem3DapplyX<Operator<OPERATOR, FEMTYPE>>(XYZ, ArrayView<const double>(x.data(), x.size()), ArrayView<double>(dofs.data, dofs.nCol*dofs.nRow), ArrayView<>(u.data, u.size), req);
        if (OPERATOR == OperatorType::GRAD) {
            int dim = Dim;
            u.Init(res.data(), dim/3, 3, dim);
            std::vector<double> tmp(dim);
            std::copy(res.begin(), res.end(), tmp.begin());
            for (uint i = 0; i < static_cast<uint>(dim / 3); ++i)
                for (int k = 0; k < 3; ++k)
                    u(i, k) = tmp[k + 3*i];
        }
        return res;
    }
    template<int OPERATOR, typename FEMTYPE, typename FUNC1, typename FUNC2>
    static void test(Ani::Tetra<const double> XYZ, std::array<double, 3> x, const FUNC1& F, const FUNC2& OpF, bool print_stat = false){
        using namespace Ani;
        auto udofs_d = interp<FEMTYPE, FUNC1>(XYZ, F);
        ArrayView<> udofs(udofs_d.data(), udofs_d.size());
        if (print_stat)
            std::cout << "udofs = " << DenseMatrix<>(udofs.data, 1, udofs.size) << std::endl;
        auto eval = eval_f<OPERATOR, FEMTYPE>(XYZ, udofs, x);
        std::vector<double> exact(Operator<OPERATOR, FEMTYPE>::Dim::value);
        OpF(x, exact.data(), exact.size(), nullptr);
        DenseMatrix<>   Uex(exact.data(), exact.size(), 1), 
                        Uev(eval.data(), eval.size(), 1);
        if (print_stat)
            std::cout   << "Uex = \n" << Uex 
                        << "\nUev = \n" << Uev << std::endl;                
        EXPECT_NEAR(Uex.ScalNorm(1, Uex, -1, Uev), 0, 100*(1 + Uex.ScalNorm(1, Uex))*DBL_EPSILON) << "Operator = " << static_cast<int>(OPERATOR) << " space = " << typeid(FEMTYPE).name();
    }
};

TEST(AniInterface, FemSpace){
    using evaluateT = AniInterface_FemSpace_evaluateT;
    using namespace Ani;

    FemSpace P0{ P0Space{} };
    FemSpace P1{ P1Space{} };
    FemSpace P2{ P2Space{} };
    FemSpace P3{ P3Space{} };
    FemSpace B4{ BubbleSpace{} };
    FemSpace RT0{ RT0Space{} };
    FemSpace CR1{ CR1Space{} };
    FemSpace ND0{ ND0Space{} };

    FemSpace MINI1 = P1 + B4;
    FemSpace MINI2 = P2 + B4;
    FemSpace VecP1 = P1^3;
    FemSpace VecP2 = P2^3;
    FemSpace CmplP2 = make_complex_raw(std::vector<FemSpace>{P2, P2, P2});
    FemSpace Cmpl1P2 = (P2*P2)*P2;
    FemSpace Cmpl2P2 = make_complex_raw({P2, make_complex_raw({P2, P2}) });
    FemSpace Cmpl3P2 = make_complex_raw({make_complex_raw({P2, P2}), P2 });
    FemSpace Cmpl3 = (P2^2)*P3;
    FemSpace Cmpl4 = P3*(P2^2);
    FemSpace P2P1 = P2 - P1;
    FemSpace P3P2 = P3 - P2;
    FemSpace P3P1 = P3 - P1;
    FemSpace MINI1P1 = MINI1 - P1;
    FemSpace P1_plus_P2P1 = P1 + P2P1;
    FemSpace P2P1_vec = P2P1^3;
    FemSpace P2P1_vec_direct = (P2^3) - (P1^3);
    FemSpace EP2P1 = ComplementFemSpace::make(P2, P1, ComplementFemSpace::EnergyComplement);
    FemSpace EP3P1 = ComplementFemSpace::make(P3, P1, ComplementFemSpace::EnergyComplement);
    FemSpace EP2P1_vec = EP2P1 ^ 3;
    FemSpace EP2P1_vec_direct = ComplementFemSpace::make(P2 ^ 3, P1 ^ 3, ComplementFemSpace::EnergyComplement);

    EXPECT_EQ(P2P1_vec.dofMap().NumDofOnTet(), P2P1_vec_direct.dofMap().NumDofOnTet());
    EXPECT_EQ(P2P1_vec.gatherType(), BaseFemSpace::BaseTypes::VectorType);
    EXPECT_EQ(P2P1_vec_direct.gatherType(), BaseFemSpace::BaseTypes::VectorType);
    EXPECT_EQ(P3P2.dofMap().NumDofOnTet(), P3.dofMap().NumDofOnTet() - P2.dofMap().NumDofOnTet());
    EXPECT_EQ(P3P1.dofMap().NumDofOnTet(), P3.dofMap().NumDofOnTet() - P1.dofMap().NumDofOnTet());
    EXPECT_EQ(MINI1P1.dofMap().NumDofOnTet(), 1u);
    EXPECT_EQ(P1_plus_P2P1.dofMap().NumDofOnTet(), P2.dofMap().NumDofOnTet());
    EXPECT_EQ(EP2P1.gatherType(), BaseFemSpace::BaseTypes::ComplementType);
    EXPECT_EQ(EP2P1.dofMap().NumDofOnTet(), P2.dofMap().NumDofOnTet() - P1.dofMap().NumDofOnTet());
    EXPECT_EQ(EP3P1.dofMap().NumDofOnTet(), P3.dofMap().NumDofOnTet() - P1.dofMap().NumDofOnTet());
    EXPECT_EQ(EP2P1_vec.dofMap().NumDofOnTet(), EP2P1_vec_direct.dofMap().NumDofOnTet());
    EXPECT_EQ(EP2P1_vec.gatherType(), BaseFemSpace::BaseTypes::VectorType);
    EXPECT_EQ(EP2P1_vec_direct.gatherType(), BaseFemSpace::BaseTypes::VectorType);

    EXPECT_TRUE(MINI1.dofMap() == DofT::merge({P1.dofMap(), B4.dofMap()}));
    EXPECT_TRUE(VecP2.dofMap() == DofT::pow(P2.dofMap(), 3));
    EXPECT_TRUE(CmplP2.dofMap() == DofT::merge({P2.dofMap(), P2.dofMap(), P2.dofMap()}));
    EXPECT_TRUE(Cmpl1P2 == CmplP2);
    EXPECT_FALSE(Cmpl2P2 == CmplP2);
    EXPECT_TRUE(Cmpl2P2.dofMap() == DofT::merge({P2.dofMap(), DofT::merge({P2.dofMap(), P2.dofMap()}) }));
    EXPECT_TRUE(Cmpl3.dofMap() == DofT::merge({ DofT::pow(P2.dofMap(), 2), P3.dofMap()}));
    EXPECT_TRUE(P1.target<P1Space>()->polyOrder() == 1);
    EXPECT_TRUE(P2.target<P2Space>()->polyOrder() == 2);
    EXPECT_FALSE(P1 == P2);
    EXPECT_FALSE(P1 == B4);
    EXPECT_FALSE(MINI1 == MINI2);
    EXPECT_FALSE(P1 == MINI2);
    EXPECT_FALSE(VecP2 == CmplP2);
    EXPECT_TRUE(MINI1 == make_union_raw({P1, B4}));
    EXPECT_TRUE(CmplP2 == make_complex_with_simplification({P2, P2*P2}));

    using Mini1T = FemUnionT<FemFix<FEM_P1>, FemFix<FEM_B4>>;
    using Mini2T = FemUnionT<FemFix<FEM_P2>, FemFix<FEM_B4>>;
    EXPECT_TRUE(MINI1.dofMap() != DofT::DofMap(Ani::Dof<Mini1T>::Map())); //BaseTypes::ComplexTemplateType != BaseTypes::ComplexType
    EXPECT_TRUE(MINI1.order() == uint(Operator<IDEN, Mini1T>::Order::value));
    EXPECT_TRUE(MINI1.dim() == uint(Operator<IDEN, Mini1T>::Dim::value));
    EXPECT_TRUE(uint(Operator<IDEN, Ani::FemVecT<3, Mini1T>>::Nfa::value) == 15);

    {
        const double Mep[] = { 
            1, 0, 0, 0, -0.25,
            0, 1, 0, 0, -0.25,
            0, 0, 1, 0, -0.25,
            0, 0, 0, 1, -0.25,
            0, 0, 0, 0,  1.00
        };
        DenseMatrix<const double> Me(Mep, 5, 5);
        auto M = MINI1.target<UnionFemSpace>()->GetOrthBasisShiftMatrix();
        EXPECT_NEAR(Me.ScalNorm(1, Me, -1, M), 0, 100*(1 + Me.ScalNorm(1, Me))*DBL_EPSILON) << "Problem with UnionSpace(P1, Bubble) orthoganalize matrix";
        M = Mini1T::GetOrthBasisShiftMatrix();
        EXPECT_NEAR(Me.ScalNorm(1, Me, -1, M), 0, 100*(1 + Me.ScalNorm(1, Me))*DBL_EPSILON) << "Problem with FemUnionT<FemFix<FEM_P1>, FemFix<FEM_B4>> orthoganalize matrix";
    }
    
    double XYp[] = {0, 0, 0, 2, 1, 1, 1, 2, 1, 2, 1, 2};
    Tetra<const double> XYZ(XYp+0, XYp+3, XYp+6, XYp+9);
    std::vector<char> wmem;
    Ani::PlainMemoryX<> mem;
    std::vector<double> udofs;
    {   
        auto check_interpolation = [&wmem, &mem, &udofs, &XYZ](const FemSpace& s, const auto& f, const DofT::TetGeomSparsity& sp, const std::vector<double> udofs_expected, bool print_eval = false)->void{
            uint nf = s.dofMap().NumDofOnTet();
            int int_order = 5;
            udofs.resize(nf);
            std::fill(udofs.begin(), udofs.end(), 0);
            mem = s.interpolateByDOFs_mem_req(int_order);
            wmem.resize(mem.enoughRawSize());
            mem.allocateFromRaw(wmem.data(), wmem.size());
            s.interpolateByDOFs(XYZ, f, ArrayView<>(udofs.data(), nf), sp, mem, nullptr, int_order);
            if (print_eval)
                std::cout << DenseMatrix<const double>(udofs.data(), 1, nf) << std::endl;
            EXPECT_TRUE(udofs_expected.size() == nf);
            if (udofs_expected.size() == nf){
                DenseMatrix<const double> U(udofs.data(), nf, 1), Ue(udofs_expected.data(), nf, 1);
                EXPECT_NEAR(Ue.ScalNorm(1, Ue, -1, U), 0, 100*(1 + Ue.ScalNorm(1, Ue))*DBL_EPSILON);
            }
        };
        auto check_interpolationT = [&wmem, &mem, &udofs, &XYZ](auto s, const auto& f, const DofT::TetGeomSparsity& sp, const std::vector<double> udofs_expected, bool print_eval = false)->void{
            using FemT = std::remove_cv_t<std::remove_reference_t<decltype(s)>>;
            const int NF = Operator<IDEN, FemT>::Nfa::value;
            udofs.resize(NF);
            std::fill(udofs.begin(), udofs.end(), 0);
            interpolateByDOFs<FemT>(XYZ, f, ArrayView<>(udofs.data(), NF), sp);
            if (print_eval)
                std::cout << DenseMatrix<const double>(udofs.data(), 1, NF) << std::endl;
            EXPECT_TRUE(udofs_expected.size() == NF);
            if (udofs_expected.size() == NF){
                DenseMatrix<const double> U(udofs.data(), NF, 1), Ue(udofs_expected.data(), NF, 1);
                EXPECT_NEAR(Ue.ScalNorm(1, Ue, -1, U), 0, 100*(1 + Ue.ScalNorm(1, Ue))*DBL_EPSILON);
            }
        };

        int shift = 0;
        auto f = [&shift](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data;
            const int nvals = 4;
            double vals[nvals] = {1, x[0], x[1]*x[1] + x[2]*x[2], 1 + x[0]*x[1]};
            for (uint d = 0; d < dim; ++d)
                res[d] = vals[(shift+d)%nvals];
            return 0;
        };
        auto f_div = [](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data; (void) dim;
            double w[] = {1, 7, 11};
            cross(x.data(), w, res);
            return 0;
        };
        auto f_rot = [](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data; (void) dim;
            double a = 3.;
            double b[] = { 1, 5, 11};
            for (int i = 0; i < 3; ++i) 
                res[i] = a*x[i] + b[i];
            return 0;    
        };
        double udofs[1] = {-1};
        mem = P0.interpolateOnDOF_mem_req(0, 5);
        wmem.resize(mem.enoughRawSize());
        mem.allocateFromRaw(wmem.data(), wmem.size());
        P0.interpolateOnDOF(XYZ, f, ArrayView<>(udofs, 1), 0, mem, nullptr, 5);
        EXPECT_NEAR(udofs[0]-1, 0, 10*DBL_EPSILON);
        udofs[0] = -1;
        P0.interpolateConstant(1, ArrayView<>(udofs, 1), DofT::TetGeomSparsity().setCell(true));
        EXPECT_NEAR(udofs[0]-1, 0, 10*DBL_EPSILON);

        //predefined spaces interpolate test
        shift = 1;
        auto sp = DofT::TetGeomSparsity().setFace(1, true);
        check_interpolation(P1, f, sp, {0, 2, 1, 2}, false);
        check_interpolation(P2, f, sp, {0, 2, 1, 2, 0, 0, 0, 1.5, 2, 1.5}, false);
        check_interpolation(P3, f, sp, {0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 5.0/3, 4.0/3, 2, 2, 4.0/3, 5.0/3, 0, 5.0/3, 0, 0}, false);
        check_interpolation(P3, f, DofT::TetGeomSparsity().setCell(true), 
                                        {0, 2, 1, 2, 2.0/3, 4.0/3, 1.0/3, 2.0/3, 2.0/3, 4.0/3, 5.0/3, 4.0/3, 2, 2, 4.0/3, 5.0/3, 1, 5.0/3, 1, 4./3}, false);
        check_interpolation(B4, f, DofT::TetGeomSparsity().setCell(true), {1.25}, false);
        check_interpolation(CR1, f, sp, {0, 5./3, 0, 0}, false);
        check_interpolation(ND0, f_div, sp, {0, 0, 0, -25/sqrt(2), 13, 10*sqrt(3)}, false);
        check_interpolation(RT0, f_rot, sp, {0, 15/sqrt(2), 0, 0}, false);

        //compound femspace interpolate tests
        check_interpolation(MINI1, f, sp, {0, 2, 1, 2, 0}, false);
        check_interpolation(MINI1, f, DofT::TetGeomSparsity().setFace(1, true).setCell(), {0, 2, 1, 2, 1.25}, false);
        check_interpolationT(Mini1T{}, f, sp, {0, 2, 1, 2, 0}, false);
        check_interpolationT(Mini1T{}, f, DofT::TetGeomSparsity().setFace(1, true).setCell(), {0, 2, 1, 2, 1.25}, false);
        check_interpolation(MINI2, f, DofT::TetGeomSparsity().setFace(1, true).setCell(), {0, 2, 1, 2, 0, 0, 0, 1.5, 2, 1.5, 1.25}, false);
        check_interpolationT(Mini2T{}, f, DofT::TetGeomSparsity().setFace(1, true).setCell(), {0, 2, 1, 2, 0, 0, 0, 1.5, 2, 1.5, 1.25}, false);
        auto P123dofs = std::vector<double>{0,2,1,2, 0,2,5,5,0,0,0,3.25,3.25,4.5, 0,3,3,3,0,0,0,0,0,0,29./9,29./9,3,3,29./9,29./9,0,29./9,0,0 };
        check_interpolation(P1*P2*P3, f, sp, P123dofs, false);
        check_interpolationT(FemCom<FemFix<FEM_P1>, FemFix<FEM_P2>, FemFix<FEM_P3>>{}, f, sp, P123dofs, false);
        check_interpolation(make_complex_raw({P1, make_complex_raw({P2, P3})}), f, sp, P123dofs, false);
        check_interpolation(make_complex_raw({make_complex_raw({P1, P2}), P3}), f, sp, P123dofs, false);
        check_interpolation(P1^3, f, sp, {0,2,1,2, 0,2,5,5, 0,3,3,3}, false);
        check_interpolation((P1^2)*P3, f, sp, {0,2,1,2, 0,2,5,5, 0,3,3,3,0,0,0,0,0,0,29./9,29./9,3,3,29./9,29./9,0,29./9,0,0}, false);
    }
    { //compound femspace evaluate tests
        using Func = std::function<int(const std::array<double, 3>&, double*, ulong, void*)>;
        auto fconv = [](auto f) -> Func{
            return [f](const std::array<double, 3>& x, double* res, ulong dim, void* user_data) -> int{
                (void) dim; (void) user_data;
                auto v = f(x);
                for (uint i = 0; i < v.size(); ++i)
                    res[i] = v[i];
                return 0;    
            };
        };
        auto fp0 = fconv([](const std::array<double, 3>& x){ (void) x; return std::array<double, 1>{1}; });
        auto fp1 = fconv([](const std::array<double, 3>& x){ return std::array<double, 1>{x[0]}; });
        auto fp2 = fconv([](const std::array<double, 3>& x){ return std::array<double, 1>{x[1]*x[1] + x[2]*x[2]}; });
        auto fp2_1 = fconv([](const std::array<double, 3>& x){ return std::array<double, 1>{1 + x[1]*x[2]}; });
        auto fp3 = fconv([](const std::array<double, 3>& x){ return std::array<double, 1>{x[1]*x[2]*x[2]}; }); 
        Func f_div = [](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data; (void) dim;
            double w[] = {1, 7, 11};
            cross(x.data(), w, res);
            return 0;
        };
        Func f_rot = [](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data; (void) dim;
            double a = 3.;
            double b[] = { 1, 5, 11};
            for (int i = 0; i < 3; ++i) 
                res[i] = a*x[i] + b[i];
            return 0;    
        };
        auto mix_f = [](std::vector<std::pair<Func, uint>> funcs) -> Func{
            std::vector<Func> fcs(funcs.size());
            std::vector<uint> shft(funcs.size()+1);
            shft[0] = 0;
            for (uint i = 0; i < funcs.size(); ++i){
                shft[i+1] = shft[i] + funcs[i].second;
                fcs[i] = std::move(funcs[i].first);
            }
            return [fcs, shft](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
                (void) user_data; (void) dim;
                assert(dim == shft.back() - shft.front() && "Wrong dimension");
                for (uint i = 0; i < fcs.size(); ++i)
                    fcs[i](x, res+shft[i], shft[i+1] - shft[i], user_data);
                return 0;    
            };
        };

        auto dfp0 = fconv([](const std::array<double, 3>& x){ (void) x; return std::array<double, 3>{0, 0, 0}; });
        auto dfp1 = fconv([](const std::array<double, 3>& x){ (void) x; return std::array<double, 3>{1, 0, 0}; });
        auto dfp2 = fconv([](const std::array<double, 3>& x){ return std::array<double, 3>{0, 2*x[1], 2*x[2]}; });
        auto dfp2_1 = fconv([](const std::array<double, 3>& x){ return std::array<double, 3>{0, x[2], x[1]}; });
        auto dfp3 = fconv([](const std::array<double, 3>& x){ return std::array<double, 3>{0, x[2]*x[2], 2*x[1]*x[2]}; });
        Func df_div = [](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data; (void) dim; (void) x;
            double w[] = {1, 7, 11};
            DenseMatrix<> grF(res, 3, 3);
            grF.SetZero();
            grF(0, 1) = w[2], grF(1, 0) = -w[2];
            grF(0, 2) = -w[1], grF(2, 0) = w[1];
            grF(1, 2) = w[0], grF(2, 1) = -w[0];
            return 0;
        };
        Func df_rot = [](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
            (void) user_data; (void) dim; (void) x;
            DenseMatrix<> grF(res, 3, 3);
            grF.SetZero();
            double a = 3.;
            //double b[] = { 1, 5, 11};
            for (int i = 0; i < 3; ++i)
                grF(i, i) = a;
            return 0;    
        };

        auto mix_df = [](std::vector<std::pair<Func, uint>> funcs) -> Func{
            std::vector<Func> fcs(funcs.size());
            std::vector<uint> shft(funcs.size()+1);
            shft[0] = 0;
            for (uint i = 0; i < funcs.size(); ++i){
                shft[i+1] = shft[i] + funcs[i].second;
                fcs[i] = std::move(funcs[i].first);
            }
            return [fcs, shft](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
                (void) dim;
                assert(dim == shft.back() - shft.front() && "Wrong dimension");
                uint nvar = (shft.back() - shft.front()) / 3;
                double tmp[100] = {0};
                for (uint i = 0; i < fcs.size(); ++i){
                    ulong ldim = shft[i+1] - shft[i];
                    DenseMatrix<> GrU(tmp, ldim/3, 3);
                    fcs[i](x, tmp, shft[i+1] - shft[i], user_data);
                    for (uint d = 0; d < ldim/3; ++d)
                        for (int k = 0; k < 3; ++k)
                            res[shft[i]/3 + d + k*nvar] = GrU(d, k);
                }
                return 0;    
            };
        };

        auto make_div = [](const Func& grad_f) -> Func{
            return [grad_f](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
                (void) user_data; (void) dim;
                assert(dim == 1 && "Wrong dimension");
                double tmp[9] = {0};
                grad_f(x, tmp, 9, user_data);
                res[0] = tmp[0] + tmp[4] + tmp[8];
                return 0;
            };
        };
        auto make_curl = [](const Func& grad_f) -> Func{
            return [grad_f](const std::array<double, 3>& x, double* res, ulong dim, void* user_data)->int{
                (void) user_data; (void) dim;
                assert(dim == 3 && "Wrong dimension");
                double tmp[9] = {0};
                grad_f(x, tmp, 9, user_data);
                DenseMatrix<> GrU(tmp, 3, 3);
                for (int i = 0; i < 3; ++i)
                    res[i] = -GrU((i+1)%3, (i+2)%3) + GrU((i+2)%3, (i+1)%3);
                return 0;
            };
        };

        auto interpolate = [&wmem, &mem, &udofs_mem = udofs, &XYZ](const FemSpace& s, const auto& f)->ArrayView<>{
            uint nf = s.dofMap().NumDofOnTet();
            int int_order = 5;
            udofs_mem.resize(nf);
            std::fill(udofs_mem.begin(), udofs_mem.end(), 0);
            mem = s.interpolateByDOFs_mem_req(int_order);
            wmem.resize(mem.enoughRawSize());
            mem.allocateFromRaw(wmem.data(), wmem.size());
            ArrayView<> udofs(udofs_mem.data(), nf);
            s.interpolateByDOFs(XYZ, f, udofs, DofT::TetGeomSparsity().setCell(true), mem, nullptr, int_order);
            return udofs;
        };
        std::vector<char> wmem1;
        Ani::PlainMemoryX<> mem1;
        auto evaluate = [&wmem = wmem1, &mem = mem1, &XYZ](OperatorType op_type, const FemSpace& s, ArrayView<> udofs, const std::array<double, 3>& x){
            ApplyOpFromSpaceView op = s.getOP(op_type);
            uint dim = s.dimOP(op_type);
            auto req = fem3DapplyX_memory_requirements(op, 1);
            wmem.resize(req.enoughRawSize());
            req.allocateFromRaw(wmem.data(), wmem.size());
            std::vector<double> res(dim);
            DenseMatrix<> u(res.data(), dim, 1);
            DenseMatrix<> dofs(udofs.data, s.dofMap().NumDofOnTet(), 1);
            fem3DapplyX<>(XYZ, ArrayView<const double>(x.data(), x.size()), dofs, op, u, req);
            if (op_type == OperatorType::GRAD) {
                u.Init(res.data(), dim/3, 3, dim);
                std::vector<double> tmp(dim);
                std::copy(res.begin(), res.end(), tmp.begin());
                for (uint i = 0; i < (dim / 3); ++i)
                    for (int k = 0; k < 3; ++k)
                        u(i, k) = tmp[k + 3*i];
            }
            return res;
        };

        std::array<double, 4> lmb{0.1, 0.2, 0.3, 0.4};
        std::array<double, 3> x{0};
        for (int i = 0; i < 4; ++i) 
            for (int k = 0; k < 3; ++k) 
                x[k] += lmb[i] * XYp[k + 3*i];
        auto test_evaluates = [interpolate, evaluate, &x](OperatorType op_type, const FemSpace& s, const auto& F, const auto& OpF, bool print_stat = false){
            auto udofs = interpolate(s, F);
            if (print_stat)
                std::cout << "udofs = " << DenseMatrix<>(udofs.data, 1, udofs.size) << std::endl;
            auto eval = evaluate(op_type, s, udofs, x);
            std::vector<double> exact(s.dimOP(op_type));
            OpF(x, exact.data(), exact.size(), nullptr);
            DenseMatrix<>   Uex(exact.data(), exact.size(), 1), 
                            Uev(eval.data(), eval.size(), 1);
            if (print_stat)
                std::cout   << "Uex = \n" << Uex 
                            << "\nUev = \n" << Uev << std::endl;                
            EXPECT_NEAR(Uex.ScalNorm(1, Uex, -1, Uev), 0, 100*(1 + Uex.ScalNorm(1, Uex))*DBL_EPSILON) << "Operator = " << static_cast<int>(op_type) << " space = " << s.typeName();
        };  

        auto p1_vec = mix_f({{fp1, 1}, {fp1, 1}, {fp1, 1}});
        auto dp1_vec = mix_df({{dfp1, 3}, {dfp1, 3}, {dfp1, 3}});
        auto p2_vec = mix_f({{fp1, 1}, {fp2, 1}, {fp2_1, 1}});
        auto dp2_vec = mix_df({{dfp1, 3}, {dfp2, 3}, {dfp2_1, 3}});   
        
        test_evaluates(IDEN, P0, fp0, fp0);
        test_evaluates(GRAD, P0, fp0, dfp0);
        test_evaluates(IDEN, P1, fp1, fp1);
        test_evaluates(GRAD, P1, fp1, dfp1);
        test_evaluates(IDEN, CR1, fp1, fp1);
        test_evaluates(GRAD, CR1, fp1, dfp1);
        test_evaluates(IDEN, P2, fp2, fp2);
        test_evaluates(GRAD, P2, fp2, dfp2);
        test_evaluates(IDEN, P2, fp2_1, fp2_1);
        test_evaluates(GRAD, P2, fp2_1, dfp2_1);
        test_evaluates(IDEN, P3, fp3, fp3);
        test_evaluates(GRAD, P3, fp3, dfp3);
        test_evaluates(IDEN, MINI1, fp1, fp1);
        test_evaluates(GRAD, MINI1, fp1, dfp1);
        evaluateT::test<IDEN, Mini1T>(XYZ, x, fp1, fp1);
        evaluateT::test<GRAD, Mini1T>(XYZ, x, fp1, dfp1);
        test_evaluates(IDEN, P1^3, p1_vec, p1_vec);
        test_evaluates(GRAD, P3^3, p1_vec, dp1_vec);
        test_evaluates(IDEN, VecP2, p2_vec, p2_vec);
        test_evaluates(GRAD, VecP2, p2_vec, dp2_vec);
        test_evaluates(DIV, VecP2, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, VecP2, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, MINI1^3, p1_vec, p1_vec);
        test_evaluates(GRAD, MINI2^3, p2_vec, dp2_vec);
        test_evaluates(DIV, MINI2^3, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, MINI2^3, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, CmplP2, p2_vec, p2_vec);
        test_evaluates(GRAD, CmplP2, p2_vec, dp2_vec);
        test_evaluates(DIV, CmplP2, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, CmplP2, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, Cmpl2P2, p2_vec, p2_vec);
        test_evaluates(GRAD, Cmpl2P2, p2_vec, dp2_vec);
        test_evaluates(DIV, Cmpl2P2, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, Cmpl2P2, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, Cmpl3P2, p2_vec, p2_vec);
        test_evaluates(GRAD, Cmpl3P2, p2_vec, dp2_vec);
        test_evaluates(DIV, Cmpl3P2, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, Cmpl3P2, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, Cmpl3, p2_vec, p2_vec);
        test_evaluates(GRAD, Cmpl3, p2_vec, dp2_vec);
        test_evaluates(DIV, Cmpl3, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, Cmpl3, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, Cmpl4, p2_vec, p2_vec);
        test_evaluates(GRAD, Cmpl4, p2_vec, dp2_vec);
        test_evaluates(DIV, Cmpl4, p2_vec, make_div(dp2_vec));
        test_evaluates(CURL, Cmpl4, p2_vec, make_curl(dp2_vec));
        test_evaluates(IDEN, ND0, f_div, f_div);
        test_evaluates(GRAD, ND0, f_div, df_div);
        test_evaluates(DIV, ND0, f_div, make_div(df_div));
        test_evaluates(CURL, ND0, f_div, make_curl(df_div));
        test_evaluates(IDEN, RT0, f_rot, f_rot);
        test_evaluates(GRAD, RT0, f_rot, df_rot);
        test_evaluates(DIV, RT0, f_rot, make_div(df_rot));
        test_evaluates(CURL, RT0, f_rot, make_curl(df_rot));

        {
            auto eval_with_coef = [&](OperatorType op_type, const FemSpace& s, const std::vector<double>& coef,
                                      const std::array<double, 3>& pt){
                return evaluate(op_type, s, ArrayView<>(const_cast<double*>(coef.data()), coef.size()), pt);
            };

            auto test_scalar_complement = [&](const FemSpace& V10, const FemSpace& V1, int preferred_row = 0){
                auto* comp = V10.target<ComplementFemSpace>();
                ASSERT_NE(comp, nullptr) << V10.typeName();
                EXPECT_EQ(comp->m_orth, ComplementFemSpace::L2Complement);
                EXPECT_EQ(V10.gatherType(), BaseFemSpace::BaseTypes::ComplementType);
                EXPECT_EQ(V10.dim(), 1u);
                const int n1 = static_cast<int>(V1.dofMap().NumDofOnTet());
                const int n0 = static_cast<int>(comp->m_V0->m_order.NumDofOnTet());
                const int n10 = static_cast<int>(V10.dofMap().NumDofOnTet());
                EXPECT_EQ(n10, n1 - n0);
                ASSERT_GT(n10, 0);
                const int row = std::min(preferred_row, n10 - 1);
                std::vector<double> unit(n10, 0.0);
                unit[row] = 1.0;

                auto f_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 1);
                    res[0] = eval_with_coef(IDEN, V10, unit, pt)[0];
                    return 0;
                };
                auto df_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 3);
                    auto g = eval_with_coef(GRAD, V10, unit, pt);
                    std::copy(g.begin(), g.end(), res);
                    return 0;
                };
                {
                    auto udofs_interp = interpolate(V10, f_basis);
                    DenseMatrix<const double> U(udofs_interp.data, n10, 1), Ue(unit.data(), n10, 1);
                    EXPECT_NEAR(Ue.ScalNorm(1, Ue, -1, U), 0, 100 * (1 + Ue.ScalNorm(1, Ue)) * DBL_EPSILON)
                        << V10.typeName();
                }
                test_evaluates(IDEN, V10, f_basis, f_basis);
                test_evaluates(GRAD, V10, f_basis, df_basis);

                for (int k = 0; k < n10; ++k){
                    double s = 0;
                    for (int j = 0; j < n1; ++j)
                        s += comp->m_basis_coefs[k * n1 + j] * comp->m_dual_coefs[k * n1 + j];
                    EXPECT_NEAR(s, 1.0, 1e-8) << V10.typeName() << " bio " << k;
                }
            };

            auto test_scalar_energy_complement = [&](const FemSpace& VE, const FemSpace& V1, int preferred_row = 0){
                auto* ec = VE.target<ComplementFemSpace>();
                ASSERT_NE(ec, nullptr) << VE.typeName();
                EXPECT_EQ(ec->m_orth, ComplementFemSpace::EnergyComplement);
                EXPECT_EQ(VE.gatherType(), BaseFemSpace::BaseTypes::ComplementType);
                EXPECT_EQ(VE.dim(), 1u);
                const int n1 = static_cast<int>(V1.dofMap().NumDofOnTet());
                const int n0 = static_cast<int>(ec->m_V0->m_order.NumDofOnTet());
                const int n10 = static_cast<int>(VE.dofMap().NumDofOnTet());
                EXPECT_EQ(n10, n1 - n0);
                ASSERT_GT(n10, 0);
                const int row = std::min(preferred_row, n10 - 1);
                std::vector<double> unit(n10, 0.0);
                unit[row] = 1.0;

                auto f_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 1);
                    res[0] = eval_with_coef(IDEN, VE, unit, pt)[0];
                    return 0;
                };
                auto df_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 3);
                    auto g = eval_with_coef(GRAD, VE, unit, pt);
                    std::copy(g.begin(), g.end(), res);
                    return 0;
                };
                {
                    auto udofs_interp = interpolate(VE, f_basis);
                    DenseMatrix<const double> U(udofs_interp.data, n10, 1), Ue(unit.data(), n10, 1);
                    EXPECT_NEAR(Ue.ScalNorm(1, Ue, -1, U), 0, 100 * (1 + Ue.ScalNorm(1, Ue)) * DBL_EPSILON)
                        << VE.typeName();
                }
                test_evaluates(IDEN, VE, f_basis, f_basis);
                test_evaluates(GRAD, VE, f_basis, df_basis);

                for (int k = 0; k < n10; ++k){
                    double s = 0;
                    for (int j = 0; j < n1; ++j)
                        s += ec->m_basis_coefs[k * n1 + j] * ec->m_dual_coefs[k * n1 + j];
                    EXPECT_NEAR(s, 1.0, 1e-8) << VE.typeName() << " bio " << k;
                }
            };

            {
                auto* comp = P2P1.target<ComplementFemSpace>();
                ASSERT_NE(comp, nullptr);
                for (auto it = P2P1.dofMap().begin(); it != P2P1.dofMap().end(); ++it)
                    EXPECT_TRUE(it->etype == DofT::EDGE_UNORIENT);
                const int row01 = complement_edge_row(*comp, 0, 1);
                ASSERT_GE(row01, 0);
                test_scalar_complement(P2P1, P2, row01);

                const int n10 = static_cast<int>(P2P1.dofMap().NumDofOnTet());
                std::vector<double> unit_vec(n10 * 3, 0.0);
                unit_vec[row01] = 1.0;
                unit_vec[n10 + row01] = 1.0;
                unit_vec[2 * n10 + row01] = 1.0;
                auto f_vec_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 3);
                    auto v = eval_with_coef(IDEN, P2P1_vec, unit_vec, pt);
                    std::copy(v.begin(), v.end(), res);
                    return 0;
                };
                auto df_vec_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 9);
                    auto g = eval_with_coef(GRAD, P2P1_vec, unit_vec, pt);
                    std::copy(g.begin(), g.end(), res);
                    return 0;
                };
                test_evaluates(IDEN, P2P1_vec, f_vec_basis, f_vec_basis);
                test_evaluates(GRAD, P2P1_vec, f_vec_basis, df_vec_basis);
                test_evaluates(IDEN, P2P1_vec_direct, f_vec_basis, f_vec_basis);
                test_evaluates(GRAD, P2P1_vec_direct, f_vec_basis, df_vec_basis);
            }

            {
                auto* ec = EP2P1.target<ComplementFemSpace>();
                ASSERT_NE(ec, nullptr);
                EXPECT_EQ(ec->m_orth, ComplementFemSpace::EnergyComplement);
                for (auto it = EP2P1.dofMap().begin(); it != EP2P1.dofMap().end(); ++it)
                    EXPECT_TRUE(it->etype == DofT::EDGE_UNORIENT);
                const int row01 = complement_edge_row(*ec, 0, 1);
                ASSERT_GE(row01, 0);
                test_scalar_energy_complement(EP2P1, P2, row01);
                test_scalar_energy_complement(EP3P1, P3, 0);

                const int n10 = static_cast<int>(EP2P1.dofMap().NumDofOnTet());
                std::vector<double> unit_vec(n10 * 3, 0.0);
                unit_vec[row01] = 1.0;
                unit_vec[n10 + row01] = 1.0;
                unit_vec[2 * n10 + row01] = 1.0;
                auto f_vec_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 3);
                    auto v = eval_with_coef(IDEN, EP2P1_vec, unit_vec, pt);
                    std::copy(v.begin(), v.end(), res);
                    return 0;
                };
                auto df_vec_basis = [&](const std::array<double, 3>& pt, double* res, ulong dim, void* user_data)->int{
                    (void) user_data; assert(dim == 9);
                    auto g = eval_with_coef(GRAD, EP2P1_vec, unit_vec, pt);
                    std::copy(g.begin(), g.end(), res);
                    return 0;
                };
                test_evaluates(IDEN, EP2P1_vec, f_vec_basis, f_vec_basis);
                test_evaluates(GRAD, EP2P1_vec, f_vec_basis, df_vec_basis);
                test_evaluates(IDEN, EP2P1_vec_direct, f_vec_basis, f_vec_basis);
                test_evaluates(GRAD, EP2P1_vec_direct, f_vec_basis, df_vec_basis);
            }

            test_scalar_complement(P3P2, P3, 7);
            test_scalar_complement(P3P1, P3, 13);
            test_scalar_complement(MINI1P1, MINI1, 0);

            // P1 + (P2 - P1) is L2-equivalent to P2 and must interpolate quadratics exactly.
            test_evaluates(IDEN, P1_plus_P2P1, fp2, fp2);
            test_evaluates(GRAD, P1_plus_P2P1, fp2, dfp2);
            test_evaluates(IDEN, P1_plus_P2P1, fp2_1, fp2_1);
            test_evaluates(GRAD, P1_plus_P2P1, fp2_1, dfp2_1);
        }

        auto compl_vec = mix_f({{fp0, 1}, {fp1, 1}, {fp2_1, 1}, {fp2, 1}, {fp1, 1}, {fp0, 1}, {f_rot, 3}});
        auto dcompl_vec = mix_df({{dfp0, 3}, {dfp1, 3}, {dfp2_1, 3}, {dfp2, 3}, {dfp1, 3}, {dfp0, 3}, {df_rot, 9}});
        test_evaluates(IDEN, P1*(P2^3)*(MINI1^2)*RT0, compl_vec, compl_vec);
        test_evaluates(GRAD, P1*(P2^3)*(MINI1^2)*RT0, compl_vec, dcompl_vec);
        evaluateT::test<IDEN, FemCom<FemFix<FEM_P1>, FemVecT<3, FemFix<FEM_P2>>, FemVecT<2, Mini1T>, FemFix<FEM_RT0>> >(XYZ, x, compl_vec, compl_vec);
        evaluateT::test<GRAD, FemCom<FemFix<FEM_P1>, FemVecT<3, FemFix<FEM_P2>>, FemVecT<2, Mini1T>, FemFix<FEM_RT0>> >(XYZ, x, compl_vec, dcompl_vec);
        auto compl_vec1 = mix_f({{fp0, 1}, {fp1, 1}, {fp2_1, 1}, {fp2, 1}, {f_rot, 3}});
        auto dcompl_vec1 = mix_df({{dfp0, 3}, {dfp1, 3}, {dfp2_1, 3}, {dfp2, 3}, {df_rot, 9}});
        evaluateT::test<IDEN, FemCom<FemFix<FEM_P1>, FemVecT<3, FemFix<FEM_P2>>, FemFix<FEM_RT0>> >(XYZ, x, compl_vec1, compl_vec1);
        evaluateT::test<GRAD, FemCom<FemFix<FEM_P1>, FemVecT<3, FemFix<FEM_P2>>, FemFix<FEM_RT0>> >(XYZ, x, compl_vec1, dcompl_vec1);

        //test for inverted tetra
        std::swap(XYZ.XY0, XYZ.XY1);
        test_evaluates(IDEN, RT0, f_rot, f_rot);
        test_evaluates(GRAD, RT0, f_rot, df_rot);
        test_evaluates(DIV, RT0, f_rot, make_div(df_rot));
        test_evaluates(CURL, RT0, f_rot, make_curl(df_rot));
        std::swap(XYZ.XY0, XYZ.XY1);
    }
    {
        auto dfunc = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
            (void) user_data; (void) iTet;
            DenseMatrix<> D(Dmem, Ddims.first, Ddims.second);
            int cnt = 0;
            for (ulong i = 0; i < Ddims.first; ++i)
                for (ulong j = 0; j < Ddims.second; ++j)
                    D(i, j) = (i * x[0] + j * x[1] + (i%2)*x[2]) * (cnt++);
            return TENSOR_GENERAL;
        };
        double coef = 9*80;
        std::array<std::array<double, 20>, 12> At_exp_m{std::array<double, 20>
                {   83,   222,   -85,  -220, 0,    774, 0,   -54, 0,  -324,   315,   594,    -9,   -27,  -1080,   -891,   3069,  -1143, -2565,   1341, },
                {  462,   612,  -384,  -690, 0,   3186, 0,   -54, 0,  -432,   432,  1188,  -378,  -378,  -3294,  -3186,   9882,  -6858, -6318,   6210, },
                { -154,  -300,   146,   308, 0,  -1197, 0,    54, 0,   378,  -360,  -648,    45,    81,   1458,   1431,  -4275,   2151,  3267,  -2385, },
                { -391,  -534,   323,   602, 0,  -2763, 0,    54, 0,   378,  -387, -1134,   342,   324,   2916,   2646,  -8676,   5850,  5616,  -5166, },
                {  953,   834,  -673, -1114, 0,   5364, 0,   270, 0,     0,    45,  1404, -1089, -1161,  -5130,  -5589,  16245, -14211, -7749,  11601, },
                { 1686,   612,  -996, -1302, 0,   7398, 0,  1080, 0,  1836, -1674,    54, -2484, -2646,  -5562,  -7722,  19764, -25488, -2916,  18360, },
                { -703,  -534,   479,   758, 0,  -3951, 0,  -216, 0,  -162,   207,  -864,   936,   864,   3456,   3726, -11214,  10656,  4806,  -8244, },
                {-1936,  -912,  1190,  1658, 0,  -8811, 0, -1134, 0, -1674,  1422,  -594,  2637,  2943,   7236,   9585, -24795,  29043,  5859, -21717, },
                { 2951,  1446, -1825, -2572, 0,  13950, 0,  1620, 0,  2376, -2223,  1188, -4167, -4347, -11232, -14391,  38547, -44505, -9855,  33039, },
                { 2622,   612, -1464, -1770, 0,  10962, 0,  1890, 0,  3456, -3456,  -756, -4266, -4266,  -7182, -10962,  27378, -39906,  -486,  27594, },
                {-2344,  -912,  1394,  1862, 0, -10215, 0, -1512, 0, -2430,  2124,  -216,  3339,  3699,   7992,  11097, -28089,  35253,  4725, -25767, },
                {-3229, -1146,  1895,  2480, 0, -14697, 0, -1998, 0, -3402,  3555,  -216,  5094,  4914,  10422,  14256, -37836,  49158,  5616, -34866, },
        };
        std::array<double, 12*20> A_exp_mp;
        DenseMatrix<> A_exp_m(A_exp_mp.data(), 12, 20);
        for (std::size_t i = 0; i < A_exp_m.nRow; ++i) 
            for (std::size_t j = 0; j < A_exp_m.nCol; ++j) 
                A_exp_m(i, j) = At_exp_m[i][j];
        auto grad_p3 = P3.getOP(GRAD);
        auto grad_3p1 = (P1^3).getOP_own(GRAD);
        mem = fem3Dtet_memory_requirements<>(grad_p3, grad_3p1, 5, 1);
        wmem.resize(mem.enoughRawSize());
        mem.allocateFromRaw(wmem.data(), wmem.size());
        std::array<double, 20*12> Ap = {0};
        DenseMatrix<> A(Ap.data(), 12, 20);
        fem3Dtet<>(XYZ, grad_p3, grad_3p1, dfunc, A, mem, 5);
        EXPECT_NEAR(DenseMatrix<>::ScalNorm(1.0/coef, A_exp_m, -1, A), 0, 10*(1 + A_exp_m.ScalNorm(1/coef, A_exp_m))*DBL_EPSILON);
    }
}


namespace {

using namespace Ani;

struct QuadSetup {
    uint nquad = 0;
    std::vector<double> xyl;
    std::vector<double> wg;
};

QuadSetup make_quad(uint quad_order){
    QuadSetup q;
    auto formula = tetrahedron_quadrature_formulas(static_cast<int>(quad_order));
    q.nquad = static_cast<uint>(formula.GetNumPoints());
    q.xyl.resize(4 * q.nquad);
    q.wg.resize(q.nquad);
    std::copy(formula.GetPointData(), formula.GetPointData() + 4 * q.nquad, q.xyl.data());
    std::copy(formula.GetWeightData(), formula.GetWeightData() + q.nquad, q.wg.data());
    return q;
}

Tetra<const double> canonical_tetra(){
    static double XYZa[12]{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    return Tetra<const double>(XYZa + 0, XYZa + 3, XYZa + 6, XYZa + 9);
}

struct DualAnimemHolder {
    std::vector<char> buf;
    PlainMemoryX<> pmx_v1, pmx_v0;
    AniMemoryX<> mem_v1, mem_v0;
};

DualAnimemHolder setup_dual_animem(const BaseFemSpace& V1, const BaseFemSpace& V0, uint nquad,
                                   const Tetra<const double>& XYZ){
    DualAnimemHolder h;
    auto req1 = V1.memOP(IDEN, nquad, 1);
    auto req0 = V0.memOP(IDEN, nquad, 1);
    h.pmx_v1.dSize = req1.Usz + req1.extraRsz;
    h.pmx_v1.iSize = req1.extraIsz + 2 * (req1.mtx_parts + 2);
    h.pmx_v1.mSize = req1.mtx_parts;
    h.pmx_v0.dSize = req0.Usz + req0.extraRsz;
    h.pmx_v0.iSize = req0.extraIsz + 2 * (req0.mtx_parts + 2);
    h.pmx_v0.mSize = req0.mtx_parts;
    const auto sz1 = h.pmx_v1.enoughRawSize();
    const auto sz0 = h.pmx_v0.enoughRawSize();
    h.buf.resize(sz1 + sz0);
    h.pmx_v1.allocateFromRaw(h.buf.data(), sz1);
    h.pmx_v0.allocateFromRaw(h.buf.data() + sz1, sz0);

    auto alloc = [](AniMemoryX<>& mem, PlainMemoryX<>& pmx, const BaseFemSpace& space, uint nq){
        auto req = space.memOP(IDEN, nq, 1);
        mem.q = nq;
        mem.f = 1;
        mem.busy_mtx_parts = 0;
        mem.U.Init(pmx.ddata, req.Usz);
        pmx.ddata += req.Usz;
        mem.extraR.Init(pmx.ddata, req.extraRsz);
        pmx.ddata += req.extraRsz;
        mem.extraI.Init(pmx.idata, req.extraIsz);
        pmx.idata += req.extraIsz;
        mem.MTXI_COL.Init(pmx.idata, req.mtx_parts + 2);
        pmx.idata += req.mtx_parts + 2;
        mem.MTXI_ROW.Init(pmx.idata, req.mtx_parts + 2);
        pmx.idata += req.mtx_parts + 2;
        mem.MTX.Init(pmx.mdata, req.mtx_parts);
    };
    alloc(h.mem_v1, h.pmx_v1, V1, nquad);
    alloc(h.mem_v0, h.pmx_v0, V0, nquad);

    double one = 1;
    double* XYZa = const_cast<double*>(XYZ.XY0.data);
    h.mem_v1.XYP.Init(XYZa, 12);
    h.mem_v1.PSI.Init(XYZa + 3, 9);
    h.mem_v1.DET.Init(&one, 1);
    h.mem_v0.XYP = h.mem_v1.XYP;
    h.mem_v0.PSI = h.mem_v1.PSI;
    h.mem_v0.DET = h.mem_v1.DET;
    return h;
}

struct BasisFunctorData {
    const BaseFemSpace* space;
    AniMemoryX<>* mem;
    int idof;
    int nfa;
};

int eval_basis_functor(const std::array<double, 3>& X, double* res, ulong dim, void* user_data){
    auto& d = *static_cast<BasisFunctorData*>(user_data);
    d.mem->XYG.Init(const_cast<double*>(X.data()), 3);
    std::array<double, 4> xyl;
    xyl[0] = 1 - (X[0] + X[1] + X[2]);
    for (int k = 0; k < 3; ++k)
        xyl[1 + k] = X[k];
    d.mem->XYL.Init(xyl.data(), 4);
    d.mem->q = 1;
    d.mem->busy_mtx_parts = 0;
    auto r = d.space->applyOP(IDEN, *d.mem, d.mem->U);
    std::fill(res, res + dim, 0.0);
    for (std::size_t p = 0; p < r.nparts; ++p)
        for (int row = r.stRow[p]; row < r.stRow[p + 1]; ++row)
            for (int col = r.stCol[p]; col < r.stCol[p + 1]; ++col)
                if (col == d.idof && row >= 0 && row < static_cast<int>(dim))
                    res[row] = r.data[p](row - r.stRow[p], col - r.stCol[p]);
    return 0;
}

void build_embedding_matrix(const BaseFemSpace& V0, const BaseFemSpace& V1, const Tetra<const double>& XYZ,
                            AniMemoryX<>& mem_v0, uint quad_order, double* C, int n0, int n1){
    PlainMemoryX<> pmx;
    for (int j = 0; j < n1; ++j){
        auto lpmx = V1.interpolateOnDOF_mem_req(j, 1, quad_order);
        pmx.dSize = std::max(pmx.dSize, lpmx.dSize);
        pmx.iSize = std::max(pmx.iSize, lpmx.iSize);
        pmx.mSize = std::max(pmx.mSize, lpmx.mSize);
    }
    std::vector<char> wmem(pmx.enoughRawSize());
    pmx.allocateFromRaw(wmem.data(), wmem.size());
    double* d0 = pmx.ddata;
    int* i0 = pmx.idata;
    DenseMatrix<>* m0 = pmx.mdata;
    const std::size_t ds = pmx.dSize, is = pmx.iSize, ms = pmx.mSize;
    BasisFunctorData fd{&V0, &mem_v0, 0, n0};
    std::vector<double> val(n1, 0.0);
    for (int i = 0; i < n0; ++i){
        fd.idof = i;
        for (int j = 0; j < n1; ++j){
            std::fill(val.begin(), val.end(), 0.0);
            pmx.ddata = d0; pmx.idata = i0; pmx.mdata = m0;
            pmx.dSize = ds; pmx.iSize = is; pmx.mSize = ms;
            V1.interpolateOnDOF(XYZ, eval_basis_functor, ArrayView<>(val.data(), n1), j, pmx, &fd, quad_order);
            C[i + j * n0] = val[j];
        }
    }
}

void build_mass_matrix(const BaseFemSpace& V1, AniMemoryX<>& mem, const QuadSetup& quad, double* M1, int n1){
    const uint gdim = V1.dim();
    std::vector<double> phi(static_cast<std::size_t>(n1) * quad.nquad * gdim, 0.0);
    for (uint q = 0; q < quad.nquad; ++q){
        mem.XYL.Init(const_cast<double*>(quad.xyl.data()) + 4 * q, 4);
        mem.WG.Init(const_cast<double*>(quad.wg.data()) + q, 1);
        mem.q = 1;
        mem.busy_mtx_parts = 0;
        auto res = V1.applyOP(IDEN, mem, mem.U);
        for (std::size_t p = 0; p < res.nparts; ++p)
            for (int d = res.stRow[p]; d < res.stRow[p + 1]; ++d)
                for (int i = res.stCol[p]; i < res.stCol[p + 1]; ++i)
                    if (d >= 0 && d < static_cast<int>(gdim) && i >= 0 && i < n1)
                        phi[q + quad.nquad * i + n1 * quad.nquad * d] =
                            res.data[p](d - res.stRow[p], i - res.stCol[p]);
    }
    for (int i = 0; i < n1; ++i)
        for (int j = i; j < n1; ++j){
            double s = 0;
            for (uint q = 0; q < quad.nquad; ++q)
                for (uint d = 0; d < gdim; ++d)
                    s += quad.wg[q] * phi[q + quad.nquad * i + n1 * quad.nquad * d]
                                    * phi[q + quad.nquad * j + n1 * quad.nquad * d];
            M1[i + j * n1] = M1[j + i * n1] = s;
        }
}

void matmul(const double* A, int n, int m, const double* B, int p, double* C){
    std::fill(C, C + n * p, 0.0);
    for (int j = 0; j < p; ++j)
        for (int k = 0; k < m; ++k)
            for (int i = 0; i < n; ++i)
                C[i + j * n] += A[i + k * n] * B[k + j * m];
}

struct ComplementMatrices {
    int n0 = 0, n1 = 0, n10 = 0;
    std::vector<double> M1, C, S;
};

ComplementMatrices build_complement_matrices(const FemSpace& V1, const FemSpace& V0, uint quad_order){
    ComplementMatrices out;
    const BaseFemSpace& b1 = *V1.base();
    const BaseFemSpace& b0 = *V0.base();
    out.n1 = static_cast<int>(V1.dofMap().NumDofOnTet());
    out.n0 = static_cast<int>(V0.dofMap().NumDofOnTet());
    out.n10 = out.n1 - out.n0;
    out.M1.assign(out.n1 * out.n1, 0.0);
    out.C.assign(out.n0 * out.n1, 0.0);
    out.S.assign(out.n0 * out.n1, 0.0);

    auto quad = make_quad(quad_order);
    auto XYZ = canonical_tetra();
    auto holder = setup_dual_animem(b1, b0, quad.nquad, XYZ);
    build_mass_matrix(b1, holder.mem_v1, quad, out.M1.data(), out.n1);
    build_embedding_matrix(b0, b1, XYZ, holder.mem_v0, quad_order, out.C.data(), out.n0, out.n1);
    matmul(out.C.data(), out.n0, out.n1, out.M1.data(), out.n1, out.S.data());
    return out;
}

double frob_norm(const double* A, int n, int m){
    double ss = 0;
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i)
            ss += A[i + j * n] * A[i + j * n];
    return std::sqrt(ss);
}

void expect_S_annihilates(const ComplementMatrices& mats, const double* psi, double tol){
    std::vector<double> r(mats.n0, 0.0);
    for (int i = 0; i < mats.n0; ++i)
        for (int j = 0; j < mats.n1; ++j)
            r[i] += mats.S[i + j * mats.n0] * psi[j];
    EXPECT_NEAR(frob_norm(r.data(), mats.n0, 1), 0.0, tol);
}

void expect_discrete_m1_orthogonality_v0(const ComplementMatrices& mats, const double* coef, double tol){
    std::vector<double> mpsi(mats.n1, 0.0);
    for (int j = 0; j < mats.n1; ++j)
        for (int l = 0; l < mats.n1; ++l)
            mpsi[j] += mats.M1[j + l * mats.n1] * coef[l];
    for (int i = 0; i < mats.n0; ++i){
        double s = 0;
        for (int j = 0; j < mats.n1; ++j)
            s += mats.C[i + j * mats.n0] * mpsi[j];
        EXPECT_NEAR(s, 0.0, tol) << "M1-orthogonality to V0 dof " << i;
    }
}

double m1_inner(const double* a, const double* b, const double* M1, int n){
    double s = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            s += a[i] * M1[i + j * n] * b[j];
    return s;
}

double m1_norm(const double* a, const double* M1, int n){
    return std::sqrt(std::max(0.0, m1_inner(a, a, M1, n)));
}

double m1_parallel_cosine(const double* a, const double* b, const double* M1, int n){
    const double na = m1_norm(a, M1, n), nb = m1_norm(b, M1, n);
    if (na < 1e-15 || nb < 1e-15)
        return 0;
    return m1_inner(a, b, M1, n) / (na * nb);
}

int edge_vertices_from_tag(const DofT::LocalOrder& lo, int& v0, int& v1){
    if (lo.etype != DofT::EDGE_UNORIENT && lo.etype != DofT::EDGE_ORIENT)
        return 0;
    static const std::array<char, 12> lookup = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
    const int e = lo.nelem;
    if (e < 0 || e >= 6)
        return 0;
    v0 = lookup[2 * e];
    v1 = lookup[2 * e + 1];
    return 1;
}

int p2_edge_dof(int i, int j){
    static const std::array<unsigned char, 6> I = {0, 0, 0, 1, 1, 2}, J = {1, 2, 3, 2, 3, 3};
    const int a = std::min(i, j), b = std::max(i, j);
    for (int e = 0; e < 6; ++e)
        if (I[e] == a && J[e] == b)
            return 4 + e;
    return -1;
}

/// Frame equivariance: ‖Ψ P₁(σ) − P₁₀(σ) Ψ‖_F / ‖Ψ‖ < tol for all 24 σ ∈ S₄.
void expect_frame_equivariance(const FemSpace& V1, const FemSpace& V10, double tol = 1e-8){
    if (V10.gatherType() != BaseFemSpace::BaseTypes::ComplementType)
        return;
    auto* comp = V10.target<ComplementFemSpace>();
    ASSERT_NE(comp, nullptr);
    const int n1 = static_cast<int>(V1.dofMap().NumDofOnTet());
    const int n10 = static_cast<int>(V10.dofMap().NumDofOnTet());
    const double* Psi = comp->m_basis_coefs.data();
    double nPsi = 0;
    for (int i = 0; i < n10 * n1; ++i)
        nPsi += Psi[i] * Psi[i];
    nPsi = std::sqrt(nPsi);
    ASSERT_GT(nPsi, 0.0);

    std::array<std::array<DofT::uchar, 4>, 24> perms;
    DofT::S4::all_permutations(perms);
    std::vector<double> P1(static_cast<std::size_t>(n1) * n1), P10(static_cast<std::size_t>(n10) * n10);
    for (int p = 0; p < 24; ++p){
        DofT::S4::build_dof_permutation(*V1.dofMap().base(), perms[p].data(), P1.data(), n1);
        DofT::S4::build_dof_permutation(*V10.dofMap().base(), perms[p].data(), P10.data(), n10);
        double d2 = 0;
        for (int i = 0; i < n10; ++i)
            for (int j = 0; j < n1; ++j){
                double a = 0, b = 0;
                for (int k = 0; k < n1; ++k)
                    a += Psi[i * n1 + k] * P1[k + j * n1];
                for (int k = 0; k < n10; ++k)
                    b += P10[i + k * n10] * Psi[k * n1 + j];
                d2 += (a - b) * (a - b);
            }
        EXPECT_LT(std::sqrt(d2) / nPsi, tol) << "frame equivariance failed for S4 perm " << p;
    }
}

/// Checks that hold for any ComplementFemSpace V10 = V1 - V0.
void expect_complement_space(const FemSpace& V1, const FemSpace& V0, const FemSpace& V10){
    const uint n1 = V1.dofMap().NumDofOnTet();
    const uint n0 = V0.dofMap().NumDofOnTet();
    ASSERT_GE(n1, n0);
    EXPECT_EQ(V10.dofMap().NumDofOnTet(), n1 - n0);
    EXPECT_EQ(V10.dim(), V1.dim());
    EXPECT_EQ(V1.dim(), V0.dim());

    if (V10.gatherType() != BaseFemSpace::BaseTypes::ComplementType)
        return; // factorized Vector/Complex of complements

    auto* comp = V10.target<ComplementFemSpace>();
    ASSERT_NE(comp, nullptr);
    const int n10 = static_cast<int>(n1 - n0);
    ASSERT_EQ(static_cast<int>(comp->m_basis_coefs.size()), n10 * static_cast<int>(n1));
    ASSERT_EQ(static_cast<int>(comp->m_dual_coefs.size()), n10 * static_cast<int>(n1));
    ASSERT_EQ(static_cast<int>(comp->m_dof_tags.size()), n10);

    const uint quad_order = 2 * V1.order() + 1;
    auto mats = build_complement_matrices(V1, V0, quad_order);
    ASSERT_EQ(mats.n10, n10);

    for (int k = 0; k < n10; ++k){
        const double* row = comp->m_basis_coefs.data() + k * mats.n1;
        expect_S_annihilates(mats, row, 1e-8);
        expect_discrete_m1_orthogonality_v0(mats, row, 1e-8);
    }

    // Biorthogonality Ψ · Bᵀ ≈ I
    for (int k = 0; k < n10; ++k){
        double s = 0;
        for (int j = 0; j < mats.n1; ++j)
            s += comp->m_basis_coefs[k * mats.n1 + j] * comp->m_dual_coefs[k * mats.n1 + j];
        EXPECT_NEAR(s, 1.0, 1e-8) << "biorthogonality diagonal " << k;
    }

    expect_frame_equivariance(V1, V10);
}


} // namespace


TEST(AniInterface, ComplementSpaces){
    FemSpace P1{P1Space{}};
    FemSpace P2{P2Space{}};
    FemSpace P3{P3Space{}};
    FemSpace MINI1 = P1 + FemSpace{BubbleSpace{}};

    // --- scalar complements ---
    {
        FemSpace V10 = P2 - P1;
        expect_complement_space(P2, P1, V10);
        auto* comp = V10.target<ComplementFemSpace>();
        ASSERT_NE(comp, nullptr);
        std::vector<bool> matched(6, false);
        int n_edge = 0;
        for (const auto& tag : comp->m_dof_tags){
            EXPECT_EQ(tag.etype, DofT::EDGE_UNORIENT);
            EXPECT_EQ(static_cast<int>(tag.stype), 0);
            EXPECT_EQ(static_cast<int>(tag.lsid), 0);
            ++n_edge;
            int v0 = -1, v1 = -1;
            ASSERT_TRUE(edge_vertices_from_tag(tag, v0, v1));
            const int ed = p2_edge_dof(v0, v1);
            ASSERT_GE(ed, 4);
            matched[ed - 4] = true;
        }
        EXPECT_EQ(n_edge, 6);
        for (int e = 0; e < 6; ++e)
            EXPECT_TRUE(matched[e]) << "missing edge " << e;

        const auto* umap = dynamic_cast<const DofT::UniteDofMap*>(V10.dofMap().base().get());
        ASSERT_NE(umap, nullptr);
        EXPECT_EQ(umap->NumDofOnTet(), 6u);
        EXPECT_EQ(umap->NumDof(DofT::EDGE_UNORIENT), 1u);
        EXPECT_EQ(umap->SymComponents(DofT::EDGE_UNORIENT), 1u); // stype 0 (volume 1)
        EXPECT_EQ(umap->m_symmetries.get(DofT::EDGE_UNORIENT, 0), 1u);
        EXPECT_EQ(umap->m_symmetries.get(DofT::EDGE_UNORIENT, 1), 0u);
        EXPECT_EQ(umap->NumDof(DofT::EDGE_ORIENT), 0u);
        EXPECT_EQ(umap->NumDof(DofT::FACE_UNORIENT), 0u);
        EXPECT_EQ(umap->NumDof(DofT::FACE_ORIENT), 0u);
        EXPECT_EQ(umap->NumDof(DofT::NODE), 0u);
        EXPECT_EQ(umap->NumDof(DofT::CELL), 0u);

        // Analytical check for Complement(P2, P1) = L² orthogonal complement.
        //
        // Theory. Same P2 nodal basis / embedding C as in the energy case. Now
        //   K = { u ∈ R^{10} : C M1 u = 0 },   dim K = n1 − n0 = 6
        // (mean-zero is automatic: constants ⊂ P1). Stab(edge 01) ≅ Z2×Z2.
        // Stab-invariants in K form a 3-space; on the 5-parameter ansatz
        //   ψ=(α,α,β,β, γ, δ,δ,δ,δ, ε)
        // the integer basis is
        //   v1 = (0, 0, 2, 2,  1, 0,0,0,0, 0),
        //   v2 = (4, 4, 4, 4,  0, 1,1,1,1, 0),
        //   v3 = (2, 2, 0, 0,  0, 0,0,0,0, 1).
        // Canonical generator = M1-projection of the edge bubble e_4 onto that span:
        //   w_can = (−8/15,−8/15, 2/15,2/15,  7/15, −1/5,−1/5,−1/5,−1/5, 2/15)
        // with ‖w_can‖_{M1}^2 = 11/4725. The continuous L² frame is the S4-orbit of
        // any generic vector in this 3-space. Duals come from the algebraic split
        // R^{n1} = span(Ψ) ⊕ V0 via B = first n10 rows of [Ψ; C]^{-1} (as columns of
        // the inverse, stored row-major in m_dual_coefs).
        {
            using uchar = DofT::uchar;
            const int n1 = 10, n10 = 6, n0 = 4;
            static const int edge_ij[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
            auto apply_vertex_perm = [&](const uchar sigma[4], const double* x, double* y){
                std::fill(y, y + n1, 0.0);
                for (int j = 0; j < 4; ++j)
                    y[sigma[j]] = x[j];
                for (int k = 0; k < 6; ++k){
                    uchar a = sigma[edge_ij[k][0]], b = sigma[edge_ij[k][1]];
                    if (a > b) std::swap(a, b);
                    int k2 = -1;
                    for (int t = 0; t < 6; ++t)
                        if (edge_ij[t][0] == a && edge_ij[t][1] == b){ k2 = t; break; }
                    ASSERT_GE(k2, 0);
                    y[4 + k2] = x[4 + k];
                }
            };
            auto edge_sigma = [&](int ea, int eb, uchar sigma[4]){
                int rem[2], nr = 0;
                for (int i = 0; i < 4; ++i)
                    if (i != ea && i != eb) rem[nr++] = i;
                sigma[0] = static_cast<uchar>(ea);
                sigma[1] = static_cast<uchar>(eb);
                sigma[2] = static_cast<uchar>(rem[0]);
                sigma[3] = static_cast<uchar>(rem[1]);
            };

            const double V01[3][10] = {
                {0, 0, 2, 2, 1, 0, 0, 0, 0, 0},
                {4, 4, 4, 4, 0, 1, 1, 1, 1, 0},
                {2, 2, 0, 0, 0, 0, 0, 0, 0, 1},
            };
            const double w_can01[10] = {
                -8.0/15.0, -8.0/15.0, 2.0/15.0, 2.0/15.0, 7.0/15.0,
                -1.0/5.0, -1.0/5.0, -1.0/5.0, -1.0/5.0, 2.0/15.0
            };

            auto mats = build_complement_matrices(P2, P1, 2 * P2.order() + 1);
            ASSERT_EQ(mats.n1, n1);
            ASSERT_EQ(mats.n0, n0);
            const double* M1 = mats.M1.data();
            const double* C = mats.C.data();

            for (int e = 0; e < n10; ++e){
                ASSERT_EQ(static_cast<int>(comp->m_dof_tags[e].nelem), e);
                uchar sigma[4];
                edge_sigma(edge_ij[e][0], edge_ij[e][1], sigma);
                double Bspan[3][10];
                for (int c = 0; c < 3; ++c)
                    apply_vertex_perm(sigma, V01[c], Bspan[c]);
                const double* psi = comp->m_basis_coefs.data() + e * n1;
                double G[9] = {0}, rhs[3] = {0};
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i){
                        double s = 0;
                        for (int k = 0; k < n1; ++k)
                            s += Bspan[i][k] * Bspan[j][k];
                        G[i + j * 3] = s;
                    }
                for (int i = 0; i < 3; ++i){
                    double s = 0;
                    for (int k = 0; k < n1; ++k)
                        s += Bspan[i][k] * psi[k];
                    rhs[i] = s;
                }
                double chol_mem[6], coef[3];
                cholesky_solve(G, rhs, 3, 1, coef, chol_mem);
                double err2 = 0, nrm2 = 0;
                for (int k = 0; k < n1; ++k){
                    double r = psi[k];
                    for (int c = 0; c < 3; ++c)
                        r -= coef[c] * Bspan[c][k];
                    err2 += r * r;
                    nrm2 += psi[k] * psi[k];
                }
                EXPECT_LT(std::sqrt(err2) / std::max(1e-30, std::sqrt(nrm2)), 1e-12)
                    << "P2-P1 mode " << e << " not in analytical Stab-span";
                double m2 = 0;
                for (int i = 0; i < n1; ++i)
                    for (int j = 0; j < n1; ++j)
                        m2 += psi[i] * M1[i + j * n1] * psi[j];
                EXPECT_NEAR(m2, 1.0, 1e-10) << "P2-P1 mode " << e << " M1-norm";
            }

            {
                double Orbit[6][10];
                for (int e = 0; e < 6; ++e){
                    uchar sigma[4];
                    edge_sigma(edge_ij[e][0], edge_ij[e][1], sigma);
                    apply_vertex_perm(sigma, w_can01, Orbit[e]);
                }
                for (int e = 0; e < n10; ++e){
                    const double* psi = comp->m_basis_coefs.data() + e * n1;
                    double Gram[36] = {0}, rhs[6] = {0};
                    for (int j = 0; j < 6; ++j)
                        for (int i = 0; i < 6; ++i){
                            double s = 0;
                            for (int a = 0; a < n1; ++a)
                                for (int b = 0; b < n1; ++b)
                                    s += Orbit[i][a] * M1[a + b * n1] * Orbit[j][b];
                            Gram[i + j * 6] = s;
                        }
                    for (int i = 0; i < 6; ++i){
                        double s = 0;
                        for (int a = 0; a < n1; ++a)
                            for (int b = 0; b < n1; ++b)
                                s += Orbit[i][a] * M1[a + b * n1] * psi[b];
                        rhs[i] = s;
                    }
                    double chol_mem[21], coef[6];
                    cholesky_solve(Gram, rhs, 6, 1, coef, chol_mem);
                    std::vector<double> recon(n1, 0.0), diff(n1, 0.0);
                    for (int a = 0; a < n1; ++a){
                        for (int i = 0; i < 6; ++i)
                            recon[a] += coef[i] * Orbit[i][a];
                        diff[a] = psi[a] - recon[a];
                    }
                    double d2 = 0, n2 = 0;
                    for (int a = 0; a < n1; ++a)
                        for (int b = 0; b < n1; ++b){
                            d2 += diff[a] * M1[a + b * n1] * diff[b];
                            n2 += psi[a] * M1[a + b * n1] * psi[b];
                        }
                    EXPECT_LT(std::sqrt(std::max(0.0, d2)) / std::max(1e-30, std::sqrt(n2)), 1e-10)
                        << "P2-P1 mode " << e << " not in analytical w_can orbit";
                }
            }

            // Duals from [Ψ; C]^{-1}: B[k,j] = BigInv[j,k] for k < n10.
            {
                std::vector<double> Big(n1 * n1, 0.0), BigInv(n1 * n1, 0.0);
                std::vector<double> lu_mem(2 * n1 * n1), imem(2 * n1);
                for (int k = 0; k < n10; ++k)
                    for (int j = 0; j < n1; ++j)
                        Big[k + j * n1] = comp->m_basis_coefs[k * n1 + j];
                for (int k = 0; k < n0; ++k)
                    for (int j = 0; j < n1; ++j)
                        Big[n10 + k + j * n1] = C[k + j * n0];
                fullPivLU_inverse(Big.data(), BigInv.data(), n1, lu_mem.data(),
                    reinterpret_cast<int*>(imem.data()));
                double err2 = 0, nrm2 = 0;
                for (int k = 0; k < n10; ++k)
                    for (int j = 0; j < n1; ++j){
                        const double pred = BigInv[j + k * n1];
                        const double d = pred - comp->m_dual_coefs[k * n1 + j];
                        err2 += d * d;
                        nrm2 += comp->m_dual_coefs[k * n1 + j] * comp->m_dual_coefs[k * n1 + j];
                    }
                EXPECT_LT(std::sqrt(err2) / std::max(1e-30, std::sqrt(nrm2)), 1e-10)
                    << "P2-P1 duals disagree with analytical [Psi;C]^{-1} formula";
            }
        }
    }
    {
        FemSpace V10 = MINI1 - P1;
        expect_complement_space(MINI1, P1, V10);
        auto* comp = V10.target<ComplementFemSpace>();
        ASSERT_NE(comp, nullptr);
        ASSERT_EQ(comp->m_dof_tags.size(), 1u);
        static const double psi_ref[5] = {-32.0 / 105.0, -32.0 / 105.0, -32.0 / 105.0, -32.0 / 105.0, 73.0 / 105.0};
        auto mats = build_complement_matrices(MINI1, P1, 2 * MINI1.order() + 1);
        const double cos = std::abs(m1_parallel_cosine(comp->m_basis_coefs.data(), psi_ref, mats.M1.data(), mats.n1));
        EXPECT_NEAR(cos, 1.0, 1e-8);
    }
    {
        FemSpace V10 = P3 - P2;
        expect_complement_space(P3, P2, V10);
        auto* comp = V10.target<ComplementFemSpace>();
        ASSERT_NE(comp, nullptr);
        int n_edge = 0, n_face = 0;
        for (const auto& tag : comp->m_dof_tags){
            if (tag.etype == DofT::EDGE_ORIENT){
                ++n_edge;
                // Sign isotype on edges → EDGE_ORIENT, stype 0 (volume 1).
                EXPECT_EQ(static_cast<int>(tag.stype), 0);
            } else if (tag.etype & DofT::FACE){
                ++n_face;
                EXPECT_EQ(static_cast<int>(tag.stype), 0);
            } else {
                FAIL() << "unexpected tag etype " << int(tag.etype);
            }
        }
        EXPECT_EQ(n_edge, 6);
        EXPECT_EQ(n_face, 4);

        const auto* umap = dynamic_cast<const DofT::UniteDofMap*>(V10.dofMap().base().get());
        ASSERT_NE(umap, nullptr);
        EXPECT_EQ(umap->NumDof(DofT::EDGE_ORIENT), 1u);
        EXPECT_EQ(umap->m_symmetries.get(DofT::EDGE_ORIENT, 0), 1u);
        EXPECT_EQ(umap->NumDof(DofT::EDGE_UNORIENT), 0u);
        EXPECT_EQ(umap->SymComponents(DofT::FACE_UNORIENT), 1u);
        EXPECT_EQ(umap->m_symmetries.get(DofT::FACE_UNORIENT, 0), 1u);
        EXPECT_EQ(umap->NumDof(DofT::FACE_UNORIENT), 1u);
        EXPECT_EQ(umap->NumDofOnTet(), 10u);
    }
    {
        FemSpace V10 = P3 - P1;
        expect_complement_space(P3, P1, V10);
        EXPECT_EQ(V10.dofMap().NumDofOnTet(), 16u);
        auto* comp = V10.target<ComplementFemSpace>();
        ASSERT_NE(comp, nullptr);
        int n_edge = 0, n_face = 0;
        for (const auto& tag : comp->m_dof_tags){
            if (tag.etype & DofT::EDGE){
                ++n_edge;
                EXPECT_EQ(static_cast<int>(tag.stype), 1);
            } else if (tag.etype & DofT::FACE){
                ++n_face;
                EXPECT_EQ(static_cast<int>(tag.stype), 0);
            } else {
                FAIL() << "unexpected tag etype";
            }
        }
        EXPECT_EQ(n_edge, 12);
        EXPECT_EQ(n_face, 4);

        const auto* umap = dynamic_cast<const DofT::UniteDofMap*>(V10.dofMap().base().get());
        ASSERT_NE(umap, nullptr);
        EXPECT_EQ(umap->SymComponents(DofT::EDGE_UNORIENT), 2u); // stype 1 (volume 2)
        EXPECT_EQ(umap->m_symmetries.get(DofT::EDGE_UNORIENT, 0), 0u);
        EXPECT_EQ(umap->m_symmetries.get(DofT::EDGE_UNORIENT, 1), 1u);
        EXPECT_EQ(umap->SymComponents(DofT::FACE_UNORIENT), 1u);
        EXPECT_EQ(umap->m_symmetries.get(DofT::FACE_UNORIENT, 0), 1u);
        EXPECT_EQ(umap->NumDof(DofT::EDGE_UNORIENT), 2u);
        EXPECT_EQ(umap->NumDof(DofT::FACE_UNORIENT), 1u);
    }

    // --- vector / complex factorization ---
    {
        FemSpace factored = (P2 - P1) ^ 3;
        FemSpace direct = (P2 ^ 3) - (P1 ^ 3);
        EXPECT_EQ(factored.gatherType(), BaseFemSpace::BaseTypes::VectorType);
        EXPECT_EQ(direct.gatherType(), BaseFemSpace::BaseTypes::VectorType);
        EXPECT_EQ(factored.dofMap().NumDofOnTet(), 18u);
        EXPECT_EQ(direct.dofMap().NumDofOnTet(), 18u);
        EXPECT_TRUE(factored.dofMap() == direct.dofMap());
    }
    {
        FemSpace V1 = (P3 ^ 2) * P2;
        FemSpace V0 = P1 ^ 3;
        FemSpace V10 = V1 - V0;
        EXPECT_EQ(V10.dofMap().NumDofOnTet(), V1.dofMap().NumDofOnTet() - V0.dofMap().NumDofOnTet());
        EXPECT_EQ(V10.dofMap().NumDofOnTet(), 38u);
        EXPECT_EQ(V10.gatherType(), BaseFemSpace::BaseTypes::ComplexType);
        EXPECT_EQ(V10.dim(), 3u);
    }
    {
        // V1×R1 - V0×(L0^k): dim(R1)=k·dim(L0)
        FemSpace V10 = (P2 * (P3 ^ 2)) - (P1 * (P1 ^ 2));
        EXPECT_EQ(V10.gatherType(), BaseFemSpace::BaseTypes::ComplexType);
        EXPECT_EQ(V10.dofMap().NumDofOnTet(), (P2 - P1).dofMap().NumDofOnTet() + ((P3 - P1) ^ 2).dofMap().NumDofOnTet());
        // Same with flattened right-hand side P1*P1*P1
        FemSpace V10f = (P2 * (P3 ^ 2)) - make_complex_raw({P1, P1, P1});
        EXPECT_EQ(V10f.dofMap().NumDofOnTet(), V10.dofMap().NumDofOnTet());
        EXPECT_EQ(V10f.gatherType(), BaseFemSpace::BaseTypes::ComplexType);
    }
}

TEST(AniInterface, EnergyComplementSpaces){
    FemSpace P1{P1Space{}};
    FemSpace P2{P2Space{}};
    FemSpace P3{P3Space{}};

    auto expect_energy_constraints = [&](const FemSpace& V1, const FemSpace& V0, const FemSpace& VE,
                                         const double* D /*nullable*/){
        ASSERT_EQ(VE.gatherType(), BaseFemSpace::BaseTypes::ComplementType);
        auto* ec = VE.target<ComplementFemSpace>();
        ASSERT_NE(ec, nullptr);
        EXPECT_EQ(ec->m_orth, ComplementFemSpace::EnergyComplement);
        const int n1 = static_cast<int>(V1.dofMap().NumDofOnTet());
        const int n0 = static_cast<int>(V0.dofMap().NumDofOnTet());
        const int n10 = static_cast<int>(VE.dofMap().NumDofOnTet());
        EXPECT_EQ(n10, n1 - n0);
        ASSERT_EQ(static_cast<int>(ec->m_basis_coefs.size()), n10 * n1);
        ASSERT_EQ(static_cast<int>(ec->m_dual_coefs.size()), n10 * n1);

        const uint quad_order = 2 * V1.order() + 1;
        auto mats = build_complement_matrices(V1, V0, quad_order); // M1, C
        ASSERT_EQ(mats.n1, n1);
        ASSERT_EQ(mats.n0, n0);

        // Energy matrix A on the regular tet (S4-symmetric), matching ComplementFemSpace::make(EnergyComplement).
        auto quad = make_quad(quad_order);
        const BaseFemSpace& b1 = *V1.base();
        const int gdim3 = static_cast<int>(3 * b1.dim());
        std::vector<double> Dloc(gdim3 * gdim3, 0.0);
        if (D) std::copy(D, D + gdim3 * gdim3, Dloc.data());
        else for (int i = 0; i < gdim3; ++i) Dloc[i + i * gdim3] = 1.0;

        auto req = b1.memGRAD(1, 1);
        PlainMemoryX<> pmx;
        pmx.dSize = req.Usz + req.extraRsz;
        pmx.iSize = req.extraIsz + 2 * (req.mtx_parts + 2);
        pmx.mSize = req.mtx_parts;
        std::vector<char> buf(pmx.enoughRawSize());
        pmx.allocateFromRaw(buf.data(), buf.size());
        AniMemoryX<> mem;
        mem.q = 1; mem.f = 1; mem.busy_mtx_parts = 0;
        mem.U.Init(pmx.ddata, req.Usz); pmx.ddata += req.Usz;
        mem.extraR.Init(pmx.ddata, req.extraRsz);
        mem.extraI.Init(pmx.idata, req.extraIsz);
        mem.MTXI_COL.Init(pmx.idata + req.extraIsz, req.mtx_parts + 2);
        mem.MTXI_ROW.Init(pmx.idata + req.extraIsz + req.mtx_parts + 2, req.mtx_parts + 2);
        mem.MTX.Init(pmx.mdata, req.mtx_parts);
        const double s3 = std::sqrt(3.0), s6 = std::sqrt(6.0);
        double XYZa[12]{0,0,0, 1,0,0, 0.5,s3/2,0, 0.5,s3/6,s6/3};
        double PSI[9], DET = inverse3x3(XYZa + 3, PSI);
        mem.XYP.Init(XYZa, 12); mem.PSI.Init(PSI, 9); mem.DET.Init(&DET, 1);

        std::vector<double> G(static_cast<std::size_t>(n1) * quad.nquad * gdim3, 0.0);
        for (uint q = 0; q < quad.nquad; ++q){
            mem.XYL.Init(quad.xyl.data() + 4 * q, 4);
            mem.WG.Init(quad.wg.data() + q, 1);
            mem.q = 1; mem.busy_mtx_parts = 0;
            auto res = b1.applyGRAD(mem, mem.U);
            for (std::size_t p = 0; p < res.nparts; ++p)
                for (int c = res.stRow[p]; c < res.stRow[p + 1]; ++c)
                    for (int i = res.stCol[p]; i < res.stCol[p + 1]; ++i)
                        if (c >= 0 && c < gdim3 && i >= 0 && i < n1)
                            G[q + quad.nquad * i + n1 * quad.nquad * c] =
                                res.data[p](c - res.stRow[p], i - res.stCol[p]);
        }
        std::vector<double> A(n1 * n1, 0.0), Dg(gdim3);
        for (int i = 0; i < n1; ++i)
            for (int j = i; j < n1; ++j){
                double s = 0;
                for (uint q = 0; q < quad.nquad; ++q){
                    std::fill(Dg.begin(), Dg.end(), 0.0);
                    for (int a = 0; a < gdim3; ++a)
                        for (int b = 0; b < gdim3; ++b)
                            Dg[a] += Dloc[a + b * gdim3] * G[q + quad.nquad * j + n1 * quad.nquad * b];
                    double loc = 0;
                    for (int a = 0; a < gdim3; ++a)
                        loc += G[q + quad.nquad * i + n1 * quad.nquad * a] * Dg[a];
                    s += quad.wg[q] * loc;
                }
                A[i + j * n1] = A[j + i * n1] = s;
            }

        // Energy orthogonality: C A Psi^T ≈ 0
        for (int k = 0; k < n10; ++k){
            std::vector<double> Apsi(n1, 0.0);
            for (int j = 0; j < n1; ++j)
                for (int l = 0; l < n1; ++l)
                    Apsi[j] += A[j + l * n1] * ec->m_basis_coefs[k * n1 + l];
            for (int i = 0; i < n0; ++i){
                double s = 0;
                for (int j = 0; j < n1; ++j)
                    s += mats.C[i + j * n0] * Apsi[j];
                EXPECT_NEAR(s, 0.0, 1e-7) << "energy orth to V0 dof " << i << " mode " << k;
            }
        }

        // Mean-zero: ∫ φ_k ≈ 0 (scalar: one mean row from M1 * ones of P0 embedding ≈ sum of mass row)
        // Use IDEN quadrature means from M1 against constant: m_j = ∫ φ_j = sum_i M1[j,i] * c_i for c=1 in P0.
        // For P1 constants: constant 1 has equal nodal values → mean of mode k is sum_j mean_j Psi[k,j].
        {
            auto XYZ_mean = canonical_tetra();
            auto holder = setup_dual_animem(b1, *V0.base(), quad.nquad, XYZ_mean);
            std::vector<double> phi(static_cast<std::size_t>(n1) * quad.nquad * b1.dim(), 0.0);
            // reuse evaluate via applyIDEN
            for (uint q = 0; q < quad.nquad; ++q){
                holder.mem_v1.XYL.Init(quad.xyl.data() + 4 * q, 4);
                holder.mem_v1.WG.Init(quad.wg.data() + q, 1);
                holder.mem_v1.q = 1; holder.mem_v1.busy_mtx_parts = 0;
                auto res = b1.applyIDEN(holder.mem_v1, holder.mem_v1.U);
                for (std::size_t p = 0; p < res.nparts; ++p)
                    for (int d = res.stRow[p]; d < res.stRow[p + 1]; ++d)
                        for (int i = res.stCol[p]; i < res.stCol[p + 1]; ++i)
                            if (d == 0 && i >= 0 && i < n1)
                                phi[q + quad.nquad * i] = res.data[p](d - res.stRow[p], i - res.stCol[p]);
            }
            for (int k = 0; k < n10; ++k){
                double mean = 0;
                for (int j = 0; j < n1; ++j){
                    double mj = 0;
                    for (uint q = 0; q < quad.nquad; ++q)
                        mj += quad.wg[q] * phi[q + quad.nquad * j];
                    mean += mj * ec->m_basis_coefs[k * n1 + j];
                }
                EXPECT_NEAR(mean, 0.0, 1e-7) << "mean-zero mode " << k;
            }
        }

        // Biorthogonality
        for (int k = 0; k < n10; ++k){
            double s = 0;
            for (int j = 0; j < n1; ++j)
                s += ec->m_basis_coefs[k * n1 + j] * ec->m_dual_coefs[k * n1 + j];
            EXPECT_NEAR(s, 1.0, 1e-8) << "biorthogonality " << k;
        }

        // Span S4-invariance: P1(σ) maps span(Psi) into itself.
        std::array<std::array<DofT::uchar, 4>, 24> perms;
        DofT::S4::all_permutations(perms);
        std::vector<double> P1m(n1 * n1), Proj(n10 * n10);
        for (int p = 0; p < 24; ++p){
            DofT::S4::build_dof_permutation(*V1.dofMap().base(), perms[p].data(), P1m.data(), n1);
            // For each mode k: P1 Psi_k should lie in span(Psi): solve min ||Psi^T c - P Psi_k|| via duals
            for (int k = 0; k < n10; ++k){
                std::vector<double> Ppsi(n1, 0.0);
                for (int i = 0; i < n1; ++i)
                    for (int j = 0; j < n1; ++j)
                        Ppsi[i] += P1m[i + j * n1] * ec->m_basis_coefs[k * n1 + j];
                std::vector<double> recon(n1, 0.0);
                for (int ell = 0; ell < n10; ++ell){
                    double c = 0;
                    for (int j = 0; j < n1; ++j)
                        c += ec->m_dual_coefs[ell * n1 + j] * Ppsi[j];
                    for (int j = 0; j < n1; ++j)
                        recon[j] += c * ec->m_basis_coefs[ell * n1 + j];
                }
                double d2 = 0, n2 = 0;
                for (int j = 0; j < n1; ++j){
                    d2 += (Ppsi[j] - recon[j]) * (Ppsi[j] - recon[j]);
                    n2 += Ppsi[j] * Ppsi[j];
                }
                EXPECT_LT(std::sqrt(d2) / std::max(1e-30, std::sqrt(n2)), 1e-7)
                    << "span invariance failed for perm " << p << " mode " << k;
            }
        }
        (void)Proj;
    };

    auto expect_energy_frame = [&](const FemSpace& V1, const FemSpace& VE, double tol = 1e-8){
        auto* ec = VE.target<ComplementFemSpace>();
        ASSERT_NE(ec, nullptr);
        EXPECT_EQ(ec->m_orth, ComplementFemSpace::EnergyComplement);
        const int n1 = static_cast<int>(V1.dofMap().NumDofOnTet());
        const int n10 = static_cast<int>(VE.dofMap().NumDofOnTet());
        const double* Psi = ec->m_basis_coefs.data();
        double nPsi = 0;
        for (int i = 0; i < n10 * n1; ++i)
            nPsi += Psi[i] * Psi[i];
        nPsi = std::sqrt(nPsi);
        ASSERT_GT(nPsi, 0.0);
        std::array<std::array<DofT::uchar, 4>, 24> perms;
        DofT::S4::all_permutations(perms);
        std::vector<double> P1m(static_cast<std::size_t>(n1) * n1), P10(static_cast<std::size_t>(n10) * n10);
        for (int p = 0; p < 24; ++p){
            DofT::S4::build_dof_permutation(*V1.dofMap().base(), perms[p].data(), P1m.data(), n1);
            DofT::S4::build_dof_permutation(*VE.dofMap().base(), perms[p].data(), P10.data(), n10);
            double d2 = 0;
            for (int i = 0; i < n10; ++i)
                for (int j = 0; j < n1; ++j){
                    double a = 0, b = 0;
                    for (int k = 0; k < n1; ++k)
                        a += Psi[i * n1 + k] * P1m[k + j * n1];
                    for (int k = 0; k < n10; ++k)
                        b += P10[i + k * n10] * Psi[k * n1 + j];
                    d2 += (a - b) * (a - b);
                }
            EXPECT_LT(std::sqrt(d2) / nPsi, tol) << "energy frame equivariance failed for S4 perm " << p;
        }
    };

    // --- primary case: EnergyComplement(P2, P1) ---
    {
        FemSpace VE = ComplementFemSpace::make(P2, P1, ComplementFemSpace::EnergyComplement);
        expect_energy_constraints(P2, P1, VE, nullptr);
        expect_energy_frame(P2, VE);
        auto* ec = VE.target<ComplementFemSpace>();
        ASSERT_NE(ec, nullptr);
        EXPECT_EQ(VE.dofMap().NumDofOnTet(), 6u);
        for (const auto& tag : ec->m_dof_tags){
            EXPECT_EQ(tag.etype, DofT::EDGE_UNORIENT);
            EXPECT_EQ(static_cast<int>(tag.stype), 0);
            EXPECT_EQ(static_cast<int>(tag.lsid), 0);
        }
        const auto* umap = dynamic_cast<const DofT::UniteDofMap*>(VE.dofMap().base().get());
        ASSERT_NE(umap, nullptr);
        EXPECT_EQ(umap->NumDof(DofT::EDGE_UNORIENT), 1u);
        EXPECT_EQ(umap->m_symmetries.get(DofT::EDGE_UNORIENT, 0), 1u);
        EXPECT_EQ(umap->NumDof(DofT::CELL), 0u);

        // Analytical check for EnergyComplement(P2, P1).
        //
        // Theory (regular reference tet, D = I). Let φ_i = λ_i(2λ_i−1), φ_{ij}=4λ_iλ_j
        // be the P2 nodal basis (dofs 0..3 vertices, 4..9 edges in order
        // (01),(02),(03),(12),(13),(23)). Embedding P1↪P2 is C with
        // λ_i ↦ φ_i + (1/2)∑_{e∋i} φ_e. Energy ker is
        //   K = { u ∈ R^{10} : C A u = 0,  ∫_T u = 0 },   dim K = n1 − n0 = 6.
        //
        // Stab(edge 01) ≅ Z2×Z2 acts by swapping {0,1} and/or {2,3}. The space of
        // Stab-invariants in K is exactly 3-dimensional with integer basis
        //   v1 = (−1,−1, 1, 1,  0, 0,0,0,0, 0),
        //   v2 = ( 4, 4, 0, 0,  1, 0,0,0,0, 1),
        //   v3 = ( 8, 8, 0, 0,  0, 1,1,1,1, 0).
        // (Derived by imposing CA·ψ=0 and mean·ψ=0 on the 4-parameter Stab-ansatz
        //  ψ=(α,α,β,β, γ, δ,δ,δ,δ, γ).) A canonical generator is the M1-projection
        // of the edge bubble e_4 onto span{v1,v2,v3}:
        //   w_can = (0, 0, −2/5, −2/5,  3/10, −1/5,−1/5,−1/5,−1/5, 3/10)
        // with ‖w_can‖_{M1}^2 = 1/630. The continuous energy frame is the S4-orbit
        // of any generic vector in this 3-space (one EDGE_UNORIENT dof per edge).
        // Concrete SVD seeds only pick a particular direction in the 3-space, so
        // we check span membership / orbit equality rather than raw floats.
        // Duals are the M1-Riesz duals B = (Ψ M1 Ψ^T)^{-1} Ψ M1.
        {
            using uchar = DofT::uchar;
            const int n1 = 10, n10 = 6;
            static const int edge_ij[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
            auto apply_vertex_perm = [&](const uchar sigma[4], const double* x, double* y){
                std::fill(y, y + n1, 0.0);
                for (int j = 0; j < 4; ++j)
                    y[sigma[j]] = x[j];
                for (int k = 0; k < 6; ++k){
                    uchar a = sigma[edge_ij[k][0]], b = sigma[edge_ij[k][1]];
                    if (a > b) std::swap(a, b);
                    int k2 = -1;
                    for (int t = 0; t < 6; ++t)
                        if (edge_ij[t][0] == a && edge_ij[t][1] == b){ k2 = t; break; }
                    ASSERT_GE(k2, 0);
                    y[4 + k2] = x[4 + k];
                }
            };
            auto edge_sigma = [&](int ea, int eb, uchar sigma[4]){
                int rem[2], nr = 0;
                for (int i = 0; i < 4; ++i)
                    if (i != ea && i != eb) rem[nr++] = i;
                sigma[0] = static_cast<uchar>(ea);
                sigma[1] = static_cast<uchar>(eb);
                sigma[2] = static_cast<uchar>(rem[0]);
                sigma[3] = static_cast<uchar>(rem[1]);
            };

            // Exact Stab-basis for edge 01 and its S4 transports.
            const double V01[3][10] = {
                {-1, -1, 1, 1, 0, 0, 0, 0, 0, 0},
                { 4,  4, 0, 0, 1, 0, 0, 0, 0, 1},
                { 8,  8, 0, 0, 0, 1, 1, 1, 1, 0},
            };
            const double w_can01[10] = {
                0, 0, -2.0/5.0, -2.0/5.0, 3.0/10.0,
                -1.0/5.0, -1.0/5.0, -1.0/5.0, -1.0/5.0, 3.0/10.0
            };

            auto mats = build_complement_matrices(P2, P1, 2 * P2.order() + 1);
            ASSERT_EQ(mats.n1, n1);
            const double* M1 = mats.M1.data();

            // Each numerical mode lies in the analytical 3-space of its edge.
            for (int e = 0; e < n10; ++e){
                ASSERT_EQ(static_cast<int>(ec->m_dof_tags[e].nelem), e);
                uchar sigma[4];
                edge_sigma(edge_ij[e][0], edge_ij[e][1], sigma);
                double Bspan[3][10];
                for (int c = 0; c < 3; ++c)
                    apply_vertex_perm(sigma, V01[c], Bspan[c]);
                const double* psi = ec->m_basis_coefs.data() + e * n1;
                double G[9] = {0}, rhs[3] = {0};
                for (int j = 0; j < 3; ++j)
                    for (int i = 0; i < 3; ++i){
                        double s = 0;
                        for (int k = 0; k < n1; ++k)
                            s += Bspan[i][k] * Bspan[j][k];
                        G[i + j * 3] = s;
                    }
                for (int i = 0; i < 3; ++i){
                    double s = 0;
                    for (int k = 0; k < n1; ++k)
                        s += Bspan[i][k] * psi[k];
                    rhs[i] = s;
                }
                double chol_mem[6], coef[3];
                cholesky_solve(G, rhs, 3, 1, coef, chol_mem);
                double err2 = 0, nrm2 = 0;
                for (int k = 0; k < n1; ++k){
                    double r = psi[k];
                    for (int c = 0; c < 3; ++c)
                        r -= coef[c] * Bspan[c][k];
                    err2 += r * r;
                    nrm2 += psi[k] * psi[k];
                }
                EXPECT_LT(std::sqrt(err2) / std::max(1e-30, std::sqrt(nrm2)), 1e-12)
                    << "EP2P1 mode " << e << " not in analytical Stab-span";
                double m2 = 0;
                for (int i = 0; i < n1; ++i)
                    for (int j = 0; j < n1; ++j)
                        m2 += psi[i] * M1[i + j * n1] * psi[j];
                EXPECT_NEAR(m2, 1.0, 1e-10) << "EP2P1 mode " << e << " M1-norm";
            }

            // Span(Ψ) = span(S4-orbit of w_can) in the M1 metric.
            {
                double Orbit[6][10];
                for (int e = 0; e < 6; ++e){
                    uchar sigma[4];
                    edge_sigma(edge_ij[e][0], edge_ij[e][1], sigma);
                    apply_vertex_perm(sigma, w_can01, Orbit[e]);
                }
                for (int e = 0; e < n10; ++e){
                    const double* psi = ec->m_basis_coefs.data() + e * n1;
                    double Gram[36] = {0}, rhs[6] = {0};
                    for (int j = 0; j < 6; ++j)
                        for (int i = 0; i < 6; ++i){
                            double s = 0;
                            for (int a = 0; a < n1; ++a)
                                for (int b = 0; b < n1; ++b)
                                    s += Orbit[i][a] * M1[a + b * n1] * Orbit[j][b];
                            Gram[i + j * 6] = s;
                        }
                    for (int i = 0; i < 6; ++i){
                        double s = 0;
                        for (int a = 0; a < n1; ++a)
                            for (int b = 0; b < n1; ++b)
                                s += Orbit[i][a] * M1[a + b * n1] * psi[b];
                        rhs[i] = s;
                    }
                    double chol_mem[21], coef[6];
                    cholesky_solve(Gram, rhs, 6, 1, coef, chol_mem);
                    std::vector<double> recon(n1, 0.0), diff(n1, 0.0);
                    for (int a = 0; a < n1; ++a){
                        for (int i = 0; i < 6; ++i)
                            recon[a] += coef[i] * Orbit[i][a];
                        diff[a] = psi[a] - recon[a];
                    }
                    double d2 = 0, n2 = 0;
                    for (int a = 0; a < n1; ++a)
                        for (int b = 0; b < n1; ++b){
                            d2 += diff[a] * M1[a + b * n1] * diff[b];
                            n2 += psi[a] * M1[a + b * n1] * psi[b];
                        }
                    EXPECT_LT(std::sqrt(std::max(0.0, d2)) / std::max(1e-30, std::sqrt(n2)), 1e-10)
                        << "EP2P1 mode " << e << " not in analytical w_can orbit";
                }
            }

            // Duals: B = (Ψ M1 Ψ^T)^{-1} Ψ M1
            {
                double Gram[36] = {0};
                for (int j = 0; j < n10; ++j)
                    for (int i = 0; i < n10; ++i){
                        double s = 0;
                        for (int a = 0; a < n1; ++a)
                            for (int b = 0; b < n1; ++b)
                                s += ec->m_basis_coefs[i * n1 + a] * M1[a + b * n1]
                                   * ec->m_basis_coefs[j * n1 + b];
                        Gram[i + j * n10] = s;
                    }
                std::vector<double> PsiM(n10 * n1, 0.0);
                for (int i = 0; i < n10; ++i)
                    for (int j = 0; j < n1; ++j){
                        double s = 0;
                        for (int k = 0; k < n1; ++k)
                            s += ec->m_basis_coefs[i * n1 + k] * M1[k + j * n1];
                        PsiM[i + j * n10] = s;
                    }
                double chol_mem[21];
                cholesky_solve(Gram, PsiM.data(), n10, n1, PsiM.data(), chol_mem);
                double err2 = 0, nrm2 = 0;
                for (int i = 0; i < n10; ++i)
                    for (int j = 0; j < n1; ++j){
                        const double d = PsiM[i + j * n10] - ec->m_dual_coefs[i * n1 + j];
                        err2 += d * d;
                        nrm2 += ec->m_dual_coefs[i * n1 + j] * ec->m_dual_coefs[i * n1 + j];
                    }
                EXPECT_LT(std::sqrt(err2) / std::max(1e-30, std::sqrt(nrm2)), 1e-10)
                    << "EP2P1 duals disagree with analytical M1-Riesz formula";
            }
        }
    }

    // Scaled isotropic D = 2 I (same ker as I up to scaling of A)
    {
        double D[9] = {0};
        D[0] = D[4] = D[8] = 2.0;
        FemSpace VE = ComplementFemSpace::make(P2, P1, ComplementFemSpace::EnergyComplement, D);
        expect_energy_constraints(P2, P1, VE, D);
        EXPECT_EQ(VE.dofMap().NumDofOnTet(), 6u);
    }

    // Vector factorization with D = I
    {
        FemSpace VE = ComplementFemSpace::make(P2 ^ 3, P1 ^ 3, ComplementFemSpace::EnergyComplement);
        EXPECT_EQ(VE.gatherType(), BaseFemSpace::BaseTypes::VectorType);
        EXPECT_EQ(VE.dofMap().NumDofOnTet(), 18u);
    }

    // P3-P1: same continuous S4 classification as L2; dim = n1-n0
    {
        FemSpace VE = ComplementFemSpace::make(P3, P1, ComplementFemSpace::EnergyComplement);
        EXPECT_EQ(VE.dofMap().NumDofOnTet(), 16u);
        expect_energy_constraints(P3, P1, VE, nullptr);
        expect_energy_frame(P3, VE);
        auto* ec = VE.target<ComplementFemSpace>();
        ASSERT_NE(ec, nullptr);
        int n_edge = 0, n_face = 0, n_cell = 0;
        for (const auto& tag : ec->m_dof_tags){
            if (tag.etype & DofT::EDGE) ++n_edge;
            else if (tag.etype & DofT::FACE) ++n_face;
            else if (tag.etype == DofT::CELL) ++n_cell;
        }
        EXPECT_EQ(n_edge + n_face + n_cell, 16);
        EXPECT_EQ(n_cell, 0);
    }
}

