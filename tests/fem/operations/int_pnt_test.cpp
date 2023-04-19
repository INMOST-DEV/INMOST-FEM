//
// Created by Liogky Alexey on 21.02.2023.
//

#include <gtest/gtest.h>
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include <chrono>

TEST(AniInterface, Operations_IntNode){
    using namespace Ani;
    auto norm = [](double a, const DenseMatrix<>& A, double b = 0.0, const DenseMatrix<>& B = DenseMatrix<>(nullptr, 0, 0)){
        return DenseMatrix<>::ScalNorm(a, A, b, B);
    };
    double  XY1p[] = {1, 1, 1},
            XY2p[] = {2, 1, 1},
            XY3p[] = {1, 2, 1},
            XY4p[] = {1, 1, 2};
    Tetra<const double> XYZ(XY1p, XY2p, XY3p, XY4p);
    using OP1 = Operator<GRAD, FemFix<FEM_P2>>;
    using OP2 = Operator<IDEN, FemVec<3, FEM_P1>>;
    //(D*GRAD(FEM_P2)) * IDEN(FEM_P1^3) at node
    auto dfunc = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void) user_data; (void) iTet;
        constexpr auto d1 = OP2::Dim::value,
            d2 = OP1::Dim::value;
        if (Ddims.first != d1 || Ddims.second != d2 )
            throw std::runtime_error("Error in expected tensor sizes");
        DenseMatrix<> D(Dmem, d1, d2);
        for (int j = 0; j < d2; ++j)
            for (int i = 0; i < d1; ++ i)
                D(i, j) = x[j%3] + i/10.0;
        return TENSOR_GENERAL;        
    };
    auto make_fuse_tensor = [](auto& f){
        return [&f](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem) -> TensorType{
            for (std::size_t r = 0; r < mem.f; ++r)
                for (std::size_t n = 0; n < mem.q; ++n){
                    DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
                    f({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]}, 
                                Dloc.data, Ddims, user_data, r);
                }
            return TENSOR_GENERAL;
        };
    };
    auto dfunc_fuse = make_fuse_tensor(dfunc);
    int node_id = 1;
    std::array<double, 12*10> A_expd = {
        0,   4, 0, 0, 0,   4.3, 0, 0, 0,   4.6, 0, 0, 
        0,   6, 0, 0, 0,   6.3, 0, 0, 0,   6.6, 0, 0, 
        0,  -1, 0, 0, 0,  -1.1, 0, 0, 0,  -1.2, 0, 0,
        0,  -1, 0, 0, 0,  -1.1, 0, 0, 0,  -1.2, 0, 0, 
        0, -16, 0, 0, 0, -17.2, 0, 0, 0, -18.4, 0, 0, 
        0,   0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0, 
        0,   0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0, 
        0,   4, 0, 0, 0,   4.4, 0, 0, 0,   4.8, 0, 0, 
        0,   4, 0, 0, 0,   4.4, 0, 0, 0,   4.8, 0, 0, 
        0,   0, 0, 0, 0,     0, 0, 0, 0,     0, 0, 0,
    };
    DenseMatrix<> A_exp(A_expd.data(), OP2::Nfa::value, OP1::Nfa::value);
    std::array<double, OP1::Nfa::value*OP2::Nfa::value> Adat;
    DenseMatrix<> A(Adat.data(), 1, Adat.size());
    std::vector<char> raw_mem;
    PlainMemory<> reqt;
    ApplyOp op1 = ApplyOp(ApplyOpFromTemplate<GRAD, FemFix<FEM_P2>>());
    ApplyOp op2 = ApplyOp(ApplyOpFromTemplate<IDEN, FemVec<3, FEM_P1>>());
    PlainMemoryX<> reqx;
    fem3Dnode<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, node_id, dfunc, A);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3Dnode_memory_requirements<OP1, OP2>(1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dnode<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, node_id, dfunc, A, reqt);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3Dnode_memory_requirements<>(*op1.m_invoker, *op2.m_invoker, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dnode<>(XYZ, node_id, *op1.m_invoker, *op2.m_invoker, dfunc, A, reqx);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();

    fem3Dnode<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, node_id, dfunc_fuse, A);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3Dnode_memory_requirements<OP1, OP2>(1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dnode<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, node_id, dfunc_fuse, A, reqt);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3Dnode_memory_requirements<DfuncTraitsFusive<>>(*op1.m_invoker, *op2.m_invoker, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dnode<DfuncTraitsFusive<>>(XYZ, node_id, *op1.m_invoker, *op2.m_invoker, dfunc_fuse, A, reqx);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
}

TEST(AniInterface, Operations_IntPnt){
    using namespace Ani;
    auto norm = [](double a, const DenseMatrix<>& A, double b = 0.0, const DenseMatrix<>& B = DenseMatrix<>(nullptr, 0, 0)){
        return DenseMatrix<>::ScalNorm(a, A, b, B);
    };
    double  XY1p[] = {0, 0, 0},
            XY2p[] = {2, 1, 1},
            XY3p[] = {1, 2, 1},
            XY4p[] = {2, 1, 2};        
    Tetra<const double> XYZ(XY1p, XY2p, XY3p, XY4p);
    using OP1 = Operator<GRAD, FemFix<FEM_P3>>;
    using OP2 = Operator<GRAD, FemVec<3, FEM_P1>>;
    //(D*GRAD(FEM_P2)) * IDEN(FEM_P1^3) at node
    auto dfunc = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void) user_data; (void) iTet;
        constexpr auto d1 = OP2::Dim::value,
            d2 = OP1::Dim::value;
        if (Ddims.first != d1 || Ddims.second != d2 )
            throw std::runtime_error("Error in expected tensor sizes");
        DenseMatrix<> D(Dmem, d1, d2);
        int cnt = 0;
        for (ulong i = 0; i < Ddims.first; ++i)
            for (ulong j = 0; j < Ddims.second; ++j)
                D(i, j) = (i * x[0] + j * x[1] + (i%2)*x[2]) * (cnt++);
        return TENSOR_GENERAL;       
    };
    auto make_fuse_tensor = [](auto& f){
        return [&f](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem) -> TensorType{
            for (std::size_t r = 0; r < mem.f; ++r)
                for (std::size_t n = 0; n < mem.q; ++n){
                    DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
                    f({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]}, 
                                Dloc.data, Ddims, user_data, r);
                }
            return TENSOR_GENERAL;
        };
    };
    auto dfunc_fuse = make_fuse_tensor(dfunc);
    auto formula = tetrahedron_quadrature_formulas(5);
    uint q = formula.GetNumPoints();
    std::vector<double> xyld(4*q), wd(q), xygd(3*q);
    std::copy(formula.GetPointData(), formula.GetPointData()+4*q, xyld.data());
    std::copy(formula.GetWeightData(), formula.GetWeightData()+q, wd.data());
    double fn[3];
    cross(XY1p, XY2p, XY3p, fn);
    double vol = std::abs((fn[0]*(XY4p[0] - XY1p[0])+fn[1]*(XY4p[1] - XY1p[1])+fn[2]*(XY4p[2] - XY1p[2])))/6;
    std::for_each(wd.begin(), wd.end(), [vol](auto& x){ x*=vol; });
    for (uint i = 0; i < q; ++i)
        for (int k = 0; k < 3; ++k)
            xygd[k + 3*i] = xyld[0+4*i]*XY1p[k]+xyld[1+4*i]*XY2p[k]+xyld[2+4*i]*XY3p[k] + xyld[3+4*i]*XY4p[k];
    ArrayView<> XYL(xyld.data(), 4*q), WG(wd.data(), q);
    ArrayView<const double> XYG(xygd.data(), 3*q);        
    
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
    std::array<double, 12*20> A_exp_mp, Admp;
    DenseMatrix<> A_exp(A_exp_mp.data(), 12, 20), A(Admp.data(), 12, 20);
    for (std::size_t i = 0; i < A_exp.nRow; ++i) 
        for (std::size_t j = 0; j < A_exp.nCol; ++j) 
            A_exp(i, j) = At_exp_m[i][j];
    std::vector<char> raw_mem;
    PlainMemory<> reqt;
    ApplyOp op1 = ApplyOp(ApplyOpFromTemplate<GRAD, FemFix<FEM_P3>>());
    ApplyOp op2 = ApplyOp(ApplyOpFromTemplate<GRAD, FemVec<3, FEM_P1>>());
    PlainMemoryX<> reqx;        

    fem3DpntL<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, XYL, WG, dfunc, A);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    // for (int i = 0; i < 12; ++i){
    //     std::cout << "{";
    //     for (int j = 0; j < 20; ++j){
    //         long r = lround(A(i, j)*80*9);
    //         std::cout << r << ", "; 
    //     }
    //     std::cout << "},\n";
    // }
    // std::cout << std::endl;
    //std::cout << A << std::endl;    

    A.SetZero();
    fem3DpntX<OP1, OP2>( XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, XYG, WG, dfunc, A);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3DpntL_memory_requirements<OP1, OP2>(q, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntL<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, XYL, WG, dfunc, A, reqt);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3DpntX_memory_requirements<OP1, OP2>(q, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntX<OP1, OP2>( XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, XYG, WG, dfunc, A, reqt);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3DpntL_memory_requirements<>(*op1.m_invoker, *op2.m_invoker, q, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntL<>(XYZ, XYL, WG, *op1.m_invoker, *op2.m_invoker, dfunc, A, reqx);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3DpntX_memory_requirements<>(*op1.m_invoker, *op2.m_invoker, q, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntX<>(XYZ, XYG, WG, *op1.m_invoker, *op2.m_invoker, dfunc, A, reqx);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();

    fem3DpntL<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, XYL, WG, dfunc_fuse, A);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    fem3DpntX<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, XYG, WG, dfunc_fuse, A);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3DpntL_memory_requirements<OP1, OP2>(q, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntL<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, XYL, WG, dfunc_fuse, A, reqt);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3DpntX_memory_requirements<OP1, OP2>(q, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntX<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, XYG, WG, dfunc_fuse, A, reqt);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3DpntL_memory_requirements<DfuncTraitsFusive<>>(*op1.m_invoker, *op2.m_invoker, q, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntL<DfuncTraitsFusive<>>(XYZ, XYL, WG, *op1.m_invoker, *op2.m_invoker, dfunc_fuse, A, reqx);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3DpntX_memory_requirements<DfuncTraitsFusive<>>(*op1.m_invoker, *op2.m_invoker, q, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DpntX<DfuncTraitsFusive<>>(XYZ, XYG, WG, *op1.m_invoker, *op2.m_invoker, dfunc_fuse, A, reqx);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
}