//
// Created by Liogky Alexey on 21.02.2023.
//

#include <gtest/gtest.h>
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include <chrono>

TEST(AniInterface, Operations_IntEdge){
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
    //int_e (D*GRAD(FEM_P2)) * IDEN(FEM_P1^3)
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
    int edge_id = 1;
    double coef = 60;
    std::array<double, 12*10> A_expd = {
        -160, 0, -30 , 0, -175, 0, -33 , 0, -190, 0, -36 , 0, 
        -30 , 0, -30 , 0, -33 , 0, -33 , 0, -36 , 0, -36 , 0, 
         20 , 0,  90 , 0,  21 , 0,  95 , 0,  22 , 0, 100 , 0, 
        -30 , 0, -30 , 0, -33 , 0, -33 , 0, -36 , 0, -36 , 0, 
         80 , 0,  40 , 0,  88 , 0,  44 , 0,  96 , 0,  48 , 0, 
        -40 , 0, -240, 0, -44 , 0, -260, 0, -48 , 0, -280, 0, 
         80 , 0,  40 , 0,  88 , 0,  44 , 0,  96 , 0,  48 , 0, 
         40 , 0,  80 , 0,  44 , 0,  88 , 0,  48 , 0,  96 , 0, 
          0 , 0,   0 , 0,   0 , 0,   0 , 0,   0 , 0,   0 , 0, 
         40 , 0,  80 , 0,  44 , 0,  88 , 0,  48 , 0,  96 , 0
    };
    DenseMatrix<> A_exp(A_expd.data(), OP2::Nfa::value, OP1::Nfa::value);
    std::array<double, OP1::Nfa::value*OP2::Nfa::value> Adat;
    DenseMatrix<> A(Adat.data(), 1, Adat.size());
    std::vector<char> raw_mem;
    PlainMemory<> reqt;
    ApplyOp op1 = ApplyOp(ApplyOpFromTemplate<GRAD, FemFix<FEM_P2>>());
    ApplyOp op2 = ApplyOp(ApplyOpFromTemplate<IDEN, FemVec<3, FEM_P1>>());
    PlainMemoryX<> reqx;

    fem3Dedge<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, edge_id, dfunc, A, 3);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3Dedge_memory_requirements<OP1, OP2>(3, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dedge<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, edge_id, dfunc, A, reqt, 3);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3Dedge_memory_requirements<>(*op1.m_invoker, *op2.m_invoker, 3, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dedge<>(XYZ, edge_id, *op1.m_invoker, *op2.m_invoker, dfunc, A, reqx, 3);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();

    fem3Dedge<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, edge_id, dfunc_fuse, A, 3);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3Dedge_memory_requirements<OP1, OP2>(3, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dedge<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, edge_id, dfunc_fuse, A, reqt, 3);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3Dedge_memory_requirements<DfuncTraitsFusive<>>(*op1.m_invoker, *op2.m_invoker, 3, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dedge<DfuncTraitsFusive<>>(XYZ, edge_id, *op1.m_invoker, *op2.m_invoker, dfunc_fuse, A, reqx, 3);
    EXPECT_NEAR(norm(1.0/coef, A_exp, -1, A), 0, 100*(1 + norm(1/coef, A_exp))*DBL_EPSILON);
    A.SetZero();
}