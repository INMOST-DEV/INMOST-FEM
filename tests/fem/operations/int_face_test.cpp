//
// Created by Liogky Alexey on 21.02.2023.
//

#include <gtest/gtest.h>
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include <chrono>

TEST(AniInterface, Operations_IntFace){
    using namespace Ani;
    auto norm = [](double a, const DenseMatrix<>& A, double b = 0.0, const DenseMatrix<>& B = DenseMatrix<>(nullptr, 0, 0)){
        return DenseMatrix<>::ScalNorm(a, A, b, B);
    };
    double  XY1p[] = {1, 1, 1},
            XY2p[] = {2, 1, 1},
            XY3p[] = {1, 2, 1},
            XY4p[] = {1, 1, 2};
    Tetra<const double> XYZ(XY1p, XY2p, XY3p, XY4p);
    double* XYZp[] = {XY1p, XY2p, XY3p, XY4p};
    struct FaceData{
        double **XYZ;
        int face_id;
    };
    using OP1 = Operator<GRAD, FemFix<FEM_P2>>;
    using OP2 = Operator<IDEN, FemVec<3, FEM_P1>>;
    //int_f ((n.D)*GRAD(FEM_P2)) * IDEN(FEM_P1^3)
    auto dfunc = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void) user_data; (void) iTet;
        constexpr auto d1 = OP2::Dim::value,
            d2 = OP1::Dim::value;
        if (Ddims.first != d1*3 || Ddims.second != d2 )
            throw std::runtime_error("Error in expected tensor sizes");
        DenseMatrix<> D(Dmem, d1*3, d2);
        for (int j = 0; j < d2; ++j)
            for (int i = 0; i < d1*3; ++ i)
                D(i, j) = x[j%3] + i/10.0;
        return TENSOR_GENERAL;        
    };
    auto ddotn = [dfunc](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
         constexpr auto d1 = OP2::Dim::value,
            d2 = OP1::Dim::value;
        if (Ddims.first != d1 || Ddims.second != d2 )
            throw std::runtime_error("Error in expected tensor sizes");
        std::array<double, d1*d2*3> Dv;
        dfunc(x, Dv.data(), {d1*3, d2}, user_data, iTet);
        auto& dat = *static_cast<FaceData*>(user_data);
        double nrm[3];
        face_normal(dat.XYZ[(dat.face_id+0)%4], dat.XYZ[(dat.face_id+1)%4], dat.XYZ[(dat.face_id+2)%4], dat.XYZ[(dat.face_id+3)%4], nrm);
        DenseMatrix<> Dd(Dv.data(), d1*3, d2);
        DenseMatrix<> D(Dmem, d1, d2);
        for (int j = 0; j < d2; ++j)
            for (int i = 0; i < d1; ++ i)
                D(i, j) = Dd(3*i+0, j)*nrm[0] + Dd(3*i+1, j)*nrm[1] + Dd(3*i+2, j)*nrm[2];
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
    auto ddotn_fuse = make_fuse_tensor(ddotn);
    int face_id = 1;
    FaceData fd{XYZp, face_id};
    std::array<double, 12*10> A_expd = {
        0.000, 2.150, 2.150, 2.150, 0.000, 2.600, 2.600, 2.600, 0.000, 3.050, 3.050, 3.050, 
        0.000, 0.900, 0.075, 0.075, 0.000, 1.050, 0.075, 0.075, 0.000, 1.200, 0.075, 0.075, 
        0.000, 0.075, 0.900, 0.075, 0.000, 0.075, 1.050, 0.075, 0.000, 0.075, 1.200, 0.075, 
        0.000, 0.075, 0.075, 0.900, 0.000, 0.075, 0.075, 1.050, 0.000, 0.075, 0.075, 1.200, 
        0.000,-4.300,-2.150,-2.150, 0.000,-5.200,-2.600,-2.600, 0.000,-6.100,-3.050,-3.050, 
        0.000,-2.150,-4.300,-2.150, 0.000,-2.600,-5.200,-2.600, 0.000,-3.050,-6.100,-3.050, 
        0.000,-2.150,-2.150,-4.300, 0.000,-2.600,-2.600,-5.200, 0.000,-3.050,-3.050,-6.100, 
        0.000, 2.050, 2.050, 1.300, 0.000, 2.500, 2.500, 1.600, 0.000, 2.950, 2.950, 1.900, 
        0.000, 2.050, 1.300, 2.050, 0.000, 2.500, 1.600, 2.500, 0.000, 2.950, 1.900, 2.950, 
        0.000, 1.300, 2.050, 2.050, 0.000, 1.600, 2.500, 2.500, 0.000, 1.900, 2.950, 2.950
    };
    DenseMatrix<> A_exp(A_expd.data(), OP2::Nfa::value, OP1::Nfa::value);
    std::array<double, OP1::Nfa::value*OP2::Nfa::value> Adat;
    DenseMatrix<> A(Adat.data(), 1, Adat.size());
    std::vector<char> raw_mem;
    PlainMemory<> reqt;
    ApplyOp op1 = ApplyOp(ApplyOpFromTemplate<GRAD, FemFix<FEM_P2>>());
    ApplyOp op2 = ApplyOp(ApplyOpFromTemplate<IDEN, FemVec<3, FEM_P1>>());
    PlainMemoryX<> reqx;

    fem3Dface<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, ddotn, A, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    fem3DfaceN<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, dfunc, A, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();

    reqt = fem3Dface_memory_requirements<OP1, OP2>(3, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dface<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, ddotn, A, reqt, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3DfaceN_memory_requirements<OP1, OP2>(3, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DfaceN<OP1, OP2>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, dfunc, A, reqt, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();

    reqx = fem3Dface_memory_requirements<>(*op1.m_invoker, *op2.m_invoker, 3, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dface<>(XYZ, face_id, *op1.m_invoker, *op2.m_invoker, ddotn, A, reqx, 3, &fd);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3DfaceN_memory_requirements<>(*op1.m_invoker, *op2.m_invoker, 3, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DfaceN<>(XYZ, face_id, *op1.m_invoker, *op2.m_invoker, dfunc, A, reqx, 3, &fd);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();

    fem3Dface<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, ddotn_fuse, A, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    fem3DfaceN<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, dfunc_fuse, A, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();

    reqt = fem3Dface_memory_requirements<OP1, OP2>(3, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dface<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, ddotn_fuse, A, reqt, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqt = fem3DfaceN_memory_requirements<OP1, OP2>(3, 1);
    raw_mem.resize(reqt.enoughRawSize());
    reqt.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DfaceN<OP1, OP2, DfuncTraitsFusive<>>( XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, face_id, dfunc_fuse, A, reqt, 3, &fd );
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();

    reqx = fem3Dface_memory_requirements<DfuncTraitsFusive<>>(*op1.m_invoker, *op2.m_invoker, 3, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3Dface<DfuncTraitsFusive<>>(XYZ, face_id, *op1.m_invoker, *op2.m_invoker, ddotn_fuse, A, reqx, 3, &fd);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
    reqx = fem3DfaceN_memory_requirements<DfuncTraitsFusive<>>(*op1.m_invoker, *op2.m_invoker, 3, 1);
    raw_mem.resize(reqx.enoughRawSize());
    reqx.allocateFromRaw(raw_mem.data(), raw_mem.size());
    fem3DfaceN<DfuncTraitsFusive<>>(XYZ, face_id, *op1.m_invoker, *op2.m_invoker, dfunc_fuse, A, reqx, 3, &fd);
    EXPECT_NEAR(norm(1.0, A_exp, -1, A), 0, 100*(1 + norm(1, A_exp))*DBL_EPSILON);
    A.SetZero();
}