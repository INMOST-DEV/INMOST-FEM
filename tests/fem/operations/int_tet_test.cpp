//
// Created by Liogky Alexey on 21.03.2022.
//

#include <gtest/gtest.h>
#define OPTIMIZER_TIMERS_FEM3DTET
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include <chrono>

TEST(AniInterface, Operations_IntTet_SpeedTest){
#ifndef NDEBUG
    EXPECT_EQ(0, 0);
    return;
#endif
    using namespace Ani;
    double XY1d[] = {1, 1, 1, 0, 0, 0};
    double XY2d[] = {2, 1, 1, 2, 1, 1};
    double XY3d[] = {1, 2, 1, 1, 2, 1};
    double XY4d[] = {1, 1, 2, 2, 1, 2};
    DenseMatrix<> XY1(XY1d+3, 3, 1, 3);
    DenseMatrix<> XY2(XY2d+3, 3, 1, 3);
    DenseMatrix<> XY3(XY3d+3, 3, 1);
    DenseMatrix<> XY4(XY4d+3, 3, 1);
    Tetra<> XYZ(XY1d, XY2d, XY3d, XY4d);
    double Ad[30*30];
    DenseMatrix<double> A(Ad, 0, 0, 30*30);
    PlainMemory<double> pMem;
    auto update_pMem = [&pMem](auto opA, auto opB, int order = 5){
        auto res = fem3Dtet_memory_requirements<decltype(opA), decltype(opB), double>(order, 1);
        if (res.iSize > pMem.iSize) pMem.iSize = res.iSize;
        if (res.dSize > pMem.dSize) pMem.dSize = res.dSize;
    };
    update_pMem(Operator<IDEN, FemFix<FEM_P1>>(), Operator<IDEN, FemFix<FEM_P1>>(), 2);
    update_pMem(Operator<GRAD, FemFix<FEM_P1>>(), Operator<GRAD, FemFix<FEM_P1>>());
    update_pMem(Operator<GRAD, FemVec<3,FEM_P1>>(), Operator<GRAD, FemVec<3,FEM_P1>>());
    update_pMem(Operator<IDEN, FemFix<FEM_P2>>(), Operator<IDEN, FemFix<FEM_P2>>());
    update_pMem(Operator<IDEN, FemFix<FEM_P3>>(), Operator<IDEN, FemFix<FEM_P3>>());
    update_pMem(Operator<GRAD, FemVec<3,FEM_P2>>(), Operator<GRAD, FemVec<3,FEM_P2>>());
    FemSpace P1{P1Space{}};
    FemSpace P1vec = P1^3;
    FemSpace P2{P2Space{}};
    FemSpace P2vec = P2^3;
    FemSpace P3{P3Space{}};
    PlainMemoryX<> rMem;
    auto update_rMem = [&rMem](auto op1, auto op2, int order = 5){
        rMem.extend_size(fem3Dtet_memory_requirements<>(op1, op2, order));
    };
    update_rMem(P1.getOP(IDEN), P1.getOP(IDEN), 2);
    update_rMem(P1.getOP(GRAD), P1.getOP(GRAD));
    update_rMem(P1vec.getOP(IDEN), P1vec.getOP(IDEN));
    update_rMem(P2.getOP(IDEN), P2.getOP(IDEN));
    update_rMem(P3.getOP(IDEN), P3.getOP(IDEN));
    update_rMem(P2vec.getOP(GRAD), P2vec.getOP(GRAD));

    std::vector<double> ddata(std::max(pMem.dSize, rMem.dSize));
    std::vector<int> idata(std::max(pMem.iSize, rMem.iSize));
    std::vector<DenseMatrix<>> mdata(rMem.mSize);
    pMem.ddata = ddata.data(); pMem.idata = idata.data();
    rMem.ddata = ddata.data(); rMem.idata = idata.data(); rMem.mdata = mdata.data();
    DynMem<> wmem;
    const int NEVALS = 25000;

    auto dfunc = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void) user_data; (void) iTet;
        DenseMatrix<> D(Dmem, Ddims.first, Ddims.second);
        int cnt = 0;
        for (ulong i = 0; i < Ddims.first; ++i)
            for (ulong j = 0; j < Ddims.second; ++j)
                D(i, j) = (i * x[0] + j * x[1] + (i%2)*x[2]) * (cnt++);
        return TENSOR_NULL;
    };
//    std::function<TensorType(const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)> dfunc(dfunc_);
    using FTraits = DfuncTraits<TENSOR_NULL, true>;
    //to stabilize processor work
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<GRAD, FemFix<FEM_P1>>, Operator<GRAD, FemFix<FEM_P1>>, FTraits>(XY1, XY2, XY3, XY4, dfunc, A,
                                                                                          pMem, 5);

    auto t_start = std::chrono::high_resolution_clock::now();
    auto set_start_timers = [&](){
        t_start = std::chrono::high_resolution_clock::now();
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        T_init_F3DTet = T_U_F3DTet = T_V_F3DTet = T_DU_F3DTet = T_A_F3DTet = 0;
#endif
    };
    auto print_state = [&](const std::string& test_name){
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout << test_name << ": " << NEVALS << " iters\n";
        std::cout << "\tFullTime = " << elapsed_time_ms << "ms" << std::endl;
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        std::cout << "\tT_init = " << T_init_F3DTet << "ms" << std::endl;
        std::cout << "\tT_U = " << T_U_F3DTet << "ms" << std::endl;
        std::cout << "\tT_V = " << T_V_F3DTet << "ms" << std::endl;
        std::cout << "\tT_DU = " << T_DU_F3DTet << "ms" << std::endl;
        std::cout << "\tT_A = " << T_A_F3DTet << "ms" << std::endl;
#endif
    };

    set_start_timers();
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>, FTraits>(XY1, XY2, XY3, XY4, dfunc, A,
                                                                                          pMem, 2);
    print_state("template IDEN(P1)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P1.getOP(IDEN), P1.getOP(IDEN), dfunc, A, rMem, 2);
    print_state("runtime IDEN(P1)");   
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P1.getOP(IDEN), P1.getOP(IDEN), dfunc, A, wmem, 2);
    print_state("runtime IDEN(P1) dyn mem");     

    set_start_timers();
    t_start = std::chrono::high_resolution_clock::now();
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<GRAD, FemFix<FEM_P1>>, Operator<GRAD, FemFix<FEM_P1>>, FTraits>(XY1, XY2, XY3, XY4, dfunc, A,
                                                                                          pMem, 5);
    print_state("template GRAD(P1)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P1.getOP(GRAD), P1.getOP(GRAD), dfunc, A, rMem, 5);
    print_state("runtime GRAD(P1)"); 
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P1.getOP(GRAD), P1.getOP(GRAD), dfunc, A, wmem, 5);
    print_state("runtime GRAD(P1) dyn mem"); 

    set_start_timers();
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<GRAD, FemVec<3, FEM_P1>>, Operator<GRAD, FemVec<3, FEM_P1>>, FTraits>(XY1, XY2, XY3, XY4,
                                                                                                dfunc, A, pMem, 5);
    print_state("template GRAD(P1vec)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P1vec.getOP(GRAD), P1vec.getOP(GRAD), dfunc, A, rMem, 5);
    print_state("runtime GRAD(P1vec)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P1vec.getOP(GRAD), P1vec.getOP(GRAD), dfunc, A, wmem, 5);
    print_state("runtime GRAD(P1vec) dyn mem");

    set_start_timers();
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<IDEN, FemFix<FEM_P2>>, Operator<IDEN, FemFix<FEM_P2>>, FTraits>(XY1, XY2, XY3, XY4, dfunc, A,
                                                                                          pMem, 5);
    print_state("template IDEN(P2)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P2.getOP(GRAD), P2.getOP(GRAD), dfunc, A, rMem, 5);
    print_state("runtime IDEN(P2)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P2.getOP(GRAD), P2.getOP(GRAD), dfunc, A, wmem, 5);
    print_state("runtime IDEN(P2) dyn mem");

    set_start_timers();
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<GRAD, FemVec<3, FEM_P2>>, Operator<GRAD, FemVec<3, FEM_P2>>, FTraits>(XY1, XY2, XY3, XY4,
                                                                                                dfunc, A, pMem, 5);
    print_state("template GRAD(P2vec)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P2vec.getOP(GRAD), P2vec.getOP(GRAD), dfunc, A, rMem, 5);
    print_state("runtime GRAD(P2vec)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P2vec.getOP(GRAD), P2vec.getOP(GRAD), dfunc, A, wmem, 5);
    print_state("runtime GRAD(P2vec) dyn mem");

    set_start_timers();
    for (int l = 0; l < NEVALS; ++l)
        fem3Dtet<Operator<IDEN, FemFix<FEM_P3>>, Operator<IDEN, FemFix<FEM_P3>>, FTraits>(XY1, XY2, XY3, XY4, dfunc, A,
                                                                                          pMem, 5);
    print_state("template IDEN(P3)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P3.getOP(GRAD), P3.getOP(GRAD), dfunc, A, rMem, 5);
    print_state("runtime IDEN(P3)");
    set_start_timers();
        for (int l = 0; l < NEVALS; ++l)
            fem3Dtet<FTraits>(XYZ, P3.getOP(GRAD), P3.getOP(GRAD), dfunc, A, wmem, 5);
    print_state("runtime IDEN(P3) dyn mem");

    EXPECT_EQ(0, 0);
    return;
}

TEST(AniInterface, Operations_IntTet_OnePointTensor){
    using namespace Ani;
    double  XY1p[] = {1, 1, 1, 0, 0, 0},
            XY2p[] = {2, 1, 1, 2, 1, 1},
            XY3p[] = {1, 2, 1, 1, 2, 1},
            XY4p[] = {1, 1, 2, 2, 1, 2};
#ifdef WITH_EIGEN
    using namespace Eigen;
    Matrix3Xd XY1{{XY1p[0], XY1p[0+3]}, {XY1p[1], XY1p[1+3]}, {XY1p[2], XY1p[2+3]}},
              XY2{{XY2p[0], XY2p[0+3]}, {XY2p[1], XY2p[1+3]}, {XY2p[2], XY2p[2+3]}},
              XY3{{XY3p[0], XY3p[0+3]}, {XY3p[1], XY3p[1+3]}, {XY3p[2], XY3p[2+3]}},
              XY4{{XY4p[0], XY4p[0+3]}, {XY4p[1], XY4p[1+3]}, {XY4p[2], XY4p[2+3]}};
//    int fusion_param = XY1.cols();
    MatrixXd A;
    A.resize(12, 20);
    #define COLX(X) X.col(1)
#else
    double Ap[12*20];
    DenseMatrix<> XY1(XY1p+3, 3, 1), XY2(XY2p+3, 3, 1), XY3(XY3p+3, 3, 1), XY4(XY4p+3, 3, 1);
    DenseMatrix<> A(Ap, 12, 20); 
    #define COLX(X) X        
#endif    
    auto dfunc = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void) user_data; (void) iTet;
#ifdef WITH_EIGEN
        Eigen::Map<Eigen::MatrixXd> D(Dmem, Ddims.first, Ddims.second);
#else
        DenseMatrix<> D(Dmem, Ddims.first, Ddims.second);
#endif
        int cnt = 0;
        for (ulong i = 0; i < Ddims.first; ++i)
            for (ulong j = 0; j < Ddims.second; ++j)
                D(i, j) = (i * x[0] + j * x[1] + (i%2)*x[2]) * (cnt++);
        return TENSOR_GENERAL;
    };
    //FemVec is special structure to create cartesian product of fem type
    //for example, FemVec<3, FEM_P1> = FEM_P1 x FEM_P1 x FEM_P1
    fem3Dtet<Operator<GRAD, FemFix<FEM_P3>>, Operator<GRAD, FemVec<3, FEM_P1>>>(COLX(XY1), COLX(XY2), COLX(XY3),
                                                                                COLX(XY4), dfunc, A);
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
    DenseMatrix<> A_exp_m(A_exp_mp.data(), 12, 20), Adm(Admp.data(), 12, 20);
    for (std::size_t i = 0; i < A_exp_m.nRow; ++i) 
        for (std::size_t j = 0; j < A_exp_m.nCol; ++j) 
            A_exp_m(i, j) = At_exp_m[i][j];
    for (std::size_t j = 0; j < Adm.nCol; ++j) 
        for (std::size_t i = 0; i < Adm.nRow; ++i) 
            Adm(i, j) = A(i, j);
    auto norm = [](double a, const DenseMatrix<>& A, double b = 0.0, const DenseMatrix<>& B = DenseMatrix<>(nullptr, 0, 0)){
        return DenseMatrix<>::ScalNorm(a, A, b, B);
    };
    
    EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Adm), 0, 10*(1 + norm(1, Adm))*DBL_EPSILON);
//    Operator<GRAD, FemCom<FemVec<3, FEM_P2>, FemVecT<3, FemFix<FEM_P3>>, FemFix<FEM_P1>>> op;
    //let's specify some tensor traits
    fem3Dtet<Operator<GRAD, FemFix<FEM_P3>>, Operator<GRAD, FemVec<3, FEM_P1>>, DfuncTraits<PerPoint, false>>(
            COLX(XY1), COLX(XY2), COLX(XY3), COLX(XY4), dfunc, A, 5, nullptr);
    for (std::size_t j = 0; j < Adm.nCol; ++j) 
        for (std::size_t i = 0; i < Adm.nRow; ++i) 
            Adm(i, j) = A(i, j); 
    EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Adm), 0, 10*(1 + norm(1, Adm))*DBL_EPSILON); 

    //here Tensor's trait as it is set in fortran version
    fem3Dtet<Operator<GRAD, FemFix<FEM_P3>>, Operator<GRAD, FemVec<3, FEM_P1>>, DfuncTraits<PerTetra, false>>(
            COLX(XY1), COLX(XY2), COLX(XY3), COLX(XY4), dfunc, A, 5);
    for (std::size_t j = 0; j < Adm.nCol; ++j) 
        for (std::size_t i = 0; i < Adm.nRow; ++i) 
            Adm(i, j) = A(i, j);
    EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Adm), 0, 10*(1 + norm(1, Adm))*DBL_EPSILON);

    //here we say to compiler that tensor always return TENSOR_GENERAL
    fem3Dtet<Operator<GRAD, FemFix<FEM_P3>>, Operator<GRAD, FemVec<3, FEM_P1>>, DfuncTraits<TENSOR_GENERAL, false>>(
            COLX(XY1), COLX(XY2), COLX(XY3), COLX(XY4), dfunc, A, 5);
    for (std::size_t j = 0; j < Adm.nCol; ++j) 
        for (std::size_t i = 0; i < Adm.nRow; ++i) 
            Adm(i, j) = A(i, j); 
    EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Adm), 0, 10*(1 + norm(1, Adm))*DBL_EPSILON); 

    //now check memory dependent version of the method
    {
        auto req = fem3Dtet_memory_requirements<Operator<GRAD, FemFix<FEM_P3>>, Operator<GRAD, FemVec<3, FEM_P1>>>(7, 1);
        std::vector<char> raw_mem;
        raw_mem.resize(req.enoughRawSize());
        req.allocateFromRaw(raw_mem.data(), raw_mem.size());
        Adm.SetZero();
        fem3Dtet<Operator<GRAD, FemFix<FEM_P3>>, Operator<GRAD, FemVec<3, FEM_P1>>, DfuncTraits<TENSOR_GENERAL, false>>(
            DenseMatrix<const double>(XY1p+3, 3, 1), DenseMatrix<const double>(XY2p+3, 3, 1), DenseMatrix<const double>(XY3p+3, 3, 1), DenseMatrix<const double>(XY4p+3, 3, 1),
            dfunc, Adm, req, 7
        );
        EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Adm), 0, 10*(1 + norm(1, Adm))*DBL_EPSILON);
    }

    //now check runtime version of the method
    {
        ApplyOp op_r1 = ApplyOp(ApplyOpFromTemplate<GRAD, FemFix<FEM_P3>>());
        ApplyOp op_r2 = ApplyOp(ApplyOpFromTemplate<GRAD, FemVec<3, FEM_P1>>());
        PlainMemoryX<> req;
        std::vector<char> raw_mem;
        req = fem3Dtet_memory_requirements<DfuncTraits<TENSOR_GENERAL, false>>(*op_r1.m_invoker, *op_r2.m_invoker, 5, 1);
        raw_mem.resize(req.enoughRawSize());
        req.allocateFromRaw(raw_mem.data(), raw_mem.size());
        Adm.SetZero();
        Tetra<> XYZ(XY1p+3, XY2p+3, XY3p+3, XY4p+3);
        fem3Dtet<DfuncTraits<TENSOR_GENERAL, false>>(XYZ, *op_r1.m_invoker, *op_r2.m_invoker, dfunc, Adm, req, 5);
        EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Adm), 0, 10*(1 + norm(1, Adm))*DBL_EPSILON);
    }

    auto make_dfunc_const = [](TensorType t = TENSOR_GENERAL){
        return [t](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
            (void)x; (void) user_data; (void) iTet;
    #ifdef WITH_EIGEN
            Eigen::Map<Eigen::MatrixXd> D(Dmem, Ddims.first, Ddims.second);
    #else
            DenseMatrix<> D(Dmem, Ddims.first, Ddims.second);
    #endif
            for (ulong i = 0; i < Ddims.first; ++i)
                for (ulong j = 0; j < Ddims.second; ++j)
                    D(i, j) = (i == j);
            return t;
        };
    };

#ifdef WITH_EIGEN
    A.resize(12, 12);
#else
    A.nRow = A.nCol = 12;
#endif
    coef = 9*160;
    Operator<GRAD, FemVec<3, FEM_P1>> op1;
    double A_exp_m1p[] = {
                160, -240,  -80,  160,    0,    0,    0,    0,    0,    0,    0,    0,
               -240, 1440, -240, -960,    0,    0,    0,    0,    0,    0,    0,    0,
                -80, -240,  400,  -80,    0,    0,    0,    0,    0,    0,    0,    0,
                160, -960,  -80,  880,    0,    0,    0,    0,    0,    0,    0,    0,
                  0,    0,    0,    0,  160, -240,  -80,  160,    0,    0,    0,    0,
                  0,    0,    0,    0, -240, 1440, -240, -960,    0,    0,    0,    0,
                  0,    0,    0,    0,  -80, -240,  400,  -80,    0,    0,    0,    0,
                  0,    0,    0,    0,  160, -960,  -80,  880,    0,    0,    0,    0,
                  0,    0,    0,    0,    0,    0,    0,    0,  160, -240,  -80,  160,
                  0,    0,    0,    0,    0,    0,    0,    0, -240, 1440, -240, -960,
                  0,    0,    0,    0,    0,    0,    0,    0,  -80, -240,  400,  -80,
                  0,    0,    0,    0,    0,    0,    0,    0,  160, -960,  -80,  880
    };
    A_exp_m.Init(A_exp_m1p, 12, 12, 12*12);
    //for full code coverage add the following tests
#define TEST_TRAIT(TT, TTr, CST) {\
        fem3Dtet<decltype(op1), decltype(op1), DfuncTraits<TT, CST> >(XY1m, XY2m, XY3m, XY4m, make_dfunc_const(TTr), Am, 5); \
        EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Am), 0, 10*(1 + norm(1, Am))*DBL_EPSILON);\
        Am.nCol = Am.nRow*Am.nCol; Am.nRow = 1; Am.SetZero();\
        \
        req = fem3Dtet_memory_requirements(op_r, op_r, 5, 1);\
        raw_mem.resize(req.enoughRawSize());\
        req.allocateFromRaw(raw_mem.data(), raw_mem.size());\
        fem3Dtet(XYZ, op_r, op_r, make_dfunc_const(TTr), Am, req, 5);\
        EXPECT_NEAR(norm(1.0/coef, A_exp_m, -1, Am), 0, 10*(1 + norm(1, Am))*DBL_EPSILON);\
        Am.nCol = Am.nRow*Am.nCol; Am.nRow = 1; Am.SetZero();\
    }

    double A_d[30*30] = {1};
    DenseMatrix<double> Am(A_d, 0, 0, 30*30);
    double XY1d[] = {1, 1, 1, 0, 0, 0};
    double XY2d[] = {2, 1, 1, 2 ,1, 1};
    double XY3d[] = {1,2, 1, 1, 2, 1};
    double XY4d[] = {1, 1, 2, 2, 1, 2};
    DenseMatrix<> XY1m(XY1d+3, 3, 1, 3);
    DenseMatrix<> XY2m(XY2d+3, 3, 1, 3);
    DenseMatrix<> XY3m(XY3d+3, 3, 1);
    DenseMatrix<> XY4m(XY4d+3, 3, 1);
    Tetra<> XYZ(XY1d+3, XY2d+3, XY3d+3, XY4d+3);
    ApplyOpFromTemplate<GRAD, FemVec<3, FEM_P1>> op_r;
    PlainMemoryX<> req;
    std::vector<char> raw_mem;

    TEST_TRAIT(PerSelection, TENSOR_GENERAL, false);
    TEST_TRAIT(TENSOR_SYMMETRIC, TENSOR_SYMMETRIC, false);
    TEST_TRAIT(TENSOR_SCALAR, TENSOR_SCALAR, false);
    TEST_TRAIT(TENSOR_NULL, TENSOR_NULL, false);

    TEST_TRAIT(PerPoint, TENSOR_GENERAL, true);
    TEST_TRAIT(PerTetra, TENSOR_GENERAL, true);
    TEST_TRAIT(PerSelection, TENSOR_SCALAR, true);
    TEST_TRAIT(TENSOR_GENERAL, TENSOR_GENERAL, true);
    TEST_TRAIT(TENSOR_SYMMETRIC, TENSOR_SYMMETRIC, true);
    TEST_TRAIT(TENSOR_SCALAR, TENSOR_SCALAR, true);
    TEST_TRAIT(TENSOR_NULL, TENSOR_NULL, true);

#undef TEST_TRAIT
#undef COLX
}

TEST(AniInterface, Operations_IntTet_FusiveTensor){
    using namespace Ani;
    double  XY1p[] = {0, 0, 0},
            XY2p[] = {2, 1, 1},
            XY3p[] = {1, 2, 1},
            XY4p[] = {2, 1, 2},
            Ap[30],
            Bp[30] = {-1, -1, -1, -1, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4};
    DenseMatrix<> XY1(XY1p, 3, 1), XY2(XY2p, 3, 1), XY3(XY3p, 3, 1), XY4(XY4p, 3, 1), A(Ap, 30, 1), B(Bp, 30, 1);
    double mu = 9*160;
    auto scal_DFunc = [mu](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet){
        (void) Ddims; (void) user_data; (void) iTet;
        Dmem[0] = x[0]*x[0];
        return TENSOR_SCALAR;
    };
    auto scal_DFunc_fuse = [scal_DFunc](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem) -> TensorType{
        for (std::size_t r = 0; r < mem.f; ++r)
            for (std::size_t n = 0; n < mem.q; ++n){
                DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
                scal_DFunc({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]}, 
                            Dloc.data, Ddims, user_data, r);
            }
        return TENSOR_SCALAR;
    };
    //template memoryless version
    fem3Dtet<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraitsFusive<>>(XY1, XY2, XY3, XY4, scal_DFunc_fuse, A, 4);
    fem3Dtet<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<PerPoint, false>>(XY1, XY2, XY3, XY4, scal_DFunc, B, 4);
    EXPECT_NEAR(A.ScalNorm(1, A, -1, B), 0, 10*(1 + A.ScalNorm(1, A))*DBL_EPSILON);
    std::vector<char> raw_mem;
    {   //template external memory version
        PlainMemory<> req;
        req = fem3Dtet_memory_requirements<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>>(4, 1);
        raw_mem.resize(req.enoughRawSize());
        req.allocateFromRaw(raw_mem.data(), raw_mem.size());
        A.SetZero();
        fem3Dtet<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraitsFusive<>>(
            XY1, XY2, XY3, XY4, scal_DFunc_fuse, A, req, 4
        );
        EXPECT_NEAR(A.ScalNorm(1, A, -1, B), 0, 10*(1 + A.ScalNorm(1, A))*DBL_EPSILON);
    }
    {   //runtime version
        ApplyOp op_r = ApplyOp(ApplyOpFromTemplate<IDEN, FemFix<FEM_P1>>());
        PlainMemoryX<> req;
        req = fem3Dtet_memory_requirements<DfuncTraitsFusive<>>(*op_r.m_invoker, *op_r.m_invoker, 4, 1);
        raw_mem.resize(req.enoughRawSize());
        req.allocateFromRaw(raw_mem.data(), raw_mem.size());
        Tetra<> XYZ(XY1p, XY2p, XY3p, XY4p);
        A.SetZero();
        fem3Dtet<DfuncTraitsFusive<>>(XYZ, *op_r.m_invoker, *op_r.m_invoker, scal_DFunc_fuse, A, req, 4);
        EXPECT_NEAR(A.ScalNorm(1, A, -1, B), 0, 10*(1 + A.ScalNorm(1, A))*DBL_EPSILON);
    }
}

TEST(AniInterface, RhsEval){
    using namespace Ani;
    double  XY1p[] = {0, 0, 0},
            XY2p[] = {2, 1, 1},
            XY3p[] = {1, 2, 1},
            XY4p[] = {2, 1, 2},
            Ap[30],
            Bp[30] = {-1, -1, -1, -1, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4, -1, -1, -1, -1, 4, 4, 4, 4, 4, 4};
    DenseMatrix<> XY1(XY1p, 3, 1), XY2(XY2p, 3, 1), XY3(XY3p, 3, 1), XY4(XY4p, 3, 1), A(Ap, 30, 1), B(Bp, 30, 1);
    using UFem = FemVec<3, FEM_P2>;
    using NoOp = Operator<IDEN, FemFix<FEM_P0>>;
    double mu = 40;
    int k = 1;
    auto scal_Dfunc = [mu, &k](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void)x; (void) user_data; (void) iTet;
        std::fill(Dmem, Dmem + Ddims.first*Ddims.second, mu*k);
        return TENSOR_SCALAR;
    };
    auto null_Dfunc = [mu](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void)x; (void) user_data; (void) iTet;
        std::fill(Dmem, Dmem + Ddims.first*Ddims.second, 1);
        return TENSOR_NULL;
    };
    // int (mu, mu, mu) * \vec{\phi}_i dx
    #define TEST_TRAIT(TT, CST) {\
    fem3Dtet<NoOp, Operator<IDEN, UFem>, DfuncTraits<TT, CST> >(XY1, XY2, XY3, XY4, scal_Dfunc, A, 2); \
    EXPECT_NEAR(A.ScalNorm(1, A, -k, B), 0, 10*(1 + A.ScalNorm(1, A))*DBL_EPSILON);\
    ++k;\
    }
    TEST_TRAIT(PerPoint, false);
    TEST_TRAIT(PerTetra, false);
    TEST_TRAIT(PerSelection, false);
    TEST_TRAIT(TENSOR_SCALAR, false);
    TEST_TRAIT(PerPoint, true);
    TEST_TRAIT(PerTetra, true);
    TEST_TRAIT(PerSelection, true);
    TEST_TRAIT(TENSOR_SCALAR, true);
    #undef TEST_TRAIT

    // int (1, 1, 1) * \vec{\phi}_i dx
    #define TEST_TRAIT(TT, CST) {\
    fem3Dtet<NoOp, Operator<IDEN, UFem>, DfuncTraits<TT, CST> >(XY1, XY2, XY3, XY4, null_Dfunc, A, 2); \
    EXPECT_NEAR(A.ScalNorm(1, A, -1/mu, B), 0, 10*(1 + A.ScalNorm(1, A))*DBL_EPSILON);\
    }
    TEST_TRAIT(PerPoint, false);
    TEST_TRAIT(PerTetra, false);
    TEST_TRAIT(PerSelection, false);
    TEST_TRAIT(TENSOR_NULL, false);
    TEST_TRAIT(PerPoint, true);
    TEST_TRAIT(PerTetra, true);
    TEST_TRAIT(PerSelection, true);
    TEST_TRAIT(TENSOR_NULL, true);
    #undef TEST_TRAIT
}