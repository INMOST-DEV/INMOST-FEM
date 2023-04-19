//
// Created by Liogky Alexey on 21.02.2023.
//

#include <gtest/gtest.h>
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include <chrono>

TEST(AniInterface, Operations_dc_on_dof){
    using namespace Ani;
    {
        const unsigned N = 4;
        int dof_id = 2;
        double bc = 1;
        std::array<double, N> F, F_exp;
        std::array<double, N*N> A, A_exp;

        for (unsigned i = 0; i < F.size(); ++i){
            F[i] = i+1;
            for (unsigned j = 0; j < F.size(); ++j)
                A[i+N*j] = (i+1)*(j+1);    
        }
        std::copy(A.begin(), A.end(), A_exp.begin());
        std::copy(F.begin(), F.end(), F_exp.begin());

        for (unsigned i = 0; i < F.size(); ++i)
            A_exp[dof_id+N*i] = A_exp[i + N*dof_id] = 0;
        A_exp[dof_id + N*dof_id] = 1;
        for (unsigned i = 0; i < N; ++i)
            F_exp[i] -= A[i + N*dof_id] * bc;
        F_exp[dof_id] = bc;

        DenseMatrix<> Am(A.data(), N, N), Fm(F.data(), N, 1), 
            Ame(A_exp.data(), N, N), Fme(F_exp.data(), N, 1);
        applyDir(Am, Fm, dof_id, bc);

        EXPECT_NEAR(Fm.ScalNorm(1, Fm, -1, Fme), 0, 10*(1 + Fm.ScalNorm(1, Fme))*DBL_EPSILON);
        EXPECT_NEAR(Am.ScalNorm(1, Am, -1, Ame), 0, 10*(1 + Am.ScalNorm(1, Ame))*DBL_EPSILON);
    }   

    {
        double Ap[] = {
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
        double Rp[12] = { 1.0e5, 1.1e5, 1.2e5, 1.3e5, 1.4e5, 1.5e5, 1.6e5, 1.7e5, 1.8e5, 1.9e5, 2.0e5, 2.1e5 };
        double Ip[] = { 1, 1,  1,
                        1, -1, 1,
                        1, 0, -2 };
        double bc_p[] = {100, 200, 300};                
        double Bp[12*12], CRp[12], memp[12*12];
        std::copy(Ap, Ap+12*12, Bp); std::copy(Rp, Rp+12, CRp);
        DenseMatrix<> Am(Ap, 12, 12, 12*12), Bm(Bp, 12, 12), Rhs(CRp, 12, 1);
        DenseMatrix<> Im(Ip, 3, 3);
        ArrayView<> bc(bc_p, 3);
        for (std::size_t i = 0; i < Im.nRow; ++i){
            double s = 0;
            for (int j = 0; j < 3; ++j)
                s += Im(i, j)*Im(i, j);
            s = sqrt(s);
            if (s > 1e-15)
                for (int j = 0; j < 3; ++j)
                    Im(i, j) /= s;
        }
        ArrayView<> mem(memp, 12*12);
        uint dof_id[] = {1, 5, 9};
        uint dc_orth[] = {1, 2};
        double Bexp_p[] = {
            160, -240/sqrt(3), -80,  160,   0,          0,   0,    0,   0,          0,   0,    0, 
            -80, 1440/sqrt(3), -80, -320, -80,  1/sqrt(2), -80, -320, -80,  1/sqrt(6), -80, -320, 
            -80, -240/sqrt(3), 400,  -80,   0,          0,   0,    0,   0,          0,   0,    0, 
            160, -960/sqrt(3), -80,  880,   0,          0,   0,    0,   0,          0,   0,    0, 
              0, -240/sqrt(3),   0,    0, 160,          0, -80,  160,   0,          0,   0,    0, 
            -80, 1440/sqrt(3), -80, -320, -80, -1/sqrt(2), -80, -320, -80,  1/sqrt(6), -80, -320, 
              0, -240/sqrt(3),   0,    0, -80,          0, 400,  -80,   0,          0,   0,    0, 
              0, -960/sqrt(3),   0,    0, 160,          0, -80,  880,   0,          0,   0,    0, 
              0, -240/sqrt(3),   0,    0,   0,          0,   0,    0, 160,          0, -80,  160, 
            -80, 1440/sqrt(3), -80, -320, -80,          0, -80, -320, -80, -2/sqrt(6), -80, -320, 
              0, -240/sqrt(3),   0,    0,   0,          0,   0,    0, -80,          0, 400,  -80, 
              0, -960/sqrt(3),   0,    0,   0,          0,   0,    0, 160,          0, -80,  880};
        DenseMatrix<> Bexp(Bexp_p, 12, 12);      
        applyVectorDirMatrix(Bm, dof_id, Im, mem, 2, dc_orth);
        EXPECT_NEAR(Bexp.ScalNorm(1, Bm, -1, Bexp), 0, 1e+3*(1 + Bexp.ScalNorm(1, Bexp))*DBL_EPSILON);
        std::copy(Ap, Ap+12*12, Bp);
        
        double Bexp_p1[] = {
            160,         0, -80,  160,   0,          0,   0,    0,    0,  -240/sqrt(6),    0,    0, 
            -40, 1/sqrt(3), -40, -160, -40,  1/sqrt(2), -40, -160,   80,  1440/sqrt(6),   80,  320, 
            -80,         0, 400,  -80,   0,          0,   0,    0,    0,  -240/sqrt(6),    0,    0, 
            160,         0, -80,  880,   0,          0,   0,    0,    0,  -960/sqrt(6),    0,    0, 
              0,         0,   0,    0, 160,          0, -80,  160,    0,  -240/sqrt(6),    0,    0, 
            -40, 1/sqrt(3), -40, -160, -40, -1/sqrt(2), -40, -160,   80,  1440/sqrt(6),   80,  320, 
              0,         0,   0,    0, -80,          0, 400,  -80,    0,  -240/sqrt(6),    0,    0, 
              0,         0,   0,    0, 160,          0, -80,  880,    0,  -960/sqrt(6),    0,    0, 
              0,         0,   0,    0,   0,          0,   0,    0,  160,   480/sqrt(6),  -80,  160, 
             80, 1/sqrt(3),  80,  320,  80,          0,  80,  320, -160, -2880/sqrt(6), -160, -640, 
              0,         0,   0,    0,   0,          0,   0,    0,  -80,   480/sqrt(6),  400,  -80, 
              0,         0,   0,    0,   0,          0,   0,    0,  160,  1920/sqrt(6),  -80,  880
        };
        Bexp.Init(Bexp_p1, 12, 12, 12*12);
        applyVectorDirMatrix(Bm, dof_id, Im, mem, 2);
        EXPECT_NEAR(Bexp.ScalNorm(1, Bm, -1, Bexp), 0, 1e+3*(1 + Bexp.ScalNorm(1, Bexp))*DBL_EPSILON);
        std::copy(Ap, Ap+12*12, Bp);

        Bexp.Init(Bexp_p, 12, 12, 12*12);
        applyVectorDir(Bm, Rhs, dof_id, Im, bc, mem, 2, dc_orth);
        EXPECT_NEAR(Bexp.ScalNorm(1, Bm, -1, Bexp), 0, 1e+3*(1 + Bexp.ScalNorm(1, Bexp))*DBL_EPSILON);
        double Rexp_p1[12] = {
            1.0e5 - (-240) * ( 1/sqrt(2)*100 + 1/sqrt(6)*200 ),
            (1.1e5 + 1.5e5 + 1.9e5)/sqrt(3),
            1.2e5 - (-240) * ( 1/sqrt(2)*100 + 1/sqrt(6)*200 ),
            1.3e5 - (-960) * ( 1/sqrt(2)*100 + 1/sqrt(6)*200 ),
            1.4e5 - (-240) * ( -1/sqrt(2)*100 + 1/sqrt(6)*200 ),
            100,
            1.6e5 - (-240) * ( -1/sqrt(2)*100 + 1/sqrt(6)*200 ),
            1.7e5 - (-960) * ( -1/sqrt(2)*100 + 1/sqrt(6)*200 ),
            1.8e5 - (-240) * ( -0/sqrt(2)*100 + -2/sqrt(6)*200 ),
            200,
            2.0e5 - (-240) * ( -0/sqrt(2)*100 + -2/sqrt(6)*200 ),
            2.1e5 - (-960) * ( -0/sqrt(2)*100 + -2/sqrt(6)*200 ),
        };
        DenseMatrix<> Rexp(Rexp_p1, 12, 1);
        EXPECT_NEAR(Rexp.ScalNorm(1, Rhs, -1, Rexp), 0, 1e+2*(1 + Rexp.ScalNorm(1, Rexp))*DBL_EPSILON);
        std::copy(Ap, Ap+12*12, Bp); std::copy(Rp, Rp+12, CRp);

        applyVectorDirResidual(Rhs, dof_id, Im, mem, 2, dc_orth);
        double Rexp_p2[12] = {
            1.0e5,
            (1.1e5 + 1.5e5 + 1.9e5)/sqrt(3),
            1.2e5,
            1.3e5,
            1.4e5,
            0,
            1.6e5,
            1.7e5,
            1.8e5,
            0,
            2.0e5,
            2.1e5,
        };
        Rexp.Init(Rexp_p2, 12, 1, 12);
        EXPECT_NEAR(Rexp.ScalNorm(1, Rhs, -1, Rexp), 0, 1e+2*(1 + Rexp.ScalNorm(1, Rexp))*DBL_EPSILON);
        std::copy(Ap, Ap+12*12, Bp); std::copy(Rp, Rp+12, CRp);
    }
}