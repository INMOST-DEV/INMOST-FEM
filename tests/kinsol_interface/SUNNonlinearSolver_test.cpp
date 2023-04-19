//
// Created by Liogky Alexey on 24.10.2022.
//
#include <gtest/gtest.h>
#include "anifem++/kinsol_interface/SUNNonlinearSolver.h"

TEST(NonLinSolvers, KinSol){
    INMOST::Solver::Initialize(nullptr, nullptr, nullptr);
    {
        //find x = arg min g(x)
        //    g(x) = 1/2 (x - b)^2 + 1/4 (x - b)^4
        //    b = (1, 1, ..., 1), x_0 = (0, 0, ..., 0)
        //
        //    F(x) = ∇ g(x) = (1 + (x - b)^2)*(x - b)
        //    J(x) = ∇ F(x) = (1 + (x - b)^2)I + 2 (x - b) (x - b)^T
        //    x = Solution of F(x) = 0
        int dim = 10;
        INMOST::Sparse::Vector F("F", 0, dim), b("b", 0, dim), x("x", 0, dim);
        INMOST::Sparse::Matrix J("J", 0, dim);
        for (int i = 0; i < dim; ++i) b[i] = 1;
        for (int i = 0; i < dim; ++i) x[i] = 0; //< set initial guess
        auto compF = [dim, &b](INMOST::Sparse::Vector& x, INMOST::Sparse::Vector& F)->int{
            double coef = 1;
            for (int i = 0; i < dim; ++i) coef += (x[i] - b[i])*(x[i] - b[i]);
            for (int i = 0; i < dim; ++i) F[i] = coef*(x[i] - b[i]);
            return 0;
        };
        auto compJ = [dim, &b](INMOST::Sparse::Vector& x, INMOST::Sparse::Matrix& J)->int{
            double coef = 1;
            for (int i = 0; i < dim; ++i) coef += (x[i] - b[i])*(x[i] - b[i]);
            for (int i = 0; i < dim; ++i) 
                for (int j = 0; j < dim; ++j)
                    J[i][j] = coef*(i==j ? 1 : 0) + 2*(x[i] - b[i])*(x[j] - b[j]);
            return 0;
        };
        INMOST::Solver ls(INMOST::Solver::INNER_ILU2);

        SUNNonlinearSolver s(ls, J, x);
        s.SetParameterReal("MaxNewtonStep", sqrt(dim));
        s.SetParameterReal("ScaledSteptol", 1e-5);
        s.SetParameterReal("FuncNormTol", 1e-8);
        s.SetAssemblerRHS(compF).SetAssemblerMAT(compJ).Init(); 
        EXPECT_TRUE(s.Solve());
        double dif_nrm = 0;
        for (int i = 0; i < dim; ++i) dif_nrm += (x[i] - b[i])*(x[i] - b[i]);
        EXPECT_NEAR(dif_nrm, 0, 1e-7);
    }
    INMOST::Solver::Finalize();
}