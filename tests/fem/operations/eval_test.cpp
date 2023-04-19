//
// Created by Liogky Alexey on 21.02.2023.
//

#include <gtest/gtest.h>
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include <chrono>

TEST(AniInterface, Operations_Eval){
    using namespace Ani;
    using std::array;
    double XYp[] = {0, 0, 0, 2, 1, 1, 1, 2, 1, 2, 1, 2};
    using UFem = FemVec<3, FEM_P2>;
    constexpr auto nfa = Operator<IDEN, UFem>::Nfa::value, dim = Operator<IDEN, UFem>::Dim::value;
    auto u_exact = [](const array<double, 3>& x)->array<double, 3>{
        return {x[0], x[1]*x[1] + x[2]*x[2], 1 + x[0]*x[1]};
    };
    auto grad_u_exact = [](const array<double, 3>& x)->array<double, 3*3>{
        return {1, 0, x[1],
                0, 2*x[1], x[0],
                0, 2*x[2], 0};
    };
    auto div_u_exact = [](const array<double, 3>& x)->array<double, 1>{
        return {1 + 2*x[1]};
    };
    auto curl_u_exact = [](const array<double, 3>& x)->array<double, 3>{
        return {x[0] - 2*x[2], -x[1], 0};
    };
    double udofs[nfa];
    std::vector<char> wmem;
    for (int k = 0, bnfa = Operator<IDEN, UFem::Base>::Nfa::value; k < bnfa; ++k){
        array<double, 3> x;
        DOF_coord<UFem::Base>::at(k, XYp, x.data());
        auto u = u_exact(x);
        for (int d = 0; d < dim; ++d)
            udofs[k + d*bnfa] = u[d]; 
    }
    auto u_eval = [udofs, XYp](const array<double, 3>& x)->array<double, 3>{
        array<double, dim> res;
        ArrayView<> u(res.data(), dim);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<IDEN, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, x.data(), dofs, u);
        return res;
    };
    auto grad_u_eval = [udofs, XYp](const array<double, 3>& x)->array<double, 3*3>{
        array<double, dim*3> res;
        ArrayView<> u(res.data(), dim*3);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<GRAD, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, x.data(), dofs, u);
        //transform to came in standard view
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < i; ++j)
                std::swap(res[i+3*j], res[j+3*i]);
        return res;
    };
    auto div_u_eval = [udofs, XYp](const array<double, 3>& x)->array<double, 1>{
        array<double, 1> res;
        ArrayView<> u(res.data(), 1);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<DIV, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, ArrayView<const double>(x.data(), x.size()), dofs, u);
        return res;
    };
    auto curl_u_eval = [udofs, XYp](const array<double, 3>& x)->array<double, 3>{
        array<double, 3> res;
        ArrayView<> u(res.data(), 3);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<CURL, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, x.data(), dofs, u);
        return res;
    };
    std::array<double, 4> lmb{0.1, 0.2, 0.3, 0.4};
    array<double, 3> x{0};
    for (int i = 0; i < 4; ++i) 
        for (int k = 0; k < 3; ++k) 
            x[k] += lmb[i] * XYp[k + 3*i];

    {
        auto ux_eval = u_eval(x), ux_exact = u_exact(x);
        DenseMatrix<> ux_e(ux_eval.data(), dim, 1), ux_a(ux_exact.data(), dim, 1);
        EXPECT_NEAR(ux_e.ScalNorm(1, ux_e, -1, ux_a), 0, 10*(1 + ux_a.ScalNorm(1, ux_a))*DBL_EPSILON);
    }
    {
        auto gr_ux_eval = grad_u_eval(x), gr_ux_exact = grad_u_exact(x);
        DenseMatrix<> gr_ux_e(gr_ux_eval.data(), dim, 3), gr_ux_a(gr_ux_exact.data(), dim, 3);
        EXPECT_NEAR(gr_ux_e.ScalNorm(1, gr_ux_e, -1, gr_ux_a), 0, 10*(1 + gr_ux_a.ScalNorm(1, gr_ux_a))*DBL_EPSILON);
    }
    {
        auto div_ux_eval = div_u_eval(x), div_ux_exact = div_u_exact(x);
        DenseMatrix<> div_ux_e(div_ux_eval.data(), 1, 1), div_ux_a(div_ux_exact.data(), 1, 1);
        EXPECT_NEAR(div_ux_e.ScalNorm(1, div_ux_e, -1, div_ux_a), 0, 10*(1 + div_ux_a.ScalNorm(1, div_ux_a))*DBL_EPSILON);
    }
    {
        auto curl_ux_eval = curl_u_eval(x), curl_ux_exact = curl_u_exact(x);
        DenseMatrix<> curl_ux_e(curl_ux_eval.data(), 3, 1), curl_ux_a(curl_ux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }
    
    auto u_eval_m = [udofs, XYp, &wmem](const array<double, 3>& x)->array<double, 3>{
        auto req = fem3DapplyX_memory_requirements<Operator<IDEN, UFem>>(1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, dim> res;
        ArrayView<> u(res.data(), dim);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<IDEN, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, ArrayView<const double>(x.data(), x.size()), dofs, u, req);
        return res;
    };
    auto grad_u_eval_m = [udofs, XYp, &wmem](const array<double, 3>& x)->array<double, 3*3>{
        auto req = fem3DapplyX_memory_requirements<Operator<GRAD, UFem>>(1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, dim*3> res;
        ArrayView<> u(res.data(), dim*3);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<GRAD, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, ArrayView<const double>(x.data(), x.size()), dofs, u, req);
        //transform to came in standard view
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < i; ++j)
                std::swap(res[i+3*j], res[j+3*i]);
        return res;
    };
    auto div_u_eval_m = [udofs, XYp, &wmem](const array<double, 3>& x)->array<double, 1>{
        auto req = fem3DapplyX_memory_requirements<Operator<DIV, UFem>>(1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 1> res;
        ArrayView<> u(res.data(), 1);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<DIV, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, ArrayView<const double>(x.data(), x.size()), dofs, u, req);
        return res;
    };
    auto curl_u_eval_m = [udofs, XYp, &wmem](const array<double, 3>& x)->array<double, 3>{
        auto req = fem3DapplyX_memory_requirements<Operator<CURL, UFem>>(1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 3> res;
        ArrayView<> u(res.data(), 3);
        ArrayView<> dofs(const_cast<double*>(udofs+0), nfa);
        fem3DapplyX<Operator<CURL, UFem>>(XYp+0, XYp+3, XYp+6, XYp+9, ArrayView<const double>(x.data(), x.size()), dofs, u, req);
        return res;
    };
    {
        auto ux_eval = u_eval_m(x), ux_exact = u_exact(x);
        DenseMatrix<> ux_e(ux_eval.data(), dim, 1), ux_a(ux_exact.data(), dim, 1);
        EXPECT_NEAR(ux_e.ScalNorm(1, ux_e, -1, ux_a), 0, 10*(1 + ux_a.ScalNorm(1, ux_a))*DBL_EPSILON);
    }
    {
        auto gr_ux_eval = grad_u_eval_m(x), gr_ux_exact = grad_u_exact(x);
        DenseMatrix<> gr_ux_e(gr_ux_eval.data(), dim, 3), gr_ux_a(gr_ux_exact.data(), dim, 3);
        EXPECT_NEAR(gr_ux_e.ScalNorm(1, gr_ux_e, -1, gr_ux_a), 0, 10*(1 + gr_ux_a.ScalNorm(1, gr_ux_a))*DBL_EPSILON);
    }
    {
        auto div_ux_eval = div_u_eval_m(x), div_ux_exact = div_u_exact(x);
        DenseMatrix<> div_ux_e(div_ux_eval.data(), 1, 1), div_ux_a(div_ux_exact.data(), 1, 1);
        EXPECT_NEAR(div_ux_e.ScalNorm(1, div_ux_e, -1, div_ux_a), 0, 10*(1 + div_ux_a.ScalNorm(1, div_ux_a))*DBL_EPSILON);
    }
    {
        auto curl_ux_eval = curl_u_eval_m(x), curl_ux_exact = curl_u_exact(x);
        DenseMatrix<> curl_ux_e(curl_ux_eval.data(), 3, 1), curl_ux_a(curl_ux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }

    Tetra<const double> XYZ(XYp+0, XYp+3, XYp+6, XYp+9);
    auto u_eval_r = [udofs, XYZ, &wmem](const array<double, 3>& x)->array<double, 3>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<IDEN, UFem>());
        auto req = fem3DapplyX_memory_requirements(*op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, dim> res;
        DenseMatrix<> u(res.data(), dim, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyX<>(XYZ, ArrayView<const double>(x.data(), x.size()), dofs, *op.m_invoker, u, req);
        return res;
    };
    auto grad_u_eval_r = [udofs, XYZ, &wmem](const array<double, 3>& x)->array<double, 3*3>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<GRAD, UFem>());
        auto req = fem3DapplyX_memory_requirements(*op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, dim*3> res;
        DenseMatrix<> u(res.data(), dim*3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyX<>(XYZ, ArrayView<const double>(x.data(), x.size()), dofs, *op.m_invoker, u, req);
        //transform to came in standard view
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < i; ++j)
                std::swap(res[i+3*j], res[j+3*i]);
        return res;
    };
    auto div_u_eval_r = [udofs, XYZ, &wmem](const array<double, 3>& x)->array<double, 1>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<DIV, UFem>());
        auto req = fem3DapplyX_memory_requirements(*op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 1> res;
        DenseMatrix<> u(res.data(), 1, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyX<>(XYZ, ArrayView<const double>(x.data(), x.size()), dofs, *op.m_invoker, u, req);
        return res;
    };
    auto curl_u_eval_r = [udofs, XYZ, &wmem](const array<double, 3>& x)->array<double, 3>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<CURL, UFem>());
        auto req = fem3DapplyX_memory_requirements(*op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 3> res;
        DenseMatrix<> u(res.data(), 3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyX<>(XYZ, ArrayView<const double>(x.data(), x.size()), dofs, *op.m_invoker, u, req);
        return res;
    };
    {
        auto ux_eval = u_eval_r(x), ux_exact = u_exact(x);
        DenseMatrix<> ux_e(ux_eval.data(), dim, 1), ux_a(ux_exact.data(), dim, 1);
        EXPECT_NEAR(ux_e.ScalNorm(1, ux_e, -1, ux_a), 0, 10*(1 + ux_a.ScalNorm(1, ux_a))*DBL_EPSILON);
    }
    {
        auto gr_ux_eval = grad_u_eval_r(x), gr_ux_exact = grad_u_exact(x);
        DenseMatrix<> gr_ux_e(gr_ux_eval.data(), dim, 3), gr_ux_a(gr_ux_exact.data(), dim, 3);
        EXPECT_NEAR(gr_ux_e.ScalNorm(1, gr_ux_e, -1, gr_ux_a), 0, 10*(1 + gr_ux_a.ScalNorm(1, gr_ux_a))*DBL_EPSILON);
    }
    {
        auto div_ux_eval = div_u_eval_r(x), div_ux_exact = div_u_exact(x);
        DenseMatrix<> div_ux_e(div_ux_eval.data(), 1, 1), div_ux_a(div_ux_exact.data(), 1, 1);
        EXPECT_NEAR(div_ux_e.ScalNorm(1, div_ux_e, -1, div_ux_a), 0, 10*(1 + div_ux_a.ScalNorm(1, div_ux_a))*DBL_EPSILON);
    }
    {
        auto curl_ux_eval = curl_u_eval_r(x), curl_ux_exact = curl_u_exact(x);
        DenseMatrix<> curl_ux_e(curl_ux_eval.data(), 3, 1), curl_ux_a(curl_ux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }

    auto curl_u_eval_l = [udofs, XYp](const array<double, 4>& lmb)->array<double, 3>{
        array<double, 3> res;
        DenseMatrix<> u(res.data(), 3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        using DMCST = DenseMatrix<const double>;
        fem3DapplyL<Operator<CURL, UFem>>(DMCST(XYp+0, 3, 1), DMCST(XYp+3, 3, 1), DMCST(XYp+6, 3, 1), DMCST(XYp+9, 3, 1), ArrayView<>(const_cast<double*>(lmb.data()), 4), dofs, u);
        return res;
    };
    auto curl_u_eval_ml = [udofs, XYp, &wmem](const array<double, 4>& lmb)->array<double, 3>{
         auto req = fem3DapplyL_memory_requirements<Operator<CURL, UFem>>(1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 3> res;
        DenseMatrix<> u(res.data(), 3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        using DMCST = DenseMatrix<const double>;
        fem3DapplyL<Operator<CURL, UFem>>(DMCST(XYp+0, 3, 1), DMCST(XYp+3, 3, 1), DMCST(XYp+6, 3, 1), DMCST(XYp+9, 3, 1), ArrayView<>(const_cast<double*>(lmb.data()), 4), dofs, u, req);
        return res;
    };
    auto curl_u_eval_rl = [udofs, XYZ, &wmem](const array<double, 4>& lmb)->array<double, 3>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<CURL, UFem>());
        auto req = fem3DapplyX_memory_requirements(*op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 3> res;
        DenseMatrix<> u(res.data(), 3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyL<>(XYZ, ArrayView<>(const_cast<double*>(lmb.data()), 4), dofs, *op.m_invoker, u, req);
        return res;
    };
    {
        auto curl_ux_eval = curl_u_eval_l(lmb), curl_ux_exact = curl_u_exact(x);
        DenseMatrix<> curl_ux_e(curl_ux_eval.data(), 3, 1), curl_ux_a(curl_ux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }
    {
        auto curl_ux_eval = curl_u_eval_ml(lmb), curl_ux_exact = curl_u_exact(x);
        DenseMatrix<> curl_ux_e(curl_ux_eval.data(), 3, 1), curl_ux_a(curl_ux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }
    {
        auto curl_ux_eval = curl_u_eval_rl(lmb), curl_ux_exact = curl_u_exact(x);
        DenseMatrix<> curl_ux_e(curl_ux_eval.data(), 3, 1), curl_ux_a(curl_ux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }

    auto Dtensor = [](const std::array<double, 3>& x, double* Dmem, std::pair<ulong, ulong> Ddims, void* user_data, int iTet)->TensorType{
        (void) user_data; (void) iTet;
        if (Ddims.first != 3 || Ddims.second != 3 )
            throw std::runtime_error("Error in expected tensor sizes");
        DenseMatrix<> D(Dmem, 3, 3);
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++ i)
                D(i, j) = x[j%3] + i/10.0;
        return TENSOR_GENERAL;        
    };
    auto curl_Du_exact = [curl_u_exact, Dtensor](std::array<double, 3> x)->std::array<double, 3>{
        std::array<double, 3> res;
        auto curl_u = curl_u_exact(x);
        std::array<double, 3*3> Dd;
        Dtensor(x, Dd.data(), {3,3}, nullptr, 0);
        DenseMatrix<> D(Dd.data(), 3, 3);
        for (int i = 0; i < 3; ++i)
            res[i] = D(i, 0) * curl_u[0] + D(i, 1) * curl_u[1] + D(i, 2) * curl_u[2];
        return res;    
    };
    auto curl_Du_rl = [udofs, XYZ, &wmem, Dtensor](const array<double, 4>& lmb)->array<double, 3>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<CURL, UFem>());
        auto req = fem3DapplyL_memory_requirements(3, *op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 3> res;
        DenseMatrix<> u(res.data(), 3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyL<>(XYZ, ArrayView<>(const_cast<double*>(lmb.data()), 4), dofs, *op.m_invoker, 3, Dtensor, u, req);
        return res;
    };
    auto curl_Du_r = [udofs, XYZ, &wmem, Dtensor](const array<double, 3>& x)->array<double, 3>{
        ApplyOp op = ApplyOp(ApplyOpFromTemplate<CURL, UFem>());
        auto req = fem3DapplyX_memory_requirements(3, *op.m_invoker, 1);
        wmem.resize(req.enoughRawSize());
        req.allocateFromRaw(wmem.data(), wmem.size());
        array<double, 3> res;
        DenseMatrix<> u(res.data(), 3, 1);
        DenseMatrix<> dofs(const_cast<double*>(udofs+0), nfa, 1);
        fem3DapplyX<>(XYZ, ArrayView<const double>(x.data(), 3), dofs, *op.m_invoker, 3, Dtensor, u, req);
        return res;
    };
    {
        auto curl_Dux_eval = curl_Du_rl(lmb), curl_Dux_exact = curl_Du_exact(x);
        DenseMatrix<> curl_ux_e(curl_Dux_eval.data(), 3, 1), curl_ux_a(curl_Dux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }
    {
        auto curl_Dux_eval = curl_Du_r(x), curl_Dux_exact = curl_Du_exact(x);
        DenseMatrix<> curl_ux_e(curl_Dux_eval.data(), 3, 1), curl_ux_a(curl_Dux_exact.data(), 3, 1);
        EXPECT_NEAR(curl_ux_e.ScalNorm(1, curl_ux_e, -1, curl_ux_a), 0, 10*(1 + curl_ux_a.ScalNorm(1, curl_ux_a))*DBL_EPSILON);
    }
}