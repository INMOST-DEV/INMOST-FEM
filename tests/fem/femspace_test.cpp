#include <gtest/gtest.h>
#include <algorithm>
#include "anifem++/fem/fem_space.h"
#include "anifem++/fem/spaces/spaces.h"
#include "anifem++/fem/operations/operations.h"

TEST(AniInterface, FemSpace){
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
        auto P123dofs = std::vector<double>{0,2,1,2, 0,2,5,5,0,0,0,3.25,3.25,4.5, 0,3,3,3,0,0,0,0,0,0,29./9,29./9,3,3,29./9,29./9,0,29./9,0,0 };
        check_interpolation(P1*P2*P3, f, sp, P123dofs, false);
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

        auto compl_vec = mix_f({{fp0, 1}, {fp1, 1}, {fp2_1, 1}, {fp2, 1}, {fp1, 1}, {fp0, 1}, {f_rot, 3}});
        auto dcompl_vec = mix_df({{dfp0, 3}, {dfp1, 3}, {dfp2_1, 3}, {dfp2, 3}, {dfp1, 3}, {dfp0, 3}, {df_rot, 9}});
        test_evaluates(IDEN, P1*(P2^3)*(MINI1^2)*RT0, compl_vec, compl_vec);
        test_evaluates(GRAD, P1*(P2^3)*(MINI1^2)*RT0, compl_vec, dcompl_vec);

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