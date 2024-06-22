#include "anifem++/autodiff/cauchy_strain_autodiff.h"
#include "gtest/gtest.h"
#include <chrono> 

TEST(AutoDiffTest, TensorAlgebra){
    using namespace Ani;
    Mtx3D<> grU(
        std::array<double, 9>{
            0.1, 0.2, 0.3,
            0.35, 0.40, 0.45,
            0.5, 0.7, 0.6
        }
    );
    SymMtx3D<> I = SymMtx3D<>::Identity();
    Mtx3D<> F = I + grU;

    Mtx3D<> E0 = (F.Transpose()*F - I) / 2;
    SymMtx3D<> E = Mech::grU_to_E(grU);
    SymMtx3D<> E1 = Mech::F_to_E(F);
    Mtx3D<> Ee(
        std::array<double, 9>{
            0.29125,  0.53000,  0.64375,
            0.53000,  0.74500,  0.90500,
            0.64375,  0.90500,  0.92625
        }
    );
    SymMtx3D<> E2, E3, E4, E5;
    for (unsigned i = 0; i < E2.continuous_size(); ++i)
        E2[i] = Ee(SymMtx3D<>::index(i).i, SymMtx3D<>::index(i).j); 
    for (auto it = E3.begin(); it != E3.end(); ++it){
        auto q = it.index();
        *it = E5[q] = E4(q.i, q.j) = Ee(q.i, q.j);
    }    
    EXPECT_NEAR(sqrt((E0-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((Ee-E).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E1-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E2-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E3-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E4-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E5-Ee).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());

    Mtx3D<> E0p2 = E0 * E0 + Mtx3D<>::Identity();
    SymMtx3D<> Ep2 = E * E + I;
    SymMtx3D<> Ep2e(
        std::array<double, 6>{
            1.780140625, 1.13180625,  1.263415625,
                         2.65495000,  1.853668750,
                                      3.091378125
        }
    );
    EXPECT_NEAR(sqrt((E0p2-Ep2e).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((Ep2-Ep2e).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());

    EXPECT_NEAR(F.Det(), 1.914, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(E.Det(), 0.011065125, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(F.Trace(), 4.1, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(E.Trace(), 1.9625, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(F.SquareFrobNorm(), 6.925, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(E.SquareFrobNorm(), 4.52646875, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(tensor_convert<Mtx3D<>>(E).SquareFrobNorm(), 4.52646875, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(tensor_convert<SymMtx3D<>>(tensor_convert<Mtx3D<>>(E)).SquareFrobNorm(), 4.52646875, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((F*F.Inv() - I).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E*E.Inv() - I).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((F*F.Adj() - F.Det()*I).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((E*E.Adj() - E.Det()*I).SquareFrobNorm()), 0.0, 100*std::numeric_limits<double>::epsilon());

    std::array<double, 3> f{1, 2, 3}, s{13, -2, -3};
    auto arr_dif_norm  = [](std::array<double, 3> a, std::array<double, 3> b)->double{ 
        std::array<double, 3> c{ a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        return sqrt(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]);
    };
    Mtx3D<> Eff(
        std::array<double, 9>{
            1,  2,  3,
            2,  4,  6,
            3,  6,  9
        }
    );
    Mtx3D<> E2fs(
        std::array<double, 9>{
            26,   24,   36,
            24,   -8,  -12,
            36,  -12,  -18
        }
    );
    EXPECT_NEAR(sqrt((Mtx3D<>::TensorSquare(f) - Eff).SquareFrobNorm()), 0.0, 20*sqrt(Eff.SquareFrobNorm())*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(sqrt((Mtx3D<>::TensorSymMul2(f, s) - E2fs).SquareFrobNorm()), 0.0, 20*sqrt(E2fs.SquareFrobNorm())*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(arr_dif_norm(F.Mul(f), std::array<double, 3>{2.4, 4.5, 6.7}), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(arr_dif_norm(E.Mul(f), std::array<double, 3>{3.2825, 4.735, 5.2325}), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(F.Dot(f, s), 14.7, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(E.Dot(f, s), 17.505, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(F.Dot(E0), (F.Transpose()*E0).Trace(), 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(E.Dot(Ep2), (E*Ep2).Trace(), 100*std::numeric_limits<double>::epsilon());

    BiSym4Tensor3D<> R(
        std::array<double, 21>{
            1 , 3 , 5 , 7 , 11, 13, 19, 
            23, 31, 37, 41, 43, 47, 53, 
            61, 67, 71, 73, 79, 83, 87
        }
    );

    double R1[3][3][3][3];
    double R1_sq_nrm = 0;
    Sym4Tensor3D<> R3;
    Tensor4Rank<3> R4;
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k)
    for (int l = 0; l < 3; ++l){
        auto& v = R1[i][j][k][l];
        v = R(i, j, k, l);
        R3(i, j, k, l) = v;
        R4(i, j, k, l) = v;
        R1_sq_nrm += v*v;
    }
    BiSym4Tensor3D<> R2;
    for (unsigned i = 0; i < R2.continuous_size(); ++i){
        auto q = BiSym4Tensor3D<>::index(i);
        R2[i] = R1[q.i][q.j][q.k][q.l];
    }

    EXPECT_NEAR(R.SquareFrobNorm(), R1_sq_nrm, R1_sq_nrm*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(R3.SquareFrobNorm(), R1_sq_nrm, R1_sq_nrm*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(R4.SquareFrobNorm(), R1_sq_nrm, R1_sq_nrm*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(tensor_convert<decltype(R3)>(R).SquareFrobNorm(), R1_sq_nrm, R1_sq_nrm*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(tensor_convert<decltype(R4)>(R).SquareFrobNorm(), R1_sq_nrm, R1_sq_nrm*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR((R - R2).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR((tensor_convert<decltype(R3)>(R) - R3).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR((tensor_convert<decltype(R4)>(R) - R4).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR((tensor_convert<decltype(R)>(R3) - R).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR((tensor_convert<decltype(R)>(R3) - R).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
} 

std::string to_short_str(const Ani::SymMtx3D<>& a){
    std::ostringstream oss;
    oss << "{ " << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < a.continuous_size(); ++i)
        oss << (abs(a[i]) > 1e-100 ? a[i] : 0.0) << ", ";
    oss << " }";
    return oss.str();
}
std::string to_short_str(const Ani::BiSym4Tensor3D<>& a) { 
    std::ostringstream oss;
    oss << "{ " << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < a.continuous_size(); ++i)
        oss << (abs(a[i]) > 1e-100 ? a[i] : 0.0) << ", ";
    oss << " }";
    return oss.str();
};

TEST(AutoDiffTest, ScalarExpr) {
    using namespace Ani;
    auto f_expr = [](double xv, double yv, int dif = 2){
        auto x = VarPool<2>::Var(0, xv, dif);
        auto y = VarPool<2>::Var(1, yv, dif);
        return sin(pow(x, y)) * exp(y);
    };
    auto fv = [](double x, double y)->double{ return sin(pow(x, y)) * exp(y); };
    auto df = [](double x, double y)->PhysArr<2>{
        auto fx = exp(y) * cos(pow(x, y))*y*pow(x, y-1);
        auto fy = exp(y) * (sin(pow(x, y)) + cos(pow(x, y))*pow(x, y)*log(x));
        return PhysArr<2>({fx, fy});
    };
    auto ddf = [](double x, double y)->SymMtx<2>{
        auto sq = [](auto x) { return x*x; };
        auto fxx = y*exp(y)*pow(x, y-2)*((y-1)*cos(pow(x, y)) - sin(pow(x, y))*y*pow(x, y));
        auto fyy = exp(y)*(sin(pow(x, y)) + 2*cos(pow(x, y))*pow(x, y)*log(x) + sq(log(x))*pow(x, y)*cos(pow(x, y)) - sq(pow(x, y)*log(x))*sin(pow(x, y)));
        auto fxy = exp(y)*pow(x, y-1)*((y+1+y*log(x))*cos(pow(x, y)) - y*log(x)*pow(x, y)*sin(pow(x, y)));
        return SymMtx<2>({fxx, fxy, fyy});
    };
    auto test = [f_expr, fv, df, ddf](double x, double y, int dif){
        auto f = f_expr(x, y, dif);
        if (dif >= 0){
            auto v = fv(x, y);
            EXPECT_NEAR(f(), v, abs(v)*100*std::numeric_limits<double>::epsilon());
        }
        if (dif >= 1){
            auto dv = df(x, y);
            EXPECT_NEAR((f.D() - dv).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
        }
        if (dif >= 2){
            auto ddv = ddf(x, y);
            EXPECT_NEAR((f.DD() - ddv).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
        }
    };
    test(3, 2, 0);
    test(3, 2, 1);
    test(3, 2, 2);

    SymMtx3D<VarPool<6>::ADVal<>> E;
    Mtx3D<> Ee(
        std::array<double, 9>{
            0.29125,  0.53000,  0.64375,
            0.53000,  0.74500,  0.90500,
            0.64375,  0.90500,  0.92625
        }
    );
    for (std::size_t i = 0; i < E.continuous_size(); ++i){
        auto q = E.index(i);
        E[q] = make_shell(VarPool<6>::Var(i, Ee(q.i, q.j), 2));
    }
    auto I1 = 3 + 2*E.Trace();
    auto I2 = 3 + 4 * E.Trace() + 2 * E.Trace()*E.Trace() - 2*E.SquareFrobNorm();
    auto I3 = (SymMtx3D<VarPool<6>::ADVal<>>::Identity(VarPool<6>::ADVal<>(1.0))+VarPool<6>::ADVal<>(2.0)*E).Det();

    auto I1m = Mech::I1<>{2, tensor_convert<SymMtx3D<>>(Ee)}; 
    auto I2m = Mech::I2<>{2, tensor_convert<SymMtx3D<>>(Ee)};
    auto I3m = Mech::I3<>{2, tensor_convert<SymMtx3D<>>(Ee)};

    auto divide_duplicates = [](const auto& m){
        auto r = m;
        for (std::size_t i = 0; i < m.continuous_size(); ++i){
            auto q = m.index(i);
            auto d = m.index_duplication(i);
            r[q] = m[q] / d;
        }
        return r;
    };
    auto convert_without_duplicates = [](const auto& m){
        BiSym4Tensor3D<> r;
        for (std::size_t i = 0; i < m.continuous_size(); ++i){
            auto d = m.index_duplication(i);
            r[i] = m[i] * d / r.index_duplication(i);
        }
        return r;
    };

    EXPECT_NEAR(I1m(), I1(), abs(I1m())*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(( I1m.D() - divide_duplicates(tensor_convert<SymMtx3D<>>(I1.D())) ).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(( I1m.DD() - convert_without_duplicates(I1.DD()) ).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(I2m(), I2(), abs(I2m())*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(( I2m.D() - divide_duplicates(tensor_convert<SymMtx3D<>>(I2.D())) ).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(( I2m.DD() - convert_without_duplicates(I2.DD()) ).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(I3m(), I3(), abs(I3m())*100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(( I3m.D() - divide_duplicates(tensor_convert<SymMtx3D<>>(I3.D())) ).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(( I3m.DD() - convert_without_duplicates(I3.DD()) ).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
}

//reorder function because in some tests second derivative data writen in wrong order
std::array<double, 21> rd(const std::array<double, 21>& arr){
    std::array<double, 21> r = arr;
    static std::array<std::size_t, 21> order = {
        0, 
        1, 6,
        2, 7, 11, 
        3, 8, 12, 15,
        4, 9, 13, 16, 18,
        5,10, 14, 17, 19, 20,
    };
    for (std::size_t i = 0; i < 21; ++i)
        r[order[i]] = arr[i];
    return r;
}

TEST(ElasticityTest, Invariants){
    using namespace Ani;
    Mtx3D<> grU(
        std::array<double, 9>{
            0.1, 0.2, 0.3,
            0.35, 0.40, 0.45,
            0.5, 0.7, 0.6
        }
    );
    SymMtx3D<> E = Mech::grU_to_E(grU);
    std::array<double, 3> f{1, 2, 3}, s{13, -2, -3}, n{0, 3, -2};
    auto normalize = [](std::array<double, 3>& f) { 
        double nrm = 0;
        for (int k = 0; k < 3; ++k) 
            nrm += f[k] * f[k];
        nrm = sqrt(nrm);
        for (int k = 0; k < 3; ++k) f[k] /= nrm;    
    };
    normalize(f); normalize(s); normalize(n);

    auto test_invariants = [E, f, s, n](unsigned char dif){
        auto I0sf = Mech::I0fs<>{dif, f, s}, I0ff = Mech::I0fs<>{dif, f, f};
        auto I4f = Mech::I4fs<>{dif, E, f, f}, I4s = Mech::I4fs<>{dif, E, s, s}, I4n = Mech::I4fs<>{dif, E, n, n}; 
        auto I4sn = Mech::I4fs<>{dif, E, s, n}, I4sf = Mech::I4fs<>{dif, E, f, s}, I4fn = Mech::I4fs<>{dif, E, f, n};
        auto I5sf = Mech::I5fs<>{dif, E, f, s};
        auto I1 = Mech::I1<>{dif, E}; 
        auto I2 = Mech::I2<>{dif, E};
        auto I3 = Mech::I3<>{dif, E};
        auto J = Mech::J<>{dif, E};
        // auto& V_ = J;
        // double V = V_();
        // SymMtx3D dV = V_.D();
        // BiSym4Tensor3D<> ddV = V_.DD();
        // std::cout << std::scientific << std::setprecision(16) 
        //           << "V = " << V << "\ndV = \n" << to_short_str(dV) << "\nddV = \n" << to_short_str(ddV) << "\n" << std::endl;
        EXPECT_NEAR(I0sf(), 0, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I0ff(), 1, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I4f(), 5.0642857142857149, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I4s(), 0.92994505494505508, 100*std::numeric_limits<double>::epsilon());  
        EXPECT_NEAR(I4n(), 0.93076923076923090, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I4sf(), 0.69357335249447383, 100*std::numeric_limits<double>::epsilon());  
        EXPECT_NEAR(I4fn(), 0.55445448886250381, 100*std::numeric_limits<double>::epsilon());   
        EXPECT_NEAR(I4sn(), 0.020352971499484673, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I5sf(), 4.1687235266504956, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I1(), 6.925, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I2(), 9.499875, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(I3(), 3.663396, 100*std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(J(), 1.914, 100*std::numeric_limits<double>::epsilon());

        if (dif >= 1){
            SymMtx3D<> dI4f(std::array<double, 6>{1.4285714285714288e-01, 2.8571428571428575e-01, 4.2857142857142860e-01, 
                                            5.7142857142857151e-01, 8.5714285714285721e-01, 1.2857142857142858e+00});
            SymMtx3D<> dI1 = SymMtx3D<>::Identity(2);   
            SymMtx3D<> dI4sf(std::array<double, 6>{5.1507875363771272e-01, 4.7545731105019634e-01, 7.1318596657529454e-01, 
                                                -1.5848577035006547e-01, -2.3772865552509820e-01, -3.5659298328764732e-01});
            SymMtx3D<> dI5sf(std::array<double, 6>{ 4.4746476186211606e+00, 3.2990794170495508e+00, 3.9756155492313923e+00, 
                                                -6.4186736991776483e-01, -8.3422947368015699e-01, -1.0584868387255000e+00});
            SymMtx3D<> dI2(std::array<double, 6>{10.685, -2.12, -2.575, 8.87, -3.62, 8.145});
            SymMtx3D<> dI3(std::array<double, 6>{7.65325, -1.38655, -2.57455, 5.71285, -2.99915, 5.63365});
            SymMtx3D<> dJ(std::array<double, 6>{1.9992816091954024e+00, -3.6221264367816119e-01, -6.7255747126436760e-01, 
                                                1.4923850574712652e+00, -7.8347701149425264e-01, 1.4716954022988502e+00});
            EXPECT_NEAR((I0ff.D()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I0sf.D()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I4f.D() - dI4f).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I4sf.D() - dI4sf).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I5sf.D() - dI5sf).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I1.D() - dI1).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I2.D() - dI2).SquareFrobNorm(), 0, 100*dI2.SquareFrobNorm()*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I3.D() - dI3).SquareFrobNorm(), 0, 100*dI3.SquareFrobNorm()*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((J.D() - dJ).SquareFrobNorm(), 0, 100*dJ.SquareFrobNorm()*std::numeric_limits<double>::epsilon());
        }   
        if (dif >= 2){
            BiSym4Tensor3D<> ddI5sf(rd(std::array<double, 21>{2.0603150145508509e+00, 9.5091462210039268e-01, 3.5659298328764721e-01, 1.4263719331505891e+00, 
                            -2.3772865552509820e-01, 1.5848577035006539e-01, 0.0000000000000000e+00, 9.5091462210039268e-01, 0.0000000000000000e+00, 
                            -6.3394308140026190e-01, 0.0000000000000000e+00, 7.1318596657529454e-01, 4.7545731105019634e-01, -4.7545731105019640e-01, 
                            -5.1507875363771283e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.4263719331505891e+00, 0.0000000000000000e+00, 
                            -4.7545731105019640e-01, -1.4263719331505893e+00}));
            BiSym4Tensor3D<> ddI2(rd(std::array<double, 21>{0, 0, -2, 0, 0, -2, 4, 0, 0, 0, 0, 0, 0, 0, -2, 4, 0, 0, 4, 0, 0}));
            BiSym4Tensor3D<> ddI3(rd(std::array<double, 21>{0, 0, -5.705, 0, 3.62, -4.98, 11.41, 0, -5.15, 0, -7.24, 2.575, 2.12, 0, -3.165, 9.96, -4.24, 0, 6.33, 0, 0}));
            BiSym4Tensor3D<> ddJ(rd(std::array<double, 21>{ -2.0883630892721823e+00, 3.7835165993928688e-01, -1.5588808773460410e+00, 7.0252444275120962e-01, 
                            8.1838619660806367e-01, -1.5372693584919124e+00, 1.4217878791847838e+00, 2.8242462750911773e-01, -8.2094319728024934e-01, 
                            -1.1636432391658880e+00, -1.0729408671327922e+00, 5.2440685496635386e-01, 2.7850923843033504e-01, 6.1089309552051452e-01, 
                            -1.1475110906687385e+00, 1.0646115192510333e+00, -8.2911876574939281e-01, 5.1713674934248233e-01, 5.0609392500524286e-01, 
                            6.0242398935367569e-01, -1.1316025899412616e+00}));
            EXPECT_NEAR((I0ff.DD()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I0sf.DD()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I4f.DD()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I4sf.DD()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I5sf.DD() - ddI5sf).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I1.DD()).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I2.DD() - ddI2).SquareFrobNorm(), 0, 100*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((I3.DD() - ddI3).SquareFrobNorm(), 0, 100*ddI3.SquareFrobNorm()*std::numeric_limits<double>::epsilon());
            EXPECT_NEAR((J.DD() - ddJ).SquareFrobNorm(), 0, 100*ddJ.SquareFrobNorm()*std::numeric_limits<double>::epsilon());
        }                                   
    };
    test_invariants(0);
    test_invariants(1);
    test_invariants(2);
}

TEST(AutoDiffTest, MatrExpr) {
    using namespace Ani;
    Mtx3D<> grU(
        std::array<double, 9>{
            0.1, 0.2, 0.3,
            0.35, 0.40, 0.45,
            0.5, 0.7, 0.6
        }
    );
    SymMtx3D<> E = Mech::grU_to_E(grU);

    auto test_operations = [E](unsigned char dif){
        auto J = Mech::J<>{dif, E};
        auto I1 = Mech::I1<>{dif, E};
        auto V0 = J * I1;
        auto V1 = -pow(J, 3.0), V1p = -pow(J, 3), V1m = -pow(1/J, -3);
        auto V2 = pow(2, J), V3 = cube(J) - sq(J);
        auto V4 = J / exp(I1);
        auto V5 = log(I1) - abs(-J); 
        auto V6 = expm1(J) + log1p(I1);
        auto V7 = cbrt(J);
        auto V8 = sin(J) * exp(-I1);
        auto V9 = cos(sqrt(abs(J)));
        auto V10 = tan(J);
        auto V11 = atan(J), V12 = asin(J-1), V13 = acos(J-1); 
        auto V14 = sinh(J), V15 = cosh(J), V16 = tanh(J);
        auto V17 = sign(-J);
        auto V18 = min(I1, J), V19 = max(I1, J);
        auto V20 = pow(J, I1);
        auto V21 = atan2(I1, -J);
        auto V22 = Ani::hypot({make_shell(I1), make_shell(J)});

        double V0e = 1.3254450000000004e+01;
        SymMtx3D<> dV0(std::array<double, 6>{{ 1.7673025143678164e+01, -2.5083225574712671e+00, -4.6574604885057465e+00, 
                                            1.4162766522988511e+01, -5.4255783045977015e+00, 1.4019490660919541e+01,  }});
        BiSym4Tensor3D<> ddV0(rd(std::array<double, 21>{{ -6.4647879564282533e+00, 1.8956599577232378e+00, -1.0795250075621336e+01, 
                    3.5198668235233921e+00, 5.6673244115108421e+00, -1.0645590307556493e+01, 1.6829214396687966e+01, 
                    1.2313652581443182e+00, -7.0301465836944637e+00, -2.0886892013387151e+00, -8.9970695278830952e+00, 
                    3.6315174706420010e+00, 1.9286764761300703e+00, 2.6634806634910575e+00, -7.9465143028810132e+00, 
                    1.4314388793801916e+01, -6.4660727401708682e+00, 2.2360570466679541e+00, 9.4328613502015379e+00, 
                    2.6048321032856991e+00, -1.9495663261478366e+00,  }}));            
        double V1e = -7.0117399440000039e+00;
        SymMtx3D<> dV1(std::array<double, 6>{{ -2.1972480750000010e+01, 3.9807850500000050e+00, 7.3915330500000014e+00, 
                                            -1.6401592350000012e+01, 8.6105596500000026e+00, -1.6174209150000003e+01,  }});
        BiSym4Tensor3D<> ddV1(rd(std::array<double, 21>{{ -2.2951502963362074e+01, 4.1581558728448318e+00, 1.5625716088362072e+01, 
                7.7208757004310353e+00, -1.1791821842672416e+01, 1.1700280743534485e+01, -4.9890503911637943e+01, 
                3.1038997521551766e+00, 2.0548979924568972e+01, -1.2788657963362077e+01, 2.9780258157327584e+01, 
                -9.0223200754310380e+00, -9.1121711099137936e+00, 6.7138299676724138e+00, 5.5620673814655213e+00, 
                -4.5490039256465515e+01, 1.5233908890086212e+01, 5.6834300969827591e+00, -3.0784792618534496e+01, 
                6.6207528987068960e+00, -1.2436525204741379e+01,  }}));
        double V2e = 3.7685250956192360e+00;
        SymMtx3D<> dV2(std::array<double, 6>{{ 5.2224085506111901e+00, -9.4615105685165779e-01, -1.7568159845785827e+00, 
                                            3.8983225019905459e+00, -2.0465536346735762e+00, 3.8442781734754154e+00,  }});
        BiSym4Tensor3D<> ddV2(rd(std::array<double, 21>{{ 1.7820929492281490e+00, -3.2286427057162537e-01, -3.8344720644008428e+00, 
                -5.9949529970082382e-01, 2.5788193461617848e+00, -3.1965719084309234e+00, 9.1161939394991300e+00, 
                -2.4100547455461530e-01, -3.9617457627208670e+00, 9.9298844276032805e-01, -5.6387811681445932e+00, 
                1.8836468683119854e+00, 1.6815708954226567e+00, -5.2130220259671378e-01, -1.8860511334766605e+00, 
                8.1083039372170767e+00, -3.1309465328451678e+00, -4.4129574953902517e-01, 5.2986741780646902e+00, 
                -5.1407513826881113e-01, 9.6564673414403668e-01,  }}));        
        double V3e = 3.3483439440000025e+00;
        SymMtx3D<> dV3(std::array<double, 6>{{ 1.4319230750000006e+01, -2.5942350500000027e+00, -4.8169830500000019e+00, 
                                            1.0688742350000009e+01, -5.6114096500000006e+00, 1.0540559150000004e+01,  }});
        BiSym4Tensor3D<> ddV3(rd(std::array<double, 21>{{ 2.2951502963362074e+01, -4.1581558728448318e+00, -9.9207160883620720e+00, 
                    -7.7208757004310353e+00, 8.1718218426724150e+00, -6.7202807435344845e+00, 3.8480503911637939e+01, 
                    -3.1038997521551766e+00, -1.5398979924568971e+01, 1.2788657963362077e+01, -2.2540258157327585e+01, 
                    6.4473200754310378e+00, 6.9921711099137944e+00, -6.7138299676724138e+00, -2.3970673814655203e+00, 
                    3.5530039256465514e+01, -1.0993908890086214e+01, -5.6834300969827591e+00, 2.4454792618534491e+01, 
                    -6.6207528987068960e+00, 1.2436525204741379e+01,  }}));
        double V4e = 1.8812765681366845e-03;
        SymMtx3D<> dV4(std::array<double, 6>{{ -1.7974530085377817e-03, -3.5601993690416233e-04, -6.6105883563997720e-04, 
                                            -2.2956832098062793e-03, -7.7008199759555558e-04, -2.3160191363886676e-03,  }});
        BiSym4Tensor3D<> ddV4(rd(std::array<double, 21>{{ -2.3879528306679698e-03, 1.0839229001194004e-03, -1.5322288751658639e-03, 
                    2.0126311366358224e-03, 8.0439434449595178e-04, -1.5109868458962105e-03, 2.0586459039381904e-03, 
                    9.8963592082903238e-04, 5.1521004285020244e-04, 5.1387799848953263e-04, 4.8556707129707817e-04, 
                    5.1544113292500233e-04, 2.7374759888640658e-04, 2.1406127236337589e-03, -1.1278922291285617e-03, 
                    1.7482479991950679e-03, -1.0290354636238659e-04, 1.8304129947179825e-03, 2.1957397168069440e-03, 
                    2.1322884127744409e-03, 6.2671455884947928e-04,  }}));
        double V5e = 2.1138052073401914e-02;
        SymMtx3D<> dV5(std::array<double, 6>{{ -1.7104729449354745e+00, 3.6221264367816119e-01, 6.7255747126436760e-01, 
                                            -1.2035763932113368e+00, 7.8347701149425264e-01, -1.1828867380389223e+00,  }});
        BiSym4Tensor3D<> ddV5(rd(std::array<double, 21>{{ 2.0049526447205781e+00, -3.7835165993928682e-01, 1.5588808773460410e+00, 
                    -7.0252444275120962e-01, -8.1838619660806367e-01, 1.5372693584919124e+00, -1.5051983237363884e+00, 
                    -2.8242462750911773e-01, 8.2094319728024956e-01, 1.0802327946142842e+00, 1.0729408671327927e+00, 
                    -5.2440685496635397e-01, -2.7850923843033515e-01, -6.1089309552051452e-01, 1.1475110906687380e+00, 
                    -1.1480219638026377e+00, 8.2911876574939281e-01, -5.1713674934248233e-01, -5.8950436955684649e-01, 
                    -6.0242398935367558e-01, 1.0481921453896577e+00,  }}));           
        double V6e = 7.8501775703041847e+00;
        SymMtx3D<> dV6(std::array<double, 6>{{ 1.3807805630495317e+01, -2.4558579578467636e+00, -4.5600440700835723e+00, 
                                            1.0370968313841859e+01, -5.3120957731608041e+00, 1.0230689239692733e+01,  }});
        BiSym4Tensor3D<> ddV6(rd(std::array<double, 21>{{ 1.2878026768867452e+01, -2.3446686562310135e+00, -9.6799115620957448e+00, 
                    -4.3535874572857463e+00, 7.2004910858162052e+00, -7.3560332038450040e+00, 2.9806189646860240e+01, 
                    -1.7502028984744193e+00, -1.2371463961038643e+01, 7.1474806185299871e+00, -1.7895081039375277e+01, 
                    5.4796681445878939e+00, 5.4610255755787342e+00, -3.7857423266088857e+00, -3.6183984253242141e+00, 
                    2.7104021101585047e+01, -9.2358288181798134e+00, -3.2047284459200784e+00, 1.8259207425071995e+01, 
                    -3.7332587514638291e+00, 6.9489227320010123e+00,  }}));
        double V7e = 1.2415969711992576e+00;
        SymMtx3D<> dV7(std::array<double, 6>{{ 4.3230616345374234e-01, -7.8321511898446661e-02, -1.4542760698001922e-01, 
                                            3.2269954148717378e-01, -1.6941182244435909e-01, 3.1822580181506877e-01,  }});
        BiSym4Tensor3D<> ddV7(rd(std::array<double, 21>{{ -7.5261386462459812e-01, 1.3635210583676702e-01, -3.4695922318643313e-01, 
                    2.5317897954062113e-01, 1.5861274590801269e-01, -3.6647262741658015e-01, 8.2715605590639757e-02, 
                    1.0178148209317284e-01, -1.0191790527137290e-01, -4.1935908548265965e-01, -1.1402930144921838e-01, 
                    9.2019462669198887e-02, 2.0536024568997432e-02, 2.2015645452362975e-01, -2.9435855648081932e-01, 
                    8.5984764063509744e-03, -1.3913279874518203e-01, 1.8636811264352010e-01, -5.5985053749980446e-02, 
                    2.1710431921493595e-01, -4.0781212941840983e-01,  }}));
        double V8e = 9.2558162928829961e-04;
        SymMtx3D<> dV8(std::array<double, 6>{{ -2.5124305611760036e-03, 1.1980272151297877e-04, 2.2245003546301199e-04, 
                    -2.3447732817242829e-03, 2.5913694581922761e-04, -2.3379301274609470e-03,  }});
        BiSym4Tensor3D<> ddV8(rd(std::array<double, 21>{{ 3.3384596709684095e-03, 3.0552770815221378e-04, 3.9416922099284727e-04, 
                    5.6730472108707289e-04, -4.9616309277439059e-04, 8.9783989764833939e-05, 2.7801652801678380e-03, 
                    1.6731500550351899e-04, 7.5564881411483516e-04, 4.0001768838908297e-03, 1.2864268361463174e-03, 
                    -4.3611529671047964e-04, -5.7983741331810884e-04, 3.6190746727913034e-04, -1.8861342400645190e-04, 
                    2.9229021022654724e-03, 5.2802435348616198e-04, 3.0019613350014559e-04, 3.4628000285727555e-03, 
                    3.4970508779668752e-04, 4.0189687797153161e-03,  }}));
        double V9e = 1.8622884424529643e-01;
        SymMtx3D<> dV9(std::array<double, 6>{{ -7.0991833826068884e-01, 1.2861689764679829e-01, 2.3881622288165893e-01, -5.2992610704374976e-01, 2.7820227801189623e-01, -5.2257948536142540e-01,  }});
        BiSym4Tensor3D<> ddV9(rd(std::array<double, 21>{{ 1.0150966214154649e+00, -1.8390647377566580e-01, 5.6251653101504318e-01, -3.4147806574529577e-01, -2.7392649821246562e-01, 5.7681974046356510e-01, -3.0066618833843384e-01, -1.3727894668399859e-01, 2.2281598982779011e-01, 5.6561539833664909e-01, 2.7378979347194909e-01, -1.6678883091457689e-01, -6.2833820433496446e-02, -2.9693855464809338e-01, 4.4947432454314340e-01, -1.7666841771002187e-01, 2.5792820674080169e-01, -2.5136613923313428e-01, -2.9399173573202986e-02, -2.9282195198425148e-01, 5.5004130830271181e-01,  }}));
        double V10e = -2.7984119613521030e+00;
        SymMtx3D<> dV10(std::array<double, 6>{{ 1.7655874823013946e+01, -3.1987395205762272e+00, -5.9394286774364549e+00, 1.3179415866808906e+01, -6.9189712834994630e+00, 1.2996703256351552e+01,  }});
        BiSym4Tensor3D<> ddV10(rd(std::array<double, 21>{{ -2.1600521920322407e+02, 3.9133967489136069e+01, -2.0251261684559658e+01, 
                7.2664062600811491e+01, -4.8133915839473111e+00, -3.5932906802669272e+01, -1.3491678499912456e+02, 
                2.9211966964402183e+01, 4.2359930054620484e+01, -1.2035886587038684e+02, 6.7945450554304188e+01, 
                -9.3953236763181032e+00, -2.3584747489554655e+01, 6.3186376777820286e+01, -4.0473366731429991e+01, 
                -1.3602656256192134e+02, 1.9025403154946233e+01, 5.3488896386641173e+01, -1.0408710578324377e+02, 
                6.2310393504882356e+01, -1.1704481215303683e+02,  }}));    
        double V11e = 1.0893378424918936e+00;
        SymMtx3D<> dV11(std::array<double, 6>{{ 4.2871795772767352e-01, -7.7671431651560624e-02, -1.4422053612096580e-01, 
                                            3.2002108709431160e-01, -1.6800567901466068e-01, 3.1558448012968443e-01,  }});
        BiSym4Tensor3D<> ddV11(rd(std::array<double, 21>{{ -1.1514031403834240e+00, 2.0860131634255222e-01, -3.5737393970027675e-01, 
                3.8733151995219595e-01, 1.3261091536072384e-01, -4.0926662452902218e-01, -2.2031444587158278e-01, 
                1.5571267501617611e-01, 6.3625145675177963e-04, -6.4156586885879419e-01, 4.5642402186337944e-02, 
                6.2499213176100826e-02, -3.3029530942013052e-02, 3.3681127205997918e-01, -3.5411646853185041e-01, 
                -2.8962485778635855e-01, -8.3961367386680974e-02, 2.8511942212506958e-01, -2.7807908063214554e-01, 
                3.3214189464815302e-01, -6.2390050006987563e-01,  }}));
        double V12e = 1.1530362474120259e+00;
        SymMtx3D<> dV12(std::array<double, 6>{{ 4.9278080051604238e+00, -8.9277786424789340e-01, -1.6577124881175664e+00, 
                                            3.6784147861732905e+00, -1.9311057888709877e+00, 3.6274191445819772e+00,  }});
        BiSym4Tensor3D<> ddV12(rd(std::array<double, 21>{{ 4.9558448521328181e+01, -8.9785733900300695e+00, -2.0466996028476805e+00, 
                -1.6671440713498896e+01, 5.3512496129072806e+00, 2.4017213371103363e+00, 4.4340152480834966e+01, 
                -6.7021517644442952e+00, -1.5760579009791709e+01, 2.7614141363460071e+01, -2.4082651271042089e+01, 
                5.1765189946750443e+00, 7.8982268169510093e+00, -1.4496958973230740e+01, 5.5727652058651120e+00, 
                4.2893662332156694e+01, -9.3393062974060825e+00, -1.2272049387593908e+01, 3.1307101199704821e+01, 
                -1.4295980626052032e+01, 2.6853792325811654e+01,  }}));
        double V13e = 4.1776007938287069e-01;
        SymMtx3D<> dV13(std::array<double, 6>{{ -4.9278080051604238e+00, 8.9277786424789340e-01, 1.6577124881175664e+00, 
                                            -3.6784147861732905e+00, 1.9311057888709877e+00, -3.6274191445819772e+00,  }});
        BiSym4Tensor3D<> ddV13(rd(std::array<double, 21>{{ -4.9558448521328188e+01, 8.9785733900300748e+00, 2.0466996028476805e+00, 
                1.6671440713498896e+01, -5.3512496129072806e+00, -2.4017213371103363e+00, -4.4340152480834966e+01, 
                6.7021517644442952e+00, 1.5760579009791709e+01, -2.7614141363460071e+01, 2.4082651271042089e+01, 
                -5.1765189946750443e+00, -7.8982268169510093e+00, 1.4496958973230740e+01, -5.5727652058651120e+00, 
                -4.2893662332156694e+01, 9.3393062974060825e+00, 1.2272049387593908e+01, -3.1307101199704821e+01, 
                1.4295980626052032e+01, -2.6853792325811654e+01,  }}));
        double V14e = 3.3163330012133065e+00;
        SymMtx3D<> dV14(std::array<double, 6>{{ 6.9251561206023897e+00, -1.2546402141601609e+00, -2.3296195329169800e+00, 
                                            5.1693565666329180e+00, -2.7138251042504367e+00, 5.0976912787157946e+00,  }});
        BiSym4Tensor3D<> ddV14(rd(std::array<double, 21>{{ 6.0220854905365089e+00, -1.0910296458241142e+00, -4.9645900124286033e+00, 
                -2.0258269623572680e+00, 3.6426322844852992e+00, -3.8247391205909889e+00, 1.4819756630279262e+01, 
                -8.1441070292311013e-01, -6.1722535632580975e+00, 3.3555271603579286e+00, -8.9111512118516796e+00, 
                2.7575786199097827e+00, 2.7121928474609960e+00, -1.7615952253231710e+00, -1.9390891084774133e+00, 
                1.3445382926856013e+01, -4.6397466586811023e+00, -1.4912357582052107e+00, 9.0368017108337853e+00, 
                -1.7371733795114310e+00, 3.2631334909839707e+00,  }}));
        double V15e = 3.4638222493275337e+00;
        SymMtx3D<> dV15(std::array<double, 6>{{ 6.6302835792935575e+00, -1.2012177436866023e+00, -2.2304245371665927e+00, 
                                            4.9492458166095723e+00, -2.5982706689103683e+00, 4.8806320303775710e+00,  }});
        BiSym4Tensor3D<> ddV15(rd(std::array<double, 21>{{ 6.9196298412582271e+00, -1.2536390104068995e+00, -4.7153215496671423e+00, 
                -2.3277604949284765e+00, 3.5578588013309056e+00, -3.5312940832540161e+00, 1.5050121579508255e+01, 
                -9.3579219555130877e-01, -6.1992103977805479e+00, 3.8556420210993418e+00, -8.9839298275235944e+00, 
                2.7220895246781120e+00, 2.7488327281177387e+00, -2.0241471012857133e+00, -1.6793093168468012e+00, 
                1.3722326737656317e+01, -4.5960821594987102e+00, -1.7134926877148684e+00, 9.2860942771654926e+00, 
                -1.9960853719523981e+00, 3.7494778039443259e+00,  }}));
        double V16e = 9.5742008755129948e-01;
        SymMtx3D<> dV16(std::array<double, 6>{{ 1.6663367634908635e-01, -3.0189256060082437e-02, -5.6055496873163704e-02, 
                                            1.2438548302105358e-01, -6.5300282941542775e-02, 1.2266106696684809e-01,  }});
        BiSym4Tensor3D<> ddV16(rd(std::array<double, 21>{{ -8.1198282462643345e-01, 1.4710806330458068e-01, -1.5086633897318186e-01, 
                2.7315067208597443e-01, 2.9330917453746348e-02, -2.0031695959721624e-01, -3.5768409924202738e-01, 
                1.0981038113867626e-01, 9.1765639934480780e-02, -4.5243967825760789e-01, 1.6056322107744847e-01, 
                -1.5833053227107263e-03, -6.0883473599658594e-02, 2.3752320838921098e-01, -1.9360700599563022e-01, 
                -3.8085200712374562e-01, 1.5970803660631111e-02, 2.0106951736806583e-01, -3.0834464162243058e-01, 
                2.3423030938006037e-01, -4.3998185567209958e-01,  }}));
        double V17e = -1.0000000000000000e+00;
        SymMtx3D<> dV17(std::array<double, 6>{{ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                                            0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,  }});
        BiSym4Tensor3D<> ddV17(rd(std::array<double, 21>{{ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00,  }}));
        double V18e = 1.9140000000000004e+00;
        SymMtx3D<> dV18(std::array<double, 6>{{ 1.9992816091954024e+00, -3.6221264367816119e-01, -6.7255747126436760e-01, 
                                            1.4923850574712652e+00, -7.8347701149425264e-01, 1.4716954022988502e+00,  }});
        BiSym4Tensor3D<> ddV18(rd(std::array<double, 21>{{ -2.0883630892721823e+00, 3.7835165993928688e-01, -1.5588808773460410e+00, 
                7.0252444275120962e-01, 8.1838619660806367e-01, -1.5372693584919124e+00, 1.4217878791847838e+00, 
                2.8242462750911773e-01, -8.2094319728024934e-01, -1.1636432391658880e+00, -1.0729408671327922e+00, 
                5.2440685496635386e-01, 2.7850923843033504e-01, 6.1089309552051452e-01, -1.1475110906687385e+00, 
                1.0646115192510333e+00, -8.2911876574939281e-01, 5.1713674934248233e-01, 5.0609392500524286e-01, 
                6.0242398935367569e-01, -1.1316025899412616e+00,  }}));
        double V19e = 6.9250000000000007e+00;
        SymMtx3D<> dV19(std::array<double, 6>{{ 2.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                                            2.0000000000000000e+00, 0.0000000000000000e+00, 2.0000000000000000e+00,  }});
        BiSym4Tensor3D<> ddV19(rd(std::array<double, 21>{{ 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 
                0.0000000000000000e+00, 0.0000000000000000e+00,  }}));        
        double V20e = 8.9628863397413852e+01;
        SymMtx3D<> dV20(std::array<double, 6>{{ 7.6470862635991000e+02, -1.1745982228852870e+02, -2.1809973349171048e+02, 
                                            6.0033010472804631e+02, -2.5406918323655151e+02, 5.9362077731450063e+02,  }});
        BiSym4Tensor3D<> ddV20(rd(std::array<double, 21>{{ 5.5444963401071300e+03, -7.9069738112428922e+02, -3.7381625145402211e+02, 
                -1.4681691555108264e+03, 5.0993826841709938e+02, -4.4433580674062625e+01, 5.4045427675555038e+03, 
                -6.3749236753776268e+02, -1.6199733580687803e+03, 3.5458238159977645e+03, -2.3236299909616355e+03, 
                4.5493687368587985e+02, 6.1928220234287051e+02, -1.3789154621909615e+03, 2.4408455543449034e+02, 
                5.2365430741413375e+03, -9.9042531674568352e+02, -1.1720865668188994e+03, 4.0456480412888786e+03, 
                -1.3653894571381024e+03, 3.4733536329089338e+03,  }}));
        double V21e = 1.8404542492224520e+00;
        SymMtx3D<> dV21(std::array<double, 6>{{ 1.9405686023526406e-01, -4.8592989732820883e-02, -9.0227602117943034e-02, 
                                            1.2605366000623117e-01, -1.0510812098892189e-01, 1.2327801918055628e-01,  }});
        BiSym4Tensor3D<> ddV21(rd(std::array<double, 21>{{ -4.1307360492796263e-01, 7.0004816092794325e-02, -2.1043844836839073e-01, 
                1.2998514245552159e-01, 1.0736777301131314e-01, -2.1073405829586039e-01, 8.4768779405774930e-02, 
                5.5308982615896618e-02, -7.7788868724727497e-02, -2.3770387789731567e-01, -1.0231026755682734e-01, 
                6.7528979394723729e-02, 3.2121305889122570e-02, 1.1963501872450766e-01, -1.6005242183777446e-01, 
                3.7950811299294153e-02, -9.3885760211597641e-02, 1.0158411094249008e-01, -1.2703628045572034e-02, 
                1.1833756824810904e-01, -2.3141962106180175e-01,  }}));  
        double V22e = 7.1846378475188306e+00;
        SymMtx3D<> dV22(std::array<double, 6>{{ 2.4603362584384838e+00, -9.6494077323524219e-02, -1.7917047836232303e-01, 
                                            2.3252981367417789e+00, -2.0871963651137526e-01, 2.3197863766725257e+00,  }});
        BiSym4Tensor3D<> ddV22(rd(std::array<double, 21>{{ -2.8578399470700633e-01, 3.3043819633249934e-02, -3.9832364103735962e-01, 
                6.1355858668481880e-02, 2.4952003817783447e-01, -3.5104094511702844e-01, 5.5451433559321262e-01, 
                3.1230175127685160e-02, -3.0041531199103350e-01, -1.9583609565243321e-01, -4.3237802327305408e-01, 
                1.7639856289947980e-01, 1.4233196224856293e-01, 6.7551822678011492e-02, -2.2632510102468026e-01, 
                4.5549205611958621e-01, -2.6391787508818260e-01, 5.7850826113713484e-02, 2.4647102585636244e-01, 
                6.7391701516359676e-02, -1.9227257695006938e-01,  }}));              
        EXPECT_FALSE(I1 < J);
        EXPECT_FALSE(I1 == J);
        EXPECT_TRUE(I1 > J);
    #define SPEC_CHECK(X)  \
            EXPECT_NEAR(X(), X##e, 100*abs(X##e)*std::numeric_limits<double>::epsilon());\
            if (dif >= 1) { EXPECT_NEAR((X.D() - d##X).SquareFrobNorm(), 0, 100*d##X.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }\
            if (dif >= 2) { EXPECT_NEAR((X.DD() - dd##X).SquareFrobNorm(), 0, 100*dd##X.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }     
        
        SPEC_CHECK(V0);
        SPEC_CHECK(V1);
        EXPECT_NEAR(V1p(), V1e, 100*abs(V1e)*std::numeric_limits<double>::epsilon());
        if (dif >= 1) { EXPECT_NEAR((V1p.D() - dV1).SquareFrobNorm(), 0, 100*dV1.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }
        if (dif >= 2) { EXPECT_NEAR((V1p.DD() - ddV1).SquareFrobNorm(), 0, 100*ddV1.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }
        EXPECT_NEAR(V1m(), V1e, 100*abs(V1e)*std::numeric_limits<double>::epsilon());
        if (dif >= 1) { EXPECT_NEAR((V1m.D() - dV1).SquareFrobNorm(), 0, 100*dV1.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }
        if (dif >= 2) { EXPECT_NEAR((V1m.DD() - ddV1).SquareFrobNorm(), 0, 100*ddV1.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); } 
        SPEC_CHECK(V2);
        SPEC_CHECK(V3);
        SPEC_CHECK(V4);
        EXPECT_NEAR(V5(), V5e, 1000*abs(V5e)*std::numeric_limits<double>::epsilon());
        if (dif >= 1) { EXPECT_NEAR((V5.D() - dV5).SquareFrobNorm(), 0, 100*dV5.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }
        if (dif >= 2) { EXPECT_NEAR((V5.DD() - ddV5).SquareFrobNorm(), 0, 100*ddV5.SquareFrobNorm()*std::numeric_limits<double>::epsilon()); }
        SPEC_CHECK(V6);
        SPEC_CHECK(V7);
        SPEC_CHECK(V8);
        SPEC_CHECK(V9);
        SPEC_CHECK(V10);
        SPEC_CHECK(V11);
        SPEC_CHECK(V12);
        SPEC_CHECK(V13);
        SPEC_CHECK(V14);
        SPEC_CHECK(V15);
        SPEC_CHECK(V16);
        SPEC_CHECK(V17);
        SPEC_CHECK(V18);
        SPEC_CHECK(V19);
        SPEC_CHECK(V20);
        SPEC_CHECK(V21);
        SPEC_CHECK(V22);

    #undef SPEC_CHECK
        // auto& V_ = V21;
        // double V = V_();
        // SymMtx3D<> dV = V_.D();
        // BiSym4Tensor3D<> ddV = V_.DD();
        // std::cout << std::scientific << std::setprecision(16) 
        //           << "V = " << V << "\ndV = \n" << to_short_str(dV) << "\nddV = \n" << to_short_str(ddV) << "\n" << std::endl;
    };
    test_operations(2);
    test_operations(1);
    test_operations(0);
}

TEST(ElasticityTest, Potential) {
    using namespace Ani;
    Mtx3D<> grU(
        std::array<double, 9>{
            0.1, 0.2, 0.3,
            0.35, 0.40, 0.45,
            0.5, 0.7, 0.6
        }
    );
    //gradU(i, j) = grad_j U_i
    // SymMtx3D<> I = SymMtx3D<>::Identity();
    // Mtx3D<> F = I + grU;
    SymMtx3D<> E = Mech::grU_to_E(grU);
    std::array<double, 3> f{1, 2, 3}, s{13, -2, -3}, n{0, 3, -2};
    auto normalize = [](std::array<double, 3>& f) { 
        double nrm = 0;
        for (int k = 0; k < 3; ++k) 
            nrm += f[k] * f[k];
        nrm = sqrt(nrm);
        for (int k = 0; k < 3; ++k) f[k] /= nrm;    
    };
    normalize(f); normalize(s); normalize(n);

    double mu = 1e2, bf = 2, bt = 7, bfs = 1;
    auto get_fung_potential = [&mu_ = mu, &bf_ = bf, &bt_ = bt, &bfs_ = bfs, &f, &s, &n](const SymMtx3D<>& E, unsigned char dif){
        Param<> mu(mu_), bf(bf_), bt(bt_), bfs(bfs_);
        auto I4f = Mech::I4fs<>{dif, E, f, f}, I4s = Mech::I4fs<>{dif, E, s, s}, I4n = Mech::I4fs<>{dif, E, n, n}; 
        auto I4sn = Mech::I4fs<>{dif, E, s, n}, I4sf = Mech::I4fs<>{dif, E, f, s}, I4fn = Mech::I4fs<>{dif, E, f, n};

        auto Lf = sq(I4f - 1.);
        auto Lt = sq(I4s - 1.) + sq(I4n - 1.) + 2 * sq(I4sn);
        auto Lfs = 2*(sq(I4sf) + sq(I4fn));
        auto Q = (bf*Lf + bt*Lt + bfs*Lfs)/4;
        auto W = mu/2*(exp(Q) - 1);
        return W;
    };
    auto test_fung_potential = [get_fung_potential, E](unsigned char dif){
        auto W = get_fung_potential(E, dif);
        double We = 2.9176927538144670e+05;
        EXPECT_NEAR(W(), We, 100*abs(We)*std::numeric_limits<double>::epsilon());
        if (dif >= 1){
            SymMtx3D<> dW(std::array<double, 6>{1.4080280037154834e+05, 5.2485781100748782e+05, 6.3710401085956546e+05, 
                    6.7827137292230572e+05, 1.0847339350607491e+06, 1.2684758573812819e+06});
            EXPECT_NEAR((W.D() - dW).SquareFrobNorm(), 0, 100*dW.SquareFrobNorm()*std::numeric_limits<double>::epsilon());        
        }
        if (dif >= 2){
            BiSym4Tensor3D<> ddW(rd(std::array<double, 21>{3.6739898255357202e+06, -2.0532925321687417e+05, 2.4447791077594794e+06, 
                    -3.8045703530224029e+05, 5.2055005228574749e+05, 2.3706135739179011e+06, 4.1064287325828377e+05, 
                    8.8641176886289415e+05, 1.7309425584248309e+06, 3.9944276013671188e+06, 6.4844952740171843e+05, 
                    1.8259060594672137e+06, 2.4932720269586053e+06, 1.5207083591833820e+06, 5.1993881001373157e+06, 
                    7.9963770678882860e+05, 2.6566406461233376e+06, 2.5817565073161321e+06, 3.6986927366370722e+06, 
                    4.3399097615182763e+06, 6.7852910070861559e+06}));
            EXPECT_NEAR((W.DD() - ddW).SquareFrobNorm(), 0, 100*ddW.SquareFrobNorm()*std::numeric_limits<double>::epsilon());        
        }            
    };
    test_fung_potential(0);
    test_fung_potential(1);
    test_fung_potential(2);
    auto W1 = get_fung_potential(E, 2);
    
    Mtx3D<> Pe = Mech::S_to_P(grU, W1.D());
    Mtx3D<> Pa(std::array<double, 9>{ 4.509858458680704e+05,  1.038418047210922e+06,  1.298303956172056e+06,
                                    1.070778720427329e+06,  1.621410426721186e+06,  2.312428048707474e+06,
                                    1.457168285266320e+06,  2.472793162646557e+06,  3.107427131782358e+06 });
    
    EXPECT_NEAR((Pe - Pa).SquareFrobNorm(), 0, Pa.SquareFrobNorm() * std::numeric_limits<double>::epsilon());                                
    Sym4Tensor3D<> dPe = Mech::dS_to_dP(grU, W1.D(), W1.DD());
    
    Tensor4Rank<3> dPa{std::array<double, 81>{
        4.6184963667921470e+06, 1.7636339819901544e+06, 2.0945029167303748e+06, 2.1256832732880423e+06, 2.5762275102377781e+06, 3.6793726420013481e+06, 2.8714401345970519e+06, 3.9357516068650903e+06, 4.9736462762056189e+06, 
        1.7636339819901548e+06, 6.0417803070283597e+06, 5.1818824394902373e+06, 5.5478649350032927e+06, 6.0598249251218438e+06, 8.2386152701922869e+06, 5.0822587439295165e+06, 9.4641889043853097e+06, 1.1486531824177410e+07, 
        2.0945029167303748e+06, 5.1818824394902382e+06, 8.2773581545378165e+06, 4.1579526460855701e+06, 7.5400756922306614e+06, 1.1061105026120828e+07, 7.3992542258476028e+06, 1.1324959291608822e+07, 1.4689145249107912e+07, 
        2.1256832732880423e+06, 5.5478649350032918e+06, 4.1579526460855687e+06, 6.1975092855339255e+06, 6.5660484636902772e+06, 9.0086489481220841e+06, 5.5829839848818062e+06, 9.5553916291824263e+06, 1.1822368718922542e+07, 
        2.5762275102377785e+06, 6.0598249251218438e+06, 7.5400756922306605e+06, 6.5660484636902781e+06, 1.3219647477369068e+07, 1.3167679423118005e+07, 8.4042085788139496e+06, 1.4242340061267292e+07, 1.8336716204414781e+07, 
        3.6793726420013485e+06, 8.2386152701922888e+06, 1.1061105026120827e+07, 9.0086489481220879e+06, 1.3167679423118006e+07, 2.1848644311127350e+07, 1.2092355303696444e+07, 2.1214034684706848e+07, 2.5866957603070479e+07, 
        2.8714401345970524e+06, 5.0822587439295165e+06, 7.3992542258476028e+06, 5.5829839848818099e+06, 8.4042085788139496e+06, 1.2092355303696441e+07, 8.7395831521721277e+06, 1.3208911462605396e+07, 1.6759233451210011e+07, 
        3.9357516068650894e+06, 9.4641889043853097e+06, 1.1324959291608818e+07, 9.5553916291824281e+06, 1.4242340061267287e+07, 2.1214034684706837e+07, 1.3208911462605394e+07, 2.3505493868805930e+07, 2.8400865225533910e+07, 
        4.9736462762056179e+06, 1.1486531824177409e+07, 1.4689145249107912e+07, 1.1822368718922544e+07, 1.8336716204414781e+07, 2.5866957603070475e+07, 1.6759233451210013e+07, 2.8400865225533914e+07, 3.7376673094446383e+07,
    }};
    EXPECT_NEAR(sqrt((dPe - tensor_convert<Sym4Tensor3D<>>(dPa)).SquareFrobNorm()), 0, 100*sqrt(dPa.SquareFrobNorm()) * std::numeric_limits<double>::epsilon());
}

TEST(ElasticityTest, HGO_SpeedTest){
    using namespace Ani;
    struct PotentialCoefs{
        double mu = 1;
        double k1[3] = {1, 1, 1}, k2[3] = {1, 1, 1}, sigma[3] = {0, 0, 0};
        std::array<double, 3> f[3]{std::array<double, 3>{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        double kappa = 1;
    };

    auto get_test_potential = [](const PotentialCoefs& c, const Mtx3D<>& grU, unsigned char dif = 2)->double{
        Param<> mu(c.mu);
        std::array<Param<>, 3> k1{Param<>{c.k1[0]}, Param<>{c.k1[1]}, Param<>{c.k1[2]}};
        std::array<Param<>, 3> k2{Param<>{c.k2[0]}, Param<>{c.k2[1]}, Param<>{c.k2[2]}};
        std::array<Param<>, 3> sigma{Param<>{c.sigma[0]}, Param<>{c.sigma[1]}, Param<>{c.sigma[2]}};
        Param<> kappa(c.kappa);
        SymMtx3D<> E = Mech::grU_to_E(grU);

        // auto I4f = Mech::I4fs<>{dif, E, c.f[0], c.f[0]}, I4s = Mech::I4fs<>{dif, E, c.f[1], c.f[1]}, I4n = Mech::I4fs<>{dif, E, c.f[2], c.f[2]}; 
        // auto I4sn = Mech::I4fs<>{dif, E, c.f[1], c.f[2]}, I4sf = Mech::I4fs<>{dif, E, c.f[0], c.f[1]}, I4fn = Mech::I4fs<>{dif, E, c.f[0], c.f[2]};
        
        auto J = Mech::J<>{dif, E};
        auto I1 = pow(J, -2.0/3) * Mech::I1<>{dif, E};
        auto q = (I1 - 3);
        auto psi_NHK = mu/2 * q;
        ADVal<> psi_aniso;
        for (std::size_t i = 0; i < 3; ++i){
            auto If = pow(J, -2.0/3) * Mech::I4fs<>{dif, E, c.f[i], c.f[i]};
            auto Ia = sigma[i] * q + (1 - sigma[i]) * (If - 1);
            psi_aniso = psi_aniso +  k1[i] / (2*k2[i]) * (exp(k2[i] * sq(Ia)) - 1);
        }
        auto psi_bulk = kappa/4*(sq(J) - 1 - 2*log(J));

        auto W = psi_NHK + psi_aniso + psi_bulk;

        Mtx3D<> Pe = Mech::S_to_P(grU, W.D());
        Sym4Tensor3D<> dPe = Mech::dS_to_dP(grU, W.D(), W.DD());
        return sqrt(Pe.SquareFrobNorm()) + sqrt(dPe.SquareFrobNorm());
    };

    std::size_t N = 100000;
    Mtx3D<> grU1;
    PotentialCoefs pc;
    auto start = std::chrono::steady_clock::now();
    double res = 0;
    for (std::size_t i = 0; i < N; ++i){ 
        grU1(0,0) = 0.1/N*i;  grU1(0,1) = 0.1/N*i; 
        res += get_test_potential(pc, grU1);   
    }
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    auto sym_dif_speed = (std::chrono::duration <double, std::micro> (diff).count()/N);

    std::cout << "res = " << res << " total_cycle(N = " << N << ") time = " << std::chrono::duration <double, std::milli> (diff).count() << "ms\n"
        << "sym_dif_speed = " << sym_dif_speed << " us / eval\n" <<  std::endl;
}