//
// Created by Liogky Alexey on 24.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_NEDELEC_0_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_NEDELEC_0_H

#include "common.h"
#include <cmath>

namespace Ani{
    template<>
    struct Operator<IDEN, FemFix<FEM_ND0>> {
        using Nfa = std::integral_constant<int, 6>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 1>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = extraI = 0;
        }
        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &Uin) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            constexpr auto nfa = Nfa::value;
            auto q = mem.q, f = mem.f;
            constexpr auto dim = Dim::value;
            assert(Uin.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            static const unsigned char IPR[4*6] = {0,1,2,3, 0,2,3,1, 0,3,1,2, 1,2,0,3, 1,3,2,0, 2,3,0,1};
            DenseMatrix<const ScalarType> XYL(mem.XYL.data, 4, q, mem.XYL.size);
            DenseMatrix<ScalarType> U(Uin.data, nRow, nCol, Uin.size);
            for (std::size_t r = 0; r < f; ++r){
                DenseMatrix<ScalarType> locU(U.data + U.nRow*nfa*r, U.nRow, nfa);
                DenseMatrix<ScalarType> locXYP(mem.XYP.data + 12*r, 3, 4);
                auto vol = mem.DET.data[r];
                for (IndexType i = 0; i < 6; ++i) {
                    std::array<unsigned char, 4> ip{IPR[4 * i], IPR[4 * i + 1], IPR[4 * i + 2], IPR[4 * i + 3]};
                    ScalarType sqr = 0;
                    for (int k = 0; k < 3; ++k) {
                        auto t = locXYP(k, ip[0]) - locXYP(k, ip[1]);
                        sqr += t * t;
                    }
                    sqr = sqrt(sqr);
                    ScalarType xyn[3], xym[3];
                    cross(&locXYP(0, ip[2]), &locXYP(0, ip[3]), &locXYP(0, ip[0]), xyn);
                    cross(&locXYP(0, ip[2]), &locXYP(0, ip[3]), &locXYP(0, ip[1]), xym);
                    sqr /= vol;
                    for (std::size_t n = 0; n < q; ++n)
                        for (int k = 0; k < Dim::value; ++k)
                            locU(k + dim * n, i) = sqr * (xyn[k] * XYL(ip[0], n) + xym[k] * XYL(ip[1], n));
                }
            }
            return U;
        };
    };

    template<>
    struct Operator<DIV, FemFix<FEM_ND0>> {
        using Nfa = std::integral_constant<int, 6>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = extraI = 0;
        }
        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &Uin) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            assert(Uin.size >= wsz && "Not enough memory for operator initialization");
            std::fill(Uin.data, Uin.data + wsz, 0);
            return {Uin.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), Uin.size};
        }
    };

    template<>
    struct Operator<CURL, FemFix<FEM_ND0>> {
        using Nfa = std::integral_constant<int, 6>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = extraI = 0;
        }
        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &Uin) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            constexpr auto nfa = Nfa::value;
            auto q = mem.q, f = mem.f;
            constexpr auto dim = Dim::value;
            assert(Uin.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            static const unsigned char IPR[4*6] = {0,1,2,3, 0,2,3,1, 0,3,1,2, 1,2,0,3, 1,3,2,0, 2,3,0,1};
            DenseMatrix<const ScalarType> XYL(mem.XYL.data, 4, q, mem.XYL.size);
            DenseMatrix<ScalarType> U(Uin.data, nRow, nCol, Uin.size);
            static ScalarType GRAD_P1_[3*4] = {-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
            std::array<double, 3*4> Vdat = {0};
            DenseMatrix<ScalarType> GRAD_P1(GRAD_P1_, 3, 4, 12);
            for (std::size_t r = 0; r < f; ++r) {
                DenseMatrix<ScalarType> lU(U.data + U.nRow*nfa*r, U.nRow, nfa, U.nRow*nfa);
                DenseMatrix<ScalarType> lXYP(mem.XYP.data + 12*r, 3, 4, 12);
                DenseMatrix<ScalarType> lPSI(mem.PSI.data + 9*r, 3, 3, 9);
                DenseMatrix<ScalarType> V(Vdat.data(), 4, 3, 4*3);
                for (IndexType i = 0; i < 3; ++i)
                    for (IndexType j = 0; j < 4; ++j){
                        ScalarType s = 0;
                        for (IndexType k = 0; k < 3; ++k)
                            s += GRAD_P1(k, j) * lPSI(k, i);
                        V(j, i) = s;
                    }
                auto vol = mem.DET.data[r];
                for (IndexType i = 0; i < 6; ++i) {
                    std::array<unsigned char, 4> ip{IPR[4 * i], IPR[4 * i + 1], IPR[4 * i + 2], IPR[4 * i + 3]};
                    ScalarType sqr = 0;
                    for (int k = 0; k < 3; ++k) {
                        auto t = lXYP(k, ip[0]) - lXYP(k, ip[1]);
                        sqr += t * t;
                    }
                    sqr = sqrt(sqr);
                    ScalarType xyn[3], xym[3];
                    cross(&lXYP(0, ip[2]), &lXYP(0, ip[3]), &lXYP(0, ip[0]), xyn);
                    cross(&lXYP(0, ip[2]), &lXYP(0, ip[3]), &lXYP(0, ip[1]), xym);
                    sqr /= vol;
                    for (int k = 0; k < Dim::value; ++k){
                        lU(k + dim * 0, i) = sqr*(
                             (V(ip[0], (k+1)%3)*xyn[(k+2)%3] + V(ip[1], (k+1)%3)*xym[(k+2)%3])
                            -(V(ip[0], (k+2)%3)*xyn[(k+1)%3] + V(ip[1], (k+2)%3)*xym[(k+1)%3]));
                    }
                    for (std::size_t n = 1; n < q; ++n)
                        for (int k = 0; k < Dim::value; ++k)
                            lU(k + dim*n, i) = lU(k + dim * 0, i);
                }    
            }
            return U;
        }
    };
    template<>
    struct Operator<GRAD, FemFix<FEM_ND0>> {
        using Nfa = std::integral_constant<int, 6>;
        using Dim = std::integral_constant<int, 9>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = extraI = 0;
        }
        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol, csz = mem.q*Nfa::value*mem.f*3;
            constexpr auto nfa = Nfa::value;
            auto q = mem.q, f = mem.f;
            constexpr auto dim = Dim::value;
            assert(U.size >= wsz && "Not enough memory for operator initialization");

            ArrayView<ScalarType> CurlU(U.data + wsz - csz, csz);
            Operator<CURL, FemFix<FEM_ND0>>::apply(mem, CurlU);
            for (std::size_t r = 0; r < f; ++r)
                for (IndexType i = 0; i < nfa ; ++i)
                    for (std::size_t n = 0; n < q; ++n){
                        ScalarType a[3] = {U.data[wsz-csz+0+n*3+q*3*(i + nfa*r)]/2, U.data[wsz-csz+1+n*3+q*3*(i + nfa*r)]/2, U.data[wsz-csz+2+n*3+q*3*(i + nfa*r)]/2};
                        ScalarType* lU = U.data + dim*(n + q*(i + r*nfa));
                        lU[0] =   0,   lU[3] =  a[2], lU[6] = -a[1];
                        lU[1] = -a[2], lU[4] =   0,   lU[7] =  a[0];
                        lU[2] =  a[1], lU[5] = -a[0], lU[8] =   0;
                    }
            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};        
        }
    };

    template<>
    struct Dof<FemFix<FEM_ND0>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(std::array<uint, DofT::NGEOM_TYPES>{0, 0, 1, 0, 0, 0});
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet >= 0 && idof_on_tet < 6 && "Wrong idof value");
            std::array<const double*, 4> XY{XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
            const static std::array<char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
            std::array<double, 3> X[2];
                for (int k = 0; k < 3; ++k)
                    X[0][k] = XY[lookup_nds[2*idof_on_tet]][k], X[1][k] = XY[lookup_nds[2*idof_on_tet+1]][k];
            double t[3], t_nrm = 0;
            for (int k = 0; k < 3; ++k){
                t[k] = X[1][k] - X[0][k];
                t_nrm += t[k]*t[k];
            }
            t_nrm = sqrt(t_nrm);
            for (int k = 0; k < 3; ++k)
                t[k] /= t_nrm;
            
            for (int r = 0; r < fusion; ++r)
                udofs[r][idof_on_tet] = 0;
            auto formula = segment_quadrature_formulas(max_quad_order);
            for (int i = 0; i < formula.GetNumPoints(); ++i){
                auto q = formula.GetPointWeight(i);
                std::array<double, 3> lX{X[0][0]*q.p[0] + X[1][0]*q.p[1], X[0][1]*q.p[0] + X[1][1]*q.p[1], X[0][2]*q.p[0] + X[1][2]*q.p[1]};
                f(lX, fstorage, 3*fusion, user_data);
                for (int r = 0; r < fusion; ++r)
                    udofs[r][idof_on_tet] += q.w * (fstorage[3*r+0] * t[0] + fstorage[3*r+1] * t[1] + fstorage[3*r+2] * t[2]);
            }
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_ND0>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_ND0>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            using namespace FT;
            auto V = det3x3(vertcat({grad_lmb.col(0), grad_lmb.col(1), grad_lmb.col(2)}));
            static const unsigned char IPR[4*6] = {0,1,2,3, 0,2,3,1, 0,3,1,2, 1,2,0,3, 1,3,2,0, 2,3,0,1};
            for (int i = 0; i < 6; ++i){
                auto l0 = grad_lmb.col(IPR[6*i+0]).T();
                auto l1 = grad_lmb.col(IPR[6*i+1]).T();
                auto l2 = grad_lmb.col(IPR[6*i+2]).T();
                auto l3 = grad_lmb.col(IPR[6*i+3]).T();
                auto e = norm(cross(l2, l3) / V);
                phi[i] = 6*e*(lmb(IPR[6*i+0], 0)*l1 - lmb(IPR[6*i+1], 0)*l0);
            } 
        }
    };

    using ND0Space = GenSpace<Nedelec, FEM_ND0>;
}
#endif //CARNUM_FEM_ANIINTERFACE_SPACES_NEDELEC_0_H