//
// Created by Liogky Alexey on 24.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_POLY_2_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_POLY_2_H

#include "common.h"

namespace Ani{
    template<>
    struct Operator<IDEN, FemFix<FEM_P2>> {
        using Nfa = std::integral_constant<int, 10>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 2>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f;
            extraR = extraI = 0;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &Uin) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            constexpr auto nfa = Nfa::value;
            auto q = mem.q, f = mem.f;
            assert(Uin.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            DenseMatrix<const ScalarType> XYL(mem.XYL.data, 4, q, mem.XYL.size);
            DenseMatrix<ScalarType> U(Uin.data, nRow, nCol, Uin.size);
            std::array<unsigned char, 6> I = {0, 0, 0, 1, 1, 2}, J = {1, 2, 3, 2, 3, 3};
            for (IndexType i = 0; i < 4; ++i)
                for (std::size_t n = 0; n < q; ++n)
                    U(n, i) = XYL(i, n) * (2*XYL(i, n) - 1);
            for (IndexType i = 4; i < nfa; ++i)
                for (std::size_t n = 0; n < q; ++n)
                    U(n, i) = 4*XYL(I[i-4], n) * XYL(J[i-4], n);

            for (std::size_t r = 1; r < f; ++r)
                for (IndexType i = 0; i < nfa; ++i)
                    for (std::size_t n = 0; n < q; ++n)
                        U(n, i+r*nfa) = U(n, i);

            return U;
        }
    };
    
    template<>
    struct Operator<GRAD, FemFix<FEM_P2>> {
        using Nfa = std::integral_constant<int, 10>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 1>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = 3*10*q;
            extraI = 0;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U) {
            auto nRow = Dim ::value*mem.q, nCol = Nfa::value*mem.f;
            auto wsz = nRow * nCol;
            constexpr auto nfa = Nfa::value;
            auto q = mem.q, f = mem.f;
            assert(U.size >= wsz && "Not enough memory for operator initialization");
            assert(mem.extraR.size >= 3*10*q && "Not enough memory for operator initialization in mem.extraR");
            (void) wsz; (void) nfa;
            DenseMatrix<ScalarType> GRAD_P2(mem.extraR.data, 3, 10*q, mem.extraR.size);
            ScalarType GRAD_P1[3*4] = {-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
            std::array<unsigned char, 6> I = {0, 0, 0, 1, 1, 2}, J = {1, 2, 3, 2, 3, 3};
            for (std::size_t n = 0; n < q; ++n){
                for (IndexType i = 0; i < 4; ++i) {
                    auto s1 = mem.XYL.data[i + 4*n];
                    for (IndexType d = 0; d < 3; ++d)
                        GRAD_P2.data[d + 3*(i + 10 * n)] = GRAD_P1[d + 3*i] * (4 * s1 - 1);
                }
                for (IndexType i = 4; i < 10; ++i) {
                    auto s1 = mem.XYL.data[I[i - 4] + 4*n], s2 = mem.XYL.data[J[i - 4] + 4*n];
                    for (IndexType d = 0; d < 3; ++d)
                        GRAD_P2.data[d + 3*(i + 10 * n)] = 4 * (GRAD_P1[d + 3*J[i - 4]] * s1 + GRAD_P1[d + 3*I[i - 4]] * s2);
                }
            }
            for (std::size_t r = 0; r < f; ++r)
                for (IndexType i = 0; i < Nfa::value; ++i)
                    for (std::size_t n = 0; n < q; ++n)
                        for (IndexType k = 0; k < Dim::value; ++k){
                            ScalarType s = 0;
                            for (IndexType j = 0; j < 3; ++j)
                                s += mem.PSI.data[j + 3*(k+3*r)] * GRAD_P2.data[j + 3*(i + 10*n)];
                            U.data[k + Dim::value * (n + q * ( i + Nfa::value*r ))] = s;
                        }

            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<>
    struct Dof<FemFix<FEM_P2>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(DofT::DofSymmetries({1}, {1, 0}));
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet >= 0 && idof_on_tet < 10 && "Wrong idof value");
            (void) max_quad_order;
            std::array<const double*, 4> XY{XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
            if (idof_on_tet < 4) {
                std::array<double, 3> X;
                std::copy(XY[idof_on_tet], XY[idof_on_tet]+3, X.data());
                f(X, fstorage, 1*fusion, user_data);
                for (int r = 0; r < fusion; ++r)  
                    udofs[r].data[idof_on_tet] = fstorage[r];  
            } else {
                int iedge = idof_on_tet - 4;
                const static std::array<char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
                std::array<double, 3> X[2];
                for (int k = 0; k < 3; ++k)
                    X[0][k] = XY[lookup_nds[2*iedge]][k], X[1][k] = XY[lookup_nds[2*iedge+1]][k];
                {
                   std::array<double, 3> lX{(X[0][0]+ X[1][0])/2, (X[0][1] + X[1][1])/2, (X[0][2] + X[1][2])/2};
                   f(lX, fstorage, 1*fusion, user_data);
                   for (int r = 0; r < fusion; ++r)
                        udofs[r][idof_on_tet] = fstorage[r];
                }
            }
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_P2>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_P2>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            (void) grad_lmb;
            for (int i = 0; i < 4; ++i){
                phi[i] = (2*lmb(i, 0) - 1)*lmb(i, 0);
            }
            phi += 4;
            for (int j = 0, l = 0; j < 4; ++j)
                for (int k = j+1; k < 4; ++k){
                    phi[l] = 4 * lmb(j, 0) * lmb(k, 0);
                    ++l;
                }
        }
    };

    using P2Space = GenSpace<Polynomial, FEM_P2>;
}
#endif //CARNUM_FEM_ANIINTERFACE_SPACES_POLY_2_H