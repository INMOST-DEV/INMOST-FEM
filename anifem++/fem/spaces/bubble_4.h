//
// Created by Liogky Alexey on 14.03.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_BUBBLE_4_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_BUBBLE_4_H

#include "common.h"
namespace Ani{
    template<>
    struct Operator<IDEN, FemFix<FEM_B4>> {
        using Nfa = std::integral_constant<int, 1>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 4>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f;
            extraR = extraI = 0;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            auto q = mem.q, f = mem.f;
            assert(U.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            const int coef = 256;
            for (std::size_t n = 0; n < q; ++n)
                U.data[n] = coef*mem.XYL.data[0 + 4 * n]*mem.XYL.data[1 + 4 * n]*mem.XYL.data[2 + 4 * n]*mem.XYL.data[3 + 4 * n];
            for (std::size_t r = 1; r < f; ++r)
                std::copy(U.data, U.data + q, U.data + q * r);

            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<>
    struct Operator<GRAD, FemFix<FEM_B4>> {
        using Nfa = std::integral_constant<int, 1>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 3>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = Dim::value * q * Nfa::value * f;
            extraR = 3*q;
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
            assert(mem.extraR.size >= 3*q && "Not enough memory for operator initialization in mem.extraR");
            (void) wsz; (void) nfa;
            ScalarType GRAD_P1[3*4] = {-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
            DenseMatrix<ScalarType> GRAD_B4(mem.extraR.data, 3, q, mem.extraR.size);
            const int coef = 256;
            for (std::size_t n = 0; n < q; ++n) {
                ScalarType l[4] = {
                    mem.XYL.data[1 + 4 * n] * mem.XYL.data[2 + 4 * n] * mem.XYL.data[3 + 4 * n],
                    mem.XYL.data[0 + 4 * n] * mem.XYL.data[2 + 4 * n] * mem.XYL.data[3 + 4 * n],
                    mem.XYL.data[0 + 4 * n] * mem.XYL.data[1 + 4 * n] * mem.XYL.data[3 + 4 * n],
                    mem.XYL.data[0 + 4 * n] * mem.XYL.data[1 + 4 * n] * mem.XYL.data[2 + 4 * n]
                };
                for (IndexType d = 0; d < 3; ++d)
                    GRAD_B4(d, n) = coef * (GRAD_P1[d + 3*0] * l[0] + GRAD_P1[d + 3*1] * l[1] + GRAD_P1[d + 3*2] * l[2] + GRAD_P1[d + 3*3] * l[3]);    
            }
            for (std::size_t r = 0; r < f; ++r)
                for (std::size_t n = 0; n < q; ++n)
                    for (IndexType k = 0; k < Dim::value; ++k){
                        ScalarType s = 0;
                        for (IndexType j = 0; j < 3; ++j)
                            s += mem.PSI.data[j + 3*(k+3*r)] * GRAD_B4.data[j + 3*n];
                        U.data[k + Dim::value * (n + q * r)] = s;
                    }
            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};        
        }
    };

    template<>
    struct Dof<FemFix<FEM_B4>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(std::array<uint, DofT::NGEOM_TYPES>{0, 0, 0, 0, 0, 1});
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet >= 0 && idof_on_tet < 1 && "Wrong idof value");
            (void) max_quad_order; (void) idof_on_tet;
            std::array<const double*, 4> XY{XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
            std::array<double, 3> X = {0, 0, 0};
            for (int i = 0; i < 4; ++i)
                for (int d = 0; d < 3; ++d)
                    X[d] += XY[i][d];
            for (int d = 0; d < 3; ++d)  
                X[d] /= 4;      
            f(X, fstorage, 1*fusion, user_data);
            for (int r = 0; r < fusion; ++r)  
                udofs[r].data[0] = fstorage[r];
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_B4>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_B4>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            (void) grad_lmb;
            phi[0] = 256 * lmb(0, 0) * lmb(1, 0) * lmb(2, 0) * lmb(3, 0);
        }
    };

    using BubbleSpace = GenSpace<Bubble, FEM_B4>;
};

#endif //CARNUM_FEM_ANIINTERFACE_SPACES_BUBBLE_4_H