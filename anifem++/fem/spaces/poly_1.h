//
// Created by Liogky Alexey on 24.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_POLY_1_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_POLY_1_H

#include "common.h"

namespace Ani{
    template<>
    struct Operator<IDEN, FemFix<FEM_P1>> {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 1>;

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
            constexpr auto nfa = Nfa::value;
            auto q = mem.q, f = mem.f;
            assert(U.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            for (IndexType i = 0; i < nfa; ++i)
                for (std::size_t n = 0; n < q; ++n)
                    U.data[n + nRow * i] = mem.XYL.data[i + 4 * n];
            for (std::size_t r = 1; r < f; ++r)
                std::copy(U.data, U.data + nfa * q, U.data + nfa * q * r);

            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<int K>
    struct OperatorDUDX_FEM_P1 {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 0>;
        using NCRD = std::integral_constant<int, K>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = Dim::value * q * Nfa::value * f;
            extraR = extraI = 0;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            assert(U.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            for (std::size_t r = 0; r < mem.f; ++r) {
                auto lPSI = mem.PSI.data + 9*r;
                ScalarType s[4*3];
                s[0] = -lPSI[0 + 3*0]-lPSI[1 + 3*0]-lPSI[2 + 3*0];
                s[1] = -lPSI[0 + 3*1]-lPSI[1 + 3*1]-lPSI[2 + 3*1];
                s[2] = -lPSI[0 + 3*2]-lPSI[1 + 3*2]-lPSI[2 + 3*2];
                s[3] = lPSI[0 + 3*0];
                s[4] = lPSI[0 + 3*1];
                s[5] = lPSI[0 + 3*2];
                s[6] = lPSI[1 + 3*0];
                s[7] = lPSI[1 + 3*1];
                s[8] = lPSI[1 + 3*2];
                s[9] = lPSI[2 + 3*0];
                s[10] = lPSI[2 + 3*1];
                s[11] = lPSI[2 + 3*2];
                for (IndexType i = 0; i < 4; ++i)
                    for (std::size_t n = 0; n < mem.q; ++n)
                            U.data[n + mem.q * (i + Nfa::value * r)] = s[K + 3*i];
            }

            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<> struct Operator<DUDX, FemFix<FEM_P1>>: public OperatorDUDX_FEM_P1<0> { };
    template<> struct Operator<DUDY, FemFix<FEM_P1>>: public OperatorDUDX_FEM_P1<1> { };
    template<> struct Operator<DUDZ, FemFix<FEM_P1>>: public OperatorDUDX_FEM_P1<2> { };

    template<>
    struct Operator<GRAD, FemFix<FEM_P1>> {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = Dim::value * q * Nfa::value * f;
            extraR = extraI = 0;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            assert(U.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            for (std::size_t r = 0; r < mem.f; ++r) {
                auto lPSI = mem.PSI.data + 9*r;
                ScalarType s[4*3];
                s[0] = -lPSI[0 + 3*0]-lPSI[1 + 3*0]-lPSI[2 + 3*0];
                s[1] = -lPSI[0 + 3*1]-lPSI[1 + 3*1]-lPSI[2 + 3*1];
                s[2] = -lPSI[0 + 3*2]-lPSI[1 + 3*2]-lPSI[2 + 3*2];
                s[3] = lPSI[0 + 3*0];
                s[4] = lPSI[0 + 3*1];
                s[5] = lPSI[0 + 3*2];
                s[6] = lPSI[1 + 3*0];
                s[7] = lPSI[1 + 3*1];
                s[8] = lPSI[1 + 3*2];
                s[9] = lPSI[2 + 3*0];
                s[10] = lPSI[2 + 3*1];
                s[11] = lPSI[2 + 3*2];
                for (IndexType i = 0; i < 4; ++i)
                    for (std::size_t n = 0; n < mem.q; ++n)
                        for (IndexType k = 0; k < 3; ++k)
                            U.data[k + Dim::value * (n + mem.q * (i + Nfa::value * r))] = s[k + 3*i];
            }

            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<>
    struct Dof<FemFix<FEM_P1>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(std::array<uint, DofT::NGEOM_TYPES>{1, 0, 0, 0, 0, 0});
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet >= 0 && idof_on_tet < 4 && "Wrong idof value");
            (void) max_quad_order;
            std::array<const double*, 4> XY{XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
            std::array<double, 3> X;
            std::copy(XY[idof_on_tet], XY[idof_on_tet]+3, X.data());
            f(X, fstorage, 1*fusion, user_data);
            for (int r = 0; r < fusion; ++r)  
                udofs[r].data[idof_on_tet] = fstorage[r];
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_P1>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_P1>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            (void) grad_lmb;
            for (int i = 0; i < 4; ++i)
                phi[i] = lmb(i, 0);
        }
    };

    using P1Space = GenSpace<Polynomial, FEM_P1>;
}
#endif //CARNUM_FEM_ANIINTERFACE_SPACES_POLY_1_H