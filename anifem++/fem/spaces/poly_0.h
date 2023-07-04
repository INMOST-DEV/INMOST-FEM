//
// Created by Liogky Alexey on 24.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_POLY_0_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_POLY_0_H

#include "common.h"
namespace Ani{
    template<>
    struct Operator<GRAD, FemFix<FEM_P0>> {
        using Nfa = std::integral_constant<int, 1>;
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
            std::fill(U.data, U.data + wsz, 0);
            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<int K>
    struct OperatorDUDX_FEM_P0 {
        using Nfa = std::integral_constant<int, 1>;
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
            std::fill(U.data, U.data + wsz, 0);
            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };
    
    template<> struct Operator<DUDX, FemFix<FEM_P0>>: public OperatorDUDX_FEM_P0<0> { };
    template<> struct Operator<DUDY, FemFix<FEM_P0>>: public OperatorDUDX_FEM_P0<1> { };
    template<> struct Operator<DUDZ, FemFix<FEM_P0>>: public OperatorDUDX_FEM_P0<2> { };

    template<>
    struct Operator<IDEN, FemFix<FEM_P0>> {
        using Nfa = std::integral_constant<int, 1>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * f;
            extraR = extraI = 0;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            assert(U.size >= wsz && "Not enough memory for operator initialization");
            std::fill(U.data, U.data + wsz, 1);
            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<>
    struct Dof<FemFix<FEM_P0>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(DofT::DofSymmetries({0}, {0}, {0}, {0}, {0}, {1, 0, 0, 0, 0}));
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet == 0 && "Wrong idof value");
            (void) idof_on_tet;
            for (int r = 0; r < fusion; ++r)
                udofs[r].data[0] = 0;
            auto formula = tetrahedron_quadrature_formulas(max_quad_order);
            for (int i = 0; i < formula.GetNumPoints(); ++i){
                auto q = formula.GetPointWeight(i);
                std::array<double, 3> X;
                for (int k = 0; k < 3; ++k) 
                    X[k] = q.p[0]*XYZ.XY0[k] + q.p[1]*XYZ.XY1[k] + q.p[2]*XYZ.XY2[k] + q.p[3]*XYZ.XY3[k]; 
                std::fill(fstorage, fstorage + fusion, 0);  
                f(X, fstorage, 1*fusion, user_data); 
                for (int r = 0; r < fusion; ++r)
                    udofs[r].data[0] += q.w * fstorage[r];
            }   
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_P0>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_P0>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            (void) lmb; (void) grad_lmb;
            phi[0] = 1;
        }
    };

    using P0Space = GenSpace<Polynomial, FEM_P0>;
}
#endif //CARNUM_FEM_ANIINTERFACE_SPACES_POLY_0_H