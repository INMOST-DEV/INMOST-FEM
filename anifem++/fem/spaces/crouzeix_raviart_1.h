//
// Created by Liogky Alexey on 24.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_CROUZEIX_RAVIANT_1_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_CROUZEIX_RAVIANT_1_H

#include "common.h"


namespace Ani{
    template<>
    struct Operator<IDEN, FemFix<FEM_CR1>> {
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
            DenseMatrix<const ScalarType> XYL(mem.XYL.data, 4, q, mem.XYL.size);
            static const unsigned char IPF3[] = {3, 0, 1, 2};
            for (IndexType i = 0; i < nfa; ++i)
                for (std::size_t n = 0; n < q; ++n)
                    U.data[n + nRow * i] = 1 - 3*mem.XYL.data[IPF3[i] + 4 * n];
            for (std::size_t r = 1; r < f; ++r)
                std::copy(U.data, U.data + nfa * q, U.data + nfa * q * r);

            return {U.data, static_cast<std::size_t>(nRow), static_cast<std::size_t>(nCol), U.size};
        }
    };

    template<>
    struct Operator<GRAD, FemFix<FEM_CR1>> {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f* Dim::value;
            extraR = 0;
            extraI = 0;
        }
        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &Uin) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            constexpr auto nfa = Nfa::value;
            constexpr auto dim = Dim::value;
            auto q = mem.q, f = mem.f;
            assert(Uin.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            DenseMatrix<ScalarType> U(Uin.data, nRow, nCol, Uin.size);
            static ScalarType GRAD_P1_[3*4] = {-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
            DenseMatrix<ScalarType> GRAD_P1(GRAD_P1_, 3, 4, 12);
            DenseMatrix<ScalarType> PSI(mem.PSI.data, 3, 3*f, mem.PSI.size);
            static const unsigned char IPF3[] = {3, 0, 1, 2};
            for (std::size_t r = 0; r < f; ++r)
                for (IndexType i = 0; i < nfa; ++i) {
                    for (IndexType k = 0; k < dim; ++k) {
                        ScalarType s = 0;
                        for (IndexType j = 0; j < 3; ++j)
                            s += PSI(j, k+3*r) * GRAD_P1(j, IPF3[i]);
                        U(k, i+nfa*r) = -3*s;
                    }
                    for (std::size_t n = 1; n < q; ++n)
                        for (IndexType k = 0; k < dim; ++k)
                            U(k + dim*n, i+nfa*r) = U(k, i+nfa*r);
                }
            return U;
        }
    };

    template<>
    struct Dof<FemFix<FEM_CR1>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(std::array<uint, DofT::NGEOM_TYPES>{0, 0, 0, 1, 0, 0});
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet >= 0 && idof_on_tet < 6 && "Wrong idof value");
            std::array<const double*, 4> XY{XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
            for (int r = 0; r < fusion; ++r)
                udofs[r][idof_on_tet] = 0;
            int iface = idof_on_tet;
            auto formula = triangle_quadrature_formulas(max_quad_order);
            for (int i = 0; i < formula.GetNumPoints(); ++i){
                auto q = formula.GetPointWeight(i);
                std::array<double, 3> lX;
                for (int k = 0; k < 3; ++k)
                    lX[k] = q.p[0]*XY[(iface+0)%4][k] + q.p[1]*XY[(iface+1)%4][k] + q.p[2]*XY[(iface+2)%4][k];
                f(lX, fstorage, 1*fusion, user_data);
                for (int r = 0; r < fusion; ++r)
                    udofs[r][idof_on_tet] += q.w * fstorage[r];
            }
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_CR1>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_CR1>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            (void) grad_lmb;
            for (int i = 0; i < 4; ++i)
                phi[i] = 1 - 3*lmb((i+3)%4, 0);
        }
    };

    using CR1Space = GenSpace<CrouzeixRaviant, FEM_CR1>;
}
#endif //CARNUM_FEM_ANIINTERFACE_SPACES_CROUZEIX_RAVIANT_1_H