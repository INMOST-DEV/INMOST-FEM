//
// Created by Liogky Alexey on 24.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_RAVIANT_THOMAS_0_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_RAVIANT_THOMAS_0_H

#include "common.h"

namespace Ani{
    template<>
    struct Operator<IDEN, FemFix<FEM_RT0>> {
        using Nfa = std::integral_constant<int, 4>;
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
            DenseMatrix<const ScalarType> XYL(mem.XYL.data, 4, q, mem.XYL.size);
            DenseMatrix<ScalarType> U(Uin.data, nRow, nCol, Uin.size);
            static const unsigned char IPF[] = {0, 1, 2, 3, 1, 2, 3, 0, 0, 2, 3, 1, 0, 1, 3, 2};
            for (std::size_t r = 0; r < f; ++r){
                auto vol = std::fabs(mem.DET.data[r]) / 2;
                DenseMatrix<ScalarType> lU(U.data + U.nRow*nfa*r, U.nRow, nfa, U.nRow*nfa);
                DenseMatrix<ScalarType> lXYP(mem.XYP.data + 12*r, 3, 4, 12);
                for (IndexType i = 0; i < nfa; ++i){
                    const unsigned char* ip = IPF + 4*i;
                    ScalarType* ix[4] = {lXYP.data + 3*ip[0], lXYP.data + 3*ip[1], lXYP.data + 3*ip[2], lXYP.data + 3*ip[3]};
                    auto sqr = tri_area(ix[0], ix[1], ix[2]);
                    sqr /= vol;
                    for (std::size_t n = 0; n < q; ++n)
                        for (IndexType j = 0; j < dim; ++j)
                            U(j + dim * n, i + nfa * r) = sqr *
                                    ( XYL(ip[0], n) * (ix[0][j] - ix[3][j]) + XYL(ip[1], n) * (ix[1][j] - ix[3][j]) + XYL(ip[2], n) * (ix[2][j] - ix[3][j]) );
                }
            }
            return U;
        }
    };

    template<>
    struct Operator<GRAD, FemFix<FEM_RT0>> {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 9>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
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
            std::fill(U.data, U.data + U.nRow * U.nCol, 0);
            static const unsigned char IPF[] = {0,1,2,3, 1,2,3,0, 0,2,3,1, 0,1,3,2};
            static ScalarType GRAD_P1_[3*4] = {-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 1};
            std::array<double, 3*4> Vdat = {0};
            DenseMatrix<ScalarType> GRAD_P1(GRAD_P1_, 3, 4, 12);
            for (std::size_t r = 0; r < f; ++r){
                DenseMatrix<ScalarType> lU(U.data + U.nRow*nfa*r, U.nRow, nfa, U.nRow*nfa);
                DenseMatrix<ScalarType> lXYP(mem.XYP.data + 12*r, 3, 4, 12);
                DenseMatrix<ScalarType> lPSI(mem.PSI.data + 9*r, 3, 3, 9);
                DenseMatrix<ScalarType> V(Vdat.data(), 4, 3, 3*4);
                for (IndexType i = 0; i < 3; ++i)
                    for (IndexType j = 0; j < 4; ++j){
                        ScalarType s = 0;
                        for (IndexType k = 0; k < 3; ++k)
                            s += GRAD_P1(k, j) * lPSI(k, i);
                        V(j, i) = s;
                    }
                ScalarType svol = std::fabs(mem.DET.data[r]) / 2;
                for (IndexType i = 0; i < nfa; ++i){
                    ScalarType vol = svol;
                    const unsigned char* ip = IPF + 4*i;
                    ScalarType* ix[4] = {lXYP.data + 3*ip[0], lXYP.data + 3*ip[1], lXYP.data + 3*ip[2], lXYP.data + 3*ip[3]};
                    auto sqr = tri_area(ix[0], ix[1], ix[2]);
                    sqr /= vol;
                    for (IndexType j = 0; j < 3; ++j)
                        for (IndexType k = 0; k < 3; ++k)
                            U(k + 3*j, i + nfa * r) = sqr *
                                    ( V(ip[0], k) * (ix[0][j] - ix[3][j]) + V(ip[1], k) * (ix[1][j] - ix[3][j]) + V(ip[2], k) * (ix[2][j] - ix[3][j]) );
                }
                for (std::size_t n = 1; n < q; ++n)
                    for (IndexType i = 0; i < nfa; ++i)
                        for (IndexType l = 0; l < dim; ++l)
                            U(l + dim * n, i + nfa * r) = U(l, i + nfa * r);

            }
            return U;
        }
    };

    template<>
    struct Operator<DIV, FemFix<FEM_RT0>> {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = 0;
            extraI = 0;
        }
        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &Uin) {
            auto nRow = Dim::value * mem.q, nCol = Nfa::value * mem.f;
            auto wsz = nRow * nCol;
            constexpr auto nfa = Nfa::value;
            //constexpr auto dim = Dim::value;
            auto q = mem.q, f = mem.f;
            assert(Uin.size >= wsz && "Not enough memory for operator initialization");
            (void) wsz;
            DenseMatrix<ScalarType> U(Uin.data, nRow, nCol, Uin.size);
            static const unsigned char IPF[] = {0,1,2,3, 1,2,3,0, 0,2,3,1, 0,1,3,2};
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
                ScalarType svol = std::fabs(mem.DET.data[r]) / 2;
                for (IndexType i = 0; i < nfa; ++i){
                    ScalarType vol = svol;
                    const unsigned char* ip = IPF + 4*i;
                    ScalarType* ix[4] = {lXYP.data + 3*ip[0], lXYP.data + 3*ip[1], lXYP.data + 3*ip[2], lXYP.data + 3*ip[3]};
                    auto sqr = tri_area(ix[0], ix[1], ix[2]);
                    sqr /= vol;
                    {
                        double s = 0;
                        for (IndexType j = 0; j < 3; ++j)
                            s += ( V(ip[0], j) * (ix[0][j] - ix[3][j]) + V(ip[1], j) * (ix[1][j] - ix[3][j]) + V(ip[2], j) * (ix[2][j] - ix[3][j]) );
                        U(0, i + nfa * r) = sqr*s;    
                    }
                    for (std::size_t n = 1; n < q; ++n)
                        U(n, i + nfa * r) = U(0, i + nfa * r);
                }    
            }
            return U;
        }

    };

    template<>
    struct Operator<CURL, FemFix<FEM_RT0>> {
        using Nfa = std::integral_constant<int, 4>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, 0>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Usz = q * Nfa::value * f * Dim::value;
            extraR = 0;
            extraI = 0;
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

    template<>
    struct Dof<FemFix<FEM_RT0>>{
        static inline DofT::UniteDofMap Map(){
            return DofT::UniteDofMap(DofT::DofSymmetries({0}, {0}, {0}, {0}, {1, 0, 0}));
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, double* fstorage, ArrayView<>* udofs, int idof_on_tet, int fusion, void* user_data = nullptr, uint max_quad_order = 5){
            assert(idof_on_tet >= 0 && idof_on_tet < 4 && "Wrong idof value");
            std::array<const double*, 4> XY{XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
            double nrm[3];
            face_normal(XY[idof_on_tet], XY[(idof_on_tet+1)%4], XY[(idof_on_tet+2)%4], XY[(idof_on_tet+3)%4], nrm);
            for (int r = 0; r < fusion; ++r)
                udofs[r][idof_on_tet] = 0;
            auto formula = triangle_quadrature_formulas(max_quad_order);
            for (int i = 0; i < formula.GetNumPoints(); ++i){
                auto q = formula.GetPointWeight(i);
                std::array<double, 3> lX;
                for (int k = 0; k < 3; ++k)
                    lX[k] = q.p[0]*XY[(idof_on_tet+0)%4][k] + q.p[1]*XY[(idof_on_tet+1)%4][k] + q.p[2]*XY[(idof_on_tet+2)%4][k];
                f(lX, fstorage, 3*fusion, user_data);
                for (int r = 0; r < fusion; ++r)
                    udofs[r][idof_on_tet] += q.w * (fstorage[3*r+0] * nrm[0] + fstorage[3*r+1] * nrm[1] + fstorage[3*r+2] * nrm[2]);
            }
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            std::array<double, Operator<IDEN, FemFix<FEM_RT0>>::Dim::value*FUSION> mem = {0};
            interpolate<EvalFunc>(XYZ, f, mem.data(), udofs.data(), idof_on_tet, FUSION, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    template<>
    struct Basis<FemFix<FEM_RT0>>{
        static void eval(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi){
            using namespace FT;
            auto V = abs(det3x3(vertcat({grad_lmb.col(0), grad_lmb.col(1), grad_lmb.col(2)})));
            for (int i = 0; i < 4; ++i)
                phi[i] = norm(grad_lmb.col((i+3)%4))/V * (
                        lmb((i+0)%4, 0)*cross( grad_lmb.col((i+1)%4).T(), grad_lmb.col((i+2)%4).T() ) + 
                        lmb((i+1)%4, 0)*cross( grad_lmb.col((i+2)%4).T(), grad_lmb.col((i+0)%4).T() ) +
                        lmb((i+2)%4, 0)*cross( grad_lmb.col((i+0)%4).T(), grad_lmb.col((i+1)%4).T() )
                        ); 
        }
    };

    using RT0Space = GenSpace<RaviantThomas, FEM_RT0>;
}
#endif //CARNUM_FEM_ANIINTERFACE_SPACES_RAVIANT_THOMAS_0_H