//
// Created by Liogky Alexey on 21.03.2022.
//

#ifndef CARNUM_OPERATORS_H
#define CARNUM_OPERATORS_H

#include "fem_memory.h"
#include "geometry.h"
#include "tetdofmap.h"
#include <array>
#include <cassert>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <cstddef>
#include <tuple>
#include <algorithm>
#include <limits>
#include <utility>


namespace Ani{
    ///Available types of finite elements
    enum FiniteElement {
        FEM_P0 = 1, ///< piecewise constant
        FEM_P1 = 2, ///< continuous piecewise linear
        FEM_P2 = 3, ///< continuous piecewise quadratic
        FEM_P3 = 4, ///< continuous piecewise cubic
        FEM_RT0 = 21, ///< lower order Raviart-Thomas finite elements
        FEM_ND0 = 25, ///< lower order Nedelec finite elements
        FEM_CR1 = 31, ///< Crouzeix-Raviart finite elements
        FEM_B4 = 1001, ///< classical bubble function
    };
    ///Available linear differential operators
    enum OperatorType {
        IDEN = 1, ///< identity operator
        GRAD = 2, ///< gradient operator
        DIV = 3, ///< divergence operator
        CURL = 4, ///< curl operator
        DUDX = 6, ///< partial derivative d/dx
        DUDY = 7, ///< partial derivative d/dy
        DUDZ = 8  ///< partial derivative d/dz
    };

    ///Compute action of a linear operators on FEM variable
    ///@tparam OPERATOR is number of OPERATOR, see OperatorType
    ///@tparam FEMTYPE is type of FEM space of variable
    template<int OPERATOR, typename FEMTYPE>
    struct Operator;

    ///Simple finite type
    ///@tparam OP is number of finite type, usually one of numbers from enum FiniteElement
    ///@see FiniteElement
    template<int OP>
    using FemFix = std::integral_constant<int, OP>;

    ///Vector of same simple finite type
    ///@tparam OP is number of finite type, usually one of numbers from enum FiniteElement
    ///@tparam DIM is count of repetitions of the simple finite type in vector
    ///@see FiniteElement
    template<int DIM, int OP>
    struct FemVec {
        using Dim = std::integral_constant<int, DIM>;
        using Base = FemFix<OP>;
    };

    namespace FemComDetails {
        template<bool isFem, int DIM, typename FEMTYPE>
        struct FemVecTImpl;

        template<bool isFem, class... Types>
        struct FemComImpl;

        template<class T>
        struct CheckFemVar;

        template<std::size_t i, typename... Items>
        struct CheckFemVars;

        template<int SUM, int OP, class... Types>
        struct DimCounter;

        template<int SUM, int OP, class... Types>
        struct NfaCounter;

        template<int SUM, int OP, class... Types>
        struct NpartsCounter;

        template<int SUM, int OP, class... Types>
        struct OrderCounter;
    }

    ///Vector of any same finite type (including composite finite types)
    ///@tparam FEMTYPE is the finite type
    ///@tparam DIM is count of repetitions of the finite type
    template<int DIM, typename FEMTYPE>
    using FemVecT = FemComDetails::FemVecTImpl<FemComDetails::CheckFemVar<FEMTYPE>::value, DIM, FEMTYPE>;

    ///Composition of any finite types
    template< class... Types >
    using FemCom = FemComDetails::FemComImpl<FemComDetails::CheckFemVars<0, Types ...>::Val::value, Types ...>;

    template<int OPERATOR, int DIM, int FEM_TYPE>
    struct Operator<OPERATOR, FemVec<DIM, FEM_TYPE>> {
        using Base = Operator<OPERATOR, FemFix<FEM_TYPE>>;
        using Nfa = std::integral_constant<int, DIM * Base::Nfa::value>;
        using Dim = std::integral_constant<int, DIM * Base::Dim::value>;
        using Order = std::integral_constant<int, Base::Order::value>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            return Base().template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        }

        template<typename ScalarType, typename IndexType>
        inline static BandDenseMatrix<DIM, ScalarType, IndexType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U){
            DenseMatrix<ScalarType> A = Base().apply(mem, U);
            BandDenseMatrix<DIM, ScalarType, IndexType> res;
            for (IndexType i = 0; i < DIM; ++i){
                res.data[i] = A;
                res.stRow[i] = Base::Dim::value*i;
                res.stCol[i] = Base::Nfa::value*i;
            }
            res.stRow[DIM] = Dim::value;
            res.stCol[DIM] = Nfa::value;
            return res;
        }
    private:
        using DummyType = typename std::enable_if<OPERATOR==IDEN || OPERATOR==GRAD || OPERATOR==DUDX || OPERATOR==DUDY || OPERATOR==DUDZ, bool>::type;
    };

    template<int OPERATOR, int DIM, typename FEM_TYPE>
    struct Operator<OPERATOR, FemVecT<DIM, FEM_TYPE>> {
        using Base = Operator<OPERATOR, FEM_TYPE>;
        using Nfa = std::integral_constant<int, DIM * Base::Nfa::value>;
        using Dim = std::integral_constant<int, DIM * Base::Dim::value>;
        using Order = std::integral_constant<int, Base::Order::value>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            return Base().template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        }

        template<typename ScalarType, typename IndexType>
        inline static auto apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U){
            auto A = convToBendMx(Base().apply(mem, U), Dim::value, Nfa::value);
            auto ANpart = decltype(A)::Nparts::value;
            BandDenseMatrix<DIM*ANpart, ScalarType, IndexType> res;
            for (IndexType i = 0; i < DIM; ++i){
                for (int k = 0; k < ANpart; ++k) {
                    res.data[i*ANpart + k] = A.data[k];
                    res.stRow[i*ANpart + k] = Base::Dim::value * i + A.stRow[k];
                    res.stCol[i*ANpart + k] = Base::Nfa::value * i + A.stCol[k];
                }
            }
            res.stRow[DIM*ANpart] = Dim::value;
            res.stCol[DIM*ANpart] = Nfa::value;
            return res;
        }
    private:
        using DummyType = typename std::enable_if<OPERATOR==IDEN || OPERATOR==GRAD || OPERATOR==DUDX || OPERATOR==DUDY || OPERATOR==DUDZ, bool>::type;
    };

    template<int OPERATOR, class... Types>
    struct Operator<OPERATOR, FemCom<Types...>> {
        template<int i>
        using Base = Operator<OPERATOR, std::tuple_element<i, typename FemCom<Types...>::Base>>;
    private:
        template<std::size_t i, typename ScalarType, typename IndexType>
        struct MemoryRequirements {
            inline static void
            Impl(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
                IndexType _Usz = 0, _extraR = 0, _extraI = 0;
                Base<i>().template memoryRequirements<ScalarType, IndexType>(f, q, _Usz, _extraR, _extraI);
                Usz += _Usz;
                if (extraR < _extraR) extraR = _extraR;
                if (extraI < _extraI) extraI = _extraI;
                MemoryRequirements< i + 1, ScalarType, IndexType>::Impl(f, q, &Usz, &extraR, &extraI);
            }
        };
        template<typename ScalarType, typename IndexType>
        struct MemoryRequirements<std::tuple_size<typename FemCom<Types...>::Base>::value, ScalarType, IndexType> {
            inline static void
            Impl(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {}
        };

        template <std::size_t I, std::size_t N, std::size_t PART, std::size_t NPART, std::size_t DIMOFFSET, std::size_t NFAOFFSET, typename ScalarType, typename IndexType>
        struct Apply{
            inline static void Impl(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U, BandDenseMatrix<NPART, ScalarType, IndexType>& res){
                auto A = convToBendMx(Base<I>().apply(mem, U));
                auto ANpart = decltype(A)::Nparts::value;
                int rUsz = 0;
                for (int k = 0; k < ANpart; ++k) {
                    res.data[PART + k] = A.data[k];
                    res.stRow[PART + k] = DIMOFFSET + A.stRow[k];
                    res.stCol[PART + k] = NFAOFFSET + A.stCol[k];
                    rUsz += (A.stRow[k + 1] - A.stRow[k]) * (A.stCol[k + 1] - A.stCol[k]);
                }
                rUsz *= mem.q * mem.f;
                U.data += rUsz;
                U.size -= rUsz;
                Apply<I+1, N, PART + ANpart, NPART, DIMOFFSET + Base<I>::Dim::value, NFAOFFSET + Base<I>::Nfa::value, ScalarType, IndexType>::Impl(mem, U, res);
            }
        };
        template <std::size_t N, std::size_t NPART, std::size_t DIMOFFSET, std::size_t NFAOFFSET, typename ScalarType, typename IndexType>
        struct Apply<N, N, NPART, NPART, DIMOFFSET, NFAOFFSET, ScalarType, IndexType>{
            inline static void Impl(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U, BandDenseMatrix<NPART, ScalarType, IndexType>& res){}
        };
    public:
        using Nfa = std::integral_constant<int, FemComDetails::NfaCounter<0, OPERATOR, Types...>::value>;
        using Dim = std::integral_constant<int, FemComDetails::DimCounter<0, OPERATOR, Types...>::value>;
        using Order = std::integral_constant<int, FemComDetails::OrderCounter<0, OPERATOR, Types...>::value>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            MemoryRequirements<0, ScalarType, IndexType>::Impl(f, q, Usz, extraR, extraI);
        }

        template<typename ScalarType, typename IndexType>
        inline static auto
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U){
            auto Nparts = FemComDetails::NpartsCounter<0, OPERATOR, Types...>::value;
            BandDenseMatrix<Nparts, ScalarType, IndexType> res;

            ArrayView<ScalarType> tmp = U;
            Apply<0, std::tuple_size<typename FemCom<Types...>::Base>::value, 0, Nparts, 0, 0, ScalarType, IndexType>::Impl(mem, tmp, res);

            res.stRow[Nparts] = Dim::value;
            res.stCol[Nparts] = Nfa::value;
            return res;
        }
    private:
        using DummyType = typename std::enable_if<OPERATOR==IDEN || OPERATOR==GRAD || OPERATOR==DUDX || OPERATOR==DUDY || OPERATOR==DUDZ, bool>::type;
    };

    template<int NCRD, int FEM_TYPE>
    struct OperatorDUDXHelper {
        using Base = Operator<GRAD, FemFix<FEM_TYPE>>;
        using Nfa = std::integral_constant<int, Base::Nfa::value>;
        using Dim = std::integral_constant<int, Base::Dim::value / 3>;
        using Order = std::integral_constant<int, Base::Order::value>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            return Base().template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U){
            DenseMatrix<ScalarType> A = Base().apply(mem, U);
            DenseMatrix<ScalarType> B(A.data, mem.q * Dim::value, A.nCol, A.size);
            for (std::size_t r = 0; r < mem.f; ++r)
                for (int i = 0; i < Base::Nfa::value; ++i)
                    for (std::size_t n = 0; n < mem.q; ++n)
                        for (int k = 0; k < Dim::value; ++k)
                            B(k + n*Dim::value, i + Nfa::value*r) = A(NCRD + 3*k + n * Base::Dim::value, i + Nfa::value*r);

            return B;
        }
    };
    template<int FEM_TYPE>
    struct Operator<DUDX, FemFix<FEM_TYPE>>: public OperatorDUDXHelper<0, FEM_TYPE> { };
    template<int FEM_TYPE>
    struct Operator<DUDY, FemFix<FEM_TYPE>>: public OperatorDUDXHelper<1, FEM_TYPE> { };
    template<int FEM_TYPE>
    struct Operator<DUDZ, FemFix<FEM_TYPE>>: public OperatorDUDXHelper<2, FEM_TYPE> { };


    template<int FEM_TYPE>
    struct Operator<DIV, FemVec<3, FEM_TYPE>>{
        using Base = Operator<GRAD, FemFix<FEM_TYPE>>;
        using Nfa = std::integral_constant<int, 3 * Base::Nfa::value>;
        using Dim = std::integral_constant<int, 1>;
        using Order = std::integral_constant<int, Base::Order::value>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Base().template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
            extraR += Usz;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U){
            auto Usz = 3*mem.q*mem.f*Base::Nfa::value;
            assert(mem.extraR.size >= Usz && "mem.extraR has not enough memory");
            mem.extraR.size -= Usz; 
            ArrayView<ScalarType> V(mem.extraR.data + mem.extraR.size, Usz); 
            DenseMatrix<ScalarType> A = Base().apply(mem, V); 
            DenseMatrix<ScalarType> lU(U.data, mem.q, 3*Base::Nfa::value*mem.f, U.size);
            for (std::size_t r = 0; r < mem.f; ++r)
                for (int i = 0; i < Base::Nfa::value; ++i)
                    for (std::size_t n = 0; n < mem.q; ++n)
                        for (int k = 0; k < 3; ++k)
                            lU(n, i + Base::Nfa::value*(k + 3 * r)) = A(k + 3*n, i + Base::Nfa::value * r);
            mem.extraR.size += Usz;
            return lU;
        }

    private:
        using DummyType = typename std::enable_if<Base::Dim::value == 3, bool>::type;
    };

    template<int FEM_TYPE>
    struct Operator<CURL, FemVec<3, FEM_TYPE>>{
        using Base = Operator<GRAD, FemFix<FEM_TYPE>>;
        using Nfa = std::integral_constant<int, 3 * Base::Nfa::value>;
        using Dim = std::integral_constant<int, 3>;
        using Order = std::integral_constant<int, Base::Order::value>;

        template<typename ScalarType, typename IndexType>
        inline static void memoryRequirements(int f, int q, std::size_t &Usz, std::size_t &extraR, std::size_t &extraI) {
            Base().template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
            extraR += Usz;
            Usz *= 3;
        }

        template<typename ScalarType, typename IndexType>
        inline static DenseMatrix<ScalarType>
        apply(AniMemory<ScalarType, IndexType> &mem, ArrayView<ScalarType> &U){
            auto Usz = 3*mem.q*mem.f*Base::Nfa::value;
            assert(mem.extraR.size >= Usz && "mem.extraR has not enough memory");
            mem.extraR.size -= Usz;
            ArrayView<ScalarType> V(mem.extraR.data + mem.extraR.size, Usz); 
            DenseMatrix<ScalarType> A = Base().apply(mem, V);
            DenseMatrix<ScalarType> lU(U.data, 3*mem.q, 3*Base::Nfa::value*mem.f, U.size);
            static const unsigned char IJL[] = {1,2,0,  2,1,0,  0,2,1, 2,0,1,  0,1,2,  1,0,2};
            std::fill(U.data, U.data + 3*mem.q*3*Base::Nfa::value*mem.f, 0);
            for (std::size_t r = 0; r < mem.f; ++r)
                for (std::size_t n = 0; n < mem.q; ++n){
                    //code below equivalent to this but more effective
                    //for (int p = 0; p < 6; ++p){
                    //    static const char sg_arr[] = {1, -1, -1, 1, 1, -1};
                    //    auto i = IJL[3*p + 0], j = IJL[3*p + 1], l = IJL[3*p + 2];
                    //    auto sg = sg_arr[p];
                    //    for (int d = 0; d < Base::Nfa::value; ++d)
                    //        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = sg*A(j + n*3, d + Base::Nfa::value*r);
                    //}
                    auto i = IJL[3*0 + 0], j = IJL[3*0 + 1], l = IJL[3*0 + 2];
                    for (int d = 0; d < Base::Nfa::value; ++d)
                        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = A(j + n*3, d + Base::Nfa::value*r);
                    i = IJL[3*1 + 0], j = IJL[3*1 + 1], l = IJL[3*1 + 2];
                    for (int d = 0; d < Base::Nfa::value; ++d)
                        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = -A(j + n*3, d + Base::Nfa::value*r);
                    i = IJL[3*2 + 0], j = IJL[3*2 + 1], l = IJL[3*2 + 2];
                    for (int d = 0; d < Base::Nfa::value; ++d)
                        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = -A(j + n*3, d + Base::Nfa::value*r);
                    i = IJL[3*3 + 0], j = IJL[3*3 + 1], l = IJL[3*3 + 2];
                    for (int d = 0; d < Base::Nfa::value; ++d)
                        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = A(j + n*3, d + Base::Nfa::value*r);
                    i = IJL[3*4 + 0], j = IJL[3*4 + 1], l = IJL[3*4 + 2];
                    for (int d = 0; d < Base::Nfa::value; ++d)
                        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = A(j + n*3, d + Base::Nfa::value*r);
                    i = IJL[3*5 + 0], j = IJL[3*5 + 1], l = IJL[3*5 + 2];
                    for (int d = 0; d < Base::Nfa::value; ++d)
                        lU(i + 3*n, d + Base::Nfa::value * (l + 3 * r)) = -A(j + n*3, d + Base::Nfa::value*r);
                }
            mem.extraR.size += Usz;

            return lU;
        }

    private:
        using DummyType = typename std::enable_if<Base::Dim::value == 3, bool>::type;
    };

    struct ApplyOpBase{
        virtual BandDenseMatrixX<> operator()(AniMemoryX<> &mem, ArrayView<> &U) const = 0;
        virtual OpMemoryRequirements getMemoryRequirements(uint nquadpoints, uint fusion = 1) const = 0;
        virtual uint Nfa() const = 0;
        virtual uint Dim() const = 0;
        virtual uint Order() const { return std::numeric_limits<uint>::max(); }
        virtual uint ActualType() const { return 0; }
        virtual bool operator==(const ApplyOpBase& otherOp) const = 0;
        bool operator!=(const ApplyOpBase& otherOp) const { return !(*this == otherOp); }
        virtual std::shared_ptr<ApplyOpBase> Copy() const = 0;
    };

    struct ApplyOp{
        std::shared_ptr<ApplyOpBase> m_invoker;

        ApplyOp() = default;
        explicit ApplyOp(const std::shared_ptr<ApplyOpBase>& base): m_invoker(base) {};
        template<typename Op>
        explicit ApplyOp(const Op& f, typename std::enable_if<std::is_base_of<ApplyOpBase, Op>::value>::type* = 0): m_invoker{new Op(f)} {}
        template<typename Op>
        explicit ApplyOp(Op&& f, typename std::enable_if<std::is_base_of<ApplyOpBase, Op>::value>::type* = 0): m_invoker{new Op(std::move(f))} {}

        BandDenseMatrixX<> operator()(AniMemoryX<> &mem, ArrayView<> &U) const { return m_invoker->operator()(mem, U); }
        OpMemoryRequirements getMemoryRequirements(uint nquadpoints, uint fusion = 1) const { return m_invoker->getMemoryRequirements(nquadpoints, fusion); }
        uint Nfa() const { return m_invoker->Nfa(); }
        uint Dim() const { return m_invoker->Dim(); }
        uint Order() const { return m_invoker->Order(); }
        uint ActualType() const { return m_invoker->ActualType(); }
        bool operator==(const ApplyOp& otherOp) const { return m_invoker.get() == otherOp.m_invoker.get() || (m_invoker && otherOp.m_invoker && *m_invoker == *otherOp.m_invoker); }
        bool operator!=(const ApplyOp& otherOp) const { return !(*this == otherOp); }
    };

    struct ApplyOpCustom: public ApplyOpBase{
        using ApplyOpFunc = std::function<BandDenseMatrixX<>(AniMemoryX<> &mem, ArrayView<> &U)>;
        using MemReqGetter = std::function<OpMemoryRequirements(uint nquadpoints, uint fusion)>;
        ApplyOpFunc m_applyOp;
        MemReqGetter m_memReq;
        uint m_dim = 0;
        uint m_nfa = 0;
        uint m_order = std::numeric_limits<uint>::max();
        uint m_unique_label = 1;

        ApplyOpCustom() = default;
        ApplyOpCustom(const ApplyOpCustom&) = default;
        ApplyOpCustom(ApplyOpCustom&&) = default;
        ApplyOpCustom(uint unique_label, ApplyOpFunc applyOp, MemReqGetter memReq, uint dim, uint nfa, uint order = std::numeric_limits<uint>::max()): m_applyOp{std::move(applyOp)},  m_memReq{std::move(memReq)}, m_dim{dim}, m_nfa{nfa}, m_order{order}, m_unique_label{unique_label} {}
        BandDenseMatrixX<> operator()(AniMemoryX<> &mem, ArrayView<> &U) const override { return m_applyOp(mem, U); }
        OpMemoryRequirements getMemoryRequirements(uint nquadpoints, uint fusion = 1) const override { return m_memReq(nquadpoints, fusion); }
        uint Nfa() const override { return m_nfa; }
        uint Dim() const override { return m_dim; }
        uint Order() const override { return m_order; }
        uint ActualType() const override { return 1; }
        bool operator==(const ApplyOpBase& otherOp) const override { return otherOp.ActualType() == ActualType() && m_unique_label == static_cast<const ApplyOpCustom&>(otherOp).m_unique_label; }
        std::shared_ptr<ApplyOpBase> Copy() const { return std::make_shared<ApplyOpCustom>(*this); }
    };
    /// @brief Generator of ApplyOp from templated spaces, shouldn't used in working codes, just helper for testing
    template<int OPERATOR, typename FEM_TYPE>
    struct ApplyOpFromTemplate: public Ani::ApplyOpBase{
        template <int Dummy, bool isDenseMatrix = true>
        struct Appl{
            using Nparts = std::integral_constant<int, 1>;
            static Ani::BandDenseMatrixX<> appl(Ani::AniMemoryX<> &mem, Ani::ArrayView<> &U){
                using namespace Ani;
                DenseMatrix<> lU = Operator<OPERATOR, FEM_TYPE>::template apply<double, int>(mem, U);
                uint bshift = mem.busy_mtx_parts > 0 ? 1 : 0;
                BandDenseMatrixX<> res(1, mem.MTX.data + mem.busy_mtx_parts, mem.MTXI_ROW.data + mem.busy_mtx_parts + bshift, mem.MTXI_COL.data + mem.busy_mtx_parts + bshift);
                res.data[0] = lU;
                res.stRow[0] = 0; res.stRow[1] = Operator<OPERATOR, FEM_TYPE>::Dim::value;
                res.stCol[0] = 0; res.stCol[1] = Operator<OPERATOR, FEM_TYPE>::Nfa::value;
                ++mem.busy_mtx_parts;
                return res;
            }
            static Ani::OpMemoryRequirements memreq(uint nquadpoints, uint fusion = 1){
                Ani::OpMemoryRequirements req;
                Ani::Operator<OPERATOR, FEM_TYPE>::template memoryRequirements<double, int>(fusion, nquadpoints, req.Usz, req.extraRsz, req.extraIsz);
                req.mtx_parts = Nparts::value;
                return req; 
            }
        };
        template <int Dummy>
        struct Appl<Dummy, false>{
            using Nparts = typename decltype(Ani::Operator<OPERATOR, FEM_TYPE>::template apply<double, int>(std::declval<Ani::AniMemory<>&>(), std::declval<Ani::ArrayView<>&>()))::Nparts;
            static Ani::BandDenseMatrixX<> appl(Ani::AniMemoryX<> &mem, Ani::ArrayView<> &U){
                using namespace Ani;
                BandDenseMatrix<Nparts::value, double, int> lU = Operator<OPERATOR, FEM_TYPE>::template apply<double, int>(mem, U);
                uint bshift = mem.busy_mtx_parts > 0 ? 1 : 0;
                BandDenseMatrixX<> res(Nparts::value, mem.MTX.data + mem.busy_mtx_parts, mem.MTXI_ROW.data + mem.busy_mtx_parts + bshift, mem.MTXI_COL.data + mem.busy_mtx_parts + bshift);
                for (std::size_t p = 0; p < Nparts::value; ++p)
                    res.data[p] = lU.data[p];
                std::copy(lU.stCol, lU.stCol + Nparts::value+1, res.stCol); 
                std::copy(lU.stRow, lU.stRow + Nparts::value+1, res.stRow); 
                mem.busy_mtx_parts += Nparts::value;
                return res;
            }
            static Ani::OpMemoryRequirements memreq(uint nquadpoints, uint fusion = 1){
                Ani::OpMemoryRequirements req;
                Ani::Operator<OPERATOR, FEM_TYPE>::template memoryRequirements<double, int>(fusion, nquadpoints, req.Usz, req.extraRsz, req.extraIsz);
                req.mtx_parts = Nparts::value;
                return req; 
            }
        };
        Ani::BandDenseMatrixX<> operator()(Ani::AniMemoryX<> &mem, Ani::ArrayView<> &U) const override { 
            constexpr bool isDenseMtx = std::is_same<Ani::DenseMatrix<>, decltype(Ani::Operator<OPERATOR, FEM_TYPE>::template apply<double, int>(std::declval<Ani::AniMemory<>&>(), std::declval<Ani::ArrayView<>&>()))>::value;
            return Appl<0, isDenseMtx>::appl(mem, U); 
        }
        Ani::OpMemoryRequirements getMemoryRequirements(uint nquadpoints, uint fusion = 1) const override {
            constexpr bool isDenseMtx = std::is_same<Ani::DenseMatrix<>, decltype(Ani::Operator<OPERATOR, FEM_TYPE>::template apply<double, int>(std::declval<Ani::AniMemory<>&>(), std::declval<Ani::ArrayView<>&>()))>::value;
            return Appl<0, isDenseMtx>::memreq(nquadpoints, fusion);
        }
        uint Nfa() const override { return Ani::Operator<OPERATOR, FEM_TYPE>::Nfa::value; }
        uint Dim() const override { return Ani::Operator<OPERATOR, FEM_TYPE>::Dim::value; }
        uint Order() const override { return Ani::Operator<OPERATOR, FEM_TYPE>::Order::value; }
        uint ActualType() const override { return -1; }
        bool operator==(const ApplyOpBase& otherOp) const override { return this == &otherOp; }
        std::shared_ptr<ApplyOpBase> Copy() const override { return std::make_shared<ApplyOpFromTemplate<OPERATOR, FEM_TYPE>>(*this); }
    };

    template<typename FEMTYPE>
    struct Dof{
        //should define static method Map() that return any child of BaseDofMap 
    };
    template<int DIM, int OP>
    struct Dof<FemVec<DIM, OP>>{
        static inline auto Map(){
            return DofT::VectorDofMapC<decltype(Dof<FemFix<OP>>::Map())>(DIM, Dof<FemFix<OP>>::Map());
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            constexpr auto LNFA = Operator<IDEN, FemFix<OP>>::Nfa::value;
            constexpr auto LDIM = Operator<IDEN, FemFix<OP>>::Dim::value;
            uint nvar = idof_on_tet / LNFA, ldof_id = idof_on_tet % LNFA;
            std::array<ArrayView<>, FUSION> new_udofs;
            for (uint r = 0; r < FUSION; ++r)
                new_udofs[r] = ArrayView<>(udofs[r].data + nvar*LNFA, LNFA);
            Dof<FemFix<OP>>::template interpolate<FUSION>(XYZ, 
                [&f, nvar](const std::array<double, 3>& X, double* res, uint dim, void* user_data)->int{
                    assert(dim == LDIM*FUSION && "Wrong expected dimension");
                    std::array<double, LDIM*DIM*FUSION> mem;
                    f(X, mem.data(), LDIM*DIM*FUSION, user_data);
                    for (uint i = 0; i < FUSION; ++i)
                        std::copy(mem.data() + nvar*LDIM + LDIM*DIM*i, mem.data() + (nvar+1)*LDIM + LDIM*DIM*i, res + i*LDIM);
                    return 0;
                }, 
                new_udofs, ldof_id, user_data, max_quad_order);      
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };
    template<int DIM, typename FEMTYPE>
    struct Dof<FemVecT<DIM, FEMTYPE>>{
        static inline auto Map(){
            return DofT::VectorDofMapC<decltype(Dof<FEMTYPE>::Map())>(DIM, Dof<FEMTYPE>::Map());
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            constexpr auto LNFA = Operator<IDEN, FEMTYPE>::Nfa::value;
            constexpr auto LDIM = Operator<IDEN, FEMTYPE>::Dim::value;
            uint nvar = idof_on_tet / LNFA, ldof_id = idof_on_tet % LNFA;
            std::array<ArrayView<>, FUSION> new_udofs;
            for (uint r = 0; r < FUSION; ++r)
                new_udofs[r] = ArrayView<>(udofs[r].data + nvar*LNFA, LNFA);
            Dof<FEMTYPE>::template interpolate<FUSION>(XYZ, 
                [&f, nvar](const std::array<double, 3>& X, double* res, uint dim, void* user_data)->int{
                    assert(dim == LDIM*FUSION && "Wrong expected dimension");
                    std::array<double, LDIM*DIM*FUSION> mem;
                    f(X, mem.data(), LDIM*DIM*FUSION, user_data);
                    for (uint i = 0; i < FUSION; ++i)
                        std::copy(mem.data() + nvar*LDIM + LDIM*DIM*i, mem.data() + (nvar+1)*LDIM + LDIM*DIM*i, res + i*LDIM);
                    return 0;
                }, 
                new_udofs, ldof_id, user_data, max_quad_order);       
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };
    template<typename ...Type>
    struct Dof<FemCom<Type...>>{
    private:
        template<int I, int N, int DIM_SHIFT, int NFA_SHIFT, uint FUSION, typename EvalFunc>
        struct Choose{
            static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data, uint max_quad_order){
                using LocalVar = std::tuple_element<I, typename FemCom<Type...>::Base>;
                constexpr auto DIM = Operator<IDEN, FemCom<Type...>>::Dim::value;
                constexpr auto LNFA = Operator<IDEN, LocalVar>::Nfa::value;
                constexpr auto LDIM = Operator<IDEN, LocalVar>::Dim::value;
                if (idof_on_tet - NFA_SHIFT < LNFA){
                    std::array<ArrayView<>, FUSION> ludofs;
                    for (uint i = 0; i < FUSION; ++i)
                        ludofs[i] = ArrayView<>(udofs[i].data + NFA_SHIFT, udofs[i].size - NFA_SHIFT);
                    Dof<LocalVar>::template interpolate<FUSION,EvalFunc >(XYZ, 
                        [&f](const std::array<double, 3>& X, double* res, uint dim, void* user_data)->int{
                            assert(dim == LDIM*FUSION && "Wrong expected dimension");
                            std::array<double, DIM*FUSION> mem;
                            f(X, mem.data(), DIM*FUSION, user_data);
                            for (uint i = 0; i < FUSION; ++i)
                                std::copy(mem.data() + DIM_SHIFT + DIM*i, mem.data() + DIM_SHIFT + DIM*i + LDIM, res + i*LDIM);
                            return 0;
                        }, 
                        ludofs, idof_on_tet - NFA_SHIFT, user_data, max_quad_order);
                } else {
                    Choose<I+1, N, DIM_SHIFT + LDIM, NFA_SHIFT + LNFA, FUSION, EvalFunc>::interpolate(XYZ, f, udofs, idof_on_tet, user_data, max_quad_order);
                }
            }
        };
        template<int N, int DIM_SHIFT, int NFA_SHIFT, uint FUSION, typename EvalFunc>
        struct Choose<N, N, DIM_SHIFT, NFA_SHIFT, FUSION, EvalFunc>{
            static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data, uint max_quad_order){
                throw std::runtime_error("Reached unreaceable code");
            }
        };
    public:
        static inline auto Map(){
            return DofT::ComplexDofMapC<decltype(Dof<Type>::Map())...>(Dof<Type>::Map()...);
        }
        template<uint FUSION = 1, typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, std::array<ArrayView<>, FUSION> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            constexpr auto K = sizeof...(Type);
            Choose<0, K, 0, 0, FUSION, EvalFunc>::interpolate(XYZ, f, udofs, idof_on_tet, 0, 0, user_data, max_quad_order);
        }
        template<typename EvalFunc>
        static inline void interpolate(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, void* user_data = nullptr, uint max_quad_order = 5){
            interpolate<1, EvalFunc>(XYZ, f, std::array<ArrayView<>, 1>{udofs}, idof_on_tet, user_data, max_quad_order);
        }
    };

    ///Structure to get position of specific tetrahedron degree of freedom for the FEM space type
    template<typename FEMTYPE>
    struct DOF_coord{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType *nodes, ScalarType *coord){
            throw std::runtime_error("Was called unimplemented method");
        }
    };

    template<>
    struct DOF_coord<FemFix<FEM_P0>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType *nodes, ScalarType *coord){
            assert(odf == 0 && "FEM_P0 have only 1 odf");
            (void) odf;
            for (int i = 0; i < 4; ++i)
                for (int k = 0; k < 3; ++k)
                    coord[k] += nodes[3 * i + k];
            for (int k = 0; k < 3; ++k) coord[k] /= 4;
        }
    };

    template<>
    struct DOF_coord<FemFix<FEM_P1>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType *nodes, ScalarType *coord){
            std::copy(nodes + 3 * odf, nodes + 3 * odf + 3, coord);
        }
    };

    template<>
    struct DOF_coord<FemFix<FEM_P2>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType  *nodes, ScalarType *coord){
            if (odf < 4)
                std::copy(nodes + 3 * odf, nodes + 3 * odf + 3, coord);
            else {
                odf -= 3;
                int i1 = 0, i2 = odf;
                if (odf > 3) {
                    i1 = 1 + (odf > 5);
                    i2 = 2 + (odf > 4);
                }
                for (int i = 0; i < 3; ++i)
                    coord[i] = (nodes[3 * i1 + i] + nodes[3 * i2 + i]) / 2;
            }
        }
    };

    template<>
    struct DOF_coord<FemFix<FEM_P3>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType  *nodes, ScalarType  *coord){
            if (odf < 4)
                std::copy(nodes + 3 * odf, nodes + 3 * odf + 3, coord);
            else if (odf >= 16) {
                odf -= 15;
                int i1 = (odf == 4), i2 = 1 + (odf > 2), i3 = 3 - (odf == 1);
                for (int i = 0; i < 3; ++i)
                    coord[i] = (nodes[3 * i1 + i] + nodes[3 * i2 + i] + nodes[3 * i3 + i]) / 3;
            } else {
                int l = odf % 2;
                odf = (odf - 4) / 2 + 1;
                int i1 = 0, i2 = odf;
                if (odf > 3) {
                    i1 = 1 + (odf > 5);
                    i2 = 2 + (odf > 4);
                }
                double w1 = (1 + !l) / 3.0, w2 = (1 + l) / 3.0;
                for (int i = 0; i < 3; ++i)
                    coord[i] = w1 * nodes[3 * i1 + i] + w2 * nodes[3 * i2 + i];
            }
        }
    };

    template<>
    struct DOF_coord<FemFix<FEM_CR1>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType  *nodes, ScalarType  *coord){
            for (int i = 0; i < 3; ++i)
                coord[i] = (nodes[i + 3*((odf + 1) % 4)] + nodes[i + 3*((odf + 2) % 4)] + nodes[i + 3*((odf + 3) % 4)]) / 3;
        }
    };

    template<int Dim, int Type>
    struct DOF_coord<FemVec<Dim, Type>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType  *nodes, ScalarType  *coord){
            return DOF_coord<typename FemVec<Dim, Type>::Base>::template at<ScalarType>(odf % Operator<IDEN, typename FemVec<Dim, Type>::Base>::Nfa::value, nodes, coord);
        }
    };

    template<int Dim, typename Type>
    struct DOF_coord<FemVecT<Dim, Type>>{
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType  *nodes, ScalarType  *coord){
            return DOF_coord<typename FemVecT<Dim, Type>::Base>::template at<ScalarType>(odf % Operator<IDEN, typename FemVecT<Dim, Type>::Base>::Nfa::value, nodes, coord);
        }
    };

    template<typename ...Type>
    struct DOF_coord<FemCom<Type...>>{
    private:
        template <std::size_t I, std::size_t N, std::size_t OFF>
        struct FindStart{
            template<typename ScalarType>
            inline static void Impl(int odf, const ScalarType  *nodes, ScalarType  *coord){
                using LocalVar = std::tuple_element<I, typename FemCom<Type...>::Base>;
                if (odf < OFF + Operator<IDEN, LocalVar>::Nfa::value){
                    DOF_coord<LocalVar>::template at<ScalarType>(odf - OFF, nodes, coord);
                } else {
                    FindStart<I+1, N, OFF + Operator<IDEN, LocalVar>::Nfa::value>(odf, nodes, coord);
                }
            }
        };
        template <std::size_t N, std::size_t OFF>
        struct FindStart<N, N, OFF>{
            template<typename ScalarType>
            inline static void Impl(int odf, const ScalarType  *nodes, ScalarType  *coord){
                assert(false && "Wrong odf");
            }
        };
    public:
        template<typename ScalarType = double>
        static inline void at(int odf, const ScalarType  *nodes, ScalarType  *coord){
            assert(odf >= 0 && "Wrong odf");
            return FindStart<0, std::tuple_size<typename FemCom<Type...>::Base>::value, 0>::Impl(odf, nodes, coord);
        }
    };
}

#include "operators.inl"

#endif //CARNUM_OPERATORS_H
