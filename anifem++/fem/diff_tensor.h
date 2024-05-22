//
// Created by Liogky Alexey on 21.03.2022.
//

#ifndef CARNUM_DIFFTENSOR_H
#define CARNUM_DIFFTENSOR_H

#include <array>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace Ani{
    ///Available tensor types (here tensor is matrix)
    enum TensorType {
        TENSOR_NULL = 1, ///< Identity tensor I ,available only if dim(OpA(u)) = dim(OpB(u))
        TENSOR_SCALAR = 2, ///< Scalar multiplied identity tensor, i.e. D*I where D is scalar
        TENSOR_SYMMETRIC = 3, ///< Symmetric tensor (maybe used to optimize calculations)
        TENSOR_GENERAL = 4 ///< General tensor
    };

    enum TensorTypeSparsity {
        PerPoint = 0, ///< Tensor type depends on input coordinate
        PerTetra = -1, ///< Tensor type depends on number of tetrahedron
        PerSelection = -2 ///< Tensor type depends on number of fuse tetrahedron selection
    };

    enum TensorTypeAggregate {
        OnePointTensor = 0, ///< require \code TensorType(const Coord<Scalar> &X, Scalar *D, TensorDims Ddims, void *user_data, int iTet) \endcode tensor function signature
        FusiveTensor = 1    ///< require \code TensorType(ArrayView<Scalar> X, ArrayView<Scalar> D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem) \endcode tensor function signature
    };
    ///Traits are used to specify Tensor
    ///Specific traits are used to optimize computation of elemental matrix
    template<int TensorTypeSparse = PerPoint, bool isConstant = false, long idim = -1, long jdim = -1, int aggregateType = OnePointTensor>
    struct DfuncTraitsCommon {
        using AggregateType = std::integral_constant<int, aggregateType>;
        ///is tensor constant?
        using IsConstant = std::integral_constant<bool, isConstant>;
        ///Set one of types
        ///if value > 0 if Tensor type is not depend on coordinate and value equals to common TensorType
        ///if value <= 0 it means tensor type depends on coordinate and value equals one of TensorTypeSparsity
        using TensorSparsity = std::integral_constant<int, TensorTypeSparse>;
        ///if positive than it's expected dimension of V, number of rows of the tensor
        using iD = std::integral_constant<long, idim>;
        ///if positive than it's expected dimension of U, number of cols of the tensor
        using jD = std::integral_constant<long, jdim>;
    };
    template<int TensorTypeSparse = PerPoint, bool isConstant = false, long idim = -1, long jdim = -1>
    using DfuncTraits = DfuncTraitsCommon<TensorTypeSparse, isConstant, idim, jdim>;
    template<long idim = -1, long jdim = -1>
    using DfuncTraitsFusive = DfuncTraitsCommon<PerSelection, false, idim, jdim, FusiveTensor>;

    template<typename Scalar = double>
    using Coord = std::array<Scalar, 3>;

    using TensorDims = std::pair<std::size_t, std::size_t>;

    ///Some helper DFUNC structure
    /// @tparam Functor is object with method TensorType()(const Coord<Scalar> &X, Scalar *D, std::pair<std::size_t, std::size_t> Ddims, void *user_data, int iTet)
    template<typename FunctorT, typename TraitsT, typename Scalar, typename IndexType = int, int agrType = OnePointTensor>
    struct _internal_DFunc_common_memoryless : public TraitsT {
        using ArrayR = ArrayView<Scalar>;
        using Traits = TraitsT;
        using Functor = FunctorT;
        const Functor &f;
        ArrayR m_mem;
        
        _internal_DFunc_common_memoryless(const Functor &f) : f{f} {}
        _internal_DFunc_common_memoryless(const Functor &f, ArrayView<Scalar> mem) : f{f}, m_mem{mem} {}

        virtual TensorType operator()(const std::array<Scalar, 3> &X, Scalar *D, std::pair<std::size_t, std::size_t> Ddims, void *user_data,
                              int iTet = 0) {
            Scalar* mem = m_mem.data;
            TensorType t = f(X, mem, Ddims, user_data, iTet);
            switch (t) {
                default: {
                    for (std::size_t i = 0; i < Ddims.second; i++)
                        for (std::size_t j = 0; j < Ddims.first; j++)
                            D[i + Ddims.second * j] = mem[j + Ddims.first * i];
                    break;
                }
                case TENSOR_SYMMETRIC:{
                    assert(Ddims.first == Ddims.second && "TENSOR_SYMMETRIC should have equal dimensions");
                    std::copy(mem, mem + Ddims.first*Ddims.second, D);
                    break;
                }
                case TENSOR_NULL:
                    break;
                case TENSOR_SCALAR:
                    D[0] = mem[0];
                    break;
            }
            return t;
        }
    };

    /// @tparam Functor is object with method TensorType()(ArrayR X, ArrayR D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem)
    template<typename FunctorT, typename TraitsT, typename Scalar, typename IndexType>
    struct _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, FusiveTensor> : public TraitsT {
        using ArrayR = ArrayView<Scalar>;
        using Traits = TraitsT;
        using Functor = FunctorT;

        const Functor &f;
        ArrayR m_mem;

        _internal_DFunc_common_memoryless(const Functor &f) : f{f} {}
        _internal_DFunc_common_memoryless(const Functor &f, ArrayR mem) : f{f}, m_mem{mem} {}

        virtual TensorType operator()(ArrayR X, ArrayR D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem) {
            Scalar* lmem = m_mem.data;
            TensorType t = f(X, D, Ddims, user_data, mem);
            switch (t) {
                default: {
                    for (std::size_t r = 0; r < mem.f; ++r)
                        for (std::size_t n = 0; n < mem.q; ++n){
                            int off = Ddims.first*Ddims.second*(n + mem.q*r);
                            std::copy(D.data + off, D.data + off + Ddims.first*Ddims.second, lmem);
                            for (uint i = 0; i < Ddims.second; i++)
                                for (uint j = 0; j < Ddims.first; j++)
                                    D[off + i + Ddims.second * j] = lmem[j + Ddims.first * i];
                        }
                    break;
                }
                case TENSOR_SYMMETRIC:
                    break;
                case TENSOR_NULL:
                    break;
                case TENSOR_SCALAR:
                    break;
            }
            return t;
        }
    };


    template<typename FunctorT, typename TraitsT, typename Scalar, typename IndexType, int MaxDSize = 9 * 9, int agrType = OnePointTensor>
    struct _internal_DFunc_common : public _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, agrType> {
        using BaseT = _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, agrType> ;
        using Functor = typename BaseT::Functor;
        using Traits = typename BaseT::Traits;

        _internal_DFunc_common(const Functor &f) : BaseT(f) {}

        TensorType operator()(const std::array<Scalar, 3> &X, Scalar *D, TensorDims Ddims, void *user_data,
                              int iTet = 0) override {
            Scalar mem[MaxDSize+7] = {0}; //+7 to suppress gcc memory warning
            assert(MaxDSize >= Ddims.first * Ddims.second && "Too small MaxDSize");
            BaseT::m_mem.Init(mem, MaxDSize);
            return BaseT::operator()(X, D, Ddims, user_data, iTet);
        };
    };

    template<typename FunctorT, typename TraitsT, typename Scalar, typename IndexType, int MaxDSize>
    struct _internal_DFunc_common<FunctorT, TraitsT, Scalar, IndexType, MaxDSize, FusiveTensor> : public _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, FusiveTensor> {
        using BaseT = _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, FusiveTensor>;
        using ArrayR = typename BaseT::ArrayR;
        using Functor = typename BaseT::Functor;
        using Traits = typename BaseT::Traits;

        _internal_DFunc_common(const Functor &f) : BaseT(f) {}

        TensorType operator()(ArrayR X, ArrayR D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem) override {
            Scalar lmem[MaxDSize+7] = {0}; //+7 to suppress gcc memory warning
            assert(MaxDSize >= Ddims.first * Ddims.second && "Too small MaxDSize");
            BaseT::m_mem.Init(lmem, MaxDSize);
            return BaseT::operator()(X, D, Ddims, user_data, mem);
        }
    };

    template<typename FunctorT, typename TraitsT, typename Scalar, typename IndexType>
    struct _internal_DFunc_common<FunctorT, TraitsT, Scalar, IndexType, -1, OnePointTensor> : public _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, OnePointTensor> {
        using BaseT = _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, OnePointTensor> ;
        using ArrayR = typename BaseT::ArrayR;
        using Functor = typename BaseT::Functor;
        using Traits = typename BaseT::Traits;

        _internal_DFunc_common(const Functor &f) : BaseT(f) {}
        _internal_DFunc_common(const Functor &f, ArrayR mem): BaseT(f, mem) {}

        TensorType operator()(const std::array<Scalar, 3> &X, Scalar *D, TensorDims Ddims, void *user_data,
                              int iTet = 0) override {
            if (BaseT::m_mem.size < Ddims.first * Ddims.second) throw std::runtime_error("Not enough memory for Dfunc handler");                    
            return BaseT::operator()(X, D, Ddims, user_data, iTet);
        };
        void setMem(ArrayR mem) { BaseT::m_mem = mem; }
        ///Get minimal required count of Scalar values for memory
        static std::size_t memReq(TensorDims Ddims) { return Ddims.first * Ddims.second; }
    };

    template<typename FunctorT, typename TraitsT, typename Scalar, typename IndexType>
    struct _internal_DFunc_common<FunctorT, TraitsT, Scalar, IndexType, -1, FusiveTensor> : public _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, FusiveTensor> {
        using BaseT = _internal_DFunc_common_memoryless<FunctorT, TraitsT, Scalar, IndexType, FusiveTensor> ;
        using ArrayR = typename BaseT::ArrayR;
        using Functor = typename BaseT::Functor;
        using Traits = typename BaseT::Traits;

        _internal_DFunc_common(const Functor &f) : BaseT(f) {}
        _internal_DFunc_common(const Functor &f, ArrayR mem): BaseT(f, mem) {}

        TensorType operator()(ArrayR X, ArrayR D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem) override {
            if (BaseT::m_mem.size < static_cast<decltype(BaseT::m_mem.size)>(Ddims.first * Ddims.second)) throw std::runtime_error("Not enough memory for Dfunc handler");                    
            return BaseT::operator()(X, D, Ddims, user_data, mem);
        };
        void setMem(ArrayR mem) { BaseT::m_mem = mem; }
        ///Get minimal required count of Scalar values for memory
        static std::size_t memReq(TensorDims Ddims) { return Ddims.first * Ddims.second; }
    };

    template<typename Functor, typename Traits, typename Scalar, typename IndexType, int MaxDSize = 9 * 9>
    using DFunc = _internal_DFunc_common<Functor, Traits, Scalar, IndexType, MaxDSize, Traits::AggregateType::value>;


    template<typename Scalar = double>
    TensorType TensorNull(const Coord<Scalar> &X, Scalar *D, TensorDims Ddims, void *user_data, int iTet){
        (void) X; (void) D; (void) Ddims; (void) user_data; (void) iTet;
        return TENSOR_NULL;
    }
    
    template<typename Scalar = double>
    auto makeConstantTensorScalar(double val){
        return [val](const Coord<Scalar> &X, Scalar *D, TensorDims Ddims, void *user_data, int iTet){
            (void) X; (void) Ddims; (void) user_data; (void) iTet;
            D[0] = val;
            return TENSOR_SCALAR;
        };
    }

    /// @tparam FieldFunction is functor of type Scalar(const std::array<Scalar, 3> &X)
    template<typename Scalar = double, typename FieldFunction>
    auto makeScalarTensor(const FieldFunction& f){
        return [&f](const Coord<Scalar> &X, double *dat, TensorDims dims, void *user_data, int iTet){
            dat[0] = f(X);
            return Ani::TENSOR_SCALAR;
        };
    }

    /// @tparam VectorFieldFunction is functor of type std::array<Scalar,N>(const std::array<Scalar, 3> &X)
    template<typename Scalar = double, int N, typename VectorFieldFunction>
    auto makeVectorTensor(const VectorFieldFunction& f){
        return [&f](const Coord<Scalar> &X, double *dat, TensorDims dims, void *user_data, int iTet){
            if (dims.first != N || dims.second != 1) throw std::runtime_error("Waited tensor with dimension " 
                + std::to_string(dims.first) + " x " + std::to_string(dims.second) + " instead " + std::to_string(N) + " x 1");
            auto res = f(X);
            std::copy(res.begin(), res.end(), dat);
            return Ani::TENSOR_GENERAL;
        };
    }

    ///Internal structure storing all data required for evaluating D*U expression
    ///@see internalFem3Dtet
    template <typename DFUNC, typename Scalar = double, typename IndexType = int>
    struct DU_comp_in{
        AniMemory<Scalar, IndexType>& mem;
        DFUNC &Dfnc;
        void *user_data;
        BandDenseMatrixX<Scalar, IndexType> U;
        IndexType idim, jdim;
        IndexType q, f;
        IndexType nfa, nfb;

        DU_comp_in(AniMemory<Scalar, IndexType>& mem, DFUNC &Dfnc, void *user_data, 
                    BandDenseMatrixX<Scalar, IndexType>& U, IndexType idim, IndexType jdim, IndexType q, IndexType f, 
                    IndexType nfa, IndexType nfb): mem{mem}, Dfnc{Dfnc}, user_data{user_data}, U{U}, idim{idim}, jdim{jdim}, q{q}, f{f}, nfa{nfa}, nfb{nfb} {}
        template<int NPART>
        DU_comp_in(AniMemory<Scalar, IndexType>& mem, DFUNC &Dfnc, void *user_data, 
                    BandDenseMatrix<NPART, Scalar, IndexType>& U, IndexType idim, IndexType jdim, IndexType q, IndexType f, 
                    IndexType nfa, IndexType nfb): mem{mem}, Dfnc{Dfnc}, user_data{user_data}, U(U), idim{idim}, jdim{jdim}, q{q}, f{f}, nfa{nfa}, nfb{nfb} {}            
    };
    ///Internal structure helping to compute D*U expression in effective manner
    ///@see internalFem3Dtet
    template<typename Scalar = double, typename IndexType = unsigned int>
    struct internal_Dfunc{
        using AniMem = AniMemory<Scalar, IndexType>;
        #define ONEPT (DFUNC::AggregateType::value == OnePointTensor)
        template <typename DFUNC, typename std::enable_if<(ONEPT && DFUNC::IsConstant::value && DFUNC::TensorSparsity::value <=0), bool>::type = true>
        inline static DenseMatrix<Scalar> applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            std::array<Scalar, 3> dummy{0};
            TensorType t = Dfnc(dummy, args.mem.DIFF.data, {jdim, idim}, args.user_data, 0);
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            switch (t) {
                case TENSOR_GENERAL:
                case TENSOR_SYMMETRIC:{
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    for (IndexType k = 0; k < jdim; ++k){
                                        auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                        Scalar s = 0;
                                        for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                            s += args.mem.DIFF.data[j + idim*k]*U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                                                              ((i - U.stCol[d1]) + isz * r)];
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * s;
                                    }
                                }
                        }
                    }
                    break;
                }
                case TENSOR_NULL:{
                    if (jdim != idim && (nfa != 1 || idim != 1))
                        throw std::runtime_error(
                                "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                    }
                    if (jdim != idim){ //nfa=1 => FEM_P0
                        for (IndexType r = 0; r < f; ++r)
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                    }
                    break;
                }
                case TENSOR_SCALAR:{
                    // condition (nfa == 1) for evaluating RHS expressions with scalar tensor and vector variable
                    if (jdim != idim && (nfa != 1 || idim != 1))
                        throw std::runtime_error(
                                "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                    auto s = args.mem.DIFF.data[0];
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = s*mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                    }
                    if (jdim != idim){ //nfa=1 => FEM_P0
                        for (IndexType r = 0; r < f; ++r)
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                    }
                    break;
                }
            }
            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(DFUNC::IsConstant::value && DFUNC::TensorSparsity::value == TENSOR_NULL), bool>::type = true>
        inline static DenseMatrix<Scalar> applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            if (jdim != idim && (nfa != 1 || idim != 1))
                throw std::runtime_error(
                        "Identity tensor defined only for compatible (with same dimensions) operators A and B");
            for (IndexType r = 0; r < f; ++r) {
                auto vol = mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = w[n];
                            auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                            for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                        (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                      ((i - U.stCol[d1]) + isz * r)];
                        }
                }
            }
            if (jdim != idim){ //nfa=1 => FEM_P0
                for (IndexType r = 0; r < f; ++r)
                    for (IndexType n = 0; n < q; ++n)
                        for (IndexType k = idim; k < jdim; ++k)
                            DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
            }

            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && DFUNC::IsConstant::value && DFUNC::TensorSparsity::value == TENSOR_SCALAR), bool>::type = true>
        inline static DenseMatrix<Scalar> applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            // consdition (nfa == 1) for evaluating RHS expressions with scalar tensor and vector variable
            if (jdim != idim && (nfa != 1 || idim != 1))
                throw std::runtime_error(
                        "Identity tensor defined only for compatible (with same dimensions) operators A and B");
            std::array<Scalar, 3> dummy{0};
            TensorType t = Dfnc(dummy, args.mem.DIFF.data, {jdim, idim}, args.user_data, 0);
            assert((t == TENSOR_SCALAR) && "Waited tensor type TENSOR_SCALAR");
            (void) t;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto s = args.mem.DIFF.data[0];
            for (IndexType r = 0; r < f; ++r) {
                auto vol = s*mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = w[n];
                            auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                            for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                        (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                      ((i - U.stCol[d1]) + isz * r)];
                        }
                }
            }
            if (jdim != idim){ //nfa=1 => FEM_P0
                for (IndexType r = 0; r < f; ++r)
                    for (IndexType n = 0; n < q; ++n)
                        for (IndexType k = idim; k < jdim; ++k)
                            DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
            }

            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && DFUNC::IsConstant::value && (DFUNC::TensorSparsity::value == TENSOR_GENERAL || DFUNC::TensorSparsity::value == TENSOR_SYMMETRIC)), bool>::type = true>
        inline static DenseMatrix<Scalar> applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            std::array<Scalar, 3> dummy{0};
            TensorType t = Dfnc(dummy, args.mem.DIFF.data, {jdim, idim}, args.user_data, 0);
            assert((t == TENSOR_GENERAL || t == TENSOR_SYMMETRIC) && "Waited tensor type TENSOR_GENERAL or TENSOR_SYMMETRIC");
            (void) t;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            for (IndexType r = 0; r < f; ++r) {
                auto vol = mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = w[n];
                            for (IndexType k = 0; k < jdim; ++k){
                                auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                Scalar s = 0;
                                for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                    s += args.mem.DIFF.data[j + idim*k]*U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                                                      ((i - U.stCol[d1]) + isz * r)];
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol* s;
                            }
                        }
                }
            }

            return DU;
        }

        template <typename DFUNC, typename std::enable_if<(ONEPT && (!DFUNC::IsConstant::value) && DFUNC::TensorSparsity::value == PerPoint), bool>::type = true>
        inline static auto applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            for (IndexType r = 0; r < f; ++r) {
                for (IndexType n = 0; n < q; ++n) {
                    auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                    std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                    TensorType t = Dfnc(x, locD, {jdim, idim}, args.user_data, r);
                    switch (t) {
                        case TENSOR_GENERAL:
                        case TENSOR_SYMMETRIC: break;
                        case TENSOR_SCALAR:{
                            if (jdim != idim && (nfa != 1 || idim != 1))
                                throw std::runtime_error(
                                        "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                            Scalar s = locD[0];
                            std::fill(locD, locD + idim*jdim, 0);
                            auto step = (jdim == idim) ? idim : 0;
                            for (IndexType i = 0; i < jdim; ++i)
                                locD[i + i * step] = s;           
                            break;
                        }
                        case TENSOR_NULL:{
                            if (jdim != idim && (nfa != 1 || idim != 1))
                                throw std::runtime_error(
                                        "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                            std::fill(locD, locD + idim*jdim, 0);
                            auto step = (jdim == idim) ? idim : 0;
                            for (IndexType i = 0; i < jdim; ++i)
                                locD[i + i * step] = 1;  
                            break;
                        }
                    }
                }
            }

            for (IndexType r = 0; r < f; ++r) {
                auto vol = mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = w[n];
                            auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                            for (IndexType k = 0; k < jdim; ++k){
                                auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                Scalar s = 0;
                                for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                    s += locD[j + idim*k]*U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                                        ((i - U.stCol[d1]) + isz * r)];
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol* s;
                            }
                        }
                }
            }

            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && (!DFUNC::IsConstant::value) && DFUNC::TensorSparsity::value == PerTetra), bool>::type = true>
        inline static auto applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            for (IndexType r = 0; r < f; ++r) {
                auto glocD = args.mem.DIFF.data + 0;
                std::array<Scalar, 3> gx{args.mem.XYG.data[3 * r * q], args.mem.XYG.data[3 * r * q + 1], args.mem.XYG.data[3 * r * q + 2]};
                TensorType t = Dfnc(gx, glocD, {jdim, idim}, args.user_data, r);
                switch (t) {
                    case TENSOR_GENERAL:
                    case TENSOR_SYMMETRIC: {
                        for (IndexType n = 1; n < q; ++n) {
                            auto locD = args.mem.DIFF.data + idim * jdim * n;
                            std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                            Dfnc(x, locD, {jdim, idim}, args.user_data, r);
                        }
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto locD = args.mem.DIFF.data + idim * jdim * n;
                                    for (IndexType k = 0; k < jdim; ++k) {
                                        auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                        Scalar s = 0;
                                        for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                            s += locD[j + idim * k] *
                                                 U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                               ((i - U.stCol[d1]) +
                                                                                                isz * r)];
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * s;
                                    }
                                }
                        }

                        break;
                    }
                    case TENSOR_SCALAR: {
                        if (jdim != idim && (nfa != 1 || idim != 1))
                            throw std::runtime_error(
                                    "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                        for (IndexType n = 1; n < q; ++n) {
                            auto locD = args.mem.DIFF.data + idim * jdim * n;
                            std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                            Dfnc(x, locD, {jdim, idim}, args.user_data, r);
                        }
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i) {
                                for (IndexType n = 0; n < q; ++n) {
                                    auto s = args.mem.DIFF.data[0 + idim * jdim * n];
                                    auto wg = s * w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                            }
                        }
                        if (jdim != idim){ //nfa=1 => FEM_P0
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                        }

                        break;
                    }
                    case TENSOR_NULL: {
                        if (jdim != idim && (nfa != 1 || idim != 1))
                            throw std::runtime_error(
                                    "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                        if (jdim != idim){ //nfa=1 => FEM_P0
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                        }
                        break;
                    }
                }
            }
            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && (!DFUNC::IsConstant::value) && DFUNC::TensorSparsity::value == PerSelection), bool>::type = true>
        inline static auto applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix <Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa * f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            auto glocD = args.mem.DIFF.data + 0;
            std::array<Scalar, 3> gx{args.mem.XYG.data[0], args.mem.XYG.data[1], args.mem.XYG.data[2]};
            TensorType t = Dfnc(gx, glocD, {jdim, idim}, args.user_data, 0);
            switch (t) {
                case TENSOR_GENERAL:
                case TENSOR_SYMMETRIC: {
                    for (IndexType n = 1; n < q; ++n){
                        auto locD = args.mem.DIFF.data + idim*jdim*(n + q*0);
                        std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + 0*q)], args.mem.XYG.data[3*(n + 0*q)+1], args.mem.XYG.data[3*(n + 0*q)+2]};
                        Dfnc(x, locD, {jdim, idim}, args.user_data, 0);
                    }
                    for (IndexType r = 1; r < f; ++r)
                        for (IndexType n = 0; n < q; ++n){
                            auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                            std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                            Dfnc(x, locD, {jdim, idim}, args.user_data, 0);
                        }

                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                                    for (IndexType k = 0; k < jdim; ++k){
                                        auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                        Scalar s = 0;
                                        for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                            s += locD[j + idim*k]*U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                                                ((i - U.stCol[d1]) + isz * r)];
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol* s;
                                    }
                                }
                        }
                    }

                    break;
                }
                case TENSOR_SCALAR: {
                    if (jdim != idim && (nfa != 1 || idim != 1))
                        throw std::runtime_error(
                                "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                    for (IndexType n = 1; n < q; ++n){
                        auto locD = args.mem.DIFF.data + idim*jdim*(n + q*0);
                        std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + 0*q)], args.mem.XYG.data[3*(n + 0*q)+1], args.mem.XYG.data[3*(n + 0*q)+2]};
                        Dfnc(x, locD, {jdim, idim}, args.user_data, 0);
                    }
                    for (IndexType r = 1; r < f; ++r)
                        for (IndexType n = 0; n < q; ++n){
                            auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                            std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                            Dfnc(x, locD, {jdim, idim}, args.user_data, r);
                        }
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = args.mem.DIFF.data[idim*jdim*(n + q*r)] * w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                    }
                    if (jdim != idim){ //nfa=1 => FEM_P0
                        for (IndexType r = 0; r < f; ++r)
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                    }

                    break;
                }
                case TENSOR_NULL:{
                    if (jdim != idim && (nfa != 1 || idim != 1))
                        throw std::runtime_error(
                                "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                    }
                    if (jdim != idim){ //nfa=1 => FEM_P0
                        for (IndexType r = 0; r < f; ++r)
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                    }
                    break;
                }
            }
            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && (!DFUNC::IsConstant::value) && (DFUNC::TensorSparsity::value == TENSOR_GENERAL || DFUNC::TensorSparsity::value ==TENSOR_SYMMETRIC)), bool>::type = true>
        inline static auto applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix <Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa * f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            for (IndexType r = 0; r < f; ++r)
                for (IndexType n = 0; n < q; ++n){
                    auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                    std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                    auto t = Dfnc(x, locD, {jdim, idim}, args.user_data, r);
                    assert((t == TENSOR_GENERAL || t == TENSOR_SYMMETRIC) && "Waited tensor type TENSOR_GENERAL or TENSOR_SYMMETRIC");
                    (void) t;
                }

            for (IndexType r = 0; r < f; ++r) {
                auto vol = mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = w[n];
                            auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                            for (IndexType k = 0; k < jdim; ++k){
                                auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                Scalar s = 0;
                                for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                    s += locD[j + idim*k]*U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                                        ((i - U.stCol[d1]) + isz * r)];
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol* s;
                            }
                        }
                }
            }

            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && (!DFUNC::IsConstant::value) && DFUNC::TensorSparsity::value == TENSOR_SCALAR), bool>::type = true>
        inline static auto applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix <Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa * f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            auto& Dfnc = args.Dfnc;
            if (jdim != idim && (nfa != 1 || idim != 1))
                throw std::runtime_error(
                        "Identity tensor defined only for compatible (with same dimensions) operators A and B");
            for (IndexType r = 0; r < f; ++r)
                for (IndexType n = 0; n < q; ++n){
                    auto locD = args.mem.DIFF.data + idim*jdim*(n + q*r);
                    std::array<Scalar, 3> x{args.mem.XYG.data[3*(n + r*q)], args.mem.XYG.data[3*(n + r*q)+1], args.mem.XYG.data[3*(n + r*q)+2]};
                    auto t = Dfnc(x, locD, {jdim, idim}, args.user_data, r);
                    assert((t == TENSOR_SCALAR) && "Waited tensor type TENSOR_SCALAR");
                    (void) t;
                }
            for (IndexType r = 0; r < f; ++r) {
                auto vol = mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = args.mem.DIFF.data[idim*jdim*(n + q*r)] * w[n];
                            auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                            for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                        (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                    ((i - U.stCol[d1]) + isz * r)];
                        }
                }
            }
            if (jdim != idim){ //nfa=1 => FEM_P0
                for (IndexType r = 0; r < f; ++r)
                    for (IndexType n = 0; n < q; ++n)
                        for (IndexType k = idim; k < jdim; ++k)
                            DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
            }

            return DU;
        }
        template <typename DFUNC, typename std::enable_if<(ONEPT && (!DFUNC::IsConstant::value) && DFUNC::TensorSparsity::value == TENSOR_NULL), bool>::type = true>
        inline static DenseMatrix<Scalar> applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix<Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa*f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            if (jdim != idim && (nfa != 1 || idim != 1))
                throw std::runtime_error(
                        "Identity tensor defined only for compatible (with same dimensions) operators A and B");
            for (IndexType r = 0; r < f; ++r) {
                auto vol = mes[r];
                for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                    auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                    for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                        for (IndexType n = 0; n < q; ++n) {
                            auto wg = w[n];
                            auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                            for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                        (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                      ((i - U.stCol[d1]) + isz * r)];
                        }
                }
            }
            if (jdim != idim){ //nfa=1 => FEM_P0
                for (IndexType r = 0; r < f; ++r)
                    for (IndexType n = 0; n < q; ++n)
                        for (IndexType k = idim; k < jdim; ++k)
                            DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
            }

            return DU;
        }
        #undef ONEPT

        template <typename DFUNC, typename std::enable_if<((DFUNC::AggregateType::value == FusiveTensor) && !(DFUNC::IsConstant::value && DFUNC::TensorSparsity::value == TENSOR_NULL)), bool>::type = true>
        inline static auto applyDU(DU_comp_in<DFUNC, Scalar, IndexType> args) {
            IndexType idim = args.idim, jdim = args.jdim, q = args.q, f = args.f, nfa = args.nfa;
            DenseMatrix <Scalar> DU;
            DU.data = args.mem.DU.data;
            DU.size = args.mem.DU.size;
            DU.nRow = jdim * q, DU.nCol = nfa * f;
            auto w = args.mem.WG.data, mes = args.mem.MES.data;
            auto& U = args.U;
            TensorType t = args.Dfnc(args.mem.XYG, args.mem.DIFF, {jdim, idim}, args.user_data, args.mem);
            switch (t) {
                case TENSOR_GENERAL:
                case TENSOR_SYMMETRIC: {
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto locD = args.mem.DIFF.data + (!DFUNC::IsConstant::value ? idim*jdim*(n + q*r) : 0);
                                    for (IndexType k = 0; k < jdim; ++k){
                                        auto jsz = U.stRow[d1 + 1] - U.stRow[d1];
                                        Scalar s = 0;
                                        for (IndexType j = U.stRow[d1]; j < U.stRow[d1 + 1]; ++j)
                                            s += locD[j + idim*k]*U.data[d1].data[(j - U.stRow[d1]) + n * jsz + U.data[d1].nRow *
                                                                                                                ((i - U.stCol[d1]) + isz * r)];
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol* s;
                                    }
                                }
                        }
                    }

                    break;
                }
                case TENSOR_SCALAR: {
                    if (jdim != idim && (nfa != 1 || idim != 1))
                        throw std::runtime_error(
                                "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = args.mem.DIFF.data[(!DFUNC::IsConstant::value ? idim*jdim*(n + q*r) : 0)] * w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                    }
                    if (jdim != idim){ //nfa=1 => FEM_P0
                        for (IndexType r = 0; r < f; ++r)
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                    }

                    break;
                }
                case TENSOR_NULL:{
                    if (jdim != idim && (nfa != 1 || idim != 1))
                        throw std::runtime_error(
                                "Identity tensor defined only for compatible (with same dimensions) operators A and B");
                    for (IndexType r = 0; r < f; ++r) {
                        auto vol = mes[r];
                        for (std::size_t d1 = 0; d1 < U.nparts; ++d1) {
                            auto isz = U.stCol[d1 + 1] - U.stCol[d1];
                            for (IndexType i = U.stCol[d1]; i < U.stCol[d1 + 1]; ++i)
                                for (IndexType n = 0; n < q; ++n) {
                                    auto wg = w[n];
                                    auto ksz = U.stRow[d1 + 1] - U.stRow[d1];
                                    for (IndexType k = U.stRow[d1]; k < U.stRow[d1 + 1]; ++k)
                                        DU.data[k + jdim * (n + q * (i + nfa * r))] = wg * vol * U.data[d1].data[
                                                (k - U.stRow[d1]) + n * ksz + U.data[d1].nRow *
                                                                              ((i - U.stCol[d1]) + isz * r)];
                                }
                        }
                    }
                    if (jdim != idim){ //nfa=1 => FEM_P0
                        for (IndexType r = 0; r < f; ++r)
                            for (IndexType n = 0; n < q; ++n)
                                for (IndexType k = idim; k < jdim; ++k)
                                    DU.data[k + jdim * (n + q * (0 + nfa * r))] = DU.data[0 + jdim * (n + q * (0 + nfa * r))];
                    }
                    break;
                }
            }
            return DU;
        }
        
    };
};

#endif //CARNUM_DIFFTENSOR_H
