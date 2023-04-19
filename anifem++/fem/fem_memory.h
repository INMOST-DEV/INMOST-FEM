//
// Created by Liogky Alexey on 21.03.2022.
//

#ifndef CARNUM_MEMORY_H
#define CARNUM_MEMORY_H
#include <type_traits>
#include <algorithm>
#include <string>
#include <iomanip>
#include <limits>
#include <memory>
#include <array>
#include <sstream>
#include <utility>
#include <cmath>

namespace Ani{
    ///Store view on external memory
    template<typename ScalarType = double>
    struct ArrayView {
        using Scalar = ScalarType;
        Scalar *data = nullptr;
        std::size_t size = 0;

        ArrayView() = default;

        ArrayView(Scalar *data, std::size_t size) : data{data}, size{size} {}
        void Init(Scalar *data, std::size_t size){ ArrayView::data = data, ArrayView::size = size; }
        void SetZero(){ std::fill(data, data + size, 0); }
        Scalar& operator[](std::size_t i) { return data[i]; }
        Scalar operator[](std::size_t i) const { return data[i]; }
        bool empty() const { return (size <= 0 || data == nullptr); }
        void clear() { size = 0, data = nullptr; }
        Scalar* begin(){ return size > 0 ? data : nullptr; }
        Scalar* end(){ return !empty() ? data + size : nullptr; }
        const Scalar* begin() const { return size > 0 ? data : nullptr; }
        const Scalar* end() const { return !empty() ? data + size : nullptr; }
    };

    ///Store dense matrix
    template<typename ScalarType = double>
    struct DenseMatrix {
        using Scalar = ScalarType;
        Scalar *data = nullptr; ///< contiguous array storing matrix in col-major format
        std::size_t nRow = 0, nCol = 0; ///< number of rows and columns of the matrix
        std::size_t size = 0; ///< size of data array (should be not less than nRow*nCol)
        DenseMatrix() = default;
        DenseMatrix(Scalar *data, std::size_t nRow, std::size_t nCol): data{data}, nRow{nRow}, nCol{nCol}, size{nRow*nCol} {}
        DenseMatrix(Scalar *data, std::size_t nRow, std::size_t nCol, std::size_t size): data{data}, nRow{nRow}, nCol{nCol}, size{size} {}
        void Init(Scalar *data, std::size_t nRow, std::size_t nCol, std::size_t size){ DenseMatrix::data = data, DenseMatrix::nRow = nRow, DenseMatrix::nCol = nCol, DenseMatrix::size = size; }
        Scalar& operator()(std::size_t i, std::size_t j){ return data[i + nRow * j]; }
        Scalar operator()(std::size_t i, std::size_t j) const { return data[i + nRow * j]; }
        Scalar& operator[](std::size_t i){ return data[i]; }
        Scalar operator[](std::size_t i) const { return data[i]; }
        void SetZero(){ std::fill(data, data + nRow*nCol, 0); }
        std::string to_string(const std::string& val_sep = " ", const std::string& row_sep = "\n") const;

        /// @return Frobenius norm of ||a*A + b*B|| if b != 0 else return ||a*A||
        static ScalarType ScalNorm(double a, const DenseMatrix<ScalarType>& A, double b = 0.0, const DenseMatrix<ScalarType>& B = DenseMatrix<ScalarType>(nullptr, 0, 0));
    };

    template<typename ScalarType>
    ScalarType DenseMatrix<ScalarType>::ScalNorm(double a, const DenseMatrix<ScalarType>& A, double b, const DenseMatrix<ScalarType>& B){
        typename std::remove_const<ScalarType>::type nrm = 0;
        if (b != 0){
            assert(B.data && B.nRow == A.nRow && B.nCol == A.nCol && "B has wrong sizes");
            for (std::size_t j = 0; j < A.nCol; ++j) 
                for (std::size_t i = 0; i < A.nRow; ++i){
                    auto val = a*A(i,j) + b*B(i,j);
                    nrm += val * val;
                }
            nrm = sqrt(nrm);
        } else {
            for (std::size_t j = 0; j < A.nCol; ++j) 
                for (std::size_t i = 0; i < A.nRow; ++i) 
                    nrm += A(i, j) * A(i, j);
            nrm = sqrt(nrm)*abs(a);
        }
        return nrm;
    }

    template<typename ScalarType>
    std::string DenseMatrix<ScalarType>::to_string(const std::string& val_sep, const std::string& row_sep) const{
        std::ostringstream oss;
        oss << std::setprecision(std::numeric_limits<ScalarType>::digits10) << std::scientific;
        auto sign_shift = [](auto x) { return (x >= 0) ? " " : ""; };
        for (std::size_t i = 0; i < nRow; ++i){
            for (std::size_t j = 0; j < nCol-1; ++j){
                oss << sign_shift((*this)(i, j)) << (*this)(i, j) << val_sep;
            }
            if (nCol > 0) oss << sign_shift((*this)(i, nCol-1)) << (*this)(i, nCol-1) << row_sep;
        }
        return oss.str();
    }
    template<typename ScalarType>
    std::ostream& operator<<(std::ostream& out, const DenseMatrix<ScalarType>& v){ return out << v.to_string(); }

    template<typename ScalarType, typename IndexType>
    struct BandDenseMatrixX;
    ///Special format for storing matrices generated for vector FEM types
    ///@see Operator<OPERATOR, FemVec<DIM, FEM_TYPE>>
    template<int NPART = 1, typename ScalarType = double, typename IndexType = int>
    struct BandDenseMatrix {
        using Scalar = ScalarType;
        using Index = IndexType;
        using Nparts = std::integral_constant<std::size_t, NPART>;
        DenseMatrix<ScalarType> data[NPART];
        IndexType stRow[NPART+1] = {0}, stCol[NPART+1] = {0};

        BandDenseMatrix() = default;
        explicit BandDenseMatrix(const BandDenseMatrixX<ScalarType, IndexType>& A);
    };

    template<typename ScalarType = double, typename IndexType = int>
    struct BandDenseMatrixX{
        using Scalar = ScalarType;
        using Index = IndexType;

        std::size_t nparts = 0;
        DenseMatrix<ScalarType>* data = nullptr;
        IndexType *stRow = nullptr, *stCol = nullptr;

        BandDenseMatrixX() = default;
        BandDenseMatrixX(std::size_t nparts, DenseMatrix<ScalarType>* data,  IndexType* stRow, IndexType* stCol): nparts{nparts}, data{data}, stRow{stRow}, stCol{stCol} {}
        template<int NPART = 1>
        explicit BandDenseMatrixX(BandDenseMatrix<NPART, ScalarType, IndexType>& A): nparts{NPART}, data{A.data}, stRow{A.stRow}, stCol{A.stCol} {}
    };

    template<int NPART, typename ScalarType, typename IndexType>
    BandDenseMatrix<NPART, ScalarType, IndexType>::BandDenseMatrix(const BandDenseMatrixX<ScalarType, IndexType>& A){
        assert(A.nparts == NPART && "Dimension mismatch");
        std::copy(A.data, A.data + NPART, data);
        std::copy(A.stRow, A.stRow + NPART+1, stRow);
        std::copy(A.stCol, A.stCol + NPART+1, stCol);
    }

    template<typename ScalarType, typename IndexType>
    inline BandDenseMatrix<1, ScalarType, IndexType> convToBendMx(DenseMatrix<ScalarType> A, IndexType dim, IndexType nf){
        BandDenseMatrix<1, ScalarType, IndexType> res;
        res.data[0] = A;
        res.stRow[0] = 0, res.stRow[1] = dim;
        res.stCol[0] = 0, res.stCol[1] = nf;
        return res;
    }
    template<int NPART, typename ScalarType, typename IndexType>
    inline BandDenseMatrix<NPART, ScalarType, IndexType> convToBendMx(BandDenseMatrix<NPART, ScalarType, IndexType> A, IndexType, IndexType){
        return A;
    }

    ///internal structure to store all data required for computing of fem elemental matrix
    ///@see internalFem3Dtet
    template<typename ScalarType = double, typename IndexType = int>
    struct AniMemory {
        using Scalar = ScalarType;
        using ArrayR = ArrayView<Scalar>;
        using ArrayI = ArrayView<IndexType>;
        ArrayR XYP;            ///< Coords of tetrahedron nodes relative to the first node: XYP = [XYP_1, ..., XYP_f], XYP_r = [P_1^r-P_1^r, P_2^r-P_1^r, P_3^r-P_1^r, P_4^r-P_1^r]
        ArrayR PSI;            ///< Transform matrices from actual tetrahedron to canonical tetrahedron: y^r = PSI_r * (x^r - x_0^r), PSI = [PSI_1, ..., PSI_f]
        ArrayR XYG;            ///< Quadrature points: XYG_r = (p_{1r}^T, ..., p_{qr}^T)^T,  XYG = [XYG_1, ..., XYG_f]
        ArrayR DET;            ///< DET_r = det(P_2^r-P_1^r, P_3^r-P_1^r, P_4^r-P_1^r), DET = [DET_1, ..., DET_f]
        ArrayR MES;            ///< measure of the integrated domain, for example for tetrahedron MES_r = abs(DET_r) / 6, MES = [MES_1, ..., MES_f]
        ArrayR NRM;            ///< normal vector to some face of tetrahedron, NRM = [NRM_1, ..., NRM_f]
        ArrayR U, V, DIFF, DU; ///< working arrays
        ArrayR XYL, WG;        ///< arrays of quadrature formula: quadrature bary points and weights
        ArrayR extraR;         ///< extra Scalar memory
        ArrayI extraI;         ///< extra IndexType memory
        std::size_t q = 0;       ///< number of quadrature points
        std::size_t f = 1;       ///< number of tetrahedrons processed at the same time (fusion param)
    };

    template<typename ScalarType = double, typename IndexType = int>
    struct AniMemoryX: public AniMemory<ScalarType> {
        ArrayView<DenseMatrix<ScalarType>> MTX;
        ArrayView<IndexType> MTXI_ROW;
        ArrayView<IndexType> MTXI_COL;
        std::size_t busy_mtx_parts = 0;
    };

    ///internal structure to allocate on stack memory required for computing of fem elemental matrix
    ///@see fem3Dtet
    template<typename ScalarType = double, int FUSION = 1, int MAX_NGAUSS = 24>
    struct MemoryLegacy {
        using Scalar = ScalarType;
        Scalar U_d[9 * MAX_NGAUSS * 30 * FUSION];
        Scalar V_d[9 * MAX_NGAUSS * 30 * FUSION];
        Scalar DU_d[9 * MAX_NGAUSS * 30 * FUSION];
        Scalar Diff_d[9 * MAX_NGAUSS * 30 * FUSION];
        Scalar extra[3 * 20 * MAX_NGAUSS * FUSION];
        Scalar XYG_d[3 * MAX_NGAUSS * FUSION];
        Scalar XYL_d[4*MAX_NGAUSS];
        Scalar W_d[MAX_NGAUSS];
        Scalar XYP_d[3 * 4 * FUSION];
        Scalar PSI_d[3 * 3 * FUSION];
        Scalar normal_d[3 * FUSION];
        Scalar CDET_d[FUSION];
        Scalar MES_d[FUSION];


        template<typename IndexType>
        AniMemory<Scalar, IndexType> getAniMemory() {
            using AniMem = AniMemory<Scalar, IndexType>;
            AniMem am;
            am.U = typename AniMem::ArrayR(U_d, 9 * MAX_NGAUSS * 30 * FUSION);
            am.V = typename AniMem::ArrayR(V_d, 9 * MAX_NGAUSS * 30 * FUSION);
            am.DU = typename AniMem::ArrayR(DU_d, 9 * MAX_NGAUSS * 30 * FUSION);
            am.DIFF = typename AniMem::ArrayR(Diff_d, 9 * MAX_NGAUSS * 30 * FUSION);
            am.extraR = typename AniMem::ArrayR(extra, 3 * 20 * MAX_NGAUSS * FUSION);
            am.XYG = typename AniMem::ArrayR(XYG_d, 3 * MAX_NGAUSS * FUSION);
            am.XYP = typename AniMem::ArrayR(XYP_d, 3 * 4 * FUSION);
            am.PSI = typename AniMem::ArrayR(PSI_d, 3 * 3 * FUSION);
            am.DET = typename AniMem::ArrayR(CDET_d, FUSION);
            am.MES = typename AniMem::ArrayR(MES_d, FUSION);
            am.NRM = typename AniMem::ArrayR(normal_d, 3 * FUSION);
            am.XYL = typename AniMem::ArrayR(XYL_d, 4*MAX_NGAUSS);
            am.WG = typename AniMem::ArrayR(W_d, MAX_NGAUSS);
            return am;
        }
    };

    ///Single contiguous memory block that enough to work with fem3Dtet
    ///@see fem3Dtet, fem3Dtet_memory_requirements
    template<typename ScalarType = double, typename IndexType = int>
    struct PlainMemory{
        ScalarType* ddata = nullptr;
        IndexType* idata = nullptr;
        std::size_t dSize = 0, iSize = 0;

        void* allocateFromRaw(void* mem_in, ulong mem_sz, ulong dsize, ulong isize){
            void* p = mem_in;
            if (dsize > 0){
                std::size_t remain = (static_cast<char*>(mem_in) + mem_sz) - static_cast<char*>(p);
                ddata = static_cast<ScalarType*>(std::align(alignof(ScalarType), sizeof(ScalarType)*dsize, p, remain));
                if (ddata == nullptr) return nullptr;
                p = ddata + dsize;
            }
            if (isize > 0){
                std::size_t remain = (static_cast<char*>(mem_in) + mem_sz) - static_cast<char*>(p);
                idata = static_cast<IndexType*>(std::align(alignof(IndexType), sizeof(IndexType)*isize, p, remain));
                if (idata == nullptr) {
                    if (dsize > 0) 
                        ddata = nullptr;
                    return nullptr;    
                }
                p = idata + isize;
            }
            dSize = dsize > 0 ? dsize : std::max(dSize, std::size_t(0));
            iSize = isize > 0 ? isize : std::max(iSize, std::size_t(0));
            return p;
        }
        void* allocateFromRaw(void* mem_in, ulong mem_sz){ return allocateFromRaw(mem_in, mem_sz, dSize, iSize); }
        std::size_t enoughRawSize() const {
            ulong dsize = dSize > 0 ? static_cast<ulong>(dSize) : 0U;
            ulong isize = iSize > 0 ? static_cast<ulong>(iSize) : 0U;
            if (isize == 0 && dsize == 0) return 0;
            return dsize*sizeof(ScalarType) + isize*sizeof(IndexType) + std::max(alignof(IndexType), alignof(ScalarType));
        }
        bool ge(const PlainMemory<ScalarType, IndexType>& o) const { return dSize >= o.dSize && iSize >= o.iSize; }
        void extend_size(const PlainMemory<ScalarType, IndexType>& o) { dSize = std::max(o.dSize, dSize), iSize = std::max(o.iSize, iSize); }
        void append_size(const PlainMemory<ScalarType, IndexType>& o) { dSize += o.dSize, iSize += o.iSize; }
    };

    template<typename ScalarType = double, typename IndexType = int>
    struct PlainMemoryX{
        ScalarType* ddata = nullptr;
        IndexType* idata = nullptr;
        DenseMatrix<ScalarType>* mdata = nullptr;
        std::size_t dSize = 0, iSize = 0, mSize = 0;

        bool allocateFromPlainMemory(ScalarType* d_in, ulong d_sz, IndexType* i_in, ulong i_sz, ulong dsize, ulong isize, ulong msize, bool matr_from_scalars = true){
            if (i_sz < isize || d_sz < dsize)
                return false;
            IndexType* lidata = isize > 0 ? i_in : idata;
            ScalarType* lddata = dsize > 0 ? d_in : ddata;
            
            if (msize > 0){
                void* st_mtx = matr_from_scalars ? reinterpret_cast<void*>(d_in + dsize) : reinterpret_cast<void*>(i_in + isize);
                std::size_t remain = matr_from_scalars ? (d_sz - dsize)*sizeof(ScalarType) : (i_sz - isize)*sizeof(IndexType);
                std::size_t algn = alignof(DenseMatrix<ScalarType>), msz = mSize*sizeof(DenseMatrix<ScalarType>);
                st_mtx = std::align(algn, msz, st_mtx, remain);
                if (st_mtx == nullptr)
                    return false;
                mdata = static_cast<DenseMatrix<ScalarType>*>(st_mtx); 
            }
            ddata = lddata;
            idata = lidata;
            iSize = isize > 0 ? isize : std::max(iSize, std::size_t(0));
            dSize = dsize > 0 ? dsize : std::max(dSize, std::size_t(0));
            mSize = msize > 0 ? msize : std::max(mSize, std::size_t(0));
            
            return true;
        }
        bool allocateFromPlainMemory(PlainMemory<ScalarType, IndexType> mem, bool matr_from_scalars = true) { return allocateFromPlainMemory(mem.ddata, mem.dSize, mem.idata, mem.iSize, dSize, iSize, mSize, matr_from_scalars); }
        PlainMemory<ScalarType, IndexType> enoughPlainMemory(bool matr_from_scalars = true) const {
            PlainMemory<ScalarType, IndexType> p;
            p.iSize = iSize;
            p.dSize = dSize;
            if (mSize > 0){
                std::size_t ad = alignof(DenseMatrix<ScalarType>);
                std::size_t ab = matr_from_scalars ? alignof(ScalarType) : alignof(IndexType);
                std::size_t sd = mSize * sizeof(DenseMatrix<ScalarType>);
                std::size_t sb = matr_from_scalars ? sizeof(ScalarType) : sizeof(IndexType);
                std::size_t* sz = matr_from_scalars ? &(p.dSize) : &(p.iSize);
                std::size_t al_add = ad > ab ? (ad / ab - 1 + (ad % ab == 0 ? 0 : 1)) : 0;  
                *sz += al_add + sd / sb + (sd % sb == 0 ? 0 : 1);
            }
            return p;
        }
        void* allocateFromRaw(void* mem_in, ulong mem_sz, ulong dsize, ulong isize, ulong msize){
            void* p = mem_in;
            ulong sz[3] = {msize, dsize, isize};
            static constexpr ulong szof[3] = {sizeof(DenseMatrix<ScalarType>), sizeof(ScalarType), sizeof(IndexType)};
            void* mem_out[3]; //mdata, ddata, idata
            std::array<std::pair<unsigned char, ulong>, 3> szalign {std::pair<unsigned char, ulong>{0, alignof(DenseMatrix<ScalarType>)}, {1, alignof(ScalarType)}, {2, alignof(IndexType)}};
            std::sort(szalign.begin(), szalign.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
            for (int j = 0; j < 3; ++j){
                unsigned char i = szalign[j].first;
                if (sz[i] > 0){
                    std::size_t remain = (static_cast<char*>(mem_in) + mem_sz) - static_cast<char*>(p);
                    mem_out[i] = std::align(szalign[j].second, szof[i]*sz[i], p, remain);
                    if (mem_out[i] == nullptr) {
                        for (int k = 0; k < j; ++k) if (sz[szalign[k].first] > 0)
                            mem_out[szalign[k].first] = nullptr;
                        return nullptr;
                    }
                    p = static_cast<char*>(mem_out[i]) + szof[i]*sz[i];
                }
            }
            mdata = static_cast<DenseMatrix<ScalarType>*>(mem_out[0]);
            ddata = static_cast<ScalarType*>(mem_out[1]);
            idata = static_cast<IndexType*>(mem_out[2]);
            mSize = msize > 0 ? msize : std::max(mSize, std::size_t(0));
            dSize = dsize > 0 ? dsize : std::max(dSize, std::size_t(0));
            iSize = isize > 0 ? isize : std::max(iSize, std::size_t(0));
            return p;
        }
        void* allocateFromRaw(void* mem_in, ulong mem_sz){ return allocateFromRaw(mem_in, mem_sz, dSize, iSize, mSize); }
        std::size_t enoughRawSize() const {
            ulong dsize = dSize > 0 ? static_cast<ulong>(dSize) : 0U;
            ulong isize = iSize > 0 ? static_cast<ulong>(iSize) : 0U;
            ulong msize = mSize > 0 ? static_cast<ulong>(mSize) : 0U;
            if (isize == 0 && dsize == 0 && msize == 0) return 0;
            return msize*sizeof(DenseMatrix<ScalarType>) + dsize*sizeof(ScalarType) + isize*sizeof(IndexType) + std::max({alignof(DenseMatrix<ScalarType>), alignof(ScalarType), alignof(IndexType)});
        }
        bool ge(const PlainMemoryX& o) const { return dSize >= o.dSize && iSize >= o.iSize && mSize >= o.mSize; }
        void extend_size(const PlainMemoryX& o) { dSize = std::max(o.dSize, dSize), iSize = std::max(o.iSize, iSize), mSize = std::max(o.mSize, mSize); }
        void append_size(const PlainMemoryX& o) { dSize += o.dSize, iSize += o.iSize, mSize += o.mSize; }
    };

    struct OpMemoryRequirements{
        std::size_t Usz = 0;
        std::size_t extraRsz = 0;
        std::size_t extraIsz = 0;
        std::size_t mtx_parts = 1;
        OpMemoryRequirements() = default;
        OpMemoryRequirements(std::size_t Usz, std::size_t extraRsz, std::size_t extraIsz, std::size_t mtxParts): Usz{Usz}, extraRsz{extraRsz}, extraIsz{extraIsz}, mtx_parts{mtxParts} {}
    };
};

#endif //CARNUM_MEMORY_H
