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
#include <cassert>
#include <cmath>
#include <vector>

typedef unsigned long int ulong;
typedef unsigned int uint;

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
        ArrayView(const ArrayView<Scalar>& s) = default;
        ArrayView(ArrayView<Scalar>&& s) = default;
        ArrayView& operator=(const ArrayView<Scalar>& s) = default;
        ArrayView& operator=(ArrayView<Scalar>&& s) = default;
        
        template<typename OtherScalar, typename = typename std::enable_if<std::is_const<Scalar>::value && std::is_same<typename std::remove_const<Scalar>::type, OtherScalar>::value>::type>
        ArrayView(ArrayView<OtherScalar> s): data{const_cast<Scalar*>(s.data)}, size{s.size} {}
    };

    ///Store dense matrix
    template<typename ScalarType = double>
    struct DenseMatrix{
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
        template<typename RT>
        DenseMatrix<ScalarType>& operator+=(const DenseMatrix<RT>& other){ 
            assert(nRow == other.nRow && nCol == other.nCol && "Sizes incompatible");
            for (std::size_t i = 0; i < nRow*nCol; ++i)
                data[i] += other.data[i];
            return *this;    
        }
        template<typename RT>
        DenseMatrix<ScalarType>& operator-=(const DenseMatrix<RT>& other){ 
            assert(nRow == other.nRow && nCol == other.nCol && "Sizes incompatible");
            for (std::size_t i = 0; i < nRow*nCol; ++i)
                data[i] -= other.data[i];
            return *this;    
        }
        template<typename RT>
        DenseMatrix<ScalarType>& operator*=(RT val){
            for (std::size_t i = 0; i < nRow*nCol; ++i)
                data[i] *= val;
            return *this;    
        }
        template<typename RT>
        DenseMatrix<ScalarType>& operator/=(RT val){
            ScalarType inv_val = ScalarType(1) / val;
            return (*this *= inv_val);
        }
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
            nrm = sqrt(nrm)*fabs(a);
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
                const auto& val = (*this)(i, j);
                oss << sign_shift(val) << val << val_sep;
            }
            if (nCol > 0) {
                const auto& val = (*this)(i, nCol-1);
                oss << sign_shift(val) << val << row_sep;
            }
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
        template<typename ScalarType1 = double, typename IndexType1 = int>
        bool ge(const PlainMemoryX<ScalarType1, IndexType1>& o) const { return dSize >= o.dSize && iSize >= o.iSize && mSize >= o.mSize; }
        template<typename ScalarType1 = double, typename IndexType1 = int>
        void extend_size(const PlainMemoryX<ScalarType1, IndexType1>& o) { dSize = std::max(o.dSize, dSize), iSize = std::max(o.iSize, iSize), mSize = std::max(o.mSize, mSize); }
        template<typename ScalarType1 = double, typename IndexType1 = int>
        void append_size(const PlainMemoryX<ScalarType1, IndexType1>& o) { dSize += o.dSize, iSize += o.iSize, mSize += o.mSize; }
    };

    template<typename ScalarType = double, typename IndexType = int>
    struct DynMem{
        using DM = DenseMatrix<ScalarType>; 
        struct Chunk{
            std::vector<ScalarType> rdata;
            std::vector<IndexType>  idata;
            std::vector<DM>         mdata;
            std::size_t rbusy = 0, ibusy = 0, mbusy = 0;
            std::size_t nparts = 0; //number of memory parts where pointers from the vectors are used
        };
        struct MemPart{  
            PlainMemoryX<ScalarType, IndexType> m_mem;
            DynMem* m_link = nullptr;
            std::size_t m_chunk_id = -1, m_part_id = -1;
        
            PlainMemory<ScalarType, IndexType>  getPlainMemory(){ return {m_mem.ddata, m_mem.idata, m_mem.dSize, m_mem.iSize}; }
            PlainMemoryX<ScalarType, IndexType> getPlainMemoryX(){ return m_mem; }
            MemPart() = default;
            MemPart(const MemPart& a) = delete;
            MemPart& operator=(const MemPart& a) = delete;
            MemPart(MemPart&& a){
                clear();
                m_mem = a.m_mem;
                m_link = a.m_link; m_chunk_id = a.m_chunk_id; m_part_id = a.m_part_id;
                a.m_link = nullptr; a.m_chunk_id = -1; a.m_part_id = -1;
                a.m_mem = PlainMemoryX<ScalarType, IndexType>(); 
            }
            MemPart& operator=(MemPart&& a){
                if (&a == this) return *this;
                clear();
                m_mem = a.m_mem;
                m_link = a.m_link; m_chunk_id = a.m_chunk_id; m_part_id = a.m_part_id;
                a.m_link = nullptr; a.m_chunk_id = -1; a.m_part_id = -1;
                a.m_mem = PlainMemoryX<ScalarType, IndexType>(); 
                return *this;
            }
            void clear(){
                if (!m_link) return;
                m_link->clearChunk(m_chunk_id, m_part_id, m_mem.dSize, m_mem.iSize, m_mem.mSize);
                m_mem = PlainMemoryX<ScalarType, IndexType>();  
                m_link = nullptr;  
            }
            ~MemPart(){
                if (!m_link) return;
                m_link->clearChunk(m_chunk_id, m_part_id, m_mem.dSize, m_mem.iSize, m_mem.mSize);
            }
            friend class DynMem<ScalarType, IndexType>;
        };
        
        virtual MemPart alloc(std::size_t dsize, std::size_t isize, std::size_t msize){
            MemPart res;
            res.m_link = this;
            Chunk* chunk = nullptr;
            if (!m_chunks.empty()) {
                for (std::size_t i = 0; i < m_chunks.size(); ++i){
                    auto* lchk =  m_chunks.data() + i;
                    if   ( (lchk->rbusy == 0 || lchk->rdata.size() >= lchk->rbusy + dsize)
                        && (lchk->ibusy == 0 || lchk->idata.size() >= lchk->ibusy + isize)
                        && (lchk->mbusy == 0 || lchk->mdata.size() >= lchk->mbusy + msize)){
                        chunk = lchk;
                        break;
                    }
                }
                if (chunk == nullptr){
                    m_chunks.resize(m_chunks.size() + 1);
                    chunk = &m_chunks.back();
                }
            } else  {
                m_chunks.resize(1);
                chunk = m_chunks.data();
            }
            res.m_chunk_id = std::distance(m_chunks.data(), chunk);
            if (dsize > 0){
                if (chunk->rbusy == 0 && chunk->rdata.size() < dsize) chunk->rdata.resize(dsize);
                res.m_mem.ddata = chunk->rdata.data() + chunk->rbusy; chunk->rbusy += dsize; res.m_mem.dSize = dsize;
            }
            if (isize > 0){
                if (chunk->ibusy == 0 && chunk->idata.size() < isize) chunk->idata.resize(isize);
                res.m_mem.idata = chunk->idata.data() + chunk->ibusy; chunk->ibusy += isize; res.m_mem.iSize = isize;
            }
            if (msize > 0){
                if (chunk->mbusy == 0 && chunk->mdata.size() < msize) chunk->mdata.resize(msize);
                res.m_mem.mdata = chunk->mdata.data() + chunk->mbusy; chunk->mbusy += msize; res.m_mem.mSize = msize;
            }
            res.m_part_id = chunk->nparts;
            chunk->nparts++;
            return res;
        }
        virtual void defragment(){
            if (m_chunks.size() <= 1) return;
            std::size_t rsz = 0, isz = 0, msz = 0;
            for (auto& chunk: m_chunks){
                rsz += chunk.rdata.size();
                isz += chunk.idata.size();
                msz += chunk.mdata.size();
                if (chunk.nparts != 0)
                    throw std::runtime_error("Defragmentation of busy memory is not allowed");
            }
            m_chunks.clear();
            m_chunks.resize(1);
            m_chunks[0].rdata.resize(rsz);
            m_chunks[0].idata.resize(isz);
            m_chunks[0].mdata.resize(msz);
        }
    // protected:
        virtual void clearChunk(std::size_t chunk_id, std::size_t part_id, std::size_t dSize, std::size_t iSize, std::size_t mSize) {
            auto& chunk = m_chunks[chunk_id];
            chunk.nparts--;
            if (chunk.nparts == part_id)
                chunk.rbusy -= dSize, chunk.ibusy -= iSize, chunk.mbusy -= mSize;
            if (chunk.nparts == 0)
                chunk.rbusy = chunk.ibusy = chunk.mbusy = 0;
        }
    public:    
    //protected:
        std::vector<Chunk> m_chunks;
    };

    template<typename ScalarType1, typename IndexType1, typename ScalarType0, typename IndexType0>
    struct DynMemT: public DynMem<ScalarType1, IndexType1>{
        DynMem<ScalarType0, IndexType0>* m_invoker;
        using DM = DenseMatrix<ScalarType1>;

        typename DynMem<ScalarType1, IndexType1>::MemPart alloc(std::size_t dsize, std::size_t isize, std::size_t msize) override{
            static_assert(sizeof(ScalarType1) <= sizeof(ScalarType0) && sizeof(IndexType1) <= sizeof(IndexType0) &&
                "This adaptor work only for emulation of storing objects that less then expected");
            typename DynMem<ScalarType0, IndexType0>::MemPart tmem = m_invoker->alloc(dsize, isize, msize);
            typename DynMem<ScalarType1, IndexType1>::MemPart mem;
            if (dsize > 0) mem.m_mem.ddata = new (tmem.m_mem.ddata) ScalarType1[dsize];
            if (isize > 0) mem.m_mem.idata = new (tmem.m_mem.idata) IndexType1[isize];
            if (msize > 0) mem.m_mem.mdata = new (tmem.m_mem.mdata) DM[msize];
            mem.m_mem.dSize = dsize, mem.m_mem.iSize = isize; mem.m_mem.mSize = msize;
            mem.m_link = this; tmem.m_link = nullptr;
            mem.m_chunk_id = tmem.m_chunk_id, mem.m_part_id = tmem.m_part_id;
            return mem;
        }
        void defragment() override { m_invoker->defragment();}
    protected:
        void clearChunk(std::size_t chunk_id, std::size_t part_id, std::size_t dSize, std::size_t iSize, std::size_t mSize) override { 
            m_invoker->clearChunk(chunk_id, part_id, dSize, iSize, mSize);
        }
    };
    template<typename ScalarType1, typename IndexType1, typename DYNMEM>
    DynMemT<ScalarType1, IndexType1, 
            typename decltype(std::declval<typename DYNMEM::Chunk>().rdata)::value_type, 
            typename decltype(std::declval<typename DYNMEM::Chunk>().idata)::value_type> makeAdaptor(DYNMEM& mem){
        using R1 = ScalarType1; 
        using I1 = IndexType1;
        using R0 = typename decltype(std::declval<typename DYNMEM::Chunk>().rdata)::value_type;
        using I0 = typename decltype(std::declval<typename DYNMEM::Chunk>().idata)::value_type;
        using DynMemT1 = DynMemT<R1, I1, R0, I0>;
        DynMemT1 m;
        m.m_invoker = &mem;
        return m;
    }

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
