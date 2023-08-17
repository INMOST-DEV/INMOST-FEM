//
// Created by Liogky Alexey on 01.07.2023.
//

#ifndef CARNUM_FUNCWRAP_INL
#define CARNUM_FUNCWRAP_INL

#include "func_wrap.h"
namespace Ani{

    template <typename TInt> template <typename Real>
    void MatSparsityView<TInt>::densify(const Real* from, Real* to, Real fill_empty) const {
        if (m_tp == DENSE || m_nnz == m_sz1*m_sz2){
            if (to == from) return;
            std::copy(from, from + m_nnz, to); 
        } else { // tp == SPARSE_CSC
            Int lda = m_sz1;
            const Int *colind = m_colind, *row = m_row;
            for (Int i = m_sz2-1; i >= 0; --i){
                Int k = lda - 1;
                for (Int j = colind[i+1]-1; j >= colind[i]; --j){
                    for (; k > row[j]; --k)
                        to[i * lda + k] = fill_empty;
                    to[i * lda + row[j]] = from[j];
                    --k;    
                }
                for (; k>=0; --k)
                    to[i * lda + k] = fill_empty;
            }
        }
    }

    template <typename TInt> template<typename Real>
    void MatSparsityView<TInt>::sparsify(const Real* from, Real* to) const {
        if (m_tp == DENSE || m_nnz == m_sz1*m_sz2){
            if (to == from) return;
            std::copy(from, from + m_nnz, to); 
        } else {
            for (Int i = 0; i < m_sz2; ++i){
                for (Int j = m_colind[i]; j < m_colind[i+1]; ++j){
                    to[j] = from[i * m_sz1 + m_row[j]];
                }
            }
        }
    }

    template <typename TInt> template<typename IntTo>
    void MatSparsityView<TInt>::copyCSC(IntTo* colind, IntTo* row) const {
        if (m_tp == MatSparsityView<Int>::DENSE){
            for (Int i = 0; i < m_sz2+1; ++i) colind[i] = i*m_sz1;
            if (m_sz2 > 0) for (Int i = 0; i < m_sz1; ++i) row[i] = i;
            for (int i = 1; i < m_sz2; ++i)
                std::copy(row, row + m_sz1, row + m_sz1*i);
        } else {
            std::copy(m_colind, m_colind+m_sz2+1, colind);
            std::copy(m_row, m_row+m_nnz, row);
        }
    }

    template <typename TInt> template <typename RandAccBoolIter, typename TVal>
    void MatSparsityView<TInt>::fillTemplate(RandAccBoolIter beg) const{
        if (m_tp == MatSparsityView<Int>::DENSE)
            std::fill(beg, beg + m_sz1*m_sz2, TVal(1));
        else  {
            std::fill(beg, beg + m_sz1*m_sz2, TVal(0));
            for (Int i = 0; i < m_sz2; ++i) {
                for (Int j = m_colind[i]; j < m_colind[i + 1]; ++j)
                    beg[i * m_sz1 + m_row[j]] = TVal(1);    
            }
        }   
    }

    template <typename TReal, typename TInt>
    std::ostream& MatFuncWrap<TReal, TInt>::print_signature(std::ostream &out) const {
        bool user_data_req = is_user_data_required();
        size_t sz_args = 0, sz_res = 0, sz_w = 0, sz_iw = 0;
        working_sizes(sz_args, sz_res, sz_w, sz_iw);
        out << "MatFuncWrap: ";
        out << "(";
        for (int i = 0; i < static_cast<int>(n_in()); ++i){
            if (i > 0) out << ", ";
            auto sp = in_sparsity(i);
            Int sz1 = sp.m_sz1, sz2 = sp.m_sz2, nnz = sp.m_nnz;
            out << "i" << i << "[" << sz1;
            if (sz2 > 1) out << "x" << sz2;
            if (nnz != sz1*sz2) out << " nnz="<<nnz;
            out << "]";
        }
        bool with_additional_memory = (sz_w > 0) || (sz_iw > 0) || (sz_args > n_in()) || (sz_res > n_out());
        if (user_data_req) out << ", user_data";
        if (with_additional_memory) {
            out << ", external mem{ ";
            if (sz_w > 0) out << "w[" << sz_w << "], ";
            if (sz_iw > 0) out << "iw[" << sz_iw << "], ";
            if (sz_args > n_in()) out << "args[" << sz_args << "], ";
            if (sz_res > n_out()) out << "res[" << sz_res << "] ";
            out << "}";
        }
        out << ")->(";
        for (int i = 0; i < static_cast<int>(n_out()); ++i){
            if (i > 0) out << ", ";
            auto sp = out_sparsity(i);
            Int sz1 = sp.m_sz1, sz2 = sp.m_sz2, nnz = sp.m_nnz;
            out << "o" << i << "[" << sz1;
            if (sz2 > 1) out << "x" << sz2;
            if (nnz != sz1*sz2) out << " nnz="<<nnz;
            out << "]";
        }
        out << ")";
        if (sz_args > n_in()) out << "; args[" << sz_args << "]";
        if (sz_res > n_out()) out << "; res[" << sz_res << "]";
        out << std::endl;
        return out;
    }

    template <ThreadPar::Type ParType, typename TReal, typename TInt>
    MatFuncWrapDynamic<ParType, TReal, TInt>& MatFuncWrapDynamic<ParType, TReal, TInt>::operator=(const MatFuncWrapDynamic<ParType, TReal, TInt>& a){
        if (this == &a) return *this;
        m_sp_in = a.m_sp_in, m_sp_out = a.m_sp_out;
        m_f = a.m_f; 
        m_sz_w = a.m_sz_w, m_sz_w = a.m_sz_iw; 
        m_user_data_required = a.m_user_data_required, m_dyn_mem_required = a.m_dyn_mem_required;
        return *this;
    }

    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(typename MatFuncWrapDynamic<ParType, Real, Int>::IOData)> f, size_t nRow, size_t nCol, size_t nw, size_t niw, bool with_user_data, bool use_dyn_mem){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ return f(MatFuncWrapDynamic<ParType, Real, Int>::IOData(args, res, w, iw, user_data, dyn_mem)), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol)), MatSparsity<Int>::make_as_dense(Int(nRow), 1)},
                nw, niw, with_user_data, use_dyn_mem};
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F)> f, size_t nRow, size_t nCol){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) w; (void) iw; (void) user_data; (void) dyn_mem; return f(args, res[0], res[1]), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol)), MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, 0, 0, false, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, void* user_data)> f, size_t nRow, size_t nCol){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)->int{ (void) w; (void) iw; (void) dyn_mem; return f(args, res[0], res[1], user_data), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol)), MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, 0, 0, true, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data)> f, size_t nRow, size_t nCol, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) dyn_mem; return f(args, res[0], res[1], w, iw, user_data), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol)), MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, nw, niw, true, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw)> f, size_t nRow, size_t nCol, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) user_data; (void) dyn_mem; return f(args, res[0], res[1], w, iw), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol)), MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, nw, niw, false, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)> f, size_t nRow, size_t nCol, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ return f(args, res[0], res[1], w, iw, user_data, dyn_mem), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol)), MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, nw, niw, true, true };
    }

    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(typename MatFuncWrapDynamic<ParType, Real, Int>::IOData)> f, size_t nRow, size_t nCol, size_t nw, size_t niw, bool with_user_data, bool use_dyn_mem){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ return f(MatFuncWrapDynamic<ParType, Real, Int>::IOData(args, res, w, iw, user_data, dyn_mem)), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol))},
                nw, niw, with_user_data, use_dyn_mem};
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A)> f, size_t nRow, size_t nCol){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) w; (void) iw; (void) user_data; (void) dyn_mem; return f(args, res[0]), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol))}, 0, 0, false, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, void* user_data)> f, size_t nRow, size_t nCol){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) w; (void) iw; (void) dyn_mem; return f(args, res[0], user_data), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol))}, 0, 0, true, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data)> f, size_t nRow, size_t nCol, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) dyn_mem; return f(args, res[0], w, iw, user_data), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol))}, nw, niw, true, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw)> f, size_t nRow, size_t nCol, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) user_data; (void) dyn_mem; return f(args, res[0], w, iw), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol))}, nw, niw, false, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)> f, size_t nRow, size_t nCol, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ return f(args, res[0], w, iw, user_data, dyn_mem), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), Int(nCol))}, nw, niw, true, true };
    }

    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(typename MatFuncWrapDynamic<ParType, Real, Int>::IOData)> f, size_t nRow, size_t nw, size_t niw, bool with_user_data, bool use_dyn_mem){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ return f(MatFuncWrapDynamic<ParType, Real, Int>::IOData(args, res, w, iw, user_data, dyn_mem)), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), 1)},
                nw, niw, with_user_data, use_dyn_mem};
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F)> f, size_t nRow){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) w; (void) iw; (void) user_data; (void) dyn_mem; return f(args, res[0]), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow))}, 0, 0, false, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, void* user_data)> f, size_t nRow, size_t nCol){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) w; (void) iw; (void) dyn_mem; return f(args, res[0], user_data), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, 0, 0, true, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw, void* user_data)> f, size_t nRow, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) dyn_mem; return f(args, res[0], w, iw, user_data), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, nw, niw, true, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw)> f, size_t nRow, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ (void) user_data; (void) dyn_mem; return f(args, res[0], w, iw), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, nw, niw, false, false };
    }
    template<ThreadPar::Type ParType, typename Real, typename Int>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)> f, size_t nRow, size_t nw, size_t niw){
        return {[f](const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)->int{ return f(args, res[0], w, iw, user_data, dyn_mem), 0; },
                std::vector<MatSparsity<Int>>(4, MatSparsity<Int>::make_as_dense(3, 1)), {MatSparsity<Int>::make_as_dense(Int(nRow), 1)}, nw, niw, true, true };
    }

}

#endif //CARNUM_FUNCWRAP_INL