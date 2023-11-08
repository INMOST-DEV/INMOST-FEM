//
// Created by Liogky Alexey on 01.07.2023.
//

#ifndef CARNUM_FUNCWRAP_H
#define CARNUM_FUNCWRAP_H

#include <ostream>
#include <functional>
#include "anifem++/fem/fem_memory.h"
#include "anifem++/fem/fem_memory_parallel.h"


namespace Ani{

    template <typename TInt = long>
    struct MatSparsityView{
        using Int = TInt;

        enum Type{
            DENSE = 0,
            SPARSE_CSC = 1
        };

        MatSparsityView() = default;
        MatSparsityView(Type tp, Int nnz, Int sz1, Int sz2, const Int* colind, const Int* row): m_tp{tp}, m_nnz{nnz}, m_sz1{sz1}, m_sz2{sz2}, m_colind{colind}, m_row{row} {}
        Type m_tp = DENSE;
        Int m_nnz = 0;
        Int m_sz1 = 0, m_sz2 = 0;
        const Int *m_colind = nullptr, *m_row = nullptr;
        
        //Convert sparsed by the sparsity matrix in raw array from to dense column major matrix
        template<typename Real>
        void densify(const Real* from, Real* to, Real fill_empty = Real(0)) const;
        template<typename Real>
        void densify(Real* inplace) { densify<Real>(inplace, inplace); }
        //Convert dense column major matrix sparsed by the sparsity matrix in sparsed raw array
        template<typename Real>
        void sparsify(const Real* from, Real* to) const;
        template<typename Real>
        void sparsify(const Real* inplace) const { sparsify<Real>(inplace, inplace); }
        //Copy colind and row arrays of CSC format
        template<typename IntTo>
        void copyCSC(IntTo* colind, IntTo* row) const;
        template<typename RandAccBoolIter, typename TVal = bool>
        void fillTemplate(RandAccBoolIter beg) const;
        static MatSparsityView<Int> make_as_dense(Int sz1, Int sz2) { return MatSparsityView<Int>(DENSE, sz1*sz2, sz1, sz2, nullptr, nullptr); }
        static MatSparsityView<Int> make_as_sparse(Int nnz, Int sz1, Int sz2, const Int* colind, const Int* row) { return MatSparsityView<Int>(SPARSE_CSC, sz1*sz2, sz1, sz2, colind, row); }
    };

    template <typename TInt = long>
    struct MatSparsity{
        using Int = TInt;
        using Type = typename MatSparsityView<TInt>::Type;

        MatSparsity() = default;
        MatSparsity(Type tp, Int nnz, Int sz1, Int sz2, const std::vector<Int>& colind, const std::vector<Int>& row): m_tp{tp}, m_nnz{nnz}, m_sz1{sz1}, m_sz2{sz2}, m_colind{colind}, m_row{row} {}
        Type m_tp = Type::DENSE;
        Int m_nnz = 0;
        Int m_sz1 = 0, m_sz2 = 0;
        std::vector<Int> m_colind, m_row;

        MatSparsityView<Int> getView() const 
            { return MatSparsityView<Int>(m_tp, m_nnz, m_sz1, m_sz2, m_colind.data(), m_row.data()); }
        template<typename Real> void densify(const Real* from, Real* to, Real fill_empty = Real(0)) const 
            { return getView().template densify<Real>(from, to, fill_empty); }
        template<typename Real> void densify(Real* inplace) 
            { return getView().template densify<Real>(inplace); }
        template<typename Real> void sparsify(const Real* from, Real* to) const 
            { return getView().template sparsify<Real>(from, to); }
        template<typename Real> void sparsify(const Real* inplace) const 
            { return getView().template sparsify<Real>(inplace); }
        template<typename IntTo> void copyCSC(IntTo* colind, IntTo* row) const 
            { return getView().template copyCSC<IntTo>(colind, row); }
        template<typename RandAccBoolIter, typename TVal = bool> void fillTemplate(RandAccBoolIter beg) const 
            { return getView().template fillTemplate<RandAccBoolIter, TVal>(beg); }
        static MatSparsity<Int> make_as_dense(Int sz1, Int sz2)
            { return MatSparsity(Type::DENSE, sz1*sz2, sz1, sz2, std::vector<Int>(), std::vector<Int>()); }
        static MatSparsity<Int> make_as_sparse(Int nnz, Int sz1, Int sz2, const std::vector<Int>& colind, const std::vector<Int>& row)
            { return MatSparsity(Type::SPARSE_CSC, nnz, sz1, sz2, colind, row); }
    };

    template <typename TReal = double, typename TInt = long>
    struct SparsedData{
        using Real = TReal;
        using Int = TInt;

        MatSparsityView<Int> m_sp;
        Real* m_dat;
        SparsedData() = default;
        SparsedData(MatSparsityView<Int> sp, Real* dat): m_sp{std::move(sp)}, m_dat{dat} {}

        void densify(Real* dense_data, Real fill_empty = Real(0)) const { m_sp.template densify<Real>(m_dat, dense_data, fill_empty); }
    };

    template <typename TReal = double, typename TInt = long>
    struct MatFuncWrap{
        using Real = TReal;
        using Int = TInt;

        ///Helper memory container
        struct Memory{
            Int   *m_iw = nullptr;
            Real  *m_w = nullptr;
            const Real **m_args = nullptr;
            Real **m_res = nullptr;
            void* user_data = nullptr;
            Int mem_id = -1;
        };

        ///Make numerical evaluation
        ///@param args is working array of const Real pointers, first n_in() elements is pointers on some input parameters
        ///@param res is working array of Real pointers, first n_out() elements will contain pointers of result matrices
        ///@param iw is working integer data
        ///@param w is working real data
        ///@param user_data is some working user-defined data (usually it is not required)
        ///@param mem_id is thread-safe index of internal allocated memory
        ///@return 0 on success or code of error
        virtual int operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr, Int mem_id = -1) const = 0;
        int operator()(Memory mem) const { return operator()(mem.m_args, mem.m_res, mem.m_w, mem.m_iw, mem.user_data, mem.mem_id); }
        
        /// Setup the function and allocated some thread-safe internal memory
        /// @note this method of memory allocation is more convenient for std::thread multi-threading
        /// @return is identificator of allocated internal memory
        virtual Int setup_and_alloc_memory() { return 0; };
        /// Release internal memory
        /// @param mem_id is thread-safe index of internal allocated memory
        virtual void release_memory(Int mem_id) { (void) mem_id; }
        /// Defragment internal allocated memory to improve memory locality
        /// @note the method is useful if the matrix function ( operator() ) 
        /// must be called many times and the same memory template is allocated for each calculation 
        virtual void defragment_memory(Int mem_id) { (void) mem_id; }

        /// Setup the function and allocate internal memory for num_threads amount of threads
        /// The identificators of allocated memory are numbers from 0 to num_threads-1
        /// @note this method of memory allocation is more convenient for OpenMP multi-threading
        /// @return true on success
        virtual std::pair<Int, Int> setup_and_alloc_memory_range(Int num_threads) { (void) num_threads; return {-1, -1}; }
        /// Release internal memory allocated by setup_and_alloc_memory_range call 
        /// @note this method of memory release is more convenient for OpenMP multi-threading
        virtual void release_memory_range(std::pair<Int, Int> id_range) { (void) id_range; }
        
        /// Release all internal memory
        virtual void release_all_memory() {}
        /// @brief Clear internally allocated memory
        virtual void clear_memory() { }

        ///Set minimal required sizes of corresponding arrays
        virtual void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const {
            sz_args = n_in();
            sz_res = n_out();
            sz_w = sz_iw = 0;
        }
        ///Return true if specific user_data required
        virtual bool is_user_data_required() const { return false; }
        virtual size_t n_in() const { return 0; }; ///< Number of input parameters
        virtual size_t n_out() const = 0; ///< Number of output matrices

        virtual MatSparsityView<Int> out_sparsity(Int res_id) const = 0;
        virtual Int out_nnz  (Int res_id) const { return out_sparsity(res_id).m_nnz; }  ///< Expected count of elements in res[res_id] result data array
        virtual Int out_size1(Int res_id) const { return out_sparsity(res_id).m_sz1; }  ///< first dimension of res[res_id] matrix
        virtual Int out_size2(Int res_id) const { return out_sparsity(res_id).m_sz2; }  ///< second dimension of res[res_id] matrix
        ///Set csc indexes in colind, row arrays if they are not NULL
        ///@param[in] res_id is index of result parameter
        ///@param[in] colind, row are arrays with out_size2()+1 and out_nnz(res_id) sizes correspondingly
        ///@param[out] colind, row are arrays with csc indexes
        virtual void out_csc(Int res_id, Int* colind, Int* row) const { out_sparsity(res_id).copyCSC(colind, row); }

        ///Next functions are just informative and are not used in computations
        ///But we advise to set them to facilitate possible debugging

        virtual MatSparsityView<Int> in_sparsity(Int arg_id) const = 0;
        virtual Int in_nnz  (Int arg_id) const { return in_sparsity(arg_id).m_nnz; }    ///< Expected count of elements in args[arg_id] input data array
        virtual Int in_size1(Int arg_id) const { return in_sparsity(arg_id).m_sz1; }    ///< first dimension of args[args_id] matrix
        virtual Int in_size2(Int arg_id) const { return in_sparsity(arg_id).m_sz2; }    ///< second dimension of args[args_id] matrix
        ///Set csc indexes in colind, row arrays if they are not NULL
        ///@param[in] arg_id is index of input parameter
        ///@param[in] colind, row are arrays with in_size2()+1 and in_nnz(arg_id) sizes correspondingly
        ///@param[out] colind, row are arrays with csc indexes
        virtual void in_csc(Int arg_id, Int* colind, Int* row) const { in_sparsity(arg_id).copyCSC(colind, row); }

        /// Return true if the wrapper prepared for computations
        virtual bool isValid() const { return false; }
        operator bool() const { return isValid(); }

        std::ostream& print_signature(std::ostream& out = std::cout) const;
    };

    template <ThreadPar::Type ParType = ThreadPar::Type::NONE, typename TReal = double, typename TInt = long>
    struct MatFuncWrapHolder{
        using Base = MatFuncWrap<TReal, TInt>;
        using Real = typename Base::Real;
        using Int = typename Base::Int;
        using Memory = typename Base::Memory;
        static const ThreadPar::Type parallel_type = ParType;

        std::shared_ptr<Base> m_invoker;

        int operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr, Int mem_id = -1) const{ return m_invoker->operator()(args, res, w, iw, user_data, mem_id); }
        int operator()(Memory mem) const { return m_invoker->operator()(std::move(mem));}
        Int setup_and_alloc_memory(){ return m_invoker->setup_and_alloc_memory(); }
        void release_memory(Int mem_id) { return m_invoker->release_memory(mem_id); }
        void defragment_memory(Int mem_id){ return m_invoker->defragment_memory(mem_id); }
        std::pair<Int, Int> setup_and_alloc_memory_range(Int num_threads){ return m_invoker->setup_and_alloc_memory_range(num_threads); }
        void release_memory_range(std::pair<Int, Int> id_range){ return m_invoker->release_memory_range(id_range); }
        void release_all_memory(){ return m_invoker->release_all_memory(); }
        void clear_memory(){ return m_invoker->clear_memory(); }
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const{ return m_invoker->working_sizes(sz_args, sz_res, sz_w, sz_iw); }
        bool is_user_data_required() const{ return m_invoker->is_user_data_required(); }
        size_t n_in() const{ return m_invoker->n_in(); }
        size_t n_out() const{ return m_invoker->n_out(); }
        MatSparsityView<Int> out_sparsity(Int res_id) const{ return m_invoker->out_sparsity(res_id); }
        Int out_nnz  (Int res_id) const{ return m_invoker->out_nnz(res_id); }
        Int out_size1(Int res_id) const{ return m_invoker->out_size1(res_id); }
        Int out_size2(Int res_id) const{ return m_invoker->out_size2(res_id); }
        void out_csc(Int res_id, Int* colind, Int* row) const{ return m_invoker->out_csc(res_id, colind, row); }
        MatSparsityView<Int> in_sparsity(Int arg_id) const{ return m_invoker->in_sparsity(arg_id); }
        Int in_nnz  (Int arg_id) const{ return m_invoker->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const{ return m_invoker->in_size1(arg_id); }
        Int in_size2(Int arg_id) const{ return m_invoker->in_size2(arg_id); }
        void in_csc(Int arg_id, Int* colind, Int* row) const{ return m_invoker->in_csc(arg_id, colind, row); }
        std::ostream& print_signature(std::ostream& out = std::cout) const{ return m_invoker->print_signature(out); }
        bool isValid() const { return m_invoker && m_invoker->isValid(); }
        operator bool() const { return isValid(); }

        MatFuncWrapHolder() = default;
        MatFuncWrapHolder(const MatFuncWrapHolder&) = default;
        MatFuncWrapHolder(MatFuncWrapHolder&&) = default;
        template<typename T>
        MatFuncWrapHolder(T&& f, typename std::enable_if<std::is_base_of<Base, T>::value>::type* = 0): m_invoker{new T(std::move(f))} {}
        MatFuncWrapHolder(std::shared_ptr<Base> f): m_invoker(std::move(f)) {}
        
        MatFuncWrapHolder& operator=(std::shared_ptr<Base> f){ return m_invoker = std::move(f), *this; }
        template<typename T, typename std::enable_if<std::is_base_of<Base, T>::value>::type* U = 0>
        MatFuncWrapHolder& operator=(T&& f){ return m_invoker = std::make_shared<T>(std::move(f)), *this; }
        template<typename T, typename std::enable_if<std::is_base_of<Base, T>::value>::type* U = 0>
        MatFuncWrapHolder& operator=(const T& f){ return m_invoker = std::make_shared<T>(f), *this; }
        MatFuncWrapHolder& operator=(const MatFuncWrapHolder&) = default;
        MatFuncWrapHolder& operator=(MatFuncWrapHolder&&) = default;
    };

    template <ThreadPar::Type ParType = ThreadPar::Type::NONE, typename TReal = double, typename TInt = long>
    struct MatFuncWrapDynamic: public MatFuncWrap<TReal, TInt>{
        using Real = TReal;
        using Int = TInt;
        using Functor = std::function<int(const Real**, Real**, Real*, Int*, void*, DynMem<Real, Int>*)>;
        struct IOData{
            const Real** args;
            Real** res;
            Real* w;
            Int* iw;
            void* user_data;
            DynMem<Real, Int>* allocator;
        };
        static const ThreadPar::Type parallel_type = ParType;

        int operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr, Int mem_id = -1) const override{
            bool empty_mem = (!m_dyn_mem_required || mem_id < 0);
            DynMem<Real, Int>* dmem = empty_mem ? nullptr : m_mem.get(mem_id);   
            return m_f(args, res, w, iw, user_data, dmem);
        }

        Int setup_and_alloc_memory() override { return m_dyn_mem_required ? m_mem.alloc() : 0; }
        void release_memory(Int mem_id) override { if (m_dyn_mem_required) m_mem.release(mem_id); }
        std::pair<Int, Int> setup_and_alloc_memory_range(Int num_threads) override { return m_dyn_mem_required ? m_mem.alloc_range(num_threads) : std::pair<Int, Int>(-num_threads, 0); }
        void release_memory_range(std::pair<Int, Int> id_range) override { if (m_dyn_mem_required) m_mem.release_range(id_range); }
        void release_all_memory() override{ if (m_dyn_mem_required) m_mem.release_all(); }
        void clear_memory() override { if (m_dyn_mem_required) m_mem.clear(); }
        void defragment_memory(Int mem_id) override { if (m_dyn_mem_required && mem_id>=0) m_mem.get(mem_id)->defragment(); }

        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
            sz_args = m_sp_in.size(), sz_res = m_sp_out.size(); sz_w = m_sz_w; sz_iw = m_sz_iw;
        }
        bool is_user_data_required() const override { return m_user_data_required; }
        size_t n_in() const override { return m_sp_in.size(); }
        size_t n_out() const override { return m_sp_out.size(); }
        MatSparsityView<Int> out_sparsity(Int res_id) const override { return m_sp_out[res_id].getView(); } 
        Int out_nnz  (Int res_id) const override { return m_sp_out[res_id].m_nnz; }  
        Int out_size1(Int res_id) const override { return m_sp_out[res_id].m_sz1; }  
        Int out_size2(Int res_id) const override { return m_sp_out[res_id].m_sz2; }
        void out_csc(Int res_id, Int* colind, Int* row) const override { m_sp_out[res_id].copyCSC(colind, row); }
        MatSparsityView<Int> in_sparsity(Int arg_id) const override { return m_sp_in[arg_id].getView(); } 
        Int in_nnz  (Int arg_id) const override { return m_sp_in[arg_id].m_nnz; }  
        Int in_size1(Int arg_id) const override { return m_sp_in[arg_id].m_sz1; }  
        Int in_size2(Int arg_id) const override { return m_sp_in[arg_id].m_sz2; }
        void in_csc(Int arg_id, Int* colind, Int* row) const override { m_sp_in[arg_id].copyCSC(colind, row); }
        bool isValid() const override { return m_f != nullptr; }
        operator bool() const { return isValid(); }

        MatFuncWrapDynamic() = default;
        MatFuncWrapDynamic(MatFuncWrapDynamic<ParType, TReal, TInt>&&) = default;
        MatFuncWrapDynamic(const MatFuncWrapDynamic<ParType, TReal, TInt>& a): 
            m_sp_in{a.m_sp_in}, m_sp_out{a.m_sp_out}, m_f{a.m_f}, m_sz_w{a.m_sz_w}, m_sz_w{a.m_sz_iw}, 
            m_user_data_required{a.m_user_data_required}, m_dyn_mem_required{a.m_dyn_mem_required} {}
        MatFuncWrapDynamic& operator=(MatFuncWrapDynamic<ParType, TReal, TInt>&&) = default;
        MatFuncWrapDynamic& operator=(const MatFuncWrapDynamic<ParType, TReal, TInt>& a);
        MatFuncWrapDynamic(Functor f, std::vector<MatSparsity<Int>> sp_in, std::vector<MatSparsity<Int>> sp_out, 
                            size_t sz_w = 0, size_t sz_iw = 0, bool with_user_data = true, bool use_dyn_mem = true):
            m_sp_in{std::move(sp_in)}, m_sp_out{std::move(sp_out)}, m_f{std::move(f)}, m_sz_w{sz_w}, m_sz_iw{sz_iw}, 
            m_user_data_required{with_user_data}, m_dyn_mem_required{use_dyn_mem} {}   

        std::vector<MatSparsity<Int>> m_sp_in, m_sp_out;
        std::function<int(const Real** args, Real** res, Real* w, Int* iw, void* user_data, Ani::DynMem<Real, Int>* dyn_mem)> m_f;
        mutable DynMemParallel<Real, Int, ThreadPar::mutex<ParType>> m_mem;
        size_t m_sz_w = 0, m_sz_iw = 0;
        bool m_user_data_required = true, m_dyn_mem_required = true;
    };

    ///Wrap some variants of elemental matrix and rhs evaluators
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(typename MatFuncWrapDynamic<ParType, Real, Int>::IOData)> f, size_t nRow, size_t nCol, size_t nw, size_t niw, bool with_user_data = true, bool use_dyn_mem = true);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F)> f, size_t nRow, size_t nCol);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, void* user_data)> f, size_t nRow, size_t nCol);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data)> f, size_t nRow, size_t nCol, size_t nw, size_t niw);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw)> f, size_t nRow, size_t nCol, size_t nw, size_t niw);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)> f, size_t nRow, size_t nCol, size_t nw, size_t niw);

    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(typename MatFuncWrapDynamic<ParType, Real, Int>::IOData)> f, size_t nRow, size_t nCol, size_t nw, size_t niw, bool with_user_data = true, bool use_dyn_mem = true);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A)> f, size_t nRow, size_t nCol);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, void* user_data)> f, size_t nRow, size_t nCol);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data)> f, size_t nRow, size_t nCol, size_t nw, size_t niw);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw)> f, size_t nRow, size_t nCol, size_t nw, size_t niw);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)> f, size_t nRow, size_t nCol, size_t nw, size_t niw);

    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(typename MatFuncWrapDynamic<ParType, Real, Int>::IOData)> f, size_t nRow, size_t nw, size_t niw, bool with_user_data = true, bool use_dyn_mem = true);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F)> f, size_t nRow);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, void* user_data)> f, size_t nRow);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw, void* user_data)> f, size_t nRow, size_t nw, size_t niw);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw)> f, size_t nRow, size_t nw, size_t niw);
    template<ThreadPar::Type ParType = ThreadPar::Type::NONE, typename Real = double, typename Int = long>
    MatFuncWrapDynamic<ParType, Real, Int> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw, void* user_data, DynMem<Real, Int>* dyn_mem)> f, size_t nRow, size_t nw, size_t niw);
}

#include "func_wrap.inl"

#endif //CARNUM_FUNCWRAP_H