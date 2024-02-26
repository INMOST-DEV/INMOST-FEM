//
// Created by Liogky Alexey on 29.03.2022.
//

#ifndef CARNUM_ASSEMBLER_H
#define CARNUM_ASSEMBLER_H

#include "global_enumerator.h"
#include "ordering.h"

namespace Ani{

struct InitValueSetFromTag{
    union GenTag{
        double v;
        INMOST::Tag t;
        INMOST::Tag* p;
        GenTag(): v(0) {}
        ~GenTag() {}
    };
    GenTag m_t;
    char m_tp = -1;

    InitValueSetFromTag() = default;
    InitValueSetFromTag(InitValueSetFromTag&& a) = default;
    InitValueSetFromTag(const InitValueSetFromTag& a){
        m_tp = a.m_tp;
        if (m_tp < 0) m_t.v = a.m_t.v;
        else if (m_tp > 0) m_t.p = a.m_t.p;
        else m_t.t = a.m_t.t;
    }
    InitValueSetFromTag& set(double v) { unset_tag(-1); m_tp = -1; m_t.v = v; return *this; }
    InitValueSetFromTag& set(INMOST::Tag t){ unset_tag(0); m_tp = 0; m_t.t = t; return *this;  }
    InitValueSetFromTag& set(INMOST::Tag* p){ unset_tag(1); m_tp = 1; m_t.p = p; return *this; }
    ~InitValueSetFromTag(){ if (m_tp == 0) m_t.t.~Tag(); }

    void operator()(ElementalAssembler& p) const {
        if (m_tp < 0){
            std::fill(p.vars->initValues.begin(), p.vars->initValues.end(), m_t.v);
            return;
        }
        INMOST::Tag t = (m_tp == 1) ? (*m_t.p) : m_t.t;
        ElementalAssembler::GatherDataOnElement(t, p, p.vars->initValues.data(), nullptr, 0);
    }; 
    operator bool() const { return m_tp < 0 || (m_tp == 1 ? (m_t.p != nullptr) : m_t.t.isValid()); }
private:
    void unset_tag(char tp){ if (tp != m_tp && m_tp == 0) m_t.t.~Tag(); }
};

struct InitValueSetTagVector{
    std::vector<INMOST::Tag> m_ts;

    InitValueSetTagVector() = default;
    InitValueSetTagVector& set(std::vector<INMOST::Tag> t){ m_ts = std::move(t); return *this; }
    void operator()(ElementalAssembler& p) const {
        ElementalAssembler::GatherDataOnElement(m_ts, p, p.vars->initValues.data(), nullptr, 0);
    }
     operator bool() const { return !m_ts.empty(); }
};

struct DefaultAssemblerTraits{
    using MatFuncT = MatFuncWrapHolder<>;
    using Real = typename MatFuncT::Real;
    using Int = typename MatFuncT::Int;
    using GlobEnumMap = GlobEnumeration;
    using ProbHandler = std::function<void(ElementalAssembler&)>;
    using InitValueSetter = std::function<void(ElementalAssembler&)>;
};

template<ThreadPar::Type ParType =
#ifdef WITH_OPENMP  
ThreadPar::Type::OMP
#else
ThreadPar::Type::STD
#endif
>
struct DefaultParallelAssemblerTraits{
    using MatFuncT = MatFuncWrapHolder<ParType>;
    using Real = typename MatFuncT::Real;
    using Int = typename MatFuncT::Int;
    using GlobEnumMap = GlobEnumeration;
    using ProbHandler = std::function<void(ElementalAssembler&)>;
    using InitValueSetter = std::function<void(ElementalAssembler&)>;
};

namespace internals{
    template<typename MatFuncT>
    struct MatFromSystemFWrap: public MatFuncWrap<typename MatFuncT::Real, typename MatFuncT::Int>{
        using Base = MatFuncT;
        using Real = typename Base::Real;
        using Int = typename Base::Int;
        using Memory = typename Base::Memory;

        bool is_mat_func = true;
        MatFuncT* m_f = nullptr;

        int operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr, Int mem_id = -1) const{ return m_f->operator()(args, res, w, iw, user_data, mem_id); }
        int operator()(Memory mem) const { return m_f->operator()(std::move(mem));}
        Int setup_and_alloc_memory(){ return m_f->setup_and_alloc_memory(); }
        void release_memory(Int mem_id) { return m_f->release_memory(mem_id); }
        void defragment_memory(Int mem_id){ return m_f->defragment_memory(mem_id); }
        std::pair<Int, Int> setup_and_alloc_memory_range(Int num_threads){ return m_f->setup_and_alloc_memory_range(num_threads); }
        void release_memory_range(std::pair<Int, Int> id_range){ return m_f->release_memory_range(id_range); }
        void release_all_memory(){ return m_f->release_all_memory(); }
        void clear_memory(){ return m_f->clear_memory(); }
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const{ return m_f->working_sizes(sz_args, sz_res, sz_w, sz_iw); }
        bool is_user_data_required() const{ return m_f->is_user_data_required(); }
        size_t n_in() const{ return m_f->n_in(); }
        size_t n_out() const{ return 1; }
        MatSparsityView<Int> out_sparsity(Int res_id) const{ return m_f->out_sparsity(res_id); }
        Int out_nnz  (Int res_id) const{ return m_f->out_nnz(res_id); }
        Int out_size1(Int res_id) const{ return m_f->out_size1(res_id); }
        Int out_size2(Int res_id) const{ return m_f->out_size2(res_id); }
        void out_csc(Int res_id, Int* colind, Int* row) const{ return m_f->out_csc(res_id, colind, row); }
        MatSparsityView<Int> in_sparsity(Int arg_id) const{ return m_f->in_sparsity(arg_id); }
        Int in_nnz  (Int arg_id) const{ return m_f->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const{ return m_f->in_size1(arg_id); }
        Int in_size2(Int arg_id) const{ return m_f->in_size2(arg_id); }
        void in_csc(Int arg_id, Int* colind, Int* row) const{ return m_f->in_csc(arg_id, colind, row); }
        std::ostream& print_signature(std::ostream& out = std::cout) const{ return m_f->print_signature(out); }
        bool isValid() const { return m_f && m_f->isValid(); }
        operator bool() const { return isValid(); }
        MatFromSystemFWrap() = default;
        MatFromSystemFWrap(MatFuncT* f, bool is_mat_func = true): is_mat_func{is_mat_func}, m_f{f} {} 
    };

    template<typename MatFuncT>
    struct RhsFromSystemFWrap: public MatFuncWrap<typename MatFuncT::Real, typename MatFuncT::Int>{
        using Base = MatFuncT;
        using Real = typename Base::Real;
        using Int = typename Base::Int;
        using Memory = typename Base::Memory;

        bool is_rhs_func = true;
        MatFuncT* m_f = nullptr;

        int operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr, Int mem_id = -1) const;
        int operator()(Memory mem) const;
        Int setup_and_alloc_memory(){ return m_f->setup_and_alloc_memory(); }
        void release_memory(Int mem_id) { return m_f->release_memory(mem_id); }
        void defragment_memory(Int mem_id){ return m_f->defragment_memory(mem_id); }
        std::pair<Int, Int> setup_and_alloc_memory_range(Int num_threads){ return m_f->setup_and_alloc_memory_range(num_threads); }
        void release_memory_range(std::pair<Int, Int> id_range){ return m_f->release_memory_range(id_range); }
        void release_all_memory(){ return m_f->release_all_memory(); }
        void clear_memory(){ return m_f->clear_memory(); }
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const{ return m_f->working_sizes(sz_args, sz_res, sz_w, sz_iw); }
        bool is_user_data_required() const{ return m_f->is_user_data_required(); }
        size_t n_in() const{ return m_f->n_in(); }
        size_t n_out() const{ return 1; }
        MatSparsityView<Int> out_sparsity(Int res_id) const{ return m_f->out_sparsity(res_id+(is_rhs_func ? 0 : 1) ); }
        Int out_nnz  (Int res_id) const{ return m_f->out_nnz(res_id+(is_rhs_func ? 0 : 1) ); }
        Int out_size1(Int res_id) const{ return m_f->out_size1(res_id+(is_rhs_func ? 0 : 1) ); }
        Int out_size2(Int res_id) const{ return m_f->out_size2(res_id+(is_rhs_func ? 0 : 1) ); }
        void out_csc(Int res_id, Int* colind, Int* row) const{ return m_f->out_csc(res_id+(is_rhs_func ? 0 : 1) , colind, row); }
        MatSparsityView<Int> in_sparsity(Int arg_id) const{ return m_f->in_sparsity(arg_id); }
        Int in_nnz  (Int arg_id) const{ return m_f->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const{ return m_f->in_size1(arg_id); }
        Int in_size2(Int arg_id) const{ return m_f->in_size2(arg_id); }
        void in_csc(Int arg_id, Int* colind, Int* row) const{ return m_f->in_csc(arg_id, colind, row); }
        std::ostream& print_signature(std::ostream& out = std::cout) const{ return m_f->print_signature(out); }
        bool isValid() const { return m_f && m_f->isValid(); }
        operator bool() const { return isValid(); }
        RhsFromSystemFWrap() = default;
        RhsFromSystemFWrap(MatFuncT* f, bool is_rhs_func = true): is_rhs_func{is_rhs_func}, m_f{f} {} 
    };
    template<typename MatFuncT>
    struct SystemFromMatRhsFWrap: public MatFuncWrap<typename MatFuncT::Real, typename MatFuncT::Int>{
        using Base = MatFuncT;
        using Real = typename Base::Real;
        using Int = typename Base::Int;
        using Memory = typename Base::Memory;

        MatFuncT* m_mat = nullptr;
        MatFuncT* m_rhs = nullptr;

        int operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr, Int mem_id = -1) const;
        int operator()(Memory mem) const;
        static Int encode_id(Int id1, Int id2) { return id1 + (id2 << 16); }
        static std::pair<Int, Int> decode_id(Int encoded_ids){ return {encoded_ids & 0xFFFF, (encoded_ids >> 16)}; }
        Int setup_and_alloc_memory();
        void release_memory(Int mem_id);
        void defragment_memory(Int mem_id);
        std::pair<Int, Int> setup_and_alloc_memory_range(Int num_threads);
        void release_memory_range(std::pair<Int, Int> id_range);
        void release_all_memory();
        void clear_memory();
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const;
        bool is_user_data_required() const{ return m_mat->is_user_data_required() || (m_rhs && m_rhs->is_user_data_required()); }
        size_t n_in() const{ return std::max(m_mat->n_in(), m_rhs ? m_rhs->n_in() : size_t(0)); }
        size_t n_out() const{ return 1 + (m_rhs ? m_rhs->n_out() : 0); }
        MatSparsityView<Int> out_sparsity(Int res_id) const{ return (res_id == 0 || !m_rhs) ? m_mat->out_sparsity(res_id) : m_rhs->out_sparsity(res_id - 1); }
        Int out_nnz  (Int res_id) const{ return (res_id == 0 || !m_rhs) ? m_mat->out_nnz(res_id) : m_rhs->out_nnz(res_id - 1);  }
        Int out_size1(Int res_id) const{ return (res_id == 0 || !m_rhs) ? m_mat->out_size1(res_id) : m_rhs->out_size1(res_id - 1); }
        Int out_size2(Int res_id) const{ return (res_id == 0 || !m_rhs) ? m_mat->out_size2(res_id) : m_rhs->out_size2(res_id - 1); }
        void out_csc(Int res_id, Int* colind, Int* row) const{ return (res_id == 0 || !m_rhs) ? m_mat->out_csc(res_id, colind, row) : m_rhs->out_csc(res_id - 1, colind, row); }
        MatSparsityView<Int> in_sparsity(Int arg_id) const{ return m_mat->in_sparsity(arg_id); }
        Int in_nnz  (Int arg_id) const{ return m_mat->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const{ return m_mat->in_size1(arg_id); }
        Int in_size2(Int arg_id) const{ return m_mat->in_size2(arg_id); }
        void in_csc(Int arg_id, Int* colind, Int* row) const{ return m_mat->in_csc(arg_id, colind, row); }
        std::ostream& print_signature(std::ostream& out = std::cout) const;
        bool isValid() const { return m_mat && m_mat->isValid(); }
        operator bool() const { return isValid(); }
        SystemFromMatRhsFWrap() = default;
        SystemFromMatRhsFWrap(MatFuncT* system_func): m_mat{system_func} {}
        SystemFromMatRhsFWrap(MatFuncT* mat_f, MatFuncT* rhs_f): m_mat{mat_f}, m_rhs{rhs_f} {} 
    };
}

struct AssmOpts{
    void*  user_data = nullptr;             ///< is user supplied data to be postponed to problem handler, may be NULL.
    double drop_val = 1e-100;               ///< set to zero elements of local matrix/rhs if they less than this value

    bool   is_mtx_sorted = false;           ///< is elements in matrix rows sorted by it's column indexes
    bool   is_mtx_include_template = false; ///< is matrix contains all elements which will be modified
    
    /// should we prefer sorted state of rows and use algorithms for ordered ranges to add/insert elements 
    /// instead unordered rows and line search algorihms?
    /// @note use_ordered_insert = true prefered when matrix have a lot of elements in row 
    bool   use_ordered_insert = false;      

    AssmOpts() = default;
    AssmOpts(void* user_data): user_data{user_data} {}
    AssmOpts(void* user_data, double drop_val): user_data{user_data}, drop_val{drop_val} {}

    AssmOpts& SetUserData(void* _user_data){ return user_data = _user_data, *this; }
    AssmOpts& SetDropVal(double drp_val){ return drop_val = drp_val, *this; }
    AssmOpts& SetIsMtxSorted(bool is_sorted) { return is_mtx_sorted = is_sorted, *this; }
    AssmOpts& SetIsMtxIncludeTemplate(bool is_templated) { return is_mtx_include_template = is_templated, *this; }
    AssmOpts& SetUseOrderedInsert(bool use) { return use_ordered_insert = use, *this; }
};
template<typename Traits = DefaultAssemblerTraits>
class AssemblerT{
#ifndef NO_ASSEMBLER_TIMERS
    struct TimerData{
        TimerWrap m_timer, m_timer_ttl;
        double  m_time_init_assemble_dat = 0,
                m_time_fill_map_template = 0,
                m_time_init_val_setter = 0,
                m_time_init_user_handler = 0,
                m_time_comp_func = 0,
                m_time_proc_user_handler = 0,
                m_time_set_elemental_res = 0,
                m_timer_total = 0;
        void reset(); 
        ElementalAssembler::TimeMessures getTimeMessures();      
    };
#endif //NO_ASSEMBLER_TIMERS
    using Real = typename Traits::Real;
    using Int = typename Traits::Int;

    struct WorkMem{
        std::vector<Real> m_A, m_F, m_w;
        std::vector<Int> m_iw;
        std::vector<const Real*> m_args;
        std::vector<Real*> m_res;
        std::vector<long> m_indexesR, m_indexesC;
        std::vector<bool> m_Ab; //m_Ab[i] is true if m_A[i] is nonzero according template
        void Clear(){
            m_A.clear(), m_F.clear(), m_w.clear();
            m_iw.clear();
            m_args.clear();
            m_res.clear();
            m_indexesR.clear(), m_indexesC.clear();
            m_Ab.clear();
        }
#ifndef NO_ASSEMBLER_TIMERS
        TimerData m_timers;
#endif
        INMOST::ElementArray<INMOST::Node> nodes; 
        INMOST::ElementArray<INMOST::Edge> edges; 
        INMOST::ElementArray<INMOST::Face> faces;
        int status = 0;
        DynMem<Real, Int> pool;
    };
    struct CommonFuncData{
        std::vector<Int> colindA, colindF, rowA, rowF;
        void Clear() { colindA.clear(); colindF.clear(); rowA.clear(); rowF.clear();}
    };
    struct OrderTempl{
        DofT::uint var_id;   ///< contiguous number of variable 
        DofT::uint dim_id;   ///< number of dimensional component in the variable
        DofT::uchar etype;   ///< type of geometrical element
        DofT::uchar nelem;   ///< number of geometrical element on the tetrahedron (from 0 to 3 for NODE or FACE, from 0 to 6 for EDGE, always 0 for CELL) 
        DofT::uint lde_id;  ///< element-local d.o.f. index for specific dim of the var (if there is no other variables on the element and dim == 0 it's index of d.o.f on the element)

        DofT::uchar stype;    ///< contiguous number of symmetry group, e.g. for cell 0 - s1, 1 - s4, 2 - s6, 3 - s12, 4 - s24
        DofT::uchar lsid;     ///< number of d.o.f. in symmetry group 
    };

    std::vector<WorkMem> m_wm;
    CommonFuncData m_fd;
    std::vector<ElementalAssembler::VarsHelper> m_helpers;
    std::vector<OrderTempl> orderC;
    std::vector<OrderTempl> orderR;
public:
    struct AssembleMode{
        bool reorder_nodes = true;  ///Reorder the nodes if they form a tetrahedron of negative volume?
        bool prepare_edges = true;  ///Prepare edges of the cell?
        bool prepare_faces = true;  ///Prepare faces of the cell?
        int  num_threads = -1;      ///Number of used concurrency threads if they available, if num_threads < 0 then will use maximal available number of threads
    };

    ///\warning Next three functions must have same input parameters
    typename Traits::MatFuncT mat_func;     ///< evaluates elemental matrix only
    typename Traits::MatFuncT rhs_func;     ///< evaluates elemental right hand-side only
    typename Traits::MatFuncT mat_rhs_func; ///< evaluates elemental matrix and rhs together

    AssembleMode m_assm_traits;
    FemExprDescr m_info;
    typename Traits::ProbHandler m_prob_handler;
    typename Traits::InitValueSetter m_init_value_setter;
    typename Traits::GlobEnumMap m_enum;
    INMOST::Mesh* m_mesh = nullptr;

    AssemblerT() = default;
    explicit AssemblerT(INMOST::Mesh *m): m_mesh(m) {}
    explicit AssemblerT(INMOST::Mesh& m): AssemblerT(&m) {}
    // ///Allocate internal memory for the problem ( must be called after all Set* calls )
    void PrepareProblem();

    AssemblerT& SetMesh(INMOST::Mesh& m) { m_mesh = &m; return *this; }
    AssemblerT& SetEnumerator(typename Traits::GlobEnumMap&& enumeration) { m_enum = std::move(enumeration); return *this; }
    AssemblerT& SetDataGatherer(typename Traits::ProbHandler&& problem_handler) { m_prob_handler = std::move(problem_handler); return *this; }
    AssemblerT& SetProbDescr(FemExprDescr info) { m_info = std::move(info); return *this; }
    AssemblerT& SetMatFunc (typename Traits::MatFuncT&& func){ mat_func = std::move(func); return *this; }
    AssemblerT& SetRHSFunc (typename Traits::MatFuncT&& func){ rhs_func = std::move(func); return *this; }
    AssemblerT& SetMatRHSFunc(typename Traits::MatFuncT&& func){ mat_rhs_func = std::move(func); return *this; }
    AssemblerT& SetInitValueSetter(typename Traits::InitValueSetter&& init_value_setter) { m_init_value_setter = std::move(init_value_setter); return *this; }

    template<typename T, typename V>
    using TCS = typename std::enable_if<std::is_copy_constructible<T>::value && std::is_same<T, V>::value, AssemblerT<Traits>&>::type;
    template<typename T, typename V>
    using TCO = typename std::enable_if<std::is_constructible<V, T>::value && !std::is_same<T, V>::value, AssemblerT<Traits>&>::type;
    template<typename T, typename V>
    using TCX = typename std::enable_if<std::is_constructible<V, T>::value, AssemblerT<Traits>&>::type;
    template<typename T> 
    TCS<T, typename Traits::GlobEnumMap> SetEnumerator(const T& enumeration) { m_enum = enumeration; return *this; }
    template<typename T> 
    TCO<T, typename Traits::GlobEnumMap> SetEnumerator(T enumeration) { m_enum = typename Traits::GlobEnumMap(std::move(enumeration)); return *this; }

    template<typename T> 
    TCS<T, typename Traits::ProbHandler> SetDataGatherer(const T& problem_handler) { m_prob_handler = problem_handler; return *this; }
    template<typename T> 
    TCO<T, typename Traits::ProbHandler> SetDataGatherer(T problem_handler) { m_prob_handler = typename Traits::ProbHandler(std::move(problem_handler)); return *this; }

    template<typename T> 
    TCS<T, typename Traits::MatFuncT> SetMatFunc (const T& func){ mat_func = func; return *this; }
    template<typename T>
    TCS<T, typename Traits::MatFuncT> SetRHSFunc (const T& func){ rhs_func = func; return *this; }
    template<typename T>
    TCS<T, typename Traits::MatFuncT> SetMatRHSFunc(const T& func){ mat_rhs_func = func; return *this; }
    template<typename T> 
    TCO<T, typename Traits::MatFuncT> SetMatFunc (T func){ mat_func = typename Traits::MatFuncT(std::move(func)); return *this; }
    template<typename T>
    TCO<T, typename Traits::MatFuncT> SetRHSFunc (T func){ rhs_func = typename Traits::MatFuncT(std::move(func)); return *this; }
    template<typename T>
    TCO<T, typename Traits::MatFuncT> SetMatRHSFunc(T func){ mat_rhs_func = typename Traits::MatFuncT(std::move(func)); return *this; }

    template<typename T>
    TCS<T, typename Traits::InitValueSetter> SetInitValueSetter(const T& init_value_setter) { return m_init_value_setter = init_value_setter, *this; }
    template<typename T>
    TCO<T, typename Traits::InitValueSetter> SetInitValueSetter(T init_value_setter) { return m_init_value_setter = typename Traits::InitValueSetter(std::move(init_value_setter)), *this; }

    

    void SaveSolution(const INMOST::Sparse::Vector& from, INMOST::Tag to) const { m_enum.CopyByEnumeration(from, to); }
    void SaveSolution(const INMOST::Sparse::Vector& from, std::vector<INMOST::Tag> to) const { m_enum.CopyByEnumeration(from, to); }
    void SaveSolution(INMOST::Tag from, INMOST::Sparse::Vector& to) const { m_enum.CopyByEnumeration(from, to); }
    void SaveSolution(INMOST::Tag from, INMOST::Tag to) const { m_enum.CopyByEnumeration(from, to); }
    void SaveSolution(INMOST::Tag from, std::vector<INMOST::Tag> to) const { m_enum.CopyByEnumeration(from, to); }
    void SaveSolution(const std::vector<INMOST::Tag>& from, INMOST::Tag to) const { m_enum.CopyByEnumeration(from, to); }
    void SaveSolution(const std::vector<INMOST::Tag>& from, INMOST::Sparse::Vector& to) const { m_enum.CopyByEnumeration(from, to); }
    /// Save variable with number iVar from vector vars to var_tag
    void SaveVar(const INMOST::Sparse::Vector& vars, int iVar, INMOST::Tag var_tag) const { m_enum.CopyVarByEnumeration(vars, var_tag, iVar); }
    /// Save variable with number iVar from tag of physical variables vars to var_tag
    void SaveVar(INMOST::Tag vars, int iVar, INMOST::Tag var_tag) const { m_enum.CopyVarByEnumeration(vars, var_tag, iVar); }
    /// Save variable with number iVar from var_tag to tag of physical variables vars
    void SaveVar(int iVar, const INMOST::Tag var_tag, INMOST::Tag vars) const { m_enum.CopyVarByEnumeration(var_tag, iVar, vars); }
    /// Copy variable with number iVar from var_tag_from to var_tag_to
    void CopyVar(int iVar, const INMOST::Tag var_tag_from, INMOST::Tag var_tag_to) const { m_enum.CopyVarByEnumeration(iVar, var_tag_from, var_tag_to); }

    int Assemble(INMOST::Sparse::Matrix &matrix, INMOST::Sparse::Vector &rhs, const AssmOpts& opts = AssmOpts());
    int AssembleMatrix(INMOST::Sparse::Matrix &matrix, const AssmOpts& opts = AssmOpts());
    int AssembleRHS(INMOST::Sparse::Vector &rhs, const AssmOpts& opts = AssmOpts());
    int AssembleTemplate(INMOST::Sparse::Matrix &matrix);

    /// Begin index for sparse matrix and vector
    long getBegInd() const { return m_enum.getBegInd();}
    /// End index for sparse matrix and vector
    long getEndInd() const { return m_enum.getEndInd();}

    void Clear();
    const ElementalAssembler::VarsHelper& GetVarHelper() const { return m_helpers[0]; }
    const typename Traits::GlobEnumMap& GetEnumerator() const { return m_enum; }
    typename Traits::GlobEnumMap& GetEnumerator() { return m_enum; }

    /// Create initial_value_setter that sets all initial values of all variables
    static InitValueSetFromTag makeInitValueSetter(double val = 0.0) { return InitValueSetFromTag().set(val); }
    static InitValueSetFromTag makeInitValueSetter(INMOST::Tag* tag_x) { return InitValueSetFromTag().set(tag_x); }
    static InitValueSetFromTag makeInitValueSetter(INMOST::Tag tag_x) { return InitValueSetFromTag().set(tag_x); }
    static InitValueSetTagVector makeInitValueSetter(std::vector<INMOST::Tag> tag_vec) { return InitValueSetTagVector().set(std::move(tag_vec)); }
    template<typename T = typename Traits::InitValueSetter>
    TCX<decltype(makeInitValueSetter(std::declval<INMOST::Tag>())), T> pullInitValFrom(INMOST::Tag  tag_x) { m_init_value_setter = T(makeInitValueSetter(tag_x)); return *this; }
    template<typename T = typename Traits::InitValueSetter>
    TCX<decltype(makeInitValueSetter(std::declval<INMOST::Tag*>())), T> pullInitValFrom(INMOST::Tag* tag_x) { m_init_value_setter = T(makeInitValueSetter(tag_x)); return *this; }
    template<typename T = typename Traits::InitValueSetter>
    TCX<decltype(makeInitValueSetter(std::declval<std::vector<INMOST::Tag>>())), T> pullInitValFrom(std::vector<INMOST::Tag> tag_vec) { m_init_value_setter = T(makeInitValueSetter(std::move(tag_vec) )); return *this; }

    template<class RandomIt>
    void GatherDataOnElement(const INMOST::Tag* var_tag, const std::size_t ntags, const INMOST::Cell& cell, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp) const;
    template<class RandomIt>
    void GatherDataOnElement(INMOST::Tag from, const INMOST::Cell& cell, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp) const;
    template<class RandomIt>
    void GatherDataOnElement(const INMOST::Tag& from, const INMOST::Cell& cell, RandomIt out, std::initializer_list<int> components = {}) const { GatherDataOnElement(from, cell, out, components.begin(), components.size()); }
    template<class RandomIt>
    void GatherDataOnElement(const std::vector<INMOST::Tag>& var_tags, const INMOST::Cell& cell, RandomIt out, std::initializer_list<int> components = {}) const { GatherDataOnElement(var_tags.data(), var_tags.size(), cell, out, components.begin(), components.size()); }

    double GetTimeInitAssembleData() const;
    double GetTimeFillMapTemplate() const;
    double GetTimeInitValSet() const;
    double GetTimeInitUserHandler() const;
    double GetTimeEvalLocFunc() const;
    double GetTimePostProcUserHandler() const;
    double GetTimeFillGlobalStructs() const;
    double GetTimeTotal() const;
private:
    template <typename T, bool IsAssignable = std::is_constructible<T, decltype(makeInitValueSetter(std::declval<double>()))>::value>
    struct TryZeroInit{ void operator()(typename Traits::InitValueSetter& init_value_setter) const { assert("InitValueSetter is not initialized"); } };
    template<typename T> struct TryZeroInit<T, true>{ void operator()(typename Traits::InitValueSetter& init_value_setter) const { init_value_setter = typename Traits::InitValueSetter(makeInitValueSetter(0.0)); } };

    template<typename MatFuncT1>
    void extend_memory_for_fem_func(MatFuncT1& func);
    void resize_work_memory(int size);
    bool fill_assemble_templates(const INMOST::ElementArray<INMOST::Node>& nodes, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::Cell& cell, std::vector<long>& indexesC, std::vector<long>& indexesR, const unsigned char* canonical_node_indexes);
    bool fill_assemble_templates(const INMOST::ElementArray<INMOST::Node>& nodes, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::Cell& cell, std::vector<long>& indexesC, std::vector<long>& indexesR, const unsigned char* canonical_node_indexes, bool is_same_template);
    int _return_assembled_status(int nthreads);
    internals::SystemFromMatRhsFWrap<typename Traits::MatFuncT> generate_mat_rhs_func();
    internals::MatFromSystemFWrap<typename Traits::MatFuncT> generate_mat_func();
    internals::RhsFromSystemFWrap<typename Traits::MatFuncT> generate_rhs_func();

#ifndef NO_ASSEMBLER_TIMERS
    TimerData m_timers;
#endif
    void reset_timers();
};

}

#ifndef WITH_OPENMP  
using Assembler = typename Ani::AssemblerT<>;
#else
using Assembler = typename Ani::AssemblerT< Ani::DefaultParallelAssemblerTraits<> >;
#endif

using AssemblerP = typename Ani::AssemblerT< Ani::DefaultParallelAssemblerTraits<> >;

#include "assembler.inl"

#endif //CARNUM_ASSEMBLER_H