//
// Created by Liogky Alexey on 09.08.2023.
//

#ifndef CARNUM_ASSEMBLER_INL
#define CARNUM_ASSEMBLER_INL

#include "assembler.h"

namespace Ani{

template <typename Traits>
template<typename MatFuncT1>
void AssemblerT<Traits>::extend_memory_for_fem_func(MatFuncT1& func) {
    if (!func) return;
    size_t sz_iw, sz_w, sz_args, sz_res;
    func.working_sizes(sz_args, sz_res, sz_w, sz_iw);
    for (auto& m_w: m_wm){
        if (m_w.m_w.size() < sz_w) m_w.m_w.resize(sz_w);
        if (m_w.m_iw.size() < sz_iw) m_w.m_iw.resize(sz_iw);
        if (m_w.m_args.size() < sz_args) m_w.m_args.resize(sz_args);
        if (m_w.m_res.size() < sz_res) m_w.m_res.resize(sz_res);
    }
}

template <typename Traits>
void AssemblerT<Traits>::resize_work_memory(int size){
    if (size <= 1) m_wm.resize(std::min(std::size_t(1), m_wm.size()));
    auto old_sz = m_wm.size();
    m_wm.resize(size);
    if (static_cast<std::size_t>(size) <= old_sz) return;
    for (int i = old_sz; i < size; ++i){
        m_wm[i].m_w.resize(m_wm[0].m_w.size());
        m_wm[i].m_iw.resize(m_wm[0].m_iw.size());
        m_wm[i].m_args.resize(m_wm[0].m_args.size());
        m_wm[i].m_res.resize(m_wm[0].m_res.size());
        m_wm[i].m_A.resize(m_wm[0].m_A.size());
        m_wm[i].m_F.resize(m_wm[0].m_F.size());
        m_wm[i].m_indexesR.resize(m_wm[0].m_indexesR.size());
        m_wm[i].m_indexesC.resize(m_wm[0].m_indexesC.size());
        m_wm[i].m_Ab.resize(m_wm[0].m_Ab.size(), true);
        m_wm[i].nodes = INMOST::ElementArray<INMOST::Node>(m_mesh, 4);
        m_wm[i].edges = INMOST::ElementArray<INMOST::Edge>(m_mesh, 6);
        m_wm[i].faces = INMOST::ElementArray<INMOST::Face>(m_mesh, 4);
    }
}

namespace internals{
    static inline long assemble_index_encode(char sign, long index){ return sign*(index+1); }
    struct AssembleDecode{
        long id;
        char sign;
    };
    static const long CODE_UNDEF = 0;
    static inline AssembleDecode assemble_index_decode(long code){ return AssembleDecode{abs(code)-1, code < 0 ? char(-1) : char(1)}; }

    static inline void print_assemble_matrix_indeces_incompatible(std::ostream& out, const INMOST::Mesh* m,
                                                              std::vector<long> &indexesC, const std::array<long, 4>& NumElem, const std::array<long, 4>& MinElem, const std::array<long, 4>& MaxElem,
                                                              long MatrSize, long BegInd, long EndInd, int nRows, int wrong_index) {
        out<<"Proc rank "<<m->GetProcessorRank()<<std::endl;
        out<<"wrong index R "<<assemble_index_decode(indexesC[wrong_index]).id<<" "<<0<<" "<<MatrSize<<std::endl;
        out<<"Num elems"<<std::endl;
        for(int i1=0;i1<4;i1++)
            out<<NumElem[i1]<<" ";
        out<<std::endl;
        out<<"Beg elems id"<<std::endl;
        for(int i1=0;i1<4;i1++)
            out<<MinElem[i1]<<" ";
        out<<"End elems id"<<std::endl;
        for(int i1=0;i1<4;i1++)
            out<<MaxElem[i1]<<" ";
        out<<std::endl;
        out<<"Begin index "<< BegInd<<" End index "<<EndInd<< std::endl;
        out<<"indexes"<<std::endl;
        for(int i1=0;i1<nRows;i1++)
            out<<assemble_index_decode(indexesC[i1]).id<<" ";
        out<<std::endl;
    }
    template<typename Real>
    static inline void print_assemble_rhs_nan(std::ostream& out, std::vector<Real>& F, INMOST::Mesh::iteratorCell it, int nRows, int nan_ind) {
        out<<"not a number in rhs"<<"\n";
        out<<"\tF["<<nan_ind <<"] = " << F[nan_ind] <<"\n";
        for(int i1=0;i1<nRows;i1++) { out << "\t\t" << F[i1] << "\n"; }
        out<<"\tnum tet " << it->DataLocalID() << ":\n";
        auto nodes = it->getNodes();
        reorderNodesOnTetrahedron(nodes);
        for (int nP = 0; nP < 4; ++nP) {
            auto p = nodes[nP];
            out << "\t\tP" << nP << ": (" << p.Coords()[0] <<", " << p.Coords()[1] << ", " << p.Coords()[2] << ")\n";
        }
        out << std::endl;
    }
    template<typename Real>
    static inline void print_assemble_matrix_nan(std::ostream& out, std::vector<Real>& A, INMOST::Mesh::iteratorCell it, int nRows, int nan_i, int nan_j) {
        out<<"not a number in matrix"<<"\n";
        out<<"num tet " << it->DataLocalID()<< " A["<<nan_i<<"]["<<nan_j<<"] = " << A[nan_j*nRows +nan_i]<<"\n";
        for(int i1=0;i1<nRows;i1++) {
            for (int j1 = 0; j1 < nRows; j1++) {
                out << A[j1 * nRows + i1] << " ";
            }
            out<<"\n";
        }

        auto nodes = it->getNodes();
        reorderNodesOnTetrahedron(nodes);
        for (int nP = 0; nP < 4; ++nP) {
            auto p = nodes[nP];
            out << "P" << nP << ": ("
                << p.Coords()[0] <<", " << p.Coords()[1] << ", " << p.Coords()[2] << ")\n";

        }
        out << std::endl;
    }
    static inline void set_elements_on_matrix_diagonal(INMOST::Sparse::Matrix& matrix, const AssmOpts& opts){
        if (!opts.is_mtx_include_template){
            auto beg = matrix.GetFirstIndex(), last = matrix.GetLastIndex();
            if (!opts.is_mtx_sorted){
                for(auto i = beg; i < last; i++) if (matrix[i].get_safe(i) == 0.0)
                    matrix[i][i] = 0.0;
            } else {
                for(auto i = beg; i < last; i++){
                    auto it = std::lower_bound(matrix[i].Begin(), matrix[i].End(), i, [](const auto& a, INMOST_DATA_ENUM_TYPE b){ return a.first < b; });
                    if (it == matrix[i].End()) matrix[i].Push(i, 0.0);
                    else if (it->first != i) {
                        auto k = std::distance(matrix[i].Begin(), it);
                        matrix[i].Resize(matrix[i].Size() + 1);
                        for (int l = 0; l < matrix[i].Size() - k - 1; ++l){
                            auto from = matrix[i].Size() - 1 - l;
                            matrix[i].GetIndex(from) = matrix[i].GetIndex(from-1); matrix[i].GetValue(from) = matrix[i].GetValue(from-1);
                        }
                        matrix[i].GetIndex(k) = i; matrix[i].GetValue(k) = 0;
                    } 
                }
            }        
        }
    }
}

template <typename Traits>
bool AssemblerT<Traits>::fill_assemble_templates(
    const INMOST::ElementArray<INMOST::Node>& nodes, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::Cell& cell, 
    std::vector<long>& indexesC, std::vector<long>& indexesR, const unsigned char* canonical_node_indexes, bool is_same_template){
    INMOST::HandleType ch = cell.GetHandle();
    std::array<const INMOST::HandleType*, 4> h{nodes.data(), edges.data(), faces.data(), &ch};
    indexesC.resize(orderC.size()); indexesR.resize(orderR.size());
    std::array<char, 6+4> esigns;
    std::fill(esigns.begin(), esigns.end(), 0);
    auto set_indexes = [&esigns, &h, m = cell.GetMeshLink(), canonical_node_indexes](
        const std::vector<OrderTempl>& order, std::vector<long>& indexes, auto compute_order_from_nat_index)->bool{
        bool has_active = false;
        for (std::size_t i = 0; i < order.size(); ++i){
            OrderTempl tl = order[i];
            auto edim = DofT::GeomTypeDim(tl.etype);
            char sign = 1;
            if (tl.etype & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT)){
                if (esigns[tl.nelem + (edim > 1) ? 6 : 0] == 0)
                    esigns[tl.nelem + (edim > 1) ? 6 : 0] = isPositivePermutationOrient(tl.etype, tl.nelem, canonical_node_indexes) ? 1 : -1;
                sign = esigns[tl.nelem + (edim > 1) ? 6 : 0];
            }
            auto reordered_lsid = DofT::DofSymmetries::index_on_reorderd_elem(tl.etype, tl.nelem, tl.stype, tl.lsid, canonical_node_indexes); 
            auto elem = INMOST::Element(m, h[edim][tl.nelem]);
            bool is_active = (elem.GetStatus() != INMOST::Element::Ghost);
            IGlobEnumeration::NaturalIndex ind(elem, tl.etype, tl.var_id, tl.dim_id, tl.lde_id);
            indexes[i - tl.lsid + reordered_lsid] = internals::assemble_index_encode(sign, compute_order_from_nat_index(ind).id);
            if (is_active) has_active = true;
        }
        return has_active;
    };
    bool has_active = set_indexes(orderC, indexesC, [&m_enum = this->m_enum](const IGlobEnumeration::NaturalIndex& ind){ return m_enum.OrderC(ind); });
    if (!is_same_template){
        auto has_active1 = set_indexes(orderR, indexesR, [&m_enum = this->m_enum](const IGlobEnumeration::NaturalIndex& ind){ return m_enum.OrderR(ind); });
        has_active = has_active || has_active1;
    } else {
        std::copy(indexesC.begin(), indexesC.end(), indexesR.begin());
        for (std::size_t i = 0; i < orderC.size(); ++i){
            OrderTempl tl = orderC[i];
            auto edim = DofT::GeomTypeDim(tl.etype);
            auto elem = INMOST::Element(cell.GetMeshLink(), h[edim][tl.nelem]);
            if (elem.GetStatus() == INMOST::Element::Ghost)
                indexesR[i] = internals::CODE_UNDEF;
        }
    }
    return has_active;
}

template <typename Traits>
bool AssemblerT<Traits>::fill_assemble_templates(
    const INMOST::ElementArray<INMOST::Node>& nodes, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::Cell& cell, 
    std::vector<long>& indexesC, std::vector<long>& indexesR, const unsigned char* canonical_node_indexes){
    return fill_assemble_templates(nodes, edges, faces, cell, indexesC, indexesR, canonical_node_indexes, m_helpers[0].same_template);
}

template <typename Traits>
void AssemblerT<Traits>::PrepareProblem(){
    if (!m_mesh) throw std::runtime_error("Mesh was not specified");
    if (m_info.TrialFuncs().NumVars() == 0) throw std::runtime_error("Description of fem expression is empty, try SetProbDescr(...)");
    if (m_enum.getMeshLink() != m_mesh) {
        m_enum.setMesh(m_mesh);
        m_enum.setVars(m_info.TrialFuncs());
        m_enum.setup();
    }
    if (!m_init_value_setter) TryZeroInit<typename Traits::InitValueSetter>{}(m_init_value_setter);

    int nRows = m_info.TrialFuncs().NumDofOnTet(); //4*nd[0] + 6*nd[1] + 4*nd[2] + nd[3];
    m_wm.resize(1);
    auto& m_w = m_wm[0];
    m_w.m_A.resize(nRows*nRows), m_w.m_F.resize(nRows);
    m_w.m_Ab.resize(nRows * nRows, true);
    m_w.m_indexesR.resize(nRows), m_w.m_indexesC.resize(nRows);
    m_w.nodes = INMOST::ElementArray<INMOST::Node>(m_mesh, 4);
    m_w.edges = INMOST::ElementArray<INMOST::Edge>(m_mesh, 6);
    m_w.faces = INMOST::ElementArray<INMOST::Face>(m_mesh, 4);

    m_helpers.resize(1);
    m_helpers[0].descr = &m_info;
    m_helpers[0].initValues.resize(nRows);
    m_helpers[0].same_template = (m_info.TrialFuncs() == m_info.TestFuncs());

    extend_memory_for_fem_func(mat_func);
    extend_memory_for_fem_func(rhs_func);
    extend_memory_for_fem_func(mat_rhs_func);
    if (mat_func && !mat_rhs_func){
        if (mat_func.out_nnz(0) != mat_func.out_size1(0) * mat_func.out_size2(0)) {
            m_fd.colindA.resize(mat_func.out_size2(0) + 1), m_fd.rowA.resize(mat_func.out_nnz(0));
            mat_func.out_csc(0, m_fd.colindA.data(), m_fd.rowA.data());
        }
    }
    if (rhs_func && !mat_rhs_func){
        if (rhs_func.out_nnz(0) != rhs_func.out_size1(0) * rhs_func.out_size2(0)) {
            m_fd.colindF.resize(rhs_func.out_size2(0) + 1), m_fd.rowF.resize(rhs_func.out_nnz(0));
            rhs_func.out_csc(0, m_fd.colindF.data(), m_fd.rowF.data());
        }
    }
    if (mat_rhs_func){
        if (mat_rhs_func.out_nnz(0) != mat_rhs_func.out_size1(0) * mat_rhs_func.out_size2(0)) {
            m_fd.colindA.resize(mat_rhs_func.out_size2(0) + 1), m_fd.rowA.resize(mat_rhs_func.out_nnz(0));
            mat_rhs_func.out_csc(0, m_fd.colindA.data(), m_fd.rowA.data());
            m_fd.colindF.resize(mat_rhs_func.out_size2(1) + 1), m_fd.rowF.resize(mat_rhs_func.out_nnz(1));
            mat_rhs_func.out_csc(1, m_fd.colindF.data(), m_fd.rowF.data());
        }
    }
    if (!rhs_func && !mat_func && !mat_rhs_func) throw std::runtime_error("Fem expression evaluator is not set");
    orderR.resize(0); orderC.resize(0);
    orderR.reserve(nRows); orderC.reserve(nRows);
    auto prepareOrder = [](const FemVarContainer& funcs, std::vector<OrderTempl>& order){
        order.resize(0); order.reserve(funcs.NumDofOnTet());
        for (auto vit = funcs.Begin(); vit != funcs.End(); ++vit){
            auto vi = *vit;
            auto vt = vi.dofmap.ActualType();
            unsigned ndim = 1;
            if (vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorType) || vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorTemplateType))
                ndim = vi.dofmap.NestedDim();
            auto nds = vi.dofmap.NumDofsOnTet();
            for (auto& n: nds) n /= ndim;
            OrderTempl tl;
            tl.var_id = vi.iVar;
            for (auto it = vi.dofmap.base()->begin(), ite = vi.dofmap.base()->end(); it != ite; ++it){
                DofT::LocalOrder v = *it;
                auto type_num = DofT::GeomTypeToNum(v.etype);
                tl.dim_id = v.leid / nds[type_num];
                tl.etype = v.etype;
                tl.nelem = v.nelem;
                tl.lde_id = v.leid % nds[type_num];
                tl.stype = v.stype;
                tl.lsid = v.lsid;
                order.push_back(tl);
            }
        }
    };
    prepareOrder(m_info.TrialFuncs(), orderC);
    if (m_helpers[0].same_template)
        orderR = orderC;
    else
        prepareOrder(m_info.TestFuncs(), orderR);
}

template <typename Traits>
int AssemblerT<Traits>::_return_assembled_status(int nthreads){
    int status = 0;
    for (int i = 0; i < nthreads; ++i)
        status = std::min(status, m_wm[i].status);
#ifndef NO_ASSEMBLER_TIMERS        
    for (int i = 0; i < nthreads; ++i){
        m_timers.m_time_init_assemble_dat += m_wm[i].m_timers.m_time_init_assemble_dat; 
        m_timers.m_time_fill_map_template += m_wm[i].m_timers.m_time_fill_map_template; 
        m_timers.m_time_init_val_setter += m_wm[i].m_timers.m_time_init_val_setter; 
        m_timers.m_time_proc_user_handler += m_wm[i].m_timers.m_time_proc_user_handler; 
        m_timers.m_time_set_elemental_res += m_wm[i].m_timers.m_time_set_elemental_res; 
        m_timers.m_time_init_user_handler  += m_wm[i].m_timers.m_time_init_user_handler ; 
        m_timers.m_time_comp_func += m_wm[i].m_timers.m_time_comp_func; 
    }
    m_timers.m_timer_total += m_timers.m_timer_ttl.elapsed();
#endif
    return status;
}

#ifndef NO_ASSEMBLER_TIMERS
    #define TIMER_SCOPE(X) { X }
    #define TIMER_NOSCOPE(X)  X,  
#else
    #define TIMER_SCOPE(X)
    #define TIMER_NOSCOPE(X)
#endif 
///This function assembles matrix and rhs of FE problem previously setted by PrepareProblem(...) function
/// @param[in,out] matrix is the global matrix to which the assembled matrix will be added, i.e. matrix <- matrix + assembled_matrix
/// @param[in,out] rhs is the global right-hand side to which the assembled RHS will be added, i.e. rhs <- rhs + assembled_rhs
/// @param[in] opts is user specified options for assembler and user supplied data to be postponed to problem handler
/// @note if opts.use_ordered_insert == true then assembled matrix will have sorted rows
/// @return  0 if assembling successful else some error_code (number < 0)
///             ATTENTION: In case of unsuccessful completion of this function matrix and rhs has wrong elements!!!:
///         -1 if some local matrix or rhs had NaN.
///         -2 if some element is not tetrahedral or has broken component connectivity cell - faces - edges - nodes
template <typename Traits>
int AssemblerT<Traits>::Assemble(INMOST::Sparse::Matrix &matrix, INMOST::Sparse::Vector &rhs, const AssmOpts& opts){
    reset_timers();
    if (!mat_rhs_func && (!mat_func || !rhs_func))
        throw std::runtime_error("System local evaluator is not specified");
    auto func = generate_mat_rhs_func();
    int nRows = m_info.TrialFuncs().NumDofOnTet();
    int nthreads = ThreadPar::get_num_threads<Traits::MatFuncT::parallel_type>(m_assm_traits.num_threads);
    resize_work_memory(nthreads);
    m_helpers.resize(nthreads, m_helpers[0]);
    rhs.SetInterval(getBegInd(), getEndInd());
    matrix.SetInterval(getBegInd(),getEndInd());
    internals::set_elements_on_matrix_diagonal(matrix, opts);
    if (opts.use_ordered_insert && !opts.is_mtx_sorted)
        ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, 
            [&A = matrix](INMOST_DATA_ENUM_TYPE lid, int nthread){ (void) nthread; std::sort(A[lid].Begin(), A[lid].End()); }, 
            static_cast<INMOST_DATA_ENUM_TYPE>(getBegInd()), 
            static_cast<INMOST_DATA_ENUM_TYPE>(getEndInd())
        );    
    std::vector<int> row_index_buffers;
    std::vector<INMOST::Sparse::Row> swap_rows;
    if (opts.use_ordered_insert){
        row_index_buffers.resize(nRows*nthreads);
        swap_rows.resize(nthreads);
    }

    INMOST::Sparse::LockService L;
    if (nthreads > 1) L.SetInterval(getBegInd(), getEndInd());    
    auto func_internal_mem_ids = func.setup_and_alloc_memory_range(nthreads);
    const bool reord_nds = m_assm_traits.reorder_nodes;
    const bool prep_ef = (m_assm_traits.prepare_edges || m_assm_traits.prepare_faces) || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE|DofT::FACE));
    const bool comp_node_perm = !m_enum.areVarsTriviallySymmetric() || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
    for (int i = 0; i < nthreads; ++i) m_wm[i].status = 0;
    TIMER_SCOPE( for (int i = 0; i < nthreads; ++i) m_wm[i].m_timers.reset(); )
    TIMER_SCOPE( m_timers.m_time_init_assemble_dat += m_timers.m_timer.elapsed_and_reset(); )
    
    auto cycle_body_func = [&row_index_buffers, &swap_rows, use_ordered_insert = opts.use_ordered_insert, is_mtx_include_template = opts.is_mtx_include_template,
                            &func, nRows, reord_nds, prep_ef, comp_node_perm, &rhs, &matrix,  &L, nthreads, drp_val = opts.drop_val, this](INMOST::Storage::integer lid, int nthread, void* user_data) mutable {
        auto& m_w = m_wm[nthread];
        if (m_w.status < 0) return;
        INMOST::Cell cell = m_mesh->CellByLocalID(lid);
        if (!cell.isValid() || cell.Hidden()) return;
        collectConnectivityInfo(cell, m_w.nodes, m_w.edges, m_w.faces, reord_nds, prep_ef);
        TIMER_SCOPE( m_w.m_timers.m_time_init_assemble_dat += m_w.m_timers.m_timer.elapsed_and_reset(); )  
        std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
        if (comp_node_perm){
            std::array<long, 4> gni;
            for (int i = 0; i < 4; ++i)
                gni[i] = m_enum.GNodeIndex(m_w.nodes[i]);    
            canonical_node_indexes = createOrderPermutation(gni.data()); 
            // std::cout << "GID:\n" << DenseMatrix<long>(gni.data(), 1, 4);    
        }
        bool has_active = fill_assemble_templates(m_w.nodes, m_w.edges, m_w.faces, cell, m_w.m_indexesC, m_w.m_indexesR, canonical_node_indexes.data(), m_helpers[nthread].same_template);
        TIMER_SCOPE( m_w.m_timers.m_time_fill_map_template += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        if (!has_active) return;

        typename Traits::MatFuncT::Memory fmem{m_w.m_iw.data(), m_w.m_w.data(), const_cast<const typename Traits::MatFuncT::Real **>(m_w.m_args.data()), m_w.m_res.data(), user_data, nthread};
        auto tp1 = func.out_nnz(0) == func.out_size1(0)*func.out_size2(0) ? MatSparsityView<Int>::DENSE : MatSparsityView<Int>::SPARSE_CSC;
        auto tp2 = func.out_nnz(1) == func.out_size1(1)*func.out_size2(1) ? MatSparsityView<Int>::DENSE : MatSparsityView<Int>::SPARSE_CSC;
        ElementalAssembler dat(&func, 
                           ElementalAssembler::MAT_RHS,
                           SparsedData<>(MatSparsityView<Int>(tp1, func.out_nnz(0), func.out_size1(0), func.out_size2(0), m_fd.colindA.data(), m_fd.rowA.data()), m_w.m_A.data()),
                           SparsedData<>(MatSparsityView<Int>(tp2, func.out_nnz(1), func.out_size1(1), func.out_size2(1), m_fd.colindF.data(), m_fd.rowF.data()), m_w.m_F.data()),
                           fmem, &m_w.m_Ab, &(m_helpers[nthread]), m_mesh, &m_w.nodes, &m_w.edges, &m_w.faces, &cell, canonical_node_indexes.data(),
                           TIMER_NOSCOPE (m_w.m_timers.getTimeMessures())  &m_w.pool );
        m_init_value_setter(dat);
        dat.update();
        TIMER_SCOPE(  m_w.m_timers.m_time_init_val_setter += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        m_prob_handler(dat);
        func.defragment_memory(nthread);
        m_w.pool.defragment();
        TIMER_SCOPE(  m_w.m_timers.m_time_proc_user_handler += m_w.m_timers.m_timer.elapsed_and_reset(); )
        // std::array<int, 4> cni1; std::copy(canonical_node_indexes.data(), canonical_node_indexes.data()+4, cni1.begin());
        // std::cout << "perm:\n" << DenseMatrix<int>(cni1.data(), 1, 4);  
        //         //<< "A:\n" << DenseMatrix<>(m_w.m_A.data(), nRows, nRows) 
        //         //<< "F:\n" << DenseMatrix<>(m_w.m_F.data(), 1, nRows) 
        // std::cout << "IR:\n" << DenseMatrix<long>(m_w.m_indexesR.data(), 1, nRows)
        //         << "IC:\n" << DenseMatrix<long>(m_w.m_indexesC.data(), 1, nRows) << std::endl;  
        if (use_ordered_insert){
            int* row_index_buffer = row_index_buffers.data() + nthread*nRows;
            for(int j = 0; j < nRows; j++) row_index_buffer[j] = j;
            std::sort(row_index_buffer, row_index_buffer + nRows, [&C = m_w.m_indexesC](auto i, auto j){ return internals::assemble_index_decode(C[i]).id < internals::assemble_index_decode(C[j]).id; });
        }

        for(int i = 0; i < nRows; i++){
            if(m_w.m_indexesR[i] != internals::CODE_UNDEF){//cols
                auto rid = internals::assemble_index_decode(m_w.m_indexesR[i]);
        #ifndef NDEBUG
                if( rid.id < getBegInd() || rid.id >= getEndInd()){
                    std::cout<<"wrong index C " << rid.id << " " << getBegInd() << " " << getEndInd() << " " << i << std::endl;
                    abort();
                }
        #endif
                if (nthreads > 1) L.Lock(rid.id);
                rhs[rid.id] += rid.sign * m_w.m_F[i];
                for(int j = 0; j < nRows; j++){
                    auto cid = internals::assemble_index_decode(m_w.m_indexesC[j]);
                    if(cid.id < 0 || cid.id >= m_enum.getMatrixSize()){
                        internals::print_assemble_matrix_indeces_incompatible(std::cout, m_mesh,
                                                                    m_w.m_indexesC, m_enum.getNumElem(), m_enum.getBegElemID(),
                                                                    m_enum.getEndElemID(), m_enum.getMatrixSize(), m_enum.getBegInd(), m_enum.getEndInd(), nRows, i);
                        abort();
                    }
                    if(!use_ordered_insert && m_w.m_Ab[j*nRows + i] && fabs(m_w.m_A[j*nRows + i]) > drp_val) {
                        matrix[rid.id][cid.id] += rid.sign*cid.sign*m_w.m_A[j * nRows + i];
                    }
                    if(m_w.m_Ab[j*nRows +i] && !std::isfinite(m_w.m_A[j*nRows +i])){
//                            internals::print_matrix_nan(std::cout, A, it, nRows, i, j);
                        if (nthreads > 1) L.UnLock(rid.id);
                        m_w.status = -1;
                        return;
                    }
                }
                if (use_ordered_insert){
                    int* row_index_buffer = row_index_buffers.data() + nthread*nRows;
                    if (is_mtx_include_template){
                        auto it = matrix[rid.id].Begin(), iend = matrix[rid.id].End();
                        for (int j = 0; j < nRows; ++j)  if (m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val) {
                            auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                            INMOST_DATA_REAL_TYPE val = rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            INMOST_DATA_ENUM_TYPE ind_q = it->first, ind_loc = cid.id;
                            auto cur_end = ((ind_loc - ind_q) >= std::distance(it, iend)) ? iend : (it + (ind_loc - ind_q + 1));
                            it = std::lower_bound(it, cur_end, ind_loc, [](const auto& a, INMOST_DATA_ENUM_TYPE b){ return a.first < b; });
                            assert(it != cur_end && it->first == ind_loc && "Matrix doesn't include full template");
                            it->second += val;
                        }
                    } else {
                        INMOST::Sparse::Row& swap_row = swap_rows[nthread];
                        swap_row.Resize(nRows + matrix[rid.id].Size());
                        auto& qrow = matrix[rid.id];
                        int j = 0, ii = 0, q = 0;
                        int j_end = nRows, ii_end = qrow.Size();
                        while (ii < ii_end && j < j_end){
                            auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                            INMOST_DATA_ENUM_TYPE ind_q = qrow.GetIndex(ii), ind_loc = cid.id;
                            INMOST_DATA_ENUM_TYPE ind = (ind_q < ind_loc) ? ind_q : ind_loc;
                            INMOST_DATA_REAL_TYPE val = (ind_q <= ind_loc) ? qrow.GetValue(ii) : rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            if (ind_q == ind_loc && m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val) 
                                val += rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            if (ind_q <= ind_loc || (m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val) ){
                                swap_row.GetIndex(q) = ind;
                                swap_row.GetValue(q) = val;
                                ++q;
                            }
                            ii += (ind_q <= ind_loc) ? 1 : 0;
                            j += (ind_q >= ind_loc) ? 1 : 0;    
                        }
                        if (ii < ii_end){
                            std::copy(qrow.Begin() + ii, qrow.End(), swap_row.Begin() + q);
                            q += ii_end - ii;
                        }
                        for(; j < j_end; ++j) if (m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val){
                            auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                            swap_row.GetIndex(q) = cid.id;
                            swap_row.GetValue(q) = rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            ++q;
                        }
                        swap_row.Resize(q);
                        qrow.Swap(swap_row);
                    }
                }
                if (nthreads > 1) L.UnLock(rid.id);
                if (!std::isfinite(m_w.m_F[i])) {
//                    internals::print_assemble_rhs_nan(std::cout, m_w.m_F, it, nRows, i);
                    m_w.status = -1;
                    return;
                }
            }
        }    
        TIMER_SCOPE(  m_w.m_timers.m_time_set_elemental_res += m_w.m_timers.m_timer.elapsed_and_reset(); )     
    };
    ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, cycle_body_func, m_mesh->FirstLocalID(INMOST::CELL), m_mesh->CellLastLocalID(), opts.user_data);
    func.release_memory_range(func_internal_mem_ids);

    return _return_assembled_status(nthreads); 
}

///This function assembles rhs of FE problem previously setted by PrepareProblem(...) function
/// @param[in,out] rhs is the global right-hand side to which the assembled RHS will be added, i.e. rhs <- rhs + assembled_rhs
/// @param[in] opts is user specified options for assembler and user supplied data to be postponed to problem handler
/// @return  0 if assembling successful else some error_code (number < 0)
///             ATTENTION: In case of unsuccessful completion of this function matrix and rhs has wrong elements!!!:
///         -1 if some local matrix or rhs had NaN.
///         -2 if some element is not tetrahedral or has broken component connectivity cell - faces - edges - nodes
template <typename Traits>
int AssemblerT<Traits>::AssembleRHS(INMOST::Sparse::Vector &rhs, const AssmOpts& opts){   
    reset_timers();
    if (!rhs_func){
        if (mat_rhs_func)
            std::cerr << "WARNING: rhs_func is not specified, so it will be generated from mat_rhs_func func" << std::endl;
        else 
            throw std::runtime_error("Right-hand side local evaluator is not specified");
    }
    auto func = generate_rhs_func();
    int nRows = m_info.TrialFuncs().NumDofOnTet();
    int nthreads = ThreadPar::get_num_threads<Traits::MatFuncT::parallel_type>(m_assm_traits.num_threads);
    resize_work_memory(nthreads);
    m_helpers.resize(nthreads, m_helpers[0]);
    rhs.SetInterval(getBegInd(), getEndInd());
    INMOST::Sparse::LockService L;
    if (nthreads > 1) L.SetInterval(getBegInd(), getEndInd());
    auto func_internal_mem_ids = func.setup_and_alloc_memory_range(nthreads);
    const bool reord_nds = m_assm_traits.reorder_nodes;
    const bool prep_ef = (m_assm_traits.prepare_edges || m_assm_traits.prepare_faces) || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE|DofT::FACE));
    const bool comp_node_perm = !m_enum.areVarsTriviallySymmetric() || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
    for (int i = 0; i < nthreads; ++i) m_wm[i].status = 0;
    TIMER_SCOPE( for (int i = 0; i < nthreads; ++i) m_wm[i].m_timers.reset(); )
    TIMER_SCOPE( m_timers.m_time_init_assemble_dat += m_timers.m_timer.elapsed_and_reset(); )
    
    auto cycle_body_func = [&func, nRows, reord_nds, prep_ef, comp_node_perm, &rhs, &L, nthreads, this](INMOST::Storage::integer lid, int nthread, void* user_data){
        auto& m_w = m_wm[nthread];
        if (m_w.status < 0) 
            return;
        INMOST::Cell cell = m_mesh->CellByLocalID(lid);
        if (!cell.isValid() || cell.Hidden()) return;
        collectConnectivityInfo(cell, m_w.nodes, m_w.edges, m_w.faces, reord_nds, prep_ef);
        TIMER_SCOPE( m_w.m_timers.m_time_init_assemble_dat += m_w.m_timers.m_timer.elapsed_and_reset(); )  
        std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
        if (comp_node_perm){
            std::array<long, 4> gni;
            for (int i = 0; i < 4; ++i)
                gni[i] = m_enum.GNodeIndex(m_w.nodes[i]);
            canonical_node_indexes = createOrderPermutation(gni.data());    
        }
        bool has_active = fill_assemble_templates(m_w.nodes, m_w.edges, m_w.faces, cell, m_w.m_indexesC, m_w.m_indexesR, canonical_node_indexes.data(), m_helpers[nthread].same_template);
        TIMER_SCOPE( m_w.m_timers.m_time_fill_map_template += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        if (!has_active) return;

        typename Traits::MatFuncT::Memory fmem{m_w.m_iw.data(), m_w.m_w.data(), const_cast<const typename Traits::MatFuncT::Real **>(m_w.m_args.data()), m_w.m_res.data(), user_data, nthread};
        auto tp2 = func.out_nnz(0) == func.out_size1(0)*func.out_size2(0) ? MatSparsityView<Int>::DENSE : MatSparsityView<Int>::SPARSE_CSC;
        ElementalAssembler dat(&func, ElementalAssembler::RHS,
                           SparsedData<>(MatSparsityView<Int>::make_as_dense(0, 0), nullptr),
                           SparsedData<>(MatSparsityView<Int>(tp2, func.out_nnz(0), func.out_size1(0), func.out_size2(0), m_fd.colindF.data(), m_fd.rowF.data()), m_w.m_F.data()),
                           fmem, &m_w.m_Ab, &(m_helpers[nthread]), m_mesh, &m_w.nodes, &m_w.edges, &m_w.faces, &cell, canonical_node_indexes.data(),
                           TIMER_NOSCOPE (m_w.m_timers.getTimeMessures())  &m_w.pool );
        m_init_value_setter(dat);
        dat.update();
        TIMER_SCOPE(  m_w.m_timers.m_time_init_val_setter += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        m_prob_handler(dat);
        func.defragment_memory(nthread);
        m_w.pool.defragment();
        TIMER_SCOPE(  m_w.m_timers.m_time_proc_user_handler += m_w.m_timers.m_timer.elapsed_and_reset(); )  
        for(int i = 0; i < nRows; i++){
            if(m_w.m_indexesR[i] != internals::CODE_UNDEF){//cols
                auto dcd = internals::assemble_index_decode(m_w.m_indexesR[i]);
        #ifndef NDEBUG
                if( dcd.id < getBegInd() || dcd.id >= getEndInd()){
                    std::cout<<"wrong index C " << dcd.id << " " << getBegInd() << " " << getEndInd() << " " << i << std::endl;
                    abort();
                }
        #endif  
                if (nthreads > 1) L.Lock(dcd.id);
                rhs[dcd.id] += dcd.sign * m_w.m_F[i];
                if (nthreads > 1) L.UnLock(dcd.id);
                if (!std::isfinite(m_w.m_F[i])) {
//                    internals::print_assemble_rhs_nan(std::cout, m_w.m_F, it, nRows, i);
                    m_w.status = -1;
                    return;
                }
            }
        }    
        TIMER_SCOPE(  m_w.m_timers.m_time_set_elemental_res += m_w.m_timers.m_timer.elapsed_and_reset(); )     
    };
    ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, cycle_body_func, m_mesh->FirstLocalID(INMOST::CELL), m_mesh->CellLastLocalID(), opts.user_data);
    func.release_memory_range(func_internal_mem_ids);
    
    return _return_assembled_status(nthreads);
}

///@brief This function adds a zero matrix of structural nonzeros to the current matrix. This function leaves the elements of each row sorted in ascending order of the column number
/// @details The meaning of this function is related to the sparse matrix storage format. 
/// This function adds elements to the input matrix corresponding to the positions of structural non-zeros in the global finite element matrix. 
/// @note The zero matrix of structural nonzeros is a matrix with elements in those positions where, 
/// in general, there may be a nonzero number when assembling the FE matrix with a given local assembler on a given mesh.
/// @param matrix is the global matrix to which the structured nonzero matrix will be added, i.e. matrix <- matrix + template_matrix
/// @return 0 if assembling successful else -1
template <typename Traits>
int AssemblerT<Traits>::AssembleTemplate(INMOST::Sparse::Matrix &matrix){
    reset_timers();
    if (!mat_func && !mat_rhs_func)
        throw std::runtime_error("Matrix local evaluator is not specified");
    auto func = generate_mat_func();
    int nRows = m_info.TrialFuncs().NumDofOnTet();
    int nthreads = ThreadPar::get_num_threads<Traits::MatFuncT::parallel_type>(m_assm_traits.num_threads);
    resize_work_memory(nthreads);
    m_helpers.resize(nthreads, m_helpers[0]);
    std::vector<int> row_index_buffers(nRows*nthreads);
    std::vector<INMOST::Sparse::Row> swap_rows(nthreads);
    matrix.SetInterval(getBegInd(),getEndInd());
    for(auto i = getBegInd(); i < getEndInd(); i++) if (matrix[i].get_safe(i) == 0.0)
        matrix[i][i] = 0.0;
    ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, 
        [&A = matrix](INMOST_DATA_ENUM_TYPE lid, int nthread){ (void) nthread; std::sort(A[lid].Begin(), A[lid].End(), [](const auto& a, const auto& b){ return a.first < b.first; }); }, 
        static_cast<INMOST_DATA_ENUM_TYPE>(getBegInd()), 
        static_cast<INMOST_DATA_ENUM_TYPE>(getEndInd())
    );    
    INMOST::Sparse::LockService L;
    if (nthreads > 1) L.SetInterval(getBegInd(), getEndInd());    
    auto func_internal_mem_ids = func.setup_and_alloc_memory_range(nthreads);
    const bool reord_nds = m_assm_traits.reorder_nodes;
    const bool prep_ef = (m_assm_traits.prepare_edges || m_assm_traits.prepare_faces) || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE|DofT::FACE));
    const bool comp_node_perm = !m_enum.areVarsTriviallySymmetric() || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
    for (int i = 0; i < nthreads; ++i) m_wm[i].status = 0;
    TIMER_SCOPE( for (int i = 0; i < nthreads; ++i) m_wm[i].m_timers.reset(); )
    TIMER_SCOPE( m_timers.m_time_init_assemble_dat += m_timers.m_timer.elapsed_and_reset(); )

    auto cycle_body_func = [&func, nRows, reord_nds, prep_ef, comp_node_perm, &row_index_buffers, &swap_rows, &matrix, &L, nthreads, this](INMOST::Storage::integer lid, int nthread){
        auto& m_w = m_wm[nthread];
        int* row_index_buffer = row_index_buffers.data() + nRows * nthread;
        INMOST::Sparse::Row& swap_row = swap_rows[nthread];
        if (m_w.status < 0) return;
        INMOST::Cell cell = m_mesh->CellByLocalID(lid);
        if (!cell.isValid() || cell.Hidden()) return;
        collectConnectivityInfo(cell, m_w.nodes, m_w.edges, m_w.faces, reord_nds, prep_ef);
        TIMER_SCOPE( m_w.m_timers.m_time_init_assemble_dat += m_w.m_timers.m_timer.elapsed_and_reset(); )
        std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
        if (comp_node_perm){
            std::array<long, 4> gni;
            for (int i = 0; i < 4; ++i)
                gni[i] = m_enum.GNodeIndex(m_w.nodes[i]);
            canonical_node_indexes = createOrderPermutation(gni.data());    
        }
        bool has_active = fill_assemble_templates(m_w.nodes, m_w.edges, m_w.faces, cell, m_w.m_indexesC, m_w.m_indexesR, canonical_node_indexes.data(), m_helpers[nthread].same_template);
        if (!has_active) return;

        auto tp1 = func.out_nnz(0) == func.out_size1(0)*func.out_size2(0) ? MatSparsityView<Int>::DENSE : MatSparsityView<Int>::SPARSE_CSC;
        MatSparsityView<Int> sp_view(tp1, func.out_nnz(0), func.out_size1(0), func.out_size2(0), m_fd.colindA.data(), m_fd.rowA.data());
        sp_view.template fillTemplate<decltype(m_w.m_Ab.begin()), bool>(m_w.m_Ab.begin()); 

        for(int j = 0; j < nRows; j++) row_index_buffer[j] = j;
        std::sort(row_index_buffer, row_index_buffer+nRows, [&C = m_w.m_indexesC](auto i, auto j){ return internals::assemble_index_decode(C[i]).id < internals::assemble_index_decode(C[j]).id; });
        TIMER_SCOPE( m_w.m_timers.m_time_fill_map_template += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        
        for(int i = 0; i < nRows; i++){
            if(m_w.m_indexesR[i] != internals::CODE_UNDEF){//cols
                auto rid = internals::assemble_index_decode(m_w.m_indexesR[i]);
        #ifndef NDEBUG
                if( rid.id < getBegInd() || rid.id >= getEndInd()){
                    std::cout<<"wrong index C " << rid.id << " " << getBegInd() << " " << getEndInd() << " " << i << std::endl;
                    abort();
                }
        #endif
                swap_row.Resize(nRows + matrix[rid.id].Size());
                if (nthreads > 1) L.Lock(rid.id);
                auto& qrow = matrix[rid.id];
                int j = 0, ii = 0, q = 0;
                int j_end = nRows, ii_end = qrow.Size();
                while (ii < ii_end && j < j_end){
                    auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                    INMOST_DATA_ENUM_TYPE ind_q = qrow.GetIndex(ii), ind_loc = cid.id;
                    INMOST_DATA_ENUM_TYPE ind = (ind_q < ind_loc) ? ind_q : ind_loc;
                    INMOST_DATA_REAL_TYPE val = (ind_q <= ind_loc) ? qrow.GetValue(ii) : 0.0;
                    if (ind_q <= ind_loc || m_w.m_Ab[row_index_buffer[j]*nRows + i]){
                        swap_row.GetIndex(q) = ind;
                        swap_row.GetValue(q) = val;
                        ++q;
                    }
                    ii += (ind_q <= ind_loc) ? 1 : 0;
                    j += (ind_q >= ind_loc) ? 1 : 0;    
                }
                if (ii < ii_end){
                    std::copy(qrow.Begin() + ii, qrow.End(), swap_row.Begin() + q);
                    q += ii_end - ii;
                }
                for(; j < j_end; ++j) if (m_w.m_Ab[row_index_buffer[j]*nRows + i]){
                    auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                    swap_row.GetIndex(q) = cid.id;
                    swap_row.GetValue(q) = 0;
                    ++q;
                }
                swap_row.Resize(q);
                qrow.Swap(swap_row);
                if (nthreads > 1) L.UnLock(rid.id);
            }
        }    
        TIMER_SCOPE(  m_w.m_timers.m_time_set_elemental_res += m_w.m_timers.m_timer.elapsed_and_reset(); )
    };

    ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, cycle_body_func, m_mesh->FirstLocalID(INMOST::CELL), m_mesh->CellLastLocalID());
    func.release_memory_range(func_internal_mem_ids);

    return _return_assembled_status(nthreads);
}

///This function assembles matrix of FE problem previously setted by PrepareProblem(...) function
/// @param[in,out] matrix is the global matrix to which the assembled matrix will be added, i.e. matrix <- matrix + assembled_matrix
/// @param[in] opts is user specified options for assembler and user supplied data to be postponed to problem handler
/// @return  0 if assembling successful else some error_code (number < 0)
///             ATTENTION: In case of unsuccessful completion of this function matrix and rhs has wrong elements!!!:
///         -1 if some local matrix or rhs had NaN.
///         -2 if some element is not tetrahedral or has broken component connectivity cell - faces - edges - nodes
template <typename Traits>
int AssemblerT<Traits>::AssembleMatrix(INMOST::Sparse::Matrix &matrix, const AssmOpts& opts){
    reset_timers();
    if (!mat_func){
        if (mat_rhs_func)
            std::cerr << "WARNING: mat_func is not specified, so it will be generated from mat_rhs_func func" << std::endl;
        else 
            throw std::runtime_error("Matrix local evaluator is not specified");
    }
    auto func = generate_mat_func();
    int nRows = m_info.TrialFuncs().NumDofOnTet();
    int nthreads = ThreadPar::get_num_threads<Traits::MatFuncT::parallel_type>(m_assm_traits.num_threads);
    resize_work_memory(nthreads);
    m_helpers.resize(nthreads, m_helpers[0]);
    matrix.SetInterval(getBegInd(),getEndInd());
    internals::set_elements_on_matrix_diagonal(matrix, opts);
    if (opts.use_ordered_insert && !opts.is_mtx_sorted)
        ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, 
            [&A = matrix](INMOST_DATA_ENUM_TYPE lid, int nthread){ (void) nthread; std::sort(A[lid].Begin(), A[lid].End()); }, 
            static_cast<INMOST_DATA_ENUM_TYPE>(getBegInd()), 
            static_cast<INMOST_DATA_ENUM_TYPE>(getEndInd())
        );    
    std::vector<int> row_index_buffers;
    std::vector<INMOST::Sparse::Row> swap_rows;
    if (opts.use_ordered_insert){
        row_index_buffers.resize(nRows*nthreads);
        swap_rows.resize(nthreads);
    }
    INMOST::Sparse::LockService L;
    if (nthreads > 1) L.SetInterval(getBegInd(), getEndInd());    
    auto func_internal_mem_ids = func.setup_and_alloc_memory_range(nthreads);
    const bool reord_nds = m_assm_traits.reorder_nodes;
    const bool prep_ef = (m_assm_traits.prepare_edges || m_assm_traits.prepare_faces) || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE|DofT::FACE));
    const bool comp_node_perm = !m_enum.areVarsTriviallySymmetric() || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
    for (int i = 0; i < nthreads; ++i) m_wm[i].status = 0;
    TIMER_SCOPE( for (int i = 0; i < nthreads; ++i) m_wm[i].m_timers.reset(); )
    TIMER_SCOPE( m_timers.m_time_init_assemble_dat += m_timers.m_timer.elapsed_and_reset(); )
    
    auto cycle_body_func = [&row_index_buffers, &swap_rows, use_ordered_insert = opts.use_ordered_insert, is_mtx_include_template = opts.is_mtx_include_template,
                            &func, nRows, reord_nds, prep_ef, comp_node_perm, &matrix,  &L, nthreads, drp_val = opts.drop_val, this](INMOST::Storage::integer lid, int nthread, void* user_data){
        auto& m_w = m_wm[nthread];
        if (m_w.status < 0) return;
        INMOST::Cell cell = m_mesh->CellByLocalID(lid);
        if (!cell.isValid() || cell.Hidden()) return;
        collectConnectivityInfo(cell, m_w.nodes, m_w.edges, m_w.faces, reord_nds, prep_ef);
        TIMER_SCOPE( m_w.m_timers.m_time_init_assemble_dat += m_w.m_timers.m_timer.elapsed_and_reset(); )  
        std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
        if (comp_node_perm){
            std::array<long, 4> gni;
            for (int i = 0; i < 4; ++i)
                gni[i] = m_enum.GNodeIndex(m_w.nodes[i]);
            canonical_node_indexes = createOrderPermutation(gni.data());    
        }
        bool has_active = fill_assemble_templates(m_w.nodes, m_w.edges, m_w.faces, cell, m_w.m_indexesC, m_w.m_indexesR, canonical_node_indexes.data(), m_helpers[nthread].same_template);
        TIMER_SCOPE( m_w.m_timers.m_time_fill_map_template += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        if (!has_active) return;

        typename Traits::MatFuncT::Memory fmem{m_w.m_iw.data(), m_w.m_w.data(), const_cast<const typename Traits::MatFuncT::Real **>(m_w.m_args.data()), m_w.m_res.data(), user_data, nthread};
        auto tp1 = func.out_nnz(0) == func.out_size1(0)*func.out_size2(0) ? MatSparsityView<Int>::DENSE : MatSparsityView<Int>::SPARSE_CSC;
        ElementalAssembler dat(&func, ElementalAssembler::MAT,
                           SparsedData<>(MatSparsityView<Int>(tp1, func.out_nnz(0), func.out_size1(0), func.out_size2(0), m_fd.colindA.data(), m_fd.rowA.data()), m_w.m_A.data()),
                           SparsedData<>(MatSparsityView<Int>::make_as_dense(0, 0), nullptr),
                           fmem, &m_w.m_Ab, &(m_helpers[nthread]), m_mesh, &m_w.nodes, &m_w.edges, &m_w.faces, &cell, canonical_node_indexes.data(),
                           TIMER_NOSCOPE (m_w.m_timers.getTimeMessures())  &m_w.pool );
        m_init_value_setter(dat);
        dat.update();
        TIMER_SCOPE(  m_w.m_timers.m_time_init_val_setter += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        m_prob_handler(dat);
        func.defragment_memory(nthread);
        m_w.pool.defragment();
        TIMER_SCOPE(  m_w.m_timers.m_time_proc_user_handler += m_w.m_timers.m_timer.elapsed_and_reset(); ) 
        if (use_ordered_insert){
            int* row_index_buffer = row_index_buffers.data() + nthread*nRows;
            for(int j = 0; j < nRows; j++) row_index_buffer[j] = j;
            std::sort(row_index_buffer, row_index_buffer + nRows, [&C = m_w.m_indexesC](auto i, auto j){ return internals::assemble_index_decode(C[i]).id < internals::assemble_index_decode(C[j]).id; });
        } 
        for(int i = 0; i < nRows; i++){
            if(m_w.m_indexesR[i] != internals::CODE_UNDEF){//cols
                auto rid = internals::assemble_index_decode(m_w.m_indexesR[i]);
        #ifndef NDEBUG
                if( rid.id < getBegInd() || rid.id >= getEndInd()){
                    std::cout<<"wrong index C " << rid.id << " " << getBegInd() << " " << getEndInd() << " " << i << std::endl;
                    abort();
                }
        #endif
                if (nthreads > 1) L.Lock(rid.id);
                for(int j = 0; j < nRows; j++){
                    auto cid = internals::assemble_index_decode(m_w.m_indexesC[j]);
                    if(cid.id < 0 || cid.id >= m_enum.getMatrixSize()){
                        internals::print_assemble_matrix_indeces_incompatible(std::cout, m_mesh,
                                                                    m_w.m_indexesC, m_enum.getNumElem(), m_enum.getBegElemID(),
                                                                    m_enum.getEndElemID(), m_enum.getMatrixSize(), m_enum.getBegInd(), m_enum.getEndInd(), nRows, i);
                        abort();
                    }
                    if(!use_ordered_insert && m_w.m_Ab[j*nRows +i] && fabs(m_w.m_A[j*nRows +i]) > drp_val) {
                        matrix[rid.id][cid.id] += rid.sign*cid.sign*m_w.m_A[j * nRows + i];
                    }
                    if(m_w.m_Ab[j*nRows +i] && !std::isfinite(m_w.m_A[j*nRows +i])){
//                            internals::print_matrix_nan(std::cout, A, it, nRows, i, j);
                        if (nthreads > 1) L.UnLock(rid.id);
                        m_w.status = -1;
                        return;
                    }
                }
                if (use_ordered_insert){
                    int* row_index_buffer = row_index_buffers.data() + nthread*nRows;
                    if (is_mtx_include_template){
                        auto it = matrix[rid.id].Begin(), iend = matrix[rid.id].End();
                        for (int j = 0; j < nRows; ++j)  if (m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val) {
                            auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                            INMOST_DATA_REAL_TYPE val = rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            INMOST_DATA_ENUM_TYPE ind_q = it->first, ind_loc = cid.id;
                            auto cur_end = ((ind_loc - ind_q) >= std::distance(it, iend)) ? iend : (it + (ind_loc - ind_q + 1));
                            it = std::lower_bound(it, cur_end, ind_loc, [](const auto& a, INMOST_DATA_ENUM_TYPE b){ return a.first < b; });
                            assert(it != cur_end && it->first == ind_loc &&  "Matrix doesn't include full template");
                            it->second += val;
                        }
                    } else {
                        INMOST::Sparse::Row& swap_row = swap_rows[nthread];
                        swap_row.Resize(nRows + matrix[rid.id].Size());
                        auto& qrow = matrix[rid.id];
                        int j = 0, ii = 0, q = 0;
                        int j_end = nRows, ii_end = qrow.Size();
                        while (ii < ii_end && j < j_end){
                            auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                            INMOST_DATA_ENUM_TYPE ind_q = qrow.GetIndex(ii), ind_loc = cid.id;
                            INMOST_DATA_ENUM_TYPE ind = (ind_q < ind_loc) ? ind_q : ind_loc;
                            INMOST_DATA_REAL_TYPE val = (ind_q <= ind_loc) ? qrow.GetValue(ii) : rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            if (ind_q == ind_loc && m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val) 
                                val += rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            if (ind_q <= ind_loc || (m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val) ){
                                swap_row.GetIndex(q) = ind;
                                swap_row.GetValue(q) = val;
                                ++q;
                            }
                            ii += (ind_q <= ind_loc) ? 1 : 0;
                            j += (ind_q >= ind_loc) ? 1 : 0;    
                        }
                        if (ii < ii_end){
                            std::copy(qrow.Begin() + ii, qrow.End(), swap_row.Begin() + q);
                            q += ii_end - ii;
                        }
                        for(; j < j_end; ++j) if (m_w.m_Ab[row_index_buffer[j]*nRows + i] && fabs(m_w.m_A[row_index_buffer[j]*nRows + i]) > drp_val){
                            auto cid = internals::assemble_index_decode(m_w.m_indexesC[row_index_buffer[j]]);
                            swap_row.GetIndex(q) = cid.id;
                            swap_row.GetValue(q) = rid.sign*cid.sign*m_w.m_A[row_index_buffer[j]*nRows + i];
                            ++q;
                        }
                        swap_row.Resize(q);
                        qrow.Swap(swap_row);
                    }
                }
                if (nthreads > 1) L.UnLock(rid.id);
            }
        }    
        TIMER_SCOPE(  m_w.m_timers.m_time_set_elemental_res += m_w.m_timers.m_timer.elapsed_and_reset(); )     
    };
    ThreadPar::parallel_for<Traits::MatFuncT::parallel_type>(nthreads, cycle_body_func, m_mesh->FirstLocalID(INMOST::CELL), m_mesh->CellLastLocalID(), opts.user_data);
    func.release_memory_range(func_internal_mem_ids);

    return _return_assembled_status(nthreads); 
}

#undef TIMER_SCOPE
#undef TIMER_NOSCOPE

template <typename Traits>
template<bool OnlyIfDataAvailable, class RandomIt>
void AssemblerT<Traits>::GatherDataOnElement(const INMOST::Tag* var_tag, const std::size_t ntags, const INMOST::Cell& cell, RandomIt out, const int* component/*[ncomp]*/, unsigned int  ncomp) const{
    auto* m = cell.GetMeshLink();
    INMOST::ElementArray<INMOST::Node> nds(m, 4); 
    INMOST::ElementArray<INMOST::Edge> eds(m, 6); 
    INMOST::ElementArray<INMOST::Face> fcs(m, 4);
    const bool prep_ef = m_info.TestFuncs().GetGeomMask() & (DofT::EDGE|DofT::FACE);
    collectConnectivityInfo(cell, nds, eds, fcs, true, prep_ef);
    const bool comp_node_perm = !m_enum.areVarsTriviallySymmetric() || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
    std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
    if (comp_node_perm){
        std::array<long, 4> gni;
        for (int i = 0; i < 4; ++i)
            gni[i] = m_enum.GNodeIndex(nds[i]);
        canonical_node_indexes = createOrderPermutation(gni.data());    
    }
    Ani::GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tag, ntags, m_info.TestFuncs(), cell, fcs, eds, nds, canonical_node_indexes.data(), out, component, ncomp);
}
template <typename Traits>
template<bool OnlyIfDataAvailable, class RandomIt>
void AssemblerT<Traits>::GatherDataOnElement(INMOST::Tag from, const INMOST::Cell& cell, RandomIt out, const int* component/*[ncomp]*/, unsigned int  ncomp) const{
    auto* m = cell.GetMeshLink();
    INMOST::ElementArray<INMOST::Node> nds(m, 4); 
    INMOST::ElementArray<INMOST::Edge> eds(m, 6); 
    INMOST::ElementArray<INMOST::Face> fcs(m, 4);
    const bool prep_ef = m_info.TestFuncs().GetGeomMask() & (DofT::EDGE|DofT::FACE);
    collectConnectivityInfo(cell, nds, eds, fcs, true, prep_ef);
    const bool comp_node_perm = !m_enum.areVarsTriviallySymmetric() || (m_info.TestFuncs().GetGeomMask() & (DofT::EDGE_ORIENT|DofT::FACE_ORIENT));
    std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
    if (comp_node_perm){
        std::array<long, 4> gni;
        for (int i = 0; i < 4; ++i)
            gni[i] = m_enum.GNodeIndex(nds[i]);
        canonical_node_indexes = createOrderPermutation(gni.data());    
    }
    Ani::GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(from, m_info.TestFuncs(), cell, fcs, eds, nds, canonical_node_indexes.data(), out, component, ncomp);
}

#ifndef NO_ASSEMBLER_TIMERS
template <typename Traits>
void AssemblerT<Traits>::TimerData::reset(){
    m_timer_ttl.reset();
    m_timer.reset();
    m_time_init_assemble_dat = 0,
    m_time_fill_map_template = 0,
    m_time_init_val_setter = 0,
    m_time_init_user_handler = 0,
    m_time_comp_func = 0,
    m_time_proc_user_handler = 0,
    m_time_set_elemental_res = 0,
    m_timer_total = 0;
}
#endif

template <typename Traits>
void AssemblerT<Traits>::reset_timers(){
#ifndef NO_ASSEMBLER_TIMERS
    m_timers.reset();
#endif
}

#ifndef NO_ASSEMBLER_TIMERS
template <typename Traits>
ElementalAssembler::TimeMessures AssemblerT<Traits>::TimerData::getTimeMessures(){
    ElementalAssembler::TimeMessures tm;
    tm.m_timer = &m_timer;
    tm.m_time_init_user_handler = &m_time_init_user_handler;
    tm.m_time_comp_func = &m_time_comp_func;
    return tm;
};
#endif

#ifndef NO_ASSEMBLER_TIMERS
#define TAKE_TIMER(X) X
#else
#define TAKE_TIMER(X) -1
#endif

template <typename Traits>
double AssemblerT<Traits>::GetTimeInitAssembleData() const { return TAKE_TIMER(m_timers.m_time_init_assemble_dat); }
template <typename Traits>
double AssemblerT<Traits>::GetTimeFillMapTemplate() const { return TAKE_TIMER(m_timers.m_time_fill_map_template); }
template <typename Traits>
double AssemblerT<Traits>::GetTimeInitValSet() const { return TAKE_TIMER(m_timers.m_time_init_val_setter); }
template <typename Traits>
double AssemblerT<Traits>::GetTimeInitUserHandler() const { return TAKE_TIMER(m_timers.m_time_init_user_handler); }
template <typename Traits>
double AssemblerT<Traits>::GetTimeEvalLocFunc() const { return TAKE_TIMER(m_timers.m_time_comp_func); }
template <typename Traits>
double AssemblerT<Traits>::GetTimePostProcUserHandler() const { return TAKE_TIMER(m_timers.m_time_proc_user_handler); }
template <typename Traits>
double AssemblerT<Traits>::GetTimeFillGlobalStructs() const { return TAKE_TIMER(m_timers.m_time_set_elemental_res); }
template <typename Traits>
double AssemblerT<Traits>::GetTimeTotal() const { return TAKE_TIMER(m_timers.m_timer_total); }

#undef TAKE_TIMER

namespace internals{

template<typename MatFuncT>
int SystemFromMatRhsFWrap<MatFuncT>::operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data, Int mem_id) const{ 
    Real* rhs_p = res[1];
    int stat1 = m_mat->operator()(args, res, w, iw, user_data, mem_id); 
    int stat2 = 0;
    if (stat1 == 0 && m_rhs){
        res[1] = rhs_p;
        stat2 = m_rhs->operator()(args, res+1, w, iw, user_data, mem_id);
    }
    return stat1+stat2;
}

template<typename MatFuncT>
int SystemFromMatRhsFWrap<MatFuncT>::operator()(Memory mem) const { 
    Real* rhs_p = mem.m_res[1];
    int stat1 = m_mat->operator()(mem); 
    int stat2 = 0;
    if (stat1 == 0 && m_rhs){
        mem.m_res[1] = rhs_p;
        stat2 = m_rhs->operator()(mem.m_args, mem.m_res+1, mem.m_w, mem.m_iw, mem.user_data, mem.mem_id);
    }
    return stat1+stat2;
}
template<typename MatFuncT>
typename SystemFromMatRhsFWrap<MatFuncT>::Int SystemFromMatRhsFWrap<MatFuncT>::setup_and_alloc_memory(){ 
    Int id1 = m_mat->setup_and_alloc_memory(); 
    Int id2 = 0;
    if (id1 >= 0 && m_rhs){
        id2 = m_rhs->setup_and_alloc_memory();
        if (id2 < 0){
            m_mat->release_memory(id1);
            return id2;
        }
    }
    return encode_id(id1, id2);
}
template<typename MatFuncT>
void SystemFromMatRhsFWrap<MatFuncT>::release_memory(Int mem_id) { 
    auto ids = decode_id(mem_id);
    m_mat->release_memory(m_rhs ? ids.first : mem_id);
    if (m_rhs){
        m_rhs->release_memory(ids.second);
    } 
}
template<typename MatFuncT>
void SystemFromMatRhsFWrap<MatFuncT>::defragment_memory(Int mem_id){ 
    auto ids = decode_id(mem_id);
    m_mat->defragment_memory(m_rhs ? ids.first : mem_id);
    if (m_rhs)
        m_rhs->defragment_memory(ids.second);
}
template<typename MatFuncT>
std::pair<typename SystemFromMatRhsFWrap<MatFuncT>::Int, typename SystemFromMatRhsFWrap<MatFuncT>::Int> SystemFromMatRhsFWrap<MatFuncT>::setup_and_alloc_memory_range(Int num_threads){ 
    auto id1 = m_mat->setup_and_alloc_memory_range(num_threads); 
    std::pair<Int, Int> id2{0, 0};
    if (id1.first >= 0 && id1.second >= 0 && m_rhs){
        id2 = m_rhs->setup_and_alloc_memory_range(num_threads);
        if (id2.first < 0 || id2.second < 0){
            m_mat->release_memory_range(id1);
            return id2;
        }
    }
    return {encode_id(id1.first, id2.first), encode_id(id1.second, id2.second)};
}
template<typename MatFuncT>
void SystemFromMatRhsFWrap<MatFuncT>::release_memory_range(std::pair<Int, Int> id_range){ 
    auto ids1 = decode_id(id_range.first), ids2 = decode_id(id_range.second);
    m_mat->release_memory_range(m_rhs ? std::pair<Int, Int>{ids1.first, ids2.first} : id_range); 
    if (m_rhs)
        m_rhs->release_memory_range(std::pair<Int, Int>{ids1.second, ids2.second});
}
template<typename MatFuncT>
void SystemFromMatRhsFWrap<MatFuncT>::release_all_memory(){ 
    m_mat->release_all_memory(); 
    if (m_rhs)
        m_rhs->release_all_memory();
}
template<typename MatFuncT>
void SystemFromMatRhsFWrap<MatFuncT>::clear_memory(){ 
    m_mat->clear_memory(); 
    if (m_rhs)
        m_rhs->clear_memory();
}
template<typename MatFuncT>
void SystemFromMatRhsFWrap<MatFuncT>::working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const{ 
    m_mat->working_sizes(sz_args, sz_res, sz_w, sz_iw); 
    if (m_rhs){
        size_t sz_args1 = 0, sz_res1 = 0, sz_w1 = 0, sz_iw1 = 0;
        m_rhs->working_sizes(sz_args1, sz_res1, sz_w1, sz_iw1);
        if (sz_args1 > sz_args) sz_args = sz_args1;
        if (sz_res1+1 > sz_res) sz_res = sz_res1+1;
        if (sz_w1 > sz_w) sz_w = sz_w1;
        if (sz_iw1 > sz_iw) sz_iw = sz_iw1;
    }
}

template<typename MatFuncT>
std::ostream& SystemFromMatRhsFWrap<MatFuncT>::print_signature(std::ostream& out) const{ 
    if (m_rhs) out << "System from different matrix and rhs functions\n";
    auto& res = m_mat->print_signature(out); 
    if (m_rhs)
        return m_rhs->print_signature(out);
    return res;    
}
template<typename MatFuncT>
int RhsFromSystemFWrap<MatFuncT>::operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data, Int mem_id) const{ 
    auto ret = m_f->operator()(args, res, w, iw, user_data, mem_id); 
    if (!is_rhs_func) std::swap(res[0], res[1]);
    return ret;
}
template<typename MatFuncT>
int RhsFromSystemFWrap<MatFuncT>::operator()(Memory mem) const { 
    auto res = m_f->operator()(mem);
    if (!is_rhs_func) std::swap(mem.m_res[0], mem.m_res[1]);
    return res;
}

} //end namespace internals

template <typename Traits>
internals::SystemFromMatRhsFWrap<typename Traits::MatFuncT> AssemblerT<Traits>::generate_mat_rhs_func() {
    if (mat_rhs_func)
        return {&mat_rhs_func};
    if (mat_func && rhs_func){
        auto res = internals::SystemFromMatRhsFWrap<typename Traits::MatFuncT>{&mat_func, &rhs_func};
        extend_memory_for_fem_func(res);
        return res;   
    }  
    return {};
}
template <typename Traits>
internals::MatFromSystemFWrap<typename Traits::MatFuncT> AssemblerT<Traits>::generate_mat_func(){
    if (mat_func)
        return {&mat_func, true};
    if (mat_rhs_func){
        auto res = internals::MatFromSystemFWrap<typename Traits::MatFuncT>{&mat_rhs_func, false};
        extend_memory_for_fem_func(res);
        return res;
    }
    return {};        
}
template <typename Traits>
internals::RhsFromSystemFWrap<typename Traits::MatFuncT> AssemblerT<Traits>::generate_rhs_func(){
    if (rhs_func)
        return {&rhs_func, true};
    if (mat_rhs_func){
        auto res = internals::RhsFromSystemFWrap<typename Traits::MatFuncT>{&mat_rhs_func, false};
        extend_memory_for_fem_func(res);
        return res;
    }
    return {};        
}

template <typename Traits>
void AssemblerT<Traits>::Clear() {
    m_wm.clear();
    m_fd.Clear();
    m_helpers.clear();
    orderC.clear();
    orderR.clear();
    mat_func = typename Traits::MatFuncT();
    rhs_func = typename Traits::MatFuncT();
    mat_rhs_func = typename Traits::MatFuncT();
    m_info.Clear();
    m_prob_handler = typename Traits::ProbHandler();
    m_init_value_setter = typename Traits::InitValueSetter();
    m_enum.Clear();
    m_mesh = nullptr;
}

};

#endif //CARNUM_ASSEMBLER_INL