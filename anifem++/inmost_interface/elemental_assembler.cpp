//
// Created by Liogky Alexey on 28.03.2022.
//

#include "elemental_assembler.h"
#include <stdexcept>
#include <string>

std::ostream &ElemMatEval::print_signature(std::ostream &out) const {
    bool user_data_req = is_user_data_required();
    size_t sz_args = 0, sz_res = 0, sz_w = 0, sz_iw = 0;
    working_sizes(sz_args, sz_res, sz_w, sz_iw);
    out << "ElemMatEval: ";
    out << "(";
    for (int i = 0; i < static_cast<int>(n_in()); ++i){
        if (i > 0) out << ", ";
        Int sz1 = in_size1(i), sz2 = in_size2(i), nnz = in_nnz(i);
        out << "i" << i << "[" << sz1;
        if (sz2 > 1) out << "x" << sz2;
        if (nnz != sz1*sz2) out << " nnz="<<nnz;
        out << "]";
    }
    bool with_additional_memory = (sz_w > 0) || (sz_iw > 0) || (sz_args > n_in()) || (sz_res > n_out());
    if (user_data_req) out << ", user_data";
    if (with_additional_memory) {
        out << ", mem{ ";
        if (sz_w > 0) out << "w[" << sz_w << "], ";
        if (sz_iw > 0) out << "iw[" << sz_iw << "], ";
        if (sz_args > n_in()) out << "args[" << sz_args << "], ";
        if (sz_res > n_out()) out << "res[" << sz_res << "] ";
        out << "}";
    }
    out << ")->(";
    for (int i = 0; i < static_cast<int>(n_out()); ++i){
        if (i > 0) out << ", ";
        Int sz1 = out_size1(i), sz2 = out_size2(i), nnz = out_nnz(i);
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

void reorderNodesOnTetrahedron(INMOST::ElementArray<INMOST::Node> &nodes) {
    assert(nodes.size() >= 4);
    double m[3][3];
    auto    crd0 = nodes[0].Coords(), crd1 = nodes[1].Coords(), 
            crd2 =  nodes[2].Coords(), crd3 =  nodes[3].Coords();
    for (int j = 0; j < 3; ++j){
        m[0][j] = crd0[j] - crd3[j];
        m[1][j] = crd1[j] - crd3[j];
        m[2][j] = crd2[j] - crd3[j];
    }  
    double det = 0;
    det += m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1]);
    det += m[0][1]*(m[1][2]*m[2][0] - m[1][0]*m[2][2]);
    det += m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    if (det < 0) 
        std::swap(nodes.data()[2], nodes.data()[3]);
}

bool operator==(const FemExprDescr::DiscrSpaceHelper &a, const FemExprDescr::DiscrSpaceHelper &b) {
    auto t = a->ActualType();
    if (t != b->ActualType()) return false;
    switch (t) {
        case FemExprDescr::SimpleType: {
            auto aptr = static_cast<FemExprDescr::SimpleDiscrSpace*>(a.get());
            auto bptr = static_cast<FemExprDescr::SimpleDiscrSpace*>(b.get());
            if (aptr == bptr) return true;
            if (aptr->driver_id != bptr->driver_id) return false;
            if (aptr->fem_id != bptr->fem_id) return false;
            if (aptr->m_shiftTetDof != bptr->m_shiftTetDof) return false;
            if (aptr->m_shiftDof != bptr->m_shiftDof) return false;
            return true;
        }
        case FemExprDescr::SimpleVectorType:{
            auto aptr = static_cast<FemExprDescr::SimpleVectorDiscrSpace*>(a.get());
            auto bptr = static_cast<FemExprDescr::SimpleVectorDiscrSpace*>(b.get());
            if (aptr == bptr) return true;
            if (aptr->m_dim != bptr->m_dim) return false;
            return static_cast<FemExprDescr::DiscrSpaceHelper>(aptr->base) == static_cast<FemExprDescr::DiscrSpaceHelper>(bptr->base);
        }
        case FemExprDescr::UniteType:{
            auto aptr = static_cast<FemExprDescr::UniteDiscrSpace*>(a.get());
            auto bptr = static_cast<FemExprDescr::UniteDiscrSpace*>(b.get());
            if (aptr == bptr) return true;
            if (aptr->m_dim != bptr->m_dim) return false;
            if (aptr->driver_id != bptr->driver_id) return false;
            if (aptr->fem_id != bptr->fem_id) return false;
            if (aptr->m_shiftTetDof != bptr->m_shiftTetDof) return false;
            if (aptr->m_shiftDof != bptr->m_shiftDof) return false;
            return true;
        }
        case FemExprDescr::ComplexType:{
            auto aptr = static_cast<FemExprDescr::ComplexSpaceHelper*>(a.get());
            auto bptr = static_cast<FemExprDescr::ComplexSpaceHelper*>(b.get());
            if (aptr == bptr) return true;
            if (aptr->m_spaces.size() != bptr->m_spaces.size()) return false;
            if (aptr->driver_id != bptr->driver_id) return false;
            if (aptr->fem_id != bptr->fem_id) return false;
            if (aptr->m_dimShift != bptr->m_dimShift) return false;
            if (aptr->m_spaceNumDofTet != bptr->m_spaceNumDofTet) return false;
            if (aptr->m_spaceNumDof != bptr->m_spaceNumDof) return false;
            if (aptr->m_spaceNumDofs != bptr->m_spaceNumDofs) return false;
            if (aptr->m_spaceNumDofsTet != bptr->m_spaceNumDofsTet) return false;
            for (int i = 0; i < static_cast<int>(aptr->m_spaces.size()); ++i)
                if (!(aptr->m_spaces[i] == bptr->m_spaces[i])) return false;
            return true;
        }
    }
    return false;
}

void ElementalAssembler::VarsHelper::Clear() {
    descr = nullptr;
    initValues.clear(), base_MemOffsets.clear(), test_MemOffsets.clear();
}

void ElementalAssembler::init(ElemMatEval *_f, ElementalAssembler::Type _f_type, ElementalAssembler::SparsedDat _loc_m,
                          ElementalAssembler::SparsedDat _loc_rhs, ElemMatEval::Memory _mem,
                          std::vector<bool> *_loc_mb, VarsHelper *_vars, const INMOST::Mesh *const _m,
                          const INMOST::ElementArray<INMOST::Node>* _nodes, const INMOST::ElementArray<INMOST::Edge>* _edges, const INMOST::ElementArray<INMOST::Face>* _faces,
                          INMOST::Mesh::iteratorCell _cell, const int *_indexesC, const int* _indexesR, int *_local_edge_index,
                          int *_local_face_index
#ifndef NO_ASSEMBLER_TIMERS
                          , TimeMessures _tmes
#endif
                          ) {
    func = _f, f_type = _f_type, loc_m = _loc_m, loc_rhs = _loc_rhs, loc_mb = _loc_mb, mem = _mem;
    vars = _vars, m = _m, nodes = _nodes, edges = _edges, faces = _faces,
    cell = _cell, indexesC = _indexesC, indexesR = _indexesR,
    local_edge_index = _local_edge_index, local_face_index = _local_face_index;
#ifndef NO_ASSEMBLER_TIMERS
    m_tmes = _tmes;
#endif
    make_result_buffer();
}

ElementalAssembler::ElementalAssembler(ElemMatEval *f, ElementalAssembler::Type f_type, ElementalAssembler::SparsedDat loc_m,
                               ElementalAssembler::SparsedDat loc_rhs, ElemMatEval::Memory mem,
                               std::vector<bool> *loc_mb, VarsHelper *vars, const INMOST::Mesh *const m,
                               const INMOST::ElementArray<INMOST::Node>* nodes, const INMOST::ElementArray<INMOST::Edge>* edges, const INMOST::ElementArray<INMOST::Face>* faces,
                               INMOST::Mesh::iteratorCell cell, const int *indexesC, const int* indexesR, int *local_edge_index,
                               int *local_face_index
#ifndef NO_ASSEMBLER_TIMERS
                               , TimeMessures _tmes
#endif
                               ):
        f_type{f_type}, func{f}, loc_m{loc_m}, loc_rhs{loc_rhs}, mem{mem}, loc_mb{loc_mb},
        vars{vars}, m{m}, nodes{nodes}, edges{edges}, faces{faces}, cell{cell}, indexesC{indexesC}, indexesR{indexesR},
        local_edge_index{local_edge_index}, local_face_index{local_face_index}
#ifndef NO_ASSEMBLER_TIMERS
        , m_tmes{_tmes}
#endif
{
    make_result_buffer();
}

void ElementalAssembler::compute(const ElemMatEval::Real **args) {
    std::copy(args, args + func->n_in(), mem.m_args);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_init_user_handler += m_tmes.m_timer->elapsed_and_reset();
#endif
    func->operator()(mem.m_args, mem.m_res, mem.m_w, mem.m_iw, mem.user_data);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_comp_func += m_tmes.m_timer->elapsed_and_reset();
#endif
    densify_result();
}

void ElementalAssembler::compute(const ElemMatEval::Real **args, void* user_data) {
    std::copy(args, args + func->n_in(), mem.m_args);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_init_user_handler += m_tmes.m_timer->elapsed_and_reset();
#endif
    func->operator()(mem.m_args, mem.m_res, mem.m_w, mem.m_iw, user_data);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_comp_func += m_tmes.m_timer->elapsed_and_reset();
#endif
    densify_result();
}

void ElementalAssembler::update(const INMOST::ElementArray<INMOST::Node>& nodes) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            _nn_p[i][j] = nodes[i].Coords()[j];
        }
    }
    if (loc_m.dat)
        std::fill(loc_m.dat, loc_m.dat + loc_m.nnz, 0);
    if (loc_rhs.dat)
        std::fill(loc_rhs.dat, loc_rhs.dat + loc_rhs.nnz, 0);
}

double *ElementalAssembler::get_nodes() { return &_nn_p[0][0]; }
const double *ElementalAssembler::get_nodes() const { return &_nn_p[0][0]; }

std::ostream &ElementalAssembler::print_matrix_and_rhs_arbitrarily(std::ostream &out, const std::set<int> &sepX,
                                                               const std::set<int> &sepY) const {
    std::vector<std::string> sm, sr;
    size_t max_sm = 0, max_sr = 3;
    for (int r = 0; r < loc_m.sz1*loc_m.sz2; ++r) {
        auto i = loc_m.dat[r];
        std::string v = std::to_string(i);
        max_sm = std::max(max_sm, v.length());
        sm.emplace_back(std::move(v));
    }
    for (int r = 0; r < loc_rhs.sz1*loc_rhs.sz2; ++r) {
        auto i = loc_rhs.dat[r];
        std::string v = std::to_string(i);
        max_sr = std::max(max_sr, v.length());
        sr.emplace_back(std::move(v));
    }
    max_sm++; max_sr++;
    std::string y_delim = " | ";
    int lda = func->out_size1(0);
    if (sepY.count(0) != 0) {
        for (unsigned i = 0; i < lda * max_sm + max_sr + y_delim.size() * sepX.size(); ++i)
            out << "-";
        out << "\n";
    }
    for (int i = 0; i < lda; ++i){
        if (sepX.count(0) != 0){
            out << y_delim;
        }
        for (int j = 0; j < lda; ++j) {
            for (int dif = max_sm - sm[j * lda + i].length(); dif > 0; --dif)
                out << " ";
            out << sm[j * lda + i];
            if (sepX.count(j+1) != 0){
                out << y_delim;
            }
        }
        out << ":";
        for (int dif = max_sm - (sr.empty() ? 0 : sr[i].length()); dif > 0; --dif)
            out << " ";
        if (!sr.empty()) out << sr[i];
        if (sepX.count(lda+1) != 0){
            out << y_delim;
        }
        out << "\n";
        if (sepY.count(i+1) != 0) {
            for (unsigned i = 0; i < lda * max_sm + max_sr + y_delim.size() * sepX.size(); ++i)
                out << "-";
            out << "\n";
        }
    }
    return out;
}

void ElementalAssembler::print_input(const double **args) const {
    std::cout << "N = " << func->n_in() << std::endl;
    for (unsigned i = 0; i < func->n_in() ; ++i) {
        int sz = func->in_size1(i) * func->in_size2(i);
        std::cout << i << " - " << sz << ": ";
        for (int j = 0; j < sz; ++j)
            std::cout << args[i][j] << " ";
        std::cout << std::endl;
    }
}

void ElementalAssembler::make_result_buffer() {
    auto _res_buf = mem.m_res;
    switch (f_type) {
        case RHS:  _res_buf[0] = loc_rhs.dat + loc_rhs.sz1*loc_rhs.sz2 - loc_rhs.nnz; break;
        case MAT:  _res_buf[0] = loc_m.dat + loc_m.sz1*loc_m.sz2 - loc_m.nnz; break;
        case MAT_RHS: {
            _res_buf[1] = loc_rhs.dat + loc_rhs.sz1*loc_rhs.sz2 - loc_rhs.nnz;
            _res_buf[0] = loc_m.dat + loc_m.sz1*loc_m.sz2 - loc_m.nnz;
            break;
        }
        default: assert("Wrong f_type");
    }
}

void ElementalAssembler::densify_result() {
    auto densify = [](SparsedDat dat, ElemMatEval::Real* st){
        for (int i = 0; i < dat.sz2; ++i) {
            int k = 0;
            for (int j = dat.colind[i]; j < dat.colind[i + 1]; ++j) {
                for (; k < dat.row[j]; ++k)
                    dat.dat[i * dat.sz1 + k] = 0;
                dat.dat[i * dat.sz1 + dat.row[j]] = st[j];
                ++k;
            }
            for (; k < dat.sz1; ++k)
                dat.dat[i * dat.sz1 + k] = 0;
        }
    };
    auto densify_with_template = [](SparsedDat dat, ElemMatEval::Real* st, std::vector<bool>* loc_mb){
        for (int i = 0; i < dat.sz2; ++i) {
            int k = 0;
            for (int j = dat.colind[i]; j < dat.colind[i + 1]; ++j) {
                for (; k < dat.row[j]; ++k)
                    dat.dat[i * dat.sz1 + k] = 0, (*loc_mb)[i * dat.sz1 + k] = false;
                dat.dat[i * dat.sz1 + dat.row[j]] = st[j], (*loc_mb)[i * dat.sz1 + dat.row[j]] = true;
                ++k;
            }
            for (; k < dat.sz1; ++k)
                dat.dat[i * dat.sz1 + k] = 0, (*loc_mb)[i * dat.sz1 + k] = false;
        }
    };
    auto _res_buf = mem.m_res;
    switch (f_type) {
        case RHS:{
            if (loc_rhs.colind != nullptr && loc_rhs.row != nullptr && loc_rhs.sz1*loc_rhs.sz2 > loc_rhs.nnz){
                densify(loc_rhs, _res_buf[0]);
            }
            break;
        }
        case MAT:{
            if (loc_m.colind != nullptr && loc_m.row != nullptr && loc_m.sz1*loc_m.sz2 > loc_m.nnz){
                densify_with_template(loc_m, _res_buf[0], loc_mb);
            } else {
                std::fill(loc_mb->begin(), loc_mb->end(), true);
            }
            break;
        }
        case MAT_RHS:{
            if (loc_rhs.colind != nullptr && loc_rhs.row != nullptr && loc_rhs.sz1*loc_rhs.sz2 > loc_rhs.nnz){
                densify(loc_rhs, _res_buf[1]);
            }
            if (loc_m.colind != nullptr && loc_m.row != nullptr && loc_m.sz1*loc_m.sz2 > loc_m.nnz){
                densify_with_template(loc_m, _res_buf[0], loc_mb);
            } else {
                std::fill(loc_mb->begin(), loc_mb->end(), true);
            }
            break;
        }
        default:
            throw std::runtime_error("Faced unknown ElementalAssembler::Type");
    }
}

int FemExprDescr::BaseDiscrSpaceHelper::NumDof() const {
    int res = 0;
    for (int t = 0; t < NGEOM_TYPES; ++t)
        res += NumDof(static_cast<GeomType>(t));
    return res;
}

int FemExprDescr::BaseDiscrSpaceHelper::NumDofOnTet() const {
    int res = 0;
    for (int t = 0; t < NGEOM_TYPES; ++t)
        res += NumDofOnTet(static_cast<GeomType>(t));
    return res;
}

FemExprDescr::LocalOrder FemExprDescr::BaseDiscrSpaceHelper::LocalOrderOnTet(FemExprDescr::GeomType t, int ldof) const {
    return LocalOrderOnTet(TetDofID(t, ldof));
}

FemExprDescr::GeomType FemExprDescr::BaseDiscrSpaceHelper::TypeOnTet(int dof) const {
    return LocalOrderOnTet(dof).etype;
}

std::array<int, FemExprDescr::NGEOM_TYPES> FemExprDescr::BaseDiscrSpaceHelper::NumDofsOnTet() const {
    std::array<int, NGEOM_TYPES> res;
    for (int t = 0; t < NGEOM_TYPES; ++t){
        auto tt = static_cast<GeomType>(t);
        res[t] = NumDofOnTet(tt);
    }
    return res;
}

std::array<bool, FemExprDescr::NGEOM_TYPES> FemExprDescr::BaseDiscrSpaceHelper::GetGeomMask() const {
    std::array<bool, NGEOM_TYPES> res;
    for (int t = 0; t < NGEOM_TYPES; ++t){
        auto tt = static_cast<GeomType>(t);
        res[t] = NumDofOnTet(tt) > 0;
    }
    return res;
}

FemExprDescr::ShiftedSpaceHelperView FemExprDescr::BaseDiscrSpaceHelper::GetNestedComponent(const int* ext_dims, int ndims) const { 
    ShiftedSpaceHelperView view; 
    if (ndims < 0) return view;
    if (ndims == 0){
        view.base = this;
        return view; 
    }
    GetNestedComponent(ext_dims, ndims, view);
    return view;
}

FemExprDescr::UniteDiscrSpace::UniteDiscrSpace(int dim, std::array<int, NGEOM_TYPES> NumDofs, unsigned int driver_id,
                                               unsigned int fem_id) :
        m_dim(dim), driver_id(driver_id), fem_id(fem_id)
{
    m_shiftDof[0] = m_shiftTetDof[0] = 0;
    std::array<int, 6> scales = {4, 6, 4, 1, 6, 4};
    for (int i = 0; i < NGEOM_TYPES; ++i) {
        m_shiftTetDof[i + 1] += m_shiftTetDof[i] + scales[i] * NumDofs[i];
        m_shiftDof[i + 1] += m_shiftDof[i] + NumDofs[i];
    }
}

FemExprDescr::GeomType FemExprDescr::UniteDiscrSpace::Type(int dof) const {
    assert(dof < NumDof() && dof >= 0 && "Wrong dof number");
    return static_cast<GeomType>(std::upper_bound(m_shiftDof.data(), m_shiftDof.data() + m_shiftDof.size(), dof) - m_shiftDof.data() - 1);
}

FemExprDescr::LocalOrder FemExprDescr::UniteDiscrSpace::LocalOrderOnTet(int dof) const {
    assert(dof < NumDofOnTet() && dof >= 0 && "Wrong dof number");
    LocalOrder lo;
    lo.etype = static_cast<GeomType>(std::upper_bound(m_shiftTetDof.data(), m_shiftTetDof.data() + m_shiftTetDof.size(), dof) - m_shiftTetDof.data() - 1);
    lo.nelem = (dof - m_shiftTetDof[lo.etype]) / NumDof(lo.etype);
    lo.loc_elem_dof_id  = (dof - m_shiftTetDof[lo.etype]) % NumDof(lo.etype);
    lo.gid = dof;
    return lo;
}

FemExprDescr::LocalOrder FemExprDescr::UniteDiscrSpace::LocalOrderOnTet(FemExprDescr::GeomType t, int ldof) const {
    assert(ldof < NumDofOnTet(t) && ldof >= 0 && "Wrong dof number");
    LocalOrder lo;
    lo.etype = t;
    lo.nelem = ldof / NumDof(t);
    lo.loc_elem_dof_id  = ldof % NumDof(t);
    lo.gid = m_shiftTetDof[t] + ldof;
    return lo;
}

FemExprDescr::GeomType FemExprDescr::UniteDiscrSpace::TypeOnTet(int dof) const {
    assert(dof < NumDofOnTet() && dof >= 0 && "Wrong dof number");
    return static_cast<GeomType>(std::upper_bound(m_shiftTetDof.data(), m_shiftTetDof.data() + m_shiftTetDof.size(), dof) - m_shiftTetDof.data() - 1);
}

void FemExprDescr::UniteDiscrSpace::GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const { 
    (void) ext_dims;
    if (ndims == 0)
        view.base = this;
    else 
        view.Clear(); 
}

int FemExprDescr::SimpleVectorDiscrSpace::TetDofID(FemExprDescr::GeomType t, int ldof) const {
    int nOdf = base->NumDofOnTet(), lOdf = base->NumDofOnTet(t);
    int component = ldof / lOdf;
    return component * nOdf + base->TetDofID(t, ldof % lOdf);
}

FemExprDescr::LocalOrder FemExprDescr::SimpleVectorDiscrSpace::LocalOrderOnTet(int dof) const {
    assert(dof < NumDofOnTet() && dof >= 0 && "Wrong dof number");
    LocalOrder lo = base->LocalOrderOnTet(dof % base->NumDofOnTet());
    lo.loc_elem_dof_id += (dof / base->NumDofOnTet()) * base->NumDof(lo.etype);
    lo.gid = dof;
    return lo;
}

const FemExprDescr::SimpleDiscrSpace* FemExprDescr::SimpleVectorDiscrSpace::GetComponent(int dim) const {
    int bdim = base->Dim();
    int vdim = dim / bdim, ldim = dim % bdim;
    return (vdim >= 0 && vdim < m_dim) ? base->GetComponent(ldim) : nullptr; 
}

void FemExprDescr::SimpleVectorDiscrSpace::GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const{ 
    if (ndims == 0){
        view.base = this;
        return;
    } else {
        assert(ndims >= 1 && "Wrong ndims");
        if (ext_dims[0] < 0 || ext_dims[0] >= m_dim) {
            view.Clear();
            return;
        }
        auto dsz = base->NumDofsOnTet();
        for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDofTet[i] += ext_dims[0]*dsz[i];
        base->GetNestedComponent(ext_dims+1, ndims-1, view);
    } 
}

FemExprDescr::ComplexSpaceHelper::ComplexSpaceHelper(std::vector<std::shared_ptr<BaseDiscrSpaceHelper>> spaces,
                                                     unsigned int driver_id, unsigned int fem_id) :
        m_spaces{std::move(spaces)}, driver_id(driver_id), fem_id(fem_id) {
    for (int t = 0; t < NGEOM_TYPES; ++t) {
        m_spaceNumDofTet[t].resize(m_spaces.size()+1);
        m_spaceNumDof[t].resize(m_spaces.size()+1);
        m_spaceNumDofTet[t][0] = m_spaceNumDof[t][0] = 0;
        for (unsigned i = 0; i < m_spaces.size(); ++i) {
            m_spaceNumDofTet[t][i+1] = m_spaceNumDofTet[t][i] + m_spaces[i]->NumDofOnTet(static_cast<GeomType>(t));
            m_spaceNumDof[t][i+1] = m_spaceNumDof[t][i] + m_spaces[i]->NumDof(static_cast<GeomType>(t));
        }
    }
    m_dimShift.resize(m_spaces.size()+1);
    m_spaceNumDofs.resize(m_spaces.size()+1);
    m_spaceNumDofsTet.resize(m_spaces.size()+1);
    m_spaceNumDofsTet[0] = m_spaceNumDofs[0] = m_dimShift[0] = 0;
    for (unsigned i = 0; i < m_spaces.size(); ++i) {
        m_dimShift[i+1] = m_dimShift[i] + m_spaces[i]->Dim();
        m_spaceNumDofs[i+1] = m_spaceNumDofs[i] + m_spaces[i]->NumDof();
        m_spaceNumDofsTet[i+1] = m_spaceNumDofsTet[i] + m_spaces[i]->NumDofOnTet();
    }
}

FemExprDescr::GeomType FemExprDescr::ComplexSpaceHelper::Type(int dof) const {
    assert(dof < BaseDiscrSpaceHelper::NumDof() && dof >= 0 && "Wrong dof number");
    int vid = std::upper_bound(m_spaceNumDofs.data(), m_spaceNumDofs.data() + m_spaceNumDofs.size(), dof) - m_spaceNumDofs.data() - 1;
    return m_spaces[vid]->Type(dof - m_spaceNumDofs[vid]);
}

FemExprDescr::GeomType FemExprDescr::ComplexSpaceHelper::TypeOnTet(int dof) const {
    assert(dof < BaseDiscrSpaceHelper::NumDofOnTet() && dof >= 0 && "Wrong dof number");
    int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof) - m_spaceNumDofsTet.data() - 1;
    return m_spaces[vid]->TypeOnTet(dof - m_spaceNumDofsTet[vid]);
}

FemExprDescr::LocalOrder FemExprDescr::ComplexSpaceHelper::LocalOrderOnTet(int dof) const {
    assert(dof < BaseDiscrSpaceHelper::NumDofOnTet() && dof >= 0 && "Wrong dof number");
    int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof) - m_spaceNumDofsTet.data() - 1;
    LocalOrder lo = m_spaces[vid]->LocalOrderOnTet(dof - m_spaceNumDofsTet[vid]);
    lo.loc_elem_dof_id += m_spaceNumDofTet[lo.etype][vid];
    lo.gid = dof;
    return lo;
}

int FemExprDescr::ComplexSpaceHelper::TetDofID(FemExprDescr::GeomType t, int ldof) const {
    int vid = std::upper_bound(m_spaceNumDofTet[t].data(), m_spaceNumDofTet[t].data() + m_spaceNumDofTet[t].size(), ldof) - m_spaceNumDofTet[t].data() - 1;
    return m_spaceNumDofsTet[vid] + m_spaces[vid]->TetDofID(t, ldof - m_spaceNumDofTet[t][vid]);
}

FemExprDescr::LocalOrder FemExprDescr::ComplexSpaceHelper::LocalOrderOnTet(FemExprDescr::GeomType t, int ldof) const {
    long vid = std::upper_bound(m_spaceNumDofTet[t].data(), m_spaceNumDofTet[t].data() + m_spaceNumDofTet[t].size(), ldof) - m_spaceNumDofTet[t].data() - 1;
    LocalOrder lo = m_spaces[vid]->LocalOrderOnTet(t, ldof - m_spaceNumDofTet[t][vid]);
    lo.loc_elem_dof_id += m_spaceNumDofTet[t][vid];
    lo.gid += m_spaceNumDofsTet[vid];
    return lo;
}

const FemExprDescr::SimpleDiscrSpace *FemExprDescr::ComplexSpaceHelper::GetComponent(int dim) const {
    long vid = std::upper_bound(m_dimShift.data(), m_dimShift.data() + m_dimShift.size(), dim) - m_dimShift.data() - 1;
    return m_spaces[vid]->GetComponent(dim - m_dimShift[vid]);
}

void FemExprDescr::ComplexSpaceHelper::GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const { 
    if (ndims == 0){
        view.base = this;
        return;
    } else {
        assert(ndims >= 1 && "Wrong ndims");
        if (ext_dims[0] < 0 || ext_dims[0] >= static_cast<int>(m_spaces.size())) {
            view.Clear();
            return;
        }
        for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDofTet[i] += m_spaceNumDofTet[i][ext_dims[0]];
        m_spaces[ext_dims[0]]->GetNestedComponent(ext_dims+1, ndims-1, view);
    } 
}

FemExprDescr::LocalOrder FemExprDescr::ShiftedSpaceHelperView::LocalOrderOnTet(int dof) const {
    LocalOrder lo = base->LocalOrderOnTet(dof);
    lo.loc_elem_dof_id += m_shiftNumDofTet[lo.etype];
    return lo;
} 

FemExprDescr::LocalOrder FemExprDescr::ShiftedSpaceHelperView::LocalOrderOnTet(GeomType t, int ldof) const {
    LocalOrder lo = base->LocalOrderOnTet(t, ldof);
    lo.loc_elem_dof_id += m_shiftNumDofTet[lo.etype];
    return lo;
}

std::array<bool, FemExprDescr::NGEOM_TYPES> FemExprDescr::GetBaseGeomMask() const {
    std::array<bool, NGEOM_TYPES> res;
    std::fill(res.begin(), res.end(), false);
    for (unsigned i = 0; i < base_funcs.size(); ++i)
        for (int t = 0; t < NGEOM_TYPES; ++t){
            if (res[t]) continue;
            if (base_funcs[i].odf->DefinedOn(static_cast<GeomType>(t))) res[t] = true;
        }
    return res;
}

std::array<int, FemExprDescr::NGEOM_TYPES> FemExprDescr::NumDofs() const {
    std::array<int, NGEOM_TYPES> res = {0};
    for (unsigned i = 0; i < base_funcs.size(); ++i)
        for (int t = 0; t < NGEOM_TYPES; ++t){
            auto tt = static_cast<GeomType>(t);
            res[t] += base_funcs[i].odf->NumDof(tt);
        }
    return res;
}

std::array<int, FemExprDescr::NGEOM_TYPES> FemExprDescr::NumDofsOnTet() const {
    std::array<int, NGEOM_TYPES> res = {0};
    for (unsigned i = 0; i < base_funcs.size(); ++i)
        for (int t = 0; t < NGEOM_TYPES; ++t){
            auto tt = static_cast<GeomType>(t);
            res[t] += base_funcs[i].odf->NumDofOnTet(tt);
        }
    return res;
}

int FemExprDescr::NumDofOnTet() const {
    int res = 0;
    for (unsigned i = 0; i < base_funcs.size(); ++i)
        res += base_funcs[i].odf->NumDofOnTet();
    return res;
}

INMOST::Storage::real ElementalAssembler::TakeElementDOF(const INMOST::Tag& tag, const FemExprDescr::LocalOrder& lo, 
    const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
    const int local_face_index[4], const int local_edge_index[6], const int local_node_index[4]){
    switch (lo.etype){
        case FemExprDescr::NODE: return nodes[local_node_index[lo.nelem]].RealArray(tag)[lo.loc_elem_dof_id];
        case FemExprDescr::EDGE: return edges[local_edge_index[lo.nelem]].RealArray(tag)[lo.loc_elem_dof_id];
        case FemExprDescr::FACE: return faces[local_face_index[lo.nelem]].RealArray(tag)[lo.loc_elem_dof_id];
        case FemExprDescr::CELL: return cell.RealArray(tag)[lo.loc_elem_dof_id];
        default: throw std::runtime_error("Faced unknown GeomType");
    }
}

std::function<INMOST::Storage::real(
    const INMOST::Tag&, const FemExprDescr::LocalOrder&, 
    const INMOST::Cell&, const INMOST::ElementArray<INMOST::Face>&, const INMOST::ElementArray<INMOST::Edge>&, const INMOST::ElementArray<INMOST::Node>&, 
    const int* /*local_face_index[4]*/, const int* /*local_edge_index[6]*/, const int* /*local_node_index[4]*/ )> 
    ElementalAssembler::GeomTakerDOF(FemExprDescr::GeomType t){
    switch (t) {
        case FemExprDescr::NODE: 
            return []( const INMOST::Tag& tag, const FemExprDescr::LocalOrder& lo, 
                const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
                const int* local_face_index/*[4]*/, const int* local_edge_index/*[6]*/, const int* local_node_index/*[4]*/){
                (void) cell; (void) faces; (void) edges; (void) local_face_index; (void) local_edge_index;  
                return nodes[local_node_index[lo.nelem]].RealArray(tag)[lo.loc_elem_dof_id];
            };
        case FemExprDescr::EDGE: 
            return []( const INMOST::Tag& tag, const FemExprDescr::LocalOrder& lo, 
                const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
                const int* local_face_index/*[4]*/, const int* local_edge_index/*[6]*/, const int* local_node_index/*[4]*/){
                (void) cell; (void) faces; (void) nodes; (void) local_face_index; (void) local_node_index; 
                return edges[local_edge_index[lo.nelem]].RealArray(tag)[lo.loc_elem_dof_id];
            };
        case FemExprDescr::FACE: 
            return []( const INMOST::Tag& tag, const FemExprDescr::LocalOrder& lo, 
                const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
                const int* local_face_index/*[4]*/, const int* local_edge_index/*[6]*/, const int* local_node_index/*[4]*/){
                (void) cell; (void) nodes; (void) edges; (void) local_node_index; (void) local_edge_index;     
                return faces[local_face_index[lo.nelem]].RealArray(tag)[lo.loc_elem_dof_id];
            };
        case FemExprDescr::CELL: 
            return []( const INMOST::Tag& tag, const FemExprDescr::LocalOrder& lo, 
                const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
                const int* local_face_index/*[4]*/, const int* local_edge_index/*[6]*/, const int* local_node_index/*[4]*/){
                (void) nodes; (void) faces; (void) edges; (void) local_face_index; (void) local_edge_index; (void) local_node_index;
                return cell.RealArray(tag)[lo.loc_elem_dof_id];
            };
        default: throw std::runtime_error("Faced unknown GeomType");    
    }
}