//
// Created by Liogky Alexey on 28.03.2022.
//

#include "elemental_assembler.h"

using namespace Ani; 

void ElementalAssembler::make_result_buffer() {
    switch (f_type) {
        case RHS:  mem.m_res[0] = loc_rhs.m_dat; break;
        case MAT:  mem.m_res[0] = loc_m.m_dat; break;
        case MAT_RHS: mem.m_res[0] = loc_m.m_dat, mem.m_res[1] = loc_rhs.m_dat; break;
        default: assert("Wrong f_type");
    }
}

void ElementalAssembler::densify_result(){
    switch (f_type) {
        case RHS:{
            loc_rhs.densify(loc_rhs.m_dat);
            break;
        }
        case MAT:{
            loc_m.densify(loc_m.m_dat); 
            loc_m.m_sp.fillTemplate<decltype(loc_mb->begin()), bool>(loc_mb->begin()); 
            break;
        }
        case MAT_RHS:{
            loc_rhs.densify(loc_rhs.m_dat);
            loc_m.densify(loc_m.m_dat);
            loc_m.m_sp.fillTemplate<decltype(loc_mb->begin()), bool>(loc_mb->begin());
            break;
        }
        default:
            throw std::runtime_error("Faced unknown ElementalAssembler::UsageType");
    }
}

void ElementalAssembler::init(MatFuncWrap<>* _f, UsageType _f_type,
              SparsedData<> _loc_m, SparsedData<> _loc_rhs, MatFuncWrap<>::Memory _mem, std::vector<bool> * _loc_mb,
              VarsHelper* _vars, const Mesh * const _m,
              const ElementArray<Node>* _nodes, const ElementArray<Edge>* _edges, const ElementArray<Face>* _faces, const Cell* _cell,
              const uchar* _node_permutation,
#ifndef NO_ASSEMBLER_TIMERS
              TimeMessures _tmes,
#endif
              DynMem<Real, Int>* _pool)
{
    func = _f, f_type = _f_type, loc_m = _loc_m, loc_rhs = _loc_rhs, mem = _mem, loc_mb = _loc_mb;
    vars = _vars, m = _m, nodes = _nodes, edges = _edges, faces = _faces,
    cell = _cell, node_permutation = _node_permutation;
#ifndef NO_ASSEMBLER_TIMERS
    m_tmes = _tmes;
#endif
    pool = _pool;
    make_result_buffer();
}

ElementalAssembler::ElementalAssembler(MatFuncWrap<>* f, UsageType f_type,
            SparsedData<> loc_m, SparsedData<> loc_rhs, MatFuncWrap<>::Memory mem, std::vector<bool> *loc_mb,
            VarsHelper* vars, const Mesh * const m,
            const ElementArray<Node>* nodes, const ElementArray<Edge>* edges, const ElementArray<Face>* faces,
            const Cell* cell,
            const uchar* node_permutation,
#ifndef NO_ASSEMBLER_TIMERS
            TimeMessures tmes,
#endif
            DynMem<Real, Int>* pool
            ):
        func{f}, f_type{f_type}, loc_m{loc_m}, loc_rhs{loc_rhs}, mem{mem}, loc_mb{loc_mb},
        vars{vars}, m{m}, nodes{nodes}, edges{edges}, faces{faces}, cell{cell}, 
        node_permutation{node_permutation}, pool{pool}
#ifndef NO_ASSEMBLER_TIMERS
        , m_tmes{tmes}
#endif
{
    make_result_buffer();
}

void ElementalAssembler::compute(const Real **args) {
    std::copy(args, args + func->n_in(), mem.m_args);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_init_user_handler += m_tmes.m_timer->elapsed_and_reset();
#endif
    func->operator()(mem.m_args, mem.m_res, mem.m_w, mem.m_iw, mem.user_data, mem.mem_id);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_comp_func += m_tmes.m_timer->elapsed_and_reset();
#endif
    densify_result();
}

void ElementalAssembler::compute(const Real **args, void* user_data) {
    std::copy(args, args + func->n_in(), mem.m_args);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_init_user_handler += m_tmes.m_timer->elapsed_and_reset();
#endif
    func->operator()(mem.m_args, mem.m_res, mem.m_w, mem.m_iw, user_data, mem.mem_id);
#ifndef NO_ASSEMBLER_TIMERS
    *m_tmes.m_time_comp_func += m_tmes.m_timer->elapsed_and_reset();
#endif
    densify_result();
}

void ElementalAssembler::update() {
    INMOST::Storage::real_array crds;
    for (int i = 0; i < 4; ++i) {
        crds = (*nodes)[i].Coords();
        for (int j = 0; j < 3; ++j) {
            _nn_p[i][j] = crds[j];
        }
    }
    if (loc_m.m_dat)
        std::fill(loc_m.m_dat, loc_m.m_dat + loc_m.m_sp.m_nnz, 0);
    if (loc_rhs.m_dat)
        std::fill(loc_rhs.m_dat, loc_rhs.m_dat + loc_rhs.m_sp.m_nnz, 0);
}
double* ElementalAssembler::get_nodes() { return &_nn_p[0][0]; }
const double* ElementalAssembler::get_nodes() const { return &_nn_p[0][0]; }
std::ostream& ElementalAssembler::print_matrix_and_rhs_arbitrarily(std::ostream &out, const std::set<int> &sepX,
                                                               const std::set<int> &sepY) const {
    std::vector<std::string> sm, sr;
    size_t max_sm = 0, max_sr = 3;
    for (int r = 0; r < loc_m.m_sp.m_sz1*loc_m.m_sp.m_sz2; ++r) {
        auto i = loc_m.m_dat[r];
        std::string v = std::to_string(i);
        max_sm = std::max(max_sm, v.length());
        sm.emplace_back(std::move(v));
    }
    for (int r = 0; r < loc_rhs.m_sp.m_sz1*loc_rhs.m_sp.m_sz2; ++r) {
        auto i = loc_rhs.m_dat[r];
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