//
// Created by Liogky Alexey on 16.01.2023.
//

#ifndef CARNUM_FEMSPACE_INL
#define CARNUM_FEMSPACE_INL

namespace Ani{
    template<typename FUNCTOR>
    inline int BaseFemSpace::FunctorContainer<FUNCTOR>::eval_functor(const std::array<double, 3>& X, double* res, uint dim, void* user_data){
        auto& fc = *static_cast<FunctorContainer<FUNCTOR>*>(user_data);
        return fc.f(X, res, dim, fc.user_data);
    }
    template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type Dummy>
    inline void BaseFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const {
        FunctorContainer<EVAL_FUNCTOR> fc{f, user_data};
        interpolateOnDOF(XYZ, FunctorContainer<EVAL_FUNCTOR>::eval_functor, udofs, idof_on_tet, fusion, mem, &fc, max_quad_order);
    }
    template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type Dummy>
    inline void BaseFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<>* udofs, int idof_on_tet, int fusion, DynMem<>& wmem, void* user_data, uint max_quad_order) const {
        auto req = interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        interpolateOnDOF(XYZ, f, udofs, idof_on_tet, fusion, mem.m_mem, user_data, max_quad_order);
    }
    template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type Dummy>
    inline void BaseFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, int idof_on_tet, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const {
        interpolateOnDOF(XYZ, f, &udofs, idof_on_tet, 1, mem, user_data, max_quad_order);
    }
    template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type Dummy>
    inline void BaseFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, int idof_on_tet, DynMem<>& wmem, void* user_data, uint max_quad_order) const{
        auto req = interpolateOnDOF_mem_req(idof_on_tet, 1, max_quad_order);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        interpolateOnDOF(XYZ, f, udofs, idof_on_tet, mem.m_mem, user_data, max_quad_order);
    }
    template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type Dummy>
    inline void BaseFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const {
        FunctorContainer<EVAL_FUNCTOR> fc{f, user_data};
        interpolateByDOFs(XYZ, FunctorContainer<EVAL_FUNCTOR>::eval_functor, udofs, sp, mem, &fc, max_quad_order);
    }
    template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type Dummy>
    inline void BaseFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, DynMem<>& wmem, void* user_data, uint max_quad_order) const {
        auto req = interpolateByDOFs_mem_req(max_quad_order);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        interpolateByDOFs(XYZ, f, udofs, sp, mem.m_mem, user_data, max_quad_order);
    }
    template<typename EVAL_FUNCTOR>
    inline void BaseFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const {
        FunctorContainer<EVAL_FUNCTOR> fc{f, user_data};
        for (auto it = sub_map.beginBySparsity(sp); it != sub_map.endBySparsity(); ++it)
            interpolateOnDOF(XYZ, FunctorContainer<EVAL_FUNCTOR>::eval_functor, udofs, it->gid, mem, &fc, max_quad_order);
    }
    template<typename EVAL_FUNCTOR>
    inline void BaseFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map, DynMem<>& wmem, void* user_data, uint max_quad_order) const {
        auto req = interpolateByDOFs_mem_req(max_quad_order);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        interpolateByDOFs(XYZ, f, udofs, sp, sub_map, mem.m_mem,user_data, max_quad_order);
    }
    inline void BaseFemSpace::interpolateConstant(double val, ArrayView<> udofs, const TetGeomSparsity& sp) const {
        for (auto it = m_order.beginBySparsity(sp); it != m_order.endBySparsity(); ++it)
            udofs[it->gid] = val;
    }
    inline void BaseFemSpace::interpolateConstant(double val, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map) const {
        for (auto it = sub_map.beginBySparsity(sp); it != sub_map.endBySparsity(); ++it)
            udofs[it->gid] = val;
    }
    inline uint BaseFemSpace::orderDiff() const {
        auto ord = order();
        auto mx = std::numeric_limits<uint>::max();
        return ord < mx ? (ord > 0 ? ord - 1 : 0) : mx;
    } 

    template<typename ...Targs>
    OpMemoryRequirements ComplexFemSpace::memOP_internal(uint nquadpoints, uint fusion, OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const {
        OpMemoryRequirements mem_req;
        mem_req.mtx_parts = 0;
        for (auto& s: m_spaces){
            auto lreq = (s.get()->*req)(nquadpoints, fusion, ts...);//s->memIDEN(nquadpoints, fusion);
            mem_req.Usz += lreq.Usz;
            mem_req.extraRsz = std::max(mem_req.extraRsz, lreq.extraRsz); 
            mem_req.extraIsz = std::max(mem_req.extraIsz, lreq.extraIsz);
            mem_req.mtx_parts += lreq.mtx_parts;
        }
        return mem_req;
    }
    template<typename ...Targs>
    BandDenseMatrixX<> ComplexFemSpace::applyOP_internal(AniMemoryX<> &mem, ArrayView<> &U, 
        BandDenseMatrixX<>(Ani::BaseFemSpace::* apl)(AniMemoryX<> &, ArrayView<> &, Targs...) const,
        OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const {
        std::size_t Ushift = 0;
        BandDenseMatrixX<> res;
        {
            auto& s = m_spaces[0];
            auto lreq = (s.get()->*req)(mem.q, mem.f, ts...);//s->memIDEN();
            ArrayView<> lU(U.data + Ushift, lreq.Usz);
            BandDenseMatrixX<> lmtx = (s.get()->*apl)(mem, lU, ts...);//s->applyIDEN(mem, lU);
            Ushift += lreq.Usz; 
            res = lmtx;
        }
        int endCol = res.stCol[res.nparts], endRow = res.stRow[res.nparts];
        for (uint i = 1; i < m_spaces.size(); ++i){
            auto& s = m_spaces[i];
            auto lreq = (s.get()->*req)(mem.q, mem.f, ts...);//s->memIDEN();
            ArrayView<> lU(U.data + Ushift, lreq.Usz);
            BandDenseMatrixX<> lmtx = (s.get()->*apl)(mem, lU, ts...);//s->applyIDEN(mem, lU);
            Ushift += lreq.Usz;
            for (std::size_t i = 1; i <= lmtx.nparts; ++i){
                res.stCol[res.nparts + i] = endCol + lmtx.stCol[i];
                res.stRow[res.nparts + i] = endRow + lmtx.stRow[i];
            }
            res.stCol[res.nparts] = endCol, res.stRow[res.nparts] = endRow;
            res.nparts += lmtx.nparts;
            endCol = res.stCol[res.nparts], endRow = res.stRow[res.nparts];
        }
        return res;
    }
    
    template<typename ...Targs>
    OpMemoryRequirements UnionFemSpace::memOP_internal(uint gdim, uint nquadpoints, uint fusion, OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const {
        OpMemoryRequirements mem_req;
        mem_req.mtx_parts = 0;
        for (auto& s: m_subsets){
            auto lreq = (s.get()->*req)(nquadpoints, fusion, ts...);//s->memIDEN(nquadpoints, fusion);
            mem_req.extraRsz = std::max(mem_req.extraRsz, lreq.extraRsz + lreq.Usz); 
            mem_req.extraIsz = std::max(mem_req.extraIsz, lreq.extraIsz);
            mem_req.mtx_parts += lreq.mtx_parts; 
        }
        mem_req.Usz += 2*gdim*nquadpoints*fusion*m_order.NumDofOnTet();

        return mem_req;
    }
    template<typename ...Targs>
    BandDenseMatrixX<> UnionFemSpace::applyOP_internal(uint gdim, AniMemoryX<> &mem, ArrayView<> &U, 
        BandDenseMatrixX<>(Ani::BaseFemSpace::* apl)(AniMemoryX<> &, ArrayView<> &, Targs...) const,
        OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const {
        auto st_busy = mem.busy_mtx_parts;
        auto nfa = m_order.NumDofOnTet();
        DenseMatrix<> llU(U.data, mem.q * gdim, mem.f*nfa);
        DenseMatrix<> lU(U.data + mem.q * gdim * mem.f*nfa, mem.q * gdim, mem.f*nfa);
        lU.SetZero(); llU.SetZero();
        int nfa_shift = 0; 
        for (uint v = 0; v < m_subsets.size(); ++v){
            OpMemoryRequirements lreq = (m_subsets[v].get()->*req)(mem.q, mem.f, ts...);
            mem.extraR.size -= lreq.Usz; 
            ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz);  
            BandDenseMatrixX<> Vx = (m_subsets[v].get()->*apl)(mem, Vm, ts...);
            auto lnfa = m_subsets[v]->m_order.NumDofOnTet();
            for (std::size_t p = 0; p < Vx.nparts; ++p){
                int nCol = Vx.stCol[p+1] - Vx.stCol[p], nRow = (Vx.stRow[p+1] - Vx.stRow[p]);
                for (std::size_t r = 0; r < mem.f; ++r)
                for (int i = 0; i < nCol; ++i)
                for (std::size_t n = 0; n < mem.q; ++n)
                for (int k = 0; k < nRow; ++k)
                    lU(k+Vx.stRow[p] + gdim * n, nfa_shift + i + Vx.stCol[p] + nfa*r) = Vx.data[p](k+nRow*n, i + nCol*r);
            }
            mem.extraR.size += lreq.Usz;  
            nfa_shift += lnfa;
        }
        for (std::size_t r = 0; r < mem.f; ++r)
            for (uint i = 0; i < nfa; ++i)
                for (uint j = 0; j < nfa; ++j) if (m_orth_coefs[j + nfa * i] != 0.0){
                    for (unsigned k = 0; k < mem.q * gdim; ++k)
                        llU(k, i + nfa*r) += m_orth_coefs[j + nfa * i]*lU(k, j + nfa*r);
                }

        uint mtxishift = st_busy > 0 ? 1 : 0;
        BandDenseMatrixX<> bres(1, mem.MTX.data + st_busy, mem.MTXI_ROW.data + st_busy + mtxishift, mem.MTXI_COL.data + st_busy + mtxishift);
        mem.busy_mtx_parts = st_busy + 1;
        bres.data[0] = llU;
        bres.stRow[0] = 0; bres.stRow[1] = gdim;
        bres.stCol[0] = 0; bres.stCol[1] = nfa;
        return bres;
    }
    template<typename ...Targs>
    inline uint UnionFemSpace::internal_take_max_val(uint(Ani::BaseFemSpace::* take_val)(Targs...) const, Targs... ts) const{
        uint res = 0;
        for (auto& s: m_subsets) 
            res = std::max(res, (s.get()->*take_val)(ts...));
        return res;
    }

    inline static FemSpace make_union_raw(const std::vector<FemSpace>& spaces){
        std::vector<std::shared_ptr<BaseFemSpace>> m_all;
        m_all.reserve(spaces.size());
        for (auto& s: spaces){
            m_all.push_back(s.base());
        }
        return FemSpace(UnionFemSpace(m_all));
    }
    inline static FemSpace make_union_with_simplification(const std::vector<FemSpace>& spaces){
        std::vector<std::shared_ptr<BaseFemSpace>> m_all;
        m_all.reserve(spaces.size());
        for (auto& s: spaces){
            if (s.gatherType() == BaseFemSpace::BaseTypes::UnionType)
                m_all.insert(m_all.end(), s.target<UnionFemSpace>()->m_subsets.begin(), s.target<UnionFemSpace>()->m_subsets.end());
            else 
                m_all.push_back(s.base());   
        }
        return FemSpace(UnionFemSpace(m_all));
    }
    inline static FemSpace make_complex_raw(const std::vector<FemSpace>& spaces){
        std::vector<std::shared_ptr<BaseFemSpace>> m_all;
        m_all.reserve(spaces.size());
        for (auto& s: spaces){
            m_all.push_back(s.base());
        }
        return FemSpace(ComplexFemSpace(m_all));
    }
    inline static FemSpace make_complex_with_simplification(const std::vector<FemSpace>& spaces){
        std::vector<std::shared_ptr<BaseFemSpace>> m_all;
        m_all.reserve(spaces.size());
        for (auto& s: spaces){
            if (s.gatherType() == BaseFemSpace::BaseTypes::ComplexType)
                m_all.insert(m_all.end(), s.target<ComplexFemSpace>()->m_spaces.begin(), s.target<ComplexFemSpace>()->m_spaces.end());
            else 
                m_all.push_back(s.base());   
        }
        return FemSpace(ComplexFemSpace(m_all));
    }
};

#endif //CARNUM_FEMSPACE_INL