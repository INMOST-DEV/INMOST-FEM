#include "fem_space.h"
#include "spaces/poly_0.h"
#include "operations/operations.h"
#include <stdexcept>
#include <sstream>

namespace Ani{
    void BaseFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, DynMem<>& wmem, void* user_data, uint max_quad_order) const{
        auto req = interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        interpolateOnDOF(XYZ, f, udofs, idof_on_tet, fusion, mem.m_mem, user_data, max_quad_order);
    }
    void BaseFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc &f, ArrayView<> udofs, const TetGeomSparsity& sp, DynMem<>& wmem, void* user_data, uint max_quad_order) const{
        auto req = interpolateByDOFs_mem_req(max_quad_order);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        interpolateByDOFs(XYZ, f, udofs, sp, mem.m_mem, user_data, max_quad_order);
    }
    BandDenseMatrixX<> BaseFemSpace::applyOP(OperatorType op, AniMemoryX<> &mem, ArrayView<> &U) const {
        switch (op) {
            case IDEN: return applyIDEN(mem, U);
            case GRAD: return applyGRAD(mem, U);
            case DIV : return applyDIV (mem, U);
            case CURL: return applyCURL(mem, U);
            case DUDX: return applyDUDX(mem, U, 0);
            case DUDY: return applyDUDX(mem, U, 1);
            case DUDZ: return applyDUDX(mem, U, 2);
            default:
                throw std::runtime_error("Faced unknown FEM linear operator");
                break;
        }
        return BandDenseMatrixX<>();
    }
    OpMemoryRequirements BaseFemSpace::memOP(OperatorType op, uint nquadpoints, uint fusion) const {
        switch (op) {
            case IDEN: return memIDEN(nquadpoints, fusion);
            case GRAD: return memGRAD(nquadpoints, fusion);
            case DIV : return memDIV (nquadpoints, fusion);
            case CURL: return memCURL(nquadpoints, fusion);
            case DUDX: return memDUDX(nquadpoints, fusion, 0);
            case DUDY: return memDUDX(nquadpoints, fusion, 1);
            case DUDZ: return memDUDX(nquadpoints, fusion, 2);
            default:
                throw std::runtime_error("Faced unknown FEM linear operator");
        }
        return OpMemoryRequirements();
    }
    uint BaseFemSpace::dimOP(OperatorType op) const {
        switch (op) {
            case IDEN: 
            case CURL: 
            case DUDX: 
            case DUDY: 
            case DUDZ: return dim();
            case GRAD: return dim()*3;
            case DIV:  return dim()/3;
            default:
                throw std::runtime_error("Faced unknown FEM linear operator");
                break;
        }
        return 0;
    }
    uint BaseFemSpace::orderOP(OperatorType op) const{
        switch (op) {
            case IDEN: return orderIDEN();
            case CURL: return orderCURL();
            case DUDX: return orderDUDX(0);
            case DUDY: return orderDUDX(1);
            case DUDZ: return orderDUDX(2);
            case GRAD: return orderGRAD();
            case DIV:  return orderDIV();
            default:
                throw std::runtime_error("Faced unknown FEM linear operator");
                break;
        }
        return order();
    }
    ApplyOpFromSpaceView BaseFemSpace::getOP(OperatorType op) const {
        return ApplyOpFromSpaceView(op, this);
    }
    PlainMemoryX<> BaseFemSpace::interpolateOnRegion_mem_req(uchar elem_type, uint max_quad_order) const{
        if (elem_type == DofT::NODE){
            DofT::TetGeomSparsity sp;
            sp.setNode(0);
            PlainMemoryX<> res;
            for (auto it = m_order.beginBySparsity(sp); it != m_order.endBySparsity(); ++it){
                auto lres = interpolateOnDOF_mem_req(it->gid, max_quad_order);
                res.dSize = std::max(res.dSize, lres.dSize);
                res.iSize = std::max(res.iSize, lres.iSize);
                res.mSize = std::max(res.mSize, lres.mSize);
            }
            return res;
        }
        uint nfa = m_order.NumDofOnTet();
        uint nquads = 0;
        auto geom_tp = DofT::DimToGeomType(DofT::GeomTypeDim(elem_type));
        switch(geom_tp) {
            case DofT::EDGE:{
                nquads = segment_quadrature_formulas(max_quad_order).GetNumPoints();
                break;
            }
            case DofT::FACE:{
                nquads = triangle_quadrature_formulas(max_quad_order).GetNumPoints();
                break;
            }
            case DofT::CELL:{
                nquads = tetrahedron_quadrature_formulas(max_quad_order).GetNumPoints();
                break; 
            }
            default: {}
        }
        PlainMemoryX<> mem_req;
        mem_req.dSize += nfa*nfa + nfa + 5*nquads;
        auto aplIDEN = getOP(IDEN);
        auto aplP0 = P0Space().getOP(IDEN);
        PlainMemoryX<> pnt_mem0 = fem3DpntL_memory_requirements<DfuncTraits<TENSOR_NULL>>(aplIDEN, aplIDEN, nquads, 1);
        PlainMemoryX<> pnt_mem1 = fem3DpntL_memory_requirements<DfuncTraits<TENSOR_GENERAL>>(aplP0, aplIDEN, nquads, 1);
        mem_req.dSize += std::max({pnt_mem0.dSize, pnt_mem1.dSize, static_cast<decltype(pnt_mem0.dSize)>(nfa*(nfa+1)/2)});
        mem_req.iSize += std::max({pnt_mem0.iSize, pnt_mem1.iSize, static_cast<decltype(pnt_mem0.dSize)>(nfa)});
        mem_req.mSize = std::max({pnt_mem0.mSize, pnt_mem1.mSize});
        return mem_req;
    }
    void BaseFemSpace::interpolateOnRegion(const Tetra<const double>& XYZ, const EvalFunctor& f, ArrayView<> udofs, uchar elem_type, int ielem, PlainMemoryX<> mem_req, void* user_data, uint max_quad_order) const{
        assert(mem_req.ge(interpolateOnRegion_mem_req(elem_type, max_quad_order)) && "Not enough of working memory");
        DofT::TetGeomSparsity sp;
        if (elem_type == DofT::NODE){
            sp.setNode(ielem);
            interpolateByDOFs(XYZ, f, udofs, sp, mem_req, user_data, max_quad_order);
            return;
        }
        uint nfa = m_order.NumDofOnTet();
        uint nquads = 0;
        auto geom_tp = DofT::DimToGeomType(DofT::GeomTypeDim(elem_type));
        auto aplIDEN = getOP(IDEN);
        auto aplP0 = P0Space().getOP(IDEN);
        auto d_alloc = [&mem_req](int sz)->double*{ 
            assert(mem_req.dSize >= static_cast<std::size_t>(sz) && "Not enough real memory");
            mem_req.dSize -= sz;
            auto res = mem_req.ddata;
            mem_req.ddata += sz;
            return res;    
        };
        auto i_alloc = [&mem_req](int sz)->int*{ 
            assert(mem_req.iSize >= static_cast<std::size_t>(sz) && "Not enough int memory");
            mem_req.iSize -= sz;
            auto res = mem_req.idata;
            mem_req.idata += sz;
            return res;    
        };
        double* am = d_alloc(nfa*nfa), *fm = d_alloc(nfa);
        double* xyl = nullptr, *wg = nullptr;
        switch(geom_tp) {
            case DofT::EDGE:{
                sp.setEdge(ielem, true);
                const static std::array<char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
                auto formula = segment_quadrature_formulas(max_quad_order);
                nquads = formula.GetNumPoints();
                xyl = d_alloc(4*nquads), wg = d_alloc(nquads);
                for (uint i = 0; i < nquads; ++i){
                    auto q = formula.GetPointWeight(i);
                    xyl[4*i + lookup_nds[2*ielem]] = q.p[0]; 
                    xyl[4*i + lookup_nds[2*ielem+1]] = q.p[1];
                    wg[i] = q.w;
                }
                break;
            }
            case DofT::FACE:{
                sp.setFace(ielem, true);
                auto formula = triangle_quadrature_formulas(max_quad_order);
                nquads = formula.GetNumPoints();
                xyl = d_alloc(4*nquads), wg = d_alloc(nquads);
                for (uint i = 0; i < nquads; ++i){
                    auto q = formula.GetPointWeight(i);
                    xyl[4*i + (ielem+0)%4] = q.p[0]; 
                    xyl[4*i + (ielem+1)%4] = q.p[1];
                    xyl[4*i + (ielem+2)%4] = q.p[2];
                    wg[i] = q.w;
                }
                break;
            }
            case DofT::CELL:{
                sp.setCell(true); 
                auto formula = tetrahedron_quadrature_formulas(max_quad_order);
                nquads = formula.GetNumPoints();
                xyl = d_alloc(4*nquads), wg = d_alloc(nquads);
                auto pp = formula.GetPointData(), wp = formula.GetWeightData();
                std::copy(pp, pp + 4 * nquads, xyl);
                std::copy(wp, wp + nquads, wg);
                break; 
            }
            default:
                throw std::runtime_error("Faced unknown geometric region");
        }
        DenseMatrix<> A(am, nfa, nfa), F(fm, nfa, 1);
        ArrayView<> XYL(xyl, nquads*4), WG(wg, nquads);
        fem3DpntL<DfuncTraits<TENSOR_NULL>>(XYZ, XYL, WG, aplIDEN, aplIDEN, TensorNull<>, A, mem_req, nullptr);
        fem3DpntL<DfuncTraits<TENSOR_GENERAL>>(XYZ, XYL, WG, aplP0, aplIDEN, 
            [&f](const Coord<>& x, double* Dmem, TensorDims Ddims, void* user_data, int iTet)->TensorType{
                (void) iTet;
                f(x, Dmem, Ddims.first, user_data);
                return TENSOR_GENERAL;
            }, F, mem_req, user_data);
        int* ids = i_alloc(nfa);
        auto& dof_map = m_order;
        int nids = 0;
        for (auto it = dof_map.beginBySparsity(sp); it != dof_map.endBySparsity(); ++it)
            ids[nids++] = it->gid;
        if (nids == 0) return;    
        std::sort(ids, ids + nids);
        for (int k = 0; k < nids; ++k)
            F(k, 0) = F(ids[k], 0);
        for (int j = 0; j < nids; ++j)
            for (int i = 0; i < nids; ++i)
                A[i + j * nids] = A(ids[i], ids[j]);
        A.Init(am, nids, nids, nfa*nfa);
        F.Init(fm, nids, 1, nfa);
        if (mem_req.dSize < static_cast<decltype(mem_req.dSize)>(nfa*(nfa+1)/2) )
            throw std::runtime_error("Not enough real memory");
        cholesky_solve(A.data, F.data, nids, 1, F.data, mem_req.ddata);
        for (int k = 0; k < nids; ++k)
            udofs[ids[k]] = F[k];        
    }

    void BaseFemSpace::evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, std::vector<Expr>& phi) const{
        phi.resize(m_order.NumDofOnTet());
        for(auto& i: phi) i = lmb.zeroes();
        evalBasisFunctions(lmb, grad_lmb, phi.data());
    }

    BandDenseMatrixX<> BaseFemSpace::applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const {
        if (dim() != 3)
            throw std::runtime_error("DIV operator supported only for 3-dimensional variables");
        auto st_busy = mem.busy_mtx_parts;
        OpMemoryRequirements lreq = memGRAD(mem.q, mem.f);
        mem.extraR.size -= lreq.Usz;    
        ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz);    
        BandDenseMatrixX<> Vx = applyGRAD(mem, Vm); 
        auto nfa = m_order.NumDofOnTet();
        DenseMatrix<> lU(U.data, mem.q * 1, mem.f*nfa);
        lU.SetZero();
        for (std::size_t p = 0; p < Vx.nparts; ++p){
            bool have_val_0 = (0 >= Vx.stRow[p] && 0 < Vx.stRow[p+1]);
            bool have_val_4 = (4 >= Vx.stRow[p] && 4 < Vx.stRow[p+1]);
            bool have_val_8 = (8 >= Vx.stRow[p] && 8 < Vx.stRow[p+1]);
            int nd = (have_val_0 ? 1 : 0) + (have_val_4 ? 1 : 0) + (have_val_8 ? 1 : 0);
            int nCol = Vx.stCol[p+1] - Vx.stCol[p], nRow = (Vx.stRow[p+1] - Vx.stRow[p]);
            if (nd > 0){
                for (std::size_t r = 0; r < mem.f; ++r)
                for (int i = 0; i < nCol; ++i)
                for (std::size_t n = 0; n < mem.q; ++n){
                    if (have_val_0)
                        lU(0 + n, Vx.stCol[p] + i + nfa*r) += Vx.data[p](0 - Vx.stRow[p] + nRow * n, i + nCol * r);
                    if (have_val_4)
                        lU(0 + n, Vx.stCol[p] + i + nfa*r) += Vx.data[p](4 - Vx.stRow[p] + nRow * n, i + nCol * r);
                    if (have_val_8)
                        lU(0 + n, Vx.stCol[p] + i + nfa*r) += Vx.data[p](8 - Vx.stRow[p] + nRow * n, i + nCol * r);
                }
            }
        }
        mem.extraR.size += lreq.Usz;    
        mem.busy_mtx_parts = st_busy + 1;
        Vx.nparts = 1;
        Vx.data[0] = lU;
        Vx.stRow[1] = 3;
        Vx.stCol[1] = nfa;
        return Vx;
    }
    OpMemoryRequirements BaseFemSpace::memDIV(uint nquadpoints, uint fusion) const {
        if (dim() != 3)
            throw std::runtime_error("DIV operator supported only for 3-dimensional variables");
        auto lreq = memGRAD(nquadpoints, fusion);
        lreq.extraRsz += lreq.Usz; 
        lreq.Usz = 1*nquadpoints*fusion*m_order.NumDofOnTet();
        return lreq;
    }
    BandDenseMatrixX<> BaseFemSpace::applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const {
        if (dim() != 3)
            throw std::runtime_error("CURL operator supported only for 3-dimensional variables");
        static const unsigned char dlookup[] = {7, 5,  2, 6,  3, 1};
        auto st_busy = mem.busy_mtx_parts;
        OpMemoryRequirements lreq = memGRAD(mem.q, mem.f);
        mem.extraR.size -= lreq.Usz;    
        ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz); 
        BandDenseMatrixX<> Vx = applyGRAD(mem, Vm); 
        auto nfa = m_order.NumDofOnTet();
        DenseMatrix<> lU(U.data, mem.q * 3, mem.f*nfa);
        lU.SetZero();
        for (std::size_t p = 0; p < Vx.nparts; ++p){
            bool have_val = (Vx.stRow[p+1]  - Vx.stRow[p] > 1 || (Vx.stRow[p] != 0 && Vx.stRow[p] != 4 && Vx.stRow[p] != 8));
            if (have_val){ 
                std::array<bool, 6> dim_trace;
                for (int k = 0; k < 6; ++k)
                    dim_trace[k] = (dlookup[k] >= Vx.stRow[p] && dlookup[k] < Vx.stRow[p+1]);       
                int nCol = Vx.stCol[p+1] - Vx.stCol[p], nRow = (Vx.stRow[p+1] - Vx.stRow[p]);
                
                for (std::size_t r = 0; r < mem.f; ++r)
                for (int i = Vx.stCol[p]; i < Vx.stCol[p+1]; ++i)
                for (std::size_t n = 0; n < mem.q; ++n)
                for (int k = 0; k < 3; ++k){
                    if (dim_trace[2*k + 0])
                        lU(k + 3*n, i + nfa*r) += Vx.data[p](dlookup[2*k + 0] - Vx.stRow[p] + nRow * n, (i-Vx.stCol[p]) + nCol * r);
                    if (dim_trace[2*k + 1])
                        lU(k + 3*n, i + nfa*r) -= Vx.data[p](dlookup[2*k + 1] - Vx.stRow[p] + nRow * n, (i-Vx.stCol[p]) + nCol * r);
                }
            }
        }
        mem.extraR.size += lreq.Usz; 
        mem.busy_mtx_parts = st_busy + 1;
        Vx.nparts = 1;
        Vx.data[0] = lU;
        Vx.stRow[1] = 3;
        Vx.stCol[1] = nfa;
        return Vx;
    }
    OpMemoryRequirements BaseFemSpace::memCURL(uint nquadpoints, uint fusion) const {
        if (dim() != 3)
            throw std::runtime_error("CURL operator supported only for 3-dimensional variables");
        auto lreq = memGRAD(nquadpoints, fusion);
        lreq.extraRsz += lreq.Usz;  
        lreq.Usz = 3U*nquadpoints*fusion*m_order.NumDofOnTet();
        return lreq;
    }
    BandDenseMatrixX<> BaseFemSpace::applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const {
        auto st_busy = mem.busy_mtx_parts;
        OpMemoryRequirements lreq = memGRAD(mem.q, mem.f);
        mem.extraR.size -= lreq.Usz;    
        ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz); 
        BandDenseMatrixX<> Vx = applyGRAD(mem, Vm);
        auto align_to_3 = [k](auto a, bool eq_or_greater = true){
            auto q = a / 3, r = a % 3;
            if (eq_or_greater){
                if (r <= k) return 3*q + k;
                else return 3*(q+1) + k;
            } else { //eq_or_less
                if (r < k) return 3*q + k - 3;
                else return 3*q + k;
            }
        };
        int upart = 0, ushift = 0;
        for (std::size_t p = 0; p < Vx.nparts; ++p){
            int dst = align_to_3(Vx.stRow[p], true), dend = align_to_3(Vx.stRow[p+1]-1, false);
            int nd = ( (dend >= dst) ? (dend - dst) / 3 + 1 : 0);
            int nCol = Vx.stCol[p+1] - Vx.stCol[p], nRow = (Vx.stRow[p+1] - Vx.stRow[p]);
            int rowShift = (dst - Vx.stRow[p]);
            if (nd > 0){
                DenseMatrix<> lU(U.data + ushift, mem.q * nd, mem.f*(Vx.stCol[p+1] - Vx.stCol[p]));
                for (std::size_t r = 0; r < mem.f; ++r)
                for (int i = 0; i < nCol; ++i)
                for (std::size_t n = 0; n < mem.q; ++n)
                for (int id = 0; id < nd; ++id)
                    lU(id + nd*n, i + nCol*r) = Vx.data[p](rowShift + 3*id + nRow * n, i + nCol * r);

                Vx.data[upart++] = lU;
                Vx.stRow[upart] = nd + Vx.stRow[upart-1];
                Vx.stCol[upart] = nCol + Vx.stCol[upart-1];
                ushift += mem.q * nd * mem.f*nCol;
            }
        }
        mem.extraR.size += lreq.Usz; 
        Vx.nparts = upart;
        mem.busy_mtx_parts = st_busy + upart;
        return Vx;
    }
    OpMemoryRequirements BaseFemSpace::memDUDX(uint nquadpoints, uint fusion, uchar k) const{
        (void) k;
        auto lreq = memGRAD(nquadpoints, fusion);
        lreq.extraRsz += lreq.Usz;  
        lreq.Usz = std::min(static_cast<decltype(lreq.Usz)>(dim()*nquadpoints*fusion*m_order.NumDofOnTet()), lreq.Usz);
        return lreq;
    }
    void BaseFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc&f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const {
        for (auto it = m_order.beginBySparsity(sp); it != m_order.endBySparsity(); ++it)
            interpolateOnDOF(XYZ, f, udofs, it->gid, mem, user_data, max_quad_order);
    }
    PlainMemoryX<> BaseFemSpace::interpolateByDOFs_mem_req(uint max_quad_order) const {
        PlainMemoryX<> res;
        for (auto it = m_order.begin(); it != m_order.end(); ++it){
            auto lres = interpolateOnDOF_mem_req(it->gid, max_quad_order);
            res.dSize = std::max(res.dSize, lres.dSize);
            res.iSize = std::max(res.iSize, lres.iSize);
            res.mSize = std::max(res.mSize, lres.mSize);
        }
        return res;
    }

    VectorFemSpace& VectorFemSpace::Init(int dim, std::shared_ptr<BaseFemSpace> space){
        m_base = std::move(space);
        m_order = pow(m_base->m_order, dim);
        return *this;
    }
    std::shared_ptr<BaseFemSpace> VectorFemSpace::subSpace(const int* ext_dims, int ndims) const { 
        int vdim =  m_order.target<DofT::VectorDofMap>()->m_dim;
        if (ndims >= 1 && ext_dims[0] < vdim && ext_dims[0] >= 0)
            return ndims == 1 ? m_base : m_base->subSpace(ext_dims+1, ndims - 1);
        return nullptr;      
    }
    std::string VectorFemSpace::typeName() const { 
        return "Vector(" + 
        std::to_string(m_base ? m_order.target<DofT::VectorDofMap>()->m_dim : 0) + ", " + 
        (m_base ? m_base->typeName() : std::string("NULL")) + 
        ")"; 
    }
    bool VectorFemSpace::operator==(const BaseFemSpace& other) const {
        if (gatherType() != other.gatherType()) return false;
        auto& o = *static_cast<const VectorFemSpace*>(&other);
        return (o.m_base.get() == m_base.get()) || (o.m_base && m_base && *o.m_base == *m_base); 
    }
    void VectorFemSpace::evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const{
        uint lnfa = m_base->m_order.NumDofOnTet();
        uint ldim = m_order.target<DofT::VectorDofMap>()->m_dim;
        m_base->evalBasisFunctions(lmb, grad_lmb, phi);
        for (uint d = 1; d < ldim; ++d)
            std::copy(phi, phi + lnfa, phi + lnfa*d);
    }    
    PlainMemoryX<> VectorFemSpace::interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const{
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        auto ldim = m_base->dim();
        auto comp_id = m_order.target<DofT::VectorDofMap>()->ComponentID(idof_on_tet);
        uint ldof_id = comp_id.cgid;
        PlainMemoryX<> res;
        res = m_base->interpolateOnDOF_mem_req(ldof_id, fusion, max_quad_order);
        res.dSize += fusion*vdim*ldim;
        return res;
    }
    void VectorFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const{
        assert(mem.ge(interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order)) && "Not enough of work memory");
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        auto lnfa = m_base->m_order.NumDofOnTet();
        auto ldim = m_base->dim();
        auto comp_id = m_order.target<DofT::VectorDofMap>()->ComponentID(idof_on_tet);
        uint nvar = comp_id.part_id, ldof_id = comp_id.cgid;
        for (int r = 0; r < fusion; ++r)
            udofs[r] = ArrayView<>(udofs[r].data + nvar*lnfa, udofs[r].size - nvar*lnfa);
        double* wmem = mem.ddata; mem.ddata += fusion*vdim*ldim; mem.dSize -= fusion*vdim*ldim;
        auto narrowing = [&f, nvar, ldim, fusion, gdim = vdim*ldim, wmem](const std::array<double, 3>& X, double* res, uint dim, void* user_data)->int{
            assert(dim == ldim*fusion && "Wrong expected dimension");
            (void) dim;
            f(X, wmem, gdim*fusion, user_data);
            for (int r = 0; r < fusion; ++r)
                std::copy(wmem + nvar*ldim + gdim*r, wmem + (nvar+1)*ldim + gdim*r, res + r*ldim);
            return 0;
        };  
        FunctorContainer<decltype(narrowing)> fc{narrowing, user_data};
        m_base->interpolateOnDOF(XYZ, FunctorContainer<decltype(narrowing)>::eval_functor, udofs, ldof_id, fusion, mem, &fc, max_quad_order);
        for (int r = 0; r < fusion; ++r)
            udofs[r] = ArrayView<>(udofs[r].data - nvar*lnfa, udofs[r].size + nvar*lnfa);    
    }
    PlainMemoryX<> VectorFemSpace::interpolateByDOFs_mem_req(uint max_quad_order) const {
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        auto ldim = m_base->dim();
        PlainMemoryX<> res;
        for (auto it = m_base->m_order.begin(); it != m_base->m_order.end(); ++it){
            auto lres = m_base->interpolateOnDOF_mem_req(it->gid, vdim, max_quad_order);
            res.extend_size(lres);
        }
        res.dSize += vdim*ldim;
        res.mSize += (sizeof(ArrayView<>)*vdim) / sizeof(DenseMatrix<>) + 1 + std::max(1UL, sizeof(ArrayView<>)/sizeof(DenseMatrix<>));
        return res; 
    }
    void VectorFemSpace::interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const{
        assert(mem.ge(interpolateByDOFs_mem_req(max_quad_order)) && "Not enough work memory");
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        auto lnfa = m_base->m_order.NumDofOnTet();
        ArrayView<>* new_udofs = nullptr;
        {
            void* storage = mem.mdata;
            std::size_t remain = mem.mSize * sizeof(DenseMatrix<>);
            void* p = std::align(alignof(ArrayView<>), sizeof(ArrayView<>)*vdim, storage, remain);
            assert(p != nullptr && "Error arised while memory allocation");
            std::size_t rem1 = remain + sizeof(DenseMatrix<>) - (static_cast<char*>(p) - static_cast<char*>(storage));
            mem.mdata = static_cast<DenseMatrix<>*>(std::align(alignof(DenseMatrix<>), sizeof(DenseMatrix<>), p, rem1));
            assert(mem.mdata != nullptr && "Error arised while matrix memory allocation");
            mem.mSize -= mem.mdata - static_cast<DenseMatrix<>*>(storage);
            new_udofs = static_cast<ArrayView<>*>(p);
        }
        for (int i = 0; i < vdim; ++i)
            new_udofs[i] = ArrayView<>(udofs.data + i*lnfa, lnfa);
        
        for (auto it = m_base->m_order.beginBySparsity(sp); it != m_base->m_order.endBySparsity(); ++it){
            m_base->interpolateOnDOF(XYZ, f, new_udofs, it->gid, vdim, mem, user_data, max_quad_order);
        }
    }
    void VectorFemSpace::realloc_rep_mtx(AniMemoryX<> &mem, BandDenseMatrixX<>& mtx, uint nrep){
        uint vdim = nrep;
        BandDenseMatrixX<>& res = mtx;
        mem.busy_mtx_parts += (vdim-1)*res.nparts;
        auto shiftCol = res.stCol[res.nparts], shiftRow = res.stRow[res.nparts];
        for (uint d = 1; d < vdim; ++d){
            std::copy(res.data, res.data + res.nparts, res.data + d*res.nparts);
            std::copy(res.stCol, res.stCol + res.nparts, res.stCol + d*res.nparts);
            std::for_each(res.stCol + d*res.nparts, res.stCol + (d+1)*res.nparts, [shift = shiftCol * d](auto& x) { x+= shift; });
            res.stCol[(d+1)*res.nparts] = shiftCol * (d+1); 
            std::copy(res.stRow, res.stRow + res.nparts, res.stRow + d*res.nparts);
            std::for_each(res.stRow + d*res.nparts, res.stRow + (d+1)*res.nparts, [shift = shiftRow * d](auto& x) { x+= shift; });
            res.stRow[(d+1)*res.nparts] = shiftRow * (d+1);   
        }
        res.nparts *= vdim;
    }
    BandDenseMatrixX<> VectorFemSpace::applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const{
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        BandDenseMatrixX<> res = m_base->applyIDEN(mem, U);
        realloc_rep_mtx(mem, res, vdim);
        return res;
    }
    OpMemoryRequirements VectorFemSpace::memIDEN(uint nquadpoints, uint fusion) const{
        OpMemoryRequirements res = m_base->memIDEN(nquadpoints, fusion);
        res.mtx_parts *= m_order.target<DofT::VectorDofMap>()->m_dim;
        return res;
    }
    BandDenseMatrixX<> VectorFemSpace::applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const{
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        BandDenseMatrixX<> res = m_base->applyGRAD(mem, U);
        realloc_rep_mtx(mem, res, vdim);
        return res;
    }
    OpMemoryRequirements VectorFemSpace::memGRAD(uint nquadpoints, uint fusion) const{
        OpMemoryRequirements res = m_base->memGRAD(nquadpoints, fusion);
        res.mtx_parts *= m_order.target<DofT::VectorDofMap>()->m_dim;
        return res;
    }
    BandDenseMatrixX<> VectorFemSpace::applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const{
        if (dim() != 3)
            throw std::runtime_error("DIV operator supported only for 3-dimensional variables");
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        if (vdim == 3){
            auto base_req = m_base->memGRAD(mem.q, mem.f);
            mem.extraR.size -= base_req.Usz; 
            ArrayView<> Vm(mem.extraR.data + mem.extraR.size, base_req.Usz); 
            auto st_busy = mem.busy_mtx_parts;
            BandDenseMatrixX<> Vx = m_base->applyGRAD(mem, Vm);
            auto lnfa = m_base->m_order.NumDofOnTet();
            DenseMatrix<> lU(U.data, mem.q, 3*lnfa*mem.f, U.size);
            if (Vx.nparts == 1){
                DenseMatrix<> V = Vx.data[0];
                for (std::size_t r = 0; r < mem.f; ++r)
                    for (uint i = 0; i < lnfa; ++i)
                        for (std::size_t n = 0; n < mem.q; ++n)
                            for (int k = 0; k < 3; ++k)
                                lU(n, i + lnfa*(k + 3 * r)) = V(k + 3*n, i + lnfa * r);
                Vx.data[0] = lU;
                Vx.stRow[1] = 1;
                Vx.stCol[1] = lnfa*3;                 
            } else {
                mem.extraR.size += base_req.Usz; 
                throw std::runtime_error("Result of applying GRAD on 1d variable should have nparts = 1");
            }
            mem.extraR.size += base_req.Usz; 
            mem.busy_mtx_parts = st_busy + 1;
            return Vx;
        } else if (vdim == 1){
            return m_base->applyDIV(mem, U);
        } else 
            throw std::runtime_error("Object has internal data inconsistency");
        return  BandDenseMatrixX<>();     
    }
    OpMemoryRequirements VectorFemSpace::memDIV(uint nquadpoints, uint fusion) const{
        if (dim() != 3)
            throw std::runtime_error("DIV operator supported only for 3-dimensional variables");
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        if (vdim == 3){
            auto res = m_base->memGRAD(nquadpoints, fusion);
            res.extraRsz += res.Usz; 
            res.Usz = 3*nquadpoints*fusion*m_base->m_order.NumDofOnTet(); 
            return res;
        } else if (vdim == 1){
            return m_base->memDIV(nquadpoints, fusion);
        } else 
            throw std::runtime_error("Object has internal data inconsistency"); 
        return OpMemoryRequirements();      
    }
    BandDenseMatrixX<> VectorFemSpace::applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const {
        if (dim() != 3)
            throw std::runtime_error("CURL operator supported only for 3-dimensional variables");
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        if (vdim == 3){
            auto base_req = m_base->memGRAD(mem.q, mem.f);
            mem.extraR.size -= base_req.Usz; 
            ArrayView<> Vm(mem.extraR.data + mem.extraR.size, base_req.Usz); 
            auto st_busy = mem.busy_mtx_parts;
            BandDenseMatrixX<> Vx = m_base->applyGRAD(mem, Vm);
            auto lnfa = m_base->m_order.NumDofOnTet();
            DenseMatrix<> lU(U.data, 3*mem.q, 3*lnfa*mem.f, U.size);
            static const unsigned char IJL[] = {1,2,0,  2,1,0,  0,2,1, 2,0,1,  0,1,2,  1,0,2};
            std::fill(U.data, U.data + 3*mem.q*3*lnfa*mem.f, 0);
            if (Vx.nparts == 1){
                DenseMatrix<> V = Vx.data[0];
                for (std::size_t r = 0; r < mem.f; ++r)
                    for (std::size_t n = 0; n < mem.q; ++n){
                        auto i = IJL[3*0 + 0], j = IJL[3*0 + 1], l = IJL[3*0 + 2];
                        for (uint d = 0; d < lnfa; ++d)
                            lU(i + 3*n, d + lnfa * (l + 3 * r)) = V(j + n*3, d + lnfa*r);
                        i = IJL[3*1 + 0], j = IJL[3*1 + 1], l = IJL[3*1 + 2];
                        for (uint d = 0; d < lnfa; ++d)
                            lU(i + 3*n, d + lnfa * (l + 3 * r)) = -V(j + n*3, d + lnfa*r);
                        i = IJL[3*2 + 0], j = IJL[3*2 + 1], l = IJL[3*2 + 2];
                        for (uint d = 0; d < lnfa; ++d)
                            lU(i + 3*n, d + lnfa * (l + 3 * r)) = -V(j + n*3, d + lnfa*r);
                        i = IJL[3*3 + 0], j = IJL[3*3 + 1], l = IJL[3*3 + 2];
                        for (uint d = 0; d < lnfa; ++d)
                            lU(i + 3*n, d + lnfa * (l + 3 * r)) = V(j + n*3, d + lnfa*r);
                        i = IJL[3*4 + 0], j = IJL[3*4 + 1], l = IJL[3*4 + 2];
                        for (uint d = 0; d < lnfa; ++d)
                            lU(i + 3*n, d + lnfa * (l + 3 * r)) = V(j + n*3, d + lnfa*r);
                        i = IJL[3*5 + 0], j = IJL[3*5 + 1], l = IJL[3*5 + 2];
                        for (uint d = 0; d < lnfa; ++d)
                            lU(i + 3*n, d + lnfa * (l + 3 * r)) = -V(j + n*3, d + lnfa*r);
                    }
                Vx.data[0] = lU;
                Vx.stRow[1] = 3;
                Vx.stCol[1] = lnfa*3;                 
            } else {
                mem.extraR.size += base_req.Usz; 
                throw std::runtime_error("Result of applying GRAD on 1d variable should have nparts = 1");
            }
            mem.extraR.size += base_req.Usz; 
            mem.busy_mtx_parts = st_busy + 1;
            return Vx;
        } else if (vdim == 1) {
            return m_base->applyCURL(mem, U);
        } else 
            throw std::runtime_error("Object has internal data inconsistency");
        return  BandDenseMatrixX<>();   
    }
    OpMemoryRequirements VectorFemSpace::memCURL(uint nquadpoints, uint fusion) const {
        if (dim() != 3)
            throw std::runtime_error("CURL operator supported only for 3-dimensional variables");
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        if (vdim == 3){
            auto res = m_base->memGRAD(nquadpoints, fusion);
            res.extraRsz += res.Usz; 
            res.Usz = 9*nquadpoints*fusion*m_base->m_order.NumDofOnTet(); 
            return res;
        } else if (vdim == 1) {
            return m_base->memCURL(nquadpoints, fusion);
        } else 
            throw std::runtime_error("Object has internal data inconsistency");  
        return OpMemoryRequirements();      
    }
    BandDenseMatrixX<> VectorFemSpace::applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const {
        auto vdim = m_order.target<DofT::VectorDofMap>()->m_dim;
        BandDenseMatrixX<> res = m_base->applyDUDX(mem, U, k);
        realloc_rep_mtx(mem, res, vdim);
        return res;
    }
    OpMemoryRequirements VectorFemSpace::memDUDX(uint nquadpoints, uint fusion, uchar k) const {
        OpMemoryRequirements res = m_base->memDUDX(nquadpoints, fusion, k);
        res.mtx_parts *= m_order.target<DofT::VectorDofMap>()->m_dim;
        return res;
    }
    ComplexFemSpace& ComplexFemSpace::Init(std::vector<std::shared_ptr<BaseFemSpace>> spaces){
        m_spaces = std::move(spaces);
        setup();
        return *this;
    }
    std::shared_ptr<BaseFemSpace> ComplexFemSpace::subSpace(const int* ext_dims, int ndims) const{ 
        int vdim =  m_spaces.size();
        if (ndims >= 1 && ext_dims[0] < vdim && ext_dims[0] >= 0)
            return ndims == 1 ? m_spaces[ext_dims[0]] : m_spaces[ext_dims[0]]->subSpace(ext_dims+1, ndims - 1);
        return nullptr;      
    }
    std::string ComplexFemSpace::typeName() const{
        std::stringstream oss;
        oss << "Complex("; 
        for (uint i = 0; i < m_spaces.size(); ++i){
            oss << m_spaces[i]->typeName();
            if (i != m_spaces.size() - 1) 
                oss << ", ";
        }
        oss << ")";
        return oss.str(); 
    }
    bool ComplexFemSpace::operator==(const BaseFemSpace& other) const {
        if (gatherType() != other.gatherType()) return false;
        auto& a = *static_cast<const ComplexFemSpace*>(&other);
        if (m_spaces.size() != a.m_spaces.size()) return false;
        for (uint i = 0; i < m_spaces.size(); ++i){
            if (m_spaces[i].get() == a.m_spaces[i].get()) continue;
            if (!(m_spaces[i].get() && a.m_spaces[i].get() && *m_spaces[i].get() == *a.m_spaces[i].get())) return false;
        }
        return true; 
    }
    void ComplexFemSpace::evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const{
        for (uint i = 0, shift = 0; i < m_spaces.size(); ++i){
            m_spaces[i]->evalBasisFunctions(lmb, grad_lmb, phi + shift);
            shift += m_spaces[i]->m_order.NumDofOnTet();
        }
    } 
    PlainMemoryX<> ComplexFemSpace::interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const{
        auto comp = m_order.target<DofT::ComplexDofMap>()->ComponentID(idof_on_tet);
        uint ldof_id = comp.cgid, gdim = dim();
        auto res = m_spaces[comp.part_id]->interpolateOnDOF_mem_req(ldof_id, fusion, max_quad_order);
        res.dSize += fusion*gdim;
        return res;
    }
    void ComplexFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const{
        assert(mem.ge(interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order)) && "Not enough of work memory");
        auto comp = m_order.target<DofT::ComplexDofMap>()->ComponentID(idof_on_tet);
        uint nvar = comp.part_id, ldof_id = comp.cgid, lshift = idof_on_tet - comp.cgid;
        //auto lnfa = m_spaces[comp.part_id]->m_order.NumDofOnTet();
        auto ldim = m_spaces[comp.part_id]->dim(), gdim = dim();
        uint dim_shift = 0;
        for (uint i = 0; i < nvar; ++i) 
            dim_shift += m_spaces[i]->dim();
        for (int r = 0; r < fusion; ++r)
            udofs[r] = ArrayView<>(udofs[r].data + lshift, udofs[r].size - lshift);
        double* wmem = mem.ddata; mem.ddata += fusion*gdim; mem.dSize -= fusion*gdim;
        auto narrowing = [&f, dim_shift, ldim, fusion, gdim, wmem](const std::array<double, 3>& X, double* res, uint dim, void* user_data)->int{
            assert(dim == ldim*fusion && "Wrong expected dimension");
            (void) dim;
            f(X, wmem, gdim*fusion, user_data);
            for (int r = 0; r < fusion; ++r)
                std::copy(wmem + dim_shift + gdim*r, wmem + dim_shift + ldim + gdim*r, res + r*ldim);
            return 0;
        };  
        FunctorContainer<decltype(narrowing)> fc{narrowing, user_data};
        m_spaces[comp.part_id]->interpolateOnDOF(XYZ, FunctorContainer<decltype(narrowing)>::eval_functor, udofs, ldof_id, fusion, mem, &fc, max_quad_order);
        for (int r = 0; r < fusion; ++r)
            udofs[r] = ArrayView<>(udofs[r].data - lshift, udofs[r].size + lshift);    
    }
    BandDenseMatrixX<> ComplexFemSpace::applyDIV_1x1x1(AniMemoryX<> &mem, ArrayView<> &U) const{
        auto st_busy = mem.busy_mtx_parts;
        auto nfa = m_order.NumDofOnTet();
        DenseMatrix<> res(U.data, mem.q, mem.f*nfa);
        for (int k = 0; k < 3; ++k){
            auto lreq = m_spaces[k]->memDUDX(mem.q, mem.f, k);
            auto lnfa = m_spaces[k]->m_order.NumDofOnTet();
            auto lshift = m_order.target<DofT::ComplexDofMap>()->m_spaceNumDofsTet[k];
            mem.extraR.size -= lreq.Usz; 
            ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz); 
            BandDenseMatrixX<> Vx = m_spaces[k]->applyDUDX(mem, Vm, k);
            if (Vx.nparts != 1){
                mem.extraR.size += lreq.Usz; 
                throw std::runtime_error("Result of applying DUDX on 1d variable should have nparts = 1");
            }
            DenseMatrix<> V = Vx.data[0];
            for (std::size_t r = 0; r < mem.f; ++r)
                for (uint i = 0; i < lnfa; ++i)
                    for (std::size_t n = 0; n < mem.q; ++n)
                        res(n, lshift + i + nfa*r) = V(n, i + lnfa * r);
            mem.extraR.size += lreq.Usz;             
            mem.busy_mtx_parts = st_busy;
        } 
        uint mtxishift = mem.busy_mtx_parts > 0 ? 1 : 0;
        BandDenseMatrixX<> bres(1, mem.MTX.data + mem.busy_mtx_parts, mem.MTXI_ROW.data + mem.busy_mtx_parts + mtxishift, mem.MTXI_COL.data + mem.busy_mtx_parts + mtxishift);
        bres.data[0] = res;
        bres.stRow[1] = 0; bres.stRow[1] = 1;
        bres.stCol[1] = 0; bres.stCol[1] = nfa;
        return bres;
    }
    BandDenseMatrixX<> ComplexFemSpace::applyDIV_1x2(AniMemoryX<> &mem, ArrayView<> &U) const{
        auto st_busy = mem.busy_mtx_parts;
        auto nfa = m_order.NumDofOnTet();
        DenseMatrix<> res(U.data, mem.q, mem.f*nfa);
        res.SetZero();
        uint dimshift = 0;
        for (uint k = 0; k < m_spaces.size(); ++k){
            auto lnfa = m_spaces[k]->m_order.NumDofOnTet();
            auto ldim = m_spaces[k]->dim();
            auto lshift = m_order.target<DofT::ComplexDofMap>()->m_spaceNumDofsTet[k];
            BandDenseMatrixX<> Vx;
            if (ldim == 1){
                OpMemoryRequirements lreq = m_spaces[k]->memDUDX(mem.q, mem.f, dimshift);
                mem.extraR.size -= lreq.Usz; 
                ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz); 
                Vx = m_spaces[k]->applyDUDX(mem, Vm, dimshift);
                if (Vx.nparts != 1){
                    mem.extraR.size += lreq.Usz; 
                    throw std::runtime_error("Result of applying DUDX on 1d variable should have nparts = 1");
                }
                DenseMatrix<> V = Vx.data[0];
                for (std::size_t r = 0; r < mem.f; ++r)
                    for (uint i = 0; i < lnfa; ++i)
                        for (std::size_t n = 0; n < mem.q; ++n)
                            res(n, lshift + i + nfa*r) = V(n, i + lnfa * r);
                mem.extraR.size += lreq.Usz;             
            } else {
                OpMemoryRequirements lreq = m_spaces[k]->memGRAD(mem.q, mem.f);
                mem.extraR.size -= lreq.Usz; 
                ArrayView<> Vm(mem.extraR.data + mem.extraR.size, lreq.Usz); 
                Vx = m_spaces[k]->applyGRAD(mem, Vm);

                for (std::size_t p = 0; p < Vx.nparts; ++p)
                    for (std::size_t r = 0; r < mem.f; ++r)
                        for (int i = Vx.stCol[p]; i < Vx.stCol[p+1]; ++i)
                            for (std::size_t n = 0; n < mem.q; ++n){
                                if (Vx.stRow[p] <= static_cast<int>(dimshift) && Vx.stRow[p+1] > static_cast<int>(dimshift)){
                                    int d = dimshift;
                                    res(n, lshift + i + nfa*r) += Vx.data[p](d - Vx.stRow[p] + (Vx.stRow[p+1] - Vx.stRow[p]) * n, (i-Vx.stCol[p]) + (Vx.stCol[p+1] - Vx.stCol[p]) * r);
                                } 
                                if (Vx.stRow[p] <= static_cast<int>(dimshift+1+3) && Vx.stRow[p+1] > static_cast<int>(dimshift+1+3)){
                                    int d = dimshift+1 + 3;
                                    res(n, lshift + i + nfa*r) += Vx.data[p](d - Vx.stRow[p] + (Vx.stRow[p+1] - Vx.stRow[p]) * n, (i-Vx.stCol[p]) + (Vx.stCol[p+1] - Vx.stCol[p]) * r);
                                }
                            }
                mem.extraR.size += lreq.Usz;             
            }
            dimshift += ldim;
            mem.busy_mtx_parts = st_busy;
        }
        uint mtxishift = mem.busy_mtx_parts > 0 ? 1 : 0;
        BandDenseMatrixX<> bres(1, mem.MTX.data + mem.busy_mtx_parts, mem.MTXI_ROW.data + mem.busy_mtx_parts + mtxishift, mem.MTXI_COL.data + mem.busy_mtx_parts + mtxishift);
        bres.data[0] = res;
        bres.stRow[1] = 0; bres.stRow[1] = 1;
        bres.stCol[1] = 0; bres.stCol[1] = nfa;
        return bres;
    }
    BandDenseMatrixX<> ComplexFemSpace::applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const {
        return applyOP_internal<>(mem, U, &BaseFemSpace::applyIDEN, &BaseFemSpace::memIDEN);
    }
    OpMemoryRequirements ComplexFemSpace::memIDEN(uint nquadpoints, uint fusion) const {
        return memOP_internal<>(nquadpoints, fusion, &BaseFemSpace::memIDEN);
    }
    BandDenseMatrixX<> ComplexFemSpace::applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const {
        return applyOP_internal<>(mem, U, &BaseFemSpace::applyGRAD, &BaseFemSpace::memGRAD);
    }
    OpMemoryRequirements ComplexFemSpace::memGRAD(uint nquadpoints, uint fusion) const {
        return memOP_internal<>(nquadpoints, fusion, &BaseFemSpace::memGRAD);
    }
    BandDenseMatrixX<> ComplexFemSpace::applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const {
        return applyOP_internal<uchar>(mem, U, &BaseFemSpace::applyDUDX, &BaseFemSpace::memDUDX, k);
    }
    OpMemoryRequirements ComplexFemSpace::memDUDX(uint nquadpoints, uint fusion, uchar k) const {
        return memOP_internal<uchar>(nquadpoints, fusion, &BaseFemSpace::memDUDX, k);
    }
    BandDenseMatrixX<> ComplexFemSpace::applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const {
        if (dim() != 3)
            throw std::runtime_error("DIV operator supported only for 3-dimensional variables");
        if (m_spaces.size() == 1){
            return m_spaces[0]->applyDIV(mem, U);
        } else if (m_spaces.size() == 3){
            return applyDIV_1x1x1(mem, U);
        } else if (m_spaces.size() == 2){
            return applyDIV_1x2(mem, U);    
        } else 
            throw std::runtime_error("Object has internal data inconsistency");
        return BandDenseMatrixX<>();      
    }
    OpMemoryRequirements ComplexFemSpace::memDIV(uint nquadpoints, uint fusion) const {
        if (dim() != 3)
            throw std::runtime_error("DIV operator supported only for 3-dimensional variables");
        if (m_spaces.size() == 1){
            return m_spaces[0]->memDIV(nquadpoints, fusion);
        } else if (m_spaces.size() == 3){
            OpMemoryRequirements lreq[3];
            for (int k = 0; k < 3; ++k) 
                lreq[k] = m_spaces[k]->memDUDX(nquadpoints, fusion, k);
            OpMemoryRequirements res;
            res.Usz = nquadpoints * fusion * m_order.NumDofOnTet();
            res.extraRsz = std::max({lreq[0].extraRsz + lreq[0].Usz, lreq[1].extraRsz + lreq[1].Usz, lreq[2].extraRsz + lreq[2].Usz}); 
            res.extraIsz = std::max({lreq[0].extraIsz, lreq[1].extraIsz, lreq[2].extraIsz});
            res.mtx_parts = std::max({res.mtx_parts, lreq[0].mtx_parts, lreq[1].mtx_parts, lreq[2].mtx_parts}) + 2;
            return res;
        } else if (m_spaces.size() == 2){
            OpMemoryRequirements lreq[2];
            for (int k = 0; k < 2; ++k)
            if (m_spaces[k]->dim() == 1){
                lreq[k] =  m_spaces[k]->memDUDX(nquadpoints, fusion, 2*k);
            } else 
                lreq[k] = m_spaces[k]->memGRAD(nquadpoints, fusion);
            OpMemoryRequirements res;
            res.Usz = nquadpoints * fusion * m_order.NumDofOnTet();
            res.extraRsz = std::max({lreq[0].extraRsz + lreq[0].Usz, lreq[1].extraRsz + lreq[1].Usz}); 
            res.extraIsz = std::max({lreq[0].extraIsz, lreq[1].extraIsz});
            res.mtx_parts = std::max({res.mtx_parts, lreq[0].mtx_parts, lreq[1].mtx_parts}) + 2;
            return res;
        } else 
            throw std::runtime_error("Object has internal data inconsistency");
        return OpMemoryRequirements();
    }
    void ComplexFemSpace::setup(){
        std::vector<DofT::DofMap> m_dof_maps(m_spaces.size());
        for(uint i = 0; i < m_spaces.size(); ++i)
            m_dof_maps[i] = m_spaces[i]->m_order;
        m_order = DofT::merge(m_dof_maps);
        m_dim = 0; m_poly_order = m_poly_order_DIV = m_poly_order_CURL = 0;
        m_poly_order_DUDX = std::array<uint, 3>{0, 0, 0};
        for(uint i = 0; i < m_spaces.size(); ++i) {
            m_dim += m_spaces[i]->dim();
            m_poly_order = std::max(m_poly_order, m_spaces[i]->order());
            m_poly_order_DIV = std::max(m_poly_order_DIV, m_spaces[i]->orderDIV());
            m_poly_order_CURL = std::max(m_poly_order_CURL, m_spaces[i]->orderCURL());
            for (int k = 0; k < 3; ++k)
                m_poly_order_DUDX[k] = std::max(m_poly_order_DUDX[k], m_spaces[i]->orderDUDX(k));
        }
    }
    UnionFemSpace& UnionFemSpace::Init(std::vector<std::shared_ptr<BaseFemSpace>> subsets){
        m_subsets = std::move(subsets);
        setup();
        return *this;
    }
    std::shared_ptr<BaseFemSpace> UnionFemSpace::subSpace(const int* ext_dims, int ndims) const{ 
        int vdim =  m_subsets.size();
        if (ndims >= 1 && ext_dims[0] < vdim && ext_dims[0] >= 0)
            return ndims == 1 ? m_subsets[ext_dims[0]] : m_subsets[ext_dims[0]]->subSpace(ext_dims+1, ndims - 1);
        return nullptr;      
    }
    std::string UnionFemSpace::typeName() const{
        std::stringstream oss;
        oss << "Union("; 
        for (uint i = 0; i < m_subsets.size(); ++i){
            oss << m_subsets[i]->typeName();
            if (i != m_subsets.size() - 1) 
                oss << ", ";
        }
        oss << ")";
        return oss.str(); 
    }
    bool UnionFemSpace::operator==(const BaseFemSpace& other) const {
        if (gatherType() != other.gatherType()) return false;
        auto& a = *static_cast<const UnionFemSpace*>(&other);
        if (m_subsets.size() != a.m_subsets.size()) return false;
        for (uint i = 0; i < m_subsets.size(); ++i){
            if (m_subsets[i].get() == a.m_subsets[i].get()) continue;
            if (!(m_subsets[i].get() && a.m_subsets[i].get() && *m_subsets[i].get() == *a.m_subsets[i].get())) return false;
        }
        return true; 
    }
    void UnionFemSpace::evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const {
        for (uint i = 0, shift = 0; i < m_subsets.size(); ++i){
            m_subsets[i]->evalBasisFunctions(lmb, grad_lmb, phi + shift);
            shift += m_subsets[i]->m_order.NumDofOnTet();
        }
        uint nf = m_order.NumDofOnTet();
        std::vector<Expr> w(nf);
        for (uint i = 0; i < nf; ++i)
            w[i] = FT::scalsum(phi, phi+nf, m_orth_coefs.data() + i*nf);
        for (uint i = 0; i < nf; ++i)
            phi[i] = std::move(w[i]);
    }
    PlainMemoryX<> UnionFemSpace::interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const{
        auto comp = m_order.target<DofT::ComplexDofMap>()->ComponentID(idof_on_tet);
        uint nsubset = comp.part_id;
        return m_subsets[nsubset]->interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order);
    }
    void UnionFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const{
        assert(mem.ge(interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order)) && "Not enough of work memory");
        auto comp = m_order.target<DofT::ComplexDofMap>()->ComponentID(idof_on_tet);
        uint nsubset = comp.part_id, ldof_id = comp.cgid, lshift = idof_on_tet - comp.cgid;
        for (int r = 0; r < fusion; ++r)
            udofs[r] = ArrayView<>(udofs[r].data + lshift, udofs[r].size - lshift);
        m_subsets[nsubset]->interpolateOnDOF(XYZ, f, udofs, ldof_id, fusion, mem, user_data, max_quad_order);    
        for (int r = 0; r < fusion; ++r)
            udofs[r] = ArrayView<>(udofs[r].data - lshift, udofs[r].size + lshift);     
    }
    void UnionFemSpace::setup(){
        std::vector<DofT::DofMap> m_dof_maps(m_subsets.size());
        uint dim = m_subsets.empty() ? 0 : m_subsets[0]->dim();
        for(uint i = 0; i < m_subsets.size(); ++i){
            if (m_subsets[i]->dim() != dim)
                throw std::runtime_error("Union space may be defined only with subsets of fem basis functions of same dimension");
            m_dof_maps[i] = m_subsets[i]->m_order;
        }
        m_order = DofT::merge(m_dof_maps);
        orthogonalize();
    } 
    DenseMatrix<const double> UnionFemSpace::GetOrthBasisShiftMatrix() const {
        if (m_orth_coefs.empty())
            return DenseMatrix<const double>(nullptr, 0, 0);
        else{
            uint nf = m_order.NumDofOnTet();
            return DenseMatrix<const double>(m_orth_coefs.data(), nf, nf);
        }    
    }
    void UnionFemSpace::orthogonalize(uint max_quad_order){
        if (m_subsets.size() <= 1) return;

        uint nf = m_order.NumDofOnTet();
        std::vector<double> Bd(nf*nf, 0.0);
        m_orth_coefs.resize(nf*nf);
        DenseMatrix<> B(Bd.data(), nf, nf);
        uint nfshi = 0, nfshj = 0;
        double XYZa[3 * 4]{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
        double one = 1;
        Tetra<const double> XYZ(XYZa+0, XYZa+3, XYZa+6, XYZa+9);
        OpMemoryRequirements req;
        uint max_nfi = 0;
        for (uint si = 0; si < m_subsets.size(); ++si){
            auto lreq = m_subsets[si]->memIDEN(1, 1);
            req.Usz = std::max(lreq.Usz, req.Usz);
            req.mtx_parts = std::max(lreq.mtx_parts, req.mtx_parts);
            req.extraIsz = std::max(lreq.extraIsz, req.extraIsz);
            req.extraRsz = std::max(lreq.extraRsz, req.extraRsz);
            max_nfi = std::max(max_nfi, m_subsets[si]->m_order.NumDofOnTet());
        }
        PlainMemoryX<> pmx;
        for (uint si = 0; si < m_subsets.size(); ++si){
            uint nfi = m_subsets[si]->m_order.NumDofOnTet();
            for (uint k = 0; k < nfi; ++k){
                auto lpmx = m_subsets[si]->interpolateOnDOF_mem_req(k, max_nfi, max_quad_order);
                pmx.dSize = std::max(pmx.dSize, lpmx.dSize);
                pmx.iSize = std::max(pmx.iSize, lpmx.iSize);
                pmx.mSize = std::max(pmx.mSize, lpmx.mSize);
            }
        }

        pmx.iSize += req.extraIsz + 2*(req.mtx_parts + 2);
        pmx.mSize += req.mtx_parts;
        pmx.dSize += req.extraRsz + req.Usz;
        auto pmx_sz = pmx.enoughRawSize();
        std::vector<char> lmem(pmx_sz);
        pmx.allocateFromRaw(lmem.data(), pmx_sz);
        AniMemoryX<> mem;
        std::array<double, 4> xyl{1, 0, 0, 0};
        mem.XYP.Init(XYZa, 3*4);
        mem.PSI.Init(XYZa+3, 3*3);
        mem.XYG.Init(nullptr, 3);
        mem.DET.Init(&one, 1);
        mem.XYL.Init(xyl.data(), 4);
        mem.WG.Init(&one, 1);
        mem.U.Init(pmx.ddata, req.Usz); pmx.ddata += req.Usz; pmx.dSize -= req.Usz;
        mem.extraR.Init(pmx.ddata, req.extraRsz); pmx.ddata += req.extraRsz; pmx.dSize -= req.extraRsz;
        mem.extraI.Init(pmx.idata, req.extraIsz); pmx.idata += req.extraIsz; pmx.iSize -= req.extraIsz;
        mem.MTXI_COL.Init(pmx.idata, req.mtx_parts+2); pmx.idata += req.mtx_parts+2; pmx.iSize -= req.mtx_parts+2;
        mem.MTXI_ROW.Init(pmx.idata, req.mtx_parts+2); pmx.idata += req.mtx_parts+2; pmx.iSize -= req.mtx_parts+2;
        mem.MTX.Init(pmx.mdata, req.mtx_parts); pmx.mdata += req.mtx_parts; pmx.mSize -= req.mtx_parts;
        mem.q = 1, mem.f = 1;
        mem.busy_mtx_parts = 0;

        auto eval_basis_functions = [&mem](const BaseFemSpace& s, std::array<double, 3> x, double* phi){
            mem.XYG.Init(x.data(), 3);
            mem.XYL[0] = 1 - (x[0] + x[1] + x[2]);
            for (int k = 0; k < 3; ++k) mem.XYL[1+k] = x[k];
            uint dim = s.dim(), nf = s.m_order.NumDofOnTet();
            std::fill(phi, phi + dim*nf, 0);
            auto res = s.applyIDEN(mem, mem.U);
            for (std::size_t p = 0; p < res.nparts; ++p)
            for (int d = res.stRow[p]; d < res.stRow[p+1]; ++d)
            for (int i = res.stCol[p]; i < res.stCol[p+1]; ++i)
                phi[d + dim*i] = res.data[p](d - res.stRow[p], i - res.stCol[p]);
            mem.XYG.Init(nullptr, 3);
            mem.busy_mtx_parts = 0;
        };

        std::vector<ArrayView<>> udofs(max_nfi);
        for (uint i = 0; i < max_nfi; ++i)
            udofs[i] = ArrayView<>(m_orth_coefs.data() + nf * i, nf);

        for (uint si = 0; si < m_subsets.size(); ++si){
            nfshj = 0;
            uint nfi = m_subsets[si]->m_order.NumDofOnTet();
            for (uint k = 0; k < nfi; ++k)
                B(k+nfshi, k+nfshi) = 1;
            for (uint sj = 0; sj < m_subsets.size(); ++sj){
                if (sj == si) {
                    nfshj += nfi; 
                    continue;
                }
                uint nfj = m_subsets[sj]->m_order.NumDofOnTet();
                auto phi_f = [&s = *m_subsets[sj], &eval_basis_functions](const std::array<double, 3>& X, double* res, uint dim, void* user_data){
                    (void) dim, (void) user_data;
                    eval_basis_functions(s, X, res);
                    return 0;
                };
                for (uint k = 0; k < nfi; ++k){
                    m_subsets[si]->interpolateOnDOF(XYZ, phi_f, udofs.data(), k, nfj, pmx, nullptr, max_quad_order);
                    for (uint l = 0; l < nfj; ++l)
                        B(nfshi + k, nfshj + l) = udofs[l][k];
                }
                nfshj += nfj;
            }
            nfshi += nfi;
        }
        DenseMatrix<> A(m_orth_coefs.data(), nf, nf);
        {
            std::vector<double> mem(2*nf*nf);
            std::vector<int> imem(2*nf);
            fullPivLU_inverse(B.data, A.data, nf, mem.data(), imem.data());
        }
        for (uint i = 0; i < nf; ++i){
            auto p = std::max_element(A.data + i*nf, A.data + (i+1)*nf, [](double a, double b){ return std::abs(a) < std::abs(b); });
            std::transform(A.data + i*nf, A.data + (i+1)*nf, A.data + i*nf, [m = std::abs(*p)](double a){ return (std::abs(a) > m*1e-7) ? a : 0.0; });
        }
    }
    OpMemoryRequirements UnionFemSpace::memIDEN(uint nquadpoints, uint fusion) const {
        return memOP_internal<>(dimIDEN(), nquadpoints, fusion, &BaseFemSpace::memIDEN);
    }  
    OpMemoryRequirements UnionFemSpace::memGRAD(uint nquadpoints, uint fusion) const {
        return memOP_internal<>(dimGRAD(), nquadpoints, fusion, &BaseFemSpace::memGRAD);
    }
    OpMemoryRequirements UnionFemSpace::memDIV(uint nquadpoints, uint fusion) const {
        return memOP_internal<>(dimDIV(), nquadpoints, fusion, &BaseFemSpace::memDIV);
    }
    OpMemoryRequirements UnionFemSpace::memCURL(uint nquadpoints, uint fusion) const {
        return memOP_internal<>(dimCURL(), nquadpoints, fusion, &BaseFemSpace::memCURL);
    }
    OpMemoryRequirements UnionFemSpace::memDUDX(uint nquadpoints, uint fusion, uchar k) const{
        return memOP_internal<uchar>(dimDUDX(), nquadpoints, fusion, &BaseFemSpace::memDUDX, k);
    }
    BandDenseMatrixX<> UnionFemSpace::applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const {
        return applyOP_internal<>(dimIDEN(), mem, U, &BaseFemSpace::applyIDEN, &BaseFemSpace::memIDEN);
    }
    BandDenseMatrixX<> UnionFemSpace::applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const {
        return applyOP_internal<>(dimGRAD(), mem, U, &BaseFemSpace::applyGRAD, &BaseFemSpace::memGRAD);
    }
    BandDenseMatrixX<> UnionFemSpace::applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const {
        return applyOP_internal<>(dimDIV(), mem, U, &BaseFemSpace::applyDIV, &BaseFemSpace::memDIV);
    }
    BandDenseMatrixX<> UnionFemSpace::applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const {
        return applyOP_internal<>(dimCURL(), mem, U, &BaseFemSpace::applyCURL, &BaseFemSpace::memCURL);
    }
    BandDenseMatrixX<> UnionFemSpace::applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const{
        return applyOP_internal<uchar>(dimDUDX(), mem, U, &BaseFemSpace::applyDUDX, &BaseFemSpace::memDUDX, k);
    }
    FemSpace operator+(const FemSpace& a, const FemSpace& b){
        BaseFemSpace::BaseTypes ta = a.gatherType(), tb = b.gatherType();
        std::vector<std::shared_ptr<BaseFemSpace>> to_union;
        if (ta != BaseFemSpace::BaseTypes::UnionType)
            to_union.push_back(a.base());
        else{ 
            auto& ss = a.target<UnionFemSpace>()->m_subsets;
            to_union.insert(to_union.end(), ss.begin(), ss.end()); 
        } 
        if (tb != BaseFemSpace::BaseTypes::UnionType)
            to_union.push_back(b.base());
        else{ 
            auto& ss = b.target<UnionFemSpace>()->m_subsets;
            to_union.insert(to_union.end(), ss.begin(), ss.end()); 
        } 
        return FemSpace(UnionFemSpace(to_union));
    }
    FemSpace operator*(const FemSpace& a, const FemSpace& b){
        BaseFemSpace::BaseTypes ta = a.gatherType(), tb = b.gatherType();
        std::vector<std::shared_ptr<BaseFemSpace>> m_merge;
        if (ta != BaseFemSpace::BaseTypes::ComplexType)
            m_merge.push_back(a.base());
        else{ 
            auto& ss = a.target<ComplexFemSpace>()->m_spaces;
            m_merge.insert(m_merge.end(), ss.begin(), ss.end()); 
        } 
        if (tb != BaseFemSpace::BaseTypes::ComplexType)
            m_merge.push_back(b.base());
        else{ 
            auto& ss = b.target<ComplexFemSpace>()->m_spaces;
            m_merge.insert(m_merge.end(), ss.begin(), ss.end()); 
        } 
        return FemSpace(ComplexFemSpace(m_merge));
    }
    FemSpace operator^(const FemSpace& a, int k){
        if (k < 0)
            throw std::runtime_error("Power to negative order is not defined");
        return FemSpace(VectorFemSpace(k, a.base()));
    }
    ApplyOpFromSpace FemSpace::getOP_own(OperatorType op) const { return ApplyOpFromSpace(op, *this); }
}