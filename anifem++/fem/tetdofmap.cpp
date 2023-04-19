#include "tetdofmap.h"
#include <numeric>

using namespace Ani;

using DofT::uchar;
using DofT::uint;

DofT::DofSparsedIterator DofT::BaseDofMap::beginBySparsity(const TetGeomSparsity& sp, bool preferGeomOrdering) const {
    LocalOrder lo;
    BeginByGeomSparsity(sp, lo, preferGeomOrdering);
    DofSparsedIterator it;
    it.sparsity = sp;
    it.ord = lo;
    it.map = this;
    it.preferGeomOrdering = preferGeomOrdering;
    return it;
}

DofT::DofSparsedIterator DofT::BaseDofMap::endBySparsity() const{
    LocalOrder lo;
    EndByGeomSparsity(lo);
    DofSparsedIterator it;
    it.ord = lo;
    it.map = this;
    return it;
}

DofT::DofIterator DofT::BaseDofMap::begin() const { return NumDofOnTet() > 0 ? DofIterator(this, LocalOrderOnTet(TetOrder(0))) : end(); }
DofT::DofIterator DofT::BaseDofMap::end() const { LocalOrder lo; lo.gid = NumDofOnTet(); return DofIterator(this, lo); }

DofT::TetGeomSparsity& DofT::TetGeomSparsity::setCell(bool with_closure) { 
    elems[3] = (1<<0);
    if (with_closure){
        elems[0] = (1<<0) | (1<<1) | (1<<2) | (1<<3);
        elems[1] = (1<<0) | (1<<1) | (1<<2) | (1<<3) | (1<<4) | (1<<5);
        elems[2] = (1<<0) | (1<<1) | (1<<2) | (1<<3);
    }
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::unsetCell(bool with_closure) { 
    if (!with_closure)
        elems[3] = 0;
    else clear();
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::setFace(int iface, bool with_closure) {
    elems[2] |= (1<<iface);
    if (with_closure){
        const static std::array<char, 12> 
                            lookup_nds = {0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3},
                            lookup_eds = {0, 1, 3, 3, 4, 5, 1, 2, 5, 0, 2, 4};
        elems[0] |= (1<<lookup_nds[3*iface]) | (1<<lookup_nds[3*iface+1]) | (1<<lookup_nds[3*iface+2]);
        elems[1] |= (1<<lookup_eds[3*iface]) | (1<<lookup_eds[3*iface+1]) | (1<<lookup_eds[3*iface+2]);                    
    }
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::unsetFace(int iface, bool with_closure){
    elems[2] &= ~(1<<iface);
    if (with_closure){
        const static std::array<char, 12> 
                            lookup_nds = {0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3},
                            lookup_eds = {0, 1, 3, 3, 4, 5, 1, 2, 5, 0, 2, 4};
        elems[0] &= ~((1<<lookup_nds[3*iface]) | (1<<lookup_nds[3*iface+1]) | (1<<lookup_nds[3*iface+2]));
        elems[1] &= ~((1<<lookup_eds[3*iface]) | (1<<lookup_eds[3*iface+1]) | (1<<lookup_eds[3*iface+2]));                    
    }
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::setEdge(int iedge, bool with_closure) {
    elems[1] |= (1<<iedge);
    if (with_closure){
        const static std::array<char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
        elems[0] |= (1<<lookup_nds[2*iedge]) | (1<<lookup_nds[2*iedge+1]);
    }
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::unsetEdge(int iedge, bool with_closure) {
    elems[1] &= ~(1<<iedge);
    if (with_closure){
        const static std::array<char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
        elems[0] &= ~((1<<lookup_nds[2*iedge]) | (1<<lookup_nds[2*iedge+1]));
    }
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::setNode(int inode){
        elems[0] |= (1 << inode);
        return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::unsetNode(int inode){
        elems[0] &= ~(1 << inode);
        return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::set(uchar elem_dim, int ielem, bool with_closure){
    switch(elem_dim){
        case 0: return setNode(ielem);
        case 1: return setEdge(ielem, with_closure);
        case 2: return setFace(ielem, with_closure);
        case 3: return setCell(with_closure);
        default: throw std::runtime_error("Wrong dimesion");
    }
    return *this;
}
DofT::TetGeomSparsity& DofT::TetGeomSparsity::unset(uchar elem_dim, int ielem, bool with_closure){
    switch(elem_dim){
        case 0: return unsetNode(ielem);
        case 1: return unsetEdge(ielem, with_closure);
        case 2: return unsetFace(ielem, with_closure);
        case 3: return unsetCell(with_closure);
        default: throw std::runtime_error("Wrong dimesion");
    }
    return *this; 
}
std::pair<std::array<uchar, 6>, uchar> DofT::TetGeomSparsity::getElemsIds(uchar elem_dim) const {
    std::array<uchar, 6> res = {0};
    int sz = 0;
    const static uchar lookup_sz[] = {4, 6, 4, 1};
    for (int i = 0; i < lookup_sz[elem_dim]; ++i)
        if (elems[elem_dim] & (1 << i)) 
            res[sz++] = i;
    return {res, sz};
}

DofT::TetGeomSparsity::Pos DofT::TetGeomSparsity::beginPos() const {
    const static uchar lookup_sz[] = {4, 6, 4, 1};
    for (uchar i = 0; i < 4; ++i) if (!empty(i)){
        for (uchar k = 0; k < lookup_sz[i]; ++k) if (elems[i] & (1 << k))
            return Pos(i, k);
    }
    return endPos();
}
DofT::TetGeomSparsity::Pos DofT::TetGeomSparsity::beginPos(uchar elem_dim) const {
    const static uchar lookup_sz[] = {4, 6, 4, 1};
    uchar i = elem_dim;
    if (!empty(i)){
        for (uchar k = 0; k < lookup_sz[i]; ++k) if (elems[i] & (1 << k))
            return Pos(i, k);
    }
    return endPos();
}
DofT::TetGeomSparsity::Pos DofT::TetGeomSparsity::nextPos(Pos p) const{
    if (!p.isValid()) return endPos();
    const static uchar lookup_sz[] = {4, 6, 4, 1};
    for (uchar k = p.elem_num+1; k < lookup_sz[p.elem_dim]; ++k) if (elems[p.elem_dim] & (1 << k))
        return Pos(p.elem_dim, k);
    for (uchar i = p.elem_dim+1; i < 4; ++i) if (!empty(i)){
        for (uchar k = 0; k < lookup_sz[i]; ++k) if (elems[i] & (1 << k))
            return Pos(i, k);
    }
    return endPos();
}
DofT::TetGeomSparsity::Pos DofT::TetGeomSparsity::nextPosOnDim(Pos p) const{
    if (!p.isValid()) return endPos();
    const static uchar lookup_sz[] = {4, 6, 4, 1};
    for (uchar k = p.elem_num+1; k < lookup_sz[p.elem_dim]; ++k) if (elems[p.elem_dim] & (1 << k))
        return Pos(p.elem_dim, k);
    return endPos();        
}

void DofT::BaseDofMap::BeginByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const {
    (void) preferGeomOrdering;
    auto ndofs = NumDofs();
    std::array<uint, 4> ndofs_compact = {ndofs[0], ndofs[1] + ndofs[2], ndofs[3] + ndofs[4], ndofs[5]};
    for (uchar i = 0; i < 4; ++i) if (ndofs_compact[i] != 0 && !sp.empty(i)){
        switch (i) {
            case 0: lo.etype = NODE; break;
            case 1: lo.etype = ndofs[1] > 0 ? EDGE_UNORIENT : EDGE_ORIENT; break;
            case 2: lo.etype = ndofs[3] > 0 ? FACE_UNORIENT : FACE_ORIENT; break;
            case 3: lo.etype = CELL; break; 
        }
        lo.nelem = sp.beginPos(i).elem_num;
        lo.leid = 0;
        lo.gid = TetDofID(lo.getGeomOrder());
        return;
    }
    EndByGeomSparsity(lo);
    return;
}
void DofT::BaseDofMap::EndByGeomSparsity(LocalOrder& lo) const{
    lo = LocalOrder();
    return;
}
void DofT::BaseDofMap::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const {
    (void) preferGeomOrdering;
    assert(isValidIndex(lo) && "Wrong local index");
    uint nloc_dof = NumDof(lo.etype);
    if (lo.leid < nloc_dof - 1){
        ++lo.leid;
        lo.gid = TetDofID(lo.getGeomOrder());
        return;
    }
    if (lo.etype == EDGE_UNORIENT && NumDof(EDGE_ORIENT) > 0){
        lo.etype = EDGE_ORIENT;
        lo.leid = 0;
        lo.gid = TetDofID(lo.getGeomOrder());
        return; 
    } else if (lo.etype == FACE_UNORIENT && NumDof(FACE_ORIENT) > 0){
        lo.etype = FACE_ORIENT;
        lo.leid = 0;
        lo.gid = TetDofID(lo.getGeomOrder());
        return;
    }
    auto dim = GeomTypeDim(lo.etype);
    auto next_pos = sp.nextPos(TetGeomSparsity::Pos(dim, lo.nelem));
    if (!next_pos.isValid()) { 
        EndByGeomSparsity(lo); 
        return; 
    }
    if (dim == next_pos.elem_dim){
        lo.nelem = next_pos.elem_num;
        lo.leid = 0;
        if (next_pos.elem_dim == 1){
            if (NumDof(EDGE_UNORIENT) > 0) 
                lo.etype = EDGE_UNORIENT;
        } else if (next_pos.elem_dim == 2){
            if (NumDof(FACE_UNORIENT) > 0) 
                lo.etype = FACE_UNORIENT;
        }
        lo.gid = TetDofID(lo.getGeomOrder());
    } else {
        TetGeomSparsity sp_new = sp;
        for (int i = 0; i < next_pos.elem_dim; ++i) sp_new.unset(i);
        BeginByGeomSparsity(sp_new, lo);
    }
    return;
}

uint DofT::BaseDofMap::NumDofOnTet(uchar etype) const {
    auto nn = NumDofsOnTet();
    uint res = 0;
    for (uint i = 0; i < NGEOM_TYPES; ++i) if (etype & NumToGeomType(i)) 
        res += nn[i];
    return res;
}
uint DofT::BaseDofMap::NumDofOnTet() const {
    auto nn = NumDofsOnTet();
    return std::accumulate(nn.begin(), nn.end(), 0U);
}
std::array<uint, DofT::NGEOM_TYPES> DofT::BaseDofMap::NumDofs() const {
    std::array<uint, NGEOM_TYPES> res;
    for (int i = 0; i < NGEOM_TYPES; ++i) res[i] = NumDof(NumToGeomType(i));
    return res;
}
std::array<uint, DofT::NGEOM_TYPES> DofT::BaseDofMap::NumDofsOnTet() const {
    std::array<uint, NGEOM_TYPES> res;
    for (int i = 0; i < NGEOM_TYPES; ++i) {
        auto t = NumToGeomType(i);
        res[i] = GeomTypeTetElems(t) * NumDof(t);
    }
    return res;
}
uint DofT::BaseDofMap::GetGeomMask() const {
    uint res = 0;
    for (int i = 0; i < NGEOM_TYPES; ++i){ 
        auto etype = NumToGeomType(i);
        if (DefinedOn(etype)) 
            res |= etype; 
    }
    return res;
}
DofT::NestedDofMap DofT::BaseDofMap::GetNestedDofMap(const int* ext_dims, int ndims) const {
    NestedDofMap view;
    GetNestedComponent(ext_dims, ndims, view);
    return view;
}
DofT::NestedDofMapView DofT::BaseDofMap::GetNestedDofMapView(const int* ext_dims, int ndims) const {
    NestedDofMapView view;
    GetNestedComponent(ext_dims, ndims, view);
    return view;
}

DofT::UniteDofMap::UniteDofMap(std::array<uint, NGEOM_TYPES> NumDofs) {
    m_shiftDof[0] = m_shiftTetDof[0] = 0;
    for (uint i = 0; i < NGEOM_TYPES; ++i) {
        m_shiftTetDof[i + 1] += m_shiftTetDof[i] + GeomTypeTetElems(NumToGeomType(i)) * NumDofs[i];
        m_shiftDof[i + 1] += m_shiftDof[i] + NumDofs[i];
    }
}
uint DofT::UniteDofMap::NumDofOnTet(uchar etype) const{
    uint r = 0;
    for (int i = 0; i < NGEOM_TYPES; ++i) if (etype & (1 << i))
        r += m_shiftTetDof[i+1] - m_shiftTetDof[i];
    return r;    
}
std::array<uint, DofT::NGEOM_TYPES> DofT::UniteDofMap::NumDofs() const { 
    std::array<uint, NGEOM_TYPES> res;
    for (int i = 0; i < NGEOM_TYPES; ++i) 
        res[i] = m_shiftDof[i+1] - m_shiftDof[i];
    return res;
} 
std::array<uint, DofT::NGEOM_TYPES> DofT::UniteDofMap::NumDofsOnTet() const{ 
    std::array<uint, NGEOM_TYPES> res;
    for (int i = 0; i < NGEOM_TYPES; ++i) 
        res[i] = m_shiftTetDof[i+1] - m_shiftTetDof[i];
    return res;
} 
DofT::LocalOrder DofT::UniteDofMap::LocalOrderOnTet(TetOrder dof) const {
    assert(isValidIndex(dof) && "Wrong dof number");
    LocalOrder lo;
    uint num = std::upper_bound(m_shiftTetDof.data(), m_shiftTetDof.data() + m_shiftTetDof.size(), dof.gid) - m_shiftTetDof.data() - 1;
    lo.etype = NumToGeomType(num);
    uint enumdof = m_shiftDof[num+1] - m_shiftDof[num];
    lo.nelem = (dof - m_shiftTetDof[num]) / enumdof;
    lo.leid  = (dof - m_shiftTetDof[num]) % enumdof;
    lo.gid = dof;
    return lo;
}
uint DofT::UniteDofMap::TypeOnTet(uint dof) const{
    assert(isValidIndex(TetOrder(dof)) && "Wrong index");
    return NumToGeomType(std::upper_bound(m_shiftTetDof.data(), m_shiftTetDof.data() + m_shiftTetDof.size(), dof) - m_shiftTetDof.data() - 1);
}
uint DofT::UniteDofMap::TetDofID(LocGeomOrder dof) const{
    assert(isValidIndex(dof) && "Wrong index");
    uint num = GeomTypeToNum(dof.etype);
    return m_shiftTetDof[num] + dof.nelem*(m_shiftDof[num+1] - m_shiftDof[num]) + dof.leid;
}
uint DofT::UniteDofMap::GetGeomMask() const{
    uint res = 0;
    for (int i = 0; i < NGEOM_TYPES; ++i) 
        res |= ((m_shiftTetDof[i+1] - m_shiftTetDof[i]) != 0) ? (1 << i) : 0;
    return res;    
}
void DofT::UniteDofMap::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const{
    assert(isValidIndex(lo) && "Wrong local index");
    uint nloc_dof = NumDof(lo.etype);
    if (lo.leid < nloc_dof - 1){
        ++lo.leid;
        ++lo.gid;
        return;
    }
    if (lo.etype == EDGE_UNORIENT && NumDof(EDGE_ORIENT) > 0){
        lo.etype = EDGE_ORIENT;
        lo.leid = 0;
        lo.gid = m_shiftTetDof[2] + (m_shiftDof[3] - m_shiftDof[2])*lo.nelem;//TetDofID(lo.getGeomOrder());
        return; 
    } else if (lo.etype == FACE_UNORIENT && NumDof(FACE_ORIENT) > 0){
        lo.etype = FACE_ORIENT;
        lo.leid = 0;
        lo.gid = m_shiftTetDof[4] + (m_shiftDof[5] - m_shiftDof[4])*lo.nelem;//TetDofID(lo.getGeomOrder());
        return;
    }
    auto dim = GeomTypeDim(lo.etype);
    auto next_pos = sp.nextPos(TetGeomSparsity::Pos(dim, lo.nelem));
    if (!next_pos.isValid()) { 
        EndByGeomSparsity(lo); 
        return; 
    }
    if (dim == next_pos.elem_dim){
        lo.nelem = next_pos.elem_num;
        lo.leid = 0;
        if (next_pos.elem_dim == 1){
            if (NumDof(EDGE_UNORIENT) > 0) 
                lo.etype = EDGE_UNORIENT;
        } else if (next_pos.elem_dim == 2){
            if (NumDof(FACE_UNORIENT) > 0) 
                lo.etype = FACE_UNORIENT;
        }
        auto num = GeomTypeToNum(lo.etype);
        lo.gid = m_shiftTetDof[num] + (m_shiftDof[num+1] - m_shiftDof[num])*lo.nelem;//TetDofID(lo.getGeomOrder());
    } else {
        TetGeomSparsity sp_new = sp;
        for (int i = 0; i < next_pos.elem_dim; ++i) sp_new.unset(i);
        BeginByGeomSparsity(sp_new, lo, preferGeomOrdering);
    }
    return;
}

bool DofT::UniteDofMap::operator==(const BaseDofMap& o) const {
    if (o.ActualType() != ActualType()) return false;
    auto& a = static_cast<const UniteDofMap&>(o);
    return m_shiftDof == a.m_shiftDof; //&& m_shiftTetDof == a.m_shiftTetDof
}

std::array<uint, DofT::NGEOM_TYPES> DofT::VectorDofMap::NumDofs() const  { 
    if (!m_dim) return std::array<uint, DofT::NGEOM_TYPES>{0};
    auto r = base->NumDofs();
    for(auto& x: r) x *= m_dim;
    return r;
}
std::array<uint, DofT::NGEOM_TYPES> DofT::VectorDofMap::NumDofsOnTet() const {
    if (!m_dim) return std::array<uint, DofT::NGEOM_TYPES>{0};
    auto r = base->NumDofsOnTet();
    for(auto& x: r) x *= m_dim;
    return r;
}
DofT::LocalOrder DofT::VectorDofMap::LocalOrderOnTet(TetOrder dof_id) const  {
    assert(isValidIndex(dof_id) && "Wrong dof number");
    auto base_dof_on_tet = base->NumDofOnTet();
    LocalOrder lo = base->LocalOrderOnTet(TetOrder(dof_id.gid % base_dof_on_tet));
    lo.leid += (dof_id.gid / base_dof_on_tet) * base->NumDof(lo.etype);
    lo.gid = dof_id.gid;
    return lo;
}
uint DofT::VectorDofMap::TetDofID(LocGeomOrder dof_id) const{ 
    assert(isValidIndex(dof_id) && "Wrong dof number");
    int nOdf = base->NumDofOnTet(), lOdf = base->NumDofOnTet(dof_id.etype);
    int component = dof_id.leid / lOdf;
    return component * nOdf + base->TetDofID(LocGeomOrder(dof_id.etype, dof_id.nelem, dof_id.leid % lOdf));
}
void DofT::VectorDofMap::GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const{
    if (ndims == 0 || ext_dims[0] < 0 || ext_dims[0] >= m_dim) {
        view.Clear();
        return;
    }
    auto dsz = base->NumDofs();
    for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDof[i] += ext_dims[0]*dsz[i];
    view.m_shiftOnTet += ext_dims[0]*base->NumDofOnTet();
    if (ndims == 1)
        view.set_base(base);
    else 
        base->GetNestedComponent(ext_dims+1, ndims-1, view);
}
std::shared_ptr<DofT::BaseDofMap> DofT::VectorDofMap::GetSubDofMap(const int* ext_dims, int ndims){
    if (ndims > 0 && *ext_dims >= 0 && *ext_dims < m_dim){
        return (ndims == 1) ? base : base->GetSubDofMap(ext_dims+1, ndims-1);
    } else 
        return nullptr;
}
void DofT::VectorDofMap::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const{
    assert(isValidIndex(lo) && "Wrong index");
    int nOdf = base->NumDofOnTet(), lOdf = base->NumDof(lo.etype);
    uint dim = lo.gid / nOdf;
    uint lcomp = lo.gid % nOdf; 
    LocalOrder ll, lend;
    ll.gid = lcomp;
    ll.etype = lo.etype;
    ll.nelem = lo.nelem;
    ll.leid = lo.leid % lOdf;
    base->EndByGeomSparsity(lend);
    if (!preferGeomOrdering){
        base->IncrementByGeomSparsity(sp, ll, false);
        if (ll != lend){
            if (ll.etype != lo.etype) lOdf = base->NumDof(ll.etype);
            lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf);
            return;
        } else {
            ++dim;
            if (dim != static_cast<uint>(m_dim)){
                base->BeginByGeomSparsity(sp, ll, false);
                if (ll != lend){
                    if (ll.etype != lo.etype) lOdf = base->NumDof(ll.etype);
                    lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf);
                    return;
                }
            }
            EndByGeomSparsity(lo);
            return;
        }
    } else {
        TetGeomSparsity ss;
        auto gdim = GeomTypeDim(lo.etype);
        ss.set(gdim, lo.nelem);
        base->IncrementByGeomSparsity(ss, ll, true);
        if (ll != lend){
            if (ll.etype != lo.etype) lOdf = base->NumDof(ll.etype);
            lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf);
            return;
        } else {
            ++dim;
            if (dim != static_cast<uint>(m_dim)){
                LocGeomOrder lgo(lo.etype, lo.nelem, 0);
                if (lo.etype == EDGE_ORIENT){
                    auto eodf = base->NumDof(EDGE_UNORIENT);
                    if (eodf > 0){
                        lgo.etype = EDGE_UNORIENT;
                        lOdf = eodf;
                    }
                }
                else if (lo.etype == FACE_ORIENT){
                    auto fodf = base->NumDof(FACE_UNORIENT);
                    if (fodf > 0){
                        lgo.etype = FACE_UNORIENT;
                        lOdf = fodf;
                    }
                }
                lo = LocalOrder(base->TetDofID(lgo) + dim*nOdf, lgo.etype, lgo.nelem, lgo.leid + dim*lOdf);
                return;
            } else {
                dim = 0;
                auto next_pos = sp.nextPos(TetGeomSparsity::Pos(gdim, lo.nelem));
                if (!next_pos.isValid()) { 
                    EndByGeomSparsity(lo); 
                    return; 
                }
                ss = sp;
                for (int i = 0; i < next_pos.elem_dim; ++i) ss.unset(i);
                for (int i = 0; i < next_pos.elem_num; ++i) ss.unset(next_pos.elem_dim, i);
                base->BeginByGeomSparsity(ss, ll, true);
                if (ll != lend){
                    lOdf = base->NumDof(ll.etype);
                    lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf);
                } else 
                    EndByGeomSparsity(lo);
                return;
            }
        }
    }
}

bool DofT::VectorDofMap::operator==(const BaseDofMap& o) const {
    if (o.ActualType() != ActualType()) return false;
    auto& a = static_cast<const VectorDofMap&>(o);
    if (m_dim != a.m_dim) return false;
    return base.get() == a.base.get() || (base.get() && a.base.get() && *base.get() == *a.base.get());
}

DofT::ComplexDofMap DofT::ComplexDofMap::makeCompressed(const std::vector<std::shared_ptr<BaseDofMap>>& spaces){
    if (spaces.size() <= 1) return ComplexDofMap(spaces);
    std::vector<std::shared_ptr<BaseDofMap>> compressed; compressed.reserve(spaces.size());
    uint j = 0;
    auto checkSame = [](const std::shared_ptr<BaseDofMap>& a, const std::shared_ptr<BaseDofMap>& b) -> bool{
        return a.get() == b.get() || (a.get() && b.get() && *a.get() == *b.get()); 
    };
    for (uint i = 1; i < spaces.size(); ++i){
        if (!checkSame(spaces[i], spaces[i-1])){
            if (j == 0){
                compressed.push_back(spaces[i-1]);
            } else {
                compressed.push_back(std::make_shared<VectorDofMap>(j+1, spaces[i-1]));
                j = 0;
            }
        } else 
            ++j;
    }
    if (j == 0){
        compressed.push_back(spaces.back());
    } else {
        compressed.push_back(std::make_shared<VectorDofMap>(j+1, spaces.back()));
        j = 0;
    }
    return ComplexDofMap(compressed);
}
DofT::ComplexDofMap::ComplexDofMap(std::vector<std::shared_ptr<BaseDofMap>> spaces): m_spaces{std::move(spaces)}{
    for (uint t = 0; t < NGEOM_TYPES; ++t) {
        auto etype = NumToGeomType(t);
        m_spaceNumDofTet[t].resize(m_spaces.size()+1);
        m_spaceNumDof[t].resize(m_spaces.size()+1);
        m_spaceNumDofTet[t][0] = m_spaceNumDof[t][0] = 0;
        for (uint i = 0; i < m_spaces.size(); ++i) {
            m_spaceNumDofTet[t][i+1] = m_spaceNumDofTet[t][i] + m_spaces[i]->NumDofOnTet(etype);
            m_spaceNumDof[t][i+1] = m_spaceNumDof[t][i] + m_spaces[i]->NumDof(etype);
        }
    }
    m_spaceNumDofsTet.resize(m_spaces.size()+1);
    m_spaceNumDofsTet[0] = 0;
    for (uint i = 0; i < m_spaces.size(); ++i) {
        m_spaceNumDofsTet[i+1] = m_spaceNumDofsTet[i] + m_spaces[i]->NumDofOnTet();
    }
}
std::array<uint, DofT::NGEOM_TYPES> DofT::ComplexDofMap::NumDofs() const{
    std::array<uint, NGEOM_TYPES> res = {0};
    for (int t = 0; t < NGEOM_TYPES; ++t) res[t] = m_spaceNumDof[t][m_spaces.size()] - m_spaceNumDof[t][0];
    return res;
}
std::array<uint, DofT::NGEOM_TYPES> DofT::ComplexDofMap::NumDofsOnTet() const {
    std::array<uint, NGEOM_TYPES> res = {0};
    for (int t = 0; t < NGEOM_TYPES; ++t) res[t] = m_spaceNumDofTet[t][m_spaces.size()] - m_spaceNumDofTet[t][0];
    return res;
}
DofT::LocalOrder DofT::ComplexDofMap::LocalOrderOnTet(TetOrder dof_id) const {
    assert(isValidIndex(dof_id) && "Wrong dof number");
    int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof_id.gid) - m_spaceNumDofsTet.data() - 1;
    LocalOrder lo = m_spaces[vid]->LocalOrderOnTet(TetOrder(dof_id.gid - m_spaceNumDofsTet[vid]));
    lo.leid += m_spaceNumDof[GeomTypeToNum(lo.etype)][vid];
    lo.gid = dof_id.gid;
    return lo;
}
uint DofT::ComplexDofMap::TypeOnTet(uint dof) const  {
    assert(isValidIndex(TetOrder(dof)) && "Wrong dof number");
    int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof) - m_spaceNumDofsTet.data() - 1;
    return m_spaces[vid]->TypeOnTet(dof - m_spaceNumDofsTet[vid]);
}
uint DofT::ComplexDofMap::TetDofID(LocGeomOrder dof_id) const {
    assert(isValidIndex(dof_id) && "Wrong index");
    auto num = GeomTypeToNum(dof_id.etype);
    int vid = std::upper_bound(m_spaceNumDof[num].data(), m_spaceNumDof[num].data() + m_spaceNumDof[num].size(), dof_id.leid) - m_spaceNumDof[num].data() - 1;
    dof_id.leid -= m_spaceNumDof[num][vid];
    return m_spaceNumDofsTet[vid] + m_spaces[vid]->TetDofID(dof_id);
}
uint DofT::ComplexDofMap::GetGeomMask() const  { ///< @return values DefinedOn(t) for all types t
    uint res = UNDEF;
    for (int i = 0; i < NGEOM_TYPES; ++i){ 
        if (m_spaceNumDof[i][m_spaces.size()] - m_spaceNumDof[i][0] > 0) 
            res |= NumToGeomType(i); 
    }
    return res;
};
void DofT::ComplexDofMap::GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const {
    if (ndims <= 0 || ext_dims[0] < 0 || ext_dims[0] >= static_cast<int>(m_spaces.size())){
        view.Clear();
        return;
    }
    for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDof[i] += m_spaceNumDof[i][ext_dims[0]] - m_spaceNumDof[i][0];
    view.m_shiftOnTet += m_spaceNumDofsTet[ext_dims[0]] - m_spaceNumDofsTet[0];
    if (ndims == 1)
        view.set_base(m_spaces[ext_dims[0]]);
    else 
        m_spaces[ext_dims[0]]->GetNestedComponent(ext_dims+1, ndims-1, view); 
}
std::shared_ptr<DofT::BaseDofMap> DofT::ComplexDofMap::GetSubDofMap(const int* ext_dims, int ndims) {
    if (ndims <= 0 || ext_dims[0] < 0 || ext_dims[0] >= static_cast<int>(m_spaces.size()))
        return nullptr;
    return ndims == 1 ? m_spaces[ext_dims[0]] : m_spaces[ext_dims[0]]->GetSubDofMap(ext_dims+1, ndims-1);
}
void DofT::ComplexDofMap::BeginByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const {
    if (!preferGeomOrdering){
        for (uint dim = 0; dim < m_spaces.size(); ++dim){
            LocalOrder ll, lend;
            m_spaces[dim]->EndByGeomSparsity(lend);
            m_spaces[dim]->BeginByGeomSparsity(sp, ll, preferGeomOrdering);
            if (ll != lend){
                lo = LocalOrder(ll.gid + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[GeomTypeToNum(ll.etype)][dim]);
                return;
            } 
        }
        EndByGeomSparsity(lo);
        return;
    } else {
        for (int num = 0; num < NGEOM_TYPES; ++num){
            auto etype = NumToGeomType(num);
            auto gdim = GeomTypeDim(etype);
            auto pos = sp.beginPos(gdim);
            if (!pos.isValid()) continue;
            for (uint dim = 0; dim < m_spaces.size(); ++dim) if (m_spaces[dim]->DefinedOn(etype)){
                m_spaces[dim]->BeginByGeomSparsity(sp, lo, preferGeomOrdering);
                lo.gid += m_spaceNumDofsTet[dim], lo.leid += m_spaceNumDof[num][dim];
                return;
            }
        }
        EndByGeomSparsity(lo);
        return;
    }
}
void DofT::ComplexDofMap::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const{
    assert(isValidIndex(lo) && "Wrong index");
    auto num = GeomTypeToNum(lo.etype);
    uint dim = std::upper_bound(m_spaceNumDof[num].data(), m_spaceNumDof[num].data() + m_spaceNumDof[num].size(), lo.leid) - m_spaceNumDof[num].data() - 1;
    LocalOrder ll(lo.gid - m_spaceNumDofsTet[dim], lo.etype, lo.nelem, lo.leid - m_spaceNumDof[num][dim]); 
    LocalOrder lend;
    m_spaces[dim]->EndByGeomSparsity(lend);
    if (!preferGeomOrdering){
        m_spaces[dim]->IncrementByGeomSparsity(sp, ll, preferGeomOrdering);
        if (ll != lend){
            if (ll.etype != lo.etype) num = GeomTypeToNum(ll.etype);
            lo = LocalOrder(ll.gid + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[num][dim]);
            return;
        } else {
            ++dim;
            if (dim != m_spaces.size()){
                m_spaces[dim]->EndByGeomSparsity(lend);
                m_spaces[dim]->BeginByGeomSparsity(sp, ll, preferGeomOrdering);
                if (ll != lend){
                    if (ll.etype != lo.etype) num = GeomTypeToNum(ll.etype);
                    lo = LocalOrder(ll.gid + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[num][dim]);
                    return;
                }
            }
            EndByGeomSparsity(lo);
            return;
        }
    } else {
        TetGeomSparsity ss;
        auto gdim = GeomTypeDim(lo.etype);
        ss.set(gdim, lo.nelem);
        LocalOrder lgo = ll;
        m_spaces[dim]->IncrementByGeomSparsity(ss, lgo, preferGeomOrdering);
        // if (ll.etype != lo.etype) num = GeomTypeToNum(ll.etype);
        if (lgo != lend){
            if (lgo.etype != lo.etype) num = GeomTypeToNum(lgo.etype);
            lo = LocalOrder(lgo.gid + m_spaceNumDofsTet[dim], lgo.etype, lgo.nelem, lgo.leid + m_spaceNumDof[num][dim]);
            return;
        } else {
            do { ++dim; } while( dim < m_spaces.size() && !m_spaces[dim]->DefinedOn(ll.etype));
            if (dim != m_spaces.size()){
                ll.leid = 0;
                LocGeomOrder lgo(ll.etype, ll.nelem, ll.leid);
                num = GeomTypeToNum(ll.etype);
                lo = LocalOrder(m_spaces[dim]->TetDofID(lgo) + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[num][dim]);
                return;
            } else {
                auto next_pos = sp.nextPos(TetGeomSparsity::Pos(gdim, lo.nelem));
                if (!next_pos.isValid()) { 
                    EndByGeomSparsity(lo); 
                    return; 
                }
                ss = sp;
                for (int i = 0; i < next_pos.elem_dim; ++i) ss.unset(i);
                for (int i = 0; i < next_pos.elem_num; ++i) ss.unset(next_pos.elem_dim, i);
                BeginByGeomSparsity(ss, lo, preferGeomOrdering);
                return;
            }
        }
    }
}

DofT::ComponentTetOrder DofT::ComplexDofMap::ComponentID(TetOrder dof_id) const{
    assert(isValidIndex(dof_id) && "Wrong dof number");
    int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof_id.gid) - m_spaceNumDofsTet.data() - 1;
    TetOrder lgid(dof_id.gid - m_spaceNumDofsTet[vid]);
    return {dof_id, lgid, static_cast<uint>(vid)};
}

bool DofT::ComplexDofMap::operator==(const BaseDofMap& o) const {
    if (o.ActualType() != ActualType()) return false;
    auto& a = static_cast<const ComplexDofMap&>(o);
    if (m_spaces.size() != a.m_spaces.size() || m_spaceNumDof != a.m_spaceNumDof) return false; // || m_spaceNumDofTet != a.m_spaceNumDofTet || m_spaceNumDofsTet != a.m_spaceNumDofsTet
    for (uint i = 0; i < m_spaces.size(); ++i){
        if (m_spaces[i].get() == a.m_spaces[i].get()) continue;
        if (!(m_spaces[i].get() && a.m_spaces[i].get() && *m_spaces[i].get() == *a.m_spaces[i].get())) return false;
    }
    return true;
}

DofT::DofMap DofT::DofMap::operator*(const DofMap& o) const{
    uint t1 = ActualType(), t2 = o.ActualType();
    if (t1 == t2){
        if (*this == o){
            if (t1 == static_cast<uint>(BaseDofMap::BaseTypes::VectorType)){
                auto bm = this->target<VectorDofMap>(), om = o.target<VectorDofMap>();
                return DofMap(std::make_shared<VectorDofMap>(bm->m_dim + om->m_dim, bm->base)); 
            } else {
                return DofMap(std::make_shared<VectorDofMap>(2, base()));
            }
        } else if (t1 == static_cast<uint>(BaseDofMap::BaseTypes::ComplexType)){
            auto bm = this->target<ComplexDofMap>(), om = o.target<ComplexDofMap>();
            std::vector<std::shared_ptr<BaseDofMap>> space_union;
            space_union.reserve(bm->m_spaces.size() + om->m_spaces.size());
            if (bm->m_spaces.size() && om->m_spaces.size()){
                space_union.insert(space_union.end(), bm->m_spaces.data(), bm->m_spaces.data() + bm->m_spaces.size() - 1);
                auto tpsp = DofMap(bm->m_spaces.back()) * DofMap(om->m_spaces.front());
                if (tpsp.ActualType() == static_cast<uint>(BaseDofMap::BaseTypes::ComplexType)) {
                    auto tm = tpsp.target<ComplexDofMap>();
                    space_union.insert(space_union.end(), tm->m_spaces.begin(), tm->m_spaces.end());
                } else {
                   space_union.push_back(tpsp.base()); 
                }
                space_union.insert(space_union.end(), om->m_spaces.data()+1, om->m_spaces.data() + om->m_spaces.size());
            } else {
                space_union.insert(space_union.end(), bm->m_spaces.begin(), bm->m_spaces.end());
                space_union.insert(space_union.end(), om->m_spaces.begin(), om->m_spaces.end());
            }
            return DofMap(std::make_shared<ComplexDofMap>(space_union));   
        } else 
            return DofMap(std::make_shared<ComplexDofMap>(std::vector<std::shared_ptr<BaseDofMap>>{this->base(), o.base()}));
    } else {
        if (t1 == static_cast<uint>(BaseDofMap::BaseTypes::ComplexType)){
            auto bm = this->target<ComplexDofMap>();
            auto tpsp = DofMap(bm->m_spaces.back()) * o;
            std::vector<std::shared_ptr<BaseDofMap>> space_union;
            if (tpsp.ActualType() == static_cast<uint>(BaseDofMap::BaseTypes::ComplexType)){
                auto tm = tpsp.target<ComplexDofMap>();
                space_union.reserve(bm->m_spaces.size() + tm->m_spaces.size());
                space_union.insert(space_union.end(), bm->m_spaces.data(), bm->m_spaces.data() + bm->m_spaces.size() - 1);
                space_union.insert(space_union.end(), tm->m_spaces.begin(), tm->m_spaces.end());
            } else {
                space_union.reserve(bm->m_spaces.size());
                space_union.insert(space_union.end(), bm->m_spaces.data(), bm->m_spaces.data() + bm->m_spaces.size() - 1);
                space_union.push_back(tpsp.base());
            }
            return DofMap(std::make_shared<ComplexDofMap>(space_union));
        } else if (t2 == static_cast<uint>(BaseDofMap::BaseTypes::ComplexType)){
            auto om = o.target<ComplexDofMap>();
            auto tpsp = (*this) * DofMap(om->m_spaces.front());
            std::vector<std::shared_ptr<BaseDofMap>> space_union;
            if (tpsp.ActualType() == static_cast<uint>(BaseDofMap::BaseTypes::ComplexType)){
                auto tm = tpsp.target<ComplexDofMap>();
                space_union.reserve(om->m_spaces.size() + tm->m_spaces.size());
                space_union.insert(space_union.end(), tm->m_spaces.begin(), tm->m_spaces.end());
                space_union.insert(space_union.end(), om->m_spaces.data()+1, om->m_spaces.data() + om->m_spaces.size());
            } else {
                space_union.reserve(om->m_spaces.size());
                space_union.push_back(tpsp.base());
                space_union.insert(space_union.end(), om->m_spaces.data()+1, om->m_spaces.data() + om->m_spaces.size());
                
            }
            return DofMap(std::make_shared<ComplexDofMap>(space_union));
        } else if (t1 == static_cast<uint>(BaseDofMap::BaseTypes::VectorType)){
            auto bm = this->target<VectorDofMap>(); 
            if (bm->base.get() == o.base().get() || (bm->base.get() &&o.base().get() && *bm->base.get() == *o.base().get())){
                return DofMap(std::make_shared<VectorDofMap>(bm->m_dim + 1, bm->base));
            }
        } else if (t2 == static_cast<uint>(BaseDofMap::BaseTypes::VectorType)){
            auto om = o.target<VectorDofMap>(); 
            if (om->base.get() == this->base().get() || (om->base.get() &&this->base().get() && *om->base.get() == *this->base().get())){
                return DofMap(std::make_shared<VectorDofMap>(om->m_dim + 1, om->base));
            }
        }
        return DofMap(std::make_shared<ComplexDofMap>(std::vector<std::shared_ptr<BaseDofMap>>{this->base(), o.base()}));
    }
}

DofT::DofMap DofT::pow(const DofMap& d, uint k){
    return DofMap(std::make_shared<VectorDofMap>(k, d.base()));
}

DofT::DofMap DofT::operator^(const DofMap& d, uint k){
    if (d.ActualType() == static_cast<uint>(BaseDofMap::BaseTypes::VectorType)){
        auto dm = d.target<VectorDofMap>();
        return DofMap(std::make_shared<VectorDofMap>(dm->m_dim*k, dm->base));
    } else {
        return DofMap(std::make_shared<VectorDofMap>(k, d.base()));
    }
}

DofT::DofMap DofT::merge(const std::vector<DofMap>& maps){
    std::vector<std::shared_ptr<BaseDofMap>> spaces;
    spaces.reserve(maps.size());
    for (uint i = 0; i < maps.size(); ++i) 
        spaces.push_back(maps[i].base());
    return DofMap(std::make_shared<ComplexDofMap>(spaces));
}

DofT::DofMap DofT::merge_with_simplifications(const std::vector<DofMap>& maps){
    if (maps.empty()) return DofMap();
    DofMap r = maps[0];
    for (uint i = 1; i < maps.size(); ++i) 
        r = r * maps[i];
    return r;
}

bool DofT::NestedDofMapBase::isValidOutputIndex(LocalOrder dof_id) const {
    if (dof_id.etype == UNDEF || !GeomTypeIsValid(dof_id.etype)) return false;
    dof_id.gid -= m_shiftOnTet;
    if (!base()->isValidIndex(dof_id.getTetOrder())) return false;
    dof_id.leid -= m_shiftNumDof[GeomTypeToNum(dof_id.etype)];
    return base()->isValidIndex(dof_id.getGeomOrder());
}

void DofT::NestedDofMap::GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const {
    if (ndims > 0 && *ext_dims == 0){
        view.m_shiftOnTet += m_shiftOnTet;
        for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDof[i] += m_shiftNumDof[i];
        if (ndims == 1) view.set_base(m_base);
        else m_base->GetNestedComponent(ext_dims+1, ndims-1, view);
    } else view.Clear();
}

std::shared_ptr<const DofT::BaseDofMap> DofT::NestedDofMap::GetSubDofMap(const int* ext_dims, int ndims) const {
    if (ndims > 0 && *ext_dims == 0)
        return (ndims == 1) ? m_base : m_base->GetSubDofMap(ext_dims+1, ndims-1);
    else 
        return nullptr; 
}

std::shared_ptr<DofT::BaseDofMap> DofT::NestedDofMap::GetSubDofMap(const int* ext_dims, int ndims) {
    if (ndims > 0 && *ext_dims == 0)
        return (ndims == 1) ? m_base : m_base->GetSubDofMap(ext_dims+1, ndims-1);
    else 
        return nullptr;
}

void DofT::NestedDofMapView::GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const {
    if (ndims > 0 && *ext_dims == 0){
        view.m_shiftOnTet += m_shiftOnTet;
        for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDof[i] += m_shiftNumDof[i];
        if (ndims == 1) view.set_base(m_base);
        else m_base->GetNestedComponent(ext_dims+1, ndims-1, view);
    } else view.Clear();
}

std::shared_ptr<const DofT::BaseDofMap> DofT::NestedDofMapView::GetSubDofMap(const int* ext_dims, int ndims) const {
    if (ndims > 0 && *ext_dims == 0)
        return (ndims == 1) ? m_base->Copy() : m_base->GetSubDofMap(ext_dims+1, ndims-1);
    else 
        return nullptr; 
}

std::shared_ptr<DofT::BaseDofMap> DofT::NestedDofMapView::GetSubDofMap(const int* ext_dims, int ndims) {
    if (ndims > 0 && *ext_dims == 0)
        return (ndims == 1) ? m_base->Copy() : const_cast<BaseDofMap*>(m_base)->GetSubDofMap(ext_dims+1, ndims-1);
    else 
        return nullptr;
}

void DofT::NestedDofMapBase::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const{
    assert(isValidOutputIndex(lo) && "Wrong local index");
    lo.gid -= m_shiftOnTet;
    lo.leid -= m_shiftNumDof[GeomTypeToNum(lo.etype)];
    base()->IncrementByGeomSparsity(sp, lo, preferGeomOrdering);
    LocalOrder end;
    base()->EndByGeomSparsity(end);
    if (lo != end){
        lo.gid += m_shiftOnTet;
        lo.leid += m_shiftNumDof[GeomTypeToNum(lo.etype)];
    } else 
        EndByGeomSparsity(lo);
}

DofT::LocalOrder DofT::NestedDofMapBase::LocalOrderOnTet(TetOrder dof_id) const {
    LocalOrder lo = base()->LocalOrderOnTet(dof_id);
    lo.leid += m_shiftNumDof[GeomTypeToNum(lo.etype)];
    lo.gid += m_shiftOnTet;
    return lo;
} 
DofT::LocalOrder DofT::NestedDofMapBase::LocalOrderOnTet(LocGeomOrder dof_id) const {
    LocalOrder lo = base()->LocalOrderOnTet(dof_id);
    lo.leid += m_shiftNumDof[GeomTypeToNum(lo.etype)];
    lo.gid += m_shiftOnTet;
    return lo;
} 

bool DofT::NestedDofMapBase::operator==(const BaseDofMap& o) const {
    if (o.ActualType() != ActualType()) return false;
    auto& a = static_cast<const NestedDofMapBase&>(o);
    if (m_shiftOnTet != a.m_shiftOnTet || m_shiftNumDof != a.m_shiftNumDof) return false;
    return base() == a.base() || (base() && a.base() && *base() == *a.base());
}