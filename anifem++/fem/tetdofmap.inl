//
// Created by Liogky Alexey on 10.01.2023.
//

namespace Ani{
namespace DofT{
inline TetGeomSparsity operator&(TetGeomSparsity a, TetGeomSparsity b){ 
    TetGeomSparsity res; 
    for (uchar d = 0; d < 4; ++d) 
        res.elems[d] = a.elems[d] & b.elems[d];
    return res;
}
inline TetGeomSparsity operator|(TetGeomSparsity a, TetGeomSparsity b){ 
    TetGeomSparsity res; 
    for (uchar d = 0; d < 4; ++d) 
        res.elems[d] = a.elems[d] | b.elems[d];
    return res;
}
inline TetGeomSparsity operator^(TetGeomSparsity a, TetGeomSparsity b){ 
    TetGeomSparsity res; 
    for (uchar d = 0; d < 4; ++d) 
        res.elems[d] = a.elems[d] ^ b.elems[d];
    return res;
}
inline TetGeomSparsity operator~(TetGeomSparsity a){ 
    TetGeomSparsity res; 
    for (uchar d = 0; d < 4; ++d) 
        res.elems[d] = ~a.elems[d];
    return res;
}

inline bool BaseDofMap::isValidIndex(LocalOrder dof_id, bool with_sym_order) const {
    auto r = isValidIndex(dof_id.getTetOrder()) && isValidIndex(dof_id.getGeomOrder());
    if (r && with_sym_order){
        auto m = SymComponents(dof_id.etype);
        r = ((m | DofSymmetries::MASK_UNDEF) || dof_id.lsid >= DofSymmetries::symmetries_amount(dof_id.etype)) ? false : (m | (1 << dof_id.stype));
    } 
    return r; 
}

inline void DofSymmetries::set(std::array<uint, 1> n_syms, std::array<uint, 2> eu_syms, std::array<uint, 2> eo_syms, std::array<uint, 3> fu_syms, std::array<uint, 3> fo_syms, std::array<uint, 5> c_syms){
    m_sym[0] = n_syms[0];
    m_sym[1] = eu_syms[0]; m_sym[2] = eu_syms[1]; m_sym[3] = eo_syms[0]; m_sym[4] = eo_syms[1];
    m_sym[5] = fu_syms[0]; m_sym[6] = fu_syms[1]; m_sym[7] = fu_syms[2]; m_sym[8] = fo_syms[0]; m_sym[9] = fo_syms[1]; m_sym[10] = fo_syms[2];
    m_sym[11] = c_syms[0]; m_sym[12] = c_syms[1];  m_sym[13] = c_syms[2]; m_sym[14] = c_syms[3]; m_sym[15] = c_syms[4];
}
inline uint DofSymmetries::get(uchar etype, uchar sym_num) const {
    if (!isInited()) return uint(-1);
    auto t = GeomTypeToNum(etype);
    const static uchar offs[7]{0, 1, 3, 5, 8, 11, 16};
    assert(!(t < 0 || t > 6 || sym_num >= offs[t+1] - offs[t]) && "Wrong arguments");
    return m_sym[offs[t]+sym_num];
}
inline void DofSymmetries::add(uchar etype, uchar sym_num, uint count){
    if (!isInited()) m_sym[0] = 0;
    auto t = GeomTypeToNum(etype);
    const static uchar offs[7]{0, 1, 3, 5, 8, 11, 16};
    assert(!(t < 0 || t > 6 || sym_num >= offs[t+1] - offs[t]) && "Wrong arguments");
    m_sym[offs[t]+sym_num] += count;
}
inline uchar DofSymmetries::symmetries_amount_by_geom_num(uchar geom_num){
    assert(geom_num < 6 && "Wrong argument");
    const static uchar amount[6]{1, 2, 2, 3, 3, 5};
    return amount[geom_num];
}
inline uchar DofSymmetries::symmetries_amount(uchar etype){ return symmetries_amount_by_geom_num(GeomTypeToNum(etype)); }
inline uchar DofSymmetries::symmetry_volume_by_geom_num(uchar geom_num, uchar sym_num){
    const static uchar offs[7]{0, 1, 3, 5, 8, 11, 16};
    const static uchar vols[16]{1, 1, 2, 1, 2, 1, 3, 6, 1, 3, 6, 1, 4, 6, 12, 24};
    assert(!(geom_num >= 6 || sym_num >= offs[geom_num+1] - offs[geom_num]) && "Wrong arguments");
    return vols[offs[geom_num]+sym_num];
}
inline uchar DofSymmetries::symmetry_volume(uchar etype, uchar sym_num){
    assert(sym_num > 0 && "Wrong argument");
    return symmetry_volume_by_geom_num(GeomTypeToNum(etype), sym_num);
}
inline bool DofSymmetries::isInited() const{ return m_sym[0] != uint(-1); }
inline uchar DofSymmetries::GetSymMask_by_geom_num(uchar geom_num) const{
    if (!isInited()) return MASK_UNDEF;
    const static uchar offs[7]{0, 1, 3, 5, 8, 11, 16};
    uchar mask = 0;
    for (int i = 0, cnt = offs[geom_num + 1] - offs[geom_num]; i < cnt; ++i) if (m_sym[offs[geom_num] + i] > 0) 
        mask |= (1 << i); 
    return mask;    
}
inline LocSymOrder DofSymmetries::GetLocSymOrder_by_geom_num(uchar geom_num, uint loc_elem_id) const{
    if (!isInited()) return LocSymOrder();
    const static uchar offs[7]{0, 1, 3, 5, 8, 11, 16};
    uint ndofs = 0;
    int i = offs[geom_num];
    uchar vol = 0;
    for (; i < offs[geom_num+1] && loc_elem_id >= ndofs; ++i) {
        vol = symmetry_volume_by_geom_num(geom_num, i - offs[geom_num]);
        ndofs += vol*m_sym[i];
    }
    return LocSymOrder(i - (offs[geom_num]+1),  (loc_elem_id - (ndofs - vol*m_sym[i]))%vol);
}
inline uchar DofSymmetries::index_on_reorderd_elem(uchar etype, uchar nelem, uchar sym_num, uchar loc_symmetry_ind, const uchar* node_permutation){
    if (etype == 0 || sym_num == 0) return loc_symmetry_ind;
    auto dim = GeomTypeDim(etype);
    if (dim == 1) {
        const static std::array<uchar, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
        return (node_permutation[lookup_nds[2*nelem]] < node_permutation[lookup_nds[2*nelem+1]]) ? loc_symmetry_ind : ((loc_symmetry_ind+1)%2);   
    } 
    if (dim == 2){
        uchar tria[3]{node_permutation[nelem], node_permutation[(nelem+1)%4], node_permutation[(nelem+2)%4]};
        {
            uchar i = 0, j = 0;
            for (uchar k = 1; k < 3; ++k){
                if (tria[k] < tria[i]) i = k;
                else if (tria[k] > tria[j]) j = k;
            }
            tria[i] = 0, tria[j] = 2, tria[3-(i+j)] = 1;
        }
        if (sym_num == 1)
            return tria[loc_symmetry_ind];
        //abc bca cab acb cba bac
        const static std::array<uchar, 18> lookup_inds{0,1,2,  1,2,0,  2,0,1,  0,2,1,  2,1,0,  1,0,2};
        uchar from[3]{lookup_inds[3*loc_symmetry_ind], lookup_inds[3*loc_symmetry_ind+1], lookup_inds[3*loc_symmetry_ind+2]};
        uchar to[3];
        for (uchar k = 0; k < 3; ++k)
            to[tria[k]] = from[k];
        uchar change = 0;
        for (uchar k = 0; k < 3; ++k)
            if (to[k] > to[(k+1)%3]) ++change; 
        return change == 1 ? to[0] : (5 - to[1]);    
    }
    //dim == 3
    if (sym_num == 1)
        return node_permutation[loc_symmetry_ind];
    //aabb abab baab abba baba bbaa    
    const static std::array<uchar, 6*4> lookup_inds_s6{
        0,0,1,1, 0,1,0,1, 1,0,0,1, 0,1,1,0, 1,0,1,0, 1,1,0,0
    };   
    // abcc bcac cabc acbc cbac bacc  bcca cacb accb cbca ccba ccab
    const static std::array<uchar, 12*4> lookup_inds_s12{
        1,2,0,0, 2,0,1,0, 0,1,2,0, 1,0,2,0, 0,2,1,0, 2,1,0,0,  
        2,0,0,1, 0,1,0,2, 1,0,0,2, 0,2,0,1, 0,0,2,1, 0,0,1,2, 
    }; 
    //abcd bcad cabd acbd cbad bacd  abdc bcda cadb acdb cbda badc
    //adbc bdca cdab adcb cdba bdac  dabc dbca dcab dacb dcba dbac
    const static std::array<uchar, 24*4> lookup_inds_s24{
        0,1,2,3, 1,2,0,3, 2,0,1,3, 0,2,1,3, 2,1,0,3, 1,0,2,3,  
        0,1,3,2, 1,2,3,0, 2,0,3,1, 0,2,3,1, 2,1,3,0, 1,0,3,2,
        0,3,1,2, 1,3,2,0, 2,3,0,1, 0,3,2,1, 2,3,1,0, 1,3,0,2,  
        3,0,1,2, 3,1,2,0, 3,2,0,1, 3,0,2,1, 3,2,1,0, 3,1,0,2,
    };
    const static uchar quick_ind_s6[10]{0, 25, 1, 3, 25, 25, 2, 4, 25, 5};
    const static uchar quick_ind_s12[19]{5, 0, 1, 25, 4, 3, 2, 6, 25, 9, 25, 25, 25, 10, 25, 8, 7, 25, 11};
    const static uchar quick_ind_s24[44]{22, 19, 25, 25, 25, 16, 25, 10, 20, 25, 21, 13, 7, 14, 25, 25, 8, 23, 18, 25, 25, 25, 25, 25, 25, 15, 9, 17, 25, 25, 11, 4, 2, 12, 25, 6, 1, 25, 5, 25, 25, 25, 3, 0};
    const uchar* lookup_inds[3]{lookup_inds_s6.data(), lookup_inds_s12.data(), lookup_inds_s24.data()};
    const uchar* quick_ind[3]{quick_ind_s6, quick_ind_s12, quick_ind_s24};
    int qid = sym_num - 2;
    uchar from[4]{lookup_inds[qid][4*loc_symmetry_ind], lookup_inds[qid][4*loc_symmetry_ind+1], lookup_inds[qid][4*loc_symmetry_ind+2], lookup_inds[qid][4*loc_symmetry_ind+3]};
    uchar to[4];
    for (uchar k = 0; k < 4; ++k)
        to[node_permutation[k]] = from[k];
    uchar unique_hash = 0;
    switch(qid){
        case 0: unique_hash = (to[3]<<0) + (to[2]<<1) + (to[1]<<2) + (to[0]<<3) - 3; break;
        case 1: unique_hash = to[0] + (to[1] << 1) + (to[2] << 2) + (to[3] << 3) + to[3] - 4; break;
        case 2: unique_hash = 5*to[1] + 6*to[2] + 14*to[3] - 16; break;
        default:{ }
    }
    return quick_ind[qid][unique_hash];  
}

namespace FemComDetails{
    template<bool isDofMapTDerivedFromBaseDofMap, typename DofMapT>
    struct VectorDofMapCImpl{};
    template<typename DofMapT>
    struct VectorDofMapCImpl<true, DofMapT>: public BaseDofMap{
        using Base = DofMapT;

        DofMapT m_base;
        int m_dim = 0;
        VectorDofMapCImpl() = default;
        VectorDofMapCImpl(int dim, DofMapT base): m_base{base}, m_dim{dim} {}
        uint ActualType() const override { return static_cast<uint>(BaseTypes::VectorTemplateType); }
        uint NumDof(uchar etype) const override { return m_dim * m_base.NumDof(etype); }
        uint NumDofOnTet(uchar etype) const override { return m_dim * m_base.NumDofOnTet(etype); }
        uint NumDofOnTet() const override { return m_dim * m_base.NumDofOnTet(); }
        std::array<uint, NGEOM_TYPES> NumDofs() const override{ auto r = m_base.NumDofs(); for(auto& x: r) x *= m_dim; return r; }
        std::array<uint, NGEOM_TYPES> NumDofsOnTet() const override { auto r = m_base.NumDofsOnTet(); for(auto& x: r) x *= m_dim; return r; }
        uint TypeOnTet(uint dof) const override { return m_base.TypeOnTet(dof % m_base.NumDofOnTet()); }
        bool DefinedOn(uchar etype) const override { return m_base.DefinedOn(etype); }
        uint GetGeomMask() const { return m_base.GetGeomMask(); }
        uint NestedDim() const override { return m_dim; }

        LocalOrder LocalOrderOnTet(TetOrder dof_id) const override;
        uint TetDofID(LocGeomOrder dof_id) const override;
        uchar SymComponents(uchar etype) const override { return m_base.SymComponents(etype); }
        std::pair<uint, LocSymOrder> TetDofIDExt(LocGeomOrder dof_id) const override;
        void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const;
        std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims);
        bool operator==(const BaseDofMap& other) const override;
        void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const;

        std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<VectorDofMapCImpl<true, DofMapT>>(*this); }
    };
    template<typename DofMapT>
    LocalOrder VectorDofMapCImpl<true, DofMapT>::LocalOrderOnTet(TetOrder dof_id) const{
        assert(isValidIndex(dof_id) && "Wrong dof number");
        auto base_dof_on_tet = m_base.NumDofOnTet();
        LocalOrder lo = m_base.LocalOrderOnTet(TetOrder(dof_id.gid % base_dof_on_tet));
        lo.leid += (dof_id.gid / base_dof_on_tet) * m_base.NumDof(lo.etype);
        lo.gid = dof_id.gid;
        return lo;
    }

    template<typename DofMapT>
    uint VectorDofMapCImpl<true, DofMapT>::TetDofID(LocGeomOrder dof_id) const{
        assert(isValidIndex(dof_id) && "Wrong dof number");
        int nOdf = m_base.NumDofOnTet(), lOdf = m_base.NumDofOnTet(dof_id.etype);
        int component = dof_id.leid / lOdf;
        return component * nOdf + m_base.TetDofID(LocGeomOrder(dof_id.etype, dof_id.nelem, dof_id.leid % lOdf));
    }
    template<typename DofMapT>
    std::pair<uint, LocSymOrder> VectorDofMapCImpl<true, DofMapT>::TetDofIDExt(LocGeomOrder dof_id) const {
        assert(isValidIndex(dof_id) && "Wrong dof number");
        int nOdf = m_base.NumDofOnTet(), lOdf = m_base.NumDofOnTet(dof_id.etype);
        int component = dof_id.leid / lOdf;
        auto id = m_base.TetDofIDExt(LocGeomOrder(dof_id.etype, dof_id.nelem, dof_id.leid % lOdf));
        return {component * nOdf + id.first, id.second};
    }

    template<typename DofMapT>
    void VectorDofMapCImpl<true, DofMapT>::GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const{
        if (ndims == 0 || ext_dims[0] < 0 || ext_dims[0] >= m_dim) {
            view.Clear();
            return;
        }
        auto dsz = m_base.NumDofs();
        for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDof[i] += ext_dims[0]*dsz[i];
        view.m_shiftOnTet += ext_dims[0]*m_base.NumDofOnTet();
        if (ndims == 1)
            view.set_base(&m_base);
        else 
            m_base.GetNestedComponent(ext_dims+1, ndims-1, view);
    }

    template<typename DofMapT>
    std::shared_ptr<BaseDofMap> VectorDofMapCImpl<true, DofMapT>::GetSubDofMap(const int* ext_dims, int ndims){
        if (ndims > 0 && *ext_dims >= 0 && *ext_dims < m_dim){
            return (ndims == 1) ? m_base.Copy() : m_base.GetSubDofMap(ext_dims+1, ndims-1);
        } else 
            return nullptr;
    }

    template<typename DofMapT>
    bool VectorDofMapCImpl<true, DofMapT>::operator==(const BaseDofMap& o) const{
        if (o.ActualType() != ActualType()) return false;
        auto& a = static_cast<const VectorDofMapCImpl<true, DofMapT>&>(o);
        if (m_dim != a.m_dim) return false;
        return m_base == a.m_base;
    }

    template<typename DofMapT>
    void VectorDofMapCImpl<true, DofMapT>::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const{
        assert(isValidIndex(lo) && "Wrong index");
        int nOdf = m_base.NumDofOnTet(), lOdf = m_base.NumDof(lo.etype);
        uint dim = lo.gid / nOdf;
        uint lcomp = lo.gid % nOdf; 
        LocalOrder ll, lend;
        ll.gid = lcomp;
        ll.etype = lo.etype;
        ll.nelem = lo.nelem;
        ll.leid = lo.leid % lOdf;
        ll.stype = lo.stype, ll.lsid = lo.lsid;
        m_base.EndByGeomSparsity(lend);
        if (!preferGeomOrdering){
            m_base.IncrementByGeomSparsity(sp, ll, false);
            if (ll != lend){
                if (ll.etype != lo.etype) lOdf = m_base.NumDof(ll.etype);
                lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf, ll.stype, ll.lsid);
                return;
            } else {
                ++dim;
                if (dim != static_cast<uint>(m_dim)){
                    m_base.BeginByGeomSparsity(sp, ll, false);
                    if (ll != lend){
                        if (ll.etype != lo.etype) lOdf = m_base.NumDof(ll.etype);
                        lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf, ll.stype, ll.lsid);
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
            m_base.IncrementByGeomSparsity(ss, ll, true);
            if (ll != lend){
                if (ll.etype != lo.etype) lOdf = m_base.NumDof(ll.etype);
                lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf, ll.stype, ll.lsid);
                return;
            } else {
                ++dim;
                if (dim != static_cast<uint>(m_dim)){
                    LocGeomOrder lgo(lo.etype, lo.nelem, 0);
                    if (lo.etype == EDGE_ORIENT){
                        auto eodf = m_base.NumDof(EDGE_UNORIENT);
                        if (eodf > 0){
                            lgo.etype = EDGE_UNORIENT;
                            lOdf = eodf;
                        }
                    }
                    else if (lo.etype == FACE_ORIENT){
                        auto fodf = m_base.NumDof(FACE_UNORIENT);
                        if (fodf > 0){
                            lgo.etype = FACE_UNORIENT;
                            lOdf = fodf;
                        }
                    }
                    auto id = m_base.TetDofIDExt(lgo);
                    lo = LocalOrder(id.first + dim*nOdf, lgo.etype, lgo.nelem, lgo.leid + dim*lOdf, id.second.stype, id.second.lsid);
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
                    m_base.BeginByGeomSparsity(ss, ll, true);
                    if (ll != lend){
                        lOdf = m_base.NumDof(ll.etype);
                        lo = LocalOrder(ll.gid + dim*nOdf, ll.etype, ll.nelem, ll.leid + dim*lOdf, ll.stype, ll.lsid);
                    } else 
                        EndByGeomSparsity(lo);
                    return;
                }
            }
        }
    }

    template<bool isDofMapTDerivedFromBaseDofMap, typename... DofMapT>
    struct ComplexDofMapCImpl {};

    template<typename... DofMapT>
    struct ComplexDofMapCImpl<true, DofMapT...>: public BaseDofMap, std::tuple<DofMapT ...>{
        using Base = std::tuple<DofMapT ...>;
        using Size = std::integral_constant<std::size_t, sizeof...(DofMapT)>;

        std::array<std::array<uint, Size::value+1>, NGEOM_TYPES> m_spaceNumDofTet;
        std::array<std::array<uint, Size::value+1>, NGEOM_TYPES> m_spaceNumDof;
        std::array<uint, Size::value+1> m_spaceNumDofsTet;

        ComplexDofMapCImpl();
        ComplexDofMapCImpl(const DofMapT&... args);
        uint NestedDim() const override { return Size::value; }
        uint ActualType() const override { return static_cast<uint>(BaseTypes::ComplexTemplateType); }
        uint NumDof(uchar etype) const override { auto t = GeomTypeToNum(etype); return m_spaceNumDof[t][Size::value] - m_spaceNumDof[t][0]; }
        uint NumDofOnTet(uchar etype) const override { auto t = GeomTypeToNum(etype); return m_spaceNumDofTet[t][Size::value] - m_spaceNumDofTet[t][0]; }

        std::array<uint, NGEOM_TYPES> NumDofs() const override;
        std::array<uint, NGEOM_TYPES> NumDofsOnTet() const override;
        LocalOrder LocalOrderOnTet(TetOrder dof_id) const override;
        uint TypeOnTet(uint dof) const override ;
        uint TetDofID(LocGeomOrder dof_id) const override;
        uchar SymComponents(uchar etype) const override;
        std::pair<uint, LocSymOrder> TetDofIDExt(LocGeomOrder dof_id) const override;
        uint GetGeomMask() const override ;
        void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const override;
        std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) override;
        bool operator==(const BaseDofMap& other) const override;
        std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<ComplexDofMapCImpl<true, DofMapT...>>(*this); }
        
        void BeginByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const override;
        void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const override;

        template<std::size_t I, std::size_t N>
        struct Getter{
            static inline void Apply(const ComplexDofMapCImpl<true, DofMapT...>& t, const BaseDofMap** out){
                *out = &std::get<I>(static_cast<const ComplexDofMapCImpl<true, DofMapT...>::Base&>(t));
                Getter<I+1, N>::Apply(t, out+1);
            }
            static inline bool isSame(const ComplexDofMapCImpl<true, DofMapT...>& t1, const ComplexDofMapCImpl<true, DofMapT...>& t2){
                bool b = std::get<I>(static_cast<const ComplexDofMapCImpl<true, DofMapT...>::Base&>(t1)) == std::get<I>(static_cast<const ComplexDofMapCImpl<true, DofMapT...>::Base&>(t2));
                return b ? Getter<I+1, N>::isSame(t1, t2) : false;
            }
        };
        template<std::size_t N>
        struct Getter<N, N>{
            static inline void Apply(const ComplexDofMapCImpl<true, DofMapT...>& t, const BaseDofMap** out){(void) t; (void) out; }
            static inline bool isSame(const ComplexDofMapCImpl<true, DofMapT...>& t1, const ComplexDofMapCImpl<true, DofMapT...>& t2){ (void) t1, (void) t2; return true; }
        };
        
        template<std::size_t I>
        ComplexDofMapCImpl<true, DofMapT...>& set(const typename std::tuple_element<I, Base>::type& v);
        std::array<const BaseDofMap*, Size::value> content() const{
            std::array<const BaseDofMap*, Size::value> res;
            Getter<0, Size::value>::Apply(*this, res.data());
            return res;
        }
        std::array<BaseDofMap*, Size::value> content(){
            std::array<BaseDofMap*, Size::value> res;
            Getter<0, Size::value>::Apply(*this, const_cast<const BaseDofMap**>(res.data()));
            return res;
        }
    };

    template<typename... DofMapT> 
    template<std::size_t I>
    ComplexDofMapCImpl<true, DofMapT...>& ComplexDofMapCImpl<true, DofMapT...>::set(const typename std::tuple_element<I, Base>::type& v){ 
        for (uint t = 0; t < NGEOM_TYPES; ++t) {
            auto etype = NumToGeomType(t);
            uint nd = v.NumDof(etype);
            uint ndt = v.NumDofOnTet(etype);
            for (uint i = I; i < Size::value; ++i){
                m_spaceNumDofTet[t][i+1] += ndt;
                m_spaceNumDof[t][i+1] += nd;
            }
        }
        uint ndts = v.NumDofOnTet();
        for (uint i = I; i < Size::value; ++i)
            m_spaceNumDofsTet[i+1] += ndts;
        std::get<I>(static_cast<Base&>(*this)) = v;
        return *this;
    }

    template<typename... DofMapT>
    ComplexDofMapCImpl<true, DofMapT...>::ComplexDofMapCImpl(): std::tuple<DofMapT ...>() {
        for (uint t = 0; t < NGEOM_TYPES; ++t) {
            std::fill(m_spaceNumDofTet[t].begin(), m_spaceNumDofTet[t].end(), 0);
            std::fill(m_spaceNumDof[t].begin(), m_spaceNumDof[t].end(), 0);
        }
        std::fill(m_spaceNumDofsTet.begin(), m_spaceNumDofsTet.end(), 0);
    } 
    template<typename... DofMapT>
    ComplexDofMapCImpl<true, DofMapT...>::ComplexDofMapCImpl(const DofMapT&... args): std::tuple<DofMapT...>(args...) {
        auto r = content();
        for (uint t = 0; t < NGEOM_TYPES; ++t) {
            auto etype = NumToGeomType(t);
            m_spaceNumDofTet[t][0] = m_spaceNumDof[t][0] = 0;
            for (uint i = 0; i < r.size(); ++i) {
                m_spaceNumDofTet[t][i+1] = m_spaceNumDofTet[t][i] + r[i]->NumDofOnTet(etype);
                m_spaceNumDof[t][i+1] = m_spaceNumDof[t][i] + r[i]->NumDof(etype);
            }
        }
        m_spaceNumDofsTet[0] = 0;
        for (uint i = 0; i < r.size(); ++i) {
            m_spaceNumDofsTet[i+1] = m_spaceNumDofsTet[i] + r[i]->NumDofOnTet();
        }
    } 
    template<typename... DofMapT>
    std::array<uint, NGEOM_TYPES> ComplexDofMapCImpl<true, DofMapT...>::NumDofs() const {
        std::array<uint, NGEOM_TYPES> res = {0};
        for (int t = 0; t < NGEOM_TYPES; ++t) res[t] = m_spaceNumDof[t][Size::value] - m_spaceNumDof[t][0];
        return res;
    }  
    template<typename... DofMapT>
    std::array<uint, NGEOM_TYPES> ComplexDofMapCImpl<true, DofMapT...>::NumDofsOnTet() const {
        std::array<uint, NGEOM_TYPES> res = {0};
        for (int t = 0; t < NGEOM_TYPES; ++t) res[t] = m_spaceNumDofTet[t][Size::value] - m_spaceNumDofTet[t][0];
        return res;
    }
    template<typename... DofMapT>
    LocalOrder ComplexDofMapCImpl<true, DofMapT...>::LocalOrderOnTet(TetOrder dof_id) const {
        assert(isValidIndex(dof_id) && "Wrong dof number");
        auto s = content();
        int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof_id.gid) - m_spaceNumDofsTet.data() - 1;
        LocalOrder lo = s[vid]->LocalOrderOnTet(TetOrder(dof_id.gid - m_spaceNumDofsTet[vid]));
        lo.leid += m_spaceNumDof[GeomTypeToNum(lo.etype)][vid];
        lo.gid = dof_id.gid;
        return lo;
    }
    template<typename... DofMapT>
    uint ComplexDofMapCImpl<true, DofMapT...>::TypeOnTet(uint dof) const  {
        assert(isValidIndex(TetOrder(dof)) && "Wrong dof number");
        auto s = content();
        int vid = std::upper_bound(m_spaceNumDofsTet.data(), m_spaceNumDofsTet.data() + m_spaceNumDofsTet.size(), dof) - m_spaceNumDofsTet.data() - 1;
        return s[vid]->TypeOnTet(dof - m_spaceNumDofsTet[vid]);
    }
    template<typename... DofMapT>
    uint ComplexDofMapCImpl<true, DofMapT...>::TetDofID(LocGeomOrder dof_id) const {
        assert(isValidIndex(dof_id) && "Wrong index");
        auto s = content();
        auto num = GeomTypeToNum(dof_id.etype);
        int vid = std::upper_bound(m_spaceNumDof[num].data(), m_spaceNumDof[num].data() + m_spaceNumDof[num].size(), dof_id.leid) - m_spaceNumDof[num].data() - 1;
        dof_id.leid -= m_spaceNumDof[num][vid];
        return m_spaceNumDofsTet[vid] + s[vid]->TetDofID(dof_id);
    }
    template<typename... DofMapT>
    std::pair<uint, LocSymOrder> ComplexDofMapCImpl<true, DofMapT...>::TetDofIDExt(LocGeomOrder dof_id) const{
        assert(isValidIndex(dof_id) && "Wrong index");
        auto s = content();
        auto num = GeomTypeToNum(dof_id.etype);
        int vid = std::upper_bound(m_spaceNumDof[num].data(), m_spaceNumDof[num].data() + m_spaceNumDof[num].size(), dof_id.leid) - m_spaceNumDof[num].data() - 1;
        dof_id.leid -= m_spaceNumDof[num][vid];
        auto id = s[vid]->TetDofIDExt(dof_id);
        return {m_spaceNumDofsTet[vid] + id.first, id.second};
    }
    template<typename... DofMapT>
    uchar ComplexDofMapCImpl<true, DofMapT...>::SymComponents(uchar etype) const{
        auto s = content();
        uchar mask = 0;
        for (auto ss: s) 
            mask |= ss->SymComponents(etype);
        return mask;
    }
    template<typename... DofMapT>
    uint ComplexDofMapCImpl<true, DofMapT...>::GetGeomMask() const  {
        uint res = UNDEF;
        for (int i = 0; i < NGEOM_TYPES; ++i){ 
            if (m_spaceNumDof[i][Size::value] - m_spaceNumDof[i][0] > 0) 
                res |= NumToGeomType(i); 
        }
        return res;
    }
    template<typename... DofMapT>
    void ComplexDofMapCImpl<true, DofMapT...>::GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const {
        if (ndims <= 0 || ext_dims[0] < 0 || ext_dims[0] >= static_cast<int>(Size::value)){
            view.Clear();
            return;
        }
        for (int i = 0; i < NGEOM_TYPES; ++i) view.m_shiftNumDof[i] += m_spaceNumDof[i][ext_dims[0]] - m_spaceNumDof[i][0];
        view.m_shiftOnTet += m_spaceNumDofsTet[ext_dims[0]] - m_spaceNumDofsTet[0];
        auto s = content();
        if (ndims == 1)
            view.set_base(s[ext_dims[0]]);
        else 
            s[ext_dims[0]]->GetNestedComponent(ext_dims+1, ndims-1, view); 
    }
    template<typename... DofMapT>
    std::shared_ptr<BaseDofMap> ComplexDofMapCImpl<true, DofMapT...>::GetSubDofMap(const int* ext_dims, int ndims) {
        if (ndims <= 0 || ext_dims[0] < 0 || ext_dims[0] >= static_cast<int>(Size::value))
            return nullptr;
        auto s = content();    
        return ndims == 1 ? s[ext_dims[0]]->Copy() : s[ext_dims[0]]->GetSubDofMap(ext_dims+1, ndims-1);
    }
    template<typename... DofMapT>
    bool ComplexDofMapCImpl<true, DofMapT...>::operator==(const BaseDofMap& o) const {
        if (o.ActualType() != ActualType()) return false;
        auto& a = static_cast<const ComplexDofMapCImpl<true, DofMapT...>&>(o);
        return Getter<0, Size::value>::isSame(*this, a);
    }
    template<typename... DofMapT>
    void ComplexDofMapCImpl<true, DofMapT...>::BeginByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const {
        auto s = content();
        if (!preferGeomOrdering){
            for (uint dim = 0; dim < s.size(); ++dim){
                LocalOrder ll, lend;
                s[dim]->EndByGeomSparsity(lend);
                s[dim]->BeginByGeomSparsity(sp, ll, preferGeomOrdering);
                if (ll != lend){
                    lo = LocalOrder(ll.gid + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[GeomTypeToNum(ll.etype)][dim], ll.stype, ll.lsid);
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
                for (uint dim = 0; dim < s.size(); ++dim) if (s[dim]->DefinedOn(etype)){
                    s[dim]->BeginByGeomSparsity(sp, lo, preferGeomOrdering);
                    lo.gid += m_spaceNumDofsTet[dim], lo.leid += m_spaceNumDof[num][dim];
                    return;
                }
            }
            EndByGeomSparsity(lo);
            return;
        }
    }
    template<typename... DofMapT>
    void ComplexDofMapCImpl<true, DofMapT...>::IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering) const{
        assert(isValidIndex(lo) && "Wrong index");
        auto s = content();
        auto num = GeomTypeToNum(lo.etype);
        uint dim = std::upper_bound(m_spaceNumDof[num].data(), m_spaceNumDof[num].data() + m_spaceNumDof[num].size(), lo.leid) - m_spaceNumDof[num].data() - 1;
        LocalOrder ll(lo.gid - m_spaceNumDofsTet[dim], lo.etype, lo.nelem, lo.leid - m_spaceNumDof[num][dim], lo.stype, lo.lsid); 
        LocalOrder lend;
        s[dim]->EndByGeomSparsity(lend);
        if (!preferGeomOrdering){
            s[dim]->IncrementByGeomSparsity(sp, ll, preferGeomOrdering);
            if (ll != lend){
                if (ll.etype != lo.etype) num = GeomTypeToNum(ll.etype);
                lo = LocalOrder(ll.gid + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[num][dim], ll.stype, ll.lsid);
                return;
            } else {
                ++dim;
                if (dim != s.size()){
                    s[dim]->EndByGeomSparsity(lend);
                    s[dim]->BeginByGeomSparsity(sp, ll, preferGeomOrdering);
                    if (ll != lend){
                        if (ll.etype != lo.etype) num = GeomTypeToNum(ll.etype);
                        lo = LocalOrder(ll.gid + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[num][dim], ll.stype, ll.lsid);
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
            s[dim]->IncrementByGeomSparsity(ss, lgo, preferGeomOrdering);
            // if (ll.etype != lo.etype) num = GeomTypeToNum(ll.etype);
            if (lgo != lend){
                if (lgo.etype != lo.etype) num = GeomTypeToNum(lgo.etype);
                lo = LocalOrder(lgo.gid + m_spaceNumDofsTet[dim], lgo.etype, lgo.nelem, lgo.leid + m_spaceNumDof[num][dim], lgo.stype, lgo.lsid);
                return;
            } else {
                do { ++dim; } while( dim < s.size() && !s[dim]->DefinedOn(ll.etype));
                if (dim != s.size()){
                    ll.leid = 0;
                    LocGeomOrder lgo(ll.etype, ll.nelem, ll.leid);
                    num = GeomTypeToNum(ll.etype);
                    lo = LocalOrder(s[dim]->TetDofID(lgo) + m_spaceNumDofsTet[dim], ll.etype, ll.nelem, ll.leid + m_spaceNumDof[num][dim], ll.stype, ll.lsid);
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

};
};

    template<typename Scalar>
    inline void applyDirMatrix(const DofT::BaseDofMap& trial_map, DenseMatrix <Scalar>& A, const DofT::TetGeomSparsity& sp){
        for (auto it = trial_map.beginBySparsity(sp); it != trial_map.endBySparsity(); ++it)
            applyDirMatrix(A, it->gid);
    }
    template<typename Scalar>
    inline void applyDirMatrix(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix <Scalar>& A, const DofT::TetGeomSparsity& sp){
        if (trial_map == test_map){
            applyDirMatrix(trial_map, A, sp);
            return;
        }
        for (auto jt = trial_map.beginBySparsity(sp, true), it = test_map.beginBySparsity(sp, true); jt != trial_map.endBySparsity() && it != test_map.endBySparsity(); ++it, ++jt){
            uint i = it->gid, j = jt->gid;
            std::fill(A.data + j * A.nRow, A.data + (j+1) * A.nRow, 0);
            for (int l = 0; l < A.nRow; ++l) A(i, l) = 0;
            A(i, j) = 1.0;
        }
    }
    template<typename Scalar>
    inline void applyDirResidual(const DofT::BaseDofMap& test_map, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp) {
        for (auto it = test_map.beginBySparsity(sp); it != test_map.endBySparsity(); ++it)
            applyDirResidual(F, it->gid);
    }
    template<typename Scalar>
    inline void applyDirResidual(const DofT::BaseDofMap& trial_map, DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp) {
        for (auto it = trial_map.beginBySparsity(sp); it != trial_map.endBySparsity(); ++it){
            applyDirMatrix(A, it->gid);
            applyDirResidual(F, it->gid);
        }
    }
    template<typename Scalar>
    inline void applyDirResidual(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp){
        applyDirMatrix(trial_map, test_map, A, sp);
        applyDirResidual(test_map, F, sp);
    }
    template<typename Scalar>
    inline void applyConstantDirByDofs(const DofT::BaseDofMap& trial_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, Scalar bc){
        for (auto it = trial_map.beginBySparsity(sp); it != trial_map.endBySparsity(); ++it){
            applyDir(A, F, it->gid, bc);
        }
    }
    template<typename Scalar>
    inline void applyDirByDofs(const DofT::BaseDofMap& trial_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, const ArrayView<const Scalar>& dofs){
        for (auto it = trial_map.beginBySparsity(sp); it != trial_map.endBySparsity(); ++it){
            applyDir(A, F, it->gid, dofs[it->gid]);
        }
    }
    template<typename Scalar>
    inline void applyConstantDirByDofs(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, Scalar bc){
        if (trial_map == test_map){
            applyConstantDirByDofs(trial_map, A, F, sp, bc);
            return;
        }
        for (auto jt = trial_map.beginBySparsity(sp, true), it = test_map.beginBySparsity(sp, true); jt != trial_map.endBySparsity() && it != test_map.endBySparsity(); ++it, ++jt){
            uint i = it->gid, j = jt->gid;
            for (int l = 0; l < A.nRow; ++l)
                F(l, 0) -= A(l, j) * bc;
            F(i, 0) = bc;

            std::fill(A.data + j * A.nRow, A.data + (j+1) * A.nRow, 0);
            for (int l = 0; l < A.nRow; ++l) A(i, l) = 0;
            A(i, j) = 1.0;
        }
    }
    template<typename Scalar>
    inline void applyDirByDofs(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, const ArrayView<const Scalar>& dofs){
        if (trial_map == test_map){
            applyConstantDirByDofs(trial_map, A, F, sp, dofs);
            return;
        }
        for (auto jt = trial_map.beginBySparsity(sp, true), it = test_map.beginBySparsity(sp, true); jt != trial_map.endBySparsity() && it != test_map.endBySparsity(); ++it, ++jt){
            uint i = it->gid, j = jt->gid;
            for (int l = 0; l < A.nRow; ++l)
                F(l, 0) -= A(l, j) * dofs[j];
            F(i, 0) = dofs[j];

            std::fill(A.data + j * A.nRow, A.data + (j+1) * A.nRow, 0);
            for (int l = 0; l < A.nRow; ++l) A(i, l) = 0;
            A(i, j) = 1.0;
        }
    }
};