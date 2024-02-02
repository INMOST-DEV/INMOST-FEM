#include "global_enumerator.h"

using namespace Ani;
using PGE = IGlobEnumeration;

void PGE::CopyByEnumeration(const INMOST::Sparse::Vector& from, INMOST::Tag to) const{
    INMOST::ElementType mask = getInmostVarElementType();
#ifndef NDEBUG
    auto i_nds = getInmostMeshNumDof();
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = beginByGeom(); it != endByGeom(); ++it){
        auto res = *it;
        res.GetNatInd().elem.RealArray(to)[res.GetEnum()->getFullIndex(res.GetNatInd()).GetElemDofId()] = from[res.GetVecInd().id];
    }
    mesh->ExchangeData(to, mask);
}

void PGE::CopyByEnumeration(INMOST::Tag from, INMOST::Sparse::Vector& to) const{
    to.SetInterval(getBegInd(), getEndInd());
    for (auto it = beginByGeom(); it != endByGeom(); ++it){
        auto res = *it;
        to[res.GetVecInd().id] = res.GetNatInd().elem.RealArray(from)[res.GetEnum()->getFullIndex(res.GetNatInd()).GetElemDofId()];
    }
}

void PGE::CopyByEnumeration(const std::vector<INMOST::Tag>& from, INMOST::Sparse::Vector& to) const{
    to.SetInterval(getBegInd(), getEndInd());
    for (auto it = beginByGeom(); it != endByGeom(); ++it){
        auto res = *it;
        auto vid = res.GetVecInd().id; auto nid = res.GetEnum()->getFullIndex(res.GetNatInd());
        to[vid] = nid.elem.RealArray(from[nid.var_id])[nid.GetSingleVarDofId()];
    }
}

void PGE::CopyByEnumeration(const INMOST::Sparse::Vector& from, std::vector<INMOST::Tag> to) const{
#ifndef NDEBUG
    assert(to.size() == static_cast<std::size_t>(getVars().NumVars()) && "Wrong number of tags");
    for (unsigned v = 0; v < to.size(); ++v) {
        auto var = getVars()[v].dofmap;
        INMOST::ElementType mask = GeomMaskToInmostElementType(var.GetGeomMask());
        auto i_nds = DofTNumDofsToInmostNumDofs(var.NumDofs());
        assert(to[v].isDefinedMask(mask) && "Tag is not defined on the geom mask");
        assert(to[v].GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    }
#endif
    for (auto it = beginByGeom(); it != endByGeom(); ++it){
        auto res = *it;
        auto vid = res.GetVecInd().id; auto nid = res.GetEnum()->getFullIndex(res.GetNatInd());
        nid.elem.RealArray(to[nid.var_id])[nid.GetSingleVarDofId()] = from[vid];
    }
    // for (unsigned v = 0; v < to.size(); ++v) {  
    //     INMOST::ElementType mask = AniGeomMaskToInmostElementType(getVars()[v].dofmap.GetGeomMask());
    //     mesh->ExchangeData(to[v], mask);
    // }
    // The code below do potentailly more actions then commennted code above 
    // but actually perform less exchanges and so more quick 
    if (mesh->GetProcessorsNumber() > 1){
        INMOST::ElementType mask = getInmostVarElementType();
        mesh->ExchangeData(to, mask);
    }
}

void PGE::CopyByEnumeration(INMOST::Tag from, INMOST::Tag to) const{
    INMOST::ElementType mask = getInmostVarElementType();
    auto i_nds = getInmostMeshNumDof();
#ifndef NDEBUG
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = mesh->BeginElement(mask); it != mesh->EndElement(); ++it){
        for (unsigned k = 0, k_sz = i_nds[INMOST::ElementNum(it->GetElementType())]; k < k_sz; ++k)
            it->RealArray(to)[k] = it->RealArray(from)[k];
    }
}

void PGE::CopyByEnumeration(INMOST::Tag from, std::vector<INMOST::Tag> to) const{
    INMOST::ElementType mask = getInmostVarElementType();
    auto& vars = getVars();
#ifndef NDEBUG
    assert(to.size() == static_cast<std::size_t>(vars.NumVars()) && "Wrong number of tags");
    for (unsigned v = 0; v < to.size(); ++v) {
        auto var = getVars()[v].dofmap;
        INMOST::ElementType mask = GeomMaskToInmostElementType(var.GetGeomMask());
        auto i_nds = DofTNumDofsToInmostNumDofs(var.NumDofs());
        assert(to[v].isDefinedMask(mask) && "Tag is not defined on the geom mask");
        assert(to[v].GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    }
#endif
    unsigned char etnum_lookup_prt[] = {0, 1, 3, 5, 6};
    for (auto it = mesh->BeginElement(mask); it != mesh->EndElement(); ++it){
        auto dim = INMOST::ElementNum(it->GetElementType());
        for (unsigned char etnum = etnum_lookup_prt[dim]; etnum < etnum_lookup_prt[dim+1]; ++etnum){
            for (int v = 0; v < vars.NumVars(); ++v){
                for (int k = 0,
                     k_shft = vars.m_spaceNumDof[etnum][v],
                     k_sz = vars.m_spaceNumDof[etnum][v+1] - vars.m_spaceNumDof[etnum][v]; 
                        k < k_sz; ++k){
                    unsigned char tshift = 0, tvshift = 0;
                    for (int lt = etnum_lookup_prt[dim]; lt < etnum; ++lt){
                        tshift += vars.m_spaceNumDof[lt][vars.NumVars()];
                        tvshift += vars.m_spaceNumDof[lt][v+1] - vars.m_spaceNumDof[lt][v];
                    }
                    it->RealArray(to[v])[k + tvshift] = it->RealArray(from)[tshift + k_shft + k]; 
                }   

            }
        }
    }
}

void PGE::CopyByEnumeration(const std::vector<INMOST::Tag>& from, INMOST::Tag to) const{
    INMOST::ElementType mask = getInmostVarElementType();
    auto& vars = getVars();
#ifndef NDEBUG
    assert(from.size() == static_cast<std::size_t>(vars.NumVars()) && "Wrong number of tags");
    auto i_nds = getInmostMeshNumDof();
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(getInmostVarElementType()) && "Tag is not defined on the geom mask");
#endif
    unsigned char etnum_lookup_prt[] = {0, 1, 3, 5, 6};
    for (auto it = mesh->BeginElement(mask); it != mesh->EndElement(); ++it){
        auto dim = INMOST::ElementNum(it->GetElementType());
        for (unsigned char etnum = etnum_lookup_prt[dim]; etnum < etnum_lookup_prt[dim+1]; ++etnum){
            for (int v = 0; v < vars.NumVars(); ++v){
                for (int k = 0,
                     k_shft = vars.m_spaceNumDof[etnum][v],
                     k_sz = vars.m_spaceNumDof[etnum][v+1] - vars.m_spaceNumDof[etnum][v]; 
                        k < k_sz; ++k){
                    unsigned char tshift = 0, tvshift = 0;
                    for (int lt = etnum_lookup_prt[dim]; lt < etnum; ++lt){
                        tshift += vars.m_spaceNumDof[lt][vars.NumVars()];
                        tvshift += vars.m_spaceNumDof[lt][v+1] - vars.m_spaceNumDof[lt][v];
                    }
                    it->RealArray(to)[tshift + k_shft + k] = it->RealArray(from[v])[k + tvshift]; 
                }   
            }
        }
    }
}

void PGE::CopyVarByEnumeration(const INMOST::Sparse::Vector& from, INMOST::Tag to, int v) const {
    auto var = getVars()[v].dofmap;
    INMOST::ElementType mask = GeomMaskToInmostElementType(var.GetGeomMask());
#ifndef NDEBUG
    auto i_nds = DofTNumDofsToInmostNumDofs(var.NumDofs());
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = beginByGeom(v); it != endByGeom(); ++it){
        auto res = *it;
        auto vid = res.GetVecInd().id; auto nid = res.GetEnum()->getFullIndex(res.GetNatInd());
        nid.elem.RealArray(to)[nid.GetSingleVarDofId()] = from[vid];
    }
    mesh->ExchangeData(to, mask);
}

void PGE::CopyVarByEnumeration(INMOST::Tag from, INMOST::Tag to, int v) const {
    auto var = getVars()[v].dofmap;
    INMOST::ElementType mask = GeomMaskToInmostElementType(var.GetGeomMask());
#ifndef NDEBUG
    auto i_nds = DofTNumDofsToInmostNumDofs(var.NumDofs());
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    unsigned char etnum_lookup_prt[] = {0, 1, 3, 5, 6};
    for (auto it = mesh->BeginElement(mask); it != mesh->EndElement(); ++it){
        auto dim = INMOST::ElementNum(it->GetElementType());
        for (unsigned char etnum = etnum_lookup_prt[dim]; etnum < etnum_lookup_prt[dim+1]; ++etnum){
            for (int k = 0,
                    k_shft = vars.m_spaceNumDof[etnum][v],
                    k_sz = vars.m_spaceNumDof[etnum][v+1] - vars.m_spaceNumDof[etnum][v]; 
                    k < k_sz; ++k){
                unsigned char tshift = 0, tvshift = 0;
                for (int lt = etnum_lookup_prt[dim]; lt < etnum; ++lt){
                    tshift += vars.m_spaceNumDof[lt][vars.NumVars()];
                    tvshift += vars.m_spaceNumDof[lt][v+1] - vars.m_spaceNumDof[lt][v];
                }
                it->RealArray(to)[k + tvshift] = it->RealArray(from)[tshift + k_shft + k];
            } 
        }
    }
}

void PGE::CopyVarByEnumeration(const INMOST::Tag from, int v, INMOST::Tag to) const {
    auto var = getVars()[v].dofmap;
    INMOST::ElementType v_mask = GeomMaskToInmostElementType(var.GetGeomMask());
#ifndef NDEBUG
    auto i_nds = getInmostMeshNumDof();
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(getInmostVarElementType()) && "Tag is not defined on the geom mask");
#endif
    unsigned char etnum_lookup_prt[] = {0, 1, 3, 5, 6};
    for (auto it = mesh->BeginElement(v_mask); it != mesh->EndElement(); ++it){
        auto dim = INMOST::ElementNum(it->GetElementType());
        for (unsigned char etnum = etnum_lookup_prt[dim]; etnum < etnum_lookup_prt[dim+1]; ++etnum){
            for (int k = 0,
                    k_shft = vars.m_spaceNumDof[etnum][v],
                    k_sz = vars.m_spaceNumDof[etnum][v+1] - vars.m_spaceNumDof[etnum][v]; 
                    k < k_sz; ++k){
                unsigned char tshift = 0, tvshift = 0;
                for (int lt = etnum_lookup_prt[dim]; lt < etnum; ++lt){
                    tshift += vars.m_spaceNumDof[lt][vars.NumVars()];
                    tvshift += vars.m_spaceNumDof[lt][v+1] - vars.m_spaceNumDof[lt][v];
                }
                it->RealArray(to)[tshift + k_shft + k] = it->RealArray(from)[k + tvshift];
            } 
        }
    }
}

void PGE::CopyVarByEnumeration(int v, const INMOST::Tag from, INMOST::Tag to) const {
    auto var = getVars()[v].dofmap;
    INMOST::ElementType mask = GeomMaskToInmostElementType(var.GetGeomMask());
    auto i_nds = DofTNumDofsToInmostNumDofs(var.NumDofs());
#ifndef NDEBUG
    assert(to.GetSize() >= *std::max_element(i_nds.begin(), i_nds.end()) && "Tag has not enough size");
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = mesh->BeginElement(mask); it != mesh->EndElement(); ++it){
        for (int k = 0, k_sz = i_nds[INMOST::ElementNum(it->GetElementType())]; k < k_sz; ++k)
            it->RealArray(to)[k] = it->RealArray(from)[k];
    }
}

PGE::NaturalIndexExt PGE::getFullIndex(INMOST::Element elem, int elem_dof_id) const {
    auto e_num = elem.GetElementNum();
    int elem_num = e_num + ((e_num>1) ? 1 : 0) + ((e_num>2) ? 1 : 0);
    uint tshift = 0;
    uint tvshift = 0;
    if (e_num == 1 || e_num == 2){
        auto lnumdofs = vars.NumDof(DofT::NumToGeomType(elem_num));
        if (static_cast<int>(lnumdofs) < elem_dof_id){
            tshift = lnumdofs;
            ++elem_num; 
        }
    }
    auto eid = DofT::ElemOrder(DofT::NumToGeomType(elem_num), elem_dof_id - tshift);
    auto cid = vars.ComponentID(eid);
    IsoElementIndex iei;
    iei.elem_type = cid.etype;
    iei.var_id = cid.part_id;
    if (tshift > 0)
        tvshift = vars.m_spaces[iei.var_id]->NumDof(DofT::NumToGeomType(elem_num-1));
    iei.dim_id = 0;
    iei.dim_elem_dof_id = cid.cgid;
    uint dof_shift = 0;
    auto vt = vars.m_spaces[iei.var_id]->ActualType();
    if (    vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorType) 
        ||  vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorTemplateType)){
        auto ndim = vars.m_spaces[iei.var_id]->NestedDim();
        auto nd = vars.m_spaces[iei.var_id]->NumDof(iei.elem_type) / ndim;
        iei.dim_id = cid.cgid / nd;
        iei.dim_elem_dof_id = cid.cgid % nd;
        dof_shift = nd * iei.dim_id;
    }
    return {NaturalIndex(elem, iei), tshift, tvshift, vars.m_spaceNumDof[elem_num][iei.var_id], dof_shift};
}

PGE::NaturalIndexExt PGE::getFullIndex(const NaturalIndex& ni) const {
    uint tshift = 0, tvshift = 0;
    if (ni.elem_type == DofT::EDGE_ORIENT || ni.elem_type == DofT::FACE_ORIENT){
        tshift = vars.NumDof(ni.elem_type>>1);
        tvshift = vars.m_spaces[ni.var_id]->NumDof(ni.elem_type>>1);
    }
    auto elem_num = DofT::GeomTypeToNum(ni.elem_type);
    uint dof_shift = 0;
    if (ni.dim_id > 0){
        auto ndim = vars.m_spaces[ni.var_id]->NestedDim();
        auto nd = vars.m_spaces[ni.var_id]->NumDof(ni.elem_type) / ndim;
        dof_shift = nd * ni.dim_id;
    }
    
    return {ni, tshift, tvshift, vars.m_spaceNumDof[elem_num][ni.var_id], dof_shift};
}

PGE::NaturalIndexExt PGE::getFullIndex(INMOST::Element elem, uint var_id, int vledi) const{
    auto e_num = elem.GetElementNum();
    int elem_num = e_num + ((e_num>1) ? 1 : 0) + ((e_num>2) ? 1 : 0);
    uint tshift = 0, tvshift = 0;
    if (e_num == 1 || e_num == 2){
        auto lnumdofs = vars.m_spaces[var_id]->NumDof(DofT::NumToGeomType(elem_num));
        if (static_cast<int>(lnumdofs) < vledi){
            tvshift = lnumdofs;
            vledi -= lnumdofs;
            tshift = vars.NumDof(DofT::NumToGeomType(elem_num));
            ++elem_num; 
        }
    }
    IsoElementIndex iei;
    iei.elem_type = DofT::NumToGeomType(elem_num);
    iei.var_id = var_id;
    iei.dim_id = 0;
    iei.dim_elem_dof_id = vledi;
    uint dof_shift = 0;
    auto vt = vars.m_spaces[iei.var_id]->ActualType();
    if (    vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorType) 
        ||  vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorTemplateType)){
        auto ndim = vars.m_spaces[iei.var_id]->NestedDim();
        auto nd = vars.m_spaces[iei.var_id]->NumDof(iei.elem_type) / ndim;
        iei.dim_id = vledi / nd;
        iei.dim_elem_dof_id = vledi % nd;
        dof_shift = nd * iei.dim_id;
    }
    return {NaturalIndex(elem, iei), tshift, tvshift, vars.m_spaceNumDof[elem_num][iei.var_id], dof_shift};
}

void PGE::iteratorByGeom::find_begin(){
    auto* p = val.enumeration;
    auto& j = val.nid;
    j.var_id = sid.var_id == ANY ? 0 : sid.var_id;
    j.dim_id = sid.dim_id == ANY ? 0 : sid.dim_id;
    j.dim_elem_dof_id = sid.dim_elem_dof_id == ANY ? 0 : sid.dim_elem_dof_id;
    INMOST::Storage::integer lid = sid.loc_elem_id == ANY ? 0 : sid.loc_elem_id;
    for (int dim = 0; dim <= 3; ++dim){
        auto etype = DofT::DimToGeomType(dim);
        if (!(sid.etype & etype)) continue; 
        if (etype & DofT::EDGE_UNORIENT) etype = DofT::EDGE_UNORIENT;
        else if (etype & DofT::FACE_UNORIENT) etype = DofT::FACE_UNORIENT;
        j.elem_type = etype;
        auto et = INMOST::ElementTypeFromDim(dim);
        // INMOST::Storage::integer num_elems = p->mesh->NumberOf(et);
        INMOST::Storage::integer lid_end = ( sid.loc_elem_id == ANY ? p->mesh->LastLocalID(et) : std::min(p->mesh->LastLocalID(et), lid + 1) );
        if (_choose_element(p, sid, j, et, lid, lid_end))
            return;
    }
    j.clear();
}
bool PGE::iteratorByGeom::_choose_var(const IGlobEnumeration* p, NaturalIndex& j, uint st_var, uint last_var){
    for (auto next_var = st_var; next_var < last_var; ++next_var){
        auto var = p->vars.m_spaces[next_var].get();
        auto nd = var->NumDof(j.elem_type);
        if (!(nd > 0)) continue;
        auto vt = var->ActualType();
        unsigned ndim = 1;
        if (vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorType) 
            ||  vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorTemplateType)){
            ndim = var->NestedDim();
            nd /= ndim;
        }
        if (static_cast<unsigned>(j.dim_id) >= ndim || static_cast<unsigned>(j.dim_elem_dof_id) >= nd) continue;
        j.var_id = next_var;
        return true;
    }
    return false;
}
bool PGE::iteratorByGeom::_choose_element(const IGlobEnumeration* p, const SliceIndex& sid, NaturalIndex& j, INMOST::ElementType et, INMOST::Storage::integer st, INMOST::Storage::integer ed){
    for (auto lid = st; lid < ed; lid = p->mesh->NextLocalID(et, lid)) {
        auto elem = p->mesh->ElementByLocalID(et, lid);
        if (!elem.isValid() || elem.GetStatus() == INMOST::Element::Ghost) continue;
        j.elem = elem;
        auto next_var = (sid.var_id == ANY) ? 0 : sid.var_id;
        auto last_var = (sid.var_id == ANY) ? p->vars.m_spaces.size() : sid.var_id+1;
        if (_choose_var(p, j, next_var, last_var))
            return true;
        if ((j.elem_type == DofT::EDGE_UNORIENT && (sid.etype & DofT::EDGE_ORIENT) )
                || (j.elem_type == DofT::FACE_UNORIENT && (sid.etype & DofT::FACE_ORIENT) )){
            j.elem_type = (j.elem_type << 1);
            if (_choose_var(p, j, next_var, last_var))
                return true;
            j.elem_type = (j.elem_type >> 1);     
        }       
        break;
    }
    return false;
}

PGE::iteratorByGeom& PGE::iteratorByGeom::operator ++() {
    val.id.clear(); 
    auto* p = val.enumeration;
    auto& j = val.nid;
    auto* var = p->vars.m_spaces[j.var_id].get();
    auto nd = var->NumDof(j.elem_type);
    auto vt = var->ActualType();
    unsigned ndim = 1;
    if (vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorType) 
        ||  vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorTemplateType)){
        ndim = var->NestedDim();
        nd /= ndim;
    }
    if (sid.dim_elem_dof_id == ANY){
        if (j.dim_elem_dof_id + 1 < static_cast<int>(nd))
            return ++j.dim_elem_dof_id, *this;
        else 
            j.dim_elem_dof_id = 0;
    }
    if (sid.dim_id == ANY){
        if (j.dim_id + 1 < static_cast<int>(ndim))
            return ++j.dim_id, *this;
        else 
            j.dim_id = 0;
    }
    if (sid.var_id == ANY){
        if (_choose_var(p, j, j.var_id+1, p->vars.m_spaces.size()))
            return *this;
    }
    if ((j.elem_type == DofT::EDGE_UNORIENT && (sid.etype & DofT::EDGE_ORIENT) )
        || (j.elem_type == DofT::FACE_UNORIENT && (sid.etype & DofT::FACE_ORIENT) )){
        auto next_var = (sid.var_id == ANY) ? 0 : sid.var_id;
        auto last_var = (sid.var_id == ANY) ? p->vars.m_spaces.size() : sid.var_id+1;
        j.elem_type = (j.elem_type << 1);
        if (_choose_var(p, j, next_var, last_var))
            return *this;
    }
    if ((j.elem_type == DofT::EDGE_ORIENT && (sid.etype & DofT::EDGE_UNORIENT) )
        || (j.elem_type == DofT::FACE_ORIENT && (sid.etype & DofT::FACE_UNORIENT) ))
        j.elem_type = (j.elem_type >> 1);

    INMOST::Storage::integer lid = j.elem.LocalID();
    auto et = j.elem.GetElementType();
    if (sid.loc_elem_id == ANY){
        lid = p->mesh->NextLocalID(j.elem.GetElementType(), lid);
        // INMOST::Storage::integer cnt = p->mesh->NumberOf(et);
        if (_choose_element(p, sid, j, et, lid, p->mesh->LastLocalID(j.elem.GetElementType())))
            return *this;
        lid = 0;
    }
    int gtnum = DofT::GeomTypeToNum(j.elem_type);
    int dim = DofT::GeomTypeDim(j.elem_type);
    if (gtnum == 1 || gtnum == 3) gtnum++;
    if (!(sid.etype >> gtnum) || j.elem_type == DofT::CELL){
        j.clear();
        return *this;
    }
    ++dim;
    for (; dim <= 3; ++dim){
        auto etype = DofT::DimToGeomType(dim);
        if (!(sid.etype & etype)) continue;
        if (etype & DofT::EDGE_UNORIENT) etype = DofT::EDGE_UNORIENT;
        else if (etype & DofT::FACE_UNORIENT) etype = DofT::FACE_UNORIENT;
        j.elem_type = etype;
        auto et = INMOST::ElementTypeFromDim(dim);
        // INMOST::Storage::integer num_elems = p->mesh->NumberOf(et);
        if (sid.loc_elem_id != ANY && (lid >= p->mesh->LastLocalID(et) || !p->mesh->ElementByLocalID(et, lid).isValid()) ) continue;
        INMOST::Storage::integer ed = (sid.loc_elem_id == ANY) ? p->mesh->LastLocalID(et) : lid + 1;
        if (_choose_element(p, sid, j, et, lid, ed))
            return *this; 
    }
    j.clear();
    return *this;
}   

void PGE::Clear() {
    std::fill(NumDof.begin(), NumDof.end(), 0);
    std::fill(NumElem.begin(), NumElem.end(), 0);
    std::fill(BegElemID.begin(), BegElemID.end(), LONG_MAX);
    std::fill(EndElemID.begin(), EndElemID.end(), -1);
    MatrSize = -1;
    BegInd = LONG_MAX;
    EndInd = -1;
    vars.Clear();
    mesh = nullptr;
}

PGE::GlobToLocIDMap PGE::createGlobalToLocalIDMap(INMOST::Mesh* m, INMOST::ElementType et, bool forced_update){
    static const std::array<std::string, 4> postfix{"_n", "_e", "_f", "_c"};
    static const std::string prefix = "IGlobEnumeration::GidToLidMap";
    static const std::string cl_prefix = "IGlobEnumeration::ContToLidMap";
    bool is_new_maps = forced_update;
    std::array<INMOST::Tag, 4> back_maps;
    std::array<INMOST::Tag, 4> cl_back_maps;
    for (auto d = 0; d <= 3; ++d) if (et & INMOST::ElementTypeFromDim(d)){
        auto nm = prefix + postfix[d];
        if (m->HaveTag(nm))
            back_maps[d] = m->GetTag(nm);
        else {   
            back_maps[d] = m->CreateTag(nm, INMOST::DATA_INTEGER, INMOST::MESH, INMOST::NONE, ENUMUNDEF);
            is_new_maps = true;
        }
        nm = cl_prefix + postfix[d];
        if (m->HaveTag(nm))
            cl_back_maps[d] = m->GetTag(nm);
        else {   
            cl_back_maps[d] = m->CreateTag(nm, INMOST::DATA_INTEGER, INMOST::MESH, INMOST::NONE, ENUMUNDEF);
            is_new_maps = true;
        }
    }
    if (!is_new_maps) {
        PGE::GlobToLocIDMap res;
        res.glob_to_loc = back_maps;
        // for (auto d = 0; d <= 3; ++d) 
        //     if (et & INMOST::ElementTypeFromDim(d)) {
        //         res.glob_to_loc[d] = m->IntegerArray(m->GetHandle(), back_maps[d]).data();
        //         res.cont_to_loc[d].data().m_dat_st = m->IntegerArray(m->GetHandle(), cl_back_maps[d]).data();
        //         res.cont_to_loc[d].data().m_dat_end = res.cont_to_loc[d].data().m_dat_st + m->IntegerArray(m->GetHandle(), cl_back_maps[d]).size();
        //     } else {
        //         res.glob_to_loc[d] = nullptr;
        //         res.cont_to_loc[d].data().m_dat_st = res.cont_to_loc[d].data().m_dat_end = nullptr;
        //     }
        return res;
    }

    INMOST::ElementType mask = INMOST::NONE;
    for (auto d = 0; d <= 3; ++d){
        auto et1 = INMOST::ElementTypeFromDim(d);
        if ((et1 & et) && !m->HaveGlobalID(et1))
        mask |= et1;
    } 
    m->AssignGlobalID(mask);
    std::array<long, 4> BegElemID, EndElemID;
    std::fill(EndElemID.begin(), EndElemID.end(), -1);
    std::fill(BegElemID.begin(), BegElemID.end(), LONG_MAX);
    for (auto it = m->BeginElement(et); it != m->EndElement(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
        auto dim = it->GetElementDimension();
        auto gid = it->GlobalID();
        if (gid < BegElemID[dim]) BegElemID[dim] = gid;
        if (gid >= EndElemID[dim]) EndElemID[dim] = gid + 1;
    }
    for (auto d = 0; d <= 3; ++d) if (et & INMOST::ElementTypeFromDim(d) && BegElemID[d] == LONG_MAX){
        BegElemID[d] = EndElemID[d] = m->TotalNumberOf(INMOST::ElementTypeFromDim(d));
    }

    auto mHdl = m->GetHandle();
    for (auto d = 0; d <= 3; ++d) if (et & INMOST::ElementTypeFromDim(d)){
        m->IntegerArray(mHdl, back_maps[d]).resize(EndElemID[d] - BegElemID[d]);
    }
    for (auto it = m->BeginElement(et); it != m->EndElement(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
        auto dim = it->GetElementDimension();
        auto gid = it->GlobalID(), lid = it->LocalID();
        m->IntegerArray(mHdl, back_maps[dim])[gid - BegElemID[dim]] = lid;
    }
    for (auto d = 0; d <= 3; ++d) if (et & INMOST::ElementTypeFromDim(d)){
        auto vals = m->IntegerArray(m->GetHandle(), cl_back_maps[d]);
        vals.resize(0);
        auto st = m->BeginElement(et), ed = m->EndElement();
        if (st == ed) continue;
        auto it = st;
        INMOST::Storage::integer cont_id = 0;
        vals.push_back(cont_id); vals.push_back(it->LocalID());
        auto prev_val = vals.back();
        for (it = std::next(it); it != ed; ++it){
            ++cont_id;
            if (prev_val + 1 == it->LocalID()) {++prev_val; continue;}
            vals.push_back(cont_id);
            vals.push_back(it->LocalID());
            prev_val = it->LocalID();
        }
        auto lcnt = prev_val - vals.back() + 1;
        auto cnt = lcnt + vals[vals.size()-2];
        vals.push_back(cnt);
        vals.push_back(vals[vals.size()-2]+lcnt);
    }


    {
        PGE::GlobToLocIDMap res;
        res.glob_to_loc = back_maps;
        // for (auto d = 0; d <= 3; ++d)  
        //     if (et & INMOST::ElementTypeFromDim(d)) {
        //         res.glob_to_loc[d] = m->IntegerArray(m->GetHandle(), back_maps[d]).data();
        //         res.cont_to_loc[d].data().m_dat_st = m->IntegerArray(m->GetHandle(), cl_back_maps[d]).data();
        //         res.cont_to_loc[d].data().m_dat_end = res.cont_to_loc[d].data().m_dat_st + m->IntegerArray(m->GetHandle(), cl_back_maps[d]).size();
        //     } else {
        //         res.glob_to_loc[d] = nullptr;
        //         res.cont_to_loc[d].data().m_dat_st = res.cont_to_loc[d].data().m_dat_end = nullptr;
        //     }
        return res;
    }
}

void PGE::setupIGlobEnumeration(){
    if (!mesh)
        throw std::runtime_error("Mesh is not specified");

    NumDof = vars.NumDofs();
    INMOST::ElementType et = INMOST::NONE;
    for (int i = 0; i < DofT::NGEOM_TYPES; ++i){
        if (NumDof[i] > 0) {
            auto dim = DofT::NumToGeomDim(i);
            auto elem = INMOST::ElementTypeFromDim(dim);
            if (!mesh->HaveGlobalID(elem))
                mesh->AssignGlobalID(elem);
            et |= elem;
        }
    }

    std::fill(EndElemID.begin(), EndElemID.end(), -1);
    std::fill(BegElemID.begin(), BegElemID.end(), LONG_MAX);
    std::fill(NumElem.begin(), NumElem.end(), 0);
    for (auto it = mesh->BeginElement(et); it != mesh->EndElement(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
        auto dim = it->GetElementDimension();
        auto gid = it->GlobalID();
        if (gid < BegElemID[dim]) BegElemID[dim] = gid;
        if (gid >= EndElemID[dim]) EndElemID[dim] = gid + 1; 
        NumElem[dim]++;
    }
    std::array<long, 4> total_geom_elements = {0, 0, 0, 0};
    for (int d = 0; d < 4; ++d) if (et & INMOST::ElementTypeFromDim(d))
        total_geom_elements[d] = mesh->TotalNumberOf(INMOST::ElementTypeFromDim(d));
    for (auto d = 0; d <= 3; ++d) if (et & INMOST::ElementTypeFromDim(d) && BegElemID[d] == LONG_MAX){
        BegElemID[d] = EndElemID[d] = total_geom_elements[d];
    }
    MatrSize = 0, BegInd = 0, EndInd = 0;
    for (int i = 0; i < DofT::NGEOM_TYPES; ++i) if (NumDof[i] > 0){
        auto d = DofT::NumToGeomDim(i);
        MatrSize += total_geom_elements[d] * NumDof[i];
        BegInd += BegElemID[d] * NumDof[i];
    }
    EndInd = BegInd;
    for (int i = 0; i < DofT::NGEOM_TYPES; ++i) if (NumDof[i] > 0){
        auto d = DofT::NumToGeomDim(i);
        EndInd += NumElem[d] * NumDof[i];
    }
}

INMOST::ElementType PGE::getInmostVarElementType() const { 
    auto nd = getInmostMeshNumDof();
    INMOST::ElementType et = INMOST::NONE;
    for (int d = 0; d < 4; ++d) if (nd[d] > 0)
        et |= INMOST::ElementTypeFromDim(d);
    return et;    
}

bool PGE::areVarsTriviallySymmetric() const{
    for (unsigned etnum = 0; etnum < DofT::NGEOM_TYPES; ++etnum) if (NumDof[etnum] > 0){
        auto smask = vars.SymComponents(DofT::NumToGeomType(etnum));
        if (smask == DofT::DofSymmetries::MASK_UNDEF || (smask >> 1))
            return false;
    }
    return true;
}

OrderedEnumerator& OrderedEnumerator::setArrangment(std::array<unsigned char, 5> order){
    auto lperm = perm;
    std::fill(lperm.begin(), lperm.end(), 255);
    for (unsigned i = 0; i < order.size(); ++i){
        auto j = order[i];
        if (j >= lperm.size())
            throw std::runtime_error("Wrong value in order");
        lperm[j] = i;    
    }
    for (unsigned j = 0; j < perm.size(); ++j)
        if (lperm[j] == 255)
            throw std::runtime_error("\"order\" must contain only unique values");
    perm = std::move(lperm); 
    return *this;       
}

std::pair<std::array<unsigned char, 5>, bool> OrderedEnumerator::getArrangement() const{
    std::array<unsigned char, 5> order;
    bool is_ok = true;
    std::fill(order.begin(), order.end(), 255);
    for (unsigned i = 0; i < perm.size(); ++i) if (perm[i] < order.size())
        order[perm[i]] = i;
    for (unsigned j = 0; j < order.size(); ++j) if (order[j] == 255) 
        is_ok = false;   
    return {order, is_ok};      
}

PGE::iteratorByVector OrderedEnumerator::getLowerOrderIndex(long global_elem_id, IsoElementIndex iei) const{
    auto etnum = DofT::GeomTypeToNum(iei.elem_type);
    auto dim = DofT::NumToGeomDim(etnum);
    std::array<std::size_t, 5> id;
    id[perm[0]] = iei.var_id;
    id[perm[1]] = iei.dim_id;
    id[perm[2]] = etnum;
    id[perm[3]] = global_elem_id >= BegElemID[dim] ? (global_elem_id - BegElemID[dim]) : 0;
    id[perm[4]] = iei.dim_elem_dof_id;
    auto it = loc_emap.lower_bound(id);
    if (it == loc_emap.end())
        return endByVector();
    else{
        long i = it->first;
        long id = (reverse ? (loc_emap.size() - i - 1) : i) + BegInd;
        auto st = beginByVector();
        return st + (id - st->GetVecInd().id); 
    }
}

void SimpleEnumerator::setup(std::string prefix){
    static const std::array<std::string, 4> postfix{"_n_", "_e_", "_f_", "_c_"};

    setupIGlobEnumeration();
    INMOST::ElementType et = getInmostVarElementType();
    auto i_nd = getInmostMeshNumDof();
    InitElemIndex[0] = BegInd;
    for(int i=1;i<4;i++)
        InitElemIndex[i] = InitElemIndex[i-1] + i_nd[i-1]*NumElem[i-1];
    
    back_maps = IGlobEnumeration::createGlobalToLocalIDMap(mesh, et);
    if (!(et & INMOST::NODE) && (!areVarsTriviallySymmetric() || (vars.GetGeomMask() & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT) )) && !mesh->HaveGlobalID(INMOST::NODE))
        mesh->AssignGlobalID(INMOST::NODE);

    for (unsigned i = 0; i < index_tags.size(); ++i) index_tags[i].Clear();
    for (int i = 0; i < 4; ++i) if (i_nd[i] > 0){
        auto loc_et = INMOST::ElementTypeFromDim(i);
        if (!mesh->HaveTag(prefix + postfix[i])){
            index_tags[i].tag = mesh->CreateTag(prefix + postfix[i], INMOST::DATA_INTEGER, loc_et, INMOST::NONE, i_nd[i]);
        } else
            throw std::runtime_error("Tag with name \"" + prefix + postfix[i] + "\" already exists");
        for (auto it = mesh->BeginElement(loc_et); it != mesh->EndElement(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
            for (unsigned k = 0; k < i_nd[i]; ++k){
                auto id = getFullIndex(it->getAsElement(), k);
                it->IntegerArray(index_tags[i].tag)[k] = operator()(id).id;
            }
        }
        mesh->ExchangeData(index_tags[i].tag, loc_et);
    }       
}

void OrderedEnumerator::setup(std::string prefix){
    static const std::array<std::string, 4> postfix{"_n_", "_e_", "_f_", "_c_"};

    setupIGlobEnumeration();
    INMOST::ElementType et = getInmostVarElementType();
    auto i_nd = getInmostMeshNumDof();
    auto nds = std::accumulate(i_nd.begin(), i_nd.end(), 0U);

    back_maps = IGlobEnumeration::createGlobalToLocalIDMap(mesh, et);
    if (!(et & INMOST::NODE) && (!areVarsTriviallySymmetric() || (vars.GetGeomMask() & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT) )) && !mesh->HaveGlobalID(INMOST::NODE))
        mesh->AssignGlobalID(INMOST::NODE);

    std::array<std::size_t, DofT::NGEOM_TYPES> num_elem;
    unsigned char _qid[6] = {0, 1, 1, 2, 2, 3};
    for (int i = 0; i < 6; ++i) num_elem[i] = NumElem[_qid[i]];
    loc_emap.setRangedExtension(arr_range_func<DofT::NGEOM_TYPES>(std::move(num_elem)), perm[3], perm[2] - (perm[3] < perm[2] ? 1 : 0));
    std::vector<std::array<std::size_t, 4>> vals; vals.reserve(nds);
    for (unsigned vi = 0; vi < vars.m_spaces.size(); ++vi){
        auto* var = vars.m_spaces[vi].get();
        auto vt = var->ActualType();
        unsigned ndim = 1;
        if (vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorType) 
            ||  vt == static_cast<uint>(DofT::BaseDofMap::BaseTypes::VectorTemplateType)){
            ndim = var->NestedDim();
        }
        auto nd = var->NumDofs();
        for (unsigned et = 0; et < nd.size(); ++et)
        for (unsigned di = 0; di < nd[et]; ++di){
            std::array<std::size_t, 4> dof_id;
            dof_id[perm[0] - (perm[3] < perm[0] ? 1 : 0)] = vi;
            dof_id[perm[1] - (perm[3] < perm[1] ? 1 : 0)] = di % ndim;
            dof_id[perm[2] - (perm[3] < perm[2] ? 1 : 0)] = et;
            dof_id[perm[4] - (perm[3] < perm[4] ? 1 : 0)] = di / ndim;
            vals.push_back(dof_id);
        }
    }
    loc_emap.setBaseSet(vals);
    loc_emap.setup();

    for (unsigned i = 0; i < index_tags.size(); ++i) index_tags[i].Clear();
    for (int i = 0; i < 4; ++i) if (i_nd[i] > 0){
        auto loc_et = INMOST::ElementTypeFromDim(i);
        if (!mesh->HaveTag(prefix + postfix[i])){
            index_tags[i].tag = mesh->CreateTag(prefix + postfix[i], INMOST::DATA_INTEGER, loc_et, INMOST::NONE, i_nd[i]);
        } else
            throw std::runtime_error("Tag with name \"" + prefix + postfix[i] + "\" already exists");
        for (auto it = mesh->BeginElement(loc_et); it != mesh->EndElement(); ++it) if (it->GetStatus() != INMOST::Element::Ghost){
            for (unsigned k = 0; k < i_nd[i]; ++k){
                auto id = getFullIndex(it->getAsElement(), k);
                it->IntegerArray(index_tags[i].tag)[k] = index_byMap(id).id;
            }
        }
        mesh->ExchangeData(index_tags[i].tag, loc_et);
    }
}

PGE::EnumerateIndex OrderedEnumerator::operator()(const NaturalIndex& physIndex) const{ 
    auto edim = GeomNumToInmostElementNum(DofT::GeomTypeToNum(physIndex.elem_type));
    long id = physIndex.elem.IntegerArray(index_tags[edim].tag)[getFullIndex(physIndex).GetElemDofId()];
    return EnumerateIndex(id);
}
PGE::EnumerateIndex OrderedEnumerator::OrderC(const NaturalIndex& physIndex) const{
    auto edim = GeomNumToInmostElementNum(DofT::GeomTypeToNum(physIndex.elem_type));
    long id = physIndex.elem.IntegerArray(index_tags[edim].tag)[getFullIndex(physIndex).GetElemDofId()];
    return EnumerateIndex(id);
}
PGE::EnumerateIndex OrderedEnumerator::index_byMap(const NaturalIndex& physIndex) const {
    std::array<std::size_t, 5> id;
    id[perm[0]] = physIndex.var_id;
    id[perm[1]] = physIndex.dim_id;
    id[perm[2]] = DofT::GeomTypeToNum(physIndex.elem_type);
    id[perm[3]] = physIndex.elem.GlobalID() - BegElemID[DofT::NumToGeomDim(id[perm[2]])];
    id[perm[4]] = physIndex.dim_elem_dof_id;
    long i = loc_emap[id];
    return EnumerateIndex((reverse ? (loc_emap.size() - i - 1) : i) + BegInd);
}
    
PGE::NaturalIndex OrderedEnumerator::operator()(EnumerateIndex vi) const {
    assert(vi.id > BegInd && "Element is GHOST");
    auto id = loc_emap[reverse ? (loc_emap.size() - 1 - (vi.id - BegInd)) : (vi.id - BegInd)];
    NaturalIndex res;
    res.var_id = id[perm[0]];
    res.dim_id = id[perm[1]];
    res.elem_type = DofT::NumToGeomType(id[perm[2]]);
    int etype_dim = DofT::NumToGeomDim(id[perm[2]]);
    auto et = INMOST::ElementTypeFromDim(etype_dim);
    //std::size_t glob_ind = MinElem[etype_dim] + id[perm[3]];
    res.dim_elem_dof_id = id[perm[4]];
    // auto loc_ind = back_maps.glob_to_loc[etype_dim][id[perm[3]]];
    auto loc_ind = mesh->IntegerArray(mesh->GetHandle(), back_maps.glob_to_loc[etype_dim])[id[perm[3]]];
    res.elem = mesh->ElementByLocalID(et, loc_ind);
    
    return res;
}
void OrderedEnumerator::Clear(){
    for (unsigned i = 0; i < index_tags.size(); ++i) index_tags[i].Clear();
    for (unsigned i = 0; i < perm.size(); ++i) perm[i] = i;
    reverse = false;
    IGlobEnumeration::Clear();
}

std::string PGE::generateUniqueTagNamePrefix(INMOST::Mesh* m) { 
    static unsigned int unique_num = 0; 
    std::string res = "_GenEnum" + std::to_string(unique_num++);
    if (m != nullptr){
        while (m->HaveTag(res))
            res = "_GenEnum" + std::to_string(unique_num++);
    }
    return res; 
}

void SimpleEnumerator::Clear() {
    for (int i = 0; i < 4; ++i) index_tags[i].Clear();
    std::fill(InitElemIndex.begin(), InitElemIndex.end(), -1);
    IGlobEnumeration::Clear();
}
PGE::EnumerateIndex SimpleEnumerator::OrderC(const NaturalIndex& physIndex) const {
    auto edim = GeomNumToInmostElementNum(DofT::GeomTypeToNum(physIndex.elem_type));
    long id = physIndex.elem.IntegerArray(index_tags[edim].tag)[getFullIndex(physIndex).GetElemDofId()];
    return EnumerateIndex(id);
}
PGE::EnumerateIndex SimpleEnumerator::operator()(const NaturalIndex& physIndex) const {
    EnumerateIndex res; res.id = EnumerateIndex::UnValid;
    if (physIndex.elem.GetStatus() != INMOST::Element::Ghost){
        auto edim = DofT::GeomTypeDim(physIndex.elem_type);
        auto iodf = getFullIndex(physIndex).GetElemDofId();
        switch (m_t){
            case ANITYPE: res.id = InitElemIndex[edim] + (physIndex.elem.GlobalID() - BegElemID[edim]) + iodf * NumElem[edim]; break;
            case MINIBLOCKS: res.id = InitElemIndex[edim] + (physIndex.elem.GlobalID() - BegElemID[edim])*getInmostMeshNumDof()[edim] + iodf * NumElem[edim]; break;
        }
    }

    return res;
}
PGE::NaturalIndex SimpleEnumerator::operator()(EnumerateIndex vi) const {
    assert(vi.id >= 0 && "Not valid vector index");
    int edim = 0;
    {
        int i = 1;
        for (i = 1; i < 4 && InitElemIndex[i] <= vi.id; ++i);
        edim = i-1;
    } 
    int iodf = -1;
    long gid = -1; 
    switch (m_t){ 
        case ANITYPE:{
            iodf = (vi.id - InitElemIndex[edim]) / NumElem[edim];
            gid = (vi.id - InitElemIndex[edim]) % NumElem[edim];
            assert(iodf >= 0 && gid >= 0 && iodf < static_cast<long>(getInmostMeshNumDof()[edim]) && "Call with number on ghost element");
            break;
        }
        case MINIBLOCKS:{
            auto nd = getInmostMeshNumDof();
            iodf = (vi.id - InitElemIndex[edim]) % nd[edim];
            gid = (vi.id - InitElemIndex[edim]) / nd[edim];
            assert(iodf >= 0 && gid >= 0 && gid < EndElemID[edim] && "Call with number on ghost element");
            break;
        }
    }
    auto elem = mesh->ElementByLocalID(INMOST::ElementTypeFromDim(edim), mesh->IntegerArray(mesh->GetHandle(), back_maps.glob_to_loc[edim])[gid]); //back_maps.glob_to_loc[edim][gid]
    auto res = getFullIndex(std::move(elem), iodf);
    return static_cast<NaturalIndex>(res);
}

void GlobEnumeration::setup(){
    using SE = SimpleEnumerator;
    using OE = OrderedEnumerator;
    switch (m_t) {
        case ANITYPE: m_invoker = std::make_unique<SE>(SimpleEnumerator::ANITYPE); break;
        case MINIBLOCKS: m_invoker = std::make_unique<SE>(SimpleEnumerator::MINIBLOCKS); break;
        case NATURAL: (m_invoker = std::make_unique<OE>()); reinterpret_cast<OE*>(m_invoker.get())->setArrangment({OE::VAR, OE::DIM, OE::ELEM_TYPE, OE::ELEM_ID, OE::DOF_ID}); break;
        case DIMUNION: (m_invoker = std::make_unique<OE>()); reinterpret_cast<OE*>(m_invoker.get())->setArrangment({OE::VAR, OE::ELEM_TYPE, OE::ELEM_ID, OE::DOF_ID, OE::DIM}); break;
        case BYELEMTYPE: (m_invoker = std::make_unique<OE>()); reinterpret_cast<OE*>(m_invoker.get())->setArrangment({OE::ELEM_TYPE, OE::VAR, OE::DIM, OE::ELEM_ID, OE::DOF_ID}); break;
        case ETDIMBLOCKS: (m_invoker = std::make_unique<OE>()); reinterpret_cast<OE*>(m_invoker.get())->setArrangment({OE::ELEM_TYPE, OE::VAR, OE::ELEM_ID, OE::DOF_ID, OE::DIM}); break;
        case NOSPECIFIED: break;
    default:
        throw  std::runtime_error("Faced unknown ASSEMBLING_TYPE");
    }
    m_invoker->setMesh(mesh);
    m_invoker->setVars(vars);
    m_invoker->setup();
    NumDof = m_invoker->getNumDofs();
    NumElem = m_invoker->getNumElem();
    BegElemID = m_invoker->getBegElemID();
    EndElemID = m_invoker->getEndElemID();
    MatrSize = m_invoker->getMatrixSize();
    BegInd = m_invoker->getBegInd();
    EndInd = m_invoker->getEndInd();
}