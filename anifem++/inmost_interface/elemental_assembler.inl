//
// Created by Liogky Alexey on 20.10.2022.
//

template<class RandomIt>
void ElementalAssembler::GatherDataOnElement(
    const INMOST::Tag& from, const ElementalAssembler::VarsHelper& vars, 
    const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
    const int local_face_index[4], const int local_edge_index[6], const int local_node_index[4],
    RandomIt out, const int* component/*[ncomp]*/, int ncomp){
    if (ncomp < 0) return;
    if (ncomp == 0){
        for (int ti = 0; ti < FemExprDescr::NGEOM_TYPES; ++ti) {
            auto t = static_cast<FemExprDescr::GeomType >(ti);
            auto choose = GeomTakerDOF(t);
            int loffset = 0;
            for (int v = 0, cnt = vars.NumBaseVars(); v < cnt; ++v){
                auto& var = vars.descr->base_funcs[v].odf;
                auto ndof = var->NumDofOnTet(t);
                auto moff = vars.BaseOffset(v);
                for (int ldof = 0; ldof < ndof; ++ldof){
                    auto lo = var->LocalOrderOnTet(t, ldof);
                    lo.loc_elem_dof_id += loffset;
                    out[lo.gid + moff] = choose(from, lo, cell, faces, edges, nodes, local_face_index, local_edge_index, local_node_index);
                }
                loffset += ndof;
            }
        }
    } else {
        assert(component[0] >= 0 && component[0] <= static_cast<int>(vars.descr->base_funcs.size()) && "Wrong number of component");
        auto var = vars.descr->base_funcs[component[0]].odf->GetNestedComponent(component+1, ncomp-1);
        auto moff = vars.BaseOffset(component[0]);
        for (int ti = 0; ti < FemExprDescr::NGEOM_TYPES; ++ti) {
            auto t = static_cast<FemExprDescr::GeomType >(ti);
            auto choose = GeomTakerDOF(t);
            auto ndof = var.NumDofOnTet(t);
            int loffset = 0;
            for (int v = 0; v < component[0]; ++v) loffset += vars.descr->base_funcs[v].odf->NumDof(t);
            for (int ldof = 0; ldof < ndof; ++ldof){
                auto lo = var.LocalOrderOnTet(t, ldof);
                lo.loc_elem_dof_id += loffset;
                out[lo.gid + moff] = choose(from, lo, cell, faces, edges, nodes, local_face_index, local_edge_index, local_node_index);
            }
        }
    }
}

template<class RandomIt>
void ElementalAssembler::GatherDataOnElement(
    const std::vector<INMOST::Tag>& from, const ElementalAssembler::VarsHelper& vars, 
    const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
    const int local_face_index[4], const int local_edge_index[6], const int local_node_index[4],
    RandomIt out, const int* component/*[ncomp]*/, int ncomp){
    if (ncomp < 0) return;
    if (ncomp == 0){
        for (int ti = 0; ti < FemExprDescr::NGEOM_TYPES; ++ti) {
            auto t = static_cast<FemExprDescr::GeomType >(ti);
            auto choose = GeomTakerDOF(t);
            for (int v = 0, cnt = vars.NumBaseVars(); v < cnt; ++v){
                auto& var = vars.descr->base_funcs[v].odf;
                auto ndof = var->NumDofOnTet(t);
                auto moff = vars.BaseOffset(v);
                for (int ldof = 0; ldof < ndof; ++ldof){
                    auto lo = var->LocalOrderOnTet(t, ldof);
                    out[lo.gid + moff] = choose(from[v], lo, cell, faces, edges, nodes, local_face_index, local_edge_index, local_node_index);
                }
            }
        }
    } else {
        assert(component[0] >= 0 && component[0] <= static_cast<int>(static_cast<int>(vars.descr->base_funcs.size())) && "Wrong number of component");
        auto var = vars.descr->base_funcs[component[0]].odf->GetNestedComponent(component+1, ncomp-1);
        auto moff = vars.BaseOffset(component[0]);
        for (int ti = 0; ti < FemExprDescr::NGEOM_TYPES; ++ti) {
            auto t = static_cast<FemExprDescr::GeomType >(ti);
            auto choose = GeomTakerDOF(t);
            auto ndof = var.NumDofOnTet(t);
            for (int ldof = 0; ldof < ndof; ++ldof){
                auto lo = var.LocalOrderOnTet(t, ldof);
                out[lo.gid + moff] = choose(from[component[0]], lo, cell, faces, edges, nodes, local_face_index, local_edge_index, local_node_index);
            }
        }
    }
}