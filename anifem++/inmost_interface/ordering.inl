//
// Created by Liogky Alexey on 01.07.2023.
//

#include "ordering.h"

namespace Ani{
    template<typename NodeType, typename ConvertType>
    void reorderNodesOnTetrahedron(INMOST::Mesh* mlink, NodeType* nodes, ConvertType conv) {
        assert(mlink != nullptr);
        assert(nodes != nullptr);
        double m[3][3];
        auto    crd0 = conv(mlink, nodes[0]).Coords(), crd1 = conv(mlink, nodes[1]).Coords(), 
                crd2 =  conv(mlink, nodes[2]).Coords(), crd3 =  conv(mlink, nodes[3]).Coords();
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
            std::swap(nodes[2], nodes[3]);
    }
    inline void reorderNodesOnTetrahedron(INMOST::Mesh* mlink, INMOST::Node* nodes){ 
        reorderNodesOnTetrahedron(mlink, nodes, [](INMOST::Mesh* mlink, INMOST::Node& n)->INMOST::Node&{ (void) mlink; return n; });
    }
    inline void reorderNodesOnTetrahedron(INMOST::Mesh* mlink, INMOST::HandleType* nodes){ 
        reorderNodesOnTetrahedron(mlink, nodes, [](INMOST::Mesh* mlink, INMOST::HandleType h)->INMOST::Node{ return INMOST::Node(mlink, h); });
    }
    inline void reorderNodesOnTetrahedron(INMOST::ElementArray<INMOST::Node> &nodes) {
        assert(nodes.size() >= 4);
        reorderNodesOnTetrahedron(nodes.GetMeshLink(), nodes.data());
    }
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::HandleType* nodes, INMOST::HandleType* edges, INMOST::HandleType* faces,
        bool reorder_nodes, bool prepare_edges_and_faces){
        assert(nodes != nullptr);
        assert(!prepare_edges_and_faces || (edges != nullptr && faces != nullptr));

        using namespace INMOST;
        Mesh * m = cell.GetMeshLink();
        auto h = cell.GetHandle();
        auto const& hc = m->HighConn(h);//< nodes of the cell
        auto const& lc = m->LowConn(h); //< faces of the cell

        std::copy(hc.data(), hc.data() + 4, nodes);
        if (reorder_nodes)
            reorderNodesOnTetrahedron(m, nodes);
        
        if (prepare_edges_and_faces){
            //load remote data to low level cache
            std::array<HandleType, 4> nds_h{nodes[0], nodes[1], nodes[2], nodes[3]};
            std::array<HandleType, 4> fh{lc[0], lc[1], lc[2], lc[3]};
            auto const& flc0 = m->LowConn(fh[0]), &flc1 = m->LowConn(fh[1]), &flc2 = m->LowConn(fh[2]);
            std::array<std::array<HandleType, 3>, 3> eh{
                std::array<HandleType, 3>{flc0[0], flc0[1], flc0[2]}, {flc1[0], flc1[1], flc1[2]}, {flc2[0], flc2[1], flc2[2]}
            };
            auto const  &nlc0 = m->LowConn(eh[0][0]), &nlc1 = m->LowConn(eh[0][1]), 
                        &nlc2 = m->LowConn(eh[1][0]), &nlc3 = m->LowConn(eh[1][1]), 
                        &nlc4 = m->LowConn(eh[2][0]), &nlc5 = m->LowConn(eh[2][1]); 
            std::array<std::array<HandleType, 4>, 3> nh{
                std::array<HandleType, 4>{nlc0[0], nlc0[1], nlc1[0], nlc1[1]}, 
                std::array<HandleType, 4>{nlc2[0], nlc2[1], nlc3[0], nlc3[1]},
                std::array<HandleType, 4>{nlc4[0], nlc4[1], nlc5[0], nlc5[1]}
            };

            //restore tetrahedron connectivity
            unsigned char fsum = 0;
            for (unsigned char fi = 0; fi < 3; ++fi){
                auto* flc = eh[fi].data(); //< edges of the face
                unsigned char sum = 0;
                {
                    auto* elc0 = nh[fi].data() + 0; //< nodes of the first edge
                    char k = -1, l = -1;
                    for (int ni = 0; ni < 4; ++ni){
                        if (elc0[0] == nds_h[ni]) k = ni;
                        if (elc0[1] == nds_h[ni]) l = ni;
                    }
                    assert(k >= 0 && l >= 0 && "Can't find nodes of the edge");
                    unsigned char id1[2] {static_cast<unsigned char>(k), static_cast<unsigned char>(l)};
                    if (k > l) std::swap(k, l);
                    edges[l - 1 + (k > 0 ? k + 1 : 0)] = flc[0];
                    sum += (k + l);
                    
                    auto* elc1 = nh[fi].data() + 2; //< nodes of the second edge
                    char p = -1, q = -1;
                    unsigned char comp = (elc1[0] == elc0[0] ? 1 : 0) + (elc1[0] == elc0[1] ? 2 : 0);
                    if (comp){
                        k = id1[comp-1];
                        for (int ni = 0; ni < 4; ++ni)
                            if (elc1[1] == nds_h[ni]) l = ni;
                        p = id1[comp%2], q = l;    
                    } else {
                        comp = (elc1[1] == elc0[0] ? 1 : 0) + (elc1[1] == elc0[1] ? 2 : 0);
                        l = id1[comp-1];
                        for (int ni = 0; ni < 4; ++ni)
                            if (elc1[0] == nds_h[ni]) k = ni;
                        p = id1[comp%2], q = k;    
                    }
                    if (k > l) std::swap(k, l);
                    edges[l - 1 + (k > 0 ? k + 1 : 0)] = flc[1];
                    sum += (k + l);

                    k = p, l = q;
                    if (k > l) std::swap(k, l);
                    edges[l - 1 + (k > 0 ? k + 1 : 0)] = flc[2];
                    sum += (k + l);
                }
                unsigned char f_id = (7 - sum/2)%4;
                faces[f_id] = fh[fi];
                fsum += f_id;
            }
            faces[6 - fsum] = fh[3];
        } 
    }
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::Node* nodes, INMOST::Edge* edges, INMOST::Face* faces, bool reorder_nodes, bool prepare_edges_and_faces){
        INMOST::HandleType hnodes[4], hedges[6], hfaces[4]; 
        collectConnectivityInfo(cell, hnodes, hedges, hfaces, reorder_nodes, prepare_edges_and_faces);
        INMOST::Mesh * m = cell.GetMeshLink();
        std::transform(hnodes, hnodes + 4, nodes, [m](INMOST::HandleType h)->INMOST::Node{ return INMOST::Node(m, h); });
        if (prepare_edges_and_faces){
            std::transform(hedges, hedges + 4, edges, [m](INMOST::HandleType h)->INMOST::Edge{ return INMOST::Edge(m, h); });
            std::transform(hfaces, hfaces + 4, faces, [m](INMOST::HandleType h)->INMOST::Face{ return INMOST::Face(m, h); });
        }
    }
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::ElementArray<INMOST::Node>& nodes, INMOST::ElementArray<INMOST::Edge>& edges, INMOST::ElementArray<INMOST::Face>& faces, bool reorder_nodes, bool prepare_edges_and_faces){
        assert(nodes.size() >= 4 && "nodes ElementArray does't have enough of memory");
        assert((!prepare_edges_and_faces || edges.size() >= 6) && "edges ElementArray does't have enough of memory");
        assert((!prepare_edges_and_faces || faces.size() >= 4) && "faces ElementArray does't have enough of memory");
        collectConnectivityInfo(cell, nodes.data(), edges.data(), faces.data(), reorder_nodes, prepare_edges_and_faces);
    }
    template<typename TEnumerator>
    inline std::array<unsigned char, 4> createOrderPermutation(const TEnumerator* gni){
        std::array<unsigned char, 4> node_permutation;
        unsigned char i = 0, j = 0;
        for (int k = 1; k < 4; ++k)
            if (gni[k] < gni[i]) i = k;
            else if (gni[k] > gni[j]) j = k;
        node_permutation[i] = 0; node_permutation[j] = 3;
        unsigned char qid = i+j-1 + (std::min(i,j) > 0 ? 1 : 0);
        static const unsigned char lookup_ind[6*2]{2,3, 1,3, 1,2, 0,3, 0,2, 0,1};
        i = lookup_ind[2*qid], j = lookup_ind[2*qid+1];
        if (gni[i] < gni[j]) node_permutation[i] = 1, node_permutation[j] = 2;
        else node_permutation[i] = 2, node_permutation[j] = 1;
        return node_permutation;
    }
    inline bool isPositivePermutationOrient(unsigned char etype, unsigned char elid, const unsigned char* node_permutation/*[4]*/){
        //assert(etype | DofT::CELL && "This function suppose cells are positive oriented and doesn't check them");
        if (etype & DofT::EDGE){
            const static std::array<unsigned char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
            return (node_permutation[lookup_nds[2*elid]] < node_permutation[lookup_nds[2*elid+1]]);
        } else if (etype & DofT::FACE){
            char change = 0;
            for (int k = 0; k < 3; ++k)
                if (node_permutation[(elid+k) % 4] > node_permutation[(elid+k+1) % 4]) ++change;
            return change == 1;    
        } else if (etype & DofT::NODE)
            return true;
        else if (etype & DofT::CELL){
            char change = 0;
            for (int k = 0; k < 4; ++k)
                if (node_permutation[k] > node_permutation[(k+1) % 4]) ++change;
            return change == 1; 
        }  
        return false;
    }
    inline DOFLocus getDOFLocus(const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index){
        using namespace INMOST;
        HandleType ch = cell.GetHandle(); 
        unsigned char local_cell_index[1]{0};
        const HandleType* h[4]{nodes.data(), edges.data(), faces.data(), &ch};
        const unsigned char* local_index[4]{local_node_index.data(), local_edge_index.data(), local_face_index.data(), local_cell_index};
        uint shift = ( lo.etype & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT) ) ? dmap.NumDof(lo.etype >> 1) : 0;
        auto dim = DofT::GeomTypeDim(lo.etype);
        uint lsid_shift = lo.leid - lo.lsid;

        std::array<unsigned char, 4> gni;
        for (int k = 0; k < 4; ++k)
            gni[local_node_index[k]] = canonical_node_indexes[k];
        char sign = 1;
        int elid = local_index[dim][lo.nelem];
        if (lo.etype & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT))
            sign = isPositivePermutationOrient(lo.etype, elid, gni.data()) ? 1 : -1;
        unsigned char nds_arr[4]{gni[0], gni[1], gni[2], gni[3]};    
        auto reordered_lsid = DofT::DofSymmetries::index_on_reorderd_elem(lo.etype, lo.nelem, lo.stype, lo.lsid, nds_arr);
        return DOFLocus{sign, Element(cell.GetMeshLink(), h[dim][elid]), shift + lsid_shift + reordered_lsid};
    }
    inline DOFLocus getDOFLocusOnElement(const DofT::BaseDofMap& dmap, uint leid, DofT::uchar etype, DofT::LocSymOrder lso, const INMOST::Element& elem, 
        const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_node_index){
        using namespace INMOST;
        uint shift = ( etype & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT) ) ? dmap.NumDof(etype >> 1) : 0;
        auto dim = DofT::GeomTypeDim(etype);
        uint lsid_shift = leid - lso.lsid;

        std::array<unsigned char, 4> gni;
        for (int k = 0; k < 4; ++k)
            gni[local_node_index[k]] = canonical_node_indexes[k];
        auto max_id = *std::max_element(gni.data(), gni.data() + dim + 1);
        for (int k = dim + 1, st = 1; k < 4; ++k, ++st)
            gni[k] = max_id + st;
        char sign = 1;
        int elid = 0;   
        if (etype & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT))
            sign = isPositivePermutationOrient(etype, elid, gni.data()) ? 1 : -1; 
        unsigned char nds_arr[4]{gni[0], gni[1], gni[2], gni[3]};
        auto reordered_lsid = DofT::DofSymmetries::index_on_reorderd_elem(etype, 0, lso.stype, lso.lsid, nds_arr);
        return DOFLocus{sign, elem, shift + lsid_shift + reordered_lsid};
    }
    inline INMOST::Storage::real takeElementDOF(const INMOST::Tag& tag, const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index){
        auto l = getDOFLocus(dmap, lo, cell, faces, edges, nodes, canonical_node_indexes, local_face_index, local_edge_index, local_node_index);
        return l.sign * l.elem.RealArray(tag)[l.elem_data_index];
    }
    inline std::pair<INMOST::Storage::real, bool> takeElementDOFifAvailable(const INMOST::Tag& tag, const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index){
        auto l = getDOFLocus(dmap, lo, cell, faces, edges, nodes, canonical_node_indexes, local_face_index, local_edge_index, local_node_index);
        INMOST::Storage::real v{};
        bool avail_data = false;
        if (l.elem.HaveData(tag)){
            auto arr = l.elem.RealArray(tag);
            if (arr.size() > l.elem_data_index){
                v = l.sign * arr[l.elem_data_index];
                avail_data = true;
            }
        }
        return std::pair<INMOST::Storage::real, bool>{v, avail_data};
    }
    namespace internals{
        template<bool OnlyIfDataAvailable = false>
        struct GDOTTake{
            static inline INMOST::Storage::real ElementDOF(const INMOST::Tag& tag, const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
                                const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
                                const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index){
                return takeElementDOF(tag, dmap, lo, cell, faces, edges, nodes, canonical_node_indexes, local_face_index, local_edge_index, local_node_index);
            }
        };
        template<>
        struct GDOTTake<true>{
            static inline std::pair<INMOST::Storage::real, bool> ElementDOF(const INMOST::Tag& tag, const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
                                const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
                                const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index){
                return takeElementDOFifAvailable(tag, dmap, lo, cell, faces, edges, nodes, canonical_node_indexes, local_face_index, local_edge_index, local_node_index);
            }
        };
        template<bool OnlyIfDataAvailable, class RandomIt>
        struct GDOTAssign{
            static inline void assign(RandomIt out, uint at, INMOST::Storage::real v){ out[at] = v; }
        };
        template<class RandomIt>
        struct GDOTAssign<true, RandomIt>{
            static inline void assign(RandomIt out, uint at, std::pair<INMOST::Storage::real, bool> v){ if (v.second) out[at] = v.first; }
        };
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void GatherDataOnElement(
        const INMOST::Tag& from, const DofT::BaseDofMap& dmap, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp, 
        const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index)
    {
        const DofT::BaseDofMap* ldmap = &dmap;
        DofT::NestedDofMapView ndmap;
        unsigned int shiftOnTet = 0;
        if (ncomp != 0){
            ndmap = ldmap->GetNestedDofMapView(component, ncomp);
            ldmap = &ndmap;
            shiftOnTet = ndmap.m_shiftOnTet;
        }
        for (char dim = 0; dim <= 3; ++dim){
            DofT::TetGeomSparsity sp; 
            sp.set(dim);
            for (auto it = ldmap->beginBySparsity(sp, true); it != ldmap->endBySparsity(); ++it){
                auto lo = *it;
                auto _v = internals::GDOTTake<OnlyIfDataAvailable>::ElementDOF(from, dmap, lo, cell, faces, edges, nodes, 
                            canonical_node_indexes, local_face_index, local_edge_index, local_node_index);
                internals::GDOTAssign<OnlyIfDataAvailable, RandomIt>::assign(out, lo.gid - shiftOnTet, _v);
            }
        }
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void GatherDataOnElement(
        const INMOST::Tag* var_tags, const std::size_t nvar_tags, const DofT::BaseDofMap& dmap, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp, 
        const std::array<unsigned char, 4>& local_face_index, const std::array<unsigned char, 6>& local_edge_index, const std::array<unsigned char, 4>& local_node_index)
        {
            unsigned int ndim = (ncomp == 0) ? int(dmap.NestedDim()) : (component[0]+1);
            if (nvar_tags < ndim)
                throw std::runtime_error("Number of tags in the vector should coincide with number of variables in dof map");
            unsigned int v_st = (ncomp == 0) ? 0 : component[0];
            const int* next_component = (ncomp == 0) ? static_cast<int*>(nullptr) : component+1;
            unsigned int next_ncomp = (ncomp == 0) ? 0 : ncomp-1;
            for (unsigned int v = v_st; v < ndim; ++v){
                int ext_dim = v;
                auto view = dmap.GetNestedDofMapView(&ext_dim, 1);
                GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags[v], *view.m_base, cell, faces, edges, nodes, canonical_node_indexes, 
                                        out + view.m_shiftOnTet, next_component, next_ncomp, local_face_index, local_edge_index, local_node_index);
            }
        }
    
    inline void reorder_mesh_function_data(std::vector<INMOST::Tag> ts, const DofT::BaseDofMap& dmap, INMOST::Tag old_node_enumerator, INMOST::Tag new_node_enumerator, std::size_t block_sz, std::array<std::size_t, 4> elem_data_shift, bool allow_compressed_data){
        using namespace INMOST;
        if (ts.empty()) return;
        Mesh* m = ts[0].GetMeshLink();
        uint nontrivial_symmetry_geom_mask = DofT::UNDEF;
        for (uint geom_num = 0; geom_num < DofT::NGEOM_TYPES; ++geom_num)
            if (dmap.SymComponents(DofT::NumToGeomType(geom_num)) & (~uint(1<<0)) )
                nontrivial_symmetry_geom_mask |= DofT::NumToGeomType(geom_num);
        if (nontrivial_symmetry_geom_mask == DofT::UNDEF) return; 
        auto gmask = GeomMaskToInmostElementType(nontrivial_symmetry_geom_mask);
        auto num_dofs = dmap.NumDofs();
        auto inum_dofs = DofTNumDofsToInmostNumDofs(num_dofs);
        auto max_dofs_per_elem = (*std::max_element(inum_dofs.begin(), inum_dofs.end()));
        std::vector<double> r(block_sz*max_dofs_per_elem);
        std::vector<int> reord(max_dofs_per_elem);
        for (auto e = m->BeginElement(gmask); e != m->EndElement(); ++e){
            auto nds = e->getNodes();
            auto mlng = std::numeric_limits<long>::max();
            std::array<long, 4> new_gni{mlng, mlng, mlng, mlng};
            for (std::size_t i = 0; i < nds.size(); ++i)
                new_gni[i] = nds[i].Integer(new_node_enumerator);
            std::array<unsigned char, 4> new_node_ixs{0, 1, 2, 3};
            new_node_ixs = Ani::createOrderPermutation(new_gni.data());
            std::array<long, 4> old_gni{mlng, mlng, mlng, mlng};
            for (std::size_t i = 0; i < nds.size(); ++i)
                old_gni[new_node_ixs[i]] = nds[i].Integer(old_node_enumerator);
            auto canonical_node_indexes = Ani::createOrderPermutation(old_gni.data());
            DofT::TetGeomSparsity sp; 
            int ielem_type_num = ElementNum(e->GetElementType());
            sp.set(ielem_type_num, 0);
            for (auto it = dmap.beginBySparsity(sp, true); it != dmap.endBySparsity(); ++it){
                auto lo = *it;
                uint shift = ( lo.etype & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT) ) ? dmap.NumDof(lo.etype >> 1) : 0;
                auto pos = getDOFLocusOnElement(dmap, lo.leid, lo.etype, lo.getLocSymOrder(), e->getAsElement(), canonical_node_indexes.data());
                reord[shift + lo.leid] = pos.sign * (pos.elem_data_index + 1);
            }
            for (Tag t: ts) if (e->HaveData(t)) {
                INMOST::Storage::real_array tv = e->RealArray(t);
                if (allow_compressed_data && tv.size() < elem_data_shift[ielem_type_num] + 2*block_sz) continue; //< consider small arrays as compressed repeated data
                assert(tv.size() >= elem_data_shift[ielem_type_num] + block_sz * inum_dofs[ielem_type_num] && "Wrong tag array size");
                auto st_data = tv.data() + elem_data_shift[ielem_type_num];
                for (std::size_t i = 0; i < inum_dofs[ielem_type_num]; ++i){
                    std::copy(st_data + block_sz*(abs(reord[i])-1), st_data + block_sz*abs(reord[i]), r.data() + block_sz*i);
                    if (reord[i] < 0) std::for_each(r.data() + block_sz*i, r.data() + block_sz*(i+1), std::negate<double>());
                }
                std::copy(r.data(), r.data() + block_sz*inum_dofs[ielem_type_num], st_data);
            }
        }
    }
}

