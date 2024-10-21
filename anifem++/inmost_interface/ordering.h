//
// Created by Liogky Alexey on 01.07.2023.
//

#ifndef CARNUM_INMOST_INTERFACE_ORDERING_H
#define CARNUM_INMOST_INTERFACE_ORDERING_H

#include "inmost.h"
#include "anifem++/fem/tetdofmap.h"
#include <array>

namespace Ani{
    inline INMOST::ElementType GeomMaskToInmostElementType(DofT::uchar ani_geom_mask){ 
        static constexpr DofT::uchar lookup[] = {DofT::NODE, DofT::EDGE, DofT::FACE, DofT::CELL};
        INMOST::ElementType res = INMOST::NONE;
        for (unsigned char d = 0; d < 4; ++d) if (ani_geom_mask & lookup[d])
            res |= INMOST::ElementTypeFromDim(d);
        return  res; 
    }
    inline int GeomNumToInmostElementNum(int num) { return DofT::NumToGeomDim(num); }
    inline std::array<uint, 4> DofTNumDofsToInmostNumDofs(const std::array<uint, DofT::NGEOM_TYPES>& num_dofs){
        return {num_dofs[0], num_dofs[1]+num_dofs[2], num_dofs[3]+num_dofs[4], num_dofs[5]};
    }

    ///Reorder nodes of tetrahedron in positive order
    inline void reorderNodesOnTetrahedron(INMOST::ElementArray<INMOST::Node>& nodes);
    /// Get cell and restore tetrahedron connectivity on the cell
    /// @warning this function doesn't allocate memory for arrays nodes, edges, faces
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::ElementArray<INMOST::Node>& nodes, INMOST::ElementArray<INMOST::Edge>& edges, INMOST::ElementArray<INMOST::Face>& faces,
        bool reorder_nodes = true, bool prepare_edges_and_faces = true);
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::Node* nodes, INMOST::Edge* edges, INMOST::Face* faces,
        bool reorder_nodes = true, bool prepare_edges_and_faces = true);
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::HandleType* nodes, INMOST::HandleType* edges, INMOST::HandleType* faces,
        bool reorder_nodes = true, bool prepare_edges_and_faces = true);
    template<typename TEnumerator>    
    inline std::array<unsigned char, 4> createOrderPermutation(const TEnumerator* global_node_index/*[4]*/);
    inline bool isPositivePermutationOrient(unsigned char etype, unsigned char elem_id, const unsigned char* node_permutation/*[4]*/);   
    /// To store inmost-specific position of the d.o.f.
    struct DOFLocus{
        char sign = 1;          ///< = +/-1 
        INMOST::Element elem;   ///< element from where required to take value
        uint elem_data_index;   ///< index in data array
    };
    inline DOFLocus getDOFLocus(const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes/*[4]*/, const std::array<unsigned char, 4>& local_face_index = {0, 1, 2, 3}, const std::array<unsigned char, 6>& local_edge_index = {0, 1, 2, 3, 4, 5}, const std::array<unsigned char, 4>& local_node_index = {0, 1, 2, 3});
    
    /// Compute index in inmost-element local data array of the d.o.f. on element with reordered nodes
    /// @param dmap is local dof map of considered object, only influence if etype == EDGE_ORIENT or etype == FACE_ORIENT
    /// @param leid is index of d.o.f. on canonical element (on element with canonical numbering of nodes of the element)
    /// @param etype is ani geometrical element type
    /// @param lso is symmetric group d.o.f. index
    /// @param elem is inmost element, it's optional argument to be postponed in return value, so it's allowed to use not Valid element
    /// @param canonical_node_indexes is array with 4 (/3/2/1) unique numbers 0,1,2,3 which shows numbers of the nodes in canonical tetrahedron (/triangle/edge/node)
    /// @param local_node_index is actual reordering of node_indexes, s.t. actual_node_index[local_node_index[k]] = canonical_node_indexes[k]
    /// @return index in inmost-element local data array
    inline DOFLocus getDOFLocusOnElement(const DofT::BaseDofMap& dmap, uint leid, DofT::uchar etype, DofT::LocSymOrder lso, const INMOST::Element& elem, const unsigned char* canonical_node_indexes, const std::array<unsigned char, 4>& local_node_index = {0, 1, 2, 3});
    
    ///Take specified degree of freedom on cell
    inline INMOST::Storage::real takeElementDOF(const INMOST::Tag& tag, const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes/*[4]*/, const std::array<unsigned char, 4>& local_face_index = {0, 1, 2, 3}, const std::array<unsigned char, 6>& local_edge_index = {0, 1, 2, 3, 4, 5}, const std::array<unsigned char, 4>& local_node_index = {0, 1, 2, 3});
    inline std::pair<INMOST::Storage::real, bool> takeElementDOFifAvailable(const INMOST::Tag& tag, const DofT::BaseDofMap& dmap, const DofT::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes/*[4]*/, const std::array<unsigned char, 4>& local_face_index = {0, 1, 2, 3}, const std::array<unsigned char, 6>& local_edge_index = {0, 1, 2, 3, 4, 5}, const std::array<unsigned char, 4>& local_node_index = {0, 1, 2, 3});

    ///Gather all odfs of the variable component on cell from tag into container out 
    /// @tparam OnlyIfDataAvailable if set to true then will gather all d.o.f.'s on the cell else will gather values only if specified d.o.f. is stored in the tag
    template<bool OnlyIfDataAvailable = false, class RandomIt>
    void GatherDataOnElement(
        const INMOST::Tag& from, const DofT::BaseDofMap& dmap, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp, 
        const std::array<unsigned char, 4>& local_face_index = {0, 1, 2, 3}, const std::array<unsigned char, 6>& local_edge_index = {0, 1, 2, 3, 4, 5}, const std::array<unsigned char, 4>& local_node_index = {0, 1, 2, 3});
    template<bool OnlyIfDataAvailable = false, class RandomIt>
    void GatherDataOnElement(
        const INMOST::Tag* var_tags, const std::size_t nvar_tags, const DofT::BaseDofMap& dmap, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const unsigned char* canonical_node_indexes, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp, 
        const std::array<unsigned char, 4>& local_face_index = {0, 1, 2, 3}, const std::array<unsigned char, 6>& local_edge_index = {0, 1, 2, 3, 4, 5}, const std::array<unsigned char, 4>& local_node_index = {0, 1, 2, 3}); 

    /// Reorder data in data-tags, that store data relative to old_node_enumerator, according to node enumeration from tag new_node_enumerator.
    /// The function is usefull to reload mesh function from saved mesh when node numbering after mesh loading is changed comparing with the saved one 
    /// @param ts is list of tags with same d.o.f. map template
    /// @param dmap is d.o.f. map
    /// @param old_node_enumerator is node enumerator relative to which data in the tags are ordered
    /// @param new_node_enumerator is node enumerator relative to which data in the tags will be ordered after reordering
    /// @param block_sz is dimension of the d.o.f.
    /// @param elem_data_shift are indexes of first data on tags (useful if tags contain a composition of several mesh functions and we want consider only one of the functions)
    /// @param allow_compressed_data should we consider data array containing only one d.o.f. as array of same d.o.f.'s equals to the first one
    inline void reorder_mesh_function_data(std::vector<INMOST::Tag> ts, const DofT::BaseDofMap& dmap, INMOST::Tag old_node_enumerator, INMOST::Tag new_node_enumerator, std::size_t block_sz = 1, std::array<std::size_t, 4> elem_data_shift = {0, 0, 0, 0}, bool allow_compressed_data = false);

    inline void reorder_fem_vars_data(std::vector<INMOST::Tag> fem_vars, const DofT::BaseDofMap& dmap, INMOST::Tag old_node_enumerator, INMOST::Tag new_node_enumerator){
        reorder_mesh_function_data(std::move(fem_vars), dmap, old_node_enumerator, new_node_enumerator);
    }
    inline void reorder_fem_vars_data(std::vector<INMOST::Tag> fem_vars, const DofT::BaseDofMap& dmap, INMOST::Tag old_node_enumerator){
        reorder_fem_vars_data(std::move(fem_vars), dmap, old_node_enumerator, old_node_enumerator.GetMeshLink()->GlobalIDTag());
    } 
}
#include "ordering.inl"

#endif //CARNUM_INMOST_INTERFACE_ORDERING_H