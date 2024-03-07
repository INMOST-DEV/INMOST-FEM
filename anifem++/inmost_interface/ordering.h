//
// Created by Liogky Alexey on 01.07.2023.
//

#ifndef CARNUM_INMOST_INTERFACE_ORDERING_H
#define CARNUM_INMOST_INTERFACE_ORDERING_H

#include "inmost.h"
#include "anifem++/fem/tetdofmap.h"
#include <array>

namespace Ani{
    ///Reorder nodes of tetrahedron in positive order
    inline void reorderNodesOnTetrahedron(INMOST::ElementArray<INMOST::Node>& nodes);
    /// Get cell and restore tetrahedron connectivity on the cell
    /// @warning this function doesn't allocate memory for arrays nodes, edges, faces
    inline void collectConnectivityInfo(
        const INMOST::Cell& cell, INMOST::ElementArray<INMOST::Node>& nodes, INMOST::ElementArray<INMOST::Edge>& edges, INMOST::ElementArray<INMOST::Face>& faces,
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
}
#include "ordering.inl"

#endif //CARNUM_INMOST_INTERFACE_ORDERING_H