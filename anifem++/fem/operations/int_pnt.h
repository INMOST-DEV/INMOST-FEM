//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_PNT_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_PNT_H
#include "core.h"
#include <stdexcept>
#include <functional>
#include <algorithm>
#include <string>

namespace Ani{
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3Dnode(const TetrasT& XYZ,
                   int node_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   void *user_data = nullptr);
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3Dnode(const TetrasT& XYZ,
                   int node_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   void *user_data = nullptr); 
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3DpntL(const TetrasT& XYZ,
                         ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   void *user_data = nullptr); 
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3DpntL(const TetrasT& XYZ,
                         ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   void *user_data = nullptr);  
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetraScalarTp>
    void fem3DpntX(const Tetra<TetraScalarTp>& XYZ,
                   const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   void *user_data = nullptr);  
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetraScalarTp>
    void fem3DpntX(const Tetra<TetraScalarTp>& XYZ,
                   const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   void *user_data = nullptr);                                                                       
    ///The function to compute elemental matrix at node N:
    /// \f$[(D \dot OpA(u)) \dot OpB(v)]_{x = x_N} \f$
    ///Memory-less version of fem3Dnode
    ///@param node_num is node number
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3Dnode(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int node_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   void *user_data = nullptr){
        fem3Dnode<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), node_num, Dfnc, A, user_data);
    }

    ///The function to compute elemental matrix at node N:
    /// \f$[(D \dot OpA(u)) \dot OpB(v)]_{x = x_N} \f$
    ///External memory-dependent version of fem3Dnode
    ///@see fem3Dnode and fem3Dtet for description of parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3Dnode(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int node_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   void *user_data = nullptr){
        fem3Dnode<OpA, OpB, FuncTraits, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), node_num, Dfnc, A, plainMemory, user_data);
    } 

    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dnode( const TetrasT& XYZ, int node_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data = nullptr);   

    ///Give memory requirement for external memory-dependent specialization of fem3Dnode
    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3Dnode_memory_requirements(int fusion = 1);

    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3Dnode_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, 1, fusion, false, false);
    }                              

    ///The function to compute elemental matrix at point inside or on boundary of tetrahedron:
    /// \f$\sum_{L \in XYL} w_L [(D \dot OpA(u)) \dot OpB(v)](x(L)) \f$
    ///Memory-less version of fem3DpntL
    ///@param XYL are barycentric coords of points for expression evaluation
    ///@param WG are weights of the points
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3DpntL(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                         ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   void *user_data = nullptr){
        fem3DpntL<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), XYL, WG, Dfnc, A, user_data);
    }                                                       
    
    ///The function to compute elemental matrix at point inside or on boundary of tetrahedron:
    /// \f$\sum_{L \in XYL} w_L [(D \dot OpA(u)) \dot OpB(v)](x(L)) \f$
    ///External memory-dependent of fem3DpntL
    ///@param XYL are barycentric coords of points for expression evaluation
    ///@param WG are weights of the points
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3DpntL(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                         ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   void *user_data = nullptr){
        fem3DpntL<OpA, OpB, FuncTraits, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), XYL, WG, Dfnc, A, plainMemory, user_data);
    }

    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3DpntL( const TetrasT& XYZ, ArrayView<> XYL, ArrayView<> WG,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data = nullptr ); 

    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3DpntL_memory_requirements(int pnt_per_tetra=1, int fusion=1){
        return fem3D_memory_requirements_base<OpA, OpB, ScalarType, IndexType, true, false>(pnt_per_tetra, fusion);
    }
    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3DpntL_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int pnt_per_tetra = 1, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, pnt_per_tetra, fusion, false, false);
    }

    ///The function to compute elemental matrix at point inside or on boundary of tetrahedron:
    /// \f$\sum_{x \in X} w_x [(D \dot OpA(u)) \dot OpB(v)](x) \f$
    ///Memory-less version of fem3DpntX
    ///@param[in] X are points for function evaluation
    ///@param[in] WG are weights of the points
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3DpntX(const ScalarType* XY0p, const ScalarType* XY1p, const ScalarType* XY2p, const ScalarType* XY3p,
                   const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   void *user_data = nullptr){
        fem3DpntX<OpA, OpB, FuncTraits, ScalarType, IndexType>(Tetra<const ScalarType>(XY0p, XY1p, XY2p, XY3p), X, WG, Dfnc, A, user_data);
    }                                                       
    
    ///The function to compute elemental matrix at point inside or on boundary of tetrahedron:
    /// \f$\sum_{x \in X} w_x [(D \dot OpA(u)) \dot OpB(v)](x) \f$
    ///External memory-dependent of fem3DpntX
    ///@param[in] X are points for function evaluation
    ///@param[in] WG are weights of the points
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3DpntX(const ScalarType* XY0p, const ScalarType* XY1p, const ScalarType* XY2p, const ScalarType* XY3p,
                   const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   void *user_data = nullptr){
        fem3DpntX<OpA, OpB, FuncTraits, ScalarType, IndexType>(Tetra<const ScalarType>(XY0p, XY1p, XY2p, XY3p), X, WG, Dfnc, A, plainMemory, user_data);
    }

    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetraScalarTp>
    void fem3DpntX( const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, ArrayView<> WG,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data = nullptr ); 

    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3DpntX_memory_requirements(int pnt_per_tetra=1, int fusion=1){
        return fem3D_memory_requirements_base<OpA, OpB, ScalarType, IndexType, true, false>(pnt_per_tetra, fusion);
    }
    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3DpntX_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int pnt_per_tetra = 1, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, pnt_per_tetra, fusion, false, true);
    }

};
#include "int_pnt.inl"
#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_PNT_H