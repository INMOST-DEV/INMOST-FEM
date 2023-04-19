//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_EDGE_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_EDGE_H
#include "core.h"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>

namespace Ani{
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3Dedge(const TetrasT& XYZ,
                   int edge_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   int order = 5,
                   void *user_data = nullptr);
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3Dedge(const TetrasT& XYZ,
                   int edge_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   int order = 5,
                   void *user_data = nullptr);               

    ///The function to compute elemental matrix for the line integral
    ///over edge E: \f$\int_e [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///Memory-less version of fem3Dedge
    ///@param edge_num is edge number, precisely,
    /// [0, {XY0, XY1}], [1, {XY0, XY2}], [2, {XY0, XY3}], [3, {XY1, XY2}], [4, {XY1, XY3}], [5, {XY2, XY3}]
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3Dedge(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int edge_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   int order = 5,
                   void *user_data = nullptr){           
        fem3Dedge<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), edge_num, Dfnc, A, order, user_data);
    }

    ///The function to compute elemental matrix for the surface integral
    ///over face E: \f$\int_e [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///External memory-dependent version of fem3Dface
    ///@see fem3Dedge and fem3Dtet for description of parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3Dedge(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int edge_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   int order = 5,
                   void *user_data = nullptr){            
        fem3Dedge<OpA, OpB, FuncTraits, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), edge_num, Dfnc, A, plainMemory, order, user_data);
    }

    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dedge( const TetrasT& XYZ, int edge_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order = 5, void *user_data = nullptr);

    ///Give memory requirement for external memory-dependent specialization of fem3Dedge
    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3Dedge_memory_requirements(int order, int fusion = 1);

    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3Dedge_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int order, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, segment_quadrature_formulas(order).GetNumPoints(), fusion, false, true);
    } 
};

#include "int_edge.inl"

#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_EDGE_H