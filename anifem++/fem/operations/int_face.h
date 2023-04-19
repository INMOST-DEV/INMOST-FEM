//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_FACE_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_FACE_H
#include "core.h"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace Ani{
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   int order = 5,
                   void *user_data = nullptr);  
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   int order = 5,
                   void *user_data = nullptr);
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   int order = 5,
                   void *user_data = nullptr);  
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   int order = 5,
                   void *user_data = nullptr);                            

    ///The function to compute elemental matrix for the surface integral
    ///over face F: \f$\int_f [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///Memory-less version of fem3Dface
    ///@param face_num is face number, precisely
    /// - face_num=0 denotes face {XY0, XY1, XY2}
    /// - face_num=1 denotes face {XY1, XY2, XY3}
    /// - face_num=2 denotes face {XY2, XY3, XY0}
    /// - face_num=3 denotes face {XY3, XY0, XY1}
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3Dface(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   int order = 5,
                   void *user_data = nullptr){           
        fem3Dface<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), face_num, Dfnc, A, order, user_data);
    }

    ///The function to compute elemental matrix for the surface integral
    ///over face F: \f$\int_f [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///External memory-dependent version of fem3Dface
    ///@see fem3Dface and fem3Dtet for description of parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3Dface(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   int order = 5,
                   void *user_data = nullptr){           
        fem3Dface<OpA, OpB, FuncTraits, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), face_num, Dfnc, A, plainMemory, order, user_data);
    }
    
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ, int face_num,
                   const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                   PlainMemoryX<> mem, int order = 5, void *user_data = nullptr); 

    ///The function to compute elemental matrix for the surface integral
    ///over face F: \f$\int_f [((D \dot OpA(u)) \dot \mathrm{N}) \dot OpB(v)] dx \f$
    ///Memory-less version of fem3DfaceN
    ///@param face_num is face number, precisely
    /// - face_num=0 denotes face {XY0, XY1, XY2}
    /// - face_num=1 denotes face {XY1, XY2, XY3}
    /// - face_num=2 denotes face {XY2, XY3, XY0}
    /// - face_num=3 denotes face {XY3, XY0, XY1}
    ///@see fem3Dtet for description of other parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3DfaceN(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   int order = 5,
                   void *user_data = nullptr){            
        fem3DfaceN<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), face_num, Dfnc, A, order, user_data);
    }
    ///The function to compute elemental matrix for the surface integral
    ///over face F: \f$\int_f [((D \dot OpA(u)) \dot \mathrm{N}) \dot OpB(v)] dx \f$
    ///External memory-dependent version of fem3Dface
    ///@see fem3DfaceN and fem3Dtet for description of parameters
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int,
            typename Functor>
    void fem3DfaceN(const DenseMatrix<const ScalarType>& XY0,
                   const DenseMatrix<const ScalarType>& XY1,
                   const DenseMatrix<const ScalarType>& XY2,
                   const DenseMatrix<const ScalarType>& XY3,
                   int face_num,
                   const Functor& Dfnc,
                   DenseMatrix<ScalarType>& A,
                   PlainMemory<ScalarType, IndexType> plainMemory,
                   int order = 5,
                   void *user_data = nullptr){           
        fem3DfaceN<OpA, OpB, FuncTraits, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), face_num, Dfnc, A, plainMemory, order, user_data);
    }

    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ, int face_num,
                   const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                   PlainMemoryX<> mem, int order = 5, void *user_data = nullptr); 

    ///Give memory requirement for external memory-dependent specialization of fem3Dface
    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3Dface_memory_requirements(int order, int fusion = 1);

    ///Give memory requirement for external memory-dependent specialization of fem3DfaceN
    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3DfaceN_memory_requirements(int order, int fusion = 1);

    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3Dface_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int order, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, triangle_quadrature_formulas(order).GetNumPoints(), fusion, false, true);
    }
    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3DfaceN_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int order, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, triangle_quadrature_formulas(order).GetNumPoints(), fusion, true, true);
    } 
};

#include "int_face.inl"
#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_FACE_H