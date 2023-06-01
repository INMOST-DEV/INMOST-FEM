//
// Created by Liogky Alexey on 01.06.2023.
//

#ifndef CARNUM_FEM_DYNAMIC_OPERATIONS_H
#define CARNUM_FEM_DYNAMIC_OPERATIONS_H

#include "eval.h"
#include "int_tet.h"
#include "int_face.h"
#include "int_edge.h"
#include "int_pnt.h"

/// Here are wrappers based on the use of dynamic memory for some available operations
namespace Ani{
    ///The function to compute elemental matrix for the volume integral
    ///over tetrahedron T: \f$\int_T [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3Dtet(  const TetrasT& XYZ, const Functor& Dfnc, DenseMatrix<ScalarType>& A, 
                    DynMem<ScalarType, IndexType>& wmem, int order = 5, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dtet(  const TetrasT& XYZ, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, int order = 5, void *user_data = nullptr );

    ///The function to compute elemental matrix for the surface integral
    ///over face F: \f$\int_f [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ, int face_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, int order = 5, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ, int face_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                   DynMem<>& wmem, int order = 5, void *user_data = nullptr);
    ///The function to compute elemental matrix for the surface integral
    ///over face F: \f$\int_f [((D \dot OpA(u)) \dot \mathrm{N}) \dot OpB(v)] dx \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ, int face_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, int order = 5, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ, int face_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                   DynMem<>& wmem, int order = 5, void *user_data = nullptr);

    ///The function to compute elemental matrix for the surface integral
    ///over face E: \f$\int_e [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3Dedge(const TetrasT& XYZ, int edge_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, int order = 5, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dedge( const TetrasT& XYZ, int edge_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, int order = 5, void *user_data = nullptr);

    ///The function to compute elemental matrix at node N:
    /// \f$[(D \dot OpA(u)) \dot OpB(v)]_{x = x_N} \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3Dnode(const TetrasT& XYZ, int node_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dnode( const TetrasT& XYZ, int node_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, void *user_data = nullptr);

    ///The function to compute elemental matrix at point inside or on boundary of tetrahedron:
    /// \f$\sum_{L \in XYL} w_L [(D \dot OpA(u)) \dot OpB(v)](x(L)) \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3DpntL(const TetrasT& XYZ, ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3DpntL( const TetrasT& XYZ, ArrayView<> XYL, ArrayView<> WG, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, void *user_data = nullptr );
    ///The function to compute elemental matrix at point inside or on boundary of tetrahedron:
    /// \f$\sum_{x \in X} w_x [(D \dot OpA(u)) \dot OpB(v)](x) \f$
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetraScalarTp>
    void fem3DpntX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, void *user_data = nullptr);
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetraScalarTp>
    void fem3DpntX( const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, ArrayView<> WG, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, void *user_data = nullptr );

    ///Evaluate \f$ Op(u)[x(by bary coord)] \f$
    template<typename Op, typename ScalarType = double, typename IndexType = int, typename TetrasT>
    void fem3DapplyL( const TetrasT& XYZ, ArrayView<ScalarType> XYL, const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU, DynMem<ScalarType, IndexType>& wmem);
    ///Evaluate \f$ Op(u)[x(by bary coord)] \f$
    template<typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& A, DynMem<>& wmem);
    ///Evaluate \f$ D[x] * Op(u)[x] \f$ where x defined by bary coords
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs, const ApplyOpBase& applyOpU, uint Ddim1, 
                    const Functor& Dfnc, DenseMatrix<>& opU, DynMem<>& wmem, void *user_data = nullptr);
    ///Evaluate \f$ Op(u)[x] \f$
    template<typename Op, typename ScalarType = double, typename IndexType = int, typename TetraScalarTp>
    void fem3DapplyX( const Tetra<TetraScalarTp>& XYZ, const ArrayView<const ScalarType> X, const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU, DynMem<ScalarType, IndexType>& wmem);
    ///Evaluate \f$ Op(u)[x] \f$
    template<typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& opA, DynMem<>& wmem);
    ///Evaluate \f$ D[x] * Op(u)[x] \f$
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, uint Ddim1, const Functor& Dfnc, DenseMatrix<>& opU, DynMem<>& wmem, void *user_data = nullptr);

};

#include "dyn_ops.inl"
#endif //CARNUM_FEM_DYNAMIC_OPERATIONS_H