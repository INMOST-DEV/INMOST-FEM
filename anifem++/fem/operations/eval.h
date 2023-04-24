//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_EVAL_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_EVAL_H
#include "core.h"
#include <stdexcept>
#include <string>
#include <algorithm>

namespace Ani{
    template<typename Op, int FUSION = 1, int MAXPNTNUM = 24, typename ScalarType = double, typename IndexType = int, typename TetrasT>
    void fem3DapplyL(
                  const TetrasT& XYZ,
                        ArrayView<ScalarType> XYL,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU);
    template<typename Op, typename ScalarType = double, typename IndexType = int, typename TetrasT>
    void fem3DapplyL(
                  const TetrasT& XYZ,
                        ArrayView<ScalarType> XYL,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU,
                  PlainMemory<ScalarType, IndexType> plainMemory); 
    template<typename Op, int MAXPNTNUM = 24, typename ScalarType = double, typename IndexType = int, typename TetraScalarTp>
    void fem3DapplyX(
                  const Tetra<TetraScalarTp>& XYZ,
                  const ArrayView<const ScalarType> X,
                  const ArrayView<ScalarType>& dofs, 
                  ArrayView<ScalarType> opU); 
    template<typename Op, typename ScalarType = double, typename IndexType = int, typename TetraScalarTp>
    void fem3DapplyX(
                  const Tetra<TetraScalarTp>& XYZ,
                  const ArrayView<const ScalarType> X,
                  const ArrayView<ScalarType>& dofs, 
                  ArrayView<ScalarType> opU,
                  PlainMemory<ScalarType, IndexType> plainMemory);                                         

    /**
     * @brief Evaluate \f$ Op(u)[x] \f$
     * Memory-less version of fem3Dapply
     * @param[in] dofs is matrix [dofs_1, ..., dofs_f] where dofs_i is vector degrees of freedom associated with i-th tetrahedron
     *          e.g. for Op = Operator<GRAD, FemFix<FEM_P1>> dofs_i is d.o.f's vector of FEM_P1 element type 
     * @param[in,out] opU: opU = [opU_1, ..., opU_f] where opU_r = [vec(opU[p_{1r}^T])^T, ..., vec(opU[p_{qr}^T])^T]^T and 
     *          vec(opU[p_{ir}^T]) is vectrized matrix computed at point p_{ir} lied on r-th tetra 
     * @param[in] XYL is barycentric coords of points for function evaluation (supposed same for all fused tetra)
     * @param[in] XY0, XY1, XY2, XY3 are 3xf matrices where r-th column contains coordinates of r-th tetrahedron
     * @tparam FUSION is maximal available number of columns in XYi arrays
     * @tparam MAXPNTNUM is maximal number of points to be evaluated at the same time
     * @tparam Op is fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
     * @return result of evaluating Op(u) at XYG point set
     * @see AniMemory
     */
    //points for evaluationg XYG_r = (p_{1r}^T, ..., p_{qr}^T)^T,  XYG = [XYG_1, ..., XYG_f]
    template<typename Op, int FUSION = 1, int MAXPNTNUM = 24, typename ScalarType = double, typename IndexType = int>
    void fem3DapplyL(
                  const DenseMatrix<const ScalarType>& XY0, const DenseMatrix<const ScalarType>& XY1,
                  const DenseMatrix<const ScalarType>& XY2, const DenseMatrix<const ScalarType>& XY3,
                        ArrayView<ScalarType> XYL,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU){
        fem3DapplyL<Op, FUSION, MAXPNTNUM, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), XYL, dofs, opU);
    }
    ///@brief Evaluate \f$ Op(u)[x] \f$
    ///External memory-dependent version of fem3Dapply
    ///@param plainMemory is external memory for the function, should contain at least such amount of memory as fem3DapplyL_memory_requirements return
    ///@see fem3DapplyL_memory_requirements
    template<typename Op, typename ScalarType = double, typename IndexType = int>
    void fem3DapplyL(
                  const DenseMatrix<const ScalarType>& XY0, const DenseMatrix<const ScalarType>& XY1,
                  const DenseMatrix<const ScalarType>& XY2, const DenseMatrix<const ScalarType>& XY3,
                        ArrayView<ScalarType> XYL,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU,
                  PlainMemory<ScalarType, IndexType> plainMemory){
        fem3DapplyL<Op, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), XYL, dofs, opU, plainMemory);
    } 
    ///@brief Evaluate \f$ D[x] * Op(u)[x] \f$
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, uint Ddim1, const Functor& Dfnc, DenseMatrix<>& opU,
                    PlainMemoryX<> mem, void *user_data = nullptr);
    ///@brief Evaluate \f$ Op(u)[x] \f$                
    template<typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& A, PlainMemoryX<> mem);

    ///Give memory requirement for external memory-dependent specialization of fem3DapplyL
    template<typename Op, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3DapplyL_memory_requirements(int pnt_per_tetra, int fusion = 1);
    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3DapplyL_memory_requirements(uint Ddim1, const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1);
    static inline PlainMemoryX<> fem3DapplyL_memory_requirements(const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1){
        return fem3DapplyND_memory_requirements_base<>(applyOpU, pnt_per_tetra, fusion, false);
    }                        

    /**
     * @brief Evaluate \f$ Op(u)[x] \f$
     * Memory-less nofuse version of fem3Dapply
     * @param[in] dofs is matrix [dofs_1, ..., dofs_f] where dofs_i is vector degrees of freedom associated with i-th tetrahedron
     *          e.g. for Op = Operator<GRAD, FemFix<FEM_P1>> dofs_i is d.o.f's vector of FEM_P1 element type 
     * @param[in,out] opU: opU = [opU_1, ..., opU_f] where opU_r = [vec(opU[p_{1r}^T])^T, ..., vec(opU[p_{qr}^T])^T]^T and 
     *          vec(opU[p_{ir}^T]) is vectrized matrix computed at point p_{ir} lied on r-th tetra 
     * @param[in] X are points for function evaluation
     * @param[in] XY0, XY1, XY2, XY3 are arrays of size 3 with tetra nodes
     * @tparam MAXPNTNUM is maximal number of points to be evaluated at the same time
     * @tparam Op is fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
     * @return result of evaluating Op(u) at XYG point set
     * @see AniMemory
     */
    template<typename Op, int MAXPNTNUM = 24, typename ScalarType = double, typename IndexType = int>
    void fem3DapplyX(
                  const ScalarType* XY0, const ScalarType* XY1,
                  const ScalarType* XY2, const ScalarType*  XY3,
                  const ArrayView<const ScalarType> X,
                  const ArrayView<ScalarType>& dofs, 
                  ArrayView<ScalarType> opU){
        fem3DapplyX<Op, MAXPNTNUM, ScalarType, IndexType>(Tetra<const ScalarType>(XY0, XY1, XY2, XY3), X, dofs, opU);
    }

    ///@brief Evaluate \f$ Op(u)[x] \f$
    ///External memory-dependent version of fem3Dapply
    ///@param plainMemory is external memory for the function, should contain at least such amount of memory as fem3DapplyX_memory_requirements return
    ///@see fem3DapplyX_memory_requirements
    template<typename Op, typename ScalarType = double, typename IndexType = int>
    void fem3DapplyX(
                  const ScalarType* XY0, const ScalarType* XY1,
                  const ScalarType* XY2, const ScalarType*  XY3,
                  const ArrayView<const ScalarType> X,
                  const ArrayView<ScalarType>& dofs, 
                  ArrayView<ScalarType> opU,
                  PlainMemory<ScalarType, IndexType> plainMemory){
        fem3DapplyX<Op, ScalarType, IndexType>(Tetra<const ScalarType>(XY0, XY1, XY2, XY3), X, dofs, opU, plainMemory);
    }            

    /**
     * @brief Evaluate \f$ Op(u)[x] \f$ at one point X
     * Memory-less nofuse version of fem3Dapply
     * @param[in] dofs is matrix [dofs_1, ..., dofs_f] where dofs_i is vector degrees of freedom associated with i-th tetrahedron
     *          e.g. for Op = Operator<GRAD, FemFix<FEM_P1>> dofs_i is d.o.f's vector of FEM_P1 element type 
     * @param[in,out] opU: opU = [opU_1, ..., opU_f] where opU_r = [vec(opU[p_{1r}^T])^T, ..., vec(opU[p_{qr}^T])^T]^T and 
     *          vec(opU[p_{ir}^T]) is vectrized matrix computed at point p_{ir} lied on r-th tetra 
     * @param[in] X is point for function evaluation
     * @param[in] XY0, XY1, XY2, XY3 are arrays of size 3 with tetra nodes
     * @tparam Op is fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
     * @return result of evaluating Op(u) at XYG point set
     * @see AniMemory
     */
    template<typename Op, typename ScalarType = double, typename IndexType = int>
    void fem3DapplyX(
                  const ScalarType* XY0, const ScalarType* XY1,
                  const ScalarType* XY2, const ScalarType* XY3,
                  const ScalarType* X,
                  const ArrayView<ScalarType> dofs, 
                  ArrayView<ScalarType>& opU){
        fem3DapplyX<Op, 1, ScalarType, IndexType>(XY0, XY1, XY2, XY3, ArrayView<const ScalarType>(X, 3), dofs, opU);            
    }
    ///@brief Evaluate \f$ D[x] * Op(u)[x] \f$
    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, uint Ddim1, const Functor& Dfnc, DenseMatrix<>& opU,
                    PlainMemoryX<> mem, void *user_data = nullptr); 
    ///@brief Evaluate \f$ Op(u)[x] \f$
    template<typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& opA, PlainMemoryX<> mem); 

    ///Give memory requirement for external memory-dependent specialization of fem3DapplyX
    template<typename Op, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> fem3DapplyX_memory_requirements(int pnt_per_tetra, int fusion = 1);
    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3DapplyX_memory_requirements(uint Ddim1, const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1);
    static inline PlainMemoryX<> fem3DapplyX_memory_requirements(const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1){
        return fem3DapplyND_memory_requirements_base<>(applyOpU, pnt_per_tetra, fusion, true);
    }
};
#include "eval.inl"
#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_EVAL_H