//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_TET_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_TET_H
#include "core.h"
#include <functional>
#include <array>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace Ani{
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3Dtet(const TetrasT& XYZ,
                  const Functor& Dfnc,
                  DenseMatrix<ScalarType>& A,
                  int order = 5,
                  void *user_data = nullptr);
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor, typename TetrasT>
    void fem3Dtet(const TetrasT& XYZ,
                  const Functor& Dfnc,
                  DenseMatrix<ScalarType>& A,
                  PlainMemory<ScalarType, IndexType> plainMemory,
                  int order = 5,
                  void *user_data = nullptr);

    ///The function to compute elemental matrix for the volume integral
    ///over tetrahedron T: \f$\int_T [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///Memory-less version of fem3Dtet
    ///@tparam OpA is left fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
    ///@tparam OpB is right fem operator
    ///@tparam FuncTraits is traits will be associated with tensor fuction Dfnc
    ///@tparam FUSION is maximal available number of columns in XYi arrays
    ///@param XY0, XY1, XY2, XY3 are 3xf matrices where r-th column contains coordinates of r-th tetrahedron
    ///@param Dfnc should be functor with signature
    ///\code
    /// TensorType(const std::array<double, 3>& x, double* Dmem, std::pair<uint, uint> Ddims, void* user_data, int iTet)
    ///\endcode
    /// where x is coordinate inside some tetrahedron, D is storage for row-major matrix with dimensions Ddims.first (nRow) and Ddims.second (nCol),
    /// user_data is user specific data block and iTet is number of tetrahedron from range from 0 to f-1 (f is number of tetrahedrons)
    ///@param[in,out] A is matrix to store result elemental matrix
    ///@param order is order of used gauss quadrature formula
    ///@param user_data is user defined data block will be propagate to Dfnc
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int, typename Functor>
    void fem3Dtet(const DenseMatrix<ScalarType>& XY0,
                  const DenseMatrix<ScalarType>& XY1,
                  const DenseMatrix<ScalarType>& XY2,
                  const DenseMatrix<ScalarType>& XY3,
                  const Functor& Dfnc,
                  DenseMatrix<ScalarType>& A,
                  int order = 5,
                  void *user_data = nullptr){                   
        fem3Dtet<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), Dfnc, A, order, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, int FUSION = 1, typename ScalarType = double, typename IndexType = int, typename Functor>
    void fem3Dtet(const DenseMatrix<const ScalarType>& XY0,
                  const DenseMatrix<const ScalarType>& XY1,
                  const DenseMatrix<const ScalarType>& XY2,
                  const DenseMatrix<const ScalarType>& XY3,
                  const Functor& Dfnc,
                  DenseMatrix<ScalarType>& A,
                  int order = 5,
                  void *user_data = nullptr){
        DenseMatrix<ScalarType>
                _XY0(const_cast<ScalarType*>(XY0.data), XY0.nRow, XY0.nCol, XY0.size),
                _XY1(const_cast<ScalarType*>(XY1.data), XY1.nRow, XY1.nCol, XY1.size),
                _XY2(const_cast<ScalarType*>(XY2.data), XY2.nRow, XY2.nCol, XY2.size),
                _XY3(const_cast<ScalarType*>(XY3.data), XY3.nRow, XY3.nCol, XY3.size);
        fem3Dtet<OpA, OpB, FuncTraits, FUSION, ScalarType, IndexType, Functor>(
                 _XY0, _XY1, _XY2, _XY3,
                 Dfnc, A, order, user_data);
    }

#ifdef WITH_EIGEN
    ///The function to compute elemental matrix for the volume integral
    ///over tetrahedron T: \f$\int_T [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///@param Dfnc is functor with the following args: X is coordinate in tetrahedron,
    /// D is memory for col-major dense matrix with sizes stored in Ddims (iDim, jDim), user_data is user specific data,
    /// iTet is local tetrahedron number in selection (start from 0)
    ///@see other specialization of fem3Dtet
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename Derived>
    void fem3Dtet(const Eigen::Vector3d& XY0, const Eigen::Vector3d& XY1, const Eigen::Vector3d& XY2, const Eigen::Vector3d& XY3,
                  const std::function<TensorType(const std::array<double, 3>& X, double* D, std::pair<uint, uint> Ddims, void* user_data, int iTet)>& Dfnc,
                  Eigen::MatrixBase<Derived> const & A, int order = 5, void *user_data = nullptr);
#endif

    ///The function to compute elemental matrix for the volume integral
    ///over tetrahedron T: \f$\int_T [(D \dot OpA(u)) \dot OpB(v)] dx \f$
    ///External memory-dependent version of fem3Dtet
    ///@tparam OpA is left fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
    ///@tparam OpB is right fem operator
    ///@tparam FuncTraits is traits will be associated with tensor fuction Dfnc
    ///@tparam FUSION is maximal available number of columns in XYi arrays
    ///@param XY0, XY1, XY2, XY3 are 3xf matrices where r-th column contains coordinates of r-th tetrahedron
    ///@param Dfnc should be functor with signature
    ///\code
    /// TensorType(const std::array<double, 3>& x, double* Dmem, std::pair<uint, uint> Ddims, void* user_data, int iTet)
    ///\endcode
    /// where x is coordinate inside some tetrahedron, D is storage for row-major matrix with dimensions Ddims.first (nRow) and Ddims.second (nCol),
    /// user_data is user specific data block and iTet is number of tetrahedron from range from 0 to f-1 (f is number of tetrahedrons)
    ///@param[in,out] A is matrix to store result elemental matrix
    ///@param plainMemory is external memory for the function, should contain at least such amount of memory as fem3Dtet_memory_requirements return
    ///@param order is order of used gauss quadrature formula
    ///@param user_data is user defined data block will be propagate to Dfnc
    ///@see fem3Dtet_memory_requirements
    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor>
    void fem3Dtet(const DenseMatrix<ScalarType>& XY0,
                  const DenseMatrix<ScalarType>& XY1,
                  const DenseMatrix<ScalarType>& XY2,
                  const DenseMatrix<ScalarType>& XY3,
                  const Functor& Dfnc,
                  DenseMatrix<ScalarType>& A,
                  PlainMemory<ScalarType, IndexType> plainMemory,
                  int order = 5,
                  void *user_data = nullptr){                     
        fem3Dtet<OpA, OpB, FuncTraits, ScalarType, IndexType>(make_tetras(XY0.data, XY1.data, XY2.data, XY3.data, XY0.nCol), Dfnc, A, plainMemory, order, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int, typename Functor>
    void fem3Dtet(const DenseMatrix<const ScalarType>& XY0,
                  const DenseMatrix<const ScalarType>& XY1,
                  const DenseMatrix<const ScalarType>& XY2,
                  const DenseMatrix<const ScalarType>& XY3,
                  const Functor& Dfnc,
                  DenseMatrix<ScalarType>& A,
                  PlainMemory<ScalarType, IndexType> plainMemory,
                  int order = 5,
                  void *user_data = nullptr){
        DenseMatrix<ScalarType>
                _XY0(const_cast<ScalarType*>(XY0.data), XY0.nRow, XY0.nCol, XY0.size),
                _XY1(const_cast<ScalarType*>(XY1.data), XY1.nRow, XY1.nCol, XY1.size),
                _XY2(const_cast<ScalarType*>(XY2.data), XY2.nRow, XY2.nCol, XY2.size),
                _XY3(const_cast<ScalarType*>(XY3.data), XY3.nRow, XY3.nCol, XY3.size);
        fem3Dtet<OpA, OpB, FuncTraits, ScalarType, IndexType, Functor>(
                _XY0, _XY1, _XY2, _XY3,
                Dfnc, A, plainMemory, order, user_data);
    }

    template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
    void fem3Dtet(  const TetrasT& XYZ, 
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order = 5, void *user_data = nullptr );

    
    ///Give memory requirement for external memory-dependent specialization of fem3Dtet
    ///@param order is order of quadrature formula will be used
    ///@param fusion is fusion parameter will be used
    ///@return PlainMemory where were set dSize and iSize fields to required amount of memory
    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int>
    PlainMemory<ScalarType, IndexType> fem3Dtet_memory_requirements(int order, int fusion = 1);

    template<typename FuncTraits = DfuncTraits<>>
    PlainMemoryX<> fem3Dtet_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, int order, int fusion = 1) {
        return fem3D_memory_requirements_base<FuncTraits>(applyOpU, applyOpV, tetrahedron_quadrature_formulas(order).GetNumPoints(), fusion, false, !std::is_same<double, qReal>::value);
    }
};

#include "int_tet.inl"

#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_INT_TET_H