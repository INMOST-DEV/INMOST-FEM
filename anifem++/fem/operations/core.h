//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_CORE_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_CORE_H

#include "anifem++/fem/fem_memory.h"
#include "anifem++/fem/operators.h"
#include "anifem++/fem/diff_tensor.h"
#include "anifem++/fem/quadrature_formulas.h"

#ifdef WITH_EIGEN
#include <Eigen/Dense>
#endif

#include <type_traits>
#include <algorithm>
#include <functional>
#include <cassert>
#include <string>
#include <stdexcept>

namespace Ani{
    ///Main general function to compute elemental FEM matrix at runtime
    ///@param applyOpU is action of linear diffential operator on trial space
    ///@param applyOpV is action of linear diffential operator on test space
    ///@param Dfnc should be Functor like DFunc structure having all traits from DfuncTraits
    /// More exactly, Dfnc should contain the following method and all traits defined in DfuncTraits structure
    ///\code
    ///TensorType operator()(const std::array<Scalar, 3> &X, Scalar *D, std::pair<uint, uint> Ddims, void *user_data, int iTet)
    ///\endcode
    /// where X {in} is coordinate inside some tetrahedron, D {out} is storage for row-major matrix with dimensions Ddims.first (nRow) and Ddims.second (nCol),
    /// user_data is user specific data block and iTet is number of tetrahedron from range from 0 to f-1 (f is number of tetrahedrons)
    ///or
    ///\code
    ///TensorType(ArrayR X, ArrayR D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem)
    ///\endcode
    /// where X contains coordinates of points inside tetrahedron where Xk[i] = X[k*3 * i] correspond to mem.XYL_k barycentric coords,
    ///D is memory to store result as D[i + Ddims.first*(j + Ddims.second*k)] = Dk(i, j), Ddims are dimensions of D tensor at point
    ///@param[in,out] A contains memory (A.data) to store resulting elemental matrix
    ///@param mem is all required memory for function work, memory should be initially inited using internalFem3DtetGeomInit
    ///@param user_data some data for Dfnc tensor
    ///@param dotWithNormal: if true then \f$ (D \mathrm{OpA}(u)) \cdot \mathbf{N} \f$ will be computed instead \f$ D \mathrm{OpA}(u) \f$,
    /// it's used to imply some Neumann-like or Robin-like BC
    template<typename DFUNC>
    void internalFem3Dtet(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, DFUNC& Dfnc, DenseMatrix<> A, 
                          AniMemoryX<>& mem, void *user_data = nullptr, bool dotWithNormal = false);

    ///Main general function to compute elemental FEM matrix with compile time operators
    ///@tparam OpA is left fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
    ///@tparam OpB is right fem operator
    ///@tparam DotWithNormal: if true then \f$ (D \mathrm{OpA}(u)) \cdot \mathbf{N} \f$ will be computed instead \f$ D \mathrm{OpA}(u) \f$,
    /// it's used to imply some Neumann-like or Robin-like BC
    ///@see internalFem3Dtet
    template<typename OpA, typename OpB,  bool DotWithNormal = false, typename DFUNC, typename Scalar = double, typename IndexType = int>
    void internalFem3Dtet(DFUNC& Dfnc,
                          DenseMatrix<Scalar> A,
                          AniMemory<Scalar, IndexType>& mem,
                          void *user_data);                      

    ///Initialize geometrical data in mem structure
    template<typename MatrTpXY0, typename MatrTpXY1, typename MatrTpXY2, typename MatrTpXY3, typename Scalar = double, typename IndexType = int>
    void internalFem3DtetGeomInit(const MatrTpXY0 &XY0, const MatrTpXY1 &XY1, const MatrTpXY2 &XY2, const MatrTpXY3 &XY3, AniMemory<Scalar, IndexType>& mem);
    template<typename MatrTpXY0, typename MatrTpXY1, typename MatrTpXY2, typename MatrTpXY3, typename Scalar = double, typename IndexType = int>
    void internalFem3DtetGeomInit_Tet(const MatrTpXY0 &XY0, const MatrTpXY1 &XY1, const MatrTpXY2 &XY2, const MatrTpXY3 &XY3, AniMemory <Scalar, IndexType> &mem);
    template<typename MatrTpXY0, typename Scalar = double, typename IndexType = int>
    void internalFem3DtetGeomInit_XYG(const MatrTpXY0 &XY0, AniMemory <Scalar, IndexType> &mem);

    /**
     * @brief General function to evaluate FEM functions at points, compute \f$ D[x]*Op(u)[x] \f$
     * @param applyOpU is action of linear diffential operator on space
     * @param[in] dofs is matrix [dofs_1, ..., dofs_f] where dofs_i is vector degrees of freedom associated with i-th tetrahedron
     *          e.g. for Op = Operator<GRAD, FemFix<FEM_P1>> dofs_i is d.o.f's vector of FEM_P1 element type 
     * @param[in,out] opU: opU = [opU_1, ..., opU_f] where opU_r = [vec(opU[p_{1r}^T])^T, ..., vec(opU[p_{qr}^T])^T]^T and 
     *          vec(opU[p_{ir}^T]) is vectrized matrix computed at point p_{ir} lied on r-th tetra 
     * @param mem is all required memeory for function work, memory should be initially inited using internalFem3DtetGeomInit
     *          mem.XYG is used as matrix of points used for evaluation and should be inited correspondingly
     * @param user_data some data for Dfnc tensor
     * @param dotWithNormal: if true then \f$ (D \mathrm{OpA}(u)) \cdot \mathbf{N} \f$ will be computed instead \f$ D \mathrm{OpA}(u) \f$, where N is stored in mem.NRM
     * @return result of evaluating D*Op(u) at XYG point set
     * @warning The function will overwrite mem.U
     * @see AniMemory
     */
    template<typename DFUNC>
    void internalFem3DApply(const ApplyOpBase& applyOpU, const DenseMatrix<const double>& dofs, DFUNC& Dfnc, TensorDims Ddims, DenseMatrix<> opU, 
                            AniMemoryX<>& mem, void *user_data = nullptr, bool dotWithNormal = false);
    /// @brief Specialization for evaluating \f$ Op(u)[x] \f$ @see internalFem3DApply
    static inline void internalFem3DApply(const ApplyOpBase& applyOpU, const DenseMatrix<const double>& dofs, DenseMatrix<> opU, AniMemoryX<>& mem);                          
    /// @brief Specialization for evaluating \f$ Op(u)[x] \f$ @see internalFem3DApply
    /// @tparam Op is fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
    template<typename Op, typename Scalar = double, typename IndexType = int>      
    void internalFem3DApply(const DenseMatrix<Scalar>& dofs, DenseMatrix<Scalar> opU, AniMemory<Scalar, IndexType>& mem);                                               

    /**
     * @brief Computes barycentric coordinates of the points relative to tetra
     * @param[in]  XY0,XY1,XY2,XY3 are vertices of tetra (should have size=3)
     * @param[in]  X is array of point
     * @param[out] XYL is array of barycentric coordinate of points X
     */
    template<typename ScalarType = double>
    void getBaryCoord(
                  const ScalarType* XY0/*[3]*/, const ScalarType* XY1/*[3]*/,
                  const ScalarType* XY2/*[3]*/, const ScalarType* XY3/*[3]*/,
                  const ArrayView<const ScalarType> X, ArrayView<ScalarType> XYL);
    template<typename ScalarType = double>
    void getBaryCoord(
                  const ScalarType* XY0/*[3]*/, const ScalarType* XY1/*[3]*/,
                  const ScalarType* XY2/*[3]*/, const ScalarType* XY3/*[3]*/,
                  const ArrayView<const ScalarType> X, 
                  ArrayView<ScalarType> XYL, ArrayView<ScalarType> PSI);

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType, bool CopyQuad = true, bool DotWithNormal = false>
    AniMemory<ScalarType, IndexType> fem3Dtet_init_animem_from_plain_memory(int q, int f, PlainMemory<ScalarType, IndexType> plainMemory);

    template<typename DFUNC, typename ScalarType, typename IndexType>
    AniMemoryX<ScalarType, IndexType> fem3D_init_animem_from_plain_memory(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, 
        DFUNC& Dfnc, uint pnt_per_tetra, uint fusion, PlainMemoryX<ScalarType, IndexType> mem, bool dotWithNormal = true, bool copyQuadVals = true);

    template<typename DFUNC, typename ScalarType, typename IndexType>
    AniMemoryX<ScalarType, IndexType> fem3DapplyD_init_animem_from_plain_memory(const ApplyOpBase& applyOpU, 
        uint Ddim1, DFUNC& Dfnc, uint pnt_per_tetra, uint fusion, PlainMemoryX<ScalarType, IndexType> mem, bool dotWithNormal = false);

    template<typename ScalarType, typename IndexType>
    AniMemoryX<ScalarType, IndexType> fem3DapplyND_init_animem_from_plain_memory(const ApplyOpBase& applyOpU, uint pnt_per_tetra, uint fusion, PlainMemoryX<ScalarType, IndexType> mem, bool copyQuadVals = true);            
    
    template<typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int>
    PlainMemoryX<ScalarType, IndexType> fem3D_memory_requirements_base(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, uint pnt_per_tetra, uint fusion, bool dotWithNormal = true, bool copyQuadVals = true);

    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int, bool CopyQuadVals = true, bool DotWithNormal = false>
    PlainMemory <ScalarType, IndexType> fem3D_memory_requirements_base(int pnt_per_tetra, int fusion = 1);

    template<typename OpA, typename OpB, typename ScalarType = double, typename IndexType = int, bool OnFace = false, bool DotWithNormal = false>
    PlainMemory <ScalarType, IndexType> fem3D_memory_requirements(int order, int fusion = 1);
    
    template<typename FuncTraits = DfuncTraits<>, typename ScalarType = double, typename IndexType = int>
    PlainMemoryX<ScalarType, IndexType> fem3DapplyD_memory_requirements_base(const ApplyOpBase& applyOpU, uint Ddim1, uint pnt_per_tetra = 1, uint fusion = 1, bool dotWithNormal = false);
    
    template<typename ScalarType = double, typename IndexType = int>
    PlainMemoryX<ScalarType, IndexType> fem3DapplyND_memory_requirements_base(const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1, bool copyQuadVals = true);

    ///Computes in fuse manner matrix mutiplication R = A^T * B
    ///For some cases Eigen matrix multiplication is more effective than hand-written cycles
    ///@param not_use_eigen if false than matrix mutiplication will be computed using Eigen
    ///@param[in] A = [A_1, ..., A_r], |A_i| = nRow x ACol, r = fuse
    ///@param[in] B = [B_1, ..., B_r], |B_i| = nRow x BCol, r = fuse
    ///@param[in,out] R = [R_1, ..., R_r] R_i = A_i^T * B_i, |R_i| = ACol x BCol
    template<typename Scalar = double, typename IndexType0 = int, typename IndexType1 = int, typename IndexType2 = int, typename IndexType3 = int>
    void fusive_AT_mul_B(bool not_use_eigen,
                            IndexType0 fuse, IndexType1 nRow,
                            Scalar* A, IndexType2 ACol,
                            Scalar* B, IndexType3 BCol,
                            Scalar* R);

    ///Computes in fuse manner matrix mutiplication R = A * B
    ///For some cases Eigen matrix multiplication is more effective than hand-written cycles
    ///@param not_use_eigen if false than matrix mutiplication will be computed using Eigen
    ///@param[in] A = [A_1, ..., A_r], |A_i| = ARow x ACol, r = fuse
    ///@param[in] B = [B_1, ..., B_r], |B_i| = ACol x BCol, r = fuse
    ///@param[in,out] R = [R_1, ..., R_r] R_i = A_i * B_i, |R_i| = ARow x BCol
    template<typename Scalar = double, typename IndexType0 = int, typename IndexType1 = int, typename IndexType2 = int, typename IndexType3 = int>
    inline void fusive_A_mul_B(bool not_use_eigen,
                                IndexType0 fuse, IndexType1 ACol,
                                Scalar* A, IndexType2 ARow,
                                Scalar* B, IndexType3 BCol,
                                Scalar* R);
    
    /// Handler to perform tensor convolution with specified normal
    template <bool DotWithNormal>
    struct DotWithNormalHelper{
        using nD = std::integral_constant<int, 1>;
        template<typename Scalar, typename IndexType>
        static inline void ProcessNormal(DenseMatrix<Scalar>& DU, AniMemory <Scalar, IndexType> &mem){ (void)DU; (void)mem; }
    };
    template <>
    struct DotWithNormalHelper<true>{
        using nD = std::integral_constant<int, 3>;
        template<typename Scalar, typename IndexType>
        static inline void ProcessNormal(DenseMatrix<Scalar>& DU, AniMemory <Scalar, IndexType> &mem){
            auto jdim = DU.nRow / mem.q / nD::value;
            auto nfA = DU.nCol / mem.f;
            for (decltype(mem.f) r = 0; r < mem.f; ++r){
                auto normal = mem.NRM.data + 3*r;
                for (decltype(nfA) i = 0; i < nfA; ++i)
                    for (decltype(mem.q) n = 0; n < mem.q; ++n)
                        for (decltype(jdim) k = 0; k < jdim; ++k){
                            DU.data[k + jdim*n + jdim*mem.q*(i + nfA*r)] = 
                                                        DU(0 + nD::value * (k + jdim*n), i + nfA*r) * normal[0] +
                                                        DU(1 + nD::value * (k + jdim*n), i + nfA*r) * normal[1] +
                                                        DU(2 + nD::value * (k + jdim*n), i + nfA*r) * normal[2];
                        }
            }
            DU.nRow /= nD::value;
        }
    };  

    template<typename TETRAS_TYPE, typename Scalar, typename IndexType>
    void internalFem3DtetGeomInit(const TETRAS_TYPE &tetras, AniMemory <Scalar, IndexType> &mem){
        internalFem3DtetGeomInit(tetras.XY0, tetras.XY1, tetras.XY2, tetras.XY3, mem);
    }                         
};

#include "core.inl"

#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_CORE_H