//
// Created by Liogky Alexey on 23.01.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_DC_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_DC_H

#include "anifem++/fem/fem_memory.h"
#include <algorithm>

namespace Ani{
    ///@see applyDir
    template<typename Scalar>
    inline void applyDirMatrix(DenseMatrix <Scalar>& A, int k){
        std::fill(A.data + k * A.nRow, A.data + (k+1) * A.nRow, 0);
        for (std::size_t i = 0; i < A.nRow; ++i) A(k, i) = 0;
        A(k, k) = 1.0;
    }

    ///Set dirichlet value on k-th tetrahedron's dof
    ///@param A is elemental matrix
    ///@param F is elemental right hand side
    ///@param k is number of tetrahedron dof
    ///@param bc is dirichlet value to be set
    template<typename Scalar>
    inline void applyDir(DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F, int k, Scalar bc) {
        for (std::size_t i = 0; i < A.nRow; ++i)
            F(i, 0) -= A(i, k) * bc;
        F(k, 0) = bc;

        applyDirMatrix<Scalar>(A, k);
    }

    ///Set zero on k-th position in F vector
    ///@see applyDir
    template<typename Scalar>
    inline void applyDirResidual(DenseMatrix <Scalar>& F, int k) {
        F(k, 0) = 0.0;
    } 

    ///Set dirichlet condition V*u = b, where V \in R^{ndc x d}, V^T*V = I_ndc, d = vec_dim(u), i.e.
    /// u = [u_1, .., u_d], and all u_i is discritized by same fem FEM_U space
    ///@param A is elemental matrix
    ///@param dof_id[d] is positions of d.o.fs associated with some basis function phi \in FEM_U for every variable u_1, u_2, ..., u_d in vector of all d.o.fs
    ///@param Vorth(d, d) is orthogonal matrix
    ///@param mem[d*std::max(A.nCol, A.nRow)] is work memory
    ///@param ndc is count of rows in V matrix
    ///@param dc_orth[ndc] is such that V(i, *) = Vorth(dc_orth[i], *), 
    /// if dc_orth == nullptr than V(i, *) = Vorth(i, *)
    ///@tparam RandomIt is some container of integers with defined operator[] (e.g. random_access_iterator or const uint*)
    template<typename Scalar, typename RandomIt>
    inline void applyVectorDirMatrix(   DenseMatrix <Scalar>& A,
                                        RandomIt dof_id, 
                                        const DenseMatrix <Scalar>& Vorth,
                                        ArrayView <Scalar> mem,
                                        const uint ndc, const uint* dc_orth = nullptr);
    ///@param F is elemental right hand side
    ///@param bc[ndc] is is dirichlet value to be set (rhs in V*u = b)
    ///@param mem[d*std::max(A.nCol, A.nRow)] is work memory
    ///@see applyVectorDirMatrix                                   
    template<typename Scalar, typename RandomIt>
    inline void applyVectorDir( DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F,
                                RandomIt dof_id, 
                                const DenseMatrix <Scalar>& Vorth, const ArrayView <Scalar>& bc,
                                ArrayView <Scalar> mem,
                                const uint ndc, const uint* dc_orth = nullptr); 
    ///Set zero on dirichlet position of rhs
    ///@param mem[d] is work memory
    ///@see applyDir
    template<typename Scalar, typename RandomIt>
    inline void applyVectorDirResidual(
                                DenseMatrix <Scalar>& F, RandomIt dof_id, 
                                const DenseMatrix <Scalar>& Vorth,
                                ArrayView <Scalar> mem,
                                const uint ndc, const uint* dc_orth = nullptr);                                                             
                                        
};

#include "dc_on_dof.inl"

#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_DC_H