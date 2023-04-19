namespace Ani{
    namespace VDInternals{
        struct get_id_base { virtual unsigned operator[](unsigned i) const { return i; } };
        struct get_id_dat: public get_id_base { const unsigned* m_id; unsigned operator[](unsigned i) const override { return m_id[i]; } };
        template<typename Scalar, typename RandomIt>
        inline void performMatrixReorthogonalization(DenseMatrix <Scalar>& A, RandomIt dof_id, const DenseMatrix <Scalar>& Vorth, ArrayView <Scalar> mem){
            uint dim = Vorth.nCol;
            if (mem.size < static_cast<decltype(mem.size)>(dim*std::max(A.nCol, A.nRow)))
                throw std::runtime_error("Not enough of memory");
            DenseMatrix<Scalar> R(mem.data, dim, A.nCol);
            for (std::size_t j = 0; j < A.nCol; ++j)
                for (uint i = 0; i < dim; ++i){
                    R(i, j) = A(dof_id[i], j);
                }   
            for (std::size_t j = 0; j < A.nCol; ++j)
                for (uint m = 0; m < dim; ++m){
                    Scalar s = 0;
                    for (uint p = 0; p < dim; ++p)
                        s += R(p, j)*Vorth(m, p);
                    A(dof_id[m], j) = s;
                }              
        }
        template<typename Scalar, typename RandomIt>
        inline void performRhsReorthogonalization(DenseMatrix <Scalar>& F, RandomIt dof_id, const DenseMatrix <Scalar>& Vorth, ArrayView <Scalar> mem){
            uint dim = Vorth.nCol;
            if (mem.size < static_cast<decltype(mem.size) >(dim) )
                throw std::runtime_error("Not enough of memory");
            DenseMatrix<Scalar> R(mem.data, dim, 1);
            for (uint m = 0; m < dim; ++m)
                R(m, 0) = F(dof_id[m], 0);
            for (uint m = 0; m < dim; ++m){
                F(dof_id[m], 0) = 0;
                for (uint p = 0; p < dim; ++p)
                    F(dof_id[m], 0) += R(p, 0)*Vorth(m, p);
            }
        }
        template<typename Scalar, typename RandomIt>
        inline void setVectorDirMtx(    DenseMatrix <Scalar>& A, 
                                        RandomIt dof_id, 
                                        const DenseMatrix <Scalar>& Vorth,
                                        ArrayView <Scalar> mem,
                                        const uint ndc, const get_id_base& id){
            uint dim = Vorth.nCol;
            DenseMatrix<Scalar> C(mem.data, A.nRow, 1);
            for (uint k = 0; k < ndc; ++k){
                for (std::size_t j = 0; j < A.nRow; ++j){
                    Scalar s = 0;
                    for (uint p = 0; p < dim; ++p)
                        s += A(j, dof_id[p])*Vorth(id[k], p);
                    C(j, 1) = s;
                }
                for (uint l = 0; l < dim; ++l)
                    for (std::size_t j = 0; j < A.nRow; ++j)
                        A(j, dof_id[l]) -= C(j, 1)*Vorth(id[k], l);
                for (std::size_t j = 0; j < A.nCol; ++j)
                    A(dof_id[id[k]], j) = 0;
                for (uint m = 0; m < dim; ++m)
                    A(dof_id[id[k]], dof_id[m]) = Vorth(id[k], m);
            }
        }
        template<typename Scalar, typename RandomIt>
        inline void setVectorDirRhs(    DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F, 
                                        RandomIt dof_id, 
                                        const DenseMatrix <Scalar>& Vorth, const ArrayView <Scalar>& bc,
                                        ArrayView <Scalar> mem,
                                        const uint ndc, const get_id_base& id){
            (void) mem;
            uint dim = Vorth.nCol;
            for (uint k = 0; k < ndc; ++k){
                for (uint m = 0; m < dim; ++m){
                    uint did = dof_id[m];
                    for (std::size_t j = 0; j < A.nRow; ++j)
                        F(j, 0) -= A(j, did)*Vorth(id[k], m)*bc[k];
                }
                F(dof_id[id[k]], 0) = bc[k]; 
            }
        }
        template<typename Scalar, typename RandomIt>
        inline void setVectorDirResidual(DenseMatrix <Scalar>& F, 
                                        RandomIt dof_id, const uint ndc, const get_id_base& id){
            for (uint k = 0; k < ndc; ++k)
                F(dof_id[id[k]], 0) = 0; 
        }
    };

    template<typename Scalar, typename RandomIt>
    inline void applyVectorDirMatrix(   DenseMatrix <Scalar>& A, 
                                        RandomIt dof_id, 
                                        const DenseMatrix <Scalar>& Vorth,
                                        ArrayView <Scalar> mem,
                                        const uint ndc, const uint* dc_orth){
        using namespace VDInternals;
        get_id_base id_base;
        get_id_dat id_dat; id_dat.m_id = dc_orth;
        get_id_base& id = *static_cast<get_id_base*>(dc_orth == nullptr ? &id_base : static_cast<get_id_base*>(&id_dat));
        performMatrixReorthogonalization<>(A, dof_id, Vorth, mem);
        setVectorDirMtx<>(A, dof_id, Vorth, mem, ndc, id);
    }

    template<typename Scalar, typename RandomIt>
    inline void applyVectorDir( DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F,
                                RandomIt dof_id, 
                                const DenseMatrix <Scalar>& Vorth, const ArrayView <Scalar>& bc,
                                ArrayView <Scalar> mem,
                                const uint ndc, const uint* dc_orth){
        using namespace VDInternals;
        get_id_base id_base;
        get_id_dat id_dat; id_dat.m_id = dc_orth;
        get_id_base& id = *static_cast<get_id_base*>(dc_orth == nullptr ? &id_base : static_cast<get_id_base*>(&id_dat));
        performMatrixReorthogonalization<>(A, dof_id, Vorth, mem);
        performRhsReorthogonalization<>(F, dof_id, Vorth, mem);
        setVectorDirRhs<>(A, F, dof_id, Vorth, bc, mem, ndc, id);
        setVectorDirMtx<>(A, dof_id, Vorth, mem, ndc, id);
    }
    
    template<typename Scalar, typename RandomIt>
    inline void applyVectorDirResidual(DenseMatrix <Scalar>& F, RandomIt dof_id, 
                                const DenseMatrix <Scalar>& Vorth,
                                ArrayView <Scalar> mem,
                                const uint ndc, const uint* dc_orth){
        using namespace VDInternals;
        get_id_base id_base;
        get_id_dat id_dat; id_dat.m_id = dc_orth;
        get_id_base& id = *static_cast<get_id_base*>(dc_orth == nullptr ? &id_base : static_cast<get_id_base*>(&id_dat));                            
        performRhsReorthogonalization<>(F, dof_id, Vorth, mem);
        setVectorDirResidual<>(F, dof_id, ndc, id);
    }
};