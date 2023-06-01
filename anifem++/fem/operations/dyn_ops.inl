namespace Ani{
    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3Dtet(  const TetrasT& XYZ, const Functor& Dfnc, DenseMatrix<ScalarType>& A, 
                    DynMem<ScalarType, IndexType>& wmem, int order, void *user_data){
        auto req = fem3Dtet_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3Dtet<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, Dfnc, A, req, order, user_data);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dtet(  const TetrasT& XYZ, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, int order, void *user_data){
        auto req = fem3Dtet_memory_requirements<FuncTraits>(applyOpU, applyOpV, order, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3Dtet<FuncTraits>(XYZ, applyOpU, applyOpV, Dfnc, A, req, order, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ, int face_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, int order, void *user_data){
        auto req = fem3Dface_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3Dface<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, face_num, Dfnc, A, req, order, user_data);
    }
    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ, int face_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, int order, void *user_data){
        auto req = fem3DfaceN_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3DfaceN<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, face_num, Dfnc, A, req, order, user_data);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dface(const TetrasT& XYZ, int face_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                   DynMem<>& wmem, int order, void *user_data){
        auto req = fem3Dface_memory_requirements<FuncTraits>(applyOpU, applyOpV, order, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3Dface<FuncTraits>(XYZ, face_num, applyOpU, applyOpV, Dfnc, A, req, order, user_data);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ, int face_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                   DynMem<>& wmem, int order, void *user_data){
        auto req = fem3DfaceN_memory_requirements<FuncTraits>(applyOpU, applyOpV, order, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DfaceN<FuncTraits>(XYZ, face_num, applyOpU, applyOpV, Dfnc, A, req, order, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3Dedge(const TetrasT& XYZ, int edge_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, int order, void *user_data){
        auto req = fem3Dedge_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3Dedge<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, edge_num, Dfnc, A, req, order, user_data);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dedge( const TetrasT& XYZ, int edge_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, int order, void *user_data){
        auto req = fem3Dedge_memory_requirements<FuncTraits>(applyOpU, applyOpV, order, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3Dedge<FuncTraits>(XYZ, edge_num, applyOpU, applyOpV, Dfnc, A, req, order, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3Dnode(const TetrasT& XYZ, int node_num, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, void *user_data){
        auto req = fem3Dnode_memory_requirements<OpA, OpB, ScalarType, IndexType>(XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3Dnode<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, node_num, Dfnc, A, req, user_data);
    } 
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dnode( const TetrasT& XYZ, int node_num, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, void *user_data){
        auto req = fem3Dnode_memory_requirements<FuncTraits>(applyOpU, applyOpV, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3Dnode<FuncTraits>(XYZ, node_num, applyOpU, applyOpV, Dfnc, A, req, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3DpntL(const TetrasT& XYZ, ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, void *user_data){
        auto req = fem3DpntL_memory_requirements<OpA, OpB, ScalarType, IndexType>(XYL.size/4, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3DpntL<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, XYL, WG, Dfnc, A, req, user_data);
    } 
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3DpntL( const TetrasT& XYZ, ArrayView<> XYL, ArrayView<> WG, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, void *user_data ){
        auto req = fem3DpntL_memory_requirements<FuncTraits>(applyOpU, applyOpV, XYL.size/4, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DpntL<FuncTraits>(XYZ, XYL, WG, applyOpU, applyOpV, Dfnc, A, req, user_data);
    }
    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetraScalarTp>
    void fem3DpntX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG, const Functor& Dfnc, DenseMatrix<ScalarType>& A,
                   DynMem<ScalarType, IndexType>& wmem, void *user_data){
        auto req = fem3DpntX_memory_requirements<OpA, OpB, ScalarType, IndexType>(X.size/3, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata;
        fem3DpntX<OpA, OpB, FuncTraits, ScalarType, IndexType>(XYZ, X, WG, Dfnc, A, req, user_data);
    }
    template<typename FuncTraits, typename Functor, typename TetraScalarTp>
    void fem3DpntX( const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, ArrayView<> WG, const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    DynMem<>& wmem, void *user_data ){
        auto req = fem3DpntX_memory_requirements<FuncTraits>(applyOpU, applyOpV, X.size/3, XYZ.fusion); 
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DpntX<FuncTraits>(XYZ, X, WG, applyOpU, applyOpV, Dfnc, A, req, user_data);
    }

    template<typename Op, typename ScalarType, typename IndexType, typename TetrasT>
    void fem3DapplyL( const TetrasT& XYZ, ArrayView<ScalarType> XYL, const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU, DynMem<ScalarType, IndexType>& wmem){
        auto req = fem3DapplyL_memory_requirements<Op, ScalarType, IndexType>(XYL.size/4, XYZ.fusion);  
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata; 
        fem3DapplyL<Op, ScalarType, IndexType>(XYZ, XYL, dofs, opU, req);          
    }
    template<typename Op, typename ScalarType, typename IndexType, typename TetraScalarTp>
    void fem3DapplyX( const Tetra<TetraScalarTp>& XYZ, const ArrayView<const ScalarType> X, const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU, DynMem<ScalarType, IndexType>& wmem){
        auto req = fem3DapplyX_memory_requirements<Op, ScalarType, IndexType>(X.size/3, XYZ.fusion);  
        auto mem = wmem.alloc(req.dSize, req.iSize, 0);
        req.ddata = mem.m_mem.ddata, req.idata = mem.m_mem.idata; 
        fem3DapplyX<Op, ScalarType, IndexType>(XYZ, X, dofs, opU, req);          
    }
    template<typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& A, DynMem<>& wmem){
        auto req = fem3DapplyL_memory_requirements(XYL.size/4, XYZ.fusion);                
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DapplyL(XYZ, XYL, dofs, applyOpU, A, req);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs, const ApplyOpBase& applyOpU, uint Ddim1, 
                    const Functor& Dfnc, DenseMatrix<>& opU, DynMem<>& wmem, void *user_data){
        auto req = fem3DapplyL_memory_requirements<FuncTraits>(Ddim1, opU, XYL.size/4, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DapplyL<FuncTraits>(XYZ, XYL, dofs, applyOpU, Ddim1, Dfnc, opU, req, user_data);
    }
    template<typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& opA, DynMem<>& wmem){
        auto req = fem3DapplyX_memory_requirements(X.size/3, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DapplyX(XYZ, X, dofs, applyOpU, opA, wmem);
    }
    template<typename FuncTraits, typename Functor, typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, uint Ddim1, const Functor& Dfnc, DenseMatrix<>& opU, DynMem<>& wmem, void *user_data){
        auto req = fem3DapplyX_memory_requirements<FuncTraits>(Ddim1, opU, X.size/3, XYZ.fusion);
        auto mem = wmem.alloc(req.dSize, req.iSize, req.mSize);
        req = mem.m_mem;
        fem3DapplyX<FuncTraits>(XYZ, X, dofs, applyOpU, Ddim1, Dfnc, opU, req, user_data);
    }
};