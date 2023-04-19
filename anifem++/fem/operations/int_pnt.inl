namespace Ani{
    template<typename OpA, typename OpB, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3Dnode_memory_requirements(int fusion){
        return fem3DpntL_memory_requirements<OpA, OpB, ScalarType, IndexType>(1, fusion);
    }

    template<typename OpA, typename OpB, typename FuncTraits, int FUSION, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3Dnode(const TetrasT& XYZ,
              int node_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A, void *user_data) {
        MemoryLegacy<ScalarType, FUSION> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        IndexType f = XYZ.fusion;
        if (f > FUSION) throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(node_num < 4 && node_num >= 0 && "Wrong node index");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        std::fill(mem.XYL.data, mem.XYL.data + 4, 0);
        mem.XYL.data[node_num] = 1., mem.WG.data[0] = 1.;
        mem.f = f, mem.q = 1;
        internalFem3DtetGeomInit(XYZ, mem);
        std::fill(mem.MES.data, mem.MES.data+f, 1);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3Dnode(const TetrasT& XYZ,
              int node_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A,
              PlainMemory <ScalarType, IndexType> plainMemory, void *user_data) {
        auto f = XYZ.fusion;
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        int q = 1;
#ifndef	NDEBUG
        auto t = fem3Dnode_memory_requirements<OpA, OpB, ScalarType, IndexType>(f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, true, false>(q, f, plainMemory);
        std::fill(mem.XYL.data, mem.XYL.data + 4*q, 0);
        mem.XYL.data[node_num] = 1., mem.WG.data[0] = 1.;

        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        internalFem3DtetGeomInit(XYZ, mem);
        std::fill(mem.MES.data, mem.MES.data+f, 1);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, int FUSION, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3DpntL(const TetrasT& XYZ,
              ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG, const Functor &Dfnc, DenseMatrix <ScalarType> &A, void *user_data) {
        MemoryLegacy<ScalarType, FUSION> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        IndexType f = XYZ.fusion, q = std::min(XYL.size/4, WG.size);
        if (f > FUSION) throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        if (q > 24)
            throw std::runtime_error("Point arrays with size more 24 are not supported for stack memory version of the function");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        mem.XYL.Init(XYL.data, 4*q);
        mem.WG.Init(WG.data, q);
        mem.f = f, mem.q = q;
        internalFem3DtetGeomInit(XYZ, mem);
        std::fill(mem.MES.data, mem.MES.data+f, 1);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3DpntL(const TetrasT& XYZ,
              ArrayView<ScalarType> XYL, ArrayView<ScalarType> WG, const Functor &Dfnc, DenseMatrix <ScalarType> &A,
              PlainMemory <ScalarType, IndexType> plainMemory, void *user_data) {
        auto f = XYZ.fusion;
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        int q = XYL.size/4;
#ifndef	NDEBUG
        auto t = fem3DpntL_memory_requirements<OpA, OpB, ScalarType, IndexType>(q, f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, false, false>(q, f, plainMemory);
        mem.XYL.Init(XYL.data, 4*q);
        mem.WG.Init(WG.data, q);

        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        internalFem3DtetGeomInit(XYZ, mem);
        std::fill(mem.MES.data, mem.MES.data+f, 1);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetraScalarTp>
    void
    fem3DpntX(const Tetra<TetraScalarTp>& XYZ,
              const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG, const Functor &Dfnc, DenseMatrix <ScalarType> &A, void *user_data) {
        MemoryLegacy<ScalarType, 1, 24> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        IndexType f = 1, q = std::min(X.size/3, WG.size);
        if (q > 24)
            throw std::runtime_error("Point arrays with size more 24 are not supported for stack memory version of the function");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        mem.WG.Init(WG.data, q);
        mem.f = f, mem.q = q;
        ScalarType XYLp[4*24];
        ArrayView<ScalarType> XYL(XYLp, 4*q);
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, mem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, XYL, mem.PSI);
        mem.XYL = XYL;
        internalFem3DtetGeomInit_XYG(XYZ.XY0, mem);

        std::fill(mem.MES.data, mem.MES.data+f, 1);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetraScalarTp>
    void
    fem3DpntX(const Tetra<TetraScalarTp>& XYZ,
              const ArrayView<const ScalarType> X, ArrayView<ScalarType> WG, const Functor &Dfnc, DenseMatrix <ScalarType> &A,
              PlainMemory <ScalarType, IndexType> plainMemory, void *user_data) {
        int f = 1, q = std::min(X.size/3, WG.size);
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
#ifndef	NDEBUG
        auto t = fem3DpntX_memory_requirements<OpA, OpB, ScalarType, IndexType>(q, f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, true, false>(q, f, plainMemory);
        
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, mem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, mem.XYL, mem.PSI);
        mem.WG.Init(WG.data, q);
        internalFem3DtetGeomInit_XYG(XYZ.XY0, mem);

        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        std::fill(mem.MES.data, mem.MES.data+f, 1);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dnode( const TetrasT& XYZ, int node_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data){
        auto f = XYZ.fusion;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Nfa() * applyOpV.Nfa() * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Nfa() * applyOpV.Nfa() * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpV.Nfa(); A.nCol = applyOpU.Nfa() * f;
        constexpr bool copyQuadVals = false;
        constexpr bool dotWithNormal = false;
        
        int q = 1;
        double qp[4] = {0, 0, 0, 0};
        double w[1] = {1};
        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3D_init_animem_from_plain_memory(applyOpU, applyOpV, func, q, f, mem, dotWithNormal, copyQuadVals);
        amem.XYL.Init(qp, 4); amem.WG.Init(w, 1);
        amem.XYL.data[node_num] = 1;
        internalFem3DtetGeomInit(XYZ, amem);
        std::fill(amem.MES.data, amem.MES.data+f, 1);
        internalFem3Dtet(applyOpU, applyOpV, func, A, amem, user_data, dotWithNormal);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3DpntL( const TetrasT& XYZ, ArrayView<> XYL, ArrayView<> WG,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data ){
        auto f = XYZ.fusion;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Nfa() * applyOpV.Nfa() * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Nfa() * applyOpV.Nfa() * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpV.Nfa(); A.nCol = applyOpU.Nfa() * f;
        constexpr bool copyQuadVals = false;
        constexpr bool dotWithNormal = false;

        int q = std::min(XYL.size/4, WG.size);
        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3D_init_animem_from_plain_memory(applyOpU, applyOpV, func, q, f, mem, dotWithNormal, copyQuadVals);
        amem.XYL.Init(XYL.data, 4*q);
        amem.WG.Init(WG.data, q);
        internalFem3DtetGeomInit(XYZ, amem);
        std::fill(amem.MES.data, amem.MES.data+f, 1);
        internalFem3Dtet(applyOpU, applyOpV, func, A, amem, user_data, dotWithNormal);
    }
    template<typename FuncTraits, typename Functor, typename TetraScalarTp>
    void fem3DpntX( const Tetra<TetraScalarTp>& XYZ, ArrayView<const double> X, ArrayView<> WG,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data ){
        int f = 1;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Nfa() * applyOpV.Nfa() * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Nfa() * applyOpV.Nfa() * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpV.Nfa(); A.nCol = applyOpU.Nfa() * f;
        constexpr bool copyQuadVals = true;
        constexpr bool dotWithNormal = false;

        int q = std::min(X.size/3, WG.size);
        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3D_init_animem_from_plain_memory(applyOpU, applyOpV, func, q, f, mem, dotWithNormal, copyQuadVals);
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, amem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, amem.XYL, amem.PSI);
        amem.WG.Init(WG.data, q);
        internalFem3DtetGeomInit_XYG(XYZ.XY0, amem);
        std::fill(amem.MES.data, amem.MES.data+f, 1);
        internalFem3Dtet(applyOpU, applyOpV, func, A, amem, user_data, dotWithNormal);
    }
};