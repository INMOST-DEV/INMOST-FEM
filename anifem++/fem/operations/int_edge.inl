namespace Ani{
    template<typename OpA, typename OpB, typename FuncTraits, int FUSION, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3Dedge(const TetrasT& XYZ,
              int edge_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A, int order, void *user_data) {
        MemoryLegacy<ScalarType, FUSION> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        if (order > 20)
            throw std::runtime_error("Quadratures more than 20-th order are not supported for stack memory version of the function");
        IndexType f = XYZ.fusion;
        if (f > FUSION)
            throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(edge_num < 6 && edge_num >= 0 && "Wrong edge index");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        auto formula = segment_quadrature_formulas(order);
        IndexType q = formula.GetNumPoints();
        std::fill(mem.XYL.data, mem.XYL.data + 4*q, 0);
        const char lookup1[6] = {0, 0, 0, 1, 1, 2}, lookup2[6] = {1, 2, 3, 2, 3, 3};
        for (int n = 0; n < q; ++n) {
            mem.XYL.data[4 * n + lookup1[edge_num]] = formula.p[2 * n + 0];
            mem.XYL.data[4 * n + lookup2[edge_num]] = formula.p[2 * n + 1];
        }
        std::copy(formula.w, formula.w + q, mem.WG.data);
        mem.f = f;
        mem.q = q;
        internalFem3DtetGeomInit(XYZ, mem);
        auto sq = [](auto x) { return x*x; };
        for (int r = 0; r < f; ++r) {
            auto x = mem.XYP.data + 12 * r;
            auto x1 = x + 3*lookup1[edge_num], x2 = x + 3*lookup2[edge_num];
            mem.MES.data[r] = sqrt(sq(x1[0]-x2[0]) + sq(x1[1]-x2[1]) + sq(x1[2]-x2[2]));
        }
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3Dedge(const TetrasT& XYZ,
              int edge_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A,
              PlainMemory <ScalarType, IndexType> plainMemory, int order, void *user_data) {
        auto f = XYZ.fusion;
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        auto formula = segment_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
#ifndef	NDEBUG
        auto t = fem3Dedge_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, true, false>(q, f, plainMemory);
        std::fill(mem.XYL.data, mem.XYL.data + 4*q, 0);
        const char lookup1[6] = {0, 0, 0, 1, 1, 2}, lookup2[6] = {1, 2, 3, 2, 3, 3};
        for (int n = 0; n < q; ++n) {
            mem.XYL.data[4 * n + lookup1[edge_num]] = formula.p[2 * n + 0];
            mem.XYL.data[4 * n + lookup2[edge_num]] = formula.p[2 * n + 1];
        }
        std::copy(formula.w, formula.w + q, mem.WG.data);

        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        internalFem3DtetGeomInit(XYZ, mem);
        auto sq = [](auto x) { return x*x; };
        for (int r = 0; r < f; ++r) {
            auto x = mem.XYP.data + 12 * r;
            auto x1 = x + 3*lookup1[edge_num], x2 = x + 3*lookup2[edge_num];
            mem.MES.data[r] = sqrt(sq(x1[0]-x2[0]) + sq(x1[1]-x2[1]) + sq(x1[2]-x2[2]));
        }
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dedge( const TetrasT& XYZ, int edge_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order, void *user_data){
        auto f = XYZ.fusion;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)> (applyOpU.Nfa() * applyOpV.Nfa() * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Nfa() * applyOpV.Nfa() * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpV.Nfa(); A.nCol = applyOpU.Nfa() * f;
        constexpr bool copyQuadVals = true;
        constexpr bool dotWithNormal = false;

        auto formula = segment_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3D_init_animem_from_plain_memory(applyOpU, applyOpV, func, q, f, mem, dotWithNormal, copyQuadVals);
        std::fill(amem.XYL.data, amem.XYL.data + 4*q, 0);
        const char lookup1[6] = {0, 0, 0, 1, 1, 2}, lookup2[6] = {1, 2, 3, 2, 3, 3};
        for (int n = 0; n < q; ++n) {
            amem.XYL.data[4 * n + lookup1[edge_num]] = formula.p[2 * n + 0];
            amem.XYL.data[4 * n + lookup2[edge_num]] = formula.p[2 * n + 1];
        }
        std::copy(formula.w, formula.w + q, amem.WG.data);
        internalFem3DtetGeomInit(XYZ, amem);
        auto sq = [](auto x) { return x*x; };
        for (int r = 0; r < f; ++r) {
            auto x = amem.XYP.data + 12 * r;
            auto x1 = x + 3*lookup1[edge_num], x2 = x + 3*lookup2[edge_num];
            amem.MES.data[r] = sqrt(sq(x1[0]-x2[0]) + sq(x1[1]-x2[1]) + sq(x1[2]-x2[2]));
        }
        internalFem3Dtet(applyOpU, applyOpV, func, A, amem, user_data, dotWithNormal);
    }

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3Dedge_memory_requirements(int order, int fusion){
        auto q = segment_quadrature_formulas(order).GetNumPoints();
        return fem3D_memory_requirements_base<OpA, OpB, ScalarType, IndexType, true, false>(q, fusion);
    }
};