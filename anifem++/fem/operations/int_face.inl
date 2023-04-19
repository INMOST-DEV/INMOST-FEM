namespace Ani{
    template<typename OpA, typename OpB, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3Dface_memory_requirements(int order, int fusion){
        return fem3D_memory_requirements<OpA, OpB, ScalarType, IndexType, true, false>(order, fusion);
    }

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3DfaceN_memory_requirements(int order, int fusion){
        return fem3D_memory_requirements<OpA, OpB, ScalarType, IndexType, true, true>(order, fusion);
    }

    template<typename OpA, typename OpB, typename FuncTraits, int FUSION, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3Dface(const TetrasT& XYZ,
              int face_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A, int order, void *user_data) {
        MemoryLegacy<ScalarType, FUSION> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        if (order > 9)
            throw std::runtime_error("Quadratures more than 9-th order are not supported for stack memory version of the function");
        IndexType f = XYZ.fusion;
        if (f > FUSION)
            throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(face_num < 4 && face_num >= 0 && "Wrong face index");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        auto formula = triangle_quadrature_formulas(order);
        IndexType q = formula.GetNumPoints();
        for (int n = 0; n < q; ++n) {
            mem.XYL.data[4 * n + face_num] = formula.p[3 * n + 0];
            mem.XYL.data[4 * n + (face_num + 1) % 4] = formula.p[3 * n + 1];
            mem.XYL.data[4 * n + (face_num + 2) % 4] = formula.p[3 * n + 2];
            mem.XYL.data[4 * n + (face_num + 3) % 4] = 0;
        }
        std::copy(formula.w, formula.w + q, mem.WG.data);
        mem.f = f;
        mem.q = q;
        internalFem3DtetGeomInit(XYZ, mem);
        for (int r = 0; r < f; ++r) {
            auto x = mem.XYP.data + 12 * r;
            mem.MES.data[r] = tri_area(x + 3 * face_num, x + 3 * ((face_num + 1) % 4),x + 3 * ((face_num + 2) % 4));
        }
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, int FUSION, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3DfaceN(const TetrasT& XYZ,
              int face_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A, int order, void *user_data) {
        MemoryLegacy<ScalarType, FUSION> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        if (order > 9)
            throw std::runtime_error("Quadratures more than 9-th order are not supported for stack memory version of the function");
        IndexType f = XYZ.fusion;
        if (f > FUSION)
            throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(face_num < 4 && face_num >= 0 && "Wrong face index");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value * DotWithNormalHelper<true>::nD::value> func(Dfnc);
        auto formula = triangle_quadrature_formulas(order);
        IndexType q = formula.GetNumPoints();
        for (int n = 0; n < q; ++n) {
            mem.XYL.data[4 * n + face_num] = formula.p[3 * n + 0];
            mem.XYL.data[4 * n + (face_num + 1) % 4] = formula.p[3 * n + 1];
            mem.XYL.data[4 * n + (face_num + 2) % 4] = formula.p[3 * n + 2];
            mem.XYL.data[4 * n + (face_num + 3) % 4] = 0;
        }
        std::copy(formula.w, formula.w + q, mem.WG.data);
        mem.f = f;
        mem.q = q;
        internalFem3DtetGeomInit(XYZ, mem);
        for (int r = 0; r < f; ++r) {
            auto x = mem.XYP.data + 12 * r;
            mem.MES.data[r] = face_normal(x + 3 * ((face_num + 0) % 4), x + 3 * ((face_num + 1) % 4),
                                          x + 3 * ((face_num + 2) % 4), x + 3 * ((face_num + 3) % 4),
                                          mem.NRM.data+3*r);
        }
        internalFem3Dtet<OpA, OpB, true>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3Dface(const TetrasT& XYZ,
              int face_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A,
              PlainMemory <ScalarType, IndexType> plainMemory, int order, void *user_data) {
        auto f = XYZ.fusion;
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        auto formula = triangle_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
#ifndef	NDEBUG
        auto t = fem3Dface_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, true, false>(q, f, plainMemory);
        for (int n = 0; n < q; ++n) {
            mem.XYL.data[4 * n + face_num] = formula.p[3 * n + 0];
            mem.XYL.data[4 * n + (face_num + 1) % 4] = formula.p[3 * n + 1];
            mem.XYL.data[4 * n + (face_num + 2) % 4] = formula.p[3 * n + 2];
            mem.XYL.data[4 * n + (face_num + 3) % 4] = 0;
        }
        std::copy(formula.w, formula.w + q, mem.WG.data);

        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        internalFem3DtetGeomInit(XYZ, mem);
        for (int r = 0; r < f; ++r) {
            auto x = mem.XYP.data + 12 * r;
            mem.MES.data[r] = tri_area(x + 3 * face_num, x + 3 * ((face_num + 1) % 4),x + 3 * ((face_num + 2) % 4));
        }
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void
    fem3DfaceN(const TetrasT& XYZ,
              int face_num, const Functor &Dfnc, DenseMatrix <ScalarType> &A,
              PlainMemory <ScalarType, IndexType> plainMemory, int order, void *user_data) {
        auto f = XYZ.fusion;
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        auto formula = triangle_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
#ifndef	NDEBUG
        auto t = fem3DfaceN_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, true, true>(q, f, plainMemory);
        for (int n = 0; n < q; ++n) {
            mem.XYL.data[4 * n + face_num] = formula.p[3 * n + 0];
            mem.XYL.data[4 * n + (face_num + 1) % 4] = formula.p[3 * n + 1];
            mem.XYL.data[4 * n + (face_num + 2) % 4] = formula.p[3 * n + 2];
            mem.XYL.data[4 * n + (face_num + 3) % 4] = 0;
        }
        std::copy(formula.w, formula.w + q, mem.WG.data);
        
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value* DotWithNormalHelper<true>::nD::value> func(Dfnc);
        internalFem3DtetGeomInit(XYZ, mem);
        for (int r = 0; r < f; ++r) {
            auto x = mem.XYP.data + 12 * r;
            mem.MES.data[r] = face_normal(x + 3 * face_num, x + 3 * ((face_num + 1) % 4),
                                          x + 3 * ((face_num + 2) % 4), x + 3 * ((face_num + 3) % 4),
                                          mem.NRM.data+3*r);
        }
        internalFem3Dtet<OpA, OpB, true>(func, A, mem, user_data);
    }

    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dface( const TetrasT& XYZ, int face_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order, void *user_data, bool dotWithNormal){
        auto f = XYZ.fusion;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Nfa() * applyOpV.Nfa() * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Nfa() * applyOpV.Nfa() * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpV.Nfa(); A.nCol = applyOpU.Nfa() * f;
        constexpr bool copyQuadVals = true;

        auto formula = triangle_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3D_init_animem_from_plain_memory(applyOpU, applyOpV, func, q, f, mem, dotWithNormal, copyQuadVals);
        for (int n = 0; n < q; ++n) {
            amem.XYL.data[4 * n + face_num] = formula.p[3 * n + 0];
            amem.XYL.data[4 * n + (face_num + 1) % 4] = formula.p[3 * n + 1];
            amem.XYL.data[4 * n + (face_num + 2) % 4] = formula.p[3 * n + 2];
            amem.XYL.data[4 * n + (face_num + 3) % 4] = 0;
        }
        std::copy(formula.w, formula.w + q, amem.WG.data);
        internalFem3DtetGeomInit(XYZ, amem);
        if (!dotWithNormal)
            for (int r = 0; r < f; ++r) {
                auto x = amem.XYP.data + 12 * r;
                amem.MES.data[r] = tri_area(x + 3 * face_num, x + 3 * ((face_num + 1) % 4),x + 3 * ((face_num + 2) % 4));
            }
        else
            for (int r = 0; r < f; ++r) {
                auto x = amem.XYP.data + 12 * r;
                amem.MES.data[r] = face_normal(x + 3 * face_num, x + 3 * ((face_num + 1) % 4),
                                            x + 3 * ((face_num + 2) % 4), x + 3 * ((face_num + 3) % 4),
                                            amem.NRM.data+3*r);
            }
        internalFem3Dtet(applyOpU, applyOpV, func, A, amem, user_data, dotWithNormal);
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dface( const TetrasT& XYZ, int face_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order, void *user_data) { 
        fem3Dface<FuncTraits>(XYZ, face_num, applyOpU, applyOpV, Dfnc, A, mem, order, user_data, false); 
    }
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3DfaceN(const TetrasT& XYZ, int face_num,
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order, void *user_data){
        fem3Dface<FuncTraits>(XYZ, face_num, applyOpU, applyOpV, Dfnc, A, mem, order, user_data, true);
    }

};