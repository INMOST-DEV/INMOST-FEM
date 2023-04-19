namespace Ani{
    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3Dtet(  const TetrasT& XYZ, 
                    const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, int order, void *user_data){
        auto f = XYZ.fusion;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Nfa() * applyOpV.Nfa() * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Nfa() * applyOpV.Nfa() * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpV.Nfa(); A.nCol = applyOpU.Nfa() * f;
        constexpr bool dotWithNormal = false;
        constexpr bool copyQuadVals = !std::is_same<double, qReal>::value;

        auto formula = tetrahedron_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3D_init_animem_from_plain_memory(applyOpU, applyOpV, func, q, f, mem, dotWithNormal, copyQuadVals);
        if (!copyQuadVals) {
            amem.XYL.Init(const_cast<qReal*>(formula.p), 4 * q);
            amem.WG.Init(const_cast<qReal*>(formula.w), q);
        } else {
            std::copy(formula.p, formula.p + 4*q, amem.XYL.data);
            std::copy(formula.w, formula.w + q, amem.WG.data);
        }
        internalFem3DtetGeomInit(XYZ, amem);
        internalFem3Dtet(applyOpU, applyOpV, func, A, amem, user_data, dotWithNormal);
    }

    template<typename OpA, typename OpB, typename FuncTraits, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3Dtet(     const TetrasT& XYZ,
                       const Functor &Dfnc, DenseMatrix<ScalarType> &A,
                       PlainMemory<ScalarType, IndexType> plainMemory, int order, void *user_data) {
        auto f = XYZ.fusion;
        assert(f > 0 && "Coordinate arrays shouldn't be free");
        assert(A.size >= static_cast<std::size_t>(OpB::Nfa::value * OpA::Nfa::value * f) && "Not enough memory to save A matrix");
        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        auto formula = tetrahedron_quadrature_formulas(order);
        auto q = formula.GetNumPoints();
#ifndef	NDEBUG
        auto t = fem3Dtet_memory_requirements<OpA, OpB, ScalarType, IndexType>(order, f);
        assert(plainMemory.ddata && plainMemory.dSize >= t.dSize && "Not enough of plain memory");
        assert((t.iSize <= 0 || (plainMemory.idata && plainMemory.iSize >= t.iSize)) && "Not enough of plain memory");
#endif
        AniMemory<ScalarType, IndexType> mem = 
            fem3Dtet_init_animem_from_plain_memory<OpA, OpB, ScalarType, IndexType, !std::is_same<ScalarType, qReal>::value, false>(q, f, plainMemory);
        if (std::is_same<ScalarType, qReal>::value) {
            mem.XYL.Init(const_cast<qReal*>(formula.p), 4 * q);
            mem.WG.Init(const_cast<qReal*>(formula.w), q);
        } else {
            std::copy(formula.p, formula.p + 4*q, mem.XYL.data);
            std::copy(formula.w, formula.w + q, mem.WG.data);
        }
        
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        internalFem3DtetGeomInit(XYZ, mem);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

#ifdef WITH_EIGEN
    template<typename OpA, typename OpB, typename FuncTraits, typename Derived>
    void fem3Dtet(const Eigen::Vector3d &XY0, const Eigen::Vector3d &XY1, const Eigen::Vector3d &XY2,
                       const Eigen::Vector3d &XY3,
                       const std::function<TensorType(const std::array<double, 3> &, double *, std::pair<uint, uint>,
                                                      void *, int)> &Dfnc, const Eigen::MatrixBase<Derived> &A,
                       int order, void *user_data) {
        MemoryLegacy<> ml;
        using IndexType = int;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.rows() != OpB::Nfa::value || A.cols() != static_cast<long int>(OpA::Nfa::value*mem.f) )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A is " +  std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + " matrix");
        if (A.rows() > 30 || A.cols() > 30 || order > 6)
            throw std::runtime_error("Too big matrices A and too big quadrature orders is not supported for stack memory version of the function");
        DFunc<decltype(Dfnc), FuncTraits, double, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        double Amem[40*40];
        DenseMatrix<double> Am{Amem, static_cast<std::size_t>(A.rows()), static_cast<std::size_t>(A.cols()), 40*40};
        auto formula = tetrahedron_quadrature_formulas(order);
        IndexType f = XY0.cols();
        IndexType q = formula.GetNumPoints();
        mem.WG.Init(const_cast<double*>(formula.w), q);
        mem.XYL.Init(const_cast<double*>(formula.p), 4*q);
        mem.f = f;
        mem.q = q;
        internalFem3DtetGeomInit(XY0, XY1, XY2, XY3, mem);
        internalFem3Dtet<OpA, OpB, false>(func, Am, mem, user_data);
        Eigen::Map<Eigen::MatrixXd, sizeof(double)> Amap(Amem, Am.nRow, Am.nCol);
        auto& A_= const_cast< Eigen::MatrixBase<Derived>& >(A);
        A_ = Amap;
    }
#endif

    template<typename OpA, typename OpB, typename FuncTraits, int FUSION, typename ScalarType, typename IndexType, typename Functor, typename TetrasT>
    void fem3Dtet(const TetrasT& XYZ,
                  const Functor &Dfnc, DenseMatrix <ScalarType> &A, int order, void *user_data) {
        MemoryLegacy<ScalarType, FUSION> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        if (A.size < OpB::Nfa::value*OpA::Nfa::value*mem.f )
            throw std::runtime_error("Expected dimensions of A is " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f) +
                                     ", but A has size = " + std::to_string(A.size) + " < " + std::to_string(OpB::Nfa::value) + "x" + std::to_string(OpA::Nfa::value*mem.f));
        if (order > 6)
            throw std::runtime_error("Quadratures more than 6-th order are not supported for stack memory version of the function");
        IndexType f = XYZ.fusion;
        if (f > FUSION)
            throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));
        assert(f > 0 && "Coordinate arrays shouldn't be free");

        A.nRow = OpB::Nfa::value; A.nCol = OpA::Nfa::value * f;
        DFunc<decltype(Dfnc), FuncTraits, ScalarType, IndexType, OpA::Dim::value * OpB::Dim::value> func(Dfnc);
        auto formula = tetrahedron_quadrature_formulas(order);
        IndexType q = formula.GetNumPoints();
        if (std::is_same<ScalarType, qReal>::value) {
            mem.XYL.Init(const_cast<qReal*>(formula.p), 4 * q);
            mem.WG.Init(const_cast<qReal*>(formula.w), q);
        } else {
            std::copy(formula.p, formula.p + 4*q, mem.XYL.data);
            std::copy(formula.w, formula.w + q, mem.WG.data);
        }
        mem.f = f;
        mem.q = q;
        internalFem3DtetGeomInit(XYZ, mem);
        internalFem3Dtet<OpA, OpB>(func, A, mem, user_data);
    }

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3Dtet_memory_requirements(int order, int fusion){
        return fem3D_memory_requirements<OpA, OpB, ScalarType, IndexType, false, false>(order, fusion);
    }
};