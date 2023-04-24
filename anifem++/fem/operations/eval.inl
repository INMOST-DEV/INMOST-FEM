namespace Ani{
    template<typename Op, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3DapplyL_memory_requirements(int pnt_per_tetra, int fusion){
        auto f = fusion, q = pnt_per_tetra;
        PlainMemory<ScalarType, IndexType> pm;
        std::size_t Usz = 0, extraR = 0, extraI = 0;
        Op::template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        extraR = std::max(extraR, static_cast<std::size_t>(1*Op::Nfa::value*f));
        IndexType DU = 0, DIFF = 0;
        IndexType XYG = 3*q*f, XYP = 3*4*f, PSI = 3*3*f, DET = f, MES = f, NRM = 0, XYL = 0, WG = 0;
        pm.dSize = Usz + DU + DIFF + XYG + XYP + PSI + DET + MES + NRM + extraR + XYL + WG;
        pm.iSize = extraI;
        return pm;
    };

    template<typename Op, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> fem3DapplyX_memory_requirements(int pnt_per_tetra, int fusion){
        auto pm = fem3DapplyL_memory_requirements<Op, ScalarType, IndexType>(pnt_per_tetra, fusion);
        IndexType XYL = 4*pnt_per_tetra;
        pm.dSize += XYL;
        return pm;
    }

    template<typename ScalarType = double>
    void inline __internal_check_fem3DapplyL(
                  const DenseMatrix<const ScalarType>& XY0, const DenseMatrix<const ScalarType>& XY1,
                  const DenseMatrix<const ScalarType>& XY2, const DenseMatrix<const ScalarType>& XY3,
                  int f, int q, int nfa, int dim,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU){
        if (dofs.nRow != static_cast<std::size_t>(nfa) || dofs.nCol != static_cast<std::size_t>(f))
            throw std::runtime_error("Expected dimension of dofs is " + std::to_string(nfa) + "x" + std::to_string(f));
        if (f <= 0) throw std::runtime_error("Coordinate arrays shouldn't be free");
        if (!(XY0.nRow == 3 && XY1.nRow == 3 && XY2.nRow == 3 && XY3.nRow == 3)) 
           throw std::runtime_error("Wrong coordinate arrays dimension, expected XYi.nRow = 3");
        if (!(XY0.nCol == static_cast<std::size_t>(f) && XY1.nCol == static_cast<std::size_t>(f) && XY2.nCol == static_cast<std::size_t>(f) && XY3.nCol == static_cast<std::size_t>(f)))
           throw std::runtime_error("Coordinate arrays doesn't consensual");
        if (opU.size < static_cast<std::size_t>(dim * q * f))
            throw std::runtime_error("opU.size = " + std::to_string(opU.size) + " but required at least " + std::to_string(dim * q * f));
    }

    template<int FUSION, int MAXPNTNUM, typename ScalarType>
    void inline __internal_check_fem3DapplyL_static(
                const DenseMatrix<const ScalarType>& XY0, const DenseMatrix<const ScalarType>& XY1,
                const DenseMatrix<const ScalarType>& XY2, const DenseMatrix<const ScalarType>& XY3,
                int f, int q, int nfa, int dim,
                const DenseMatrix<ScalarType>& dofs, 
                DenseMatrix<ScalarType>& opU){
        __internal_check_fem3DapplyL(XY0, XY1, XY2, XY3, f, q, nfa, dim, dofs, opU);            
        if (f > FUSION)
            throw std::runtime_error("This version of the function not supported fusion that more than " + std::to_string(FUSION));    
        if (q > MAXPNTNUM*4)
           throw std::runtime_error("This version of the function not supported more than " + std::to_string(MAXPNTNUM) + " points");
    }

    template<typename Op, typename ScalarType = double, typename IndexType = int>
    inline PlainMemory<ScalarType, IndexType> __internal_fem3DapplyL_setmem(AniMemory<ScalarType, IndexType>& mem, int q, int f, int nfa, int dim, PlainMemory<ScalarType, IndexType> plainMemory){
        (void) dim;
        std::size_t Usz = 0, extraR = 0, extraI = 0;
        Op::template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        extraR = std::max(extraR, static_cast<std::size_t>(1*nfa*f));
        //IndexType DU_sz = 0, DIFF_sz = 0;
        IndexType XYG_sz = 3*q*f, XYP_sz = 3*4*f, PSI_sz = 3*3*f, DET_sz = f, MES_sz = f;// NRM_sz = 0, XYL_sz = 0, WG_sz = 0;
        ScalarType* p = plainMemory.ddata;
        mem.XYG.Init(p, XYG_sz); p += XYG_sz;
        mem.XYP.Init(p, XYP_sz); p += XYP_sz;
        mem.PSI.Init(p, PSI_sz); p += PSI_sz;
        mem.MES.Init(p, MES_sz); p += MES_sz;
        mem.NRM.data = nullptr;
        mem.DET.Init(p, DET_sz); p += DET_sz;
        mem.U.Init(p, Usz); p += Usz;
        mem.extraR.Init(p, extraR); p += extraR;
        if (extraI > 0) {
            mem.extraI.Init(plainMemory.idata, plainMemory.iSize);
            plainMemory.idata += extraI; plainMemory.iSize -= extraI;
        }
        plainMemory.dSize = plainMemory.ddata + plainMemory.dSize - p;
        plainMemory.ddata = p;

        return plainMemory;
    }

    template<typename Op, int FUSION, int MAXPNTNUM, typename ScalarType, typename IndexType, typename TetrasT>
    void fem3DapplyL(
                  const TetrasT& XYZ,
                        ArrayView<ScalarType> XYL,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU){
        constexpr auto nfa = Op::Nfa::value, dim = Op::Dim::value;
        auto f = XYZ.fusion, q = XYL.size / 4;
        __internal_check_fem3DapplyL_static<FUSION, MAXPNTNUM, ScalarType>(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, f, q, nfa, dim, dofs, opU);
        opU.nRow = dim * q, opU.nCol = f;
        MemoryLegacy<ScalarType, FUSION, MAXPNTNUM> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        
        mem.XYL = XYL; mem.f = f; mem.q = q;
        internalFem3DtetGeomInit(XYZ, mem);
        internalFem3DApply<Op, ScalarType, IndexType>(dofs, opU, mem);
    } 

    template<typename Op, int MAXPNTNUM, typename ScalarType, typename IndexType, typename TetraScalarTp>
    void fem3DapplyX(
                  const Tetra<TetraScalarTp>& XYZ,
                  const ArrayView<const ScalarType> X,
                  const ArrayView<ScalarType>& _dofs, 
                  ArrayView<ScalarType> _opU){
        DenseMatrix<ScalarType> dofs(_dofs.data, _dofs.size, 1), opU(_opU.data, _opU.size, 1);
        ScalarType XYLp[4*MAXPNTNUM];
        ArrayView<ScalarType> XYL(XYLp, 4*X.size/3);
        
        constexpr auto nfa = Op::Nfa::value, dim = Op::Dim::value;
        constexpr int f = 1;
        auto q = X.size/3;
        __internal_check_fem3DapplyL_static<1, MAXPNTNUM, ScalarType>(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, f, q, nfa, dim, dofs, opU);
        opU.nRow = dim * q, opU.nCol = f;
        MemoryLegacy<ScalarType, 1, MAXPNTNUM> ml;
        auto mem = ml.template getAniMemory<IndexType>();
        mem.f = f; mem.q = q;
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, mem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, XYL, mem.PSI);
        mem.XYL = XYL;
        internalFem3DtetGeomInit_XYG(XYZ.XY0, mem);
        internalFem3DApply<Op, ScalarType, IndexType>(dofs, opU, mem);
    }

    template<typename Op, typename ScalarType, typename IndexType, typename TetraScalarTp>
    void fem3DapplyX(
                  const Tetra<TetraScalarTp>& XYZ,
                  const ArrayView<const ScalarType> X,
                  const ArrayView<ScalarType>& _dofs, 
                  ArrayView<ScalarType> _opU,
                  PlainMemory<ScalarType, IndexType> plainMemory){
        DenseMatrix<ScalarType> dofs(_dofs.data, _dofs.size, 1), opU(_opU.data, _opU.size, 1);
        ScalarType* XYLp = plainMemory.ddata;
        plainMemory.ddata += 4 * (X.size/3), plainMemory.dSize -= 4 * (X.size/3);
        ArrayView<ScalarType> XYL(XYLp, 4*X.size/3);

        constexpr auto nfa = Op::Nfa::value, dim = Op::Dim::value;
        constexpr int f = 1;
        auto q = X.size/3;
        __internal_check_fem3DapplyL<ScalarType>(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, f, q, nfa, dim, dofs, opU);
        opU.nRow = dim * q, opU.nCol = f;
        AniMemory<ScalarType, IndexType> mem;
        __internal_fem3DapplyL_setmem<Op, ScalarType, IndexType>(mem, q, f, nfa, dim, plainMemory);
        mem.q = q; mem.f = f;
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, mem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, XYL, mem.PSI);
        mem.XYL = XYL;
        internalFem3DtetGeomInit_XYG(XYZ.XY0, mem);
        internalFem3DApply<Op, ScalarType, IndexType>(dofs, opU, mem);
    }

    template<typename Op, typename ScalarType, typename IndexType, typename TetrasT>
    void fem3DapplyL(
                  const TetrasT& XYZ,
                        ArrayView<ScalarType> XYL,
                  const DenseMatrix<ScalarType>& dofs, 
                  DenseMatrix<ScalarType>& opU,
                  PlainMemory<ScalarType, IndexType> plainMemory){            
        constexpr auto nfa = Op::Nfa::value, dim = Op::Dim::value;
        auto f = XYZ.fusion, q = XYL.size / 4;
        __internal_check_fem3DapplyL<ScalarType>(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, f, q, nfa, dim, dofs, opU);
        opU.nRow = dim * q, opU.nCol = f;
        AniMemory<ScalarType, IndexType> mem;

        __internal_fem3DapplyL_setmem<Op, ScalarType, IndexType>(mem, q, f, nfa, dim, plainMemory);
        mem.q = q; mem.f = f;
        mem.XYL = XYL;

        internalFem3DtetGeomInit(XYZ, mem);
        internalFem3DApply<Op, ScalarType, IndexType>(dofs, opU, mem);
    }

    template<typename FuncTraits, typename Functor, typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, uint Ddim1, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data){
        auto f = XYZ.fusion;
        int q = XYL.size/4, nD = 1;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(Ddim1 * q * f / nD))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(Ddim1 * q * f / nD) + " but A has size = " + std::to_string(A.size));
        A.nRow = Ddim1*q/nD; A.nCol = f;
        // constexpr bool dotWithNormal = false;

        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3DapplyD_init_animem_from_plain_memory(applyOpU, Ddim1, func, q, f, mem, false);
        amem.XYL.Init(XYL.data, 4*q);
        std::fill(amem.WG.data, amem.WG.data+q, 1);
        internalFem3DtetGeomInit(XYZ, amem);
        std::fill(amem.MES.data, amem.MES.data+f, 1);
        internalFem3DApply(applyOpU, DenseMatrix<const double>(dofs.data, dofs.nRow, dofs.nCol), func, {Ddim1, applyOpU.Dim()}, A, amem, user_data, false);
    }
    template<typename FuncTraits>
    PlainMemoryX<> fem3DapplyL_memory_requirements(uint Ddim1, const ApplyOpBase& applyOpU, uint pnt_per_tetra, uint fusion){
        return fem3DapplyD_memory_requirements_base<FuncTraits>(applyOpU, Ddim1, pnt_per_tetra, fusion, false);
    }
    template<typename TetrasT>
    void fem3DapplyL(const TetrasT& XYZ, ArrayView<> XYL, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& A, PlainMemoryX<> mem){
        auto f = XYZ.fusion;
        int q = XYL.size/4;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Dim() * q * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Dim() * q * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpU.Dim()*q; A.nCol = f;

        auto amem = fem3DapplyND_init_animem_from_plain_memory(applyOpU, q, f, mem, false);
        amem.XYL.Init(XYL.data, 4*q);
        internalFem3DtetGeomInit(XYZ, amem);
        internalFem3DApply(applyOpU, DenseMatrix<const double>(dofs.data, dofs.nRow, dofs.nCol), A, amem);
    }
    template<typename FuncTraits, typename Functor, typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, uint Ddim1, const Functor& Dfnc, DenseMatrix<>& A,
                    PlainMemoryX<> mem, void *user_data){
        auto f = XYZ.fusion;
        int q = X.size/3, nD = 1;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(Ddim1 * q * f / nD))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(Ddim1 * q * f / nD) + " but A has size = " + std::to_string(A.size));
        A.nRow = Ddim1*q/nD; A.nCol = f;
        // constexpr bool dotWithNormal = false;

        DFunc<decltype(Dfnc), FuncTraits, double, int, -1> func(Dfnc);
        auto amem = fem3DapplyD_init_animem_from_plain_memory(applyOpU, Ddim1, func, q, f, mem, false);
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, amem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, amem.XYL, amem.PSI);
        std::fill(amem.WG.data, amem.WG.data+q, 1);
        internalFem3DtetGeomInit_XYG(XYZ.XY0, amem);
        std::fill(amem.MES.data, amem.MES.data+f, 1);
        internalFem3DApply(applyOpU, DenseMatrix<const double>(dofs.data, dofs.nRow, dofs.nCol), func, {Ddim1, applyOpU.Dim()}, A, amem, user_data, false);
    }
    template<typename FuncTraits>
    PlainMemoryX<> fem3DapplyX_memory_requirements(uint Ddim1, const ApplyOpBase& applyOpU, uint pnt_per_tetra, uint fusion){
        return fem3DapplyD_memory_requirements_base<FuncTraits>(applyOpU, Ddim1, pnt_per_tetra, fusion, false);
    }
    template<typename TetraScalarTp>
    void fem3DapplyX(const Tetra<TetraScalarTp>& XYZ, const ArrayView<const double> X, const DenseMatrix<>& dofs,
                    const ApplyOpBase& applyOpU, DenseMatrix<>& A, PlainMemoryX<> mem){
        auto f = XYZ.fusion;
        int q = X.size/3;
        if (f <= 0) return;
        if (A.size < static_cast<decltype(A.size)>(applyOpU.Dim() * q * f))
            throw std::runtime_error("Not enough memory for local matrix, expected size = " + std::to_string(applyOpU.Dim() * q * f) + " but A has size = " + std::to_string(A.size));
        A.nRow = applyOpU.Dim()*q; A.nCol = f;

        auto amem = fem3DapplyND_init_animem_from_plain_memory(applyOpU, q, f, mem, true);
        internalFem3DtetGeomInit_Tet(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, amem);
        getBaryCoord(XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data, X, amem.XYL, amem.PSI);
        internalFem3DtetGeomInit_XYG(XYZ.XY0, amem);
        internalFem3DtetGeomInit(XYZ, amem);
        internalFem3DApply(applyOpU, DenseMatrix<const double>(dofs.data, dofs.nRow, dofs.nCol), A, amem);
    }
};