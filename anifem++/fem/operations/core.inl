#ifdef OPTIMIZER_TIMERS_FEM3DTET
#include <chrono>
#endif

namespace Ani{
#ifdef OPTIMIZER_TIMERS_FEM3DTET
    static std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> Inter_st1, Inter_end1;
    static double T_init_F3DTet = 0, T_U_F3DTet = 0, T_V_F3DTet = 0, T_DU_F3DTet = 0, T_A_F3DTet = 0;
#endif
    
    ///Help to choose memory block of V expression depending on it's type
    ///@see internalFem3Dtet
    template<bool IsOperatorsSame = false>
    struct CompV {
        template<typename SecondOp, typename Amat, typename MemT>
        static inline auto computeV(Amat& A, MemT& mem){
            (void)A; return SecondOp::apply(mem, mem.V);
        }
    };
    template<>
    struct CompV<true> {
        template<typename SecondOp, typename Amat, typename MemT>
        static inline auto computeV(Amat& A, MemT& mem){
            (void) mem; return A;
        }
    };

    template<typename Scalar, typename IndexType0, typename IndexType1, typename IndexType2, typename IndexType3>
    inline void fusive_AT_mul_B(bool not_use_eigen,
                                IndexType0 fuse, IndexType1 nRow,
                                Scalar* A, IndexType2 ACol,
                                Scalar* B, IndexType3 BCol,
                                Scalar* R){
        (void) not_use_eigen;
#ifdef WITH_EIGEN
        if (not_use_eigen){
#endif
            for (IndexType0 r = 0; r < fuse; ++r)
                for (IndexType3 ia = 0; ia < BCol; ++ia)
                    for (IndexType2 ib = 0; ib < ACol; ++ib) {
                        Scalar s = 0;
                        for (IndexType1 j = 0; j < nRow; ++j)
                            s += B[j + nRow*(ia + BCol * r)] *
                                 A[j + nRow*(ib + ACol * r)];

                        R[ib + ACol * (ia + BCol * r)] = s;
                    }
#ifdef WITH_EIGEN
        }else{
            for (IndexType0 r = 0; r < fuse; ++r) {
                using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
                Eigen::Map<Mat, sizeof(Scalar)> eA(A + nRow*ACol* r, nRow, ACol);
                Eigen::Map<Mat, sizeof(Scalar)> eB(B + nRow * BCol * r, nRow, BCol);
                Eigen::Map<Mat, sizeof(Scalar)> eR(R + ACol * BCol * r, ACol, BCol);
                auto expr = eA.transpose() * eB;
                eR = expr;
            }
        }
#endif
    }

    template<typename Scalar, typename IndexType0, typename IndexType1, typename IndexType2, typename IndexType3>
    inline void fusive_A_mul_B(bool not_use_eigen,
                                IndexType0 fuse, IndexType1 ACol,
                                Scalar* A, IndexType2 ARow,
                                Scalar* B, IndexType3 BCol,
                                Scalar* R){
        (void) not_use_eigen;
        auto nRow = ACol;
#ifdef WITH_EIGEN
        if (not_use_eigen){
#endif
            std::fill(R, R + ARow * BCol, 0);
            for (IndexType0 r = 0; r < fuse; ++r)
                for (IndexType1 j = 0; j < nRow; ++j)
                    for (IndexType2 ib = 0; ib < ARow; ++ib) 
                        for (IndexType3 ia = 0; ia < BCol; ++ia)
                            R[ib + ARow * (ia + BCol * r)] += A[ib + ARow*(j +  nRow*r)] * B[j + nRow*(ia + BCol * r)];   
                  
#ifdef WITH_EIGEN
        }else{
            for (IndexType0 r = 0; r < fuse; ++r) {
                using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
                Eigen::Map<Mat, sizeof(Scalar)> eA(A + ARow*nRow*r, ARow, nRow);
                Eigen::Map<Mat, sizeof(Scalar)> eB(B + nRow * BCol * r, nRow, BCol);
                Eigen::Map<Mat, sizeof(Scalar)> eR(R + ARow * BCol * r, ARow, BCol);
                auto expr = eA * eB;
                eR = expr;
            }
        }
#endif
    }

    template<typename ScalarType>
    void getBaryCoord(
                  const ScalarType* XY0/*[3]*/, const ScalarType* XY1/*[3]*/,
                  const ScalarType* XY2/*[3]*/, const ScalarType* XY3/*[3]*/,
                  const ArrayView<const ScalarType> X, ArrayView<ScalarType> XYL){
        assert(X.size/3 <= XYL.size/4 && "XYL have not enough memory");
        ScalarType T[3*3], Tinv[3*3];
        for(int i = 0; i < 3; ++i) T[i + 3*0] = XY0[i] - XY3[i];
        for(int i = 0; i < 3; ++i) T[i + 3*1] = XY1[i] - XY3[i];
        for(int i = 0; i < 3; ++i) T[i + 3*2] = XY2[i] - XY3[i];
        inverse3x3(T, Tinv);
        int r = X.size/3;
        XYL.SetZero();
        for (int n = 0; n < r; ++n){
            ScalarType dr[3];
            for (int i = 0; i < 3; ++i) 
                dr[i] = X[i + n*3] - XY3[i];
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    XYL[4*n + i] += Tinv[i + 3*j]*dr[j];
            XYL[4*n + 3] = 1;
            for (int i = 0; i < 3; ++i) XYL[4*n + 3] -= XYL[4*n + i];
        }
    }

    template<typename ScalarType>
    void getBaryCoord(
                  const ScalarType* XY0/*[3]*/, const ScalarType* XY1/*[3]*/,
                  const ScalarType* XY2/*[3]*/, const ScalarType* XY3/*[3]*/,
                  const ArrayView<const ScalarType> X, 
                  ArrayView<ScalarType> XYL, ArrayView<ScalarType> PSI){
        (void) XY1; (void) XY2; (void) XY3;            
        XYL.SetZero();
        for (std::size_t n = 0; n < X.size/3; ++n){
            ScalarType dr[3];
            for (int i = 0; i < 3; ++i) 
                dr[i] = X[i + 3*n] - XY0[i];
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    XYL[4*n + i + 1] += PSI[i + 3*j]*dr[j];
            XYL[4*n + 0] = 1;
            for (int i = 0; i < 3; ++i) XYL[4*n + 0] -= XYL[4*n + i+1];    
        }
    }

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType, bool CopyQuad, bool DotWithNormal>
    AniMemory<ScalarType, IndexType> fem3Dtet_init_animem_from_plain_memory(int q, int f, PlainMemory<ScalarType, IndexType> plainMemory){
        AniMemory<ScalarType, IndexType> mem;
        IndexType XYG_sz = 3*q*f;
        IndexType XYP_sz = 3*4*f;
        IndexType PSI_sz = 3*3*f;
        IndexType DET_sz = f;
        IndexType MES_sz = f;

        ScalarType* p = plainMemory.ddata;
        mem.XYG.Init(p, XYG_sz); p += XYG_sz;
        mem.XYP.Init(p, XYP_sz); p += XYP_sz;
        mem.PSI.Init(p, PSI_sz); p += PSI_sz;
        mem.MES.Init(p, MES_sz); p += MES_sz;
        if (!DotWithNormal){
            mem.NRM.data = nullptr;
        } else {
             mem.NRM.Init(p, 3*f); p += 3*f;
        }
        mem.DET.Init(p, DET_sz); p += DET_sz;
        if (CopyQuad){
            mem.XYL.Init(p, 4 * q); p += 4*q;
            mem.WG.Init(p, q); p += q;
        }
        std::size_t Usz = 0;
        std::size_t Vsz = 0;
        std::size_t extraR = 0, extraI = 0;
        OpA::template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        if (!std::is_same<OpA, OpB>::value){
            std::size_t extraRV = 0, extraIV = 0;
            OpB::template memoryRequirements<ScalarType, IndexType>(f, q, Vsz, extraRV, extraIV);
            extraR = std::max(extraR, extraRV); extraI = std::max(extraI, extraIV);
        }
        auto nD = DotWithNormalHelper<DotWithNormal>::nD::value;
        IndexType DU_sz = nD * OpB::Dim::value * q * OpA::Nfa::value * f;
        IndexType DIFF_sz = OpB::Dim::value * std::max(nD*OpA::Dim::value, OpA::Nfa::value) * q * f;
        mem.U.Init(p, Usz); p += Usz;
        mem.V.Init((Vsz > 0) ? p : nullptr, Vsz); p += (Vsz > 0) ? Vsz : 0;
        mem.DU.Init(p, DU_sz); p += DU_sz;
        mem.DIFF.Init(p, DIFF_sz); p += DIFF_sz;
        mem.extraR.Init(p, extraR); p += extraR;
        if (extraI > 0) mem.extraI.Init(plainMemory.idata, plainMemory.iSize);
        mem.q = q;
        mem.f = f;
        return mem;
    }

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType, bool CopyQuadVals, bool DotWithNormal>
    PlainMemory <ScalarType, IndexType> fem3D_memory_requirements_base(int pnt_per_tetra, int fusion){
        auto f = fusion;
        auto q = pnt_per_tetra;
        PlainMemory<ScalarType, IndexType> pm;
        std::size_t Usz = 0;
        std::size_t Vsz = 0;
        std::size_t extraR = 0, extraI = 0;
        OpA::template memoryRequirements<ScalarType, IndexType>(f, q, Usz, extraR, extraI);
        if (!std::is_same<OpA, OpB>::value){
            std::size_t extraRV = 0, extraIV = 0;
            OpB::template memoryRequirements<ScalarType, IndexType>(f, q, Vsz, extraRV, extraIV);
            extraR = std::max(extraR, extraRV); extraI = std::max(extraI, extraIV);
        }
        extraR = std::max(extraR, static_cast<std::size_t>(OpB::Nfa::value*OpA::Nfa::value*f));
        auto nD = DotWithNormalHelper<DotWithNormal>::nD::value;
        IndexType DU = OpB::Dim::value * nD * q * OpA::Nfa::value * f;
        IndexType DIFF = OpB::Dim::value * std::max(OpA::Dim::value * nD, OpA::Nfa::value) * q * f;
        IndexType XYG = 3*q*f;
        IndexType XYP = 3*4*f;
        IndexType PSI = 3*3*f;
        IndexType DET = f;
        IndexType MES = f;
        IndexType NRM = DotWithNormal ? 3*f : 0;
        IndexType XYL = 0, WG = 0;
        if (CopyQuadVals) XYL = 4*q, WG = q;
        pm.dSize = Usz + Vsz + DU + DIFF + XYG + XYP + PSI + DET + MES + NRM + extraR + XYL + WG;
        pm.iSize = extraI;
        return pm;
    }

    template<typename MatrTpXY0, typename MatrTpXY1, typename MatrTpXY2, typename MatrTpXY3, typename Scalar, typename IndexType>
    void internalFem3DtetGeomInit_Tet(const MatrTpXY0 &XY0, const MatrTpXY1 &XY1, const MatrTpXY2 &XY2, const MatrTpXY3 &XY3, AniMemory <Scalar, IndexType> &mem) {
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        auto q = mem.q;
        auto f = mem.f;
        using std::abs;
        DenseMatrix<Scalar> XYP{mem.XYP.data, 3, 4*f, mem.XYP.size};
        DenseMatrix<Scalar> det{mem.DET.data, 1, f, mem.DET.size};
        DenseMatrix<Scalar> PSI{mem.PSI.data, 3, 3*f, mem.PSI.size};
        DenseMatrix<Scalar> XYG{mem.XYG.data, 3*q, f, mem.XYG.size};
        // constexpr auto dimA = OpA::Dim::value, dimB = OpB::Dim::value;
        // constexpr auto nfA = OpA::Nfa::value, nfB = OpB::Nfa::value;
        for (std::size_t r = 0; r < f; ++r){
            for (IndexType i = 0; i < 3; ++i)
                XYP.data[i + 12*r] = 0;
            for (IndexType i = 0; i < 3; ++i)
                XYP.data[i + 3 + 12*r] = XY1(i, r) - XY0(i, r);
            for (IndexType i = 0; i < 3; ++i)
                XYP.data[i + 6 + 12*r] = XY2(i, r) - XY0(i, r);
            for (IndexType i = 0; i < 3; ++i)
                XYP.data[i + 9 + 12*r] = XY3(i, r) - XY0(i, r);
            det.data[r] = inverse3x3(XYP.data + 3 + 12*r, PSI.data + 9*r);
            mem.MES.data[r] = abs(det.data[r]) / 6;
        }
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_init_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
#endif
    }

template<typename MatrTpXY0, typename Scalar, typename IndexType>
    void internalFem3DtetGeomInit_XYG(const MatrTpXY0 &XY0, AniMemory <Scalar, IndexType> &mem) {
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        auto q = mem.q;
        auto f = mem.f;
        for (std::size_t r = 0; r < f; ++r)
            for (std::size_t n = 0; n < q; ++n){
                for (int k = 0; k < 3; ++k){
                    Scalar s = XY0(k, r);
                    for (int l = 0; l < 3; ++l)
                        s += mem.XYL.data[l+1 + 4 * n]*mem.XYP.data[k + 3*(l+1) + 12*r];
                    mem.XYG.data[k + 3*(n + q * r)] = s;
                }
            }
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_init_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
#endif
    }

    template<typename MatrTpXY0, typename MatrTpXY1, typename MatrTpXY2, typename MatrTpXY3, typename Scalar, typename IndexType>
    void internalFem3DtetGeomInit(const MatrTpXY0 &XY0, const MatrTpXY1 &XY1, const MatrTpXY2 &XY2, const MatrTpXY3 &XY3, AniMemory <Scalar, IndexType> &mem) {
        internalFem3DtetGeomInit_Tet(XY0, XY1, XY2, XY3, mem);
        internalFem3DtetGeomInit_XYG(XY0, mem);
    }

    template<typename OpA, typename OpB, bool DotWithNormal, typename DFUNC, typename Scalar, typename IndexType>
    void internalFem3Dtet(DFUNC &Dfnc, DenseMatrix <Scalar> A, AniMemory <Scalar, IndexType> &mem, void *user_data) {
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        constexpr auto dimA = OpA::Dim::value, dimB = OpB::Dim::value;
        constexpr auto nfA = OpA::Nfa::value, nfB = OpB::Nfa::value;
        assert(A.nRow == nfB && A.nCol == nfA*mem.f && "A matrix has wrong dimensions");
        auto U = OpA::apply(mem, mem.U);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_U_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        auto V = CompV<std::is_same<OpA, OpB>::value>::template computeV<OpB>(U, mem);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_V_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        auto Um = convToBendMx(U, static_cast<IndexType>(dimA), static_cast<IndexType>(nfA));
        auto Vm = convToBendMx(V, static_cast<IndexType>(dimB), static_cast<IndexType>(nfB));
        auto nD = DotWithNormalHelper<DotWithNormal>::nD::value;
        DU_comp_in<DFUNC, Scalar, IndexType> DU_dat(mem, Dfnc, user_data, Um, dimA, dimB * nD, mem.q, mem.f, nfA, nfB);
        if (decltype(Um)::Nparts::value > 1) std::fill(mem.DU.data, mem.DU.data + mem.q*mem.f*dimB*nfA*nD, 0);
        DenseMatrix<Scalar> DU = internal_Dfunc<Scalar, IndexType>::applyDU(DU_dat);
        DotWithNormalHelper<DotWithNormal>::ProcessNormal(DU, mem);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_DU_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif

        constexpr auto nparts = decltype(Vm)::Nparts::value;
        //conditions dimB * mem.q <= 8 || nfA + nfB <= 16 was chosen empirically
        //it seems for this dimensions Eigen multiplication more effective than hand-writen cycle
        if (nparts == 1) {
            fusive_AT_mul_B(dimB * mem.q <= 8 || nfA + nfB <= 16, mem.f, mem.q*dimB, Vm.data[0].data, nfB, DU.data, nfA, A.data);
        } else {
            Scalar *stDr = mem.DIFF.data, *stAr = mem.extraR.data;
            DenseMatrix<Scalar> Dr[nparts];
            DenseMatrix<Scalar> Ar[nparts];
            for (std::size_t db = 0; db < nparts; ++db){
                auto ldim = Vm.stRow[db + 1] - Vm.stRow[db];
                auto nrow = mem.q * ldim;
                auto ncol = nfA*mem.f;
                auto sz = nrow*ncol;
//                DenseMatrix<Scalar, IndexType> lDr(stDr, nrow, ncol, sz);
                Dr[db].Init(stDr, nrow, ncol, sz);
                auto& lDr = Dr[db];
                stDr += sz;

                for (std::size_t j = 0; j < ncol; ++j)
                    for (std::size_t n = 0; n < mem.q; ++n)
                        for (IndexType i = Vm.stRow[db]; i < Vm.stRow[db + 1]; ++i)
                            lDr(i - Vm.stRow[db] + ldim * n, j) = mem.DU.data[i + dimB * (n + mem.q * j )];
                auto lnfb = Vm.stCol[db + 1] - Vm.stCol[db];
                Ar[db].Init(stAr,lnfb, nfA*mem.f, lnfb*nfA*mem.f);
                stAr += lnfb*nfA*mem.f;
                fusive_AT_mul_B(ldim * mem.q <= 8 || nfA + lnfb <= 16, mem.f, nrow, Vm.data[db].data, lnfb, lDr.data, nfA, Ar[db].data);
            }
            for (std::size_t j = 0; j < nfA*mem.f; ++j)
                for (std::size_t db = 0; db < nparts; ++db){
                    for (IndexType i = Vm.stCol[db]; i < Vm.stCol[db + 1]; ++i)
                        A.data[i + nfB * j] = Ar[db](i - Vm.stCol[db], j);
                }
            //actually below code is  ineffective
//                for (IndexType r = 0; r < mem.f; ++r)
//                for (IndexType ia = 0; ia < nfA; ++ia)
//                for (IndexType db = 0; db < nparts; ++db) {
//                    auto ibsz = Vm.stCol[db + 1] - Vm.stCol[db];
//                    for (IndexType ib = Vm.stCol[db]; ib < Vm.stCol[db + 1]; ++ib) {
//                        Scalar s = 0;
//                        for (IndexType n = 0; n < mem.q; ++n) {
//                            auto kbsz = Vm.stRow[db + 1] - Vm.stRow[db];
//                            for (IndexType kb = Vm.stRow[db]; kb < Vm.stRow[db + 1]; ++kb)
//                                s += DU.data[kb + dimB * (n + mem.q * (ia + r * nfA))] * Vm.data[db].data[
//                                        kb - Vm.stRow[db] +
//                                        kbsz * (n + mem.q * (ib - Vm.stCol[db] + ibsz * r))];
//                        }
//                        A.data[ib + nfB * (ia + nfA * r)] = s;
//                    }
//                }
        }

#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_A_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
    }

    template<typename Op, typename Scalar, typename IndexType>      
    void internalFem3DApply(const DenseMatrix<Scalar>& dofs, DenseMatrix<Scalar> opU, AniMemory<Scalar, IndexType>& mem){
        constexpr auto dim = Op::Dim::value;
        constexpr auto nf = Op::Nfa::value;
        assert(opU.nRow == dim*mem.q && opU.nCol == mem.f && "opU matrix has wrong dimensions");
        assert(dofs.nRow == nf && "Count of dofs doesn't correspond to FEM type");
        assert(dofs.nCol == mem.f && "Number of dofs doesn't correspond to fusive number of tetra");
        auto V = Op::apply(mem, mem.U);
        auto Vm = convToBendMx(V, static_cast<IndexType>(dim), static_cast<IndexType>(nf));
        constexpr auto nparts = decltype(Vm)::Nparts::value;
        //conditions dimB * mem.q <= 8 || nfA + nfB <= 16 was chosen empirically
        //it seems for this dimensions Eigen multiplication more effective than hand-writen cycle
        if (nparts == 1) {
            fusive_A_mul_B(opU.nRow <= 8 || nf <= 16, mem.f, nf, Vm.data[0].data, opU.nRow, dofs.data, 1, opU.data);
        } else {
            double* tmp_opU = (mem.q <= 1) ? opU.data : mem.extraR.data;
            for (std::size_t db = 0; db < nparts; ++db)
                for (std::size_t r = 0; r < mem.f; ++r){
                    auto VmRow = Vm.stRow[db+1] - Vm.stRow[db];
                    auto VmCol = (Vm.stCol[db+1] - Vm.stCol[db])/mem.f;
                    double* lVm = Vm.data[db].data + mem.q*VmRow * VmCol * r; 
                    const double* ldofs = dofs.data + r*dofs.nRow + Vm.stCol[db];
                    double* lopU = tmp_opU + r*dim*mem.q + Vm.stRow[db]*mem.q;
                    fusive_A_mul_B(VmCol<= 8, mem.f, VmCol, lVm, mem.q*VmRow, const_cast<double*>(ldofs), 1, lopU);  
                }
            if (mem.q > 1) {
                for (std::size_t n = 0; n < mem.q; ++n)
                for (std::size_t db = 0; db < nparts; ++db){
                    auto ldim = Vm.stRow[db+1] - Vm.stRow[db];
                    double* from = tmp_opU + Vm.stRow[db]*mem.q + ldim*n;
                    double* to = opU.data + dim*n + Vm.stRow[db];
                    std::copy(from, from + ldim, to);
                }
            }    
        }
    }

    template<typename DFUNC>
    void internalFem3Dtet(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV,
                        DFUNC& Dfnc, DenseMatrix<> A, AniMemoryX<>& mem, void *user_data, bool dotWithNormal){
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        const auto dimA = applyOpU.Dim(), dimB = applyOpV.Dim();
        const auto nfA = applyOpU.Nfa(), nfB = applyOpV.Nfa();
        assert(A.nRow == static_cast<decltype(A.nRow)>(nfB) && A.nCol == static_cast<decltype(A.nCol)>(nfA*mem.f) && "A matrix has wrong dimensions");
        auto U = applyOpU(mem, mem.U);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_U_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif 
        Ani::BandDenseMatrixX<> V = (applyOpU == applyOpV) ? U : applyOpV(mem, mem.V);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_V_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        uint nD = !dotWithNormal ? DotWithNormalHelper<false>::nD::value : DotWithNormalHelper<true>::nD::value;
        DU_comp_in<DFUNC, double, int> DU_dat(mem, Dfnc, user_data, U, dimA, dimB * nD, mem.q, mem.f, nfA, nfB);
        if (U.nparts > 1)
            std::fill(mem.DU.data, mem.DU.data + mem.q*mem.f*dimB*nfA*nD, 0);
        DenseMatrix<> DU = internal_Dfunc<double, int>::applyDU(DU_dat);   
        if (dotWithNormal) 
            DotWithNormalHelper<true>::ProcessNormal(DU, mem);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_DU_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        uint nparts = V.nparts;
        if (nparts == 1) {
            fusive_AT_mul_B(dimB * mem.q <= 8 || nfA + nfB <= 16, mem.f, mem.q*dimB, V.data[0].data, nfB, DU.data, nfA, A.data);
        } else {
            double *stDr = mem.DIFF.data, *stAr = mem.extraR.data;
            assert(static_cast<long>(mem.MTX.size) - static_cast<long>(mem.busy_mtx_parts) >= static_cast<long>((2*nparts)));
            DenseMatrix<>* Dr = mem.MTX.data + mem.MTX.size - nparts; //[nparts]
            DenseMatrix<>* Ar = mem.MTX.data + mem.MTX.size - 2*nparts; //[nparts]
            for (uint db = 0; db < nparts; ++db){
                auto ldim = V.stRow[db + 1] - V.stRow[db];
                auto nrow = mem.q * ldim;
                auto ncol = nfA*mem.f;
                auto sz = nrow*ncol;
                Dr[db].Init(stDr, nrow, ncol, sz);
                auto& lDr = Dr[db];
                stDr += sz;

                for (uint j = 0; j < ncol; ++j)
                    for (std::size_t n = 0; n < mem.q; ++n)
                        for (int i = V.stRow[db]; i < V.stRow[db + 1]; ++i)
                            lDr(i - V.stRow[db] + ldim * n, j) = mem.DU.data[i + dimB * (n + mem.q * j )];
                auto lnfb = V.stCol[db + 1] - V.stCol[db];
                Ar[db].Init(stAr,lnfb, nfA*mem.f, lnfb*nfA*mem.f);
                stAr += lnfb*nfA*mem.f;
                fusive_AT_mul_B(ldim * mem.q <= 8 || nfA + lnfb <= 16, mem.f, nrow, V.data[db].data, lnfb, lDr.data, nfA, Ar[db].data);
            }
            for (uint j = 0; j < nfA*mem.f; ++j)
                for (uint db = 0; db < nparts; ++db){
                    for (int i = V.stCol[db]; i < V.stCol[db + 1]; ++i)
                        A.data[i + nfB * j] = Ar[db](i - V.stCol[db], j);
                }
        }
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_A_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
#endif               
    }

    static inline void internalFem3DApply(const ApplyOpBase& applyOpU, const DenseMatrix<const double>& dofs, DenseMatrix<> opU, AniMemoryX<>& mem){
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        const unsigned dimA = applyOpU.Dim(), nfA = applyOpU.Nfa();
        assert(opU.nRow == static_cast<std::size_t>(dimA*mem.q) && opU.nCol == mem.f && "A matrix has wrong dimensions");
        Ani::BandDenseMatrixX<> U = applyOpU(mem, mem.U);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_U_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif 
        auto nparts = U.nparts;
        if (nparts == 1) {
            fusive_A_mul_B(dimA * mem.q <= 8 || nfA <= 16, mem.f, nfA, U.data[0].data, dimA * mem.q, const_cast<double*>(dofs.data), 1, opU.data);
        } else {
            double* tmp_opU = (mem.q <= 1) ? opU.data : mem.extraR.data;
            for (std::size_t db = 0; db < nparts; ++db)
                for (std::size_t r = 0; r < mem.f; ++r){
                    auto VmRow = U.stRow[db+1] - U.stRow[db];
                    auto VmCol = (U.stCol[db+1] - U.stCol[db])/mem.f;
                    double* lVm = U.data[db].data + mem.q*VmRow * VmCol * r; 
                    const double* ldofs = dofs.data + r*dofs.nRow + U.stCol[db];
                    double* lopU = tmp_opU + r*dimA*mem.q + U.stRow[db]*mem.q;
                    fusive_A_mul_B(VmCol<= 8, mem.f, VmCol, lVm, mem.q*VmRow, const_cast<double*>(ldofs), 1, lopU);  
                }
            if (mem.q > 1) {
                for (std::size_t n = 0; n < mem.q; ++n)
                for (std::size_t db = 0; db < nparts; ++db){
                    auto ldim = U.stRow[db+1] - U.stRow[db];
                    double* from = tmp_opU + U.stRow[db]*mem.q + ldim*n;
                    double* to = opU.data + dimA*n + U.stRow[db];
                    std::copy(from, from + ldim, to);
                }
            }     
        }
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_V_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
    }
    
    template<typename DFUNC>
    void internalFem3DApply(const ApplyOpBase& applyOpU, const DenseMatrix<const double>& dofs, DFUNC& Dfnc, TensorDims Ddims, DenseMatrix<> opU, 
                            AniMemoryX<>& mem, void *user_data, bool dotWithNormal){
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        const auto dimA = applyOpU.Dim(), nfA = applyOpU.Nfa();
        uint nD = !dotWithNormal ? DotWithNormalHelper<false>::nD::value : DotWithNormalHelper<true>::nD::value;
        assert(opU.nRow == static_cast<decltype(opU.nRow)>(Ddims.first*mem.q/nD) && opU.nCol == static_cast<decltype(opU.nCol)>(mem.f) && "opU matrix has wrong dimensions");
        Ani::BandDenseMatrixX<> U = applyOpU(mem, mem.U);
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_U_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif 
        auto nparts = U.nparts;
        if (nparts == 1) {
            fusive_A_mul_B(dimA * mem.q <= 8 || nfA <= 16, mem.f, nfA, U.data[0].data, dimA * mem.q, const_cast<double*>(dofs.data), 1, mem.V.data);
        } else {
            double* tmp_opU = (mem.q <= 1) ? mem.V.data : mem.extraR.data;
            for (std::size_t db = 0; db < nparts; ++db)
                for (std::size_t r = 0; r < mem.f; ++r){
                    auto VmRow = U.stRow[db+1] - U.stRow[db];
                    auto VmCol = (U.stCol[db+1] - U.stCol[db])/mem.f;
                    double* lVm = U.data[db].data + mem.q*VmRow * VmCol * r; 
                    const double* ldofs = dofs.data + r*dofs.nRow + U.stCol[db];
                    double* lopU = tmp_opU + r*dimA*mem.q + U.stRow[db]*mem.q;
                    fusive_A_mul_B(VmCol<= 8, mem.f, VmCol, lVm, mem.q*VmRow, const_cast<double*>(ldofs), 1, lopU);  
                }
            if (mem.q > 1) {
                for (std::size_t n = 0; n < mem.q; ++n)
                for (std::size_t db = 0; db < nparts; ++db){
                    auto ldim = U.stRow[db+1] - U.stRow[db];
                    double* from = tmp_opU + U.stRow[db]*mem.q + ldim*n;
                    double* to = mem.V.data + dimA*n + U.stRow[db];
                    std::copy(from, from + ldim, to);
                }
            }    
        }
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_V_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
        Inter_st1 = std::chrono::high_resolution_clock::now();
#endif
        auto Vmx1 = convToBendMx(DenseMatrix<>(mem.V.data, static_cast<int>(dimA * mem.q), static_cast<int>(mem.f)), static_cast<int>(dimA), static_cast<int>(1));
        BandDenseMatrixX<> Vmx(Vmx1);
        DU_comp_in<DFUNC, double, int> DU_dat{mem, Dfnc, user_data, Vmx, static_cast<int>(dimA), static_cast<int>(Ddims.first), static_cast<int>(mem.q), static_cast<int>(mem.f), 1, 1};
        DenseMatrix<> DU = internal_Dfunc<double, int>::applyDU(DU_dat);   
        if (dotWithNormal) 
            DotWithNormalHelper<true>::ProcessNormal(DU, mem);
        std::copy(DU.data, DU.data + Ddims.first*mem.q*mem.f/nD, opU.data); 
#ifdef OPTIMIZER_TIMERS_FEM3DTET
        Inter_end1 = std::chrono::high_resolution_clock::now();
        T_DU_F3DTet += std::chrono::duration<double, std::milli>(Inter_end1 - Inter_st1).count();
#endif
    }

    template<typename FuncTraits>
    size_t _internal_func_handler_memory_requirements(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, bool dotWithNormal){
        return static_cast<size_t>(DFunc<void*, FuncTraits, double, int, -1>::memReq(TensorDims(applyOpV.Dim()*(dotWithNormal ? 3 : 1), applyOpU.Dim())));
    }
    template<typename FuncTraits, typename ScalarType, typename IndexType>
    PlainMemoryX<ScalarType, IndexType> fem3D_memory_requirements_base(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, uint pnt_per_tetra, uint fusion, bool dotWithNormal, bool copyQuadVals){
        auto r = applyOpU.getMemoryRequirements(pnt_per_tetra, fusion);
        int Vprts = r.mtx_parts;
        if (applyOpU != applyOpV){
            auto rV = applyOpV.getMemoryRequirements(pnt_per_tetra, fusion);
            Vprts = rV.mtx_parts;
            r.Usz += rV.Usz, r.extraRsz = std::max(r.extraRsz, rV.extraRsz), r.extraIsz = std::max(r.extraIsz, rV.extraIsz), r.mtx_parts += rV.mtx_parts;
        }
        IndexType f = fusion, q = pnt_per_tetra;
        r.extraRsz = std::max(r.extraRsz, static_cast<std::size_t>(applyOpU.Nfa()*applyOpV.Nfa()*f));
        uint nD = dotWithNormal ? 3U : 1U;
        IndexType DU = applyOpV.Dim() * nD * q * applyOpU.Nfa() * f;
        IndexType DIFF = applyOpV.Dim() * std::max(applyOpU.Dim() * nD, applyOpU.Nfa()) * q * f;
        IndexType XYG = 3*q*f, XYP = 3*4*f, PSI = 3*3*f, DET = f, MES = f;
        IndexType NRM = dotWithNormal ? 3*f : 0;
        IndexType XYL = copyQuadVals ? 4*q : 0, WG = copyQuadVals ? q : 0;
        PlainMemoryX<ScalarType, IndexType> pm;
        pm.dSize = r.Usz + DU + DIFF + XYG + XYP + PSI + DET + MES + NRM + r.extraRsz + XYL + WG;
        pm.dSize += _internal_func_handler_memory_requirements<FuncTraits>(applyOpU, applyOpV, dotWithNormal);
        pm.iSize = r.extraIsz + (4 + 2*r.mtx_parts);
        pm.mSize = r.mtx_parts + (Vprts > 1 ? 2*Vprts : 0);

        return pm;
    }
    template<typename DFUNC, typename ScalarType, typename IndexType>
    AniMemoryX<ScalarType, IndexType> 
        fem3D_init_animem_from_plain_memory(const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, DFUNC& Dfnc, uint pnt_per_tetra, uint fusion, 
                                                PlainMemoryX<ScalarType, IndexType> mem, bool dotWithNormal, bool copyQuadVals){
        PlainMemoryX<ScalarType, IndexType> req = fem3D_memory_requirements_base<typename DFUNC::Traits>(applyOpU, applyOpV, pnt_per_tetra, fusion, dotWithNormal, copyQuadVals);
        if (mem.mSize < req.mSize || mem.dSize < req.dSize || mem.iSize < req.iSize) 
            throw std::runtime_error("Not enough memory, expected minimal sizes = " + 
                            std::to_string(req.mSize) + ":" + std::to_string(req.dSize) +":"+ std::to_string(req.iSize)+" but found memory have sizes = " + 
                            std::to_string(mem.mSize) + ":" + std::to_string(mem.dSize) +":"+ std::to_string(mem.iSize));
        AniMemoryX<ScalarType, IndexType> res;
        ScalarType* pr = mem.ddata;
        IndexType* pi = mem.idata;
        auto* pm = mem.mdata;
        IndexType f = fusion, q = pnt_per_tetra;
        IndexType csz = 0;

        csz = 3*q*f; res.XYG.Init(pr, csz); pr += csz;
        csz = 3*4*f; res.XYP.Init(pr, csz); pr += csz;
        csz = 3*3*f; res.PSI.Init(pr, csz); pr += csz;
        csz = f; res.DET.Init(pr, csz); pr += csz;
        csz = f; res.MES.Init(pr, csz); pr += csz;
        if (dotWithNormal){
            csz = 3*f; res.NRM.Init(pr, csz); pr += csz;
        } else {
            res.NRM.Init(nullptr, 0);
        }
        if (copyQuadVals){
            csz = 4*q; res.XYL.Init(pr, csz); pr += csz;
            csz = q; res.WG.Init(pr, csz); pr += csz;
        } else {
            res.XYL.Init(nullptr, 0);
            res.WG.Init(nullptr, 0);
        }
        uint nD = dotWithNormal ? 3U : 1U;
        csz = applyOpV.Dim() * nD * q * applyOpU.Nfa() * f; res.DU.Init(pr, csz); pr += csz;
        csz = applyOpV.Dim() * std::max(applyOpU.Dim() * nD, applyOpU.Nfa()) * q * f; res.DIFF.Init(pr, csz); pr += csz;
        auto r = applyOpU.getMemoryRequirements(pnt_per_tetra, fusion);
        csz = r.Usz; res.U.Init(pr, csz); pr += csz;
        int Vprts = r.mtx_parts;
        if (applyOpU != applyOpV){
            auto rV = applyOpV.getMemoryRequirements(pnt_per_tetra, fusion);
            csz = rV.Usz; res.V.Init(pr, csz); pr += csz;
            Vprts = rV.mtx_parts;
            r.Usz += rV.Usz, r.extraRsz = std::max(r.extraRsz, rV.extraRsz), r.extraIsz = std::max(r.extraIsz, rV.extraIsz), r.mtx_parts += rV.mtx_parts;
        } else{
            res.V.Init(nullptr, 0);
        }
        csz = r.extraRsz; res.extraR.Init(pr, csz); pr += csz;
        csz = r.extraIsz; res.extraI.Init(pi, csz); pi += csz;
        res.q = q; res.f = f;
        csz = 2 + 1*r.mtx_parts; res.MTXI_ROW.Init(pi, csz); pi += csz;
        csz = 2 + 1*r.mtx_parts; res.MTXI_COL.Init(pi, csz); pi += csz;
        csz = r.mtx_parts+(Vprts > 1 ? 2*Vprts : 0); res.MTX.Init(pm, csz); pm += csz;
        csz = DFUNC::memReq(TensorDims{nD * applyOpV.Dim(), applyOpU.Dim()}); Dfnc.setMem(ArrayView<>(pr, csz)); pr += csz;
        return res;
    }
    template<typename FuncTraits, typename ScalarType, typename IndexType>
    PlainMemoryX<ScalarType, IndexType> fem3DapplyD_memory_requirements_base(const ApplyOpBase& applyOpU, uint Ddim1, uint pnt_per_tetra, uint fusion, bool dotWithNormal){
        auto r = applyOpU.getMemoryRequirements(pnt_per_tetra, fusion);
        IndexType f = fusion, q = pnt_per_tetra;
        r.extraRsz = std::max(static_cast<int>(r.extraRsz), static_cast<int>(applyOpU.Nfa()*1*f));
        IndexType V = applyOpU.Dim()*q*f;
        if (r.mtx_parts > 1 && q > 1)
            r.extraRsz = std::max(r.extraRsz, static_cast<std::size_t>(V));
        //uint nD = dotWithNormal ? 3U : 1U;
        IndexType DU = Ddim1 * q * 1 * f;
        IndexType DIFF = Ddim1 * applyOpU.Dim() * q * f;
        IndexType XYG = 3*q*f, XYP = 3*4*f, PSI = 3*3*f, DET = f, MES = f;
        IndexType NRM = dotWithNormal ? 3*f : 0;
        IndexType XYL = 4*q, WG = q;
        PlainMemoryX<ScalarType, IndexType> pm;
        pm.dSize = r.Usz + V + DU + DIFF + XYG + XYP + PSI + DET + MES + NRM + r.extraRsz + XYL + WG;
        pm.dSize += static_cast<size_t>(DFunc<void*, FuncTraits, double, int, -1>::memReq(TensorDims(Ddim1, applyOpU.Dim())));
        pm.iSize = r.extraIsz + (4 + 2*r.mtx_parts);
        pm.mSize = r.mtx_parts;

        return pm;
    }
    template<typename DFUNC, typename ScalarType, typename IndexType>
    AniMemoryX<ScalarType, IndexType> 
        fem3DapplyD_init_animem_from_plain_memory(const ApplyOpBase& applyOpU, uint Ddim1, DFUNC& Dfnc, uint pnt_per_tetra, uint fusion, 
                                                PlainMemoryX<ScalarType, IndexType> mem, bool dotWithNormal){
        PlainMemoryX<ScalarType, IndexType> req = fem3DapplyD_memory_requirements_base<typename DFUNC::Traits>(applyOpU, Ddim1, pnt_per_tetra, fusion, dotWithNormal);
        if (mem.mSize < req.mSize || mem.dSize < req.dSize || mem.iSize < req.iSize) 
            throw std::runtime_error("Not enough memory, expected minimal sizes = " + 
                            std::to_string(req.mSize) + ":" + std::to_string(req.dSize) +":"+ std::to_string(req.iSize)+" but found memory have sizes = " + 
                            std::to_string(mem.mSize) + ":" + std::to_string(mem.dSize) +":"+ std::to_string(mem.iSize));
        AniMemoryX<ScalarType, IndexType> res;
        ScalarType* pr = mem.ddata;
        IndexType* pi = mem.idata;
        auto* pm = mem.mdata;
        IndexType f = fusion, q = pnt_per_tetra;
        IndexType csz = 0;

        csz = 3*q*f; res.XYG.Init(pr, csz); pr += csz;
        csz = 3*4*f; res.XYP.Init(pr, csz); pr += csz;
        csz = 3*3*f; res.PSI.Init(pr, csz); pr += csz;
        csz = f; res.DET.Init(pr, csz); pr += csz;
        csz = f; res.MES.Init(pr, csz); pr += csz;
        if (dotWithNormal){
            csz = 3*f; res.NRM.Init(pr, csz); pr += csz;
        } else {
            res.NRM.Init(nullptr, 0);
        }
        csz = 4*q; res.XYL.Init(pr, csz); pr += csz;
        csz = q; res.WG.Init(pr, csz); pr += csz;
        //uint nD = dotWithNormal ? 3U : 1U;
        csz = Ddim1 * q * 1 * f; res.DU.Init(pr, csz); pr += csz;
        csz = Ddim1 * applyOpU.Dim() * q * f; res.DIFF.Init(pr, csz); pr += csz;
        auto r = applyOpU.getMemoryRequirements(pnt_per_tetra, fusion);
        csz = r.Usz; res.U.Init(pr, csz); pr += csz;
        csz = applyOpU.Dim()*q*f; res.V.Init(pr, csz); pr += csz;
        csz = r.extraRsz; res.extraR.Init(pr, csz); pr += csz;
        csz = r.extraIsz; res.extraI.Init(pi, csz); pi += csz;
        res.q = q; res.f = f;
        csz = 2 + 1*r.mtx_parts; res.MTXI_ROW.Init(pi, csz); pi += csz;
        csz = 2 + 1*r.mtx_parts; res.MTXI_COL.Init(pi, csz); pi += csz;
        csz = r.mtx_parts; res.MTX.Init(pm, csz); pm += csz;
        csz = DFUNC::memReq(TensorDims(Ddim1, applyOpU.Dim())); Dfnc.setMem(ArrayView<>(pr, csz)); pr += csz;
        return res;
    }
    template<typename ScalarType, typename IndexType>
    PlainMemoryX<ScalarType, IndexType> fem3DapplyND_memory_requirements_base(const ApplyOpBase& applyOpU, uint pnt_per_tetra, uint fusion, bool copyQuadVals){
        auto r = applyOpU.getMemoryRequirements(pnt_per_tetra, fusion);
        IndexType f = fusion, q = pnt_per_tetra;
        if (static_cast<std::size_t>(applyOpU.Nfa()*1*fusion) > r.extraRsz)
            r.extraRsz = applyOpU.Nfa()*1*fusion;
        if (r.mtx_parts > 1 && q > 1)
            r.extraRsz = std::max(r.extraRsz, static_cast<std::size_t>(applyOpU.Dim()*q*f));    
        IndexType XYG = 3*q*f, XYP = 3*4*f, PSI = 3*3*f, DET = f, MES = f;
        IndexType XYL = copyQuadVals ? 4*q : 0, WG = 0;
        PlainMemoryX<ScalarType, IndexType> pm;
        pm.dSize = r.Usz + XYG + XYP + PSI + DET + MES + r.extraRsz + XYL + WG;
        pm.iSize = r.extraIsz + (4 + 2*r.mtx_parts);
        pm.mSize = r.mtx_parts;

        return pm;
    }
    template<typename ScalarType, typename IndexType>
    AniMemoryX<ScalarType, IndexType> 
        fem3DapplyND_init_animem_from_plain_memory(const ApplyOpBase& applyOpU, uint pnt_per_tetra, uint fusion, PlainMemoryX<ScalarType, IndexType> mem, bool copyQuadVals){
        PlainMemoryX<ScalarType, IndexType> req = fem3DapplyND_memory_requirements_base<>(applyOpU, pnt_per_tetra, fusion, copyQuadVals);
        if (mem.mSize < req.mSize || mem.dSize < req.dSize || mem.iSize < req.iSize) 
            throw std::runtime_error("Not enough memory, expected minimal sizes = " + 
                            std::to_string(req.mSize) + ":" + std::to_string(req.dSize) +":"+ std::to_string(req.iSize)+" but found memory have sizes = " + 
                            std::to_string(mem.mSize) + ":" + std::to_string(mem.dSize) +":"+ std::to_string(mem.iSize));
        AniMemoryX<ScalarType, IndexType> res;
        ScalarType* pr = mem.ddata;
        IndexType* pi = mem.idata;
        auto* pm = mem.mdata;
        IndexType f = fusion, q = pnt_per_tetra;
        IndexType csz = 0;

        csz = 3*q*f; res.XYG.Init(pr, csz); pr += csz;
        csz = 3*4*f; res.XYP.Init(pr, csz); pr += csz;
        csz = 3*3*f; res.PSI.Init(pr, csz); pr += csz;
        csz = f; res.DET.Init(pr, csz); pr += csz;
        csz = f; res.MES.Init(pr, csz); pr += csz;
        res.NRM.Init(nullptr, 0);
        if (copyQuadVals){
            csz = 4*q; res.XYL.Init(pr, csz); pr += csz;
        } else {
            res.XYL.Init(nullptr, 0);
        }
        res.WG.Init(nullptr, 0);
        res.DU.Init(nullptr, 0);
        res.DIFF.Init(nullptr, 0);
        auto r = applyOpU.getMemoryRequirements(pnt_per_tetra, fusion);
        csz = r.Usz; res.U.Init(pr, csz); pr += csz;
        res.V.Init(nullptr, 0);
        csz = r.extraRsz; res.extraR.Init(pr, csz); pr += csz;
        csz = r.extraIsz; res.extraI.Init(pi, csz); pi += csz;
        res.q = q; res.f = f;
        csz = 2 + 1*r.mtx_parts; res.MTXI_ROW.Init(pi, csz); pi += csz;
        csz = 2 + 1*r.mtx_parts; res.MTXI_COL.Init(pi, csz); pi += csz;
        csz = r.mtx_parts; res.MTX.Init(pm, csz); pm += csz;
        return res;
    }

    template<typename OpA, typename OpB, typename ScalarType, typename IndexType, bool OnFace, bool DotWithNormal>
    PlainMemory <ScalarType, IndexType> fem3D_memory_requirements(int order, int fusion) {
        auto f = fusion;
        auto q = (!OnFace) ? tetrahedron_quadrature_formulas(order).GetNumPoints() : triangle_quadrature_formulas(order).GetNumPoints();
        return fem3D_memory_requirements_base<OpA, OpB, ScalarType, IndexType, OnFace || !std::is_same<ScalarType, qReal>::value, DotWithNormal>(q, f);
    }

};