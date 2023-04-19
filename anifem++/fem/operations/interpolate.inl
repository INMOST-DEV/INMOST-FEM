namespace Ani{
    template<typename FEMTYPE, typename EvalFunc>
    static inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, const DofT::TetGeomSparsity& sp, void* user_data, uint max_quad_order){
        auto dof_map = Dof<FEMTYPE>::Map();
        for (auto it = dof_map.beginBySparsity(sp); it != dof_map.endBySparsity(); ++it)
            Dof<FEMTYPE>::interpolate(XYZ, f, udofs, it->gid, user_data, max_quad_order);
    }
    template<typename FEMTYPE>
    static inline void interpolateConstant(double val, ArrayView<> udofs, const DofT::TetGeomSparsity& sp){
        auto dof_map = Dof<FEMTYPE>::Map();
        for (auto it = dof_map.beginBySparsity(sp); it != dof_map.endBySparsity(); ++it)
            udofs[it->gid] = val;
    }
    template<typename FEMTYPE, typename ScalarType, typename IndexType>
    PlainMemory <ScalarType, IndexType> interpolateByRegion_memory_requirements(unsigned char elem_type, uint max_quad_order){
        PlainMemory <ScalarType, IndexType> res;
        if (elem_type == DofT::NODE) 
            return res;
        constexpr auto NFA = Operator<IDEN, FEMTYPE>::Nfa::value;
        IndexType nquads = 0;
        switch(DofT::DimToGeomType(DofT::GeomTypeDim(elem_type))) {
            case DofT::EDGE:{
                nquads = segment_quadrature_formulas(max_quad_order).GetNumPoints();
                break;
            }
            case DofT::FACE:{
                nquads = triangle_quadrature_formulas(max_quad_order).GetNumPoints();
                break;
            }
            case DofT::CELL:{
                nquads = tetrahedron_quadrature_formulas(max_quad_order).GetNumPoints();
                break; 
            }
        }
        res.dSize += NFA*NFA + NFA + 5*nquads;
        res.iSize += 0;
        auto pnt_mem0 = fem3DpntL_memory_requirements<Operator<IDEN, FEMTYPE>, Operator<IDEN, FEMTYPE>, ScalarType, IndexType>(nquads);
        auto pnt_mem1 = fem3DpntL_memory_requirements<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FEMTYPE>, ScalarType, IndexType>(nquads);
        res.dSize += std::max({pnt_mem0.dSize, pnt_mem1.dSize, static_cast<IndexType>(NFA*(NFA+1)/2)});
        res.iSize += std::max({pnt_mem0.iSize, pnt_mem1.iSize, static_cast<IndexType>(NFA)});

        return res;
    }
    template<typename FEMTYPE, typename EvalFunc, typename ScalarType, typename IndexType>
    static inline void interpolateByRegion(const Tetra<const ScalarType>& XYZ, const EvalFunc& func, ArrayView<ScalarType> udofs, unsigned char elem_type, uint ielem, 
                    PlainMemory<ScalarType, IndexType> plainMemory, void* user_data, uint max_quad_order){
        DofT::TetGeomSparsity sp;
        if (elem_type == DofT::NODE){
            sp.setNode(ielem);
            interpolateByDOFs(XYZ, func, udofs, sp, user_data, max_quad_order);
            return;
        }

        constexpr auto NFA = Operator<IDEN, FEMTYPE>::Nfa::value;
        auto d_alloc = [&plainMemory](auto sz)->ScalarType*{ 
            if (plainMemory.dSize > sz){
                plainMemory.dSize -= sz;
                auto res = plainMemory.ddata;
                plainMemory.ddata += sz;
                return res;
            } else 
                throw std::runtime_error("Not enough real memory");
            return nullptr;    
        };
        auto i_alloc = [&plainMemory](auto sz)->IndexType*{ 
            if (plainMemory.iSize > sz){
                plainMemory.iSize -= sz;
                auto res = plainMemory.idata;
                plainMemory.idata += sz;
                return res;
            } else 
                throw std::runtime_error("Not enough int memory");
            return nullptr;    
        };
        ScalarType* am = d_alloc(NFA*NFA);
        ScalarType* fm = d_alloc(NFA);
        ScalarType* xyl = nullptr, wg = nullptr;
        IndexType nquads = 0;
        switch(DofT::DimToGeomType(DofT::GeomTypeDim(elem_type))) {
            case DofT::EDGE:{
                sp.setEdge(ielem, true);
                const static std::array<char, 12> lookup_nds = {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3};
                auto formula = segment_quadrature_formulas(max_quad_order);
                nquads = formula.GetNumPoints();
                xyl = d_alloc(4*nquads), wg = d_alloc(nquads);
                for (int i = 0; i < nquads; ++i){
                    auto q = formula.GetPointWeight(i);
                    xyl[4*i + lookup_nds[2*ielem]] = q.p[0]; 
                    xyl[4*i + lookup_nds[2*ielem+1]] = q.p[1];
                    wg[i] = q.w;
                }
                break;
            }
            case DofT::FACE:{
                sp.setFace(ielem, true);
                auto formula = triangle_quadrature_formulas(max_quad_order);
                nquads = formula.GetNumPoints();
                xyl = d_alloc(4*nquads), wg = d_alloc(nquads);
                for (int i = 0; i < nquads; ++i){
                    auto q = formula.GetPointWeight(i);
                    xyl[4*i + (ielem+0)%4] = q.p[0]; 
                    xyl[4*i + (ielem+1)%4] = q.p[1];
                    xyl[4*i + (ielem+2)%4] = q.p[2];
                    wg[i] = q.w;
                }
                break;
            }
            case DofT::CELL:{
                sp.setCell(true); 
                auto formula = tetrahedron_quadrature_formulas(max_quad_order);
                nquads = formula.GetNumPoints();
                xyl = d_alloc(4*nquads), wg = d_alloc(nquads);
                auto pp = formula.GetPointData();
                auto wp = formula.GetWeightData();
                std::copy(pp, pp + 4 * nquads, xyl);
                std::copy(wp, wp + nquads, wg);
                break; 
            }          
            default:
                throw std::runtime_error("Faced unknown geometric region");
        }

        DenseMatrix<ScalarType> A(am, NFA, NFA), F(fm, NFA, 1);
        ArrayView<ScalarType> XYL(xyl, nquads*4), WG(wg, nquads);
        fem3DpntL<Operator<IDEN, FEMTYPE>, Operator<IDEN, FEMTYPE>, DfuncTraits<TENSOR_NULL, true>>(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, XYL, WG, TensorNull<ScalarType>, A, plainMemory, nullptr);
        fem3DpntL<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FEMTYPE>>(XYZ.XY0, XYZ.XY1, XYZ.XY2, XYZ.XY3, XYL, WG,
            [&func](const std::array<ScalarType, 3>& x, ScalarType* Dmem, TensorDims Ddims, void* user_data, int iTet)->TensorType{
                (void) iTet;
                func(x, Dmem, Ddims.first, user_data);
                return TENSOR_GENERAL;
            }, 
            F, plainMemory, user_data);

        IndexType* ids = i_alloc(NFA);
        auto dof_map = Dof<FEMTYPE>::Map();
        int nids = 0;
        for (auto it = dof_map.beginBySparsity(sp); it != dof_map.endBySparsity(); ++it)
            ids[nids++] = it->gid;
        if (nids == 0) return;    
        std::sort(ids, ids + nids);

        for (int k = 0; k < nids; ++k)
            F(k, 0) = F(ids[k], 0);
        for (int j = 0; j < nids; ++j)
            for (int i = 0; i < nids; ++i)
                A[i + j * nids] = A(ids[i], ids[j]);
        A.Init(am, nids, nids, NFA*NFA);
        F.Init(fm, nids, 1, NFA);
        if (plainMemory.dSize < NFA*(NFA+1)/2)
            throw std::runtime_error("Not enough real memory");
        cholesky_solve(A.data, F.data, nids, 1, F.data, plainMemory.ddata);
        for (int k = 0; k < nids; ++k)
            udofs[ids[k]] = F[k];
    }
    template<typename FEMTYPE, typename EvalFunc, typename ScalarType, typename IndexType, uint STATIC_MEMORY_SIZE>
    static inline void interpolateByRegion(const Tetra<const ScalarType>& XYZ, const EvalFunc& func, ArrayView<ScalarType> udofs, unsigned char elem_type, uint ielem, void* user_data, uint max_quad_order){
        char mem[STATIC_MEMORY_SIZE];
        PlainMemory<ScalarType, IndexType> pmem = interpolateByRegion_memory_requirements<FEMTYPE, ScalarType, IndexType>(elem_type, max_quad_order);
        assert(pmem.enoughRawSize() <=  STATIC_MEMORY_SIZE && "Not enough static memory, try memory dependent version of interpolateByRegion function");
        pmem.allocateFromRaw(mem, STATIC_MEMORY_SIZE);
        interpolateByRegion<FEMTYPE>(XYZ, func, udofs, elem_type, ielem, pmem, user_data, max_quad_order);
    }
}