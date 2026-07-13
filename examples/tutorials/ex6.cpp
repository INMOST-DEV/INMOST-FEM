//
// Created by Liogky Alexey on 10.07.2026.
//

/**
 * Here we show how to solve a FEM problem with aposteriori error estimation
 * 
 * This program generates and solves a finite element system for the stationary diffusion problem
 * \f[
 * \begin{aligned}
 *  -\mathrm{div}\ \mathbf{D}\ &\mathrm{grad}\ u\ &= \mathbf{F}\ in\  \Omega          \\
 *                             &               u\ &= U_0\        on\  \partial \Omega \\
 * \end{aligned}
 * \f]
 * where Ω = [0,1]^3
 * The user-defined coefficients are
 *    D(x)   = (1 + x^2)*I - positive definite tensor
 *    F(x)   = 1           - right-hand side
 *    U_0(x) = 0           - essential (Dirichlet) boundary condition
 *
 * @see src/Tutorials/PackageFEM/main_simple.f in [Ani3d library](https://sourceforge.net/projects/ani3d/)
 */

#include "prob_args.h"
#include "anifem++/utils/adapt_mesh.h"

using namespace INMOST;

int main(int argc, char* argv[]){
    InputArgs p;
    std::string prob_name = "ex6";
    p.save_prefix = prob_name + "_out"; 
    p.parseArgs_mpi(&argc, &argv);
    unsigned quad_order = p.max_quad_order;

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);
    
    std::unique_ptr<Mesh> mptr = GenerateCube(INMOST_MPI_COMM_WORLD, p.axis_sizes, p.axis_sizes, p.axis_sizes);
    Mesh* m = mptr.get();
    // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    m->ExchangeGhost(1,NODE);
    print_mesh_sizes(m);

    using namespace Ani;
    //generate FEM space from it's name
    FemSpace UFem = FemSpace{P1Space{}};
    uint unf = UFem.dofMap().NumDofOnTet();
    auto& dofmap = UFem.dofMap();
    // To reuse BC mask for error space we will set mask on all types of dirichlet elements
    auto inmost_mask = NODE|EDGE|FACE; //GeomMaskToInmostElementType(UFem.dofMap().GetGeomMask()) | GeomMaskToInmostElementType(WFem.dofMap().GetGeomMask()); 
    auto ani_mask = DofT::NODE|DofT::EDGE|DofT::FACE; // UFem.dofMap().GetGeomMask() | WFem.dofMap().GetGeomMask()
    // Set boundary labels
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, inmost_mask, NONE, 1);
    for (auto it = m->BeginElement(inmost_mask); it != m->EndElement(); ++it){
        std::array<double, 3> c;
        it->Centroid(c.data());
        int lbl = 0;
        for (auto k = 0; k < 3; ++k)
            if (abs(c[k] - 0) < 10*std::numeric_limits<double>::epsilon())
                lbl |= (1 << (2*k+0));
            else if (abs(c[k] - 1) < 10*std::numeric_limits<double>::epsilon()) 
                lbl |= (1 << (2*k+1));   
        it->Integer(BndLabel) = lbl;
    }

    // Define tensors from the problem
    auto D_tensor = [](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) user_data; (void) iTet;
        D[0] = 1 + X[0]*X[0];  
        // D[0] = 1; (void) X;
        return Ani::TENSOR_SCALAR;
    };
    auto F_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        F[0] = 1;
        return Ani::TENSOR_SCALAR;
    };
    auto U_0 = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ (void) X; (void) dim; (void) user_data; res[0] = 0.0; return 0;}; 

    // Define tag to store result
    Tag u = createFemVarTag(m, *dofmap.target<>(), "u");

    // Define user structure to store input data used in local assembler
    struct BndMarker{
        std::array<int, 4> n = {0};
        std::array<int, 6> e = {0};
        std::array<int, 4> f = {0};

        DofT::TetGeomSparsity getSparsity() const {
            DofT::TetGeomSparsity sp;
            for (int i = 0; i < 4; ++i) if (n[i] > 0)
                sp.setNode(i);
            for (int i = 0; i < 6; ++i) if (e[i] > 0)
                sp.setEdge(i);  
            for (int i = 0; i < 4; ++i) if (f[i] > 0)
                sp.setFace(i);
            return sp;    
        }
        void fillFromBndTag(Tag lbl, Ani::DofT::uint geom_mask, const ElementArray<Node>& nodes, const ElementArray<Edge>& edges, const ElementArray<Face>& faces) {
            if (geom_mask & DofT::NODE){
                for (unsigned i = 0; i < n.size(); ++i) 
                    n[i] = nodes[i].Integer(lbl);
            }
            if (geom_mask & DofT::EDGE){
                for (unsigned i = 0; i < e.size(); ++i) 
                    e[i] = edges[i].Integer(lbl);
            }
            if (geom_mask & DofT::FACE){
                for (unsigned i = 0; i < f.size(); ++i) 
                    f[i] = faces[i].Integer(lbl);
            }
        }
    };
    struct ProbLocData{
        //save labels used to apply boundary conditions
        BndMarker lbl;

        ArrayView<> udofs;
    };
    
    // Create implementation of local_assembler requiring actual fem space on input
    auto local_assembler_gen = [&D_tensor, &F_tensor, &U_0](const BaseFemSpace& ufem, const BaseFemSpace& wfem, unsigned order, bool skip_dc, 
            const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, DynMem<double, long>* fem_alloc) -> void{
        (void) w, (void) iw;
        unsigned unf = ufem.dofMap().NumDofOnTet();
        unsigned wnf = wfem.dofMap().NumDofOnTet();
        DenseMatrix<> A(Adat, wnf, unf), F(Fdat, wnf, 1);
        A.SetZero(); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        // here we set DfuncTraits<TENSOR_SCALAR, false> because we know that D_tensor is TENSOR_SCALAR on all elements
        // and also we know that it's NOT constant
        // elemental stiffness matrix <grad(P1), D grad(P1)> 
        auto grad_w = wfem.getOP(GRAD), iden_w = wfem.getOP(IDEN); 
        auto grad_u = ufem.getOP(GRAD); 
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        fem3Dtet<DfuncTraits<TENSOR_SCALAR, false>>(XYZ, grad_u, grad_w, D_tensor, A, adapt_alloc, order, user_data); 
        // elemental right hand side vector <F, P1>
        fem3Dtet<DfuncTraits<TENSOR_SCALAR, true>>( XYZ, iden_p0, iden_w, F_tensor, F, adapt_alloc, order, user_data);

        // read node labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // choose boundary parts of the tetrahedron 
        if (!skip_dc){
            DofT::TetGeomSparsity sp = dat.lbl.getSparsity();
            if (sp.empty()) return;
            bool homogenousDC = false;
            if (!homogenousDC){
                auto mem = fem_alloc->alloc(wnf, 0, 0);
                ArrayView<> dof_values(mem.getPlainMemory().ddata, mem.getPlainMemory().dSize);
                wfem.interpolateByDOFs(XYZ, U_0, dof_values, sp, adapt_alloc, user_data);
                applyDirByDofs(wfem.dofMap(), A, F, sp, ArrayView<const double>(dof_values.data, dof_values.size));
            } else {
                applyDirResidual(wfem.dofMap(), A, F, sp);
            }
        }
    };
    
    auto local_data_gatherer_gen = [&BndLabel, geom_mask = ani_mask](std::function<Ani::DynMem<double, long>::MemPart(ElementalAssembler& p, ProbLocData& data)> callback, ElementalAssembler& p)->void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.lbl.fillFromBndTag(BndLabel, geom_mask, *p.nodes, *p.edges, *p.faces);
        Ani::DynMem<double, long>::MemPart mpart = callback(p, data);

        p.compute(args, &data);
    };

    using namespace std::placeholders;
    using LocalAssmFunc = std::function<void(const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, DynMem<double, long>* fem_alloc)>;
    using LocalGatherFunc = std::function<void(ElementalAssembler& p)>;
    
    auto solve_linear_problem = [pRank](Mesh* m, LocalAssmFunc local_assm_func, LocalGatherFunc local_data_gatherer, const BaseFemSpace& fem, Tag res, std::string solver_name, std::string solver_prefix, std::string problem_name){
        static int unum = 0;
        unum++;
        unsigned nf = fem.dofMap().NumDofOnTet();
        //define assembler
        Assembler discr(m);
        discr.SetMatRHSFunc(GenerateElemMatRhs(local_assm_func, nf, nf, 0, 0));
        {
            //create global degree of freedom enumenator
            auto Var0Helper = GenerateHelper(fem);
            FemExprDescr fed;
            fed.PushTrialFunc(Var0Helper, "u" + std::to_string(unum));
            fed.PushTestFunc(Var0Helper, "phi_u" + std::to_string(unum));
            discr.SetProbDescr(std::move(fed));
        }
        discr.SetDataGatherer(local_data_gatherer);
        discr.PrepareProblem();
        Sparse::Matrix A("A" + std::to_string(unum));
        Sparse::Vector x("x" + std::to_string(unum)), b("b" + std::to_string(unum));
        //assemble matrix and right-hand side
        TimerWrap m_timer_total; m_timer_total.reset();
        discr.Assemble(A, b);
        double total_assm_time = m_timer_total.elapsed();
        // Print Assembler timers
        if (pRank == 0) {
            std::cout << "#dofs = " << discr.m_enum.getMatrixSize() << std::endl; 
            std::cout << "Assembler timers:"
            #ifndef NO_ASSEMBLER_TIMERS
                    << "\n\tInit assembler: " << discr.GetTimeInitAssembleData() << "s"
                    << "\n\tInit guess    : " << discr.GetTimeInitValSet() << "s"
                    << "\n\tInit handler  : " << discr.GetTimeInitUserHandler() << "s"
                    << "\n\tLocal evaluate: " << discr.GetTimeEvalLocFunc() << "s"
                    << "\n\tLocal postproc: " << discr.GetTimePostProcUserHandler() << "s"
                    << "\n\tComp glob inds: " << discr.GetTimeFillMapTemplate() << "s"
                    << "\n\tGlobal remap  : " << discr.GetTimeFillGlobalStructs() << "s"
            #endif          
                    << "\n\tTotal         : " << total_assm_time << "s" 
                    << std::endl;
        }

        //setup linear solver and solve assembled system
        Solver solver(solver_name, solver_prefix);
        solver.SetMatrix(A);
        solver.Solve(b, x);
        print_linear_solver_status(solver, problem_name, true);
        
        //copy result to the tag and save solution
        discr.SaveVar(x, 0, res);
        discr.Clear();
    };

    {
        LocalAssmFunc local_assm_func = std::bind(local_assembler_gen, std::cref(*UFem.target<>()), std::cref(*UFem.target<>()), quad_order, false, _1, _2, _3, _4, _5, _6, _7);
        LocalGatherFunc local_data_gatherer = std::bind(local_data_gatherer_gen, [](ElementalAssembler&, ProbLocData&)->Ani::DynMem<double, long>::MemPart{ return Ani::DynMem<double, long>::MemPart{}; }, _1);
        solve_linear_problem(m, local_assm_func, local_data_gatherer, *UFem.base(), u, p.lin_sol_nm, p.lin_sol_prefix, "problem");
    }
    bool compute_aposteriori_error = true;
    if (compute_aposteriori_error) {   // Compute aposteriori error
        FemSpace WFem = FemSpace(Bubble2Space{});
        // unsigned wnf = WFem.dofMap().NumDofOnTet();
        Tag err = createFemVarTag(m, *WFem.dofMap().target<>(), "err");
        
        auto gather_udofs_callback = [unf, u, &udmap = dofmap](ElementalAssembler& p, ProbLocData& d) -> Ani::DynMem<double, long>::MemPart {
            Ani::DynMem<double, long>::MemPart res = p.pool->alloc(unf, 0, 0);
            d.udofs.Init(res.m_mem.ddata, unf);
            Ani::GatherDataOnElement(u, *(udmap.target<>()), *p.cell, *p.faces, *p.edges, *p.nodes, p.node_permutation, d.udofs.data, nullptr, 0);
            return res;
        };
        LocalGatherFunc local_data_gatherer = std::bind(local_data_gatherer_gen, gather_udofs_callback, _1);
        LocalAssmFunc local_assm_err = [local_assembler_gen, &Ufem = *UFem.target<>(), &Wfem = *WFem.target<>(), order = quad_order](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, DynMem<double, long>* fem_alloc) -> void{
            unsigned unf = Ufem.dofMap().NumDofOnTet();
            unsigned wnf = Wfem.dofMap().NumDofOnTet();
            auto& dat = *static_cast<ProbLocData*>(user_data);
            DofT::TetGeomSparsity sp = dat.lbl.getSparsity();

            local_assembler_gen(Wfem, Wfem, order, true, XY, Adat, Fdat, w, iw, user_data, fem_alloc);
            DenseMatrix<> A(Adat, wnf, wnf), F(Fdat, wnf, 1);

            auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
            auto mem = adapt_alloc.alloc(wnf*(unf + 1), 0, 0);

            DenseMatrix<> B(mem.m_mem.ddata, wnf, unf), T(mem.m_mem.ddata + wnf*unf, wnf, 1);
            local_assembler_gen(Ufem, Wfem, order, true, XY, B.data, T.data, w, iw, user_data, fem_alloc);
            T.SetZero();
            for (unsigned i = 0; i < wnf; ++i)
            for (unsigned j = 0; j < unf; ++j)
                T[i] += B(i, j) * dat.udofs[j];
            
            for (unsigned i = 0; i < wnf; ++i)
                F[i] -= T[i];
            
            if (!sp.empty())
                applyDirResidual(Wfem.dofMap(), A, F, sp);
        };
        solve_linear_problem(m, local_assm_err, local_data_gatherer, *WFem.base(), err, p.lin_sol_nm, p.lin_sol_prefix, "error estimation");
        
        auto grad_err_op = WFem.getOP(GRAD);
        auto& err_dmap = *WFem.dofMap().target<>();

        // squared energy seminorm over volume: ∫(∇err)^2 dx / ∫ dx
        double ints[2] = {0, 0};
        integrate_vector_func(ints, 2, m,
            [err, &err_dmap, grad_err_op](Cell c, DenseMatrix<const double> XYL, DenseMatrix<const double> X, DenseMatrix<> out, DynMem<>& mem){
                (void) X;
                const unsigned q = XYL.nCol;
                auto gmem = mem.alloc(3 * q, 0, 0);
                DenseMatrix<> grad(gmem.m_mem.ddata, 3, q);
                eval_op_var_by_barycentric_points(grad, mem, c, XYL, grad_err_op, err_dmap, err);
                for (unsigned n = 0; n < q; ++n){
                    out(0, n) = grad(0, n)*grad(0, n) + grad(1, n)*grad(1, n) + grad(2, n)*grad(2, n);
                    out(1, n) = 1.0;
                }
            }, quad_order);
        if (pRank == 0)
            std::cout << "sqrt(||(∇err)^2||_L2 / |Ω|) = " << sqrt(ints[0] / ints[1]) << std::endl;
        
        Tag metrics = m->CreateTag("metrics", DATA_REAL, NODE, NONE, 6);
        MetricsConstructTraits mtraits; mtraits.verbosity = true;
        construct_metrics(WFem, err, metrics, mtraits);

        // L2-project grad(err) onto vector P1 space
        FemSpace GradErrFem = FemSpace{P1Space{}} ^ 3;
        Tag grad_err = createFemVarTag(m, *GradErrFem.dofMap().target<>(), "grad_err");
        make_l2_project(
            [err, &err_dmap, grad_err_op](Cell c, DenseMatrix<const double> XYL, DenseMatrix<const double> X, DenseMatrix<> out, DynMem<>& mem){
                (void) X;
                eval_op_var_by_barycentric_points(out, mem, c, XYL, grad_err_op, err_dmap, err);
            }, m, grad_err, GradErrFem, static_cast<int>(quad_order));
    }
    bool compute_actual_p2p1_error = true;
    if (compute_actual_p2p1_error) {
        // Solve the same problem in P2 and form err2 = u_P2 - u_P1
        FemSpace U2Fem = FemSpace{P2Space{}};
        Tag u2 = createFemVarTag(m, *U2Fem.dofMap().target<>(), "u2");
        {
            LocalAssmFunc local_assm_func = std::bind(local_assembler_gen, std::cref(*U2Fem.target<>()), std::cref(*U2Fem.target<>()), quad_order, false, _1, _2, _3, _4, _5, _6, _7);
            LocalGatherFunc local_data_gatherer = std::bind(local_data_gatherer_gen, [](ElementalAssembler&, ProbLocData&)->Ani::DynMem<double, long>::MemPart{ return Ani::DynMem<double, long>::MemPart{}; }, _1);
            solve_linear_problem(m, local_assm_func, local_data_gatherer, *U2Fem.base(), u2, p.lin_sol_nm, p.lin_sol_prefix, "problem_P2");
        }

        Tag err2 = createFemVarTag(m, *U2Fem.dofMap().target<>(), "err2");
        auto iden_u1 = UFem.getOP(IDEN);
        auto iden_u2 = U2Fem.getOP(IDEN);
        auto& u1_dmap = *UFem.dofMap().target<>();
        auto& u2_dmap = *U2Fem.dofMap().target<>();

        // squared energy seminorm over volume: ∫(∇err2)^2 dx / ∫ dx
        double ints[2] = {0, 0};
        integrate_vector_func(ints, 2, m,
            [u2, &u2_dmap = u2_dmap, grad_u2_op = U2Fem.getOP(GRAD), u1 = u, &u1_dmap = u1_dmap, grad_u1_op = UFem.getOP(GRAD)](Cell c, DenseMatrix<const double> XYL, DenseMatrix<const double> X, DenseMatrix<> out, DynMem<>& mem){
                (void) X;
                const unsigned q = XYL.nCol;
                auto gmem = mem.alloc(6 * q, 0, 0);
                DenseMatrix<> grad2(gmem.m_mem.ddata, 3, q), grad1(gmem.m_mem.ddata + 3*q, 3, q);
                eval_op_var_by_barycentric_points(grad2, mem, c, XYL, grad_u2_op, u2_dmap, u2);
                eval_op_var_by_barycentric_points(grad1, mem, c, XYL, grad_u1_op, u1_dmap, u1);
                for (unsigned n = 0; n < q; ++n){
                    std::array<double, 3> g{grad2(0, n) - grad1(0, n), grad2(1, n) - grad1(1, n), grad2(2, n) - grad1(2, n)};
                    out(0, n) = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
                    out(1, n) = 1.0;
                }
            }, quad_order);
        if (pRank == 0)
            std::cout << "sqrt(||(∇err2)^2||_L2 / |Ω|) = " << sqrt(ints[0] / ints[1]) << std::endl;

        make_l2_project(
            [u, u2, &u1_dmap, &u2_dmap, iden_u1, iden_u2](Cell c, DenseMatrix<const double> XYL, DenseMatrix<const double> X, DenseMatrix<> out, DynMem<>& mem){
                (void) X;
                const unsigned q = XYL.nCol;
                auto vmem = mem.alloc(2 * q, 0, 0);
                DenseMatrix<> v2(vmem.m_mem.ddata, 1, q), v1(vmem.m_mem.ddata + q, 1, q);
                eval_op_var_by_barycentric_points(v2, mem, c, XYL, iden_u2, u2_dmap, u2);
                eval_op_var_by_barycentric_points(v1, mem, c, XYL, iden_u1, u1_dmap, u);
                for (unsigned n = 0; n < q; ++n)
                    out(0, n) = v2(0, n) - v1(0, n);
            }, m, err2, U2Fem, static_cast<int>(quad_order));
        
        Tag metrics = m->CreateTag("metrics2", DATA_REAL, NODE, NONE, 6);
        MetricsConstructTraits mtraits; mtraits.verbosity = true;
        // mtraits.vertex_projection_strategy = 1;
        // mtraits.distrib_heuristics = false;
        construct_metrics(U2Fem, u2, metrics, mtraits);

        FemSpace GradErrFem = FemSpace{P1Space{}} ^ 3;
        Tag grad_err2 = createFemVarTag(m, *GradErrFem.dofMap().target<>(), "grad_err2");
        auto grad_err2_op = U2Fem.getOP(GRAD);
        make_l2_project(
            [err2, &err2_dmap = u2_dmap, grad_err2_op](Cell c, DenseMatrix<const double> XYL, DenseMatrix<const double> X, DenseMatrix<> out, DynMem<>& mem){
                (void) X;
                eval_op_var_by_barycentric_points(out, mem, c, XYL, grad_err2_op, err2_dmap, err2);
            }, m, grad_err2, GradErrFem, static_cast<int>(quad_order));
    }

    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    mptr.reset();
    InmostFinalize();

    return 0;
}