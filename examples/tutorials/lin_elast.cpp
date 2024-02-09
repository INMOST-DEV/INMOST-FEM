// Created by Liogky Alexey on 09.02.2024.
//

/**
 * This program generates  and solves a finite element system for the stationary linear elasticity problem
 *
 * \f[
 * \begin{aligned}
 *   \mathrm{div}\ \sigma (\varepsilon) + f &= 0\    in\  \Omega  \\
 *          \mathbf{u}                      &= \mathbf{u}_0\  on\  \Gamma_D\\
 *                  \sigma \cdot \mathbf{N} &= -p \mathbf{N}\ on\  \Gamma_P\\  
 *                  \sigma \cdot \mathbf{N} &= 0\    on\  \Gamma_0\\
 *                  \sigma                  &= C_{(ij)(kl)} \varepsilon_{(kl)}\\
 *   C_{(ij)(kl)} &= \lambda \delta_{ij}\delta_{kl} + \mu (\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})\\
 *   \varepsilon &= \frac{1}{2}(\mathrm{grad}\ \mathbf{u} + (\mathrm{grad}\ \mathbf{u})^T)
 * \end{aligned}
 * \f]
 * 
 *
 * where where Ω = [0,10] x [0,1]^2, Γ_D = {0}x[0,1]^2, Γ_P = [0,10]x[0,1]x{0}, Γ_0 = ∂Ω \ (Γ_D ⋃ Γ_P)            
 * The user-defined coefficients are 
 *  μ(x)   = 3.0         - first Lame coefficient
 *  λ(x)   = 1.0         - second Lame coefficient
 *  f(x)   = { 0, 0, 0 } - external body forces
 *  u_0(x) = { 0, 0, 0 } - essential (Dirichlet) boundary condition
 *  p(x)   = 0.001         - pressure on bottom part
 * 
 */

#include "prob_args.h"

using namespace INMOST;

int main(int argc, char* argv[]){
    InputArgs1Var p;
    std::string prob_name = "lin_elast";
    p.save_prefix = prob_name + "_out"; 
    p.parseArgs_mpi(&argc, &argv);
    unsigned quad_order = p.max_quad_order;

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    std::unique_ptr<Mesh> mptr = GenerateParallelepiped(INMOST_MPI_COMM_WORLD, {10*p.axis_sizes, p.axis_sizes, p.axis_sizes}, {10, 1, 1}, {0, 0, 0});
    Mesh* m = mptr.get();
    // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    m->ExchangeGhost(1,NODE);
    print_mesh_sizes(m);

    using namespace Ani;
    //generate FEM space from it's name
    FemSpace UFem = choose_space_from_name(p.USpace)^3;
    uint unf = UFem.dofMap().NumDofOnTet();
    auto& dofmap = UFem.dofMap();
    auto mask = GeomMaskToInmostElementType(dofmap.GetGeomMask()) | FACE;
    const int INTERNAL_PART = 0, FREE_BND = 1, DIRICHLET_BND = 2, PRESSURE_BND = 4;
    
    // Set boundary labels on all boundaries
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, mask, NONE, 1);
    auto bmrk = m->CreateMarker();
    m->MarkBoundaryFaces(bmrk);
    for (auto it = m->BeginElement(mask); it != m->EndElement(); ++it) it->Integer(BndLabel) = INTERNAL_PART;
    for (auto it = m->BeginFace(); it != m->EndFace(); ++it) if (it->GetMarker(bmrk)) {
        std::array<double, 3> c;
        it->Centroid(c.data());
        int lbl = FREE_BND;
        if (abs(c[2] - 0) < 10*std::numeric_limits<double>::epsilon()) lbl = PRESSURE_BND;
        if (abs(c[0] - 0) < 10*std::numeric_limits<double>::epsilon()) lbl = DIRICHLET_BND;
        auto set_label = [BndLabel, lbl](const auto& elems){
            for (unsigned ni = 0; ni < elems.size(); ni++)
                elems[ni].Integer(BndLabel) |= lbl;
        };
        if (mask & NODE) set_label(it->getNodes());
        if (mask & EDGE) set_label(it->getEdges());
        if (mask & FACE) set_label(it->getFaces());
    }
    m->ReleaseMarker(bmrk);

    // Define tensors from the problem
    auto C_tensor = [](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet) -> TensorType{
        assert(dims.first == 9 && dims.second == 9 && "Wrong dimesions");
        (void) X; (void) dims; (void) user_data; (void) iTet;
        DenseMatrix<> C(D, 9, 9); 
        double lambda = 1.0, mu = 3.0;
        //C_{(ij)(kl)} = \lambda \detla_{ij}\delta_{kl} + \mu (\detla_{ik}\delta_{jl} + \detla_{il}\delta_{jk})
        C.SetZero();
        for (int i = 0; i < 3; ++i)      
            for (int l = 0; l < 3; ++l){ 
                int j = i, k = l;
                C(i+3*j, k+3*l) += lambda;
                k = i, j = l;
                C(i+3*j, k+3*l) += mu;
                C(i+3*j, l+3*k) += mu;
        }
        return Ani::TENSOR_SYMMETRIC;
    };
    auto F_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        F[0] = 0, F[1] = 0, F[2] = 0; 
        return Ani::TENSOR_GENERAL;
    };
    auto U_0 = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ (void) X; (void) dim; (void) user_data; res[0] = res[1] = res[2] = 0.0; return 0;};
    auto P_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        DenseMatrix<> pm(F, 9, 1);
        pm.SetZero();
        double p = 0.001;
        pm(0 + 0*3, 0) = p;
        pm(1 + 1*3, 0) = p;
        pm(2 + 2*3, 0) = p;
        return Ani::TENSOR_GENERAL;
    }; 

    // Define tag to store result
    Tag u = createFemVarTag(m, *dofmap.target<>(), "u");

    // Define user structure to store input data used in local assembler
    struct BndMarker{
        std::array<int, 4> n = {0};
        std::array<int, 6> e = {0};
        std::array<int, 4> f = {0};

        DofT::TetGeomSparsity getSparsity(int type) const {
            DofT::TetGeomSparsity sp;
            for (int i = 0; i < 4; ++i) if (n[i] & type)
                sp.setNode(i);
            for (int i = 0; i < 6; ++i) if (e[i] & type)
                sp.setEdge(i);  
            for (int i = 0; i < 4; ++i) if (f[i] & type)
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
    };

    // define elemental assembler of local matrix and rhs
    std::function<void(const double**, double*, double*, double*, long*, void*, DynMem<double, long>*)> local_assembler =
        [&C_tensor, &F_tensor, &P_tensor, &U_0, order = quad_order, unf, &UFem](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, DynMem<double, long>* fem_alloc) -> void{
        (void) w, (void) iw;
        DenseMatrix<> A(Adat, unf, unf), F(Fdat, unf, 1);
        A.SetZero(); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Gmem = adapt_alloc.alloc(unf, 0, 0);
        DenseMatrix<> G(Gmem.getPlainMemory().ddata, unf, 1);

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        // here we set DfuncTraits<TENSOR_SCALAR, false> because we know that D_tensor is TENSOR_SCALAR on all elements
        // and also we know that it's NOT constant
        // elemental stiffness matrix <grad(P1), C grad(P1)>   
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN);  
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        fem3Dtet<DfuncTraits<TENSOR_SYMMETRIC>>(XYZ, grad_u, grad_u, C_tensor, A, adapt_alloc, order, user_data); 
        // elemental right hand side vector <F, P1>
        fem3Dtet<DfuncTraits<TENSOR_GENERAL>>( XYZ, iden_p0, iden_u, F_tensor, F, adapt_alloc, order, user_data);

        // read labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // apply Neumann BC
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3DfaceN<DfuncTraits<TENSOR_GENERAL>> ( XYZ, k, iden_p0, iden_u, P_tensor, G, adapt_alloc, order, user_data);
                F -= G;
            }
        }

        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity sp = dat.lbl.getSparsity(DIRICHLET_BND);

        //set dirichlet condition
        if (!sp.empty()){
            auto mem = fem_alloc->alloc(unf, 0, 0);
            ArrayView<> dof_values(mem.getPlainMemory().ddata, mem.getPlainMemory().dSize);
            UFem.interpolateByDOFs(XYZ, U_0, dof_values, sp, adapt_alloc, user_data);
            applyDirByDofs(*(UFem.dofMap().target<>()), A, F, sp, ArrayView<const double>(dof_values.data, dof_values.size));
        }  
    };
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel, geom_mask = UFem.dofMap().GetGeomMask() | DofT::FACE](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.lbl.fillFromBndTag(BndLabel, geom_mask, *p.nodes, *p.edges, *p.faces);

        p.compute(args, &data);
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatRHSFunc(GenerateElemMatRhs(local_assembler, unf, unf, 0, 0));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*UFem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();
    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
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
    Solver solver(p.lin_sol_nm, p.lin_sol_prefix);
    solver.SetMatrix(A);
    solver.Solve(b, x);
    print_linear_solver_status(solver, prob_name, true);
    
    //copy result to the tag and save solution
    discr.SaveSolution(x, u);
    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}