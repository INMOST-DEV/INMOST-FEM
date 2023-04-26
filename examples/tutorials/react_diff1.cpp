/**
 * This program generates and solves a finite element system for the stationary reaction-diffusion problem
 * \f[
 * \begin{aligned}
 *  -\mathrm{div}\ \mathbf{K}\ &\mathrm{grad}\ u\ + &\mathbf{A} u &= \mathbf{F}\ in\  \Omega  \\
 *                             &                    &           u &= U_0\        on\  \Gamma_D\\
 *            &\mathbf{K} \frac{du}{d\mathbf{n}}    &             &= G_0\        on\  \Gamma_N\\
 *            &\mathbf{K} \frac{du}{d\mathbf{n}}\ + &\mathbf{S} u &= G_1\        on\  \Gamma_R\\
 * \end{aligned}
 * \f]
 * where Ω = [0,1]^3, Γ_D = {0}x[0,1]^2, Γ_R = {1}x[0,1]^2, Γ_N = ∂Ω \ (Γ_D ⋃ Γ_R)
 * The user-defined coefficients are
 *    K(x)   - positive definite tensor
 *    A(x)   - non-negative reaction
 *    F(x)   - right-hand side
 *    U_0(x) - essential (Dirichlet) boundary condition
 *    G_0(x) - Neumann boundary condition
 *    G_1(x) - Robin boundary condition
 *    S(x)   - Robin boundary coefficient
 *
 * @see src/Tutorials/PackageFEM/main_bc.f in [Ani3d library](https://sourceforge.net/projects/ani3d/)
 */

#include "prob_args.h"

using namespace INMOST;

int main(int argc, char* argv[]){
    InputArgs p;
    std::string prob_name = "react_diff1";
    p.save_prefix = prob_name + "_out";
    p.parseArgs_mpi(&argc, &argv);

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    std::unique_ptr<Mesh> mptr = GenerateCube(INMOST_MPI_COMM_WORLD, p.axis_sizes, p.axis_sizes, p.axis_sizes);
    Mesh* m = mptr.get();
    // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    m->ExchangeGhost(1,NODE);
    print_mesh_sizes(m);

    using namespace Ani;
    //specify used templated FEM space
    using UFem = FemFix<FEM_P1>; //you can replace FEM_P2 on any 1D space and the code will remain correct
    constexpr auto UNF = Operator<IDEN, UFem>::Nfa::value;
    auto dofmap = Dof<UFem>::Map();
    auto mask = AniGeomMaskToInmostElementType(dofmap.GetGeomMask()) | FACE; 
    // Set boundary labels
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, mask, NONE, 1);
    for (auto it = m->BeginElement(mask); it != m->EndElement(); ++it){
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
    auto K_tensor = [](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) user_data; (void) iTet;    
        Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
        K.SetZero();
        K(0, 0) = K(1, 1) = K(2, 2) = 1;
        K(0, 1) = K(1, 0) = -1;
        return Ani::TENSOR_SYMMETRIC;
    };
    auto F_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        double f = X[0] + X[1] + X[2];
        F[0] = f*f - 2;
        return Ani::TENSOR_SCALAR;
    };
    auto A_tensor = [](const Coord<> &X, double *A, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        A[0] = 1;
        return Ani::TENSOR_SCALAR;
    };
    auto S_tensor = [](const Coord<> &X, double *S, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        S[0] = 1;
        return Ani::TENSOR_SCALAR;
    };
    auto U_0 = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ 
        (void) dim; (void) user_data; 
        double f = X[0] + X[1] + X[2];
        res[0] = f * f; 
        return 0;
    }; 
    auto G_0 = [](const Coord<> &X, double *g0, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) user_data; (void) iTet;
        double f = X[0] + X[1] + X[2];
        g0[0] = -2*f;
        return Ani::TENSOR_SCALAR;
    };
    auto G_1 = [](const Coord<> &X, double *g1, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) user_data; (void) iTet;
        double f = X[0] + X[1] + X[2];
        g1[0] = f*(f+2);
        return Ani::TENSOR_SCALAR;
    };

    // Define user structure to store input data used in local assembler
    struct ProbLocData{
        //save labels used for apply boundary conditions
        std::array<int, 4> nlbl = {0};
        std::array<int, 6> elbl = {0};
        std::array<int, 4> flbl = {0};
    };

    // define elemental assembler of local matrix and rhs
    std::function<void(const double**, double*, double*, void*)> local_assembler =
            [&K_tensor, &F_tensor, &A_tensor, &S_tensor, &U_0, &G_0, &G_1, order = p.max_quad_order](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data) -> void{
        double Bdat[UNF * UNF] = {0};
        DenseMatrix<> A(Adat, UNF, UNF), F(Fdat, UNF, 1), B(Bdat, UNF, UNF), G(Bdat, UNF, 1);
        A.SetZero(); F.SetZero(); B.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        
        // elemental stiffness matrix <K grad(P1), grad(P1)>
        fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SYMMETRIC, true>>( XYZ, K_tensor, A, order );
        // elemental mass matrix <A P1, P1>
        fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>( XYZ, A_tensor, B, order );
        A += B;

        // elemental right hand side vector <F, P1>
        fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>>(XYZ, F_tensor, F, order);

        // read user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // apply Neumann and Robin BC
        for (int k = 0; k < 4; ++k){
            if (dat.flbl[k] == (1 << 4)){ //Neumann BC
                fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>> ( XYZ, k, G_0, G, order );
                F += G;
            } else if (dat.flbl[k] == (1 << 5)) { //Robin BC
                fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>> ( XYZ, k, G_1, G, order );
                F += G;
                fem3Dface<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>>( XYZ, k, S_tensor, B, order );
                A += B;
            }
        }

        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity sp;
        for (int i = 0; i < 4; ++i) if (dat.nlbl[i] & (1|2))
            sp.setNode(i);
        for (int i = 0; i < 6; ++i) if (dat.elbl[i] & (1|2))
            sp.setEdge(i);  
        for (int i = 0; i < 4; ++i) if (dat.flbl[i] & (1|2))
            sp.setFace(i);
        
        //set dirichlet condition
        if (!sp.empty()){
            std::array<double, UNF> dof_values;
            interpolateByDOFs<UFem>(XYZ, U_0, ArrayView<>(dof_values.data(), UNF), sp);
            applyDirByDofs(Dof<UFem>::Map(), A, F, sp, ArrayView<const double>(dof_values.data(), UNF));
        }
    };
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        auto geom_mask = Dof<UFem>::Map().GetGeomMask();
        if (geom_mask & DofT::NODE){
            for (unsigned i = 0; i < data.nlbl.size(); ++i) 
                data.nlbl[i] = (*p.nodes)[i].Integer(BndLabel);
        }
        if (geom_mask & DofT::EDGE){
            for (unsigned i = 0; i < data.elbl.size(); ++i) 
                data.elbl[i] = (*p.edges)[p.local_edge_index[i]].Integer(BndLabel);
        }
        {//required for Neumann and Robin BC
            for (unsigned i = 0; i < data.flbl.size(); ++i) 
                data.flbl[i] = (*p.faces)[p.local_face_index[i]].Integer(BndLabel);
        }

        p.compute(args, &data);
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatRHSFunc(GenerateElemMatRhs(local_assembler, UNF, UNF));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper<UFem>();
        FemExprDescr fed;
        fed.PushVar(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();

    // Define tag to store result
    Tag u = createFemVarTag(m, dofmap, "u");

    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
    //assemble matrix and right-hand side
    discr.Assemble(A, b);
    // Print Assembler timers
    if (pRank == 0) {
        std::cout << "#dofs = " << discr.m_enum->MatrSize << std::endl; 
        std::cout << "Assembler timers:"
                  << "\n\tInit assembler: " << discr.GetTimeInitAssembleData() << "s"
                  << "\n\tInit guess    : " << discr.GetTimeInitValSet() << "s"
                  << "\n\tInit handler  : " << discr.GetTimeInitUserHandler() << "s"
                  << "\n\tLocal evaluate: " << discr.GetTimeEvalLocFunc() << "s"
                  << "\n\tLocal postproc: " << discr.GetTimePostProcUserHandler() << "s"
                  << "\n\tComp glob inds: " << discr.GetTimeFillMapTemplate() << "s"
                  << "\n\tGlobal remap  : " << discr.GetTimeFillGlobalStructs() << "s"
                  << "\n\tTotal         : " << discr.GetTimeTotal() << "s" << std::endl;
    }

    //setup linear solver and solve assembled system
    Solver solver(p.lin_sol_nm, p.lin_sol_prefix);
    solver.SetMatrix(A);
    solver.Solve(b, x);
    print_linear_solver_status(solver, prob_name, true);
    
    //copy result to the tag and save solution
    discr.SaveVar(x, 0, u);

    //Compare result with analytical solution
    auto U_analytic = [](const Coord<> &X)->double{ return (X[0] + X[1] + X[2])*(X[0] + X[1] + X[2]); };
    auto U_eval = [&discr, &u](const Cell& c, const Coord<> &X)->double{
        std::array<double, UNF> dofs;
        discr.GatherDataOnElement(u, c, dofs.data());
        auto nds = c.getNodes();
        reorderNodesOnTetrahedron(nds);
        double XY[4][3] = {0};
        for (int ni = 0; ni < 4; ++ni)
            for (int k = 0; k < 3; ++k)
                XY[ni][k] = nds[ni].Coords()[k];
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);        
        
        double val = 0;
        fem3DapplyX<Operator<IDEN, UFem>>( XYZ, ArrayView<const double>(X.data(), 3), ArrayView<>(dofs.data(), UNF), ArrayView<>(&val,1) );
        return val;
    };
    auto err = [&U_analytic, &U_eval](const Cell& c, const Coord<> &X)->double{
        double Ua = U_analytic(X), Ue = U_eval(c, X);
        return (Ua - Ue)*(Ua - Ue);
    };
    double L2nrm = sqrt(integrate_scalar_func(m, err, 4));
    if (pRank == 0)
        std::cout << "||U_eval - U_exct||_L2 / ||U_exct||_L2 = " << L2nrm / sqrt(8.6) << "\n";


    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}

