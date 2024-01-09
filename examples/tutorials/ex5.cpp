//
// Created by Liogky Alexey on 05.01.2024.
//

/**
 * Here we show how to code FEM problem for runtime finite element spaces with thread-safe allocator
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

using namespace INMOST;

int main(int argc, char* argv[]){
    InputArgs1Var p;
    std::string prob_name = "ex5";
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
    FemSpace UFem = choose_space_from_name(p.USpace);
    uint unf = UFem.dofMap().NumDofOnTet();
    auto& dofmap = UFem.dofMap();
    auto mask = GeomMaskToInmostElementType(dofmap.GetGeomMask()); 
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
    auto D_tensor = [](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) user_data; (void) iTet;
        D[0] = 1 + X[0]*X[0];  
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
    };

    std::function<void(const double**, double*, double*, double*, long*, void*, DynMem<double, long>*)> local_assembler =
        [&D_tensor, &F_tensor, &U_0, order = quad_order, unf, &UFem](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, DynMem<double, long>* fem_alloc) -> void{
        (void) w, (void) iw;
        DenseMatrix<> A(Adat, unf, unf), F(Fdat, unf, 1);
        A.SetZero(); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        // here we set DfuncTraits<TENSOR_SCALAR, false> because we know that D_tensor is TENSOR_SCALAR on all elements
        // and also we know that it's NOT constant
        // elemental stiffness matrix <grad(P1), D grad(P1)>   
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN);  
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        fem3Dtet<DfuncTraits<TENSOR_SCALAR, false>>(XYZ, grad_u, grad_u, D_tensor, A, adapt_alloc, order, user_data); 
        // elemental right hand side vector <F, P1>
        fem3Dtet<DfuncTraits<TENSOR_SCALAR, true>>( XYZ, iden_p0, iden_u, F_tensor, F, adapt_alloc, order, user_data);

        // read node labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity sp = dat.lbl.getSparsity();

        //set dirichlet condition
        if (!sp.empty()){
            auto mem = fem_alloc->alloc(unf, 0, 0);
            ArrayView<> dof_values(mem.getPlainMemory().ddata, mem.getPlainMemory().dSize);
            UFem.interpolateByDOFs(XYZ, U_0, dof_values, sp, adapt_alloc, user_data);
            applyDirByDofs(*(UFem.dofMap().target<>()), A, F, sp, ArrayView<const double>(dof_values.data, dof_values.size));
        }  
    };
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel, geom_mask = UFem.dofMap().GetGeomMask()](ElementalAssembler& p) -> void{
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
    discr.SaveVar(x, 0, u);
    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}