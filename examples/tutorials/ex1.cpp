//
// Created by Liogky Alexey on 05.04.2022.
//

/**
 * Here we show the simplest way to write code used the P_1 templated space, 
 * the code essentially uses the structure of P_1 space and does not allow you to change the space without writing additional code 
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

#include "cmd_ex1_3.h"

using namespace INMOST;

int main(int argc, char* argv[]){
    InputArgs p;
    std::string prob_name = "ex1";
    p.save_prefix = prob_name + "_out"; 
    p.parseArgs_mpi(&argc, &argv);

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    std::unique_ptr<Mesh> mptr = GenerateCube(INMOST_MPI_COMM_WORLD, p.axis_sizes[0], p.axis_sizes[1], p.axis_sizes[2]);
    Mesh* m = mptr.get();
    // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    m->ExchangeGhost(1,NODE);
    print_mesh_sizes(m);

    using namespace Ani;
    //specify used templated FEM space
    using UFem = FemFix<FEM_P1>;
    constexpr auto UNF = Operator<IDEN, UFem>::Nfa::value;
    // Set boundary labels
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, NODE, NONE, 1);
    for (auto it = m->BeginNode(); it != m->EndNode(); ++it){
        int lbl = 0;
        for (auto k = 0; k < 3; ++k)
            if (abs(it->Coords()[k] - 0) < 10*std::numeric_limits<double>::epsilon())
                lbl |= (1 << (2*k+0));
            else if (abs(it->Coords()[k] - 1) < 10*std::numeric_limits<double>::epsilon()) 
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
    auto U_0 = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ (void) X; (void) dim; (void) user_data; res[0] = 0.0; return 0; }; 

    // Define tag to store result
    Tag u = m->CreateTag("u", DATA_REAL, NODE, NONE, 1);

    // Define user structure to store input data used in local assembler
    struct ProbLocData{
        //save labels used for apply boundary conditions
        std::array<int, 4> nlbl = {0};
    };

    // define elemental assembler of local matrix and rhs
    std::function<void(const double**, double*, double*, void*)> local_assembler =
            [&D_tensor, &F_tensor, &U_0](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data) -> void{
        DenseMatrix<> A(Adat, UNF, UNF), F(Fdat, UNF, 1);
        A.SetZero(); F.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        //here we set DfuncTraits<TENSOR_SCALAR, false> because we know that D_tensor is TENSOR_SCALAR on all elements
        // and also we know that it's not constant
        // elemental stiffness matrix <grad(P1), D grad(P1)>
        fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SCALAR, false>>(XYZ, D_tensor, A, 2);

        // elemental right hand side vector <F, P1>
        fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(XYZ, F_tensor, F, 2);

        // read node labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // impose dirichlet condition
        for (int i = 0; i < 4; ++i)
            if (dat.nlbl[i] > 0) {
                double bc = 0;
                U_0(std::array<double, 3>{XY[i][0], XY[i][1], XY[i][2]}, &bc, 1, nullptr);
                applyDir(A, F, i, bc); // set Dirichlet BC
            }
    };
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        std::fill(data.nlbl.begin(), data.nlbl.end(), 0);
        for (unsigned i = 0; i < data.nlbl.size(); ++i) 
            data.nlbl[i] = (*p.nodes)[i].Integer(BndLabel);

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
    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
    //initialization of assembled matrix and rhs is optional
    A.SetInterval(discr.getBegInd(), discr.getEndInd()); //initialized as free matrix
    b.SetInterval(discr.getBegInd(), discr.getEndInd()); //initialized by zeros
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
    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    discr.Clear();
    mptr.reset();
    InmostFinalize();
}
