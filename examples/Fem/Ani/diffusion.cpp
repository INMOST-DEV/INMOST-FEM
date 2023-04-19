//
// Created by Liogky Alexey on 05.04.2022.
//

#include "inmost.h"
#include "example_common.h"
#include "carnum/MeshGen/mesh_gen.h"
#include "carnum/Fem/AniInterface/Fem3Dtet.h"
#include "carnum/Fem/AniInterface/ForAssembler.h"
#include "carnum/Fem/Assembler.h"
#include <numeric>

using namespace INMOST;

void process_command_line_arguments(int argc, char* argv[], std::array<double, 3>& axis_sizes, std::string& save_dir) {
    auto print_help_message = [](std::string prefix = "", bool is_exit = false) {
        std::cout << prefix << "Help message: " << "\n";
        std::cout << prefix << " Command line options:" << "\n";
        std::cout << prefix << "  -sz, --sizes        <Axis sizes for generated mesh: nx, ny, nz [default: 8 8 8]>"
                  << "\n";
        std::cout << prefix << "  -t,  --target       <Directory to save solution [default: \"./\"]>" << "\n";
        std::cout << prefix << "  -h,  --help         <Print this message and cancel>" << "\n";
        std::cout << std::endl;
        if (is_exit) exit(0);
    };
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-sz" || std::string(argv[i]) == "--sizes") {
            for (unsigned j = 0; j < axis_sizes.size() &&
                            (i + 1 < argc && (argv[i + 1][0] != '-' || std::isdigit(argv[i + 1][1]))); ++j) {
                try {
                    axis_sizes[j] = std::stod(argv[++i]);
                } catch (std::exception &e) {
                    std::cout << "Waited " << "axis_sizes" << "[" << j << "] but error happens: '" << e.what()
                              << "'\n";
                    --i;
                }
            }
            continue;
        }
        if (std::string(argv[i]) == "-t" || std::string(argv[i]) == "--target") {
            if (i + 1 < argc) {
                save_dir = argv[++i];
                if (save_dir.back() != '/') save_dir += "/";
            }
            continue;
        }
        if (true) {
            std::cerr << "Faced unparsed value: \"" + std::string(argv[i]) + "\"" << std::endl;
            print_help_message("", true);
        }
    }
}

/**
 * This program generates and solves a finite element system for the diffusion problem
 * \f[
 * \begin{aligned}
 *   -\mathrm{div}\ \mathbf{K}\ &\mathrm{grad}\ u\ + &\mathbf{A} u &= \mathbf{F}\ in\  \Omega  \\
 *                             &                    &           u &= U_0\        on\  \Gamma_D\\
 *            &\mathbf{K} \frac{du}{d\mathbf{n}}    &             &= G_0\        on\  \Gamma_N\\
 *            &\mathbf{K} \frac{du}{d\mathbf{n}}\ + &\mathbf{S} u &= G_1\        on\  \Gamma_R\\
 * \end{aligned}
 * \f]
 * where Ω = [0,1]^3, Γ_D = {0}x[0,1]^2, Γ_R = {1}x[0,1]^2, Gamma_N = ∂Ω \ (Γ_D ⋃ Γ_R)
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
int main(int argc, char* argv[]){
    int pRank = 0, pCount = 0;
    InmostInit(&argc, &argv, "", pRank, pCount);
    std::array<double, 3> axis_sizes{8, 8, 8};
    std::string save_name = "solution.pvtk", save_dir = "";
    
    process_command_line_arguments(argc, argv, axis_sizes, save_dir);

    std::unique_ptr<Mesh> mptr = GenerateCube(INMOST_MPI_COMM_WORLD, axis_sizes[0], axis_sizes[1], axis_sizes[2]);
    Mesh* m = mptr.get();
    // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    m->ExchangeGhost(1,NODE);
    {
        long nN = m->TotalNumberOf(NODE), nE = m->TotalNumberOf(EDGE), nF = m->TotalNumberOf(FACE), nC = m->TotalNumberOf(CELL);
        if (pRank == 0) {
            std::cout << "Mesh info:"
                      << " #N " << nN
                      << " #E " << nE
                      << " #F " << nF
                      << " #T " << nC << std::endl;
        }
    }

    // Set boundary labels
    Tag BndLabel = m->CreateTag("Label", DATA_INTEGER, NODE|FACE, NONE, 1);
    for (auto it = m->BeginNode(); it != m->EndNode(); ++it) {
        it->Integer(BndLabel) = 0;
        if (!it->Boundary()) continue;
    }
    for (auto it = m->BeginFace(); it != m->EndFace(); ++it){
        auto nodes = it->getNodes();
        it->Integer(BndLabel) = 0;
        if (!it->Boundary()) continue;
        int r[6] = {0};
        for (unsigned n = 0; n < nodes.size(); ++n)
            for (int i = 0; i < 3; ++i) {
                if (fabs(nodes[n]->Coords()[i] - 0.0) < 10*DBL_EPSILON) r[i]++;
                else if (fabs(nodes[n]->Coords()[i] - 1.0) < 10*DBL_EPSILON) r[i+3]++;
            }
        for (int i = 0; i < 6; ++i)
            if (r[i] == 3) {
                if (i == 0)
                    it->Integer(BndLabel) = (1 << 0);   // Dirichelt boundary
                else if (i == 3)
                    it->Integer(BndLabel) = (1 << 2);   // Robin boundary
                else
                    it->Integer(BndLabel) = (1 << 1);   // Neumann boundary
            }
        for (unsigned n = 0; n < nodes.size(); ++n)
            nodes[n].Integer(BndLabel) |= it->Integer(BndLabel);
    }


    using namespace Ani;
    // Define tensors from the problem
    auto K_tensor =
            [](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
                (void) X; (void) user_data; (void) iTet;    
                Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
                K.SetZero();
                K(0, 0) = K(1, 1) = K(2, 2) = 1;
                K(0, 1) = K(1, 0) = -1;
                return Ani::TENSOR_SYMMETRIC;
        };
    auto A_tensor =
            [](const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet) {
                (void) X; (void) dims; (void) user_data; (void) iTet;
                dat[0] = 1;
                return Ani::TENSOR_SCALAR;
            };
    auto F_tensor =
            [](const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet) {
                (void) X; (void) dims; (void) user_data; (void) iTet;
                dat[0] = 1;
                return Ani::TENSOR_SCALAR;
            };
    auto U_0 = [](const Coord<> &X){
        return X[0] + X[1] + X[2];
    };
    auto G_0 = [](const Coord<> &X){
        return X[0] - X[1];
    };
    auto G_1 = [](const Coord<> &X){
        return X[0] + X[2];
    };
    auto S_tensor =
            [](const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet) {
                (void) X; (void) dims; (void) user_data; (void) iTet;
                dat[0] = 1;
                return Ani::TENSOR_SCALAR;
            };
    // structure to create tensors from G_0, G_1
    auto tensor_wrap = [](std::function<double(const std::array<double, 3> &X)> f){
        return [f](const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet){
            (void) dims; (void) user_data; (void) iTet;
            dat[0] = f(X);
            return Ani::TENSOR_SCALAR;
        };
    };
    // User structure for local assembler
    struct NodeLabelsData{
        std::array<int, 4> nlbl;
        std::array<int, 4> flbl;
    };

    std::function<void(const double**, double*, double*, void*)> local_assembler =
            [&K_tensor, &A_tensor, &F_tensor, &U_0, &G_0, &G_1, S_tensor, tensor_wrap](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data) -> void{
        DenseMatrix<> A(Adat, 4, 4), F(Fdat, 4, 1);
        A.SetZero(); F.SetZero();
        const DenseMatrix<const double>
                                    XY0(XY[0], 3, 1),
                                    XY1(XY[1], 3, 1),
                                    XY2(XY[2], 3, 1),
                                    XY3(XY[3], 3, 1);
        //here we set DfuncTraits<TENSOR_SYMMETRIC, true> because we know that K_tensor is TENSOR_SYMMETRIC on all elements
        // and also we know that it's constant
        // elemental stiffness matrix <grad(P1), K grad(P1)>
        fem3Dtet<Operator<GRAD, FemFix<FEM_P1>>, Operator<GRAD, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SYMMETRIC, true>>(
                XY0, XY1, XY2, XY3, K_tensor, A, 2
                );
        double Bdat[4*4];
        DenseMatrix<> B(Bdat, 4, 4);
        // elemental mass matrix <P1, A P1>
        fem3Dtet<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR, true>>(
                XY0, XY1, XY2, XY3, A_tensor, B, 2
        );
        for (int i = 0; i < 16; ++i) A.data[i] += B.data[i];

        // elemental right hand side vector <F, P1>
        fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR, true>>(
                XY0, XY1, XY2, XY3, F_tensor, F, 2
        );

        // read node labels from user_data
        auto& udat = *static_cast<NodeLabelsData*>(user_data);
        auto& nlbl = udat.nlbl;
        auto& flbl = udat.flbl;

        DenseMatrix<> G(Bdat, 4, 1);
        for (int k = 0; k < 4; ++k){
            switch (flbl[k]) {
                // impose Neumann BC
                case (1 << 1): {
                    fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR>> (
                              XY0, XY1, XY2, XY3, k, tensor_wrap(G_0), G, 2
                    );
                    for (int i = 0; i < 4; ++i) F.data[i] += G.data[i];
                    break;
                }
                // impose Robin BC
                case (1 << 2): {
                    fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR>> (
                            XY0, XY1, XY2, XY3, k, tensor_wrap(G_1), G, 2
                    );
                    for (int i = 0; i < 4; ++i) F.data[i] += G.data[i];
                    fem3Dface<Operator<IDEN, FemFix<FEM_P1>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR>> (
                            XY0, XY1, XY2, XY3, k, S_tensor, B, 3
                    );
                    for (int i = 0; i < 16; ++i) A.data[i] += B.data[i];
                    break;
                }
            }
        }

        // impose dirichlet condition
        for (int i = 0; i < 4; ++i)
            if (nlbl[i] & 1) {
                std::array<double, 3> x;
                DOF_coord<FemFix<FEM_P1>>::at(i, XY[0], x.data()); // compute coordinate of i-th degree of freedom
                applyDir(A, F, i, U_0(x));  // set Dirichlet BC
            }
    };
    auto local_data_gatherer = [&BndLabel](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        NodeLabelsData data;
        for (int i = 0; i < 4; ++i) {
            data.nlbl[i] = (*p.nodes)[i].Integer(BndLabel);
            data.flbl[i] = (*p.faces)[p.local_face_index[i]].Integer(BndLabel);
        }
        p.compute(args, &data);
    };

    Assembler discr(m);
    discr.SetMatRHSFunc(GenerateElemMatRhs(local_assembler, 4, 4));
    {
        auto Var0Helper = GenerateHelper<FemFix<FEM_P1>>();
        FemExprDescr fed;
        fed.PushVar(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();
    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
    A.SetInterval(discr.getBegInd(), discr.getEndInd()); //initialized as free matrix
    b.SetInterval(discr.getBegInd(), discr.getEndInd()); //initialized by zeros
    discr.Assemble(A, b);
    // Print Assembler timers
    if (pRank == 0) {
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

    Solver solver(Solver::INNER_MPTILUC);
    solver.SetMatrix(A);
    bool success_solve = solver.Solve(b, x);
    if(!success_solve){
        auto& s = solver;
        if (pRank == 0) std::cout << "diffusion" << ": solution failed\n";
        for (int p = 0; p < pCount; ++p) {
            BARRIER;
            if (pRank != p) continue;
            std::cout << "\tIterations " << s.Iterations() << " Residual " << s.Residual() << ". "
                      << "Precond time = " << s.PreconditionerTime() << ", solve time = " << s.IterationsTime() << std::endl;
            std::cout << "\tRank " << pRank << " failed to solve system. ";
            std::cout << "Reason: " << s.GetReason() << std::endl;
        }
        exit(-1);
    }
    else{
        auto& s = solver;
        if(pRank == 0) {
            std::cout << "diffusion" << ":\n";
            std::string _s_its = std::to_string(s.Iterations()), s_its = "    ";
            std::copy(_s_its.begin(), _s_its.end(), s_its.begin() + 3 - _s_its.size());

            std::cout << "\tsolved_succesful: #lits " << s_its << " residual " << s.Residual() << ", "
                      << "preconding = " << s.PreconditionerTime() << "s, solving = " << s.IterationsTime() << "s" << std::endl;
        }
    }

    // Create tag to store solution
    Tag Sol = m->CreateTag("u", DATA_REAL, NODE, NONE, 1);
    discr.SaveVar(x, 0, Sol);
    m->Save(save_dir + save_name);

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}
