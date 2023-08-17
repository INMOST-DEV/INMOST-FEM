/**
 * This program generates and solves a finite element system for the stationary reaction-diffusion problem
 * \f[
 * \begin{aligned}
 *   \frac{\partial u}{\partial t} -\mathrm{div}\ \mathbb{D}\ \mathrm{grad}\ u + 
 *              \mathbf{v} \cdot \mathrm{grad}\ u &= f,\ in\ \Omega\\
 *                                              u &= g,\ on\ \Gamma_1\\
 *                                              u &= 0,\ on\ \Gamma_2\\
 *                                              u|_{t = 0} &= 0,\ in\ \Omega
 * \end{aligned}
 * \f]
 * where Ω = [0,1]^3, Γ_1 = {0}x[0.25,0.75]^2, Γ_2 = ∂Ω \ Γ_1, Theta = [0, T]
 * The user-defined coefficients are
 *    D(x) = diag(1e-4, 1e-4, 1e-4)
 *    v(x) = (1, 0, 0)^T
 *    g(x) = 1
 *    f(x) = 0
 *    T = 0.5
 *
 * @see src/Tutorials/MultiPackage/UnsteadyConDif/main.f in [Ani3d library](https://sourceforge.net/projects/ani3d/)
 */

#include "prob_args.h"

struct InputArgsProblem: public InputArgs{
    double T = 0.5, dt = 0.015;
    uint freq_save = 50;
    bool save_initial = false;

    uint parseArg(int argc, char* argv[], bool print_messages = true) override{
        #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
        else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
        auto get_double = [](const std::string& s) -> double{
            std::istringstream iss(s);
            double f = NAN;
            iss >> f; 
            if (!(iss.eof() && !iss.fail())) 
                throw std::runtime_error("Expected real value, but found = \"" + s + "\"");
            return f;        
        };
        auto get_bool = [](std::string s0)->bool{
            bool res = true;
            std::transform(s0.begin(), s0.end(), s0.begin(), [](char c){ return std::tolower(c); });
            if (s0 == "true") res = true;
            else if (s0 == "false") res = false;
            else {
                int r = stoi(s0);
                if (r == 0) res = false;
                else if (r == 1) res = true;
                else 
                    throw std::runtime_error("Expected bool value? but found = \"" + s0 + "\"");
            }
            return res;
        };
        uint i = 0;
        if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--t_final") == 0){
            GETARG( T = get_double(argv[++i]); )
            return i+1; 
        } else if (strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "--time_step") == 0){
            GETARG( dt = get_double(argv[++i]); )
            return i+1; 
        } else if (strcmp(argv[i], "-fs") == 0 || strcmp(argv[i], "--freq_save") == 0){
            GETARG( freq_save = std::stoi(argv[++i]); )
            return i+1;
        } else if (strcmp(argv[i], "--save_init") == 0){
            GETARG( save_initial = get_bool(argv[++i]); )
            return i+1;
        } else 
            return InputArgs::parseArg(argc, argv, print_messages);
        #undef GETARG
    }
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override{
        out << prefix << "T = " << T << ", dt = " << dt << "\n"
            << prefix << "freq_save = " << freq_save << ", save_initial = " << save_initial << "\n";
        InputArgs::print(out, prefix);
    }

protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override{
        out << prefix << "  -T , --t_final   DVAL    <Set final time, default=\"" << T << "\">\n"
            << prefix << "  -dt, --time_step DVAL    <Set time step, default=\"" << dt << "\">\n"
            << prefix << "  -fs, --freq_save IVAL    <Set frequency of saving results, default=" << freq_save << ">\n"
            << prefix << "       --save_init BVAL    <Should we save first iteration, default=" << save_initial << ">\n";
        InputArgs::printArgsDescr(out, prefix);
    }
};

using namespace INMOST;

int main(int argc, char* argv[]){
    InputArgsProblem p;
    std::string prob_name = "unsteady_conv_dif";
    p.save_prefix = prob_name + "_out"; 
    p.axis_sizes = 12;
    p.parseArgs_mpi(&argc, &argv);

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    std::unique_ptr<Mesh> mptr = GenerateCube(INMOST_MPI_COMM_WORLD, 4*p.axis_sizes, 4*p.axis_sizes, 4*p.axis_sizes);
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
        if (lbl == 0)          
            it->Integer(BndLabel) = 0;
        else if (
             (lbl & (1 << 0)) 
          && abs(it->Coords()[1] - 0.5) < 0.25 + 10*std::numeric_limits<double>::epsilon()
          && abs(it->Coords()[2] - 0.5) < 0.25 + 10*std::numeric_limits<double>::epsilon())
            it->Integer(BndLabel) = 1;
        else 
            it->Integer(BndLabel) = 2;        
    }

    // Define tensors from the problem
    auto D_tensor = [](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) user_data; (void) iTet; (void) X;
        D[0] = 1e-4;  
        return Ani::TENSOR_SCALAR;
    };
    auto V_tensor = [](const Coord<> &X, double *V, TensorDims dims, void *user_data, int iTet) {
        (void) dims; (void) user_data; (void) iTet; (void) X;
        V[0] = 1;
        V[1] = 0;
        V[2] = 0;  
        return Ani::TENSOR_GENERAL;
    };
    auto G_bnd = [](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ 
        (void) X; (void) dim; (void) user_data; 
        res[0] = 1; 
        return 0;
    };
    auto F_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        F[0] = 0;
        return Ani::TENSOR_SCALAR;
    };

    struct ProbLocData{
        //save labels used for apply boundary conditions
        std::array<int, 4> nlbl = {0};
    };

    // function for evaluation of grid Péclet number
    auto eval_Pe = [&V_tensor, &D_tensor](Tetra<const double> XYZ, void* user_data = nullptr)->double{
        double Pe = 0;
        double vc[3] = {0};
        auto t = V_tensor(XYZ.centroid(), vc, {3, 1}, user_data, 0);
        assert(t == TENSOR_GENERAL && "Wrong V tensor");
        if (abs(vc[0]) < 1e-15 && abs(vc[1]) < 1e-15 && abs(vc[2]) < 1e-15)
            return Pe;
        double v2 = vc[0]*vc[0] + vc[1]*vc[1] + vc[2]*vc[2];
        double v_nrm = sqrt(v2);
        double Dc[9] = {0};
        t = D_tensor(XYZ.centroid(), Dc, {3, 3}, user_data, 0);
        double Dv = 0;
        if (t == TENSOR_NULL) Dc[0] = 1;
        switch (t){
            case TENSOR_NULL:
            case TENSOR_SCALAR: Dv = Dc[0]*v2; break;
            case TENSOR_SYMMETRIC:
            case TENSOR_GENERAL: {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        Dv += vc[i]*vc[j]*Dc[i + 3 * j];
                break; 
            }       
        }
        Dv /= v2;
        Pe = XYZ.diameter() * v_nrm / Dv;  
        return Pe;  
    };

    //assemble M
    std::function<void(const double**, double*, void*)> M_assembler = 
     [&V_tensor, &D_tensor, &eval_Pe, order = p.max_quad_order](const double** XY/*[4]*/, double* Mdat, void* user_data) -> void{
        double Bdat[UNF * UNF] = {0};
        DenseMatrix<> M(Mdat, UNF, UNF), B(Bdat, UNF, UNF);
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);

        // elemental mass matrix <P1, P1>
        fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_NULL>>( XYZ, TensorNull<>, M, order );
        
        double Pe = eval_Pe(XYZ, user_data);
        double delta = Pe > 1 ? 0.01 : 0;
        if (delta > 0) {
            // elemental mass matrix <v P1, grad(P1)>
            fem3Dtet<Operator<IDEN, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, true>>( XYZ, V_tensor, B, order );
            B *= delta;
            M += B;
        }
        //Here we do not require to remove Dirichlet parts of the matrix
    };
    //assemble M + 2/3 * dt * A and 2/3 * dt * F
    std::function<void(const double**, double*, double*, void*)> LF_assembler = 
     [&F_tensor, &G_bnd, &V_tensor, &D_tensor, &eval_Pe, order = p.max_quad_order, dt = p.dt](const double** XY/*[4]*/, double* Ldat, double* Fdat, void* user_data) -> void{
        double Bdat[UNF * UNF] = {0}, Mdat[UNF * UNF] = {0}, Adat[UNF * UNF] = {0};
        DenseMatrix<> L(Ldat, UNF, UNF), M(Mdat, UNF, UNF), A(Adat, UNF, UNF), B(Bdat, UNF, UNF);
        double Gdat[UNF] = {0};
        DenseMatrix<> F(Fdat, UNF, 1), G(Gdat, UNF, 1);
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);

        // elemental mass matrix <P1, P1>
        fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_NULL>>( XYZ, TensorNull<>, M, order );
        // elemental mass matrix <D grad(P1), grad(P1)>
        fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SCALAR, true>>( XYZ, D_tensor, A, order );

        // elemental mass matrix <v grad(P1), P1>
        fem3Dtet<Operator<GRAD, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_GENERAL, true>>( XYZ, V_tensor, B, order );
        A += B;

        using P0 = FemFix<FEM_P0>;  
        // elemental mass matrix <F P0, P1> 
        fem3Dtet<Operator<IDEN, P0>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>( XYZ, F_tensor, F, order );

        double Pe = eval_Pe(XYZ, user_data);
        double delta = Pe > 1 ? 0.01 : 0;
        if (delta > 0) {
            for (int i = 0; i < UNF; ++i)
            for (int j = 0; j < UNF; ++j)
                M(i, j) += delta * B(j, i);
            auto VV = [&V_tensor](const Coord<> &X, double *VV, TensorDims dims, void *user_data, int iTet){
                (void) dims;
                double V[3] = {0};
                V_tensor(X, V, {3, 1}, user_data, iTet);
                for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    VV[i + 3*j] = V[i]*V[j];
                return TENSOR_SYMMETRIC;    
            };
            // elemental mass matrix <v grad(P1), v grad(P1)>
            fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SYMMETRIC, true>>( XYZ, VV, B, order );
            B *= delta;
            A += B;

            auto FV_tensor = [&F_tensor, &V_tensor](const Coord<> &X, double *FV, TensorDims dims, void *user_data, int iTet){
                (void) dims;
                V_tensor(X, FV, {3, 1}, user_data, iTet);
                double f = 0;
                F_tensor(X, &f, {1, 1}, user_data, iTet);
                for (int  i = 0; i < 3; ++i)
                    FV[i] *= f;
                return TENSOR_GENERAL;    
            };
            fem3Dtet<Operator<IDEN, P0>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, true>>( XYZ, FV_tensor, G, order );
            G *= delta;
            F += G;
        }
        F *= 2 * dt / 3;

        double Dt = 2 * dt / 3;
        for (int j = 0; j < UNF; ++j)
        for (int i = 0; i < UNF; ++i)
            L(i, j) = M(i, j) + Dt * A(i, j);

        auto& dat = *static_cast<ProbLocData*>(user_data);    
        DofT::TetGeomSparsity sp_1, sp_2;
        DofT::TetGeomSparsity* spp[2] = {&sp_1, &sp_2};
        for (int i = 0; i < 4; ++i) if (dat.nlbl[i] > 0)
            spp[dat.nlbl[i]-1]->setNode(i);
        //set dirichlet condition    
        if (!sp_1.empty() || !sp_2.empty()){
            std::array<double, UNF> dof_values;
            interpolateByDOFs<UFem>(XYZ, G_bnd, ArrayView<>(dof_values.data(), UNF), sp_1);
            interpolateConstant<UFem>(0.0, ArrayView<>(dof_values.data(), UNF), sp_2);
            applyDirByDofs(Dof<UFem>::Map(), L, F, sp_1|sp_2, ArrayView<const double>(dof_values.data(), UNF));
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
    discr.SetMatRHSFunc(GenerateElemMatRhs(LF_assembler, UNF, UNF));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper<UFem>();
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();
    if (pRank == 0) 
        std::cout << "#dofs = " << discr.m_enum.getMatrixSize() << std::endl; 

    //prepare work matrices and vectors
    Sparse::Matrix L("L"), M("M");
    Sparse::Vector U[3]{Sparse::Vector("U0"), Sparse::Vector("U1"), Sparse::Vector("U2")}, F("F"), b("b");
    discr.Assemble(L, F);
    discr.SetMatFunc(GenerateElemMat(M_assembler, UNF, UNF));
    discr.AssembleMatrix(M);
    int vbeg = discr.getBegInd(), vend = discr.getEndInd();
    for (int k = 0; k < 3; ++k)
        U[k].SetInterval(vbeg, vend);
    b.SetInterval(vbeg, vend);
    // Define tag to save results
    Tag u = m->CreateTag("u", DATA_REAL, NODE, NONE, 1);
    
    //set initial values
    for (auto n = m->BeginNode(); n != m->EndNode(); ++n) if (n->GetStatus() != Element::Ghost){
        auto lbl = n->Integer(BndLabel);
        auto rx = n->Coords();
        std::array<double, 3> X{ rx[0], rx[1], rx[2] };
        double val = 0;
        if (lbl == 1)
            G_bnd(X, &val, 1, nullptr);
        IGlobEnumeration::NaturalIndex nid(n->getAsElement(), DofT::NODE, 0, 0, 0);   
        auto vec_id = discr.m_enum.OrderR(nid);
        U[0][vec_id.id] = U[1][vec_id.id] = U[2][vec_id.id] = val;
    }

    //function to set boundary value
    auto project_bc = [&](int k){
        for (auto n = m->BeginNode(); n != m->EndNode(); ++n) if (n->GetStatus() != Element::Ghost){
            auto lbl = n->Integer(BndLabel);
            if (lbl == 0) continue;
            auto rx = n->Coords();
            std::array<double, 3> X{ rx[0], rx[1], rx[2] };
            double val = 0;
            if (lbl == 1)
                G_bnd(X, &val, 1, nullptr);
            IGlobEnumeration::NaturalIndex nid(n->getAsElement(), DofT::NODE, 0, 0, 0);   
            auto vec_id = discr.m_enum.OrderR(nid);
            U[k][vec_id.id] = val;    
        }
    };

    //setup linear solver and solve assembled system
    Solver solver(p.lin_sol_nm, p.lin_sol_prefix);
    solver.SetMatrix(L);
    if (pRank == 0)
        std::cout << "\tsolver: preconding = " << solver.PreconditionerTime() << "s" << std::endl;
    Solver::OrderInfo info;
    info.PrepareMatrix(M, 0);
    if (pRank == 0)
        std::cout << "Prepare matrix" << std::endl;

    //time loop
    double t = 0;
    int k = -1;
    while (t < p.T){
        k++;
        if (k % p.freq_save == 0 && (k > 0 || p.save_initial)){
            discr.SaveSolution(U[(k+2)%3], u);
            if (pRank == 0)
                std::cout << "Save vector into tag" << std::endl;
            m->Save(p.save_dir + p.save_prefix + "_" + std::to_string(t) + ".pvtu");
            if (pRank == 0)
                std::cout << "Save solution into \"" << (p.save_dir + p.save_prefix + "_" + std::to_string(t) + ".pvtu") << "\" file" << std::endl;
        }
        double dt = p.dt;
        bool same_dt = true;
        if (p.T - t < dt){
            dt = p.T - t;
            same_dt = false;
        }
        t += dt;
        if (pRank == 0)
            std::cout << "Time step " << k << ", time = " << t << std::endl;
        for (auto i = vbeg; i < vend; ++i)
            b[i] = 4.0/3 * U[(k+2)%3][i] - 1.0/3 * U[(k+1)%3][i];
        info.PrepareVector(b);
        info.Update(b);
        M.MatVec(1, b, 0, U[k%3]);
        info.RestoreVector(b);
        for (auto i = vbeg; i < vend; ++i){
            b[i] = U[k%3][i] + F[i];
            U[k%3][i] = U[(k+2)%3][i];
        }
        solver.Solve(b, U[k%3]);
        if (pRank == 0)
            std::cout << "\tsolver: #lits " << solver.Iterations() << " residual " << solver.Residual() << ". "
                      << "solving = " << solver.IterationsTime() << "s" << std::endl;
        project_bc(k%3);

        if (same_dt == false) {
            double rt = dt / p.dt;
            double w0 = rt * (rt + 1) / 2,
                   w1 = (1 - rt)*(1 + rt),
                   w2 = -(1 - rt) * rt / 2;
            for (auto i = vbeg; i < vend; ++i)
                U[k%3][i] = w0 * U[k%3][i] + w1 * U[(k+2)%3][i] + w2 * U[(k+1)%3][i];     
            break;
        }
    }
    info.RestoreMatrix(M);

    discr.SaveSolution(U[k%3], u);
    if (pRank == 0)
        std::cout << "Save vector into tag" << std::endl;
    m->Save(p.save_dir + p.save_prefix + "_" + std::to_string(p.T) + ".pvtu");
    if (pRank == 0)
        std::cout << "Save solution into \"" << (p.save_dir + p.save_prefix + "_" + std::to_string(p.T) + ".pvtu") << "\" file" << std::endl;

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}