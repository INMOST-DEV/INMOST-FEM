// Created by Liogky Alexey on 09.02.2024.
//

/**
 * This program generates  and solves a finite element system for the stationary nonlinear incompressible elasticity problem
 *
 * \f[
 * \begin{aligned}
 *   \mathrm{div}\ \mathbb{P}  + f &= 0\    in\  \Omega  \\
 *                           J - 1 &= 0\    in\  \Omega  \\
 *          \mathbf{u}             &= \mathbf{u}_0\  on\  \Gamma_D\\
 *     \mathbb{P} \cdot \mathbf{N} &= -p_{ext}\ \mathrm{adj}\ \mathbb{F}^T \mathbf{N}\ on\  \Gamma_P\\  
 *     \mathbb{P} \cdot \mathbf{N} &= 0\    on\  \Gamma_0\\
 *                      \mathbb{P} &= \mathbb{F} \cdot \mathbb{S} - p \mathrm{adj}\ \mathbb{F}^T\\ 
 *   \mathbb{F}_{ij} &= \mathbb{I}_{ij} + \mathrm{grad}_j \mathbf{u}_i,\ J = \mathrm{det }\mathbb{F}\\
 *        \mathbb{S} &= \lambda\ \mathrm{tr}\ \mathbb{E}\ \mathbb{I} + 2 \mu \mathbb{E} \\
 *        \mathbb{E} &= \frac{\mathbb{F}^T \cdot \mathbb{F}  - \mathbb{I} } {2} \\
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
 *  p_{ext}(x)   = 0.001         - pressure on bottom part
 * 
 */

#include "prob_args.h"
#include "anifem++/autodiff/cauchy_strain_autodiff.h"

using namespace INMOST;

struct InputArgs1VarNonLin: public InputArgs2Var{
    using ParentType = InputArgs2Var;
    double nlin_rel_err = 1e-8, nlin_abs_err = 1e-8;
    int nlin_maxit = 10;
    double lin_abs_scale = 0.01;

    uint parseArg(int argc, char* argv[], bool print_messages = true) override {
        #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
            else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
        uint i = 0;
        if (strcmp(argv[i], "-re") == 0 || strcmp(argv[i], "--rel_error") == 0){
            GETARG(nlin_rel_err = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-ae") == 0 || strcmp(argv[i], "--abs_error") == 0){
            GETARG(nlin_abs_err = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-ni") == 0 || strcmp(argv[i], "--maxits") == 0){
            GETARG(nlin_maxit = std::stoi(argv[++i]);)
            return i+1; 
        } if (                               strcmp(argv[i], "--latol_scl") == 0){
            GETARG(lin_abs_scale = std::stod(argv[++i]);)
            return i+1; 
        } else 
            return ParentType::parseArg(argc, argv, print_messages);
        #undef GETARG
    }
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override{
        out << prefix << "newton: stop tolerances: rel_tol = " << nlin_rel_err << ", abs_tol = " << nlin_abs_err << "\n";
        out << prefix << "newton: maximum iteration number = " << nlin_maxit << "\n";
        out << prefix << "newton: linear solver absolute tolerance scale = " << lin_abs_scale << "\n";
        ParentType::print(out, prefix);
    }

protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override{
        out << prefix << "  -re, --rel_error DVAL    <Set stop relative residual norm for newton method, default=\"" << nlin_rel_err << "\">\n";
        out << prefix << "  -ae, --abs_error DVAL    <Set stop absolute residual norm for newton method, default=\"" << nlin_abs_err << "\">\n";
        out << prefix << "  -ni, --maxits    IVAL    <Set maximum number of newton method iterations, default=\"" << nlin_maxit << "\">\n";
        out << prefix << "       --latol_scl IVAL    <Set newton linear solver absolute tolerance scale, default=\"" << lin_abs_scale << "\">\n";
        ParentType::printArgsDescr(out, prefix);
    }
};

int main(int argc, char* argv[]){
    InputArgs1VarNonLin p;
    std::string prob_name = "nonlin_elast_incompress";
    p.save_prefix = prob_name + "_out"; 
    p.axis_sizes = 5;
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
    FemSpace PFem = choose_space_from_name(p.PSpace);
    uint unf = UFem.dofMap().NumDofOnTet(), pnf = PFem.dofMap().NumDofOnTet();
    auto& udofmap = UFem.dofMap();
    auto mask = GeomMaskToInmostElementType(udofmap.GetGeomMask()) | FACE;
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

    struct PotentialParams{
        double mu, lambda;
    };
    auto Potential = [](PotentialParams p, SymMtx3D<> E, double p_coef, unsigned char dif = 2){
        auto I1 = Mech::I1<>{dif, E}; 
        auto I2 = Mech::I2<>{dif, E};
        auto J = Mech::J<>{dif, E};
        Param<> mu(p.mu), lambda(p.lambda);
        
        auto W_svk = (lambda + 2*mu)/8 * sq(I1 - 3) + mu*(I1 - 3) - mu/2*(I2 - 3) - p_coef * (J - 1);
        return W_svk;
    };
    auto P_func = [Potential](PotentialParams p, const Mtx3D<>& grU, double p_coef) -> Mtx3D<>{ return Mech::S_to_P(grU, Potential(p, Mech::grU_to_E(grU), p_coef, 1).D()); };
    auto dP_func = [Potential](PotentialParams p, const Mtx3D<>& grU, double p_coef) -> Sym4Tensor3D<> { auto W = Potential(p, Mech::grU_to_E(grU), p_coef, 2); return Mech::dS_to_dP(grU, W.D(), W.DD()); };
    auto dJ_func = [](const Mtx3D<>& grU)->Mtx3D<>{ return Mech::S_to_P(grU, Mech::J<>(1, Mech::grU_to_E(grU)).D()); };
    auto ddJ_func = [](const Mtx3D<>& grU)->Sym4Tensor3D<>{ auto J = Mech::J<>(2, Mech::grU_to_E(grU)); return Mech::dS_to_dP(grU, J.D(), J.DD()); };
    auto comp_gradU = [gradUFEM = UFem.getOP(GRAD)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> udofs, DynMem<>& alloc)->Mtx3D<>{
        Mtx3D<> grU;
        DenseMatrix<> A(grU.m_dat.data(), 9, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(udofs.data, udofs.size, 1), gradUFEM, A, alloc);
        return grU;
    };
    auto comp_lagrange_coef = [idenPFEM = PFem.getOP(IDEN)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> pdofs, DynMem<>& alloc)->double{
        double p = NAN;
        DenseMatrix<> pa(&p, 1, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(pdofs.data, pdofs.size, 1), idenPFEM, pa, alloc);
        return p;
    };
    auto W_params = [](const Cell& c, const Coord<> &X) -> PotentialParams{
        (void) c, (void) X;
        PotentialParams r;
        r.lambda = 1.0, r.mu = 3.0;
        return r;
    };
    auto external_pressure = [](const Cell& c, const Coord<> &X) -> double{
        (void) c, (void) X;
        return 0.001;
    };

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
        BndMarker lbl;      //< save labels used to apply boundary conditions
        ArrayView<> udofs;  //< save elemental dofs to evaluate grad_j u_i (x)
        ArrayView<> pdofs;  //< save elemental dofs to evaluate Lagrange coefficient p(x)

        //some helper data to be postponed to tensor functions
        const Tetra<const double>* pXYZ = nullptr;    
        DynMem<>* palloc = nullptr;
        const Cell* c = nullptr;
    };

    auto P_tensor = [comp_gradU, comp_lagrange_coef, P_func, W_params](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double p_coef = comp_lagrange_coef(X, *p.pXYZ, p.pdofs, *p.palloc);
        PotentialParams c = W_params(*p.c, X);
        auto P = P_func(c, grU, p_coef);
        std::copy(P.m_dat.data(), P.m_dat.data() + 9, D);
        return Ani::TENSOR_GENERAL;
    };
    auto dP_tensor = [comp_gradU, comp_lagrange_coef, dP_func, W_params](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double p_coef = comp_lagrange_coef(X, *p.pXYZ, p.pdofs, *p.palloc);
        PotentialParams c = W_params(*p.c, X);
        auto dP = tensor_convert<Tensor4Rank<3>>(dP_func(c, grU, p_coef));
        std::copy(dP.m_dat.data(), dP.m_dat.data() + 81, D);
        return Ani::TENSOR_SYMMETRIC;
    };
    auto dlagr_tensor = [comp_gradU, dJ_func](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        auto P = dJ_func(grU);
        std::copy(P.m_dat.data(), P.m_dat.data() + 9, D);
        return Ani::TENSOR_GENERAL;
    };
    auto lagr_rhs_tensor = [comp_gradU, dJ_func](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        D[0] = (Mech::J<>(0, Mech::grU_to_E(grU))() - 1);

        return Ani::TENSOR_GENERAL;
    };
    auto pressure_tensor = [comp_gradU, dJ_func, external_pressure](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double p0 = external_pressure(*p.c, X);
        auto P = p0*dJ_func(grU);
        std::copy(P.m_dat.data(), P.m_dat.data() + 9, D);
        return Ani::TENSOR_GENERAL;
    };
    auto dpressure_tensor = [comp_gradU, ddJ_func, external_pressure](const Coord<> &X, double *D, TensorDims dims, void *user_data, int iTet){
        (void) dims; (void) iTet;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grU = comp_gradU(X, *p.pXYZ, p.udofs, *p.palloc);
        double p0 = external_pressure(*p.c, X);
        auto dP = tensor_convert<Tensor4Rank<3>>(p0*ddJ_func(grU));
        std::copy(dP.m_dat.data(), dP.m_dat.data() + 81, D);
        return Ani::TENSOR_GENERAL;
    };
    auto F_tensor = [](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        F[0] = 0, F[1] = 0, F[2] = 0; 
        return Ani::TENSOR_GENERAL;
    };

    // Define tag to store result
    Tag u = createFemVarTag(m, *udofmap.target<>(), "u");
    Tag p_tag = createFemVarTag(m, *PFem.dofMap().target<>(), "p");
    std::vector<Tag> up{u, p_tag};

    //define function for gathering data from every tetrahedron to send them to elemental assembler

    auto local_data_gatherer = [&BndLabel, unf, pnf, geom_mask = (UFem.dofMap().GetGeomMask() | DofT::FACE)](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.lbl.fillFromBndTag(BndLabel, geom_mask, *p.nodes, *p.edges, *p.faces);
        data.udofs.Init(p.vars->begin(0), unf);
        data.pdofs.Init(p.vars->begin(1), pnf);
        data.c = p.cell;

        p.compute(args, &data);
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_jacobian_assembler = 
        [unf, pnf, dP_tensor, dpressure_tensor, dlagr_tensor, &UFem, &PFem, order = quad_order](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc){
        (void) rw, (void) iw;
        DenseMatrix<> A(Adat, unf+pnf, unf+pnf); A.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc((unf+pnf)*(unf+pnf), 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf+pnf, unf+pnf); B.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto& d = *static_cast<ProbLocData*>(user_data);
        d.pXYZ = &XYZ, d.palloc = &adapt_alloc;
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN), iden_p = PFem.getOP(IDEN);

        // elemental stiffness matrix <dP grad(P2^3), grad(P2^3)> 
        fem3Dtet<DfuncTraits<>>(XYZ, grad_u, grad_u, dP_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            for (std::size_t j = 0; j < unf; ++j)
                A(i, j) = B(i, j);
        // \int (dP(\nabla u)/dp p) : \nabla \phi  
        // \int q d(1 - J)/dF : \nabla u
        fem3Dtet<DfuncTraits<>>(XYZ, grad_u, iden_p, dlagr_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            for (std::size_t j = 0; j < pnf; ++j)
                A(i, unf+j) = A(unf+j, i) = -B(j, i);

        // read labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // apply Neumann BC
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3DfaceN<DfuncTraits<>> ( XYZ, k, grad_u, iden_u, dpressure_tensor, B, adapt_alloc, order, &d);
                for (uint i = 0; i < unf; ++i)
                    for (uint j = 0; j < unf; ++j)
                        A(i, j) += B(i, j);
            }
        }

        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity sp = dat.lbl.getSparsity(DIRICHLET_BND);
        if (!sp.empty())
            applyDirMatrix(*UFem.dofMap().target<>(), A, sp); 
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_residual_assembler = 
        [unf, pnf, P_tensor, pressure_tensor, lagr_rhs_tensor, F_tensor, &UFem, &PFem, order = quad_order, grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN)](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc){
        (void) rw, (void) iw;
        DenseMatrix<> F(Adat, unf+pnf, 1); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc(unf+pnf, 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf+pnf, 1); B.SetZero();

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto& d = *static_cast<ProbLocData*>(user_data);
        d.pXYZ = &XYZ, d.palloc = &adapt_alloc;
        auto grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN), iden_p = PFem.getOP(IDEN);  
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;  

        // elemental stiffness matrix <P, grad(P2^3)> 
        fem3Dtet<DfuncTraits<>>(XYZ, iden_p0, grad_u, P_tensor, B, adapt_alloc, order, &d); 
        for (std::size_t i = 0; i < unf; ++i)
            F[i] = B[i];
        // elemental right hand side vector <-(J-1), P1>
        fem3Dtet<>(XYZ, iden_p0, iden_p, lagr_rhs_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < pnf; ++i)
            F[i+unf] = -B[i];    
        // elemental right hand side vector <F, P2^3>
        fem3Dtet<DfuncTraits<>>(XYZ, iden_p0, iden_u, F_tensor, B, adapt_alloc, order, &d);
        for (std::size_t i = 0; i < unf; ++i)
            F[i] -= B[i];

        // read labels from user_data
        auto& dat = *static_cast<ProbLocData*>(user_data);
        // apply Neumann BC
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3DfaceN<DfuncTraits<>> ( XYZ, k, iden_p0, iden_u, pressure_tensor, B, adapt_alloc, order, &d);
                for (std::size_t i = 0; i < unf; ++i)
                    F[i] += B[i];
            }
        }

        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity sp = dat.lbl.getSparsity(DIRICHLET_BND);
        if (!sp.empty())
            applyDirResidual(*UFem.dofMap().target<>(), F, sp);
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatFunc(GenerateElemMat(local_jacobian_assembler, unf+pnf, unf+pnf, 0, 0));
    discr.SetRHSFunc(GenerateElemRhs(local_residual_assembler, unf+pnf, 0, 0));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*UFem.base()), Var1Helper = GenerateHelper(*PFem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        fed.PushTrialFunc(Var1Helper, "p");
        fed.PushTestFunc(Var1Helper, "phi_p");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(local_data_gatherer);
    discr.PrepareProblem();
    discr.pullInitValFrom(up);
    if (pRank == 0) std::cout << "#dofs = " << discr.m_enum.getMatrixSize() << std::endl;

    //get parallel interval and allocate parallel vectors
    auto i0 = discr.getBegInd(), i1 = discr.getEndInd();
    Sparse::Matrix  A( "A" , i0, i1, m->GetCommunicator());
    Sparse::Vector  x( "x" , i0, i1, m->GetCommunicator()), 
                    dx("dx", i0, i1, m->GetCommunicator()), 
                    b( "b" , i0, i1, m->GetCommunicator());
    //discr.AssembleTemplate(A);  //< preallocate memory for matrix (to accelerate matrix assembling), this call is optional
    // set options to use preallocated matrix state (to accelerate matrix assembling)
    Ani::AssmOpts opts = Ani::AssmOpts()/*.SetIsMtxIncludeTemplate(true)
                                        .SetUseOrderedInsert(true)
                                        .SetIsMtxSorted(true)*/; //< setting this parameters is optional

    //setup linear solver
    Solver lin_solver(p.lin_sol_nm, p.lin_sol_prefix);
    lin_solver.SetParameterReal("drop_tolerance", 1e-2);
    lin_solver.SetParameterReal("reuse_tolerance", 1e-3);
    auto assemble_R = [&discr, &up](const Sparse::Vector& x, Sparse::Vector &b) -> int{
        discr.SaveSolution(x, up);
        std::fill(b.Begin(), b.End(), 0.0);
        return discr.AssembleRHS(b);
    };
    auto assemble_J = [&discr, &up, opts](const Sparse::Vector& x, Sparse::Matrix &A) -> int{
        discr.SaveSolution(x, up);
        std::for_each(A.Begin(), A.End(), [](INMOST::Sparse::Row& row){ for (auto vit = row.Begin(); vit != row.End(); ++vit) vit->second = 0.0; });
        return discr.AssembleMatrix(A, opts);
    };
    auto vec_norm = [m](const Sparse::Vector& x)->double{
        double lsum = 0, gsum = 0;
        for (auto itx = x.Begin(); itx != x.End(); ++itx)
            lsum += (*itx) * (*itx);
        gsum = m->Integrate(lsum);    
        // #ifdef USE_MPI
        //     MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, x.GetCommunicator());
        // #else
        //     gsum = lsum;
        // #endif
        return sqrt(gsum);
    };
    auto vec_saxpy = [](double a, const Sparse::Vector& xv, double b, const Sparse::Vector& yv, Sparse::Vector& zv) {
        auto itz = zv.Begin();
        for (auto itx = xv.Begin(), ity = yv.Begin();
            itx != xv.End() && ity != yv.End() && itz != zv.End(); ++itx, ++ity, ++itz)
            *itz = a * (*itx) + b * (*ity);
    };

    {
        TimerWrap m_timer_total; m_timer_total.reset();
        assemble_R(x, b);
        double anrm = vec_norm(b), rnrm = 1;
        double anrm0 = anrm;
        int ni = 0;
        if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        while (rnrm >= p.nlin_rel_err && anrm >= p.nlin_abs_err && ni < p.nlin_maxit){
            assemble_J(x, A);
            if (pRank == 0) std::cout << "--- Compute preconditioner ---" << std::endl;
            lin_solver.SetMatrix(A);
            if (pRank == 0) std::cout << "--- Solve linear system ---" << std::endl;
            if (std::stod(lin_solver.GetParameter("absolute_tolerance")) > p.lin_abs_scale*anrm)
                lin_solver.SetParameterReal("absolute_tolerance", p.lin_abs_scale*anrm);
            lin_solver.Solve(b, dx);
            print_linear_solver_status(lin_solver, prob_name, true);
            vec_saxpy(1, x, -1, dx, x);
            assemble_R(x, b);
            anrm = vec_norm(b);
            rnrm = anrm / anrm0;
            ni++;
            if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        }
        double total_sol_time =  m_timer_total.elapsed();
        if (pRank == 0) std::cout << "Total solution time: " << total_sol_time << "s" << std::endl;
    }

    //copy result to the tag and save solution
    discr.SaveSolution(x, up);
    m->Save(p.save_dir + p.save_prefix + ".pvtu");

    discr.Clear();
    mptr.reset();
    InmostFinalize();

    return 0;
}