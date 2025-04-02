/* 
 * This problem generates and solves the elasto-plastic cylinder expansion problem in symmetric formulation (quarter of cylinder is considered).
 * \f[
 * \begin{aligned}
 *   \mathrm{div}\ \sigma (\varepsilon^e) + f   &= 0\    in\  \Omega  \\
 *      u_x                         &= 0\     on\  \Gamma_{left}
 *      u_y                         &= 0\     on\  \Gamma_{bottom}
 *      \sigma \cdot \mathbf{N}     &= q \mathbf{N}\ on\  \Gamma_{internal}\\  
 *      \sigma \cdot \mathbf{N}     &= 0\    on\  \Gamma_{free}\\
 *   \sigma                      &= C_{(ij)(kl)} \varepsilon^e_{(kl)}\\
 *   C_{(ij)(kl)}                &= \lambda \delta_{ij}\delta_{kl} + \mu (\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})\\
 *   \varepsilon &= \varepsilon^e + \varepsilon^p
 *   \varepsilon &= \frac{1}{2}(\mathrm{grad}\ \mathbf{u} + (\mathrm{grad}\ \mathbf{u})^T)
 * \end{aligned}
 * \f]
 * References:
 * (1) https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html
 * (2) https://github.com/a-latyshev/convex-plasticity/blob/main/tutorials/plasticity/plasticity.ipynb
*/

#include "prob_args.h"
#include "anifem++/autodiff/cauchy_strain_autodiff.h"

using namespace INMOST;

double E = 7e4, nu = 0.3; // Young modulus (Pa), Poisson coefficient
double mu = E/(2*(1+nu)), lambda = E*nu/((1+nu)*(1-2*nu)); // Lame parameters (Pa)
double Et = E/100; // tangent modulus
double sigma_yield = 250; // yield strength (Pa)
double H = E*Et/(E-Et); // hardening modulus
double Re = 1.3, Ri = 1.; // mesh external and internal radii
std::string mesh_fname = "../../../data/mesh/thick_cylinder.msh";

struct InputArgs1VarNonLin: public InputArgs1Var{
    using ParentType = InputArgs1Var;
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
        } if (strcmp(argv[i], "-Re") == 0 || strcmp(argv[i], "--external-radius") == 0){
            GETARG(Re = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-E") == 0 || strcmp(argv[i], "--young-modulus") == 0){
            GETARG(E = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-nu") == 0 || strcmp(argv[i], "--poisson-coef") == 0){
            GETARG(nu = std::stod(argv[++i]);)
            return i+1;
        } if (strcmp(argv[i], "-sig0") == 0 || strcmp(argv[i], "--sigma-yield") == 0){
            GETARG(sigma_yield = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-Et") == 0 || strcmp(argv[i], "--tangent-modulus") == 0){
            GETARG(Et = std::stod(argv[++i]);)
            H = E*Et/(E-Et);
            return i+1; 
        } if (strcmp(argv[i], "--latol_scl") == 0){
            GETARG(lin_abs_scale = std::stod(argv[++i]);)
            return i+1; 
        } if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mesh-name") == 0){
            GETARG(mesh_fname = argv[++i];)
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
        out << prefix << "  -re,   --rel_error       DVAL    <Set stop relative residual norm for newton method, default=\"" << nlin_rel_err << "\">\n";
        out << prefix << "  -ae,   --abs_error       DVAL    <Set stop absolute residual norm for newton method, default=\"" << nlin_abs_err << "\">\n";
        out << prefix << "  -ni,   --maxits          IVAL    <Set maximum number of newton method iterations, default=\"" << nlin_maxit << "\">\n";
        out << prefix << "         --latol_scl       DVAL    <Set newton linear solver absolute tolerance scale, default=\"" << lin_abs_scale << "\">\n";
        out << prefix << "  -Re,   --external-radius DVAL    <Set external radius of cylinder (if mesh was changed), default=\"" << Re << "\">\n";
        out << prefix << "  -E,    --young-modulus   DVAL    <Set Young modulus of cylinder , default=\"" << E << "\">\n";
        out << prefix << "  -nu,   --poisson-coef    DVAL    <Set Poisson coefficient of cylinder , default=\"" << nu << "\">\n";
        out << prefix << "  -sig0, --sigma-yield     DVAL    <Set yield stress (von Mises criterium) , default=\"" << sigma_yield << "\">\n";
        out << prefix << "  -Et,   --tangent-modulus DVAL    <Set plastic tangent modulus , default=\"" << Et << "\">\n";
        out << prefix << "  -m,    --mesh-name       SVAL    <Set mesh file name , default=\"" << mesh_fname << "\">\n";
        ParentType::printArgsDescr(out, prefix);
    }
};

using namespace Ani;
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
    ArrayView<> sigma_dofs, alpha_dofs; //< elemental dofs of plastic variables

    //some helper data to be postponed to tensor functions
    const Tetra<const double>* pXYZ = nullptr;
    DynMem<>* palloc = nullptr;
    const Cell* c = nullptr;
    double time = 0.0;
};

int main(int argc, char* argv[]){
    InputArgs1VarNonLin p;
    std::string prob_name = "elastoplastic";
    p.save_prefix = prob_name + "_out"; 
    p.parseArgs_mpi(&argc, &argv);
    unsigned quad_order = p.max_quad_order;

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);

    Mesh* m = new Mesh;
    //m->SetFileOption("VERBOSITY","2");
    if(m->isParallelFileFormat(mesh_fname))
    {
        m->Load(mesh_fname);
        // Constructing of Ghost cells in 1 layers connected via nodes is required for FEM Assemble method
    }
    else if(pRank == 0)
    {
        m->Load(mesh_fname);
    }
    //m->ExchangeGhost(1,NODE);
    RepartMesh(m,true);
    m->AssignGlobalID(NODE|EDGE|FACE|CELL);

    print_mesh_sizes(m);

    //generate FEM space from it's name
    FemSpace UFem = choose_space_from_name(p.USpace)^3;
    uint unf = UFem.dofMap().NumDofOnTet();
    auto& dofmap = UFem.dofMap();
    uint nquad = tetrahedron_quadrature_formulas(quad_order).GetNumPoints();

    auto mask = GeomMaskToInmostElementType(dofmap.GetGeomMask()) | FACE;
    const int INTERNAL_PART = 0, FREE_BND = 1, DIRICHLET_BND_U = 2, DIRICHLET_BND_V = 4, PRESSURE_BND = 8, DIRICHLET_BND_W = 16;
    
    // Set boundary labels on all boundaries
    Tag BndLabel = m->CreateTag("bnd_label", DATA_INTEGER, mask, NONE, 1);
    auto bmrk = m->CreateMarker();
    m->MarkBoundaryFaces(bmrk);
    for (auto it = m->BeginElement(mask); it != m->EndElement(); ++it) it->Integer(BndLabel) = INTERNAL_PART;
    for (auto it = m->BeginFace(); it != m->EndFace(); ++it) if (it->GetMarker(bmrk)) {
        std::array<double, 3> c, nrm;
        it->Centroid(c.data()); it->UnitNormal(nrm.data());
        int lbl = INTERNAL_PART;
        // left symmetry plane:  u = 0
        if (fabs(c[0] - 0) < 10*std::numeric_limits<double>::epsilon()) lbl = DIRICHLET_BND_U;
        // right symmetry plane: v = 0
        else if (fabs(c[1] - 0) < 10*std::numeric_limits<double>::epsilon()) lbl = DIRICHLET_BND_V;
        // faces normal to Z:    w = 0
        else if (fabs(nrm[2]) > 1.0-1.0e-4)
            lbl = DIRICHLET_BND_W;
        // external boundary
        else if (sqrt(c[0]*c[0]+c[1]*c[1]) > 0.5*(::Re+::Ri))
            lbl = FREE_BND;
        // internal boundary:    p = p0
        else 
            lbl = PRESSURE_BND;
        auto set_label = [BndLabel, lbl](const auto& elems){
            for (unsigned ni = 0; ni < elems.size(); ni++)
                elems[ni].Integer(BndLabel) |= lbl;
        };
        if (mask & NODE) set_label(it->getNodes());
        if (mask & EDGE) set_label(it->getEdges());
        if (mask & FACE) set_label(it->getFaces());
    }
    m->ExchangeData(BndLabel, mask);
    m->ReleaseMarker(bmrk);

    // Define tag to store result
    TagRealArray u0 = createFemVarTag(m, *dofmap.target<>(), "u0"); // previous time step value
    TagRealArray du = createFemVarTag(m, *dofmap.target<>(), "du"); // increment to the previous time step value
    // storage for additional plastic unknowns
    TagRealArray tag_alpha = m->CreateTag("tag_alpha", DATA_REAL, CELL, NONE, 2*nquad); // alpha, alpha0
    TagRealArray tag_sigma = m->CreateTag("tag_sigma", DATA_REAL, CELL, NONE, 2*6*nquad); // sigma, sigma0

    auto comp_gradDeltaU = [gradUFEM = UFem.getOP(GRAD)](const Coord<> &X, const Tetra<const double>& XYZ, Ani::ArrayView<> udofs, DynMem<>& alloc)->Mtx3D<>{
        Mtx3D<> grU;
        DenseMatrix<> A(grU.m_dat.data(), 9, 1);
        fem3DapplyX(XYZ, ArrayView<const double>(X.data(), 3), DenseMatrix<>(udofs.data, udofs.size, 1), gradUFEM, A, alloc);
        return grU;
    };
    auto external_pressure = [](const Cell& c, const Coord<> &X, double time) -> double{
        (void) c, (void) X;
        return time*2./sqrt(3)*::sigma_yield*log(::Re/::Ri);
    };
    auto sigma_func = [](const SymMtx3D<>& deps){
        return ::lambda*deps.Trace()*SymMtx3D<>::Identity() + 2*::mu*deps;
    };
    // Radial return algorithm 
    auto compute_plastic = [sigma_func](const SymMtx3D<>& deps, struct ProbLocData& p, SymMtx3D<>& n_elas, SymMtx3D<>& sigma, double& beta, int iQuad, bool jacobian = false)
    {
        SymMtx3D<> sig_old;
        std::copy(p.sigma_dofs.begin()+iQuad*12+0, p.sigma_dofs.begin()+iQuad*12+6, sig_old.begin());
        auto alpha_n = p.alpha_dofs[iQuad*2+0];

        SymMtx3D<> sig_elas = sig_old + sigma_func(deps);
        SymMtx3D<> s = sig_elas - sig_elas.Trace()/3 * SymMtx3D<>::Identity();
        double sig_eq = sqrt(3./2 * s.SquareFrobNorm())+1.0e-20;
        // linear isotropic hardening law: H(alpha) = sigma_yield + H * alpha
        double f_elas = sig_eq - (::sigma_yield + ::H*alpha_n);
        // linear isotropic hardening law leads to analytical expression for dp:
        double dp = (f_elas + fabs(f_elas))/2 / (3*::mu + ::H);
        n_elas = (dp > 0) * s / sig_eq;
        beta = 3*::mu * dp /sig_eq;
        sigma = sig_elas - beta*s;
        if(!jacobian) // update plastic variables
        {
            p.alpha_dofs[iQuad*2+1] = alpha_n + dp;
            std::copy(sigma.begin(), sigma.end(), p.sigma_dofs.begin()+iQuad*12+6);
        }
    };
    auto sigma_resid_tensor = [comp_gradDeltaU, compute_plastic](const std::array<double, 3>& x, double *Dmem, TensorDims Ddims, void * user_data, int iQuad){
        (void) Ddims;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grDeltaU = comp_gradDeltaU(x, *p.pXYZ, p.udofs, *p.palloc);
        auto deps = grDeltaU.Sym();
        double beta; SymMtx3D<> n_elas, sigma;
        compute_plastic(deps, p, n_elas, sigma, beta, iQuad, false);
        Mtx3D<> sigma_resid(sigma);
        std::copy(sigma_resid.m_dat.data(), sigma_resid.m_dat.data() + 9, Dmem);
        return Ani::TENSOR_GENERAL;
    };
    auto sigma_tang_tensor = [comp_gradDeltaU, compute_plastic](const std::array<double, 3>& x, double *Dmem, TensorDims Ddims, void * user_data, int iQuad){
        (void) Ddims;
        auto& p = *static_cast<ProbLocData*>(user_data);
        auto grDeltaU = comp_gradDeltaU(x, *p.pXYZ, p.udofs, *p.palloc);
        auto deps = grDeltaU.Sym();
        double beta; SymMtx3D<> n_elas, sigma;
        compute_plastic(deps, p, n_elas, sigma, beta, iQuad, true);
        auto sigma_tang = tensor_convert<Tensor4Rank<3>>(
            (::lambda + 2./3*::mu*beta) * BiSym4Tensor3D<>::TensorSquare(SymMtx3D<>::Identity())
            + 2*::mu*(1-beta)*BiSym4Tensor3D<>::Identity()
            - 3*::mu*(1./(1+::H/(3*::mu)) - beta)*BiSym4Tensor3D<>::TensorSquare(n_elas));
        std::copy(sigma_tang.m_dat.data(), sigma_tang.m_dat.data()+81, Dmem);
        return Ani::TENSOR_GENERAL;
    };
    auto pressure_tensor = [external_pressure](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iQuad){
        (void) dims; (void) iQuad;
        auto& p = *static_cast<ProbLocData*>(user_data);
        double p0 = external_pressure(*p.c, X, p.time);
        DenseMatrix<> pm(F, 9, 1);
        pm.SetZero();
        pm(0 + 0*3, 0) = p0;
        pm(1 + 1*3, 0) = p0;
        pm(2 + 2*3, 0) = p0;
        return Ani::TENSOR_GENERAL;
    };
    auto sigma_resid_fuse_tensor = [sigma_resid_tensor](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem){
        for(std::size_t r = 0; r < mem.f; ++r) // over tets
        for(std::size_t n = 0; n < mem.q; ++n) // over quad points
        {
            DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
            sigma_resid_tensor({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]}, 
                            Dloc.data, Ddims, user_data, n);
        }
        return Ani::TENSOR_GENERAL;
    };
    auto sigma_tang_fuse_tensor = [sigma_tang_tensor](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem){
        for(std::size_t r = 0; r < mem.f; ++r) // over tets
        for(std::size_t n = 0; n < mem.q; ++n) // over quad points
        {
            DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*(n + mem.q*r), Ddims.first, Ddims.second);
            sigma_tang_tensor({X.data[3*(n + mem.q*r) + 0], X.data[3*(n + mem.q*r) + 1], X.data[3*(n + mem.q*r) + 2]}, 
                            Dloc.data, Ddims, user_data, n);
        }
        return Ani::TENSOR_GENERAL;
    };

    double T = 0.0;
    //define function for gathering data from every tetrahedron to send them to elemental assembler
    auto local_data_gatherer = [&BndLabel, unf, UFem, geom_mask = (UFem.dofMap().GetGeomMask() | DofT::FACE), &T, nquad, tag_sigma, tag_alpha](ElementalAssembler& p) -> void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.lbl.fillFromBndTag(BndLabel, geom_mask, *p.nodes, *p.edges, *p.faces);
        data.udofs.Init(p.vars->begin(0), unf);
        data.c = p.cell;
        data.time = T;
        // pull tag storage for plastic variables into user_data
        Storage::real_array dat_alpha = tag_alpha[*p.cell], dat_sigma = tag_sigma[*p.cell];
        data.alpha_dofs.Init(dat_alpha.data(), 2*nquad);
        data.sigma_dofs.Init(dat_sigma.data(), 2*6*nquad);

        p.compute(args, &data);
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_jacobian_assembler = 
        [unf, sigma_tang_fuse_tensor, &UFem, grad_u = UFem.getOP(GRAD), order = quad_order](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc){
        (void) rw, (void) iw;
        DenseMatrix<> A(Adat, unf, unf); A.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc(unf*unf, 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf, unf); B.SetZero();
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);

        auto& dat = *static_cast<ProbLocData*>(user_data);
        dat.pXYZ = &XYZ, dat.palloc = &adapt_alloc;

        // elemental stiffness matrix <C grad(P1), grad(P1)> 
        fem3Dtet<DfuncTraitsFusive<>>(XYZ, grad_u, grad_u, sigma_tang_fuse_tensor, A, adapt_alloc, order, &dat);
        // Neumann BC are constant, jacobian contribution is zero

        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity spx = dat.lbl.getSparsity(DIRICHLET_BND_U), spy = dat.lbl.getSparsity(DIRICHLET_BND_V), spz = dat.lbl.getSparsity(DIRICHLET_BND_W);
        const auto& dofmap = UFem.dofMap();
        if (!spx.empty())
        {
            int ext_dims[2] = {0,2};// x and z displacements are fixed
            applyDirMatrix(dofmap.GetNestedDofMap(ext_dims+0,1), A, spx);
            applyDirMatrix(dofmap.GetNestedDofMap(ext_dims+1,1), A, spx);
        }
        if (!spy.empty())
        {
            int ext_dims[2] = {1,2};// y and z displacements are fixed
            applyDirMatrix(dofmap.GetNestedDofMap(ext_dims+0,1), A, spy);
            applyDirMatrix(dofmap.GetNestedDofMap(ext_dims+1,1), A, spy);
        }
        if (!spz.empty())
        {
            int ext_dims[1] = {2};// z displacement is fixed
            applyDirMatrix(dofmap.GetNestedDofMap(ext_dims+0,1), A, spz);
        }
    };
    std::function<void(const double**, double*, double*, long*, void*, DynMem<double, long>*)> local_residual_assembler = 
        [unf, sigma_resid_fuse_tensor, pressure_tensor, &UFem, grad_u = UFem.getOP(GRAD), iden_u = UFem.getOP(IDEN), order = quad_order](const double** XY/*[4]*/, double* Adat, double* rw, long* iw, void* user_data, DynMem<double, long>* fem_alloc){
        (void) rw, (void) iw;
        DenseMatrix<> F(Adat, unf, 1); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        auto Bmem = adapt_alloc.alloc(unf, 0, 0);
        DenseMatrix<> B(Bmem.getPlainMemory().ddata, unf, 1); B.SetZero();
        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);

        auto& dat = *static_cast<ProbLocData*>(user_data);
        dat.pXYZ = &XYZ, dat.palloc = &adapt_alloc;
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;

        // elemental stiffness matrix <sigma_resid, grad(P1)> 
        fem3Dtet<DfuncTraitsFusive<>>(XYZ, iden_p0, grad_u, sigma_resid_fuse_tensor, F, adapt_alloc, order, &dat); 
        // apply Neumann BC
        for (int k = 0; k < 4; ++k){
            if (dat.lbl.f[k] & PRESSURE_BND){ //Neumann BC
                fem3DfaceN<DfuncTraits<TENSOR_GENERAL>> ( XYZ, k, iden_p0, iden_u, pressure_tensor, B, adapt_alloc, order, &dat);
                F += B;
            }
        }

        // choose boundary parts of the tetrahedron 
        DofT::TetGeomSparsity spx = dat.lbl.getSparsity(DIRICHLET_BND_U), spy = dat.lbl.getSparsity(DIRICHLET_BND_V), spz = dat.lbl.getSparsity(DIRICHLET_BND_W);
        const auto& dofmap = UFem.dofMap();
        if (!spx.empty())
        {
            int ext_dims[2] = {0,2};// x and z displacements are fixed
            applyDirResidual(dofmap.GetNestedDofMap(ext_dims+0,1), F, spx);
            applyDirResidual(dofmap.GetNestedDofMap(ext_dims+1,1), F, spx);
        }
        if (!spy.empty())
        {
            int ext_dims[2] = {1,2};// y and z displacements are fixed
            applyDirResidual(dofmap.GetNestedDofMap(ext_dims+0,1), F, spy);
            applyDirResidual(dofmap.GetNestedDofMap(ext_dims+1,1), F, spy);
        }
        if (!spz.empty())
        {
            int ext_dims[1] = {2};// z displacement is fixed
            applyDirResidual(dofmap.GetNestedDofMap(ext_dims+0,1), F, spz);
        }
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatFunc(GenerateElemMat(local_jacobian_assembler, unf, unf, 0, 0));
    discr.SetRHSFunc(GenerateElemRhs(local_residual_assembler, unf, 0, 0));
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
    discr.pullInitValFrom(du);

    //get parallel interval and allocate parallel vectors
    auto i0 = discr.getBegInd(), i1 = discr.getEndInd();
    Sparse::Matrix  A( "A" , i0, i1, m->GetCommunicator());
    Sparse::Vector  x( "x" , i0, i1, m->GetCommunicator()),
                    dx("dx", i0, i1, m->GetCommunicator()), 
                    b( "b" , i0, i1, m->GetCommunicator());
    discr.AssembleTemplate(A);  //< preallocate memory for matrix (to accelerate matrix assembling), this call is optional
    // set options to use preallocated matrix state (to accelerate matrix assembling)
    Ani::AssmOpts opts = Ani::AssmOpts().SetIsMtxIncludeTemplate(true)
                                        .SetUseOrderedInsert(true)
                                        .SetIsMtxSorted(true); //< setting this parameters is optional

    //setup linear solver
    Solver lin_solver(p.lin_sol_nm, p.lin_sol_prefix);
    auto assemble_R = [&discr, &du](const Sparse::Vector& x, Sparse::Vector &b) -> int {
        discr.SaveSolution(x, du);
        std::fill(b.Begin(), b.End(), 0.0);
        return discr.AssembleRHS(b);
    };
    auto assemble_J = [&discr, &du, opts](const Sparse::Vector& x, Sparse::Matrix &A) -> int {
        std::for_each(A.Begin(), A.End(), [](INMOST::Sparse::Row& row){ for (auto vit = row.Begin(); vit != row.End(); ++vit) vit->second = 0.0; });
        return discr.AssembleMatrix(A, opts);
    };
    auto vec_norm = [m](const Sparse::Vector& x)->double {
        double lsum = 0, gsum = 0;
        for (auto itx = x.Begin(); itx != x.End(); ++itx)
            lsum += (*itx) * (*itx);
        gsum = m->Integrate(lsum);
        return sqrt(gsum);
    };
    auto vec_saxpy = [](double a, const Sparse::Vector& xv, double b, const Sparse::Vector& yv, Sparse::Vector& zv) {
        auto itz = zv.Begin();
        for (auto itx = xv.Begin(), ity = yv.Begin();
            itx != xv.End() && ity != yv.End() && itz != zv.End(); ++itx, ++ity, ++itz)
            *itz = a * (*itx) + b * (*ity);
    };

    constexpr int Nincr = 20;
    // times as in FeniCS example:
    // import numpy as np; Nincr = 20; np.linspace(0, 1.1, Nincr+1)[1:]**0.5
    std::array<double,Nincr> times
        {0.23452079, 0.33166248, 0.40620192, 0.46904158, 0.52440442, \
         0.57445626, 0.62048368, 0.66332496, 0.70356236, 0.74161985, \
         0.77781746, 0.81240384, 0.84557673, 0.87749644, 0.90829511, \
         0.93808315, 0.96695398, 0.99498744, 1.02225242, 1.04880885};
    int step = 1;
    std::ofstream ofs;
    if(pRank == 0)
    {
        ofs = std::ofstream("cylinder_load_displacement.csv");
        ofs << "displacement_inner_boundary;applied_pressure\n";
    }
    TimerWrap m_timer_total; m_timer_total.reset();
    while(step < Nincr+1)
    {
        T = times[step-1];
        TimerWrap m_timer_step; m_timer_step.reset();
        std::fill(x.Begin(), x.End(), 0.0); // x <- 0
        assemble_R(x, b); // 0 = x -> du, b
        double anrm = vec_norm(b), rnrm = 1;
        double anrm0 = anrm;
        int ni = 0;
        if (pRank == 0)
        {
            double p0 = external_pressure(m->BeginCell()->self(), std::array<double,3>{0,0,0}, T);
            std::cout << "Start step " << step << ": T " << T << " " << " load " << p0 << std::endl;
            std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        }
        while (rnrm >= p.nlin_rel_err && anrm >= p.nlin_abs_err && ni < p.nlin_maxit){
            assemble_J(x, A);
            lin_solver.SetMatrix(A); // x -> du, A
            if (std::stod(lin_solver.GetParameter("absolute_tolerance")) > p.lin_abs_scale*anrm)
               lin_solver.SetParameterReal("absolute_tolerance", p.lin_abs_scale*anrm);
            lin_solver.Solve(b, dx);
            print_linear_solver_status(lin_solver, prob_name, true);
            vec_saxpy(1, x, -1, dx, x); // x = b <- x - dx
            assemble_R(x, b); // x -> du, b
            anrm = vec_norm(b);
            rnrm = anrm / anrm0;
            ni++;
            if (pRank == 0) std::cout << prob_name << ":\n\tnit = " << ni << ": newton residual = " << anrm << " ( rel = " << rnrm << " )" <<  std::endl;
        }
        {
            // update plastic variables
            for(int k = 0; k < m->CellLastLocalID(); ++k) if(m->isValidCell(k))
            {
                Cell c = m->CellByLocalID(k);
                Storage::real_array dat_alpha = tag_alpha[c];
                Storage::real_array dat_sigma = tag_sigma[c];
                for(std::size_t iQuad = 0; iQuad < nquad; ++iQuad)
                {
                    dat_alpha[2*iQuad+0] = dat_alpha[2*iQuad+1];
                    std::copy(dat_sigma.data()+iQuad*12+6, dat_sigma.data()+iQuad*12+12, dat_sigma.data()+iQuad*12);
                }
            }
            // u0 <- u0 + du
            discr.SaveSolution(u0, b); // u0 -> b
            vec_saxpy(1, b, +1, x, x); // x  <- b + x = u0 + du
            discr.SaveSolution(x, u0); // u0 <- x     = u0 + du
        }
        double u_max = -1.0;
        for(int k = 0; k < m->NodeLastLocalID(); ++k) if(m->isValidNode(k))
        {
            Storage::real_array u_node = u0[m->NodeByLocalID(k)];
            double ur = sqrt(u_node[0]*u_node[0] + u_node[1]*u_node[1]);
            if(ur > u_max) u_max = ur;
        }
        m->AggregateMax(u_max);
        if(pRank == 0)
        {
            std::cout << "Finished step " << step << std::endl;
            ofs << u_max << ';' << T << '\n';
        }
        m->Save(p.save_dir + p.save_prefix + "_" + std::to_string(step-1) + ".pvtu");
        ++step;
        double step_sol_time =  m_timer_step.elapsed();
        if (pRank == 0) std::cout << "Step solution time: " << step_sol_time << "s" << std::endl;
    }
    double total_sol_time =  m_timer_total.elapsed();
    if (pRank == 0)
    {
        ofs.close();
        std::cout << "Total solution time: " << total_sol_time << "s" << std::endl;
    }

    discr.Clear();
    delete m;
    InmostFinalize();

    return 0;
}