#include "utils.h"
#include "anifem++/fem/spaces/spaces.h"
#include "anifem++/inmost_interface/fem.h"

using namespace INMOST;

void InmostInit(int* argc, char** argv[], const std::string& solver_db, int& pRank, int& pCount){
#if defined(USE_MPI)
    int is_inited = 0;
    MPI_Initialized(&is_inited);
    if (!is_inited) MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &pCount); // Get the total number of processors used
    if (pRank == 0)
        std::cout << "Running with MPI using\n";
#else
    pRank = 0, pCount = 1;
    std::cout << "Running without MPI using\n";
#endif
    Mesh::Initialize(argc, argv);
#ifdef USE_PARTITIONER
    Partitioner::Initialize(argc, argv);
#endif
    Solver::Initialize(argc, argv, solver_db.c_str());
}

void InmostFinalize(){
    Solver::Finalize();
#ifdef USE_PARTITIONER
    Partitioner::Finalize();
#endif
    Mesh::Finalize();
#if defined(USE_MPI)
    int flag = 0;
    MPI_Finalized(&flag);
    if (!flag) MPI_Finalize();
#endif
}

Ani::FemSpace choose_space_from_name(const std::string& name){
    using namespace Ani;
    std::map<std::string, int> conv {
        {"P0", 0}, {"P1", 1}, {"P2", 2}, {"P3", 3},
        {"MINI", 4}, {"MINI1", 4}, {"MINI2", 5}, {"MINI3", 6},
        {"CR1", 7}, {"MINI_CR", 8}, {"MINI_CR1", 8}
    };
    auto it = conv.find(name);
    if (it == conv.end())
        throw std::runtime_error("Faced unknown space name = \"" + name + "\"");
    switch (it->second){
        case 0: return Ani::FemSpace{ P0Space{} };
        case 1: return Ani::FemSpace{ P1Space{} };
        case 2: return Ani::FemSpace{ P2Space{} };
        case 3: return Ani::FemSpace{ P3Space{} };
        case 4: return Ani::FemSpace{ P1Space{} } + Ani::FemSpace{ BubbleSpace{} };
        case 5: return Ani::FemSpace{ P2Space{} } + Ani::FemSpace{ BubbleSpace{} };
        case 6: return Ani::FemSpace{ P3Space{} } + Ani::FemSpace{ BubbleSpace{} };
        case 7: return Ani::FemSpace{ CR1Space{} };
        case 8: return Ani::FemSpace{ CR1Space{} } + Ani::FemSpace{ BubbleSpace{} };
        
    }
    throw std::runtime_error("Doesn't find space with specified name = \"" + name + "\"");
    return FemSpace{};   
}

void print_linear_solver_status(INMOST::Solver& s, const std::string& prob_name, bool exit_on_fail){
    int pRank = 0, pCount = 1;
    #if defined(USE_MPI)
        MPI_Comm_rank(MPI_COMM_WORLD, &pRank);  // Get the rank of the current process
        MPI_Comm_size(MPI_COMM_WORLD, &pCount); // Get the total number of processors used
    #endif
    bool success_solve = s.IsSolved();
    if(!success_solve){
        if (pRank == 0) std::cout << prob_name << ":\n\tsolution failed:\n";
        for (int p = 0; p < pCount; ++p) {
            BARRIER;
            if (pRank != p) continue;
            std::cout << "\t               : #lits " << s.Iterations() << " residual " << s.Residual() << ". "
                      << "preconding = " << s.PreconditionerTime() << "s, solving = " << s.IterationsTime() << "s" << std::endl;
            std::cout << "\tRank " << pRank << " failed to solve system. ";
            std::cout << "Reason: " << s.GetReason() << std::endl;
        }
        if (exit_on_fail)
            exit(-1);
    }
    else{
        if(pRank == 0) {
            std::cout << prob_name << ":\n";
            std::string _s_its = std::to_string(s.Iterations()), s_its = "    ";
            std::copy(_s_its.begin(), _s_its.end(), s_its.begin() + 3 - _s_its.size());

            std::cout << "\tsolved_succesful: #lits " << s_its << " residual " << s.Residual() << ", "
                      << "preconding = " << s.PreconditionerTime() << "s, solving = " << s.IterationsTime() << "s" << std::endl;
        }
    }
}

Tag createFemVarTag(Mesh* m, const Ani::DofT::BaseDofMap& dofmap, const std::string& tag_name, bool use_fixed_size){
    auto mask = Ani::GeomMaskToInmostElementType(dofmap.GetGeomMask()); 
    auto ndofs = Ani::DofTNumDofsToInmostNumDofs(dofmap.NumDofs());
    INMOST_DATA_ENUM_TYPE sz;
    if (!use_fixed_size){
        sz = ndofs[0];
        for (unsigned i = 1; i < ndofs.size(); ++i)
            if (ndofs[i] != 0){
                if (sz == 0) sz = ndofs[i];
                else if (sz != ndofs[i]) {
                    sz = ENUMUNDEF;
                    break;
                }
            }
    } else {
        sz = *std::max_element(ndofs.begin(), ndofs.end()); 
    }    
    Tag u = m->CreateTag(tag_name, DATA_REAL, mask, NONE, sz);
    if (sz == ENUMUNDEF){
        for (auto it = m->BeginElement(mask); it != m->EndElement(); ++it)
            it->RealArrayDV(u).resize(ndofs[it->GetElementNum()], 0.0);
    }
    return u;  
}

void make_l2_project(const std::function<void(Cell c, std::array<double, 3> X, double* res)>& func, Mesh* m, Tag res_tag, Ani::FemSpace fem, int order){
    using namespace Ani;
    uint unf = fem.dofMap().NumDofOnTet();
    auto data_gatherer = [](ElementalAssembler& p)->void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        
        p.compute(args, const_cast<Cell*>(p.cell));
    };
    auto F_tensor = [func](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet)->TensorType {
        (void) dims; (void) iTet;
        Cell& c = *static_cast<Cell*>(user_data);
        func(c, X, F);

        return Ani::TENSOR_GENERAL;
    };
    std::function<void(const double**, double*, double*, double*, long*, void*, DynMem<double, long>*)> local_assm = 
        [fem, unf, order, F_tensor](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, Ani::DynMem<double, long>* fem_alloc){
        (void) w, (void) iw;
        DenseMatrix<> A(Adat, unf, unf), F(Fdat, unf, 1);
        A.SetZero(); F.SetZero();
        auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
        Ani::Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
        auto iden_u = fem.getOP(IDEN);  
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;

        fem3Dtet<DfuncTraits<TENSOR_NULL, true>>(XYZ, iden_u, iden_u, TensorNull<>, A, adapt_alloc, order); 
        // elemental right hand side vector <F, P1>
        fem3Dtet<DfuncTraits<TENSOR_GENERAL, false>>( XYZ, iden_p0, iden_u, F_tensor, F, adapt_alloc, order, user_data);
    };
    //define assembler
    Assembler discr(m);
    discr.SetMatRHSFunc(GenerateElemMatRhs(local_assm, unf, unf, 0, 0));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*fem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(data_gatherer);
    discr.PrepareProblem();
    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
    discr.Assemble(A, b);
    Solver solver(Solver::INNER_ILU2);
    solver.SetParameterReal("absolute_tolerance", 1e-20);
    solver.SetParameterReal("relative_tolerance", 1e-10);
    solver.SetMatrix(A);
    solver.Solve(b, x);
    print_linear_solver_status(solver, "projection", true);

    //copy result to the tag and save solution
    discr.SaveSolution(x, res_tag);
    solver.Clear();
    discr.Clear();
}

std::pair<Tag, bool> create_or_load_problem_tag(INMOST::Mesh *m, const Ani::DofT::BaseDofMap &dofmap, std::string var_name, bool load_if_exists){
    bool have_tag = m->HaveTag(var_name); 
    if (have_tag){
        if (!load_if_exists)
            throw std::runtime_error("Found unexpected tag \"" + var_name +  "\"");
        else
            return {m->GetTag(var_name), true};    
    } else 
        return {createFemVarTag(m, dofmap, var_name), false};
}

void solve_stationary_diffusion(   
    INMOST::Mesh* m, Ani::FemSpace fem, 
    INMOST::Tag res_tag, 
    const std::function<bool(INMOST::Element e)> is_dirichlet, const std::function<void(INMOST::Element e, std::array<double, 3> X, Ani::ArrayView<> dirichlet_value_out)> dirichlet_func,
    const std::function<void(INMOST::Face f, std::array<double, 3> X, Ani::ArrayView<> neumann_value_out)> neumann_func,
    const std::function<Ani::TensorType(INMOST::Cell c, std::array<double, 3> X, Ani::DenseMatrix<> Dout)> diffusion_tensor,
    const std::function<void(INMOST::Cell c, std::array<double, 3> X, Ani::ArrayView<> Fout)> right_hand_side,
    int quad_order
    ){
    using namespace Ani;
    uint unf = fem.dofMap().NumDofOnTet();
    uint geom_mask = fem.dofMap().GetGeomMask();
    // Define memory requirements for elemental assembler
    PlainMemoryX<> mem_req;
    {
        auto grad_u = fem.getOP(GRAD), iden_u = fem.getOP(IDEN);
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        mem_req = fem3Dtet_memory_requirements<DfuncTraits<>>(grad_u, grad_u, quad_order);
        // mem_req.extend_size(fem3Dtet_memory_requirements<DfuncTraits<>>(iden_u, iden_u, quad_order));
        mem_req.extend_size(fem3Dtet_memory_requirements<DfuncTraits<>>(iden_p0, iden_u, quad_order));
        mem_req.extend_size(fem3Dface_memory_requirements<DfuncTraits<>>(iden_p0, iden_u, quad_order));
        mem_req.extend_size(fem.interpolateByDOFs_mem_req());
    }
    PlainMemory<> shrt_req = mem_req.enoughPlainMemory();

    // Define tensors from the problem
    auto K_tensor = [func = diffusion_tensor](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
        (void) iTet;
        Cell& c = *static_cast<Cell*>(user_data);
        auto t = func(c, X, Ani::DenseMatrix<>(Kdat, dims.first, dims.second));

        return t;
    };
    auto F_tensor = [func = right_hand_side](const Coord<> &X, double *F, TensorDims dims, void *user_data, int iTet) {
        (void) X; (void) dims; (void) user_data; (void) iTet;
        Cell& c = *static_cast<Cell*>(user_data);
        auto sz = dims.first*dims.second;
        func(c, X, ArrayView<>(F, sz));

        return sz == 1 ? Ani::TENSOR_SCALAR : Ani::TENSOR_GENERAL;
    };
    auto U_0 = [func = dirichlet_func](const Coord<> &X, double* res, ulong dim, void* user_data)->int{ 
        INMOST::Element& e = *static_cast<INMOST::Element*>(user_data);
        func(e, X, ArrayView<>(res, dim));
        return 0;
    }; 
    auto G_0 = [func = neumann_func](const Coord<> &X, double *g0, TensorDims dims, void *user_data, int iTet) {
        (void) iTet;
        Face& f = *static_cast<Face*>(user_data);
        auto sz = dims.first*dims.second;
        func(f, X, ArrayView<>(g0, sz));
        return sz == 1 ? Ani::TENSOR_SCALAR : Ani::TENSOR_GENERAL;
    };
    // Define user structure to store input data used in local assembler
    struct ProbLocData{
        const ElementArray<Node>* nodes;
        const ElementArray<Edge>* edges;
        const ElementArray<Face>* faces;
        const Cell* c;
        DofT::TetGeomSparsity sp;
        unsigned char is_neumann = 0;
        const unsigned char* node_permutation = nullptr;
    };
    auto bmrk = m->CreateMarker();
    if (bmrk == InvalidMarker())
        throw std::runtime_error("Can't create marker");
    m->MarkBoundaryFaces(bmrk);
    auto data_gatherer = [geom_mask, is_dirichlet, bmrk](ElementalAssembler& p)->void{
        double *nn_p = p.get_nodes();
        const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
        ProbLocData data;
        data.nodes = p.nodes;
        data.edges = p.edges;
        data.faces = p.faces;
        data.c = p.cell;
        if (geom_mask & DofT::NODE)
            for (unsigned i = 0; i < 4U; ++i) if (is_dirichlet((*p.nodes)[i]))
                data.sp.setNode(i);
        if (geom_mask & DofT::EDGE)
            for (unsigned i = 0; i < 6U; ++i) if (is_dirichlet((*p.edges)[i]))
                data.sp.setEdge(i);
        if (geom_mask & DofT::FACE)
            for (unsigned i = 0; i < 4U; ++i) {
                if (is_dirichlet((*p.faces)[i]))
                    data.sp.setFace(i);
                else if ((*p.faces)[i].GetMarker(bmrk))
                    data.is_neumann |= (1 << i);
            }
        if ((geom_mask & DofT::CELL) && is_dirichlet(*p.cell) )
            data.sp.setCell();
        data.node_permutation = p.node_permutation;

        p.compute(args, &data);
    };
    // define elemental assembler of local matrix and rhs
    std::function<void(const double**, double*, double*, double*, long*, void*)> local_assm =
            [&K_tensor, &F_tensor, &U_0, &G_0, order = quad_order, mem_req, shrt_req, unf, &fem, geom_mask](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data) -> void{
        auto& dat = *static_cast<ProbLocData*>(user_data);
        PlainMemory<> _mem = shrt_req;
        _mem.ddata = w, _mem.idata = reinterpret_cast<int*>(iw);
        PlainMemoryX<> mem = mem_req;
        mem.allocateFromPlainMemory(_mem);
        // read user_data
        Cell cell = *dat.c;
        DenseMatrix<>   A(Adat, unf, unf), F(Fdat, unf, 1), 
                        // B(w + shrt_req.dSize, unf, unf), 
                        G(w + shrt_req.dSize, unf, 1);
        A.SetZero(); 
        // B.SetZero();
        F.SetZero(); 

        Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);

        auto grad_u = fem.getOP(GRAD), iden_u = fem.getOP(IDEN);
        ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
        // elemental stiffness matrix <K grad(P1), grad(P1)>
        fem3Dtet<DfuncTraits<>>(XYZ, grad_u, grad_u, K_tensor, A, mem, order, &cell);
        // // elemental mass matrix <A P1, P1>
        // fem3Dtet<DfuncTraits<>>(XYZ, iden_u, iden_u, A_tensor, B, mem, order);
        // A += B;

        // elemental right hand side vector <F, P1>
        fem3Dtet<DfuncTraits<>>( XYZ, iden_p0, iden_u, F_tensor, F, mem, order, &cell);

        // apply Neumann and Robin BC
        for (int k = 0; k < 4; ++k) if (dat.is_neumann & (1 << k)) {
            Face f = (*dat.faces)[k];
            fem3Dface<DfuncTraits<>>( XYZ, k, iden_p0, iden_u, G_0, G, mem, order, &f);
            F += G;
        }
        
        //set dirichlet condition
        if (!dat.sp.empty()){
            ArrayView<> dof_values(w + shrt_req.dSize, unf);
            for (auto it = fem.dofMap().beginBySparsity(dat.sp); it != fem.dofMap().endBySparsity(); ++it){
                auto lo = *it;
                DOFLocus l = getDOFLocus(*(fem.dofMap().target<>()), lo, *dat.c, *dat.faces, *dat.edges, *dat.nodes, dat.node_permutation);
                fem.interpolateOnDOF(XYZ, U_0, dof_values, lo.gid, mem, &l.elem, order);
            }
            applyDirByDofs(*(fem.dofMap().target<>()), A, F, dat.sp, ArrayView<const double>(dof_values.data, dof_values.size));
        }
    };

    //define assembler
    Assembler discr(m);
    discr.SetMatRHSFunc(GenerateElemMatRhs(local_assm, unf, unf, shrt_req.dSize+unf*unf, shrt_req.iSize));
    {
        //create global degree of freedom enumenator
        auto Var0Helper = GenerateHelper(*fem.base());
        FemExprDescr fed;
        fed.PushTrialFunc(Var0Helper, "u");
        fed.PushTestFunc(Var0Helper, "phi_u");
        discr.SetProbDescr(std::move(fed));
    }
    discr.SetDataGatherer(data_gatherer);
    discr.PrepareProblem();
    Sparse::Matrix A("A");
    Sparse::Vector x("x"), b("b");
    discr.Assemble(A, b);
    Solver solver(Solver::INNER_ILU2);
    solver.SetParameterReal("absolute_tolerance", 1e-20);
    solver.SetParameterReal("relative_tolerance", 1e-10);
    solver.SetMatrix(A);
    solver.Solve(b, x);
    print_linear_solver_status(solver, "projection", true);

    //copy result to the tag and save solution
    discr.SaveSolution(x, res_tag);
    solver.Clear();
    discr.Clear();
    for (auto f = m->BeginFace(); f != m->EndFace(); ++f) f->RemMarker(bmrk);
    m->ReleaseMarker(bmrk);
}

void make_laplace(INMOST::Mesh* m, Ani::FemSpace fem, INMOST::Tag res_tag, const std::function<bool(INMOST::Element e)> is_dirichlet, const std::function<void(INMOST::Element e, std::array<double, 3> X, Ani::ArrayView<> dirichlet_value_out)> dirichlet_func, int order){
    solve_stationary_diffusion(m, fem, res_tag, is_dirichlet, dirichlet_func, 
        [](INMOST::Face, std::array<double, 3>, Ani::ArrayView<> out){ out.SetZero(); },
        [](INMOST::Cell, std::array<double, 3>, Ani::DenseMatrix<> Dout){ Dout.SetEye(1); return Ani::TENSOR_NULL; },
        [](INMOST::Cell, std::array<double, 3>, Ani::ArrayView<> Fout){ Fout.SetZero(); },
        order
    );
}