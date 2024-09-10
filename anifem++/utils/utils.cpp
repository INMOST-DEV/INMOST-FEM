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