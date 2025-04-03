//
// Created by Liogky Alexey on 22.03.2025.
//

/**
 * Here we show how to operate with mesh functions disretized at quadrature points
 * 
 * This program generates and solves a system for L2 projection of quadrature-located point 
 * on some finite element space
 * 
 * \f[
 * \int_{\Omega} \mathbf{f}^h \cdot \mathbf{\phi}\ d^3\mathbf{x} = \int_{\Omega} \mathbf{f}^q \cdot \mathbf{\phi}\ d^3\mathbf{x}
 * \f]
 * where 
 * Ω = [0,1]^3, 
 * f^q correspond to f(x) = (x+y, z^2, e^{x+z})
 * 
 * There are two possible ways to store a mesh function at quadrature points:
 *  1) Based on a global node numbering, such as GlobalID,
 *  2) Based on cell-local node numbering (every cell have method getNodes()).
 * Here, the 1) option is demonstrated. In this case, a small overhead arises during matrix assembly due to 
 * the call to `copy_quadrature_data_from_proper_to_custom_order`. However, this approach simplifies working with mesh saving 
 * and reading (not all INMOST mesh formats correctly preserve the tetrahedron node sequence after saving and reloading).
 */

 #include "prob_args.h"

using namespace INMOST;

struct InputArgsQuadFunc: public InputArgs1Var{
    std::string mesh_file = "", func_name = "quad_f";

    uint parseArg(int argc, char* argv[], bool print_messages = true) override {
        #define GETARG(X)   if (i+1 < static_cast<uint>(argc)) { X }\
            else { if (print_messages) std::cerr << "ERROR: Not found argument after \"" << argv[i] <<"\"" << std::endl; exit(-1); }
        uint i = 0;
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mesh_file") == 0){
            GETARG(mesh_file = argv[++i];)
            return i+1; 
        } 
        if (strcmp(argv[i], "-fn") == 0 || strcmp(argv[i], "--func_name") == 0){
            GETARG(func_name = argv[++i];)
            return i+1; 
        } else 
            return InputArgs1Var::parseArg(argc, argv, print_messages);
        #undef GETARG   
    }
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const override{
        out << prefix << "mesh_file = \"" << mesh_file << "\"\n";
        out << prefix << "func_name = \"" << func_name << "\"\n";
        InputArgs1Var::print(out, prefix);
    }

protected:
    void printArgsDescr(std::ostream& out = std::cout, const std::string& prefix = "") override{
        out << prefix << "  -m , --mesh_file STR     <Set file with mesh containing quadrature mesh function, default=\"" << mesh_file << "\">\n";
        out << prefix << "  -fn, --func_name STR     <Name of tag storing quadrature mesh function, default=\"" << func_name << "\">\n";
        InputArgs::printArgsDescr(out, prefix);
    }
};

/// Reorder in-place quadrature data stored relative old_node_enumerator to be stored relative new_node_enumerator
void reorder_quadrature_function(Tag t, int quad_order, INMOST::Tag old_node_enumerator, INMOST::Tag new_node_enumerator, unsigned func_dim){
    auto formula = tetrahedron_quadrature_formulas(quad_order);
    assert(formula.IsSymmetric() && "reorder_quadrature_function currently support only symmetric quadraure point distributions");
    assert(old_node_enumerator.isDefined(INMOST::NODE) && new_node_enumerator.isDefined(INMOST::NODE) && "Wrong node enumerator");
    auto _csym = formula.GetSymmetryPartition();
    std::array<Ani::DofT::uint, 5> csym;
    for (int i = 0; i < 5; ++i) csym[i] = static_cast<Ani::DofT::uint>(_csym[i]);
    Ani::DofT::DofSymmetries sym({0}, {0}, {0}, {0}, {0}, csym);
    Ani::DofT::UniteDofMap dmap(sym);
    Ani::reorder_mesh_function_data({t}, dmap, old_node_enumerator, new_node_enumerator, func_dim);
}

/// Rearrange cell_local_data, which is ordered properly, to be ordered relative to node_ids order
template<typename iterator1, typename iterator2>
void reordered_copy_data_of_quadrature_function(std::array<long, 4> node_ids, int quad_order, iterator1 cell_local_data, iterator2 to, unsigned func_dim){
    auto formula = tetrahedron_quadrature_formulas(quad_order);
    assert(formula.IsSymmetric() && "reordered_copy_data_of_quadrature_function currently support only symmetric quadraure point distributions");
    auto sym = formula.GetSymmetryPartition();
    bool comp_node_perm = ((abs(sym[1]) + abs(sym[2]) + abs(sym[3]) + abs(sym[4])) != 0);
    std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
    if (comp_node_perm)
        canonical_node_indexes = Ani::createOrderPermutation(node_ids.data());
    std::size_t shift = 0, sz = func_dim;
    for (unsigned ist = 0; ist < sym.size(); ++ist) if (sym[ist] > 0){
        auto vol = Ani::DofT::DofSymmetries::symmetry_volume(Ani::DofT::CELL, ist);
        for (int lsid = 0; lsid < vol; ++lsid){
            auto reordered_lsid = Ani::DofT::DofSymmetries::index_on_reorderd_elem(Ani::DofT::CELL, 0, ist, lsid, canonical_node_indexes.data());
            for (int is = 0; is < sym[ist]; ++is)
                std::copy(cell_local_data + shift + sz*(is*vol + reordered_lsid), cell_local_data + shift + sz*(is*vol + reordered_lsid+1) , to + shift + sz*(is*vol + lsid));
        }
        shift += sz * sym[ist] * vol;
    }
}
/// Copy data from 'from' to 'to' changing ordering of nodes from proper (relative GlobalID) to custom (same as nodes ordering in 'nds') 
void copy_quadrature_data_from_proper_to_custom_order(const double* from, double* to, int quad_order, const Mesh* mlink, const INMOST::HandleType* nds/*[4]*/, unsigned func_dim){
    std::array<long, 4> gni;
    for (int i = 0; i < 4; ++i)
        gni[i] = INMOST::Node(const_cast<INMOST::Mesh*>(mlink), nds[i]).GlobalID();
    reordered_copy_data_of_quadrature_function(gni, quad_order, from, to, func_dim);
}
/// Copy data from 'from' to 'to' changing ordering of nodes from custom (same as nodes ordering in 'nds')  to proper (relative GlobalID) 
void copy_quadrature_data_from_custom_to_proper_order(const double* from, double* to, int quad_order, const Mesh* mlink, const INMOST::HandleType* nds/*[4]*/, unsigned func_dim){
    std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
    std::array<long, 4> gni;
    for (int i = 0; i < 4; ++i)
        gni[i] = INMOST::Node(const_cast<INMOST::Mesh*>(mlink), nds[i]).GlobalID();
    canonical_node_indexes = Ani::createOrderPermutation(gni.data());
    std::array<long, 4> rev_ids;
    for (int i = 0; i < 4; ++i)
        rev_ids[canonical_node_indexes[i]] = i;
    reordered_copy_data_of_quadrature_function(rev_ids, quad_order, from, to, func_dim);
}

int main(int argc, char* argv[]){
    InputArgsQuadFunc p;
    std::string prob_name = "quad_func";
    p.save_prefix = prob_name + "_out";
    p.USpace = "P2";
    p.max_quad_order = 5;
    p.parseArgs_mpi(&argc, &argv);

    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);
    if (pRank == 0) p.print();
    
    {
        std::unique_ptr<Mesh> mptr;
        Tag quad_func;
        unsigned quad_order = p.max_quad_order;
        static const uint FuncDim = 3;
        auto f_func = [](std::array<double, 3> x)->std::array<double, 3>{ return {x[0]+x[1], x[2]*x[2], exp(x[0]+x[2])}; };

        auto formula = tetrahedron_quadrature_formulas(quad_order);
        auto nq = formula.GetNumPoints();
        bool from_file = false; //< mode of the program
        if (p.mesh_file.empty()){
            if (pRank == 0) std::cout << "Generate cube mesh [0, 1]^3" << std::endl;
            mptr = GenerateCube(INMOST_MPI_COMM_WORLD, p.axis_sizes, p.axis_sizes, p.axis_sizes);
            mptr->AssignGlobalID(NODE|EDGE|FACE|CELL);

            quad_func = mptr->CreateTag(p.func_name, DATA_REAL, CELL, NONE, FuncDim*nq);
            for (auto c = mptr->BeginCell(); c != mptr->EndCell(); ++c){
                auto nds = c->getNodes();
                auto r = c->RealArray(quad_func);
                std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
                std::array<long, 4> gni;
                for (int i = 0; i < 4; ++i)
                    gni[i] = nds[i].GlobalID();
                canonical_node_indexes = Ani::createOrderPermutation(gni.data());

                double P[12]{};
                for (int n = 0; n < 4; ++n)
                    for (int k = 0; k < 3; ++k)
                        P[3*canonical_node_indexes[n] + k] = nds[n].Coords()[k];
            
                for (int q = 0; q < formula.GetNumPoints(); ++q){
                    auto wp = formula.GetPointWeight(q);
                    std::array<double, 3> x{0, 0, 0};
                    for (int d = 0; d < 3; ++d)
                        x[d] = wp.p[0] * P[3*0 + d] + wp.p[1] * P[3*1 + d] + wp.p[2] * P[3*2 + d] + wp.p[3] * P[3*3 + d];
                    INMOST_DATA_REAL_TYPE* f = r.data() + FuncDim*q;
                    auto f_v = f_func(x);
                    std::copy(f_v.data(), f_v.data() + std::min(FuncDim, 3U), f);
                }
            }
            from_file = false;
        } else { //< read quadrature mesh function from the mesh
            mptr = std::make_unique<Mesh>("mesh_from_file");
            mptr->SetCommunicator(INMOST_MPI_COMM_WORLD);
            if (mptr->isParallelFileFormat(p.mesh_file)){ //< read the mesh in parallel mode
                if (pRank == 0) std::cout << "Read mesh from parallel file \"" << p.mesh_file << "\"" << std::endl;
                mptr->SetFileOption("PMF_DUP_GID", "1"); //< load stored node enumeration
                mptr->Load(p.mesh_file);
            } else if (pRank == 0){ //< read the mesh in sequential mode
                std::cout << "Read mesh from sequential file \"" << p.mesh_file << "\"" << std::endl;
                mptr->Load(p.mesh_file);
            }
            {   //remove internal tags if it have
                std::vector<std::string> tmp_tag_names;
                mptr->ListTagNames(tmp_tag_names);
                for (const auto& name: tmp_tag_names)
                    if ((name.size() > 0 && name[0] == '_') || (name.size() > 18 && name.substr(0, 18) == "IGlobEnumeration::"))
                    mptr->DeleteTag(mptr->GetTag(name));
            }
            if (!mptr->HaveTag("DUP_GLOBAL_ID")){ //< have only for pmf
                Tag dup_gid = mptr->CreateTag("DUP_GLOBAL_ID", DATA_INTEGER, NODE, NONE, 1);
                mptr->Enumerate(NODE, dup_gid); //< save node enumeration
            }
            RepartMesh(mptr.get()); //< divide the mesh between processors
            mptr->AssignGlobalID(NODE|EDGE|FACE|CELL);

                if (!mptr->HaveTag(p.func_name))
                    throw std::runtime_error("Loaded mesh from file \"" + p.mesh_file + "\" does't have tag \"" + p.func_name + "\"");
                // if (!mptr->HaveTag("DUP_GLOBAL_ID"))
                //     throw std::runtime_error("Impossible properly reorder quadrature data if we don't know the nodal ordering");
            quad_func = mptr->GetTag(p.func_name);
                if (!quad_func.isDefined(CELL)) 
                    throw std::runtime_error("Loaded mesh function \"" + p.func_name + "\" doesn't defined on CELL");
                if (quad_func.GetDataType() != DATA_REAL)
                    throw std::runtime_error("Mesh function must be real valued but it store \"" + std::string(INMOST::DataTypeName(quad_func.GetDataType())) + "\"");
                if (quad_func.GetSize() != FuncDim*nq)
                    for (auto c = mptr->BeginCell(); c != mptr->EndCell(); ++c)
                        if (c->RealArray(quad_func).size() != FuncDim*nq)
                            throw std::runtime_error("Quadrature mesh function should store exactly "+ std::to_string(FuncDim*nq) + " values on every cell");
            Tag old_gid = mptr->GetTag("DUP_GLOBAL_ID");
                if (old_gid.GetDataType() != DATA_INTEGER || !old_gid.isDefined(NODE))
                    throw std::runtime_error("Wrong \"DUP_GLOBAL_ID\" tag");
            reorder_quadrature_function(quad_func, quad_order, old_gid, mptr->GlobalIDTag(), FuncDim);
            
            // we don't need old_gid anymore
            mptr->DeleteTag(old_gid);
            
            from_file = true;
        }
        Mesh* m = mptr.get();
        m->ExchangeGhost(1,NODE);
        print_mesh_sizes(m);
        
        using namespace Ani;
        //generate FEM space from it's name
        FemSpace fem = choose_space_from_name(p.USpace)^FuncDim;
        uint unf = fem.dofMap().NumDofOnTet();
        struct ProbLocData{
            ArrayView<> quad_func_data;
        };
        auto data_gatherer = [nq, quad_func, quad_order](ElementalAssembler& p)->void{
            double *nn_p = p.get_nodes();
            const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
            ProbLocData data;
            auto quad_cell_sz = FuncDim*nq;
            auto data_chunk = p.pool->alloc(quad_cell_sz, 0, 0);
            data.quad_func_data = ArrayView<>(data_chunk.m_mem.ddata, quad_cell_sz);
            auto arr = p.cell->RealArray(quad_func);
            copy_quadrature_data_from_proper_to_custom_order(arr.data(), data.quad_func_data.data, quad_order, p.m, p.nodes->data(), FuncDim);

            p.compute(args, &data);
        };
        auto F_tensor = [](ArrayView<> X, ArrayView<> D, TensorDims Ddims, void *user_data, const AniMemory<>& mem) -> TensorType{
            (void) X; (void) mem;
            auto nq = mem.q; //< count of quadrature points
            ProbLocData& dat = *static_cast<ProbLocData*>(user_data);
            for (std::size_t q = 0; q < nq; ++q){ //< cycle over all quadrature points
                DenseMatrix<> Dloc(D.data + Ddims.first*Ddims.second*q, Ddims.first, Ddims.second);
                for (std::size_t d = 0; d < FuncDim; ++d)
                    Dloc[d] = dat.quad_func_data[q*FuncDim + d];
            }
            
            return TENSOR_GENERAL;
        };
        std::function<void(const double**, double*, double*, double*, long*, void*, DynMem<double, long>*)> local_assm = 
            [fem, unf, quad_order, F_tensor](const double** XY/*[4]*/, double* Adat, double* Fdat, double* w, long* iw, void* user_data, Ani::DynMem<double, long>* fem_alloc){
            (void) w, (void) iw;
            DenseMatrix<> A(Adat, unf, unf), F(Fdat, unf, 1);
            A.SetZero(); F.SetZero();
            auto adapt_alloc = makeAdaptor<double, int>(*fem_alloc);
            Ani::Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);
            auto iden_u = fem.getOP(IDEN);  
            ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;

            fem3Dtet<DfuncTraits<TENSOR_NULL, true>>(XYZ, iden_u, iden_u, TensorNull<>, A, adapt_alloc, quad_order); //< we can use here other quad_order
            // elemental right hand side vector <F, P2>
            fem3Dtet<DfuncTraitsFusive<>>( XYZ, iden_p0, iden_u, F_tensor, F, adapt_alloc, quad_order, user_data); //< we must use here quad_order corresponding to quadrature mesh function
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

        // Define tag to store result
        Tag res_tag = createFemVarTag(m, *fem.dofMap().target<>(), "res");
        Sparse::Matrix A("A");
        Sparse::Vector x("x"), b("b");
        discr.Assemble(A, b);
        Solver solver(Solver::INNER_ILU2);
        solver.SetParameterReal("absolute_tolerance", 1e-20);
        solver.SetParameterReal("relative_tolerance", 1e-10);
        solver.SetMatrix(A);
        solver.Solve(b, x);
        print_linear_solver_status(solver, prob_name, true);

        //copy result to the tag and save solution
        discr.SaveSolution(x, res_tag);

        // if (!from_file)//< compare result with usual l2 projection of analytical function
        {   
            (void) from_file;
            Tag comp_tag = createFemVarTag(m, *fem.dofMap().target<>(), "cmp");
            make_l2_project(
                [f_func](INMOST::Cell c, std::array<double, 3> X, double* res){
                    (void) c; 
                    auto f = f_func(X); 
                    std::copy(f.data(), f.data() + std::min(f.size(), std::size_t(FuncDim)), res);
                }, m, comp_tag, fem, quad_order);
            auto err = [res_tag, comp_tag, &discr, &fem](INMOST::Cell c, std::array<double, 3> X)->double{
                std::array<double, 3> res, comp;
                eval_op_var_at_point(res.data(), c, X, fem.getOP(IDEN), discr, res_tag, nullptr, 0);
                eval_op_var_at_point(comp.data(), c, X, fem.getOP(IDEN), discr, comp_tag, nullptr, 0);
                auto sq = [](auto x){return x*x; };
                return sq(res[0] - comp[0]) + sq(res[1] - comp[1]) + sq(res[2] - comp[2]);
            };
            double L2nrm = sqrt(integrate_scalar_func(m, err, quad_order));
            if (pRank == 0)
                std::cout << "||f^{q,h} - f^h||_L2 = " << L2nrm << "\n";
        }

        {   //don't save internal tags
            std::vector<std::string> tmp_tag_names;
            m->ListTagNames(tmp_tag_names);
            for (const auto& name: tmp_tag_names)
                if ((name.size() > 0 && name[0] == '_') || (name.size() > 18 && name.substr(0, 18) == "IGlobEnumeration::"))
                m->SetFileOption("Tag:" + name, "nosave,noload");
        }
        m->Save(p.save_dir + p.save_prefix + ".pvtu");

        solver.Clear();
        discr.Clear();
        m->Clear();
        mptr.reset();
    }

    InmostFinalize();

    return 0;
}