//
// Created by Liogky Alexey on 16.12.2022.
//

#include "inmost.h"
#include "example_common.h"
#include "carnum/MeshGen/mesh_gen.h"
#include "carnum/Fem/AniInterface/Fem3Dtet.h"
#include "carnum/Fem/AniInterface/ForAssembler.h"
#include "carnum/Fem/Assembler.h"
#include <numeric>

#if __cplusplus >= 201703L
#include <filesystem>
using namespace std::filesystem;
#else
#include <experimental/filesystem>
using namespace std::experimental::filesystem;
#endif


/** The program solves the nonstationary diffusion problem
 * \f[
 * \left\{ 
 * \begin{aligned}
 *   \frac{\partial u}{\partial t} &= \mathrm{div}\ \mathbf{K}\ &\mathrm{grad}\ u\ - &\mathbf{A} u + \mathbf{F}\ in\ \Omega \otimes [0, T] \\
 *                             &                    &           u &= U_0\        at\  t = 0\ in\  \Omega\\
 *                             &                    &           u &= G_D\        on\  \Gamma_D \otimes [0, T]\\
 *            &\mathbf{K} \frac{du}{d\mathbf{n}}    &             &= G_N\        on\  \Gamma_N \otimes [0, T]\\
 *            &\mathbf{K} \frac{du}{d\mathbf{n}}\ + &\mathbf{S} u &= G_R\        on\  \Gamma_R \otimes [0, T]\\
 * \end{aligned}
 * \f]
 * where Ω = [0,1]^3, Γ_D = {0}x[0,1]^2, Γ_R = {1}x[0,1]^2, Γ_N = ∂Ω \ (Γ_D ⋃ Γ_R)
 * The user-defined coefficients are
 *    K(x,t)   - positive definite tensor
 *    A(x,t)   - non-negative reaction
 *    F(x,t)   - right-hand side term
 *    G_D(x,t) - essential (Dirichlet) boundary condition
 *    G_N(x,t) - Neumann boundary condition
 *    G_R(x,t) - Robin boundary condition
 *    S(x,t)   - Robin boundary coefficient
 *    U_0(x)   - initial condition
 *
 *  For space discretization P1 finite elements are used. 
 *  Time discretization expressed by explicit or implicit scheme (implemented both variants)
 * 
 **/

using namespace Ani;
struct DiffProbParams{
    struct DiffProbTensors{
        using ICTensor = std::function<double(const Coord<> X, void *user_data)>;
        using BCTensor = std::function<double(const double t, const Coord<> &X, void *user_data)>;
        using DynamicTensor = std::function<TensorType(const double t, const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet)>;
        DynamicTensor   K, A, F, S;
        BCTensor        G_D, G_N, G_R;
        ICTensor        U_0;

        DiffProbTensors(){
            K = [](const double t, const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
                (void) t; (void) X; (void) user_data; (void) iTet;
                Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
                K.SetZero();
                K(0, 0) = K(1, 1) = K(2, 2) = 1;
                K(0, 1) = K(1, 0) = -1;
                return Ani::TENSOR_SYMMETRIC;
            };
            A = [](const double t, const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet) {
                (void) t; (void) X; (void) dims; (void) user_data; (void) iTet;
                dat[0] = 1;
                return Ani::TENSOR_SCALAR;
            };
            F = [](const double t, const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet) {
                (void) t; (void) X; (void) dims; (void) user_data; (void) iTet;
                dat[0] = 1;
                return Ani::TENSOR_SCALAR;
            };
            S = [](const double t, const Coord<> &X, double *dat, TensorDims dims, void *user_data, int iTet) {
                (void) t; (void) X; (void) dims; (void) user_data; (void) iTet;
                dat[0] = 1;
                return Ani::TENSOR_SCALAR;
            };
            G_D = [](const double t, const Coord<> &X, void *user_data){
                (void) t; (void) user_data;
                return X[0] + X[1] + X[2];
            };
            G_N = [](const double t, const Coord<> &X, void *user_data){
                (void) t; (void) user_data;
                return X[0] - X[1];
            };
            G_R = [](const double t, const Coord<> &X, void *user_data){
                (void) t; (void) user_data;
                return X[0] + X[2];
            };
            U_0 = [](const Coord<> &X, void *user_data){
                (void) X; (void) user_data;
                return 0.0;
            };
        } 
    };
    enum TimeScheme{
        EXPLICIT_SCHEME = 0,
        IMPLICIT_SCHEME = 1,
        NUM_SCHEMES
    };

    DiffProbTensors tensors = DiffProbTensors();
    std::array<unsigned, 3> axis_sizes{8, 8, 8}; ///< cube axis partition
    TimeScheme time_scheme = IMPLICIT_SCHEME;
    double T = 10, dt = 0.5;
    std::string lin_sol_db = "";
    std::string save_dir = "", save_prefix = "dyndiff";
    std::string lin_sol_nm = "inner_mptiluc", lin_sol_prefix = "";

    void parseArgs(int argc, char* argv[], bool print_messages = true){
        #define GETARG(X)   if (i+1 < argc) { X }\
                        else { if (print_messages) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
        auto is_double = [](const std::string& s) -> std::pair<bool, double>{
            std::istringstream iss(s);
            double f = NAN;
            iss >> f; 
            return {iss.eof() && !iss.fail(), f};    
        };       
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                if (print_messages) printArgsHelpMessage();
                #if defined(USE_MPI)
                    int inited = 0;
                    MPI_Initialized(&inited);
                    if (inited) MPI_Finalize();
                #endif
                exit(0);
            } else if (strcmp(argv[i], "-sz") == 0 || strcmp(argv[i], "--sizes") == 0) {
                unsigned j = 0;
                for (; j < axis_sizes.size() && i+1 < argc && is_double(argv[i+1]).first; ++j)
                    axis_sizes[j] = is_double(argv[++i]).second;
                if (j != axis_sizes.size()) throw std::runtime_error("Waited " + std::to_string(axis_sizes.size()) + " arguments for command \"" + std::string(argv[i-j]) + "\" but found only " + std::to_string(j));    
                continue;
            } else if (strcmp(argv[i], "-nm") == 0 || strcmp(argv[i], "--name") == 0) {
                GETARG(save_prefix = argv[++i];)
                continue;
            } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--target") == 0) {
                GETARG(save_dir = argv[++i];)
                if (save_dir.back() != '/') save_dir += '/';
                continue;
            } else if (strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "--time_step") == 0) {
                GETARG(dt = std::stod(argv[++i]);)
                continue;
            } else if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--time_end") == 0) {
                GETARG(T = std::stod(argv[++i]);)
                continue;
            } else if (strcmp(argv[i], "-ts") == 0 || strcmp(argv[i], "--tmscheme") == 0) {
                GETARG(
                    auto ti = std::stoi(argv[++i]);
                    if (ti < 0 || ti >= NUM_SCHEMES) throw std::runtime_error("Wrong time scheme number");
                    else time_scheme = static_cast<TimeScheme>(ti);
                    )
                continue;
            } else if (strcmp(argv[i], "-db") == 0 || strcmp(argv[i], "--lnslvdb") == 0) {
                GETARG(lin_sol_db = argv[++i];)
                continue;
            } else if (strcmp(argv[i], "-ls") == 0 || strcmp(argv[i], "--lnslv") == 0) {
                std::array<std::string*, 2> plp = {&lin_sol_nm, &lin_sol_prefix};
                unsigned j = 0;
                for (; j < plp.size() && i+1 < argc; ++j)
                    *(plp[j]) = argv[++i];    
                continue;
            } else {
                if (print_messages) {
                    std::cerr << "Faced unknown command \"" << argv[i] << "\"" << "\n";
                    printArgsHelpMessage();
                }
                exit(-1);
            }
        }
        #undef GETARG
        if (!save_dir.empty()){
            path p(save_dir);
            if (!exists(status(p)) || !is_directory(status(p))){
                if (print_messages) std::cerr << ("save_dir = \"" + save_dir + "\" is not exists") << std::endl;
                exit(-2);
            }
        }
        if (!save_dir.empty() && save_dir.back() != '/') save_dir += "/";
    }
    static DiffProbParams FromInputArgs(int* argc, char** argv[]){
        int pRank = 0; 
    #if defined(USE_MPI)
        int inited = 0;
        MPI_Initialized(&inited);
        if (!inited) MPI_Init(argc, argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
    #endif
        DiffProbParams p;
        p.parseArgs(*argc, *argv, pRank == 0);
        return p;
    }


    static void printArgsHelpMessage(std::ostream& out = std::cout, const std::string& prefix = ""){
        out << prefix << "Help message: " << "\n"
            << prefix << " Command line options: " << "\n"
            << prefix << "  -sz, --sizes     IVAL[3] <Sets the number of segments of the cube [0; 1]^3 partition along the coordinate axes Ox, Oy, Oz, default=8 8 8>" << "\n"
            << prefix << "  -t , --target    PATH    <Directory to save results, default=\"\">" << "\n"
            << prefix << "  -nm, --name      STR     <Prefix for saved results, default=\"dyndiff\">" << "\n"
            << prefix << "  -dt, --time_step DVAL    <Specify time step, default=0.5>\n"  
            << prefix << "  -T , --time_end  DVAL    <Specify final time, default=10>\n"
            << prefix << "  -ts, --tmscheme  IVAL    <Choose time discretization scheme: 0 - EXPLICIT, 1 - IMPLICIT, default=IMPLICIT>\n" 
            << prefix << "  -db, --lnslvdb   FILE    <Specify linear solver data base, default=\"\">\n"
            << prefix << "  -ls, --lnslv     STR[2]  <Set linear solver name and prefix, default=\"inner_mptiluc\" \"\">\n"
            << prefix << "  -h , --help              <Print this message and exit>" << "\n";
    }
    void print(std::ostream& out = std::cout, const std::string& prefix = "") const {
        out << prefix << "mesh axis partition = " << axis_sizes[0] << " x " << axis_sizes[1] << " x " << axis_sizes[2] << "\n"
            << prefix << "save_dir  = \"" << save_dir << "\" save_prefix = \"" << save_prefix << "\"\n";
        out << prefix << "time_scheme = ";
        switch (time_scheme) {
            case EXPLICIT_SCHEME: out << "EXPLICIT"; break;
            case IMPLICIT_SCHEME: out << "IMPLICIT"; break;   
            default:              out << "UNKNOWN" ; break;
        }    
        out << ": T = " << T << " with dt = " << dt << "\n"
            << prefix << "linsol = \"" << lin_sol_nm << "\" prefix = \"" << lin_sol_prefix  << "\" database = \"" << lin_sol_db << "\"" << "\n";     
    }
    friend std::ostream& operator<<(std::ostream& out, const DiffProbParams& p);
};
std::ostream& operator<<(std::ostream& out, const DiffProbParams& p){ return p.print(out), out; }

using namespace INMOST;
struct DynDiffProblem{
    using UFem = FemFix<FEM_P1>;
    static const int BC_DIRICHLET_LBL = (1 << 0);
    static const int BC_NEUMANN_LBL = (1 << 1);
    static const int BC_ROBIN_LBL = (1 << 2);

    // User structure for local assembler
    struct ProbLocData{
        using StaticTensor = std::function<TensorType(const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet)>;
        std::array<int, 4> nlbl;
        std::array<int, 4> flbl;

        double dt, t;
        ArrayView<> udofs, udofs_prev;
        ArrayView<const double> XY;

        void print(std::ostream& out = std::cout) const {
            out << "nlbl = " << DenseMatrix<const int>(nlbl.data(), 1, 4)
                << "flbl = " << DenseMatrix<const int>(flbl.data(), 1, 4) << "\n"
                << "udofs = " << DenseMatrix<>(udofs.data, 1, udofs.size) << "\n"
                << "udofs_prev = " << DenseMatrix<>(udofs.data, 1, udofs.size) << "\n";
            out << "XY = \n" << DenseMatrix<const double>(XY.data, 3, XY.size/3) << "\n";
        }
        friend std::ostream& operator<<(std::ostream& out, const ProbLocData& p);
        static StaticTensor make_tensor_wrapper(const DiffProbParams::DiffProbTensors::DynamicTensor& t){
            return [&t](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                auto dat = static_cast<ProbLocData*>(user_data);
                return t(dat->t, X, out, dims, user_data, iTet);
            };
        }
        static StaticTensor make_tensor_wrapper(const DiffProbParams::DiffProbTensors::BCTensor& t){
            return [&t](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                (void) dims; (void) iTet;
                auto dat = static_cast<ProbLocData*>(user_data);
                out[0] = t(dat->t, X, user_data);
                return TENSOR_SCALAR;
            };
        }
        static StaticTensor make_tensor_wrapper(const DiffProbParams::DiffProbTensors::ICTensor& t){
            return [&t](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                (void) dims; (void) iTet;
                out[0] = t(X, user_data);
                return TENSOR_SCALAR;
            };
        }
        static StaticTensor make_tensor_wrapper_prev(const DiffProbParams::DiffProbTensors::DynamicTensor& t){
            return [&t](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                auto dat = static_cast<ProbLocData*>(user_data);
                return t(dat->t - dat->dt, X, out, dims, user_data, iTet);
            };
        }
        static StaticTensor make_tensor_wrapper_prev(const DiffProbParams::DiffProbTensors::BCTensor& t){
            return [&t](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                (void) dims; (void) iTet;
                auto dat = static_cast<ProbLocData*>(user_data);
                out[0] = t(dat->t - dat->dt, X, user_data);
                return TENSOR_SCALAR;
            };
        }
        double getU(const Coord<> &X) const {
            double U = 0;
            ArrayView<> Ua(&U, 1);
            fem3DapplyX<Operator<IDEN, UFem>>(XY.data, XY.data+3, XY.data+6, XY.data+9, X.data(), udofs, Ua);
            return U;
        }
        double getUprev(const Coord<> &X) const {
            double U = 0;
            ArrayView<> Ua(&U, 1);
            fem3DapplyX<Operator<IDEN, UFem>>(XY.data, XY.data+3, XY.data+6, XY.data+9, X.data(), udofs_prev, Ua);
            return U;
        }
        std::array<double, 3> getGradUprev(const Coord<> &X) const {
            std::array<double, 3> grUp = {0};
            ArrayView<> grU(grUp.data(), grUp.size());
            fem3DapplyX<Operator<GRAD, UFem>>(XY.data, XY.data+3, XY.data+6, XY.data+9, X.data(), udofs_prev, grU); 
            return grUp;
        };

    };

    DynDiffProblem(DiffProbParams p): m_p{p}, discr(nullptr) {
#if defined(USE_MPI)
        MPI_Comm_rank(INMOST_MPI_COMM_WORLD, &pRank);
        MPI_Comm_size(INMOST_MPI_COMM_WORLD, &pCount);
#endif
    }
    DynDiffProblem& GenerateMesh(bool print_size = true){
        if (pRank == 0) std::cout << "--- Generate mesh for cube domain ---" << std::endl;
        m = GenerateCube(INMOST_MPI_COMM_WORLD, m_p.axis_sizes[0], m_p.axis_sizes[1], m_p.axis_sizes[2]);
        m->ExchangeGhost(1,NODE);
        long nN = m->TotalNumberOf(NODE), nE = m->TotalNumberOf(EDGE), nF = m->TotalNumberOf(FACE), nC = m->TotalNumberOf(CELL);
        if (pRank == 0 && print_size) {
            std::cout << "Mesh info:"
                      << " #N " << nN
                      << " #E " << nE
                      << " #F " << nF
                      << " #T " << nC << std::endl;
        }
        bnd = m->CreateTag("lbl", DATA_INTEGER, NODE|FACE, NONE, 1);
        for (auto it = m->BeginNode(); it != m->EndNode(); ++it) 
            it->Integer(bnd) = 0;
        for (auto it = m->BeginFace(); it != m->EndFace(); ++it){
            auto nodes = it->getNodes();
            it->Integer(bnd) = 0;
            if (!it->Boundary()) continue;
            std::array<int, 6> r = {0};
            for (unsigned n = 0; n < nodes.size(); ++n)
                for (int i = 0; i < 3; ++i) {
                    if (fabs(nodes[n]->Coords()[i] - 0.0) < 10*DBL_EPSILON) r[i]++;
                    else if (fabs(nodes[n]->Coords()[i] - 1.0) < 10*DBL_EPSILON) r[i+3]++;
                }
            for (int i = 0; i < 6; ++i)
                if (r[i] == 3) {
                    if (i == 0)
                        it->Integer(bnd) = BC_DIRICHLET_LBL;
                    else if (i == 3)
                        it->Integer(bnd) = BC_ROBIN_LBL;
                    else
                        it->Integer(bnd) = BC_NEUMANN_LBL;
                }
            for (unsigned n = 0; n < nodes.size(); ++n)
                nodes[n].Integer(bnd) |= it->Integer(bnd);
        }
        discr.m_mesh = m.get();
        return *this;
    }
    DynDiffProblem& SetProblemDefinition(){
        st_time = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now());
        if (pRank == 0) std::cout << "--- Allocate storage for FEM variables ---" << std::endl;
        {
            auto Var0Helper = GenerateHelper<UFem>();
            auto amask = Var0Helper->GetGeomMask();
            ElementType et = 0;
            for (unsigned i = 0; i < amask.size(); ++i) 
                if (amask[i]) 
                    et |= ElementTypeFromDim(i);
            u = m->CreateTag("u", DATA_REAL, et, NONE, 1);
            uprev = m->CreateTag("uprev", DATA_REAL, et, NONE, 1);

            FemExprDescr fed;
            fed.PushVar(Var0Helper, "u");
            fed.PushTestFunc(Var0Helper, "phi_u");
            discr.SetProbDescr(std::move(fed));
        }
        if (pRank == 0) std::cout << "--- Set initial condition ---" << std::endl;
        {
            ElementType et = 0;
            for (int i = 0; i < 4; ++i) 
                if (u.isDefinedByDim(i))
                    et |= ElementTypeFromDim(i);
            for(auto it = m->BeginElement(et); it != m->EndElement(); ++it){
                auto nds = it->getNodes();
                std::array<double, 3> X = {0, 0, 0};
                for (unsigned ni = 0; ni < nds.size(); ++ni) 
                    for (int k = 0; k < 3; ++k)
                        X[k] += nds[ni].Coords()[k];
                if (nds.size()) 
                    for (int k = 0; k < 3; ++k) 
                        X[k] /= nds.size();
                it->Real(uprev) = it->Real(u) = m_p.tensors.U_0(X, nullptr);
            }
        }
        if (pRank == 0) std::cout << "--- Set problem definition ---" << std::endl;
        std::size_t elem_nfa = Operator<IDEN, UFem>::Nfa::value;
        discr.SetMatRHSFunc(GenerateElemMatRhs(get_local_assembler(m_p.tensors, m_p.time_scheme), elem_nfa, elem_nfa));
        discr.SetDataGatherer(get_local_data_gatherer());
        discr.PrepareProblem();
        discr.pullInitValFrom(u);

        return *this;
    }
    DynDiffProblem& InitLinSolver(){
        ls = std::make_shared<INMOST::Solver>(m_p.lin_sol_nm, m_p.lin_sol_prefix);
        return *this;
    }
    bool Solve(bool save_temp_meshes = true){
        auto save_temporary_meshes = [&](){
            if (save_temp_meshes) {
                std::string save_name = m_p.save_dir + m_p.save_prefix + "_" + std::to_string(t_cur);
                m->Save(save_name + ".pvtu"); 
                if (pRank == 0) std::cout << "Intermediate result saved in \"" << save_name << ".pvtu\"" << std::endl;
            } 
        };

        if (pRank == 0) std::cout << "--- Time loop ---" << std::endl;
        t_cur = 0;
        save_temporary_meshes();   
        discr.SaveSolution(u, x);
        double dt = m_p.dt;
        while(abs(t_cur - m_p.T) > 1e-9*abs(m_p.T)){
            dt_cur = dt;
            if (t_cur + dt > m_p.T) dt_cur = m_p.T - t_cur;
            t_cur += dt_cur;
            if (pRank == 0) {
                auto loc_time = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now());
                std::cout << "\n    t = " << t_cur << ":" << " working time = " <<  std::chrono::duration<double, std::ratio<1,1>>(loc_time - st_time).count() << "s" << std::endl;
            }
            if (pRank == 0) std::cout << "--- Start assemble linear system ---" << std::endl;
            discr.Assemble(A, b);
            if (pRank == 0) std::cout << "--- Compute preconditioner ---" << std::endl;
            ls->SetMatrix(A);
            if (pRank == 0) std::cout << "--- Start solve linear system ---" << std::endl;
            slvFlag = ls->Solve(b, x); 
            printLSolverStatus("nonstationary diffusion");
            discr.SaveSolution(x, u);
            discr.CopyVar(0, u, uprev);
            save_temporary_meshes();
        }
        return slvFlag;
    }
    void printLSolverStatus(std::string probName = "", bool exit_on_false = true){
        auto print_ls_state = [](Solver& s, const std::string& prefix = "", std::ostream& out = std::cout){
            //add spaces to the number of iterations string up to 4 characters
            std::string _s_its = std::to_string(s.Iterations());
            std::string s_its(std::max(4U, static_cast<unsigned>(_s_its.size())), ' ');
            std::copy(_s_its.begin(), _s_its.end(), s_its.begin() + s_its.size() - _s_its.size());
            out << prefix << "#LinIts = " << s_its << " Residual " << s.Residual() << ", "
                << "Precond time = " << s.PreconditionerTime() << "s, Solve time = " << s.IterationsTime() << "s\n";
        };
        if(!slvFlag){
            if (pRank == 0) std::cout << probName << ": solution failed\n";
            for (int p = 0; p < pCount; ++p) {
                BARRIER;
                if (pRank != p) continue;
                print_ls_state(*ls, "\t");
                std::cout << "\tRank " << pRank << " failed to solve system. ";
                std::cout << "Reason: " << ls->GetReason() << std::endl;
            }
            if (exit_on_false) exit(-1);
        }
        else if (pRank == 0){
            std::cout << probName << ": solved_succesful\n";
            print_ls_state(*ls, "\t");
        }
    }
    std::function<void (ElementalAssembler &p)> get_local_data_gatherer(){
        return [this](ElementalAssembler& p)->void{
            double *nn_p = p.get_nodes();
            const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
            constexpr auto unfa = Operator<IDEN, UFem>::Nfa::value;
            ProbLocData data;
            for (int i = 0; i < 4; ++i)
                data.nlbl[i] = (*p.nodes)[i].Integer(bnd);
            for (int i = 0; i < 4; ++i)
                data.flbl[i] = (*p.faces)[p.local_face_index[i]].Integer(bnd);

            double vardat[1*unfa];
            //Set udofs from initial values propogated by previous call discr.pullInitValFrom(u);
            data.udofs.Init(p.vars->initValues.data() + p.vars->base_MemOffsets[0], unfa); 
            //Set udofs_prev values from data on tag uprev
            data.udofs_prev.Init(vardat+0*unfa, unfa);
            ElementalAssembler::GatherDataOnElement(uprev, p, data.udofs_prev.data, {});
            data.dt = dt_cur, data.t = t_cur;

            p.compute(args, &data);
        };
    }
    static std::function<void(const double**, double*, double*, void*)> get_local_assembler_implicit(const DiffProbParams::DiffProbTensors& t){
        return [&t](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data){
            auto& dat = *static_cast<ProbLocData*>(user_data);
            dat.XY.Init(XY[0], 3*4);
            const DenseMatrix<const double> XY0(XY[0], 3, 1), XY1(XY[1], 3, 1), XY2(XY[2], 3, 1), XY3(XY[3], 3, 1), XYFull(XY[0] + 0, 3, 4);
            constexpr auto UNFA = Operator<IDEN, UFem>::Nfa::value;
            DenseMatrix<> A(Adat, UNFA, UNFA), F(Fdat, UNFA, 1);
            A.SetZero(), F.SetZero();
            std::array<double, UNFA*UNFA> Bp; 
            DenseMatrix<> B(Bp.data(), UNFA, UNFA, Bp.size()), G(Bp.data(), UNFA, 1);

            // elemental mass matrix <P1, P1>/dt
            fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(
                XY0, XY1, XY2, XY3, makeConstantTensorScalar(1.0/dat.dt), A, 2
            );
            // elemental stiffness matrix <grad(P1), K grad(P1)>
            fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SYMMETRIC, true>>(
                XY0, XY1, XY2, XY3, ProbLocData::make_tensor_wrapper(t.K), B, 2, user_data
            );
            for (unsigned i = 0; i < Bp.size(); ++i) 
                A[i] += B[i];
            // elemental mass matrix <P1, A P1>
            fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(
                    XY0, XY1, XY2, XY3, ProbLocData::make_tensor_wrapper(t.A), B, 2, user_data
            );
            for (unsigned i = 0; i < Bp.size(); ++i) 
                A[i] += B[i];
            
            // elemental right hand side vector <u^n, P1>/dt
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, false>>(
                    XY0, XY1, XY2, XY3, [](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                        (void) dims; (void) iTet;
                        auto& dat = *static_cast<ProbLocData*>(user_data); 
                        out[0] = dat.getUprev(X) / dat.dt;
                        return TENSOR_SCALAR; }, F, 2, user_data
            );
            // elemental right hand side vector <F, P1>
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(
                    XY0, XY1, XY2, XY3, ProbLocData::make_tensor_wrapper(t.F), G, 2, user_data
            );
            for (int i = 0; i < UNFA; ++i) 
                F[i] += G[i];

            auto& nlbl = dat.nlbl;
            auto& flbl = dat.flbl;

            for (int k = 0; k < 4; ++k){
                switch (flbl[k]) {
                    // impose Neumann BC
                    case (BC_NEUMANN_LBL): {
                        fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>> (
                                XY0, XY1, XY2, XY3, k, ProbLocData::make_tensor_wrapper(t.G_N), G, 2, user_data
                        );
                        for (int i = 0; i < UNFA; ++i) 
                            F[i] += G[i];
                        break;
                    }
                    // impose Robin BC
                    case (BC_ROBIN_LBL): {
                        fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>> (
                                XY0, XY1, XY2, XY3, k, ProbLocData::make_tensor_wrapper(t.G_R), G, 2, user_data
                        );
                        for (int i = 0; i < UNFA; ++i) 
                            F[i] += G[i];
                        fem3Dface<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>> (
                                XY0, XY1, XY2, XY3, k, ProbLocData::make_tensor_wrapper(t.S), B, 3, user_data
                        );
                        for (int i = 0; i < UNFA*UNFA; ++i) 
                            A[i] += B[i];
                        break;
                    }
                }
            }

            // impose dirichlet condition
            for (int i = 0; i < UNFA; ++i)
                if (nlbl[i] & BC_DIRICHLET_LBL) {
                    std::array<double, 3> x;
                    DOF_coord<UFem>::at(i, XY[0], x.data()); // compute coordinate of i-th degree of freedom
                    applyDir(A, F, i, t.G_D(dat.t, x, user_data));  // set Dirichlet BC
                }
        };
    }
    static std::function<void(const double**, double*, double*, void*)> get_local_assembler_explicit(const DiffProbParams::DiffProbTensors& t){
        return [&t](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data){
            auto& dat = *static_cast<ProbLocData*>(user_data);
            dat.XY.Init(XY[0], 3*4);
            const DenseMatrix<const double> XY0(XY[0], 3, 1), XY1(XY[1], 3, 1), XY2(XY[2], 3, 1), XY3(XY[3], 3, 1), XYFull(XY[0] + 0, 3, 4);
            constexpr auto UNFA = Operator<IDEN, UFem>::Nfa::value;
            DenseMatrix<> A(Adat, UNFA, UNFA), F(Fdat, UNFA, 1);
            std::array<double, UNFA*UNFA> Bp; 
            DenseMatrix<> B(Bp.data(), UNFA, UNFA, Bp.size()), G(Bp.data(), 4, 1);

            // elemental mass matrix <P1, P1>/dt
            fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(
                XY0, XY1, XY2, XY3, makeConstantTensorScalar(1.0/dat.dt), A, 2
            );

            // elemental right hand side vector <u^n, P1>/dt
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, false>>(
                    XY0, XY1, XY2, XY3, [](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                        (void) dims; (void) iTet;
                        auto& dat = *static_cast<ProbLocData*>(user_data); 
                        out[0] = dat.getUprev(X) / dat.dt;
                        return TENSOR_SCALAR; }, F, 2, user_data
            );
            // elemental right hand side vector <K grad u^n, grad(P1)>
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(
                    XY0, XY1, XY2, XY3, [&K = t.K](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                        (void) dims;
                        auto& dat = *static_cast<ProbLocData*>(user_data); 
                        auto grU = dat.getGradUprev(X);
                        double Kdat[3*3];
                        auto t = K(dat.t - dat.dt, X, Kdat, TensorDims(3, 3), user_data, iTet);
                        assert(t != TENSOR_SCALAR && t != TENSOR_NULL);
                        (void) t;
                        std::fill(out, out + 3, 0);
                        for (int j = 0; j < 3; ++j)
                            for (int i = 0; i < 3; ++i)
                                out[i] += Kdat[i + 3*j] * grU[j];
                        assert(dims.first == 3 && dims.second == 1);
                        return TENSOR_GENERAL; 
                        }, G, 2, user_data
            );
            for (int i = 0; i < UNFA; ++i) 
                F[i] -= G[i];
            // elemental right hand side vector <A u^n, P1>
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, false>>(
                    XY0, XY1, XY2, XY3, [&A = t.A](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                        (void) dims;
                        auto& dat = *static_cast<ProbLocData*>(user_data); 
                        auto U = dat.getUprev(X);
                        double Av;
                        A(dat.t - dat.dt, X, &Av, TensorDims(1, 1), user_data, iTet);
                        out[0] = Av*U;
                        return TENSOR_SCALAR; 
                        }, F, 2, user_data
            );
            for (int i = 0; i < UNFA; ++i) 
                F[i] -= G[i];
            // elemental right hand side vector <F, P1>
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(
                    XY0, XY1, XY2, XY3, ProbLocData::make_tensor_wrapper_prev(t.F), G, 2, user_data
            );
            for (int i = 0; i < UNFA; ++i) 
                F[i] += G[i];

            auto& nlbl = dat.nlbl;
            auto& flbl = dat.flbl;

            for (int k = 0; k < 4; ++k){
                switch (flbl[k]) {
                    // impose Neumann BC
                    case (1 << 1): {
                        fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR>> (
                                XY0, XY1, XY2, XY3, k, ProbLocData::make_tensor_wrapper_prev(t.G_N), G, 2, user_data
                        );
                        for (int i = 0; i < UNFA; ++i) 
                            F[i] += G[i];
                        break;
                    }
                    // impose Robin BC
                    case (1 << 2): {
                        fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, FemFix<FEM_P1>>, DfuncTraits<TENSOR_SCALAR>> (
                                XY0, XY1, XY2, XY3, k, ProbLocData::make_tensor_wrapper_prev(t.G_R), G, 2, user_data
                        );
                        for (int i = 0; i < UNFA; ++i) 
                            F[i] += G[i];
                        fem3Dface<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR>> (
                                XY0, XY1, XY2, XY3, k, [&S = t.S](const Coord<> X, double* out, TensorDims dims, void *user_data, int iTet){
                                    (void) dims;
                                    auto& dat = *static_cast<ProbLocData*>(user_data); 
                                    auto U = dat.getUprev(X);
                                    double Av;
                                    S(dat.t - dat.dt, X, &Av, TensorDims(1, 1), user_data, iTet);
                                    out[0] = Av*U;
                                    return TENSOR_SCALAR; 
                                    }, G, 3, user_data
                        );
                        for (int i = 0; i < UNFA; ++i) 
                            F[i] -= G[i];
                        break;
                    }
                }
            }

            // impose dirichlet condition
            for (int i = 0; i < 4; ++i)
                if (nlbl[i] & 1) {
                    std::array<double, 3> x;
                    DOF_coord<UFem>::at(i, XY[0], x.data()); // compute coordinate of i-th degree of freedom
                    applyDir(A, F, i, t.G_D(dat.t - dat.dt, x, user_data));  // set Dirichlet BC
                }
        };
    }
    static std::function<void(const double**, double*, double*, void*)> get_local_assembler(const DiffProbParams::DiffProbTensors& t, DiffProbParams::TimeScheme ts){
        switch (ts) {
            case DiffProbParams::EXPLICIT_SCHEME: return get_local_assembler_explicit(t);
            case DiffProbParams::IMPLICIT_SCHEME: return get_local_assembler_implicit(t);
            default: throw std::runtime_error("Faced unknown time discretization scheme");
        }
        return std::function<void(const double**, double*, double*, void*)>();
    }



    DiffProbParams m_p;
    std::unique_ptr<Mesh> m;
    Tag bnd;
    Tag u, uprev;

private:
    int pRank = 0, pCount = 1;
    double t_cur = 0; 
    double dt_cur = 0;
    std::shared_ptr<INMOST::Solver> ls;
    Sparse::Matrix A;
    Sparse::Vector x, b;
    Assembler discr;  
    bool slvFlag = false;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> st_time;
};

std::ostream& operator<<(std::ostream& out, const DynDiffProblem::ProbLocData& p){ return p.print(out), out; }

int main(int argc, char* argv[]){
    int pRank = 0, pCount = 1;
    auto p = DiffProbParams::FromInputArgs(&argc, &argv);
    InmostInit(&argc, &argv, p.lin_sol_db, pRank, pCount);  
    if (pRank == 0) std::cout << p << std::endl;

    DynDiffProblem pr(p);
    pr.GenerateMesh()
      .SetProblemDefinition()
      .InitLinSolver()
      .Solve();

    return 0;
}