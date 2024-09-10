//
// Created by Liogky Alexey on 24.10.2022.
//


#include "inmost.h"
#include "anifem++/utils/utils.h"
#include "carnum/MeshGen/mesh_gen.h"
#include "carnum/Fem/AniInterface/Fem3Dtet.h"
#include "carnum/Fem/AniInterface/ForAssembler.h"
#include "carnum/Fem/Assembler.h"
#include "carnum/Solver/SUNNonlinearSolver.h"
#include <numeric>
#if __cplusplus >= 201703L
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

#ifdef WITH_CASADI
#include <casadi/casadi.hpp>
#endif

#if __cplusplus >= 201703L
using namespace std::filesystem;
#else
using namespace std::experimental::filesystem;
#endif

using namespace INMOST;

template<typename T, std::size_t N>
struct arrayw: public std::array<T, N>{
     arrayw(const T* st, const T* ed){ 
        if (ed - st != N) throw std::runtime_error("Wrong initial arrayw size");
        std::copy(st, ed, std::array<T, N>::data()); 
    }
    arrayw(std::size_t sz){ if (sz != N) throw std::runtime_error("Wrong arrayw resize"); }
    void resize(std::size_t sz){
        if (sz != N) throw std::runtime_error("Wrong arrayw resize");
    }
};

using namespace Ani;
struct ProblemDefinition{
    using UFem = FemVec<3, FEM_P2>;
    using PFem = FemFix<FEM_P1>;
    using rMatrix3d = INMOST::Matrix<double, arrayw<double, 3*3>>;
    struct ProbData{
        //for time stepping and p(t) eval
        double t, dt, T;
        //physical parameters
        double rho, mu, b, p_scl;
        //for computing of boundary conditions
        std::array<int, 4> flbl;
        std::array<int, 4> nlbl;
        std::array<int, 6> elbl;
        //for nonlinear tensors:
        ArrayView<> udofs, um1dofs, um2dofs;
        ArrayView<> pdofs;
        ArrayView<const double> XY;
        void print(std::ostream& out = std::cout) const {
            out << "t = " << t << " dt = " << dt << " T = " << T << "\n";
            out << "rho = " << rho << " mu = " << mu << " b = " << b << " p_scl = " << p_scl <<"\n";
            out << "nlbl = " << DenseMatrix<const int>(nlbl.data(), 1, 4);
            out << "elbl = " << DenseMatrix<const int>(elbl.data(), 1, 6);
            out << "flbl = " << DenseMatrix<const int>(flbl.data(), 1, 4) << '\n';
            out << "udofs = " << DenseMatrix<>(udofs.data, 1, udofs.size) << "\n";
            out << "um1dofs = " << DenseMatrix<>(um1dofs.data, 1, um1dofs.size) << "\n";
            out << "um2dofs = " << DenseMatrix<>(um2dofs.data, 1, um2dofs.size) << "\n";
            out << "pdofs = " << DenseMatrix<>(pdofs.data, 1, pdofs.size) << "\n";
            out << "XY = \n" << DenseMatrix<const double>(XY.data, 3, XY.size/3) << "\n";
        } 
    };

    double  rho = 1, //mg/mm^3
            mu = 2, p_scl = 1e-2, //kPa
            b = 4,
            T = 100, dt = 1;  //ms 
    Mesh& m;
    Tag Label, u, p, um1, um2;
    std::vector<Tag> up;

    ProblemDefinition(Mesh& m): m{m}, A("A"), x("x"), discr(m) {
        pRank = 0, pCount = 0;
#if defined(USE_MPI)
        MPI_Comm_rank(INMOST_MPI_COMM_WORLD, &pRank);
        MPI_Comm_size(INMOST_MPI_COMM_WORLD, &pCount);
#endif
    }

    ProblemDefinition& InitLinSolver(){
        ls = std::make_shared<INMOST::Solver>(INMOST::Solver::INNER_MPTILUC);
        ls->SetParameterReal("relative_tolerance", 1e-20);
        double ltau = 9e-2;
        ls->SetParameterReal("drop_tolerance", ltau);
        ls->SetParameterReal("reuse_tolerance", 0.1 * ltau);
        ls->SetParameterEnum("maximum_iterations", 300);
        return *this;
    }
    ProblemDefinition& InitNonLinSolver(){
        if (pRank == 0) std::cout << "--- Initialize nonlinear solver ---" << std::endl;
        auto assemble_F = [&discr = this->discr, &up = this->up](Sparse::Vector& x, Sparse::Vector &b) -> int{
            discr.SaveSolution(x, up);
            std::fill(b.Begin(), b.End(), 0.0);
            discr.AssembleRHS(b);
            return 0;
        };
        auto assemble_J = [&discr = this->discr, &up = this->up](Sparse::Vector& x, Sparse::Matrix &A) -> int{
            discr.SaveSolution(x, up);
            std::for_each(A.Begin(), A.End(), [](INMOST::Sparse::Row& row){ for (auto vit = row.Begin(); vit != row.End(); ++vit) vit->second = 0.0; });
            discr.AssembleMatrix(A);
            return 0;
        };
        x.SetInterval(discr.getBegInd(), discr.getEndInd());
        nls = std::make_shared<SUNNonlinearSolver>(*ls, A, x);
        nls->SetInfoHandlerFn([pRank = this->pRank](const char *module, const char *function, char *msg){
            if (pRank == 0) std::cout << "[" << module << "] " << function << "\n   " << msg << std::endl;
        });
        nls->GetLinearSolverContent()->verbosity = 1;
        nls->SetVerbosityLevel(2);
        nls->SetAssemblerRHS(assemble_F).SetAssemblerMAT(assemble_J);
        double max_step = (2000*sqrt(m.Integrate(x.Size())) + 1);
        nls->SetParameterReal("MaxNewtonStep", max_step);
        nls->SetParameterReal("ScaledSteptol", 1e-7);
        nls->SetParameterReal("FuncNormTol", 1e-9);
        nls->Init();
        return *this;
    }
    ProblemDefinition& SetInitialCondition(){
        if (pRank == 0) std::cout << "--- Set initial condition ---" << std::endl;
        u = m.CreateTag("u", DATA_REAL, NODE|EDGE, NONE, 3);
        um1 = m.CreateTag("uprev", DATA_REAL, NODE|EDGE, NONE, 3);
        um2 = m.CreateTag("upprev", DATA_REAL, NODE|EDGE, NONE, 3);
        p = m.CreateTag("p", DATA_REAL, NODE, NONE, 1);
        up = std::vector<Tag>{u, p};   
        {
            std::vector<Tag> upp = {u, um1, um2, p};
            for (auto e = m.BeginElement(NODE|EDGE); e != m.EndElement(); ++e) {
                for (unsigned v = 0; v < upp.size(); ++v) { 
                    if (!upp[v].isDefined(e->GetElementType())) continue;
                    auto arr = e->RealArray(upp[v]);
                    std::fill(arr.begin(), arr.end(), 0.0);
                }
            } 
        }
        return *this;
    }
    ProblemDefinition& SetProblemDefinition(){
        if (pRank == 0) std::cout << "--- Set problem definition ---" << std::endl;
        constexpr auto elem_nfa = Operator<IDEN, FemCom<UFem, PFem>>::Nfa::value;
        discr.SetMatFunc(GenerateElemMat(get_local_mtx_assembler(tensors), elem_nfa, elem_nfa));
        discr.SetRHSFunc(GenerateElemRhs(get_local_rhs_assembler(tensors), elem_nfa));
        {
            auto Var0Helper = GenerateHelper<UFem>(),
                 Var1Helper = GenerateHelper<PFem>();
            FemExprDescr fed;
            fed.PushVar(Var0Helper, "u"); fed.PushTestFunc(Var0Helper, "phi_u");
            fed.PushVar(Var1Helper, "p"); fed.PushTestFunc(Var1Helper, "phi_p");
            discr.SetProbDescr(std::move(fed));
        }
        discr.SetDataGatherer(get_local_data_gatherer());
        discr.PrepareProblem();
        discr.pullInitValFrom(up);
        return *this;
    }
    bool Solve(bool save_steps = false, std::string save_prefix = ""){
        if (pRank == 0) std::cout << "--- Time loop ---" << std::endl;
        if (save_steps) m.Save(save_prefix + "res_" + std::to_string(t_cur) + ".vtk");
        discr.SaveSolution(up, x);
        while(t_cur < T){
            double dt_loc = dt;
            if (t_cur + dt > T) dt_loc = T - t_cur;
            dt = dt_loc;
            t_cur += dt;
            if (pRank == 0) std::cout << "\n    t = " << t_cur << ":"  << std::endl;
            slvFlag = nls->Solve(); printNLSolverStatus("BeamBending"); 
            discr.SaveSolution(x, up);
            for (auto e = m.BeginElement(NODE|EDGE); e != m.EndElement(); ++e) {
                auto arr = e->RealArray(u), arrm1 = e->RealArray(um1), arrm2 = e->RealArray(um2);
                std::copy(arrm1.begin(), arrm1.end(), arrm2.begin());
                std::copy(arr.begin(), arr.end(), arrm1.begin());
            }
            if (save_steps) {
                std::string save_name = save_prefix + "res_" + std::to_string(t_cur) + ".vtu";
                m.Save(save_name); 
                if (pRank == 0) std::cout << "Intermediate result saved in \"" << save_name << "\"" << std::endl;
            }
        }
        return slvFlag;
    }
    ProblemDefinition& PrepareTensors(bool is_static = true){
#ifdef WITH_CASADI
        if (!is_static)
            tensors.InitCasadi(*this);
#endif
        return *this;
    }
#ifdef WITH_CASADI
    ProblemDefinition& GenerateTermsFromSymbolic(){
        using namespace casadi;
        SX x = SX::sym("x",3,3);
        using TT = SX; //MX
        Function Det("det3d", {x}, {SX::det(x)});
        TT u = TT::sym("u", 3, 1), um1 = TT::sym("um1", 3, 1), um2 = TT::sym("um2", 3, 1);
        TT p = TT::sym("p");
        TT mu = TT::sym("mu"), b = TT::sym("b"), rho = TT::sym("rho"), dt = TT::sym("dt"), p0 = TT::sym("p0");
        TT grU = TT::sym("grU", 3, 3); //grad_j u_i
        TT phi = TT::sym("phi", 3), q = TT::sym("q");
        TT n = TT::sym("normal", 3);
        TT grPhi = TT::sym("grPhi", 3, 3); //grad_j phi_i
        TT I = TT::eye(3);
        TT F = I + grU;
        TT Eps = (grU + grU.T())/2;
        TT E = (grU + grU.T() + TT::mtimes(grU.T(), grU))/2;
        //Linear case maybe useful for debug purposes
        //TT J = 1+TT::trace(grU); //< Linear elasticity
        //TT W = mu*b / 2 * TT::trace(TT::mtimes(Eps.T(), Eps)) - p*(J-1); //< Linear elasticity
        TT J = Det({F})[0]; //<NonLinear elasticity
        TT W = mu/2*(TT::exp(b*TT::trace(TT::mtimes(E.T(), E))) - 1) - p*(J - 1); //<NonLinear elasticity

        TT P = TT::gradient(W, grU); //< same as dW/dF
        
        TT mass_term_j = TT::dot(rho*(u - 2*um1 + um2)/TT::sq(dt), phi);
        TT stiff_term_j = TT::dot(P, grPhi); //P_ij (grad_j phi_i)_{ij}
        TT pstiff_term_j = -TT::dot(J-1, q);
        TT neumann_term_j = p0*TT::mtimes({phi.T(), TT::gradient(J-1, grU), n});
        TT mass_term_rhs = TT::jacobian(mass_term_j, phi);
        //< for coincendence with AniFem we sholud use grPhi^T = grad_i phi_j
        TT AniGrPhi = TT::sym("AniGrPhi", 3, 3); //grad_i phi_j
        TT stiff_term_ani = TT::substitute(stiff_term_j, grPhi, AniGrPhi.T());
        TT stiff_term_rhs = TT::jacobian(stiff_term_ani, AniGrPhi);
        TT pstiff_term_rhs = TT::jacobian(pstiff_term_j, q);
        TT neumann_term_rhs = TT::jacobian(TT::jacobian(neumann_term_j, n), phi);

        TT mass_term_mtx = TT::jacobian(mass_term_rhs, u);
           stiff_term_ani = TT::substitute(stiff_term_rhs, grU, AniGrPhi.T());
        TT stiff_term_mtx_grU = TT::substitute(TT::jacobian(stiff_term_ani, AniGrPhi), AniGrPhi, grU.T());
        TT stiff_term_mtx_p = TT::jacobian(stiff_term_rhs, p);
        TT neumann_term_ani = TT::substitute(neumann_term_rhs, grU, AniGrPhi.T());
        TT neumann_term_mtx_grU = TT::substitute(TT::jacobian(neumann_term_ani, AniGrPhi), AniGrPhi, grU.T());
        Function    mrhs("mass_rhs", {u, um1, um2, rho, dt}, {mass_term_rhs}, {"u", "um1", "um2", "rho", "dt"}, {"mass_rhs"}),
                    srhs("stiff_rhs", {grU, p, mu, b}, {stiff_term_rhs}, {"grU", "p", "mu", "b"}, {"stiff_rhs"}),
                    psrhs("pstiff_rhs", {grU}, {pstiff_term_rhs}, {"grU"}, {"pstiff_rhs"}),
                    nrhs("neumann_rhs", {grU, p0}, {neumann_term_rhs}, {"grU", "p0"}, {"neumann_rhs"}),
                    mmtx("mass_mtx_u", {rho, dt}, {mass_term_mtx}, {"rho", "dt"}, {"mass_rhs"}),
                    smtxg("stiff_mtx_u", {grU, p, mu, b}, {stiff_term_mtx_grU}, {"grU", "p", "mu", "b"}, {"stiff_mtx_u"}),
                    smtxp("stiff_mtx_p", {grU, mu, b}, {stiff_term_mtx_p}, {"grU", "mu", "b"}, {"stiff_mtx_p"}),
                    nmtxg("neumann_mtx_u", {grU, p0}, {neumann_term_mtx_grU}, {"grU", "p0"}, {"neumann_mtx_u"});
        std::vector<Function> tensors = {mrhs, srhs, psrhs, nrhs, mmtx, smtxg, smtxp, nmtxg};            
        std::string gen_file_name = "BendBeamTensors", gen_file_ext = ".c", gen_dir = "generated/";
        CodeGenerator gen(gen_file_name+gen_file_ext);
        for (auto& i: tensors) gen.add(i);
        create_directory(gen_dir);
        gen.generate(gen_dir);
        Importer imp;
        for (int p = 0; p < pCount; ++p){
            if (p == pRank && pRank == 0){
                imp = Importer(gen_dir+gen_file_name+gen_file_ext, "shell", {
                                {"folder", gen_dir},
                                {"cleanup", false},
                                {"compiler_flags", {"-O3"}},
                                {"verbose", true},
                                {"temp_suffix", false},
                                {"name", gen_dir+gen_file_name} //<library name
                            });
            } else if (p == pRank && pRank != 0){
                std::string lib_ext = ".so";
    #ifdef _WIN32
                lib_ext = ".dll";
    #endif
                imp = Importer(gen_dir+gen_file_name+lib_ext, "dll");
            }  
            if (p == 0) { BARRIER; }
        }
        Function mrhsf = external("mass_rhs", imp), 
                 srhsf = external("stiff_rhs", imp),
                 psrhsf = external("pstiff_rhs", imp),
                 nrhsf = external("neumann_rhs", imp), 
                 mmtxf = external("mass_mtx_u", imp), 
                 smtxgf = external("stiff_mtx_u", imp),
                 smtxpf = external("stiff_mtx_p", imp), 
                 nmtxgf = external("neumann_mtx_u", imp);
        m_funcs = std::vector<Function>{mrhsf, srhsf, psrhsf, nrhsf, mmtxf, smtxgf, smtxpf, nmtxgf};         
        func_bufs.resize(m_funcs.size());
        for (unsigned i = 0; i < m_funcs.size(); ++i) 
            func_bufs[i] = std::make_shared<FunctionBuffer>(m_funcs[i]);
        return *this;    
    }
#endif
    struct ProblemTensors{
        using Tensor = std::function<Ani::TensorType(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet)>;
        Tensor  mass_rhs, 
                stiff_rhs,
                pstiff_rhs, 
                neumann_rhs, 
                mass_mtx_u,
                stiff_mtx_u,
                stiff_mtx_p,
                neumann_mtx_u,
                pstiff_mtx_u;
        ProblemTensors(){
            InitStatic();
        }        
        void InitStatic(){
            mass_rhs = ProblemDefinition::mass_rhs;
            stiff_rhs = ProblemDefinition::Peval;
            pstiff_rhs = ProblemDefinition::Jeval;
            neumann_rhs = ProblemDefinition::NeumannEval;
            mass_mtx_u = [](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) X; (void) dims; (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                Pdat[0] = dat.rho/(dat.dt*dat.dt);
                return Ani::TENSOR_SCALAR;
            };
            stiff_mtx_u = ProblemDefinition::dP_div_dF;
            stiff_mtx_p = ProblemDefinition::dP_div_dp;
            neumann_mtx_u = ProblemDefinition::NeumannD;
            pstiff_mtx_u = nullptr;
        }
#ifdef WITH_CASADI
        void InitCasadi(ProblemDefinition& p){
           auto densify_result = [](casadi::Function& f, double* res){
                auto sp = f.sparsity_out(0);
                if (sp.nnz() == sp.size1()*sp.size2()) return;
                int lda = sp.size1();
                const long long *colind = sp.colind(), *row = sp.row();
                for (int i = sp.size2()-1; i >= 0; --i){
                    int k = lda - 1;
                    for (int j = colind[i+1]-1; j >= colind[i]; --j){
                        for (; k > row[j]; --k)
                            res[i * lda + k] = 0;
                        res[i * lda + row[j]] = res[j];
                        --k;    
                    }
                    for (; k>=0; --k)
                        res[i * lda + k] = 0;
                }
           }; 
           mass_rhs = [&f = *(p.func_bufs[0]), densify_result, &ff = p.m_funcs[0]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                //u, um1, um2, rho, dt
                auto u = getU(X, user_data, 0), um1 = getU(X, user_data, -1), um2 = getU(X, user_data, -2);
                auto& dat = *static_cast<ProbData*>(user_data);
                f.set_arg(0, u.data(), u.size()*sizeof(double));
                f.set_arg(1, um1.data(), um1.size()*sizeof(double));
                f.set_arg(2, um2.data(), um2.size()*sizeof(double));
                f.set_arg(3, &dat.rho, 1*sizeof(double));
                f.set_arg(4, &dat.dt, 1*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);

                return Ani::TENSOR_GENERAL;
           };
           stiff_rhs = [&f = *(p.func_bufs[1]), densify_result, &ff = p.m_funcs[1]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                //grU, p, mu, b
                auto grUp = getGradU(X, user_data);
                double p = get_p(X, user_data);
                f.set_arg(0, grUp.data(), grUp.size()*sizeof(double));
                f.set_arg(1, &p, 1*sizeof(double));
                f.set_arg(2, &dat.mu, 1*sizeof(double));
                f.set_arg(3, &dat.b, 1*sizeof(double));
                
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                return Ani::TENSOR_GENERAL;
           };
           pstiff_rhs = [&f = *(p.func_bufs[2]), densify_result, &ff = p.m_funcs[2]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                //grU
                auto grUp = getGradU(X, user_data);
                f.set_arg(0, grUp.data(), grUp.size()*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                return Ani::TENSOR_GENERAL;
           };
           neumann_rhs = [&f = *(p.func_bufs[3]), densify_result, &ff = p.m_funcs[3]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                //grU, p0
                auto grUp = getGradU(X, user_data);
                double p0 = ProblemDefinition::p_0(dat.t, dat.T, dat.p_scl);
                f.set_arg(0, grUp.data(), grUp.size()*sizeof(double));
                f.set_arg(1, &p0, 1*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                return Ani::TENSOR_GENERAL;
           };
           mass_mtx_u = [&f = *(p.func_bufs[4]), densify_result, &ff = p.m_funcs[4]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                //rho, dt
                auto grUp = getGradU(X, user_data);
                f.set_arg(0, &dat.rho, grUp.size()*sizeof(double));
                f.set_arg(1, &dat.dt, 1*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                return Ani::TENSOR_GENERAL;
           };
           stiff_mtx_u = [&f = *(p.func_bufs[5]), densify_result, &ff = p.m_funcs[5]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                //grU, p, mu, b
                auto grUp = getGradU(X, user_data);
                double p = get_p(X, user_data);
                f.set_arg(0, grUp.data(), grUp.size()*sizeof(double));
                f.set_arg(1, &p, 1*sizeof(double));
                f.set_arg(2, &dat.mu, 1*sizeof(double));
                f.set_arg(3, &dat.b, 1*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                
                return Ani::TENSOR_GENERAL;
           };
           stiff_mtx_p = [&f = *(p.func_bufs[6]), densify_result, &ff = p.m_funcs[6]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                //grU, mu, b
                auto grUp = getGradU(X, user_data);
                f.set_arg(0, grUp.data(), grUp.size()*sizeof(double));
                f.set_arg(1, &dat.mu, 1*sizeof(double));
                f.set_arg(2, &dat.b, 1*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                return Ani::TENSOR_GENERAL;
           };
           neumann_mtx_u = [&f = *(p.func_bufs[7]), densify_result, &ff = p.m_funcs[7]](const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
                (void) iTet;
                auto& dat = *static_cast<ProbData*>(user_data);
                //grU, p0
                auto grUp = getGradU(X, user_data);
                double p0 = ProblemDefinition::p_0(dat.t, dat.T, dat.p_scl);
                f.set_arg(0, grUp.data(), grUp.size()*sizeof(double));
                f.set_arg(1, &p0, 1*sizeof(double));
                f.set_res(0, Pdat, dims.first*dims.second*sizeof(double));
                f._eval();
                densify_result(ff, Pdat);
                return Ani::TENSOR_GENERAL;
           };
           pstiff_mtx_u = nullptr;
        }
#endif
    };

    static double DetA(const rMatrix3d& A){
        double res = 0;
        for (int i = 0; i < 3; ++i)
            res += A(i, 0)*A((i+1)%3, 1)*A((i+2)%3, 2) - A(i, 2)*A((i+1)%3, 1)*A((i+2)%3, 0);
        return res;    
    }
    template <typename MTX>
    static void dDetA_div_dA(const rMatrix3d& A, MTX& R){
        //J = det(A) = 1.0/6 e_{ikn}e_{jmq} A_{ij} A_{km} A_{nq} 
        // => JA^{-T} = d(det(A)) / d(A) = 1.0/2 e_{ikn}e_{jmq} A_{km} A_{nq}
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j){ 
                R(i, j) = 0;
                for (int k = 0; k < 3; ++k) {if (k == i) continue;
                    for (int m = 0; m < 3; ++m) {if (m == j) continue;
                        int n = 3 - (i + k), q = 3 - (j + m);
                        char sg = ((k == (i+1)%3) == (m == (j+1)%3)) ? 1 : -1;
                        R(i, j) += sg * A(k, m) * A(n, q);
                    }
                }
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) /= 2;        
    }
    template <typename MTX>
    static void d2DetA_div_ddA(const rMatrix3d& A, MTX& R){
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) {if (k == i) continue;
                    for (int m = 0; m < 3; ++m) {if (m == j) continue;
                        int n = 3 - (i + k), q = 3 - (j + m);
                        char sg = ((k == (i+1)%3) == (m == (j+1)%3)) ? 1 : -1;
                        R(i + 3*j, k + 3*m) += sg*A(n, q);
                    }
                }         
    }
    static rMatrix3d dDetA_div_dA1(const rMatrix3d& A){
        rMatrix3d R(3, 3);
        std::fill(R.data(), R.data() + 9, 0);
        dDetA_div_dA(A, R);

        return R;
    }
    static INMOST::Matrix<double, arrayw<double, 9*9>> d2DetA_div_ddA1(const rMatrix3d& A){
        INMOST::Matrix<double, arrayw<double, 9*9>> R(9, 9);
        std::fill(R.data(), R.data() + 9*9, 0);
        d2DetA_div_ddA(A, R);
        return R;
    }
    static std::array<double, 3> getU(const Coord<> &X, void *user_data, int i){
       std::array<double, 3> Up;
        auto& dat = *static_cast<ProbData*>(user_data);
        ArrayView<> U(Up.data(), Up.size());
        ArrayView<>* dofs[3] = {&dat.um2dofs, &dat.um1dofs, &dat.udofs};
        fem3DapplyX<Operator<IDEN, UFem>>(dat.XY.data, dat.XY.data+3, dat.XY.data+6, dat.XY.data+9, X.data(), *(dofs[2+i]), U);
        return Up; 
    }
    static std::array<double, 3*3> getGradU(const Coord<> &X, void *user_data, bool save_ani_format = false){
        std::array<double, 3*3> grUp = {0};
        auto& dat = *static_cast<ProbData*>(user_data);
        ArrayView<> grU(grUp.data(), grUp.size());
        fem3DapplyX<Operator<GRAD, UFem>>(dat.XY.data, dat.XY.data+3, dat.XY.data+6, dat.XY.data+9, X.data(), dat.udofs, grU);
        if (save_ani_format) return grUp;
        //we get matrix grU_{ij} = grad_i U_j, but usually work with grad_j U_i so we transpose output
        for (int i = 0; i < 3; ++i) 
            for (int j = 0; j < i; ++j) 
                std::swap(grU[i+3*j], grU[j+3*i]); 
        return grUp;
    };
    static double get_p(const Coord<> &X, void *user_data){
        double res = 0;
        auto& dat = *static_cast<ProbData*>(user_data);
        ArrayView<> p(&res, 1);
        fem3DapplyX<Operator<IDEN, PFem>>(dat.XY.data, dat.XY.data+3, dat.XY.data+6, dat.XY.data+9, X.data(), dat.pdofs, p); 
        return res;
    };  
    static Ani::TensorType Peval(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 9 && dims.second == 1 && "Wrong Peval dims");
        (void) dims; (void) iTet;
        auto& dat = *static_cast<ProbData*>(user_data);
        auto mu = dat.mu, b = dat.b;
        auto grUp = getGradU(X, user_data);
        double p = get_p(X, user_data);
        raMatrix PT(shell<double>(Pdat, 9), 3, 3);
        rMatrix3d grU = raMatrix(shell<double>(grUp.data(), 9), 3, 3).Transpose();
        rMatrix3d F(3, 3), Eps(3, 3), E(3, 3), K(3, 3), Pp(3, 3), Pe(3, 3); 
        auto I = rMatrix3d::Unit(3);
        F = I + grU;
        Eps = (grU + grU.Transpose())/2;
        E = Eps + (grU.Transpose()*grU)/2;
        double trE2 = 0;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                trE2 += E(i, j)*E(i, j);
        double r = mu * b * exp(b*trE2);
        K = F*E;
        Pe = r*K;
        Pp = -p*dDetA_div_dA1(F);
        //we should transpose tensor due to Ani numeration of grad \phi = (grad_i \phi_j)_{i + 3*j}
        //to get P_{ji} grad_i \phi_j = P_{j + 3*i} * (grad_i \phi_j)_{i + 3*j}
        PT = Pe + Pp;
        //PT = (Pe + Pp).Transpose();  
        return Ani::TENSOR_GENERAL;  
    }
    static Ani::TensorType Jeval(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 1 && dims.second == 1 && "Wrong Jeval dims");
        (void) dims; (void) iTet;
        auto grUp = getGradU(X, user_data);
        raMatrix grUT(shell<double>(grUp.data(), 9), 3, 3);
        rMatrix3d FT(3, 3); 
        auto I = rMatrix3d::Unit(3);       
        FT = I + grUT;
        Pdat[0] = -(DetA(FT) - 1);
        return Ani::TENSOR_GENERAL;
    }
    static Ani::TensorType dP_div_dF(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 9 && dims.second == 9 && "Wrong dP_div_dF dims");
        (void) dims; (void) iTet;
        auto& dat = *static_cast<ProbData*>(user_data);
        auto mu = dat.mu, b = dat.b;
        auto grUp = getGradU(X, user_data);
        double p = get_p(X, user_data);
        raMatrix PT(shell<double>(Pdat, 9*9), 9, 9);
        rMatrix3d grU = raMatrix(shell<double>(grUp.data(), 9), 3, 3).Transpose();
        rMatrix3d F(3, 3), Eps(3, 3), E(3, 3), e(3, 3), K(3, 3); 
        auto I = rMatrix3d::Unit(3);                
        F = I + grU;
        Eps = (grU + grU.Transpose())/2;
        E = Eps + (grU.Transpose()*grU)/2;
        e = Eps + (grU*grU.Transpose())/2;
        #define R(i, j, k, l) (grU(i, j)*I(k,l) + I(i,j)*grU(k,l) + grU(i,j)*grU(k,l))/2
        //we should transpose tensor due to Ani numeration of grad \phi = (grad_i \phi_j)_{i + 3*j}
        // so P_{ij, km} -> P(j+3*i, m+3*k)
        ////PT(JJ+3*II, MM+3*KK)
        #define _P(II,JJ,KK,MM) PT(MM+3*KK, JJ+3*II)
        double trE2 = 0;
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                trE2 += E(i, j)*E(i, j);
        double r = mu * b * exp(b*trE2);
        K = F*E;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        _P(i, j, k, l) = r * (  2*b*K(i,j)*K(k,l) + 
                                                (I(i,k)*E(j,l) + e(i,k)*I(j,l) + R(i,l,k,j)) + 
                                                (I(i,k)*I(j,l) + I(i,l)*I(j,k))/2
                                             ); 
        auto ddF = d2DetA_div_ddA1(F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int m = 0; m < 3; ++m)
                        _P(i, j, k, m) += -p*ddF(i+3*j, k+3*m);  
        #undef _P 
        #undef R
        return Ani::TENSOR_GENERAL;                             
    };
    static Ani::TensorType dP_div_dp(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 9 && dims.second == 1 && "Wrong dP_div_dp dims");
        (void) dims; (void) iTet;
        auto grUp = getGradU(X, user_data);
        raMatrix PT(shell<double>(Pdat, 9), 3, 3);
        rMatrix3d grU = raMatrix(shell<double>(grUp.data(), 9), 3, 3).Transpose();
        rMatrix3d F(3, 3), Pp(3, 3);
        auto I = rMatrix3d::Unit(3); 
        F = I + grU;
        Pp = -dDetA_div_dA1(F);
        PT = Pp;
        //PT = Pp.Transpose();

        return Ani::TENSOR_GENERAL;
    }
    static Ani::TensorType mass_rhs(const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 3 && dims.second == 1 && "Wrong mass_rhs");
        (void) dims; (void) iTet;
        auto u = getU(X, user_data, 0), um1 = getU(X, user_data, -1), um2 = getU(X, user_data, -2);
        auto& dat = *static_cast<ProbData*>(user_data);
        for (unsigned i = 0; i < u.size(); ++i)
            Kdat[i] = dat.rho*(u[i] - 2*um1[i] + um2[i])/(dat.dt*dat.dt);
        return TENSOR_GENERAL;
    }
    static double p_0(double t, double T, double p_scl) { return p_scl * t / T; };
    static Ani::TensorType NeumannEval(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 9 && dims.second == 1 && "Wrong NeumannEval dims");
        dP_div_dp(X, Pdat, dims, user_data, iTet);
        raMatrix PT(shell<double>(Pdat, 9), 3, 3);
        auto& dat = *static_cast<ProbData*>(user_data);
        double p0 = p_0(dat.t, dat.T, dat.p_scl);
        PT *= -p0;

        return Ani::TENSOR_GENERAL;
    }
    static Ani::TensorType NeumannD(const Coord<> &X, double *Pdat, TensorDims dims, void *user_data, int iTet){
        assert(dims.first == 9 && dims.second == 9 && "Wrong NeumannD dims");
        (void) dims; (void) iTet;
        auto grUp = getGradU(X, user_data);
        raMatrix PT(shell<double>(Pdat, 9*9), 9, 9);
        rMatrix3d grU = raMatrix(shell<double>(grUp.data(), 9), 3, 3).Transpose();
        auto& dat = *static_cast<ProbData*>(user_data);
        double p0 = p_0(dat.t, dat.T, dat.p_scl);
        rMatrix3d F(3, 3);
        auto I = rMatrix3d::Unit(3);        
        F = I + grU;
        //PT(JJ+3*II, MM+3*KK)
        #define _P(II,JJ,KK,MM) PT(MM+3*KK, JJ+3*II)
        auto ddF = d2DetA_div_ddA1(F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int m = 0; m < 3; ++m)
                        _P(i, j, k, m) += p0*ddF(i+3*j, k+3*m);  
        #undef _P
        
        return Ani::TENSOR_GENERAL;
    };
    static std::function<void(const double**, double*, void*)> get_local_mtx_assembler(ProblemTensors t){
        return [t](const double** XY/*[4]*/, double* Adat, void* user_data){
            auto& dat = *static_cast<ProbData*>(user_data);
            dat.XY.Init(XY[0], 3*4);
            const DenseMatrix<const double> XY0(XY[0], 3, 1), XY1(XY[1], 3, 1), XY2(XY[2], 3, 1), XY3(XY[3], 3, 1), XYFull(XY[0] + 0, 3, 4);
            constexpr auto UNFA = Operator<IDEN, UFem>::Nfa::value, PNFA = Operator<IDEN, PFem>::Nfa::value;
            DenseMatrix<> A(Adat, UNFA+PNFA, UNFA+PNFA);
            std::array<double, (UNFA+PNFA)*(UNFA+PNFA)> Bp; DenseMatrix<> B(Bp.data(), UNFA, UNFA, Bp.size());
            // \int \rho u_i \phi_i / dt^2
            fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(XY0, XY1, XY2, XY3, t.mass_mtx_u, B, 4, user_data);
            for (int i = 0; i < UNFA; ++i)
                for (int j = 0; j < UNFA; ++j)
                    A(i, j) = B(i, j);
                
            // \int  (dP(\nabla u, p)/dF : \nabla u) : \nabla \phi
            fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, t.stiff_mtx_u, B, 5, user_data);
            for (int i = 0; i < UNFA; ++i)
                for (int j = 0; j < UNFA; ++j)
                    A(i, j) += B(i, j);
            
            // \int (dP(\nabla u)/dp p) : \nabla \phi  
            // \int q d(J - 1)/dF : \nabla u      
            fem3Dtet<Operator<IDEN, PFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, t.stiff_mtx_p, B, 5, user_data);
            for (int i = 0; i < UNFA; ++i)
                for (int j = 0; j < PNFA; ++j)
                    A(i, UNFA+j) = A(UNFA+j, i) = B(i, j);
            
            //Impose BC
            //  Neumann
            for (int k = 0; k < 4; ++k){
                if (dat.flbl[k] == 2){
                    fem3DfaceN<Operator<GRAD, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_GENERAL>> (XY0, XY1, XY2, XY3, k, t.neumann_mtx_u, B, 4, user_data);
                    for (int i = 0; i < UNFA; ++i)
                        for (int j = 0; j < UNFA; ++j)
                            A(i, j) += B(i, j);
                }
            }
            // Dirichlet
            constexpr auto UNFA1 = Operator<IDEN, UFem::Base>::Nfa::value;
            std::array<bool, 10> dc = {false};
            for (int i = 0; i < 4; ++i) 
                dc[i] = (dat.nlbl[i] & 1);
            for (int i = 0; i < 6; ++i) 
                dc[i+4] = (dat.elbl[i] & 1);    
            
            for (int k = 0; k < 10; ++k) if (dc[k])
                for (int dim = 0; dim < UFem::Dim::value; ++dim)
                        applyDirMatrix(A, k+dim*UNFA1);  
        };                   
    }
    static std::function<void(const double**, double*, void*)> get_local_rhs_assembler(ProblemTensors t){
        return [t](const double** XY/*[4]*/, double* Fdat, void* user_data){
            auto& dat = *static_cast<ProbData*>(user_data);
            dat.XY.Init(XY[0], 3*4);
            const DenseMatrix<const double> XY0(XY[0], 3, 1), XY1(XY[1], 3, 1), XY2(XY[2], 3, 1), XY3(XY[3], 3, 1), XYFull(XY[0] + 0, 3, 4);
            constexpr auto UNFA = Operator<IDEN, UFem>::Nfa::value, PNFA = Operator<IDEN, PFem>::Nfa::value;
            DenseMatrix<> F(Fdat, UNFA+PNFA, 1);
            std::array<double, (UNFA+PNFA)> Bp; DenseMatrix<> B(Bp.data(), UNFA, 1, UNFA+PNFA);
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_GENERAL>>(XY0, XY1, XY2, XY3, t.mass_rhs, B, 4, user_data);
            for (int i = 0; i < UNFA; ++i)
                F(i, 0) = B(i, 0);    
    
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL>>(XY0, XY1, XY2, XY3, t.stiff_rhs, B, 5, user_data);
            for (int i = 0; i < UNFA; ++i)
                F(i, 0) += B(i, 0);
            
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, PFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, t.pstiff_rhs, B, 4, user_data);
            for (int i = 0; i < PNFA; ++i)
                F(i+UNFA, 0) = B(i, 0);
            
            for (int k = 0; k < 4; ++k){
                if (dat.flbl[k] == 2){
                    fem3DfaceN<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_GENERAL>> (XY0, XY1, XY2, XY3, k, t.neumann_rhs, B, 4, user_data); 
                    for (int i = 0; i < UNFA; ++i)
                        F(i, 0) += B(i, 0);    
                }
            }
            constexpr auto UNFA1 = Operator<IDEN, UFem::Base>::Nfa::value;
            std::array<bool, 10> dc = {false};
            for (int i = 0; i < 4; ++i) 
                dc[i] = (dat.nlbl[i] & 1);
            for (int i = 0; i < 6; ++i) 
                dc[i+4] = (dat.elbl[i] & 1);

            for (int k = 0; k < 10; ++k) if (dc[k])
                for (int dim = 0; dim < UFem::Dim::value; ++dim)
                    applyDirResidual(F, k+dim*UNFA1);   
            return;   
        };                 
    }
    std::function<void (ElementalAssembler &p)> get_local_data_gatherer(){
        return [this](ElementalAssembler& p)->void{
            double *nn_p = p.get_nodes();
            const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
            constexpr auto unfa = Operator<IDEN, UFem>::Nfa::value, pnfa = Operator<IDEN, PFem>::Nfa::value;
            ProbData data;
            for (int i = 0; i < 4; ++i)
                data.nlbl[i] = (*p.nodes)[i].Integer(Label);
            for (int i = 0; i < 4; ++i)
                data.flbl[i] = (*p.faces)[p.local_face_index[i]].Integer(Label);
            for (int i = 0; i < 6; ++i)    
                data.elbl[i] = (*p.edges)[p.local_edge_index[i]].Integer(Label);
            

            double vardat[2*unfa];
            data.udofs.Init(p.vars->initValues.data() + p.vars->base_MemOffsets[0], unfa); 
            data.pdofs.Init(p.vars->initValues.data() + p.vars->base_MemOffsets[1], pnfa);
            data.um1dofs.Init(vardat, unfa), data.um2dofs.Init(vardat+unfa, unfa);
            Tag utag = up[0];

            //instead: 
            //      discr.SetInitValueSetter(Assembler::makeInitValueSetter(up));
            //in code below we may use the code: 
            //      ElementalAssembler::GatherDataOnElement(up, p, p.vars->initValues.data(), std::array<int, 0>{});
            up[0] = um1;
            ElementalAssembler::GatherDataOnElement(up, p, data.um1dofs.data, 0);
            up[0] = um2;
            ElementalAssembler::GatherDataOnElement(up, p, data.um2dofs.data, 0);
            up[0] = utag;

            data.dt = dt; data.t = t_cur;
            data.p_scl = p_scl, data.T = T; 
            data.rho = rho, data.mu = mu, data.b = b;
            p.compute(args, &data);
        };
    }
    void printNLSolverStatus(std::string prob_nm = "", bool exit_on_false = true) const{
        auto success_solve = slvFlag;
        auto rank = pRank, npc = pCount;
        SUNNonlinearSolver& s = *nls;
        if(!success_solve){
            if (exit_on_false){
                std::cout << prob_nm << ":\n"
                        <<"\t#NonLinIts = "<<s.GetNumNolinSolvIters() << " Residual = "<< s.GetResidualNorm()
                        << "\n\t#linIts = " << s.GetNumLinIters() << " #funcEvals = " << s.GetNumLinFuncEvals() << " #jacEvals = " << s.GetNumJacEvals()
                        << "\n\t#convFails = " << s.GetNumLinConvFails() << " #betaCondFails = " << s.GetNumBetaCondFails() << " #backtrackOps = " << s.GetNumBacktrackOps() << "\n";
                std::cout<<"\t"<<rank<< " / " << npc << " failed to solve system. ";
                std::cout << "Reason: " << s.GetReason() << std::endl;
                exit(-1);
            } else {
            if (rank == 0)
                    std::cout << prob_nm << ":\n"
                        <<"\tnot_converged #NonLinIts = "<<s.GetNumNolinSolvIters() << " Residual = "<< s.GetResidualNorm()
                        << "\n\t#linIts = " << s.GetNumLinIters() << " #funcEvals = " << s.GetNumLinFuncEvals() << " #jacEvals = " << s.GetNumJacEvals()
                        << "\n\t#convFails = " << s.GetNumLinConvFails() << " #betaCondFails = " << s.GetNumBetaCondFails() << " #backtrackOps = " << s.GetNumBacktrackOps() << "\n"
                        << "\n\tMatAssmTime = " << s.GetMatAssembleTime() << "s RHSAssmTime = " << s.GetRHSAssembleTime() << "s"
                        << "Reason: " << s.GetReason() << std::endl; 
            }
        }
        else{
            if(rank == 0)
                std::cout << prob_nm << ":\n"
                        <<"\tsolved_succesful #NonLinIts = "<<s.GetNumNolinSolvIters() << " Residual = "<< s.GetResidualNorm()
                        << "\n\t#linIts = " << s.GetNumLinIters() << " #funcEvals = " << s.GetNumLinFuncEvals() << " #jacEvals = " << s.GetNumJacEvals()
                        << "\n\t#convFails = " << s.GetNumLinConvFails() << " #betaCondFails = " << s.GetNumBetaCondFails() << " #backtrackOps = " << s.GetNumBacktrackOps()
                        << "\n\tMatAssmTime = " << s.GetMatAssembleTime() << "s RHSAssmTime = " << s.GetRHSAssembleTime() << "s" << std::endl;
        }
    }
    
    private:
    int pRank = 0, pCount = 1;
    double t_cur = 0; 
    std::shared_ptr<INMOST::Solver> ls;
    std::shared_ptr<SUNNonlinearSolver> nls; 
    Sparse::Matrix A;
    Sparse::Vector x;
    Assembler discr;  
    bool slvFlag = false; 
    ProblemTensors tensors;
    #ifdef WITH_CASADI
    std::vector<casadi::Function> m_funcs;
    std::vector<std::shared_ptr<casadi::FunctionBuffer>> func_bufs;
    bool use_casadi_tensors = false;
    #endif 
};




/** The program generates and solves a nonlinear finite element system for 
 * the nonstationary bending of incompessible beam
 * \f[
 * \left\{ 
 * \begin{aligned}
 *	  &\rho \frac{\partial^2 \mathbf{u}^i}{\partial t^2} &- \nabla_{j} \mathbb{P}^{ij} &= 0,\ X \in \Omega\\
 *	  &                                                  &                       J - 1 &= 0,\ X \in \Omega\\
 *	  &                                                  &                  \mathbf{u} &= 0,\ X \in \partial \Omega_D\\
 *	  &                                                  & \mathbb{P} \cdot \mathbf{n} &= 0,\ X \in \partial \Omega_N\\
 *	  &                                                  & \mathbb{P} \cdot \mathbf{n} &= -p_0(t) \mathrm{adj} \mathbb{F}^T \cdot \mathbf{n},\ X \in \partial \Omega_p\\
 * \end{aligned}
 * \right. \text{, } \ 
 * \begin{aligned}
 *	  &             \mathbb{P} = 2\mathbb{F}\frac{\partial \psi}{\partial \mathbb{C}} - pJ\mathbb{F}^{-T}\\
 *	  &       \psi(\mathbb{E}) = \frac{\mu}{2} (e^{b\cdot\mathrm{tr} \mathbb{E}^2} - 1)\\
 *    &             \mathbb{E} = \frac{\mathbb{C} - \mathbb{I}}{2}\\
 *    &\mathrm{adj} \mathbb{F} = \det(\mathbb{F}) \mathbb{F}^{-1}\\
 *    &                      J = \det(\mathbb{F})\\
 *    &             \mathbb{F} = \mathbb{I} + \nabla \mathbf{u},\ \mathbb{C} = \mathbb{F}^T \cdot \mathbb{F} 
 * \end{aligned}
 * \f]
 * with \f$p_0(t) = p_{scl} \frac{t}{T}\f$
 * We use the P2 finite elements for the displacement u and P1 finite elements for the pressure p 
 * Time derivative term descretized with central difference scheme. 
 * Final discrete scheme is
 * \f[
 *  \begin{aligned}
 *    \int_\Omega\rho \frac{\mathbf{u}^{n+1} - 2\mathbf{u}^{n} + \mathbf{u}^{n-1}}{\Delta^2 t} \cdot \mathbf{\phi} d\Omega + 
 *          &\int_\Omega \mathbb{P}(\nabla \mathbf{u}^{n+1}) : \nabla \mathbb{\phi} d\Omega 
 *                  + \int_{\partial \Omega_p} p_0^{n+1} \mathbf{n}\cdot (\mathrm{adj}\ \mathbb{F}(\nabla \mathbf{u}^{n+1})) \cdot \mathbf{\phi} dS &= 0\\
 *          &\int_\Omega (J(\nabla \mathbf{u}^{n+1}) - 1) q d\Omega &= 0
 *  \end{aligned}
 * \f]
 **/
int main(int argc, char* argv[]){
    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, "", pRank, pCount);
    std::string fmesh = "../../../data/mesh/beam.msh",
                save_dir = "";
    //[m] = mg, [x] = mm, [t] = ms, [p] = kPa
    double  rho = 1, //mg/mm^3
            mu = 2, p_scl = 1e-2, //kPa
            b = 4,
            T = 100, dt = 1;  //ms        
    
    //process command line args
    auto print_help_message = [pRank](const std::string& prefix = "", const std::pair<bool, int>& exit_exec = {false, 0}, std::ostream& out = std::cout){
         if (pRank == 0){
            out     << prefix << "Help message: " << "\n"
                    << prefix << " Command line options: " << "\n"
                    << prefix << "  -m , --mesh   FILE    <Input mesh file, default=\"../../../data/mesh/beam.msh\">" << "\n"
                    << prefix << "  -t , --target PATH    <Directory to save results, default=\"\">" << "\n"
                    << prefix << "  -dt, --time_step DVAL <Time step of time disretization, default=1.0>" << "\n"
                    << prefix << "  -T , --end_time DVAL  <Final time of time integaration, default=100.0>" << "\n"
                    << prefix << "  -p , --prms DVAL[4]   <Set problem parameters rho mu b p_scl, default=1.0 2.0 4.0 1e-2>" << "\n"
                    << prefix << "  -h , --help           <Print this message>" << "\n"
                    << std::endl;
         }
        if (exit_exec.first) exit(exit_exec.second);
    };
    auto print_input_params = [&](const std::string& prefix = "", std::ostream& out = std::cout){
        if (pRank == 0){
            out     << prefix << "mesh_file = \"" << fmesh << "\"" << "\n"
                    << prefix << "save_dir  = \"" << save_dir << "\"" << "\n"
                    << prefix << "rho = " << rho << " mu = " << mu << " b = " << b << " p_scl = " << p_scl << "\n"
                    << prefix << "T = " << T << " dt = " << dt << "\n"
                    << std::endl;
        }
    };  
    #define GETARG(X)   if (i+1 < argc) { X }\
                        else { if (pRank ==0) std::cerr << "ERROR: Not found argument" << std::endl; exit(-1); }
    for (int i = 1; i < argc; i++) {
               if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help_message("", {true, 0});
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mesh") == 0) {
            GETARG(fmesh = argv[++i];)
            continue;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--target") == 0) {
            GETARG(save_dir = argv[++i];)
            continue;
        } else if (strcmp(argv[i], "-dt") == 0 || strcmp(argv[i], "--time_step") == 0) {
            GETARG(dt = std::stod(argv[++i]);)
            continue;
        } else if (strcmp(argv[i], "-T") == 0 || strcmp(argv[i], "--end_time") == 0) {
            GETARG(T = std::stod(argv[++i]);)
            continue;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prms") == 0) {
            std::array<double*, 4> prms = {&rho, &mu, &b, &p_scl};
            for (int j = 0; j < 4; ++j)
                if (i + 1 < argc){
                    std::string s(argv[i+1]);
                    if (std::isdigit(s[0]) || (s[0] == '-' && (s[1] == '.' || std::isdigit(s[1])))){
                        *(prms[j]) = std::stod(s);
                        ++i;
                    } else break;
                }
            continue;    
        } else {
            if (pRank == 0) std::cerr << "Faced unknown command \"" << argv[i] << "\"" << "\n";
            print_help_message("  ", {true, -1});
        }
    }
    #undef GETARG
    if (!save_dir.empty()){
        path p(save_dir);
        if (!exists(status(p)) || !is_directory(status(p))){
            if (pRank == 0) std::cout << ("save_dir = \"" + save_dir + "\" is not exists") << std::endl;
            BARRIER;
            abort();
        }
    }
    if (!save_dir.empty() && save_dir.back() != '/') save_dir += "/";
    print_input_params();     
    {
        Mesh m;
        if (pRank == 0) {
            std::cout << "--- Load mesh ---" << std::endl;
            m.Load(fmesh);   
        }
        RepartMesh(&m);
        Tag gmsh = m.GetTag("GMSH_TAGS");
        Tag Label = m.CreateTag("Lbl", DATA_INTEGER, NODE|EDGE|FACE, NONE, 1);
        for (auto f = m.BeginElement(NODE|EDGE|FACE); f != m.EndElement(); ++f) 
            f->Integer(Label) = 0;
        for (auto f = m.BeginFace(); f != m.EndFace(); ++f){
            f->Integer(Label) = f->Integer(gmsh);
            auto nds = f->getNodes();
            for (unsigned i = 0; i < nds.size(); ++i)
                if (nds[i].Integer(Label) > 0) nds[i].Integer(Label) |= f->Integer(Label);
                else nds[i].Integer(Label) = f->Integer(Label);
            auto eds = f->getEdges();
            for (unsigned i = 0; i < eds.size(); ++i)
                if (eds[i].Integer(Label) > 0) eds[i].Integer(Label) |= f->Integer(Label);
                else eds[i].Integer(Label) = f->Integer(Label);
        }
        m.ExchangeData(Label, NODE|EDGE|FACE);    
        if (pRank == 0) std::cout << "--- Mesh loaded ---" << std::endl;

        ProblemDefinition pr(m);    
        pr.rho = rho, pr.mu = mu, pr.p_scl = p_scl, pr.b = b;
        pr.T = T, pr.dt = dt;
        pr.Label = Label;
#ifdef WITH_CASADI
        pr.GenerateTermsFromSymbolic();
#endif
        pr.PrepareTensors(false)  
          .SetInitialCondition()
          .SetProblemDefinition()
          .InitLinSolver()
          .InitNonLinSolver()
          .Solve(true, save_dir); 
    }

    InmostFinalize();
    return 0;
}