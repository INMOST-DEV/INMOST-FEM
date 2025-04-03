//
// Created by Liogky Alexey on 21.10.2022.
//


#include "inmost.h"
#include "anifem++/utils/utils.h"
#include "anifem++/fem/operations/eval.h"
#include "anifem++/inmost_interface/elemental_assembler.h"
#include "anifem++/inmost_interface/assembler.h"
#include <numeric>
#if __cplusplus >= 201703L
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

#if __cplusplus >= 201703L
using namespace std::filesystem;
#else
using namespace std::experimental::filesystem;
#endif

using namespace INMOST;

/** The program generates and solves a nonlinear finite element system for the
 * Navier-Stokes problem:
 * \f[
 * \begin{aligned} 
 *  & \frac{\partial u}{\partial t} - \frac{\mu}{\rho} \mathrm{div}\ \mathrm{grad}\ u 
 *                                                  + (u\cdot \mathrm{grad}) u + &\mathrm{grad}\ p &= f\   in\ \Omega\\ 
 *  &                                                                            &\mathrm{div}\  u &= 0\   in\ \Omega\\ 
 *  &                                                                            &               u &= g\   on\ \partial \Omega_1 \cup \partial \Omega_2\\
 *  &                                                             & \frac{\partial u}{\partial n}  &= 0\   on\ \partial \Omega_3\\
 *  &                                                             &                              u &= u_0\  on\ t = 0   
 * \end{aligned}
 * \f]
 * 
 * We use the P2 finite elements for the velocity u and P1
 * finite elements for the pressure p and add SUPG component for stability
 * 
 * Solution of nonlinear problem performed with backward Euler scheme with 
 * extrapolation of velocity from previous step
 * 
 * g(x) = u_{stationary}(x)
 * u_0(x) = (1-perturb) u_{stationary}
 * f = (0, 0, 0)^T
 * u_{stationary}(x) = (0, max_vel*(1 - (x^2 + z^2)), 0)^T
 */
int main(int argc, char* argv[]){
    int pRank = 0, pCount = 1;
    InmostInit(&argc, &argv, "", pRank, pCount);
    std::string fmesh = "../../../data/mesh/cilinder.vtu",
                save_dir = "";
    // units are SI, except length in cm           
    double rhof = 105e-5, muf = 42e-6, max_vel = 165.;
    double dt = 0.01, T = 10;  
    int max_quadrature_order = 2; 
    double perturb = 0.01;         

    //process command line args
    auto print_help_message = [pRank](const std::string& prefix = "", const std::pair<bool, int>& exit_exec = {false, 0}, std::ostream& out = std::cout){
         if (pRank == 0){
            out     << prefix << "Help message: " << "\n"
                    << prefix << " Command line options: " << "\n"
                    << prefix << "  -m, --mesh   FILE   <Input mesh file, default=\"../../../data/mesh/cilinder.vtu\">" << "\n"
                    << prefix << "  -t, --target PATH   <Directory to save results, default=\"\">" << "\n"
                    << prefix << "  -dt, --time_step DVAL <Time step of time disretization, default=0.01>" << "\n"
                    << prefix << "  -T, --end_time DVAL <Final time of time integaration, default=10.0>" << "\n"
                    << prefix << "  -p, --prms DVAL[3] <Set problem parameters rhof muf max_vel, default=105e-5 42e-6 165>" << "\n"
                    << prefix << "  -q, --quad_order IVAL <Maximal FEM quadrature formula order, default=2>" << "\n"
                    << prefix << "      --perturb DVAL <Perturbation of initial guess, default=0.01>" << "\n"
                    << prefix << "  -h, --help          <Print this message>" << "\n"
                    << std::endl;
         }
        if (exit_exec.first) exit(exit_exec.second);
    };
    auto print_input_params = [&](const std::string& prefix = "", std::ostream& out = std::cout){
        if (pRank == 0){
            out     << prefix << "mesh_file = \"" << fmesh << "\"" << "\n"
                    << prefix << "save_dir  = \"" << save_dir << "\"" << "\n"
                    << prefix << "rhof = " << rhof << " muf = " << muf << " max_vel = " << max_vel << "\n"
                    << prefix << "T = " << T << " dt = " << dt << "\n"
                    << prefix << "perturbation = " << perturb << "\n"
                    << prefix << "Max quadrature order = " << max_quadrature_order << "\n"
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
        } else if (strcmp(argv[i], "--perturb") == 0) {
            GETARG(perturb = std::stod(argv[++i]);)
            continue;
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quad_order") == 0) {
            GETARG(max_quadrature_order = std::stol(argv[++i]);)
            continue;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prms") == 0) {
            std::array<double*, 3> prms = {&rhof, &muf, &max_vel};
            for (int j = 0; j < 3; ++j)
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
    if (perturb < 0) perturb = 0;
    else if (perturb > 1) perturb = 1;
    print_input_params();

    {//we should embrace mesh definition to enforce call of Mesh destructor 
     //before MPI will be finalized
        //Read the mesh in sequential mode and divide it between processors
        Mesh m;
        if (pRank == 0) {
            std::cout << "--- Load mesh ---" << std::endl;
            m.Load(fmesh);   
        }
        RepartMesh(&m);
        //Label NODE|EDGE|FACE|CELL
        //Mark the points
        Tag Label = m.GetTag("Label");
        if (pRank == 0) std::cout << "--- Mesh loaded ---" << std::endl;
        
        using namespace Ani;
        using UFem = FemVec<3, FEM_P2>;
        using PFem = FemFix<FEM_P1>;
        struct ProbData{
            //for computing of boundary conditions
            std::array<int, 4> nlbl;
            //std::array<int, 4> flbl;
            //for convec tensor:
            ArrayView<> uprev;
            ArrayView<const double> XY; 
        };
        // Define tensors from the problem
        auto _get_u = [](const Coord<> &X, void *user_data)->std::array<double, 3>{
            std::array<double, 3> udat;
            auto& dat = *static_cast<ProbData*>(user_data);
            ArrayView<> u(udat.data(), 3);
            fem3DapplyX<Operator<IDEN, UFem>>(dat.XY.data, dat.XY.data+3, dat.XY.data+6, dat.XY.data+9, X.data(), dat.uprev, u);

            return udat;
        };
        auto mass = TensorNull<>;
        auto visc1 = makeConstantTensorScalar(muf/rhof);
        auto convec = [_get_u](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) iTet;
            //K_{k}{ji} \nabla_i v_j w_k, K_{k}{ji} = v_prev_i \delta_{jk}
            Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
            K.SetZero();
            auto u = _get_u(X, user_data);

            for (int k = 0; k < 3; ++k)
                for (int i = 0; i < 3; ++i)
                    K(k, i + 3*k) = u[i]; //j=k      
            return Ani::TENSOR_GENERAL;
        };
        auto pressure = TensorNull<>;
        auto rhs_time = [_get_u](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) dims; (void) iTet;
            auto u = _get_u(X, user_data);
            for (int i = 0; i < 3; ++i) Kdat[i] = u[i];
            return Ani::TENSOR_GENERAL;
        };
        auto rhsf = [](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) X; (void) dims; (void) user_data; (void) iTet;
            std::fill(Kdat, Kdat + 3, 0);
            return Ani::TENSOR_GENERAL;
        };
        auto supg_mass = [_get_u](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) iTet;
            //K_{ji}{k} v_k \nabla_i w_j , K_{ji}{k} = v_prev_i \delta_{jk}
            Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
            K.SetZero();
            auto u = _get_u(X, user_data);
            for (int k = 0; k < 3; ++k)
                for (int i = 0; i < 3; ++i)
                    K(i + 3*k, k) = u[i]; //j=k      
            return Ani::TENSOR_GENERAL;
        };
        auto supg_convec = [_get_u](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) iTet;
            //K{ji}{lk} \nabla_k v_l \nabla_i v_j, K{ji}{lk} = v_prev_k v_prev_i \delta_{lj}
            Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
            K.SetZero();
            std::array<double, 3> u = _get_u(X, user_data);// {0};// _get_u(X, user_data);
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int i = 0; i < 3; ++i)
                        K(i + 3*j, k + 3*j) = u[k]*u[i]; //l == j
            return Ani::TENSOR_GENERAL;    
        };
        auto supg_rhs_time = [_get_u](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) iTet;
            //K{ji}{} \nabla_i v_j, K{ji}{} = v_prev_i v_prev_j
            Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
            K.SetZero();
            auto u = _get_u(X, user_data);
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    K(i + 3*j, 0) = u[i]*u[j];
            return Ani::TENSOR_GENERAL;    
        };
        auto supg_const = [_get_u](const Coord<> &X, double *Kdat, TensorDims dims, void *user_data, int iTet) {
            (void) iTet;
            //K{i}{} \nabla_i v, K{i}{} = v_prev_i
            Ani::DenseMatrix<> K(Kdat, dims.first, dims.second);
            K.SetZero();
            auto u = _get_u(X, user_data);
            for (int i = 0; i < 3; ++i)
                K(i, 0) = u[i];
            return Ani::TENSOR_GENERAL;    
        };
        auto U_bc = [val = max_vel](const Coord<> &X)->std::array<double, 3>{
            return {0, val * (1.0 - (X[0]*X[0] + X[2]*X[2])), 0};
        };

        std::function<void(const double**, double*, double*, void*)> local_assembler =
            [&mass, &visc1, &convec, &pressure, &rhs_time, &rhsf, 
             &supg_mass, &supg_convec, &supg_rhs_time, &supg_const, _get_u, &U_bc,
             &dt, nu = muf/rhof, mord = max_quadrature_order](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data) -> void{
            auto& dat = *static_cast<ProbData*>(user_data);
            dat.XY.Init(XY[0], 3*4);
            const DenseMatrix<const double>
                                    XY0(XY[0], 3, 1),
                                    XY1(XY[1], 3, 1),
                                    XY2(XY[2], 3, 1),
                                    XY3(XY[3], 3, 1),
                                    XYFull(XY[0] + 0, 3, 4);
            constexpr auto UNFA = Operator<IDEN, UFem>::Nfa::value, PNFA = Operator<IDEN, PFem>::Nfa::value; //30 and 4
            DenseMatrix<> A(Adat, UNFA+PNFA, UNFA+PNFA), F(Fdat, UNFA+PNFA, 1);
            std::array<double, (UNFA+PNFA)*(UNFA+PNFA)> Bp; DenseMatrix<> B(Bp.data(), UNFA, UNFA);
            
            // compute element matrix
            fem3Dtet<Operator<IDEN, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_NULL, true>>(XY0, XY1, XY2, XY3, mass, B, std::min(mord, 4));
            for (int j = 0; j < UNFA; ++j) 
                for (int i = 0; i < UNFA; ++i) 
                    A(i, j) = B(i, j) / dt;
            fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(XY0, XY1, XY2, XY3, visc1, B, std::min(mord, 2));
            for (int j = 0; j < UNFA; ++j) 
                for (int i = 0; i < UNFA; ++i) 
                    A(i, j) += B(i, j);
            fem3Dtet<Operator<GRAD, UFem>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, convec, B, std::min(mord, 5), user_data);
            for (int j = 0; j < UNFA; ++j) 
                for (int i = 0; i < UNFA; ++i) 
                    A(i, j) += B(i, j);
            B.Init(Bp.data(), UNFA, PNFA, Bp.size());
            fem3Dtet<Operator<IDEN, PFem>, Operator<DIV, UFem>, DfuncTraits<TENSOR_NULL, true>>(XY0, XY1, XY2, XY3, pressure, B, std::min(mord, 2));
            for (int j = 0; j < PNFA; ++j) 
                for (int i = 0; i < UNFA; ++i) 
                    A(UNFA+j, i) = A(i, UNFA+j) = -B(i, j);
            
            // compute element RHS
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, rhs_time, F, std::min(mord, 4), user_data);
            F.nRow = UNFA+PNFA;
            B.Init(Bp.data(), UNFA, 1, B.size);
            //fem3Dtet<Operator<IDEN, FemVec<3, FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(XY0, XY1, XY2, XY3, rhsf, B, 2);
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(XY0, XY1, XY2, XY3, rhsf, B, std::min(mord, 2));
            for (int i = 0; i < UNFA; ++i) F.data[i] = F.data[i]/dt + B.data[i];
            
            // add SUBG contribution
            double XYPp[4*3]; DenseMatrix<> XYP(XYPp, 3, 3);
            double Gp[3*3]; DenseMatrix<> G(Gp, 3, 3); G.SetZero();
            for (int j = 0; j < 3; ++j) 
                for (int i = 0; i < 3; ++i) 
                    XYP(i, j) = XY[j+1][i] - XY[0][i];
            for (int j = 0; j < 3; ++j) 
                for (int k = 0; k <= j; ++k) 
                    for (int i = 0; i < 3; ++i) 
                        G(j, k) += XYP(i, j) * XYP(i, k);
            for (int j = 0; j < 3; ++j) 
                for (int k = j+1; k < 3; ++k) 
                    G(j, k) = G(k, j);
            double Ginvp[3*3]; DenseMatrix<> Ginv(Ginvp, 3, 3);
            auto detG = inverse3x3(Gp, Ginvp);
            std::array<double, 10> lapl = {0};
            for (int j = 0; j < 3; ++j) 
                for (int i = 0; i < 3; ++i) 
                    lapl[0] += Ginv(i, j);
            lapl[0] *= 4;
            for (int k = 0; k < 3; ++k) {
                lapl[k+1] = 4*Ginv(k, k);
                lapl[k+4] = -8*(Ginv(0, k) + Ginv(1, k) + Ginv(2, k));
            }
            lapl[7] = 8*Ginv(1, 0), lapl[8] = 8*Ginv(2,0), lapl[9] = 8*Ginv(2,1);
            
            //     compute mass matrix
            std::array<double, UNFA*UNFA> mp; DenseMatrix<> m(mp.data(), UNFA, UNFA);
            fem3Dtet<Operator<IDEN, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, supg_mass, m, std::min(mord, 5), user_data);
            for (int j = 0; j < UNFA; ++j) 
                for (int i = 0; i < UNFA; ++i) 
                    m(i, j) /= dt;
            //     compute convection matrix
            std::array<double, UNFA*UNFA> bp; DenseMatrix<> b(bp.data(), UNFA, UNFA);
            fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, supg_convec, b, std::min(mord, 5), user_data);
            //     compute pressure matrix
            std::array<double, UNFA*PNFA> qp; DenseMatrix<> q(qp.data(), UNFA, PNFA);
            fem3Dtet<Operator<GRAD, PFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, supg_mass, q, std::min(mord, 5), user_data);
            //     compute rhs time matrix
            std::array<double, UNFA*1> f2p; DenseMatrix<> f2(f2p.data(), UNFA, 1);
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, supg_rhs_time, f2, std::min(mord, 5), user_data);
            for (int i = 0; i < UNFA; ++i) f2(i, 0) /= dt;
            //     compute viscosity matrix
            std::array<double, UNFA*1> f3p; DenseMatrix<> f3(f3p.data(), UNFA, 1);
            fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<GRAD, FemFix<FEM_P2>>, DfuncTraits<TENSOR_GENERAL, false>>(XY0, XY1, XY2, XY3, supg_const, f3, std::min(mord, 5), user_data);

            constexpr auto UNFA1 = Operator<IDEN, UFem::Base>::Nfa::value;
            std::array<double, UNFA1*UNFA1> gp; DenseMatrix<> g(gp.data(), UNFA1, UNFA1);
            for (int i = 0; i < UNFA1; ++i)
                for (int j = 0; j < UNFA1; ++j)
                    g(i, j) = -nu*lapl[j]*f3(i, 0);  
            Coord<> xc;
            for (int i = 0; i < 3; ++i) xc[i] = (XY[0][i] + XY[1][i] + XY[2][i] + XY[3][i])/4;
            auto uc = _get_u(xc, user_data);
            std::array<std::array<int, 4>, 4> id = {std::array<int, 4>{1, 3, 2, 3}, {0, 2, 3, 2}, {0, 3, 1, 3}, {0, 1, 2, 1}};
            DenseMatrix<> B1(XYPp, 4, 3);
            for (int i = 0; i < 4; ++i) 
                for (int j = 0; j < 3; ++j)
                    B1(i, j) = (XY[id[i][0]][(j+1)%3] - XY[id[i][1]][(j+1)%3])*(XY[id[i][2]][(j+2)%3] - XY[id[i][3]][(j+2)%3]) - (XY[id[i][0]][(j+2)%3] - XY[id[i][1]][(j+2)%3])*(XY[id[i][2]][(j+1)%3] - XY[id[i][3]][(j+1)%3]);
            for (int i = 0; i < 3; ++i) 
                for (int j = 0; j <= i; ++j)
                    G(i, j) = B1(0, i)*B1(0, j) + B1(1, i)*B1(1, j) + B1(2, i)*B1(2, j) + B1(3, i)*B1(3, j);
            for (int i = 0; i < 3; ++i) 
                for (int j = i+1; j < 3; ++j) 
                    G(i, j) = G(j, i);
            auto detB2 = detG;
            for (int i = 0; i < 3; ++i) 
                for (int j = 0; j < 3; ++j) 
                    G(i, j) /= detB2;
            double fbnrm = 0, fbnrm1 = 0, fbnrm2 = 4.0 / (dt * dt);
            for (int i = 0; i < 3; ++i) 
                for (int j = 0; j < 3; ++j){
                    fbnrm += uc[i] * uc[j] * G(i, j);
                    fbnrm1 += G(i, j) * G(i, j);
                }
            fbnrm1 *= 60 * nu *nu;
            double delta = 1.0/sqrt(fbnrm2 + fbnrm + fbnrm1);

            for (int j = 0; j < UNFA; ++j) 
                for (int i = 0; i < UNFA; ++i) 
                    A(i, j) += delta * (m(i, j) + b(i, j) + (((i/UNFA1)==(j/UNFA1)) ? g(i%UNFA1, j%UNFA1) : 0) );
            for (int j = UNFA; j < UNFA + PNFA; ++j) 
                for (int i = 0; i < UNFA; ++i)
                    A(i, j) += delta * q(i, j - UNFA);
            for (int i = 0; i < UNFA; ++i) F.data[i] += delta * f2.data[i];
            /* END SUPG contribution */

            //Impose dirichlet boundary conditions
            //  at vertices
            for (int k = 0; k < 4; ++k){
                if (dat.nlbl[k] > 0 && dat.nlbl[k] != 3){
                    std::array<double, 3> x;
                    DOF_coord<UFem::Base>::at(k, XY[0], x.data()); // compute coordinate of k-th degree of freedom
                    auto ebc = U_bc(x);
                    for (int dim = 0; dim < UFem::Dim::value; ++dim)
                        applyDir(A, F, k+dim*UNFA1, ebc[dim]);
                }
            }
            //  at mid-points of edges
            for (int i = 1, k = 3; i <= 3; ++i)
                for (int j = i + 1; j <= 4; ++j){
                    ++k;
                    auto i1 = dat.nlbl[i-1], i2 = dat.nlbl[j-1];
                    if (!(i1*i2 == 1 || i1*i2 > 3)) continue;
                    std::array<double, 3> x;
                    DOF_coord<UFem::Base>::at(k, XY[0], x.data());
                    std::array<double, 3UL> ebc = U_bc(x);
                    if (i1 != 3 && i2 != 3)
                    for (int dim = 0; dim < UFem::Dim::value; ++dim)
                        applyDir(A, F, k+dim*UNFA1, ebc[dim]);
                }
            return;
        };

        // create solution storage
        if (pRank == 0) std::cout << "--- Set initial guess ---" << std::endl;
        Tag ue = m.CreateTag("u_stat", DATA_REAL, NODE|EDGE, NONE, 3);
        Tag pe = m.CreateTag("p_stat", DATA_REAL, NODE, NONE, 1);
        // set initial guess
        for (auto e = m.BeginNode(); e != m.EndNode(); ++e) {
            auto x = e->Coords()[0], y = e->Coords()[1], z = e->Coords()[2];
            e->RealArray(ue)[0] = 0;
            e->RealArray(ue)[1] = max_vel * (1.0 - (x*x + z*z));
            e->RealArray(ue)[2] = 0;
            e->Real(pe) = 2.0 * muf * max_vel * (10.0 - 2*y) / rhof;
        }
        for (auto e = m.BeginEdge(); e != m.EndEdge(); ++e) {
            double x = 0, y = 0, z = 0;
            auto nd = e->getNodes();
            for (unsigned n = 0; n < nd.size(); ++n)
                x += nd[n].Coords()[0], y += nd[n].Coords()[1], z += nd[n].Coords()[2];
            x /= nd.size(), y /= nd.size(), z /= nd.size();
            e->RealArray(ue)[0] = 0;
            e->RealArray(ue)[1] = max_vel * (1.0 - (x*x + z*z));
            e->RealArray(ue)[2] = 0;
        }
        Tag u = m.CreateTag("u", DATA_REAL, NODE|EDGE, NONE, 3);
        Tag p = m.CreateTag("p", DATA_REAL, NODE, NONE, 1);
        double error_scale = 1 - perturb;
        std::vector<Tag> up = {u, p};
        {   //copy solution
            std::array<Tag, 2> upe = {ue, pe};
            for (auto e = m.BeginElement(NODE|EDGE); e != m.EndElement(); ++e){
                auto nds = e->getNodes();
                bool is_dirichlet = false;
                if (nds.size() == 1) is_dirichlet = (nds[0].Integer(Label)>0 && nds[0].Integer(Label)!=3);
                if (nds.size() == 2) {
                    int i1 = nds[0].Integer(Label), i2 = nds[1].Integer(Label);
                    is_dirichlet = (i1*i2 == 1 || i1*i2 > 3) && (i1 != 3) && (i2 != 3);
                }
                for (int i = 0; i < 2; ++i){
                    auto& var = up[i], &vare = upe[i];
                    if (var.isDefined(e->GetElementType())){
                        auto dat = e->RealArray(var), date = e->RealArray(vare);
                        for (unsigned k = 0; k < dat.size(); ++k) dat[k] = (is_dirichlet ? 1 : error_scale) * date[k];
                    }   
                }
            }
        }
        auto local_data_gatherer = [&Label, &up](ElementalAssembler& p) -> void{
            double *nn_p = p.get_nodes();
            const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
            double udat[Operator<IDEN, UFem>::Nfa::value];
            ProbData data;
            for (int i = 0; i < 4; ++i) {
                data.nlbl[i] = (*p.nodes)[i].Integer(Label);
                //data.flbl[i] = (*p.nodes)[i].Integer(BndLabel);
            }
            data.uprev.Init(udat, Operator<IDEN, UFem>::Nfa::value);
            ElementalAssembler::GatherDataOnElement(up, p, udat, 0);
            //std::cout << "u_prev = \n" << DenseMatrix<const double>(data.uprev.data, 10, 3) << std::endl;
            p.compute(args, &data);
        };

        if (pRank == 0) std::cout << "--- Set problem definition ---" << std::endl;
        Assembler discr(m);
        constexpr auto elem_nfa = Operator<IDEN, FemCom<UFem, PFem>>::Nfa::value;
        discr.SetMatRHSFunc(GenerateElemMatRhs(local_assembler, elem_nfa, elem_nfa));
        {
            auto Var0Helper = GenerateHelper<UFem>(),
                 Var1Helper = GenerateHelper<PFem>();
            FemExprDescr fed;
            fed.PushTrialFunc(Var0Helper, "u"); fed.PushTestFunc(Var0Helper, "phi_u");
            fed.PushTrialFunc(Var1Helper, "p"); fed.PushTestFunc(Var1Helper, "phi_p");
            discr.SetProbDescr(std::move(fed));
        }
        discr.SetDataGatherer(local_data_gatherer);
        discr.PrepareProblem();
        Sparse::Matrix A("A");
        Sparse::Vector x("x"), b("b");
        A.SetInterval(discr.getBegInd(), discr.getEndInd()); //initialized as free matrix
        b.SetInterval(discr.getBegInd(), discr.getEndInd()); //initialized by zeros
        Solver solver(Solver::INNER_MPTILUC);
        solver.SetParameterReal("relative_tolerance", 1e-10);
        solver.SetParameterReal("absolute_tolerance", 1e-10);
        solver.SetParameterReal("drop_tolerance", 5e-2);
        solver.SetParameterReal("reuse_tolerance", 5e-3);
        bool success_solve = true;
        auto print_solver_status = [&s = solver, &success_solve, pRank, pCount](double tcur, double tfinal, const std::string& prefix = ""){
            if(!success_solve){
                if (pRank == 0) std::cout << prefix << "Navier-stokes["<< tcur << " / " << tfinal << "]" << ": solution failed\n";
                for (int p = 0; p < pCount; ++p) {
                    BARRIER;
                    if (pRank != p) continue;
                    std::cout << prefix << " Iterations " << s.Iterations() << " Residual " << s.Residual() << ". "
                            << "Precond time = " << s.PreconditionerTime() << ", solve time = " << s.IterationsTime() << std::endl;
                    std::cout << prefix << " Rank " << pRank << " failed to solve system. ";
                    std::cout << "Reason: " << s.GetReason() << std::endl;
                }
                exit(-1);
            }
            else{
                if(pRank == 0) {
                    std::cout << prefix << "Navier-stokes["<< tcur << " / " << tfinal << "]" << ":\n";
                    std::string _s_its = std::to_string(s.Iterations()), s_its = "    ";
                    std::copy(_s_its.begin(), _s_its.end(), s_its.begin() + 3 - _s_its.size());

                    std::cout << prefix << " solved_succesful: #lits " << s_its << " residual " << s.Residual() << ", "
                            << "preconding = " << s.PreconditionerTime() << "s, solving = " << s.IterationsTime() << "s" << std::endl;
                }
            }
        };
        if (pRank == 0) std::cout << "--- Save analytic solution ---" << std::endl;
        discr.SaveSolution(up, x); //< set initial guess on working vector

        if (pRank == 0) std::cout << "--- Time loop ---" << std::endl;
        double tcur = 0;
        while (tcur < T){
            auto dtloc = dt;
            if (tcur + dt > T){
                dtloc = T - tcur;
                tcur = T;
                dt = dtloc;
            } else {
                tcur += dt;
            }
            if (pRank == 0) std::cout << "    t = " << tcur << ":"  << std::endl;
            if (pRank == 0) std::cout << "    - Assemble matrix and rhs" << std::endl;
            discr.Assemble(A, b);
            if (pRank == 0) std::cout << "    - Set preconditioner" << std::endl;
            solver.SetMatrix(A);
            if (pRank == 0) std::cout << "    - Solve" << std::endl;
            success_solve = solver.Solve(b, x);
            print_solver_status(tcur, T, "      ");
            discr.SaveSolution(x, up);
            {
                double nrm = 0;
                for (auto e = m.BeginElement(NODE|EDGE); e != m.EndElement(); ++e){
                    if (e->GetStatus() == INMOST::Element::Ghost) continue; //ignore ghost elements
                    for (auto& var: up)
                        if (var.isDefined(e->GetElementType())){
                            auto dat = e->RealArray(var);
                            for (auto& i : dat) nrm += i * i;
                        }   
                }
                nrm = sqrt(m.Integrate(nrm));
                if (pRank == 0) std::cout << "    - Solution norm = " << nrm << std::endl;
            }
            m.Save(save_dir + "res_" + std::to_string(tcur) + ".vtk");
        }

        auto aggregate_relative_error = [&m](const std::vector<Tag> vars, const std::vector<Tag> evars, INMOST::ElementType t){
            if (vars.size() != evars.size()) throw std::runtime_error("Wrong input");
            std::vector<double> nrm(vars.size()*2);
            for (auto e = m.BeginElement(t); e != m.EndElement(); ++e){
                if (e->GetStatus() == INMOST::Element::Ghost) continue; //ignore ghost elements
                for (unsigned v = 0; v < vars.size(); ++v){
                    auto& var = vars[v], &evar = evars[v];
                    if (var.isDefined(e->GetElementType()) && evar.isDefined(e->GetElementType())){
                        auto dat = e->RealArray(var), date = e->RealArray(evar);
                        for (unsigned i = 0; i < dat.size(); ++i)
                            nrm[2*v] = date[i] * date[i], nrm[2*v + 1] = (dat[i] - date[i]) * (dat[i] - date[i]);
                    }
                }  
            }
            m.Integrate(nrm.data(), nrm.size());
            for (auto& i : nrm) i = sqrt(i);
            return nrm;
        };
        auto nrmu = aggregate_relative_error({u}, {ue}, NODE|EDGE);
        if (pRank == 0) std::cout << "Velocity relative error = " << nrmu[1] / nrmu[0] << std::endl;
        auto nrmp = aggregate_relative_error({p}, {pe}, NODE);
        if (pRank == 0) std::cout << "Pressure relative error = " << nrmp[1] / nrmp[0] << std::endl;
        auto nrms = aggregate_relative_error({u, p}, {ue, pe}, NODE|EDGE);
        if (pRank == 0) std::cout << "Entire solution relative error = " << nrms[1] / nrms[0] << std::endl;
        if (pRank == 0) std::cout<<"Finished!"<<std::endl;
    }

    InmostFinalize();
    return 0;
}