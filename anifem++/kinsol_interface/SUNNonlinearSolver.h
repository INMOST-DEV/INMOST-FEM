//
// Created by Liogky Alexey on 23.10.2022.
//

#ifndef CARNUM_SOLVER_SUNNONLINEARSOLVER_H
#define CARNUM_SOLVER_SUNNONLINEARSOLVER_H

#include "inmost.h"
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_linearsolver.h>
#include <kinsol/kinsol.h>
#include <regex>

///To store internal data of N_Vector of Inmost type (SUNDIALS_NVEC_CUSTOM)
///@see N_Vector in sundials_nvector.h
struct _N_VectorContent_Inmost {
    bool own_data;                  ///< data ownership flag
    INMOST::Sparse::Vector* data;   ///< pointer to INMOST vector
};
typedef struct _N_VectorContent_Inmost* N_VectorContent_Inmost;

#if SUNDIALS_VERSION_MAJOR >= 6
    #define SUN_CTX(X) SUNContext X
    #define SUN_CTX_COMMA(X) SUN_CTX(X), 
#else
    #define SUN_CTX(X) 
    #define SUN_CTX_COMMA(X)      
#endif

/// @brief  Set INMOST interface operations on N_Vector @return 0 on success
int          N_VInitOps_Inmost(N_Vector v);
/// @brief  Create empty (without content, but with operations) N_Vector of Inmost type @return not NULL on success
N_Vector     N_VNewEmpty_Inmost(SUN_CTX(ctx));
/// @brief  Construct N_Vector of Inmost type that doesn't own by it's data @return not NULL on success
N_Vector     N_VNew_Inmost(SUN_CTX_COMMA(ctx) INMOST::Sparse::Vector* data);
/// @brief  Construct N_Vector of Inmost type that own by it's data @return not NULL on success
N_Vector     N_VNew_Inmost(SUN_CTX_COMMA(ctx) std::string name);

/// @brief N_Vector operation nvgetvectorid(...) @return SUNDIALS_NVEC_CUSTOM @see N_Vector_Ops in sundials_nvector.h
N_Vector_ID  N_VGetVectorID_Inmost(N_Vector v);
/// @brief N_Vector operation nvdestroy(...) @see N_Vector_Ops in sundials_nvector.h
void         N_VDestroy_Inmost(N_Vector v);
/// @brief N_Vector operation nvgetlength(...) @return Length of the vector @see N_Vector_Ops in sundials_nvector.h
sunindextype N_VGetLength_Inmost(N_Vector v);
/// @brief N_Vector operation nvcloneempty(...) @see N_Vector_Ops in sundials_nvector.h
N_Vector     N_VCloneEmpty_Inmost(N_Vector w);
/// @brief N_Vector operation nvclone(...) @see N_Vector_Ops in sundials_nvector.h
N_Vector     N_VClone_Inmost(N_Vector w);
/// @brief N_Vector operation nvspace(...) @return approximate count of reals (in lrw) and integers (in liw) required to store vector v @see N_Vector_Ops in sundials_nvector.h
void         N_VSpace_Inmost(N_Vector v, sunindextype *lrw, sunindextype *liw);
/// @brief N_Vector operation nvgetcommunicator(...) @return MpiComm associated with vector @see N_Vector_Ops in sundials_nvector.h
void*        N_VGetCommunicator_Inmost(N_Vector v);
/// @brief N_Vector operation nvlinearsum(...): z = ax + by @see N_Vector_Ops in sundials_nvector.h
void         N_VLinearSum_Inmost(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z);
/// @brief N_Vector operation nvconst(...): z_i = c @see N_Vector_Ops in sundials_nvector.h
void         N_VConst_Inmost(realtype c, N_Vector z);
/// @brief N_Vector operation nvprod(...): z_i = x_i*y_i @see N_Vector_Ops in sundials_nvector.h
void         N_VProd_Inmost(N_Vector x, N_Vector y, N_Vector z);
/// @brief N_Vector operation nvdiv(...): z_i = x_i/y_i @see N_Vector_Ops in sundials_nvector.h
void         N_VDiv_Inmost(N_Vector x, N_Vector y, N_Vector z);
/// @brief N_Vector operation nvscale(...): z = cx @see N_Vector_Ops in sundials_nvector.h
void         N_VScale_Inmost(realtype c, N_Vector x, N_Vector z);
/// @brief N_Vector operation nvabs(...): z = |x| @see N_Vector_Ops in sundials_nvector.h
void         N_VAbs_Inmost(N_Vector x, N_Vector z);
/// @brief N_Vector operation nvinv(...): z_i = 1/x_i @see N_Vector_Ops in sundials_nvector.h
void         N_VInv_Inmost(N_Vector x, N_Vector z);
/// @brief N_Vector operation nvaddconst(...): z_i = x_i + b @see N_Vector_Ops in sundials_nvector.h
void         N_VAddConst_Inmost(N_Vector x, realtype b, N_Vector z);
/// @brief N_Vector operation nvdotprodlocal(...) @return x'*y' where x' and y' are parts of vectors x and y owned by current processor  @see N_Vector_Ops in sundials_nvector.h
realtype     N_VDotProdLocal_Inmost(N_Vector x, N_Vector y);
/// @brief N_Vector operation nvdotprod(...) @return x*y @see N_Vector_Ops in sundials_nvector.h
realtype     N_VDotProd_Inmost(N_Vector x, N_Vector y);
/// @brief N_Vector operation nvmaxnormlocal(...) @return max|x'_i| where x' is part of vector x owned by current processor  @see N_Vector_Ops in sundials_nvector.h
realtype     N_VMaxNormLocal_Inmost(N_Vector x);
/// @brief N_Vector operation nvmaxnorm(...) @return max|x_i| @see N_Vector_Ops in sundials_nvector.h
realtype     N_VMaxNorm_Inmost(N_Vector x);
/// @return (w'_i*x'_i)*(w'_i*x'_i) where w' and x' are parts of vectors w and x owned by current processor 
realtype     N_VWSqrSumLocal_Inmost(N_Vector x, N_Vector w);
/// @brief N_Vector operation nvwrmsnorm(...) @return ((Σ_i (w_i*x_i)*(w_i*x_i))/dim(x))^{1/2} @see N_Vector_Ops in sundials_nvector.h
realtype     N_VWrmsNorm_Inmost(N_Vector x, N_Vector w);
/// @return Σ_i (id'_i > 0)*(w'_i*x'_i)*(w'_i*x'_i) where id', w' and x' are parts of vectors id, w and x owned by current processor 
realtype     N_VWSqrSumMaskLocal_Inmost(N_Vector x, N_Vector w, N_Vector id);
/// @brief N_Vector operation nvwrmsnorm(...) @return ((Σ_i (id_i > 0)*(w_i*x_i)*(w_i*x_i))/dim(x))^{1/2} @see N_Vector_Ops in sundials_nvector.h
realtype     N_VWrmsNormMask_Inmost(N_Vector x, N_Vector w, N_Vector id);
/// @brief N_Vector operation nvminlocal(...) @return min (x'_i) where x' is part of vector x owned by current processor  @see N_Vector_Ops in sundials_nvector.h
realtype     N_VMinLocal_Inmost(N_Vector x);
/// @brief N_Vector operation nvmin(...) @return min(x_i) @see N_Vector_Ops in sundials_nvector.h
realtype     N_VMin_Inmost(N_Vector x);
/// @brief N_Vector operation nvwl2norm(...) @return ((Σ_i (w_i*x_i)*(w_i*x_i)))^{1/2} @see N_Vector_Ops in sundials_nvector.h
realtype     N_VWL2Norm_Inmost(N_Vector x, N_Vector w);
/// @brief N_Vector operation nvl1normlocal(...) @return Σ_i |x'_i| where x' is part of vector x owned by current processor @see N_Vector_Ops in sundials_nvector.h
realtype     N_VL1NormLocal_Inmost(N_Vector x);
/// @brief N_Vector operation nvl1norm(...) @return Σ_i |x_i| @see N_Vector_Ops in sundials_nvector.h
realtype     N_VL1Norm_Inmost(N_Vector x);
/// @brief N_Vector operation nvcompare(...): z_i = ((x_i >= c) ? 1 : 0) @see N_Vector_Ops in sundials_nvector.h
void         N_VCompare_Inmost(realtype c, N_Vector x, N_Vector z);
/// @brief N_Vector operation nvinvtestlocal(...): z'_i = 1/x'_i if x'_i != 0 
/// @return true if x'_i != 0 forall i where x' is part of vector x owned by current processor @see N_Vector_Ops in sundials_nvector.h
booleantype  N_VInvTestLocal_Inmost(N_Vector x, N_Vector z);
/// @brief N_Vector operation nvinvtest(...): z_i = 1/x_i if x_i != 0 @return true if x_i != 0 forall i @see N_Vector_Ops in sundials_nvector.h
booleantype  N_VInvTest_Inmost(N_Vector x, N_Vector z);
/// @brief N_Vector operation nvconstrmasklocal(...). m is vector of constrain violations, c is constraints 
/// if c_i = 0 -> m_i = 0;
/// if |c_i| = 1 -> m_i = (c_i * x_i <= 0);
/// if |c_i| = 2 -> m_i = (c_i * x_i < 0);
/// @return false if there are constraint violations on current processor subvectors @see N_Vector_Ops in sundials_nvector.h
booleantype  N_VConstrMaskLocal_Inmost(N_Vector c, N_Vector x, N_Vector m);
/// @brief N_Vector operation nvconstrmask(...). m is vector of constrain violations, c is constraints 
/// if c_i = 0 -> m_i = 0;
/// if |c_i| = 1 -> m_i = (c_i * x_i <= 0);
/// if |c_i| = 2 -> m_i = (c_i * x_i < 0);
/// @return false if there are constraint violations @see N_Vector_Ops in sundials_nvector.h
booleantype  N_VConstrMask_Inmost(N_Vector c, N_Vector x, N_Vector m);
/// @brief N_Vector operation nvminquotientlocal(...) @return min (num'_i / denum'_i) performed on current processor subvectors @see N_Vector_Ops in sundials_nvector.h
realtype     N_VMinQuotientLocal_Inmost(N_Vector num, N_Vector denom);
/// @brief N_Vector operation nvminquotient(...) @return min (num_i / denum_i) @see N_Vector_Ops in sundials_nvector.h
realtype     N_VMinQuotient_Inmost(N_Vector num, N_Vector denom);
/// @brief N_Vector operation nvlinearcombination(...): z = c_1*X_1+c_2*X_2+..+c_n*X_n @return 0 on success @see N_Vector_Ops in sundials_nvector.h
int          N_VLinearCombination_Inmost(int nvec, realtype* c, N_Vector* X, N_Vector z);
/// @brief N_Vector operation nvlinearcombination(...): dotprods_i = x*Y_i @return 0 on success @see N_Vector_Ops in sundials_nvector.h
int          N_VDotProdMulti_Inmost(int nvec, N_Vector x, N_Vector* Y, realtype* dotprods);
/// @brief N_Vector operation nvcontent(...) @return content of N_Vector @see N_Vector_Ops in sundials_nvector.h
N_VectorContent_Inmost
             N_VContent_Inmost(N_Vector x);


///To store internal data of SUNMatrix of Inmost type (SUNMATRIX_CUSTOM)
///@see SUNMatrix in sundials_matrix.h
struct _SUNMatrixContent_Inmost {
    bool own_data;                  ///< data ownership flag
    INMOST::Sparse::Matrix* data;   ///< pointer to INMOST matrix
    int overlap;                    ///< parallel overlap for matrix, used for MatVec operation
};
typedef struct _SUNMatrixContent_Inmost* SUNMatrixContent_Inmost;

/// @brief  Set INMOST interface operations on SUNMatrix @return 0 on success
int          SUNMatInitOps_Inmost(SUNMatrix v);
/// @brief  Create empty (without content, but with operations) SUNMatrix of Inmost type @return not NULL on success
SUNMatrix    SUNMatNewEmpty_Inmost(SUN_CTX(ctx));
/// @brief  Construct SUNMatrix of Inmost type that doesn't own by it's data @return not NULL on success
SUNMatrix    SUNMatInmost(SUN_CTX_COMMA(ctx) INMOST::Sparse::Matrix* data, int overlap = 0);
/// @brief  Construct SUNMatrix of Inmost type that own by it's data @return not NULL on success
SUNMatrix    SUNMatInmost(SUN_CTX_COMMA(ctx) std::string name);

/// @brief SUNMatrix operation destroy(...) @see SUNMatrix_Ops in sundials_matrix.h
void         SUNMatDestroy_Inmost(SUNMatrix m);
/// @brief SUNMatrix operation getid(...) @return SUNMATRIX_CUSTOM @see SUNMatrix_Ops in sundials_matrix.h
SUNMatrix_ID SUNMatGetID_Inmost(SUNMatrix A);
/// @brief SUNMatrix operation cloneempty(...) @see SUNMatrix_Ops in sundials_matrix.h
SUNMatrix    SUNMatCloneEmpty_Inmost(SUNMatrix A);
/// @brief SUNMatrix operation clone(...) @see SUNMatrix_Ops in sundials_matrix.h
SUNMatrix    SUNMatClone_Inmost(SUNMatrix A);
/// @brief SUNMatrix operation space(...) @return approximate count of reals (in lenrw) and integers (in leniw) required to store matrix A @see SUNMatrix_Ops in sundials_matrix.h
int          SUNMatSpace_Inmost(SUNMatrix A, long int *lenrw, long int *leniw);
/// @brief SUNMatrix operation zero(...): A = 0 @return 0 on success @see SUNMatrix_Ops in sundials_matrix.h
int          SUNMatZero_Inmost(SUNMatrix A);
/// @brief SUNMatrix operation copy(...): B <- A @return 0 on success @see SUNMatrix_Ops in sundials_matrix.h
int          SUNMatCopy_Inmost(SUNMatrix A, SUNMatrix B);
/// @brief SUNMatrix operation matvec(...): y = A*x @return 0 on success @see SUNMatrix_Ops in sundials_matrix.h
int          SUNMatMatvec_Inmost(SUNMatrix A, N_Vector x, N_Vector y);
/// @brief SUNMatrix operation content(...) @return content of SUNMatrix @see N_Vector_Ops in sundials_matrix.h
SUNMatrixContent_Inmost
             SUNMatContent_Inmost(SUNMatrix A);

///To store internal data of SUNLinearSolver of Inmost type (SUNLINEARSOLVER_CUSTOM)
///@see SUNMatrix in sundials_linearsolver.h
struct _SUNLinearSolverContent_Inmost {
    bool own_data;                  ///< data ownership flag 
    INMOST::Solver* data;           ///< pointer to INMOST solver
    int updated_mat;                ///< internal matix status
    int verbosity;                  ///< level of verbosity (default = 0)
};
typedef struct _SUNLinearSolverContent_Inmost* SUNLinearSolverContent_Inmost;

/// @brief  Set INMOST interface operations on SUNLinearSolver @return 0 on success
int                  SUNLinSolInitOps_Inmost(SUNLinearSolver LS);
/// @brief  Create empty (without content, but with operations) SUNLinearSolver of Inmost type @return not NULL on success
SUNLinearSolver      SUNLinSolNewEmpty_Inmost(SUN_CTX(ctx));
/// @brief  Construct SUNLinearSolver of Inmost type that doesn't own by it's data @return not NULL on success
SUNLinearSolver      SUNLinSolInmost(SUN_CTX_COMMA(ctx) INMOST::Solver* data);

/// @brief SUNLinearSolver operation free(...) @return content of SUNLinearSolver @see N_Vector_Ops in sundials_linearsolver.h
int                  SUNLinSolFree_Inmost(SUNLinearSolver S);
/// @brief SUNLinearSolver operation gettype(...) @return SUNLINEARSOLVER_MATRIX_ITERATIVE @see N_Vector_Ops in sundials_linearsolver.h
SUNLinearSolver_Type SUNLinSolGetType_Inmost(SUNLinearSolver S);
/// @brief SUNLinearSolver operation getid(...) @return SUNLINEARSOLVER_CUSTOM @see N_Vector_Ops in sundials_linearsolver.h
SUNLinearSolver_ID   SUNLinSolGetID_Inmost(SUNLinearSolver S);
/// @brief SUNLinearSolver operation initialize(...) @return 0 on success @see N_Vector_Ops in sundials_linearsolver.h
int                  SUNLinSolInitialize_Inmost(SUNLinearSolver S);
/// @brief SUNLinearSolver operation setup(...): create preconditioner for A matrix @return 0 on success @see N_Vector_Ops in sundials_linearsolver.h
int                  SUNLinSolSetup_Inmost(SUNLinearSolver S, SUNMatrix A);
/// @brief SUNLinearSolver operation solve(...)
/// @param[in] S solver
/// @param[in] A,b are matrix and rhs
/// @param[in,out] x is initial guess on input and approximate solution on output
/// @param[in] tol is absolute tolerance for solver
/// @return 0 on success @see N_Vector_Ops in sundials_linearsolver.h
int                  SUNLinSolSolve_Inmost(SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, realtype tol);
/// @brief SUNLinearSolver operation numiters(...). @return Number of linear iterations in the last solution @see N_Vector_Ops in sundials_linearsolver.h
int                  SUNLinSolNumIters_Inmost(SUNLinearSolver LS);
/// @brief SUNLinearSolver operation resnorm(...). @return l2-norm of residual of the last solution @see N_Vector_Ops in sundials_linearsolver.h
realtype             SUNLinSolResNorm_Inmost(SUNLinearSolver LS);
/// @brief SUNLinearSolver operation content(...) @return content of SUNLinearSolver @see N_Vector_Ops in sundials_linearsolver.h
SUNLinearSolverContent_Inmost
                     SUNLinSolContent_Inmost(SUNLinearSolver LS);
#undef SUN_CTX_COMMA
#undef SUN_CTX


///The wrapper of Sundials KinSol solver adapted for work with INMOST::Sparse Matrices, Vectors and INMOST Solvers
class SUNNonlinearSolver {
public:
    using Vector = INMOST::Sparse::Vector;
    using Matrix = INMOST::Sparse::Matrix;
    using Solver = INMOST::Solver;
    using NVContent = N_VectorContent_Inmost;
    using SMContent = SUNMatrixContent_Inmost;
    using LSContent = SUNLinearSolverContent_Inmost;
    using KinSol = void*;
    using ErrorHandler = std::function<void(int error_code, const char *module, const char *function, char *msg)>;
    using InfoHandler = std::function<void(const char *module, const char *function, char *msg)>;
    enum SolveStrategy{
        NONE = KIN_NONE,            ///< use classical inexact Newton method x_{n+1} = x_n - q, q ≈ J^{-1}*F(x_n), J = dF/dx
        LINESEARCH = KIN_LINESEARCH,///< use modified inexact Newton method x_{n+1} = x_n - lambda * q, q ≈ J^{-1}*F(x_n), lambda is computed on alpha-condition and Wolfe’s curvature condition
        PICARD = KIN_PICARD         ///< use Picard method x_{n+1} = x_n - L^{-1}F(x_n), where L is given constant matrix, for example if the function F(u) = L*u - N(u) splited in linaear L*u and N(u) nonlinear part
    };
    /// The choice of the accuracy with which the linear system is solved is made on the basis of inequality
    ///\f[
    ///   ||F + J*q||_{u_scale} < eta*||F||_{u_scale}
    ///\f]
    enum EtaChoice{
        CHOISE1 = KIN_ETACHOICE1,   ///< Eisenstat and Walker choice 1: S. C. Eisenstat and H. F. Walker. Choosing the Forcing Terms in an Inexact Newton Method.
        CHOISE2 = KIN_ETACHOICE2,   ///< Eisenstat and Walker choice 2: S. C. Eisenstat and H. F. Walker. Choosing the Forcing Terms in an Inexact Newton Method.
        CONSTANT = KIN_ETACONSTANT  ///< eta = constant
    };
    enum ParamType{
        REAL,
        INTEGER,
        BOOLEAN,
        ETACHOISEENUM
    };
    struct ParamDiscr{
        std::string name;
        std::string discr;
        std::string default_val;
        ParamType type;
    };

    std::function<int(Vector& x, Matrix &A, Vector &b)> assm;
    std::function<int(Vector& x, Vector &b)> assmRHS;
    std::function<int(Vector& x, Matrix &A)> assmMAT;
    N_Vector x = nullptr;
    SUNMatrix A = nullptr;
    SUNLinearSolver LS = nullptr;
#if SUNDIALS_VERSION_MAJOR >= 6
    sundials::Context m_ctx;    
#endif 
    KinSol kin = nullptr;


    SUNNonlinearSolver(void* comm = nullptr);
#if SUNDIALS_VERSION_MAJOR >= 6
    SUNNonlinearSolver(sundials::Context&& ctx);
#endif 
    SUNNonlinearSolver(SUNNonlinearSolver&&) noexcept = default;
    SUNNonlinearSolver(Solver& s, Matrix& m, Vector& x);
    ~SUNNonlinearSolver();

    SUNNonlinearSolver& Init();
    bool Solve(SolveStrategy s = LINESEARCH);

    SUNNonlinearSolver& SetMatrix(Matrix& A);
    SUNNonlinearSolver& SetMatrix(std::string name);
    SUNNonlinearSolver& SetInitialGuess(Vector& x);
    SUNNonlinearSolver& SetLinearSolver(Solver& s);
    /// @param assm is a function that calculates the jacobian[in,out] J and the right-hand side[in,out] F at a point[in] x
    SUNNonlinearSolver& SetAssembler(std::function<int(Vector& x, Matrix &J, Vector &F)> assm);
    /// @param assm is a function that calculates the right-hand side[in,out] F at a point[in] x
    SUNNonlinearSolver& SetAssemblerRHS(std::function<int(Vector& x, Vector &F)> assm);
    /// @param assm is a function that calculates the jacobian[in,out] J at a point[in] x
    SUNNonlinearSolver& SetAssemblerMAT(std::function<int(Vector& x, Matrix &J)> assm); 
    SUNNonlinearSolver& SetErrHandlerFn(ErrorHandler eh);
    SUNNonlinearSolver& SetInfoHandlerFn(InfoHandler ih);
    SUNNonlinearSolver& SetErrHandlerFile(std::ostream& out);
    SUNNonlinearSolver& SetInfoHandlerFile(std::ostream& out);
    SUNNonlinearSolver& SetVerbosityLevel(int lvl);
    /**@brief Set specific sign constraint on solution
     *
     * If constraints[i] is \n 
	 *   0.0 then no constraint is imposed on sol[i] \n
	 *   1.0 then sol[i] will be constrained to sol[i] ≥ 0.0 \n
	 *  −1.0 then sol[i] will be constrained to sol[i] ≤ 0.0 \n
	 *   2.0 then sol[i] will be constrained to sol[i] > 0.0 \n
	 *  −2.0 then sol[i] will be constrained to sol[i] < 0.0 \n
     **/
    SUNNonlinearSolver& SetConstraints(Vector& constraints);

    /**
     * @brief Set one of values from the table
     * 
     * Name             | Desription                                                          |  Default value          | Type
     * ---------------- | ------------------------------------------------------------------- | ----------------------- | -------------
     * MaxIters         | "Max. number of nonlinear iterations                                | 200                     | INTEGER
     * UseInitSetup     | Does make initial matrix setup?                                     | true                    | BOOLEAN
     * UseResMon        | Does perform residual monitoring?                                   | true                    | BOOLEAN
     * MaxSetupCalls    | Max. iterations without matrix setup                                | 10                      | INTEGER
     * MaxSubSetupCalls | Max. iterations between checks by the residual monitoring algorithm | 5                       | INTEGER
     * EtaForm          | Form of η coefficient                                               | CHOICE1                 | ETACHOISEENUM
     * EtaConstValue    | Constant value of η for constant eta choice                         | 0.1                     | REAL
     * EtaGamma         | Values of γ for ETECHOICE2                                          | 0.9                     | REAL
     * EtaAlpha         | Values of α for ETECHOICE2                                          | 2.0                     | REAL
     * ResMonWmin       | Value of ω_{min}                                                    | 1e-5                    | REAL
     * ResMonWmax       | Value of ω_{max}                                                    | 0.9                     | REAL
     * ResMonConst      | Constant value of ω^∗                                               | 0.9                     | REAL
     * UseMinEps        | Do set lower bound on epsilon?                                      | true                    | BOOLEAN
     * MaxNewtonStep    | Max. scaled length of Newton step                                   | 1000\f$||D_u*u_0||_2\f$ | REAL
     * MaxBetaFails     | Max. number nonlinear its with β-condition failures                 | 10                      | INTEGER
     * FuncNormTol      | Function-norm stopping tolerance                                    | (DBL_EPSILON)^(1/3)     | REAL
     * ScaledSteptol    | Scaled-step stopping tolerance                                      | (DBL_EPSILON)^(2/3)     | REAL
     * MAA              | Anderson Acceleration subspace size                                 | 0                       | INTEGER
     * DampingAA        | Anderson Acceleration damping parameter beta (0 < beta ≤ 1.0)       | 1                       | REAL
     * 
     * @see GetAvaliableParametrs() and Sundials KinSol documentation
     */
    bool SetParameter(std::string param, std::string val);
    bool SetParameterReal(std::string param, double val); ///< @see GetAvaliableParametrs()
    bool SetParameterIntegral(std::string param, int val); ///< @see GetAvaliableParametrs()
 
    /// @return list of avaliable parameters for this solver
    /// @warning MaxSetupCalls % MaxSubSetupCalls must be equal 0
    /// @warning MAA should be setted before call Init() and should be less than MaxIters
    static std::vector<ParamDiscr> GetAvaliableParametrs();
    std::string GetReason();
    int GetReasonFlag() const { return reason_flag; } ///< on success return >= 0
    long GetNumFuncEvals(){      long res =  0; KINGetNumFuncEvals(kin, &res);       return res; }
    long GetNumNolinSolvIters(){ long res =  0; KINGetNumNonlinSolvIters(kin, &res); return res; }
    long GetNumBetaCondFails(){  long res =  0; KINGetNumBetaCondFails(kin, &res);   return res; }
    long GetNumBacktrackOps(){   long res =  0; KINGetNumBacktrackOps(kin, &res);    return res; }
    double GetResidualNorm(){  double res = -1; KINGetFuncNorm(kin, &res);           return res; }
    double GetStepLength(){    double res = -1; KINGetStepLength(kin, &res);         return res; }
    long GetNumJacEvals(){       long res =  0; KINGetNumJacEvals(kin, &res);        return res; }
    long GetNumLinFuncEvals(){   long res =  0; KINGetNumLinFuncEvals(kin, &res);    return res; }
    long GetNumLinIters(){       long res =  0; KINGetNumLinIters(kin, &res);        return res; }
    long GetNumLinConvFails(){   long res =  0; KINGetNumLinConvFails(kin, &res);    return res; }
    long GetLastLinFlag(){       long res =  0; KINGetLastLinFlag(kin, &res);        return res; }
    std::string GetLinReturnFlagName(){ return KINGetLinReturnFlagName(GetLastLinFlag()); }
    Solver* GetLinearSolver(){ auto c = SUNLinSolContent_Inmost(LS); return c ? c->data : nullptr; }
    Matrix* GetMatrix(){       auto c = SUNMatContent_Inmost(A)    ; return c ? c->data : nullptr; }
    Vector* GetVectorX(){      auto c = N_VContent_Inmost(x)       ; return c ? c->data : nullptr; }
    LSContent GetLinearSolverContent() { return SUNLinSolContent_Inmost(LS); }
    SMContent GetMatrixContent() { return SUNMatContent_Inmost(A); }
    NVContent GetVectorXContent() { return N_VContent_Inmost(x); };
    double GetMatAssembleTime() const;
    double GetRHSAssembleTime() const;

private:
    static int assmRHS_interface(N_Vector x, N_Vector b, void *this_obj);
    static int assmMAT_interface(N_Vector x, N_Vector b, SUNMatrix J, void *this_obj, N_Vector tmp1, N_Vector tmp2);
    static void errorHandler(int error_code, const char *module, const char *function, char *msg, void *user_data);
    static void infoHandler(const char *module, const char *function, char *msg, void *user_data);

    InfoHandler ihand;
    ErrorHandler ehand;
    N_Vector u_scale = nullptr;
    double _lmat_assm_time = 0, _lrhs_assm_time = 0;
    double _gmat_assm_time = 0, _grhs_assm_time = 0;

    int reason_flag = KIN_MEM_NULL;
    double _egamma = 0.9, _ealpha = 2.0;
    double _wmin = 1e-5, _wmax = 0.9;
};

std::ostream& operator<<(std::ostream& out, SUNNonlinearSolver::ParamType t);

#endif //CARNUM_SOLVER_SUNNONLINEARSOLVER_H
