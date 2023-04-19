//
// Created by Liogky Alexey on 23.10.2022.
//

#include "SUNNonlinearSolver.h"

#define NV_CONTENT_P(v)    ( static_cast<N_VectorContent_Inmost>(v->content) )
#define NV_OWN_DATA_P(v)   ( NV_CONTENT_P(v)->own_data )
#define NV_DATA_P(v)       ( NV_CONTENT_P(v)->data )

#define SM_CONTENT_P(v)    ( static_cast<SUNMatrixContent_Inmost>(v->content) )
#define SM_OWN_DATA_P(v)   ( SM_CONTENT_P(v)->own_data )
#define SM_DATA_P(v)       ( SM_CONTENT_P(v)->data )

#define LS_CONTENT_P(v)    ( static_cast<SUNLinearSolverContent_Inmost>(v->content) )
#define LS_OWN_DATA_P(v)   ( LS_CONTENT_P(v)->own_data )
#define LS_DATA_P(v)       ( LS_CONTENT_P(v)->data )

int N_VInitOps_Inmost(N_Vector v) {
    if (v == NULL) return -1;

    /* Attach operations */

    /* constructors, destructors, and utility operations */
    v->ops->nvgetvectorid     = N_VGetVectorID_Inmost;
    v->ops->nvclone           = N_VClone_Inmost;
    v->ops->nvcloneempty      = N_VCloneEmpty_Inmost;
    v->ops->nvdestroy         = N_VDestroy_Inmost;
    v->ops->nvspace           = N_VSpace_Inmost;
    v->ops->nvgetarraypointer = NULL;
    v->ops->nvsetarraypointer = NULL;
    v->ops->nvgetcommunicator = N_VGetCommunicator_Inmost;
    v->ops->nvgetlength       = N_VGetLength_Inmost;

    /* standard vector operations */
    v->ops->nvlinearsum    = N_VLinearSum_Inmost;
    v->ops->nvconst        = N_VConst_Inmost;
    v->ops->nvprod         = N_VProd_Inmost;
    v->ops->nvdiv          = N_VDiv_Inmost;
    v->ops->nvscale        = N_VScale_Inmost;
    v->ops->nvabs          = N_VAbs_Inmost;
    v->ops->nvinv          = N_VInv_Inmost;
    v->ops->nvaddconst     = N_VAddConst_Inmost;
    v->ops->nvdotprod      = N_VDotProd_Inmost;
    v->ops->nvmaxnorm      = N_VMaxNorm_Inmost;
    v->ops->nvwrmsnormmask = N_VWrmsNormMask_Inmost;
    v->ops->nvwrmsnorm     = N_VWrmsNorm_Inmost;
    v->ops->nvmin          = N_VMin_Inmost;
    v->ops->nvwl2norm      = N_VWL2Norm_Inmost;
    v->ops->nvl1norm       = N_VL1Norm_Inmost;
    v->ops->nvcompare      = N_VCompare_Inmost;
    v->ops->nvinvtest      = N_VInvTest_Inmost;
    v->ops->nvconstrmask   = N_VConstrMask_Inmost;
    v->ops->nvminquotient  = N_VMinQuotient_Inmost;

    v->ops->nvlinearcombination = N_VLinearCombination_Inmost;
    v->ops->nvscaleaddmulti = NULL;
    v->ops->nvdotprodmulti = N_VDotProdMulti_Inmost;
    /* vector array operations are disabled (NULL) by default */

    /* local reduction operations */
    v->ops->nvdotprodlocal     = N_VDotProdLocal_Inmost;
    v->ops->nvmaxnormlocal     = N_VMaxNormLocal_Inmost;
    v->ops->nvminlocal         = N_VMinLocal_Inmost;
    v->ops->nvl1normlocal      = N_VL1NormLocal_Inmost;
    v->ops->nvinvtestlocal     = N_VInvTestLocal_Inmost;
    v->ops->nvconstrmasklocal  = N_VConstrMaskLocal_Inmost;
    v->ops->nvminquotientlocal = N_VMinQuotientLocal_Inmost;
    v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_Inmost;
    v->ops->nvwsqrsummasklocal = N_VWSqrSumMaskLocal_Inmost;

    /* OPTIONAL XBraid interface operations are disabled (NULL) by default*/

    /* debugging functions */
    v->ops->nvprint     = NULL;
    v->ops->nvprintfile = NULL;

    return 0;
}

#if SUNDIALS_VERSION_MAJOR >= 6
    #define SUN_CTX(X) X
    #define SUN_CTX_COMMA(X) SUN_CTX(X), 
    #define SUN_TYPED_CTX(X) SUNContext X
    #define SUN_TYPED_CTX_COMMA(X) SUN_TYPED_CTX(X), 
#else
    #define SUN_CTX(X)
    #define SUN_CTX_COMMA(X)
    #define SUN_TYPED_CTX(X)
    #define SUN_TYPED_CTX_COMMA(X)       
#endif

N_Vector N_VNewEmpty_Inmost(SUN_TYPED_CTX(ctx)) {
    N_Vector v;
    N_VectorContent_Inmost content;

    v = NULL;
    v = N_VNewEmpty(SUN_CTX(ctx));
    if (v == NULL) return(NULL);

    N_VInitOps_Inmost(v);

    /* Create content */
    content = NULL;
    content = (N_VectorContent_Inmost) malloc(sizeof *content);
    if (content == NULL) { N_VDestroy(v); return(NULL); }
    /* Attach content */
    v->content = content;
    content->own_data = false;
    content->data = NULL;

    return v;
}

N_Vector N_VNew_Inmost(SUN_TYPED_CTX_COMMA(ctx) INMOST::Sparse::Vector *data) {
    N_Vector v = N_VNewEmpty_Inmost(SUN_CTX(ctx));
    auto content = static_cast<N_VectorContent_Inmost>(v->content);

    content->data = data;
    content->own_data = false;

    return v;
}

N_Vector N_VNew_Inmost(SUN_TYPED_CTX_COMMA(ctx) std::string name) {
    N_Vector v = N_VNewEmpty_Inmost(SUN_CTX(ctx));
    auto content = static_cast<N_VectorContent_Inmost>(v->content);

    content->data = new INMOST::Sparse::Vector(name);
    content->own_data = true;

    return v;
}

/*
 * -----------------------------------------------------------------
 * implementation of vector operations
 * -----------------------------------------------------------------
 */

N_Vector_ID N_VGetVectorID_Inmost(N_Vector v) {
    (void) v; return SUNDIALS_NVEC_CUSTOM;
}
sunindextype N_VGetLength_Inmost(N_Vector v){
    return NV_DATA_P(v)->Size();
}

void N_VDestroy_Inmost(N_Vector v) {
    if (v == NULL) return;
    if (v->content != NULL){
        auto content = static_cast<N_VectorContent_Inmost>(v->content);
        if (content->data != NULL){
            if (content->own_data){
                delete content->data;
                content->own_data = false;
            }
            content->data = NULL;
        } else content->own_data = false;
        free(v->content); v->content = NULL;
    }
    N_VFreeEmpty(v);
}

N_Vector N_VCloneEmpty_Inmost(N_Vector w) {
    N_Vector v;
    N_VectorContent_Inmost content;

    if (w == NULL) return(NULL);

    /* Create vector */
    v = NULL;
    v = N_VNewEmpty(SUN_CTX(w->sunctx));
    if (v == NULL) return(NULL);

    /* Attach operations */
    if (N_VCopyOps(w, v)) { N_VDestroy(v); return(NULL); }

    /* Create content */
    content = NULL;
    content = (N_VectorContent_Inmost) malloc(sizeof *content);
    if (content == NULL) { N_VDestroy(v); return(NULL); }

    /* Attach content */
    v->content = content;

    /* Initialize content */
    content->own_data      = false;
    content->data          = NULL;

    return(v);
}

N_Vector N_VClone_Inmost(N_Vector w) {
    N_Vector v = N_VCloneEmpty_Inmost(w); if (v == NULL) return(NULL);
    auto wdat = NV_DATA_P(w);
    auto content = NV_CONTENT_P(v);
    static int id = 0;
    if (wdat){
        std::string wname = wdat->GetName();
        std::regex re("_cln[0-9]+");
        std::smatch m;
        std::string name;
        if (std::regex_search(wname, m, re)){
            auto it1 = m[0].first;
            it1 += 4; //strlen("_cln");
            // auto snum = wname.substr(it1 - wname.cbegin(), m[0].second - wname.cbegin());
            // unsigned num = atoi(snum.c_str());
            name = m.prefix().str() + "_cln" + std::to_string(++id) + m.suffix().str();
        } else {
            name = wname + "_cln" + std::to_string(++id);
        }
        content->data = new INMOST::Sparse::Vector(name, wdat->GetFirstIndex(), wdat->GetLastIndex(), wdat->GetCommunicator());
        assert(!content->data->Empty());
        content->own_data = true;
    }
    return v;
}

void N_VSpace_Inmost(N_Vector v, sunindextype *lrw, sunindextype *liw) {
    int npes = 1;
#ifdef USE_MPI
    INMOST_MPI_Comm comm = NV_DATA_P(v)->GetCommunicator();
    MPI_Comm_size(comm, &npes);
#endif

    *lrw = NV_DATA_P(v)->Size();
    *liw = (2 /*in interval*/ + 1 /*in name*/)*npes;

    return;
}

template<typename T, typename std::enable_if<!std::is_pointer<T>::value, char>::type* = nullptr>
static void* choose_void_internals(T val){
    (void) val;
    return NULL;
}

template<typename T, typename std::enable_if<std::is_pointer<T>::value, char>::type* = nullptr>
static void* choose_void_internals(T val){
    return val;
}

template<typename T = INMOST_MPI_Comm, typename std::enable_if<std::is_integral<T>::value, void>::type* = nullptr>
static void* _N_VGetCommunicator_Inmost_internals(N_Vector v){
#ifdef USE_MPI
    static INMOST_MPI_Comm comm;
    comm = NV_DATA_P(v)->GetCommunicator();
    return &comm;
#else
    (void) v;
    return NULL;
#endif
}
template<typename T = INMOST_MPI_Comm, typename std::enable_if<std::is_pointer<T>::value, char>::type* = nullptr>
static void*  _N_VGetCommunicator_Inmost_internals(N_Vector v){
#ifdef USE_MPI
    return NV_DATA_P(v)->GetCommunicator();
#else
    (void) v;
    return NULL;
#endif
}
void* N_VGetCommunicator_Inmost(N_Vector v){
    return _N_VGetCommunicator_Inmost_internals<INMOST_MPI_Comm>(v);
}

void N_VLinearSum_Inmost(realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z) {
    auto xv = NV_DATA_P(x), yv = NV_DATA_P(y), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), ity = yv->Begin(), itz = zv->Begin();
         itx != xv->End() && ity != yv->End() && itz != zv->End(); ++itx, ++ity, ++itz)
        *itz = a * (*itx) + b * (*ity);
}

void N_VConst_Inmost(realtype c, N_Vector z) {
    auto zv = NV_DATA_P(z);
    for (auto it = zv->Begin(); it != zv->End(); ++it)
        *it = c;
}

void N_VProd_Inmost(N_Vector x, N_Vector y, N_Vector z) {
    auto xv = NV_DATA_P(x), yv = NV_DATA_P(y), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), ity = yv->Begin(), itz = zv->Begin();
         itx != xv->End() && ity != yv->End() && itz != zv->End(); ++itx, ++ity, ++itz)
        *itz = (*itx) * (*ity);
}

void N_VDiv_Inmost(N_Vector x, N_Vector y, N_Vector z) {
    auto xv = NV_DATA_P(x), yv = NV_DATA_P(y), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), ity = yv->Begin(), itz = zv->Begin();
         itx != xv->End() && ity != yv->End() && itz != zv->End(); ++itx, ++ity, ++itz)
        *itz = (*itx) / (*ity);
}

void N_VScale_Inmost(realtype c, N_Vector x, N_Vector z) {
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    if (z == x) {
        for (auto it = xv->Begin(); it != xv->End(); ++it) *it *= c;
        return;
    }
    if (c == RCONST(1)){
        for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz) *itz = *itx;
        return;
    }
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        *itz = c * (*itx);
}

void N_VAbs_Inmost(N_Vector x, N_Vector z) {
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        *itz = std::abs(*itx);
}

void N_VInv_Inmost(N_Vector x, N_Vector z) {
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        *itz = RCONST(1)/(*itx);
}

void N_VAddConst_Inmost(N_Vector x, realtype b, N_Vector z) {
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        *itz = (*itx) + b;
}

realtype N_VDotProdLocal_Inmost(N_Vector x, N_Vector y) {
    realtype ldot = 0;
    auto xv = NV_DATA_P(x), yv = NV_DATA_P(y);
    for (auto itx = xv->Begin(), ity = yv->Begin(); itx != xv->End() && ity != yv->End(); ++itx, ++ity)
        ldot += (*itx) * (*ity);
    return ldot;
}

realtype N_VDotProd_Inmost(N_Vector x, N_Vector y) {
    realtype ldot = N_VDotProdLocal_Inmost(x, y), gdot = 0;
#ifdef USE_MPI
    MPI_Allreduce(&ldot, &gdot, 1, MPI_DOUBLE, MPI_SUM, NV_DATA_P(x)->GetCommunicator());
#else
    gdot = ldot;
#endif
    return gdot;
}

realtype N_VMaxNormLocal_Inmost(N_Vector x) {
    realtype lmax = 0;
    auto xv = NV_DATA_P(x);
    for (auto itx = xv->Begin(); itx != xv->End(); ++itx)
        lmax = std::max(std::abs(*itx), lmax);
    return lmax;
}

realtype N_VMaxNorm_Inmost(N_Vector x) {
    realtype lmax = N_VMaxNormLocal_Inmost(x), gmax = 0;
#ifdef USE_MPI
    MPI_Allreduce(&lmax, &gmax, 1, MPI_DOUBLE, MPI_MAX, NV_DATA_P(x)->GetCommunicator());
#else
    gmax = lmax;
#endif    
    return gmax;
}

realtype N_VWSqrSumLocal_Inmost(N_Vector x, N_Vector w) {
    realtype lsum = 0;
    auto xv = NV_DATA_P(x), wv = NV_DATA_P(w);
    for (auto itx = xv->Begin(), itw = wv->Begin(); itx != xv->End() && itw != wv->End(); ++itx, ++itw) {
        auto prod = (*itx) * (*itw);
        lsum += prod * prod;
    }
    return lsum;
}

realtype N_VWrmsNorm_Inmost(N_Vector x, N_Vector w) {
    realtype lsum = N_VWSqrSumLocal_Inmost(x, w), gsum = 0;
    auto xv = NV_DATA_P(x);
#ifdef USE_MPI
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, xv->GetCommunicator());
#else
    gsum = lsum;
#endif    
    return std::sqrt(gsum / xv->Size());
}

realtype N_VWSqrSumMaskLocal_Inmost(N_Vector x, N_Vector w, N_Vector id) {
    realtype lsum = 0;
    auto xv = NV_DATA_P(x), wv = NV_DATA_P(w), iv = NV_DATA_P(id);
    for (auto itx = xv->Begin(), itw = wv->Begin(), itd = iv->Begin(); itx != xv->End() && itw != wv->End() && itd != iv->End(); ++itx, ++itw, ++itd) {
        if (*itd > RCONST(0)) {
            auto prod = (*itx) * (*itw);
            lsum += prod * prod;
        }
    }
    return lsum;
}

realtype N_VWrmsNormMask_Inmost(N_Vector x, N_Vector w, N_Vector id) {
    realtype lsum = N_VWSqrSumMaskLocal_Inmost(x, w, id), gsum = 0;
    auto xv = NV_DATA_P(x);
#ifdef USE_MPI    
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, xv->GetCommunicator());
#else
    gsum = lsum;
#endif    
    return std::sqrt(gsum / xv->Size());
}

realtype N_VMinLocal_Inmost(N_Vector x) {
    realtype lmin = BIG_REAL;
    auto xv = NV_DATA_P(x);
    for (auto itx = xv->Begin(); itx != xv->End(); ++itx)
        lmin = std::min((*itx), lmin);

    return lmin;
}

realtype N_VMin_Inmost(N_Vector x) {
    realtype lmin = N_VMinLocal_Inmost(x), gmin = BIG_REAL;
#ifdef USE_MPI
    MPI_Allreduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, NV_DATA_P(x)->GetCommunicator());
#else
    gmin = lmin;
#endif    
    return gmin;
}

realtype N_VWL2Norm_Inmost(N_Vector x, N_Vector w) {
    realtype lsum = 0, gsum = 0;
    auto xv = NV_DATA_P(x), wv = NV_DATA_P(w);
    for (auto itx = xv->Begin(), itw = wv->Begin(); itx != xv->End() && itw != wv->End(); ++itx, ++itw) {
        auto prod = (*itx) * (*itw);
        lsum += prod * prod;
    }
#ifdef USE_MPI
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, xv->GetCommunicator());
#else
    gsum = lsum;
#endif    
    return std::sqrt(gsum);
}

realtype N_VL1NormLocal_Inmost(N_Vector x) {
    realtype lsum = 0;
    auto xv = NV_DATA_P(x);
    for (auto itx = xv->Begin(); itx != xv->End(); ++itx) lsum += std::abs(*itx);

    return lsum;
}

realtype N_VL1Norm_Inmost(N_Vector x) {
    realtype lsum = N_VL1NormLocal_Inmost(x), gsum = 0;
#ifdef USE_MPI    
    MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, NV_DATA_P(x)->GetCommunicator());
#else
    gsum = lsum;
#endif    
    return gsum;
}

void N_VCompare_Inmost(realtype c, N_Vector x, N_Vector z) {
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        *itz = (std::abs(*itx) >= c) ? RCONST(1) : RCONST(0);
}

booleantype N_VInvTestLocal_Inmost(N_Vector x, N_Vector z) {
    char lval = 1;
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        if (*itx == RCONST(0)) lval = 0;
        else *itz = RCONST(1) / (*itx);
    return (lval == 0) ? SUNFALSE : SUNTRUE;
}

booleantype N_VInvTest_Inmost(N_Vector x, N_Vector z) {
    char lval = 1, gval = 1;
    auto xv = NV_DATA_P(x), zv = NV_DATA_P(z);
    for (auto itx = xv->Begin(), itz = zv->Begin(); itx != xv->End() && itz != zv->End(); ++itx, ++itz)
        if (*itx == RCONST(0)) lval = 0;
        else *itz = RCONST(1) / (*itx);
#ifdef USE_MPI        
    MPI_Allreduce(&lval, &gval, 1, MPI_CHAR, MPI_MIN, xv->GetCommunicator());
#else
    gval = lval;
#endif    
    return (gval == 0) ? SUNFALSE : SUNTRUE;
}

booleantype N_VConstrMaskLocal_Inmost(N_Vector c, N_Vector x, N_Vector m) {
    char temp = 0;
    auto cv = NV_DATA_P(c), xv = NV_DATA_P(x), mv = NV_DATA_P(m);
    for (auto itx = xv->Begin(), itc = cv->Begin(), itm = mv->Begin(); itx != xv->End() && itc != cv->End() && itm != mv->End(); ++itx, ++itc, ++itm) {
        *itm = RCONST(0.0);

        /* Continue if no constraints were set for the variable */
        if (*itc == RCONST(0.0))
            continue;

        /* Check if a set constraint has been violated */
        bool test = (abs(*itc) > RCONST(1.5) && (*itx)*(*itc) <= RCONST(0.0)) ||
                    (abs(*itc) > RCONST(0.5)   && (*itx)*(*itc) < RCONST(0.0));
        if (test) {
            *itm = RCONST(1);
            temp = 1;
        }
    }
    return (temp == 1) ? SUNFALSE : SUNTRUE;
}

booleantype N_VConstrMask_Inmost(N_Vector c, N_Vector x, N_Vector m) {
    char temp = 0, temp2 = 0;
    auto res = N_VConstrMaskLocal_Inmost(c, x, m);
    if (res == SUNFALSE) temp = 1;
#ifdef USE_MPI    
    MPI_Allreduce(&temp, &temp2, 1, MPI_CHAR, MPI_MAX, NV_DATA_P(x)->GetCommunicator());
#else
    temp2 = temp;
#endif    
    return (temp2 == 1) ? SUNFALSE : SUNTRUE;
}

realtype N_VMinQuotientLocal_Inmost(N_Vector num, N_Vector denom) {
    realtype lmin = BIG_REAL;
    auto xv = NV_DATA_P(num), yv = NV_DATA_P(denom);
    for (auto itx = xv->Begin(), ity = yv->Begin();
         itx != xv->End() && ity != yv->End(); ++itx, ++ity) {
        if ((*ity) == RCONST(0)) continue;
        lmin = std::min(lmin, (*itx) / (*ity));
    }
    return lmin;
}

realtype N_VMinQuotient_Inmost(N_Vector num, N_Vector denom) {
    realtype lmin = N_VMinQuotientLocal_Inmost(num ,denom), gmin = BIG_REAL;
#ifdef USE_MPI    
    MPI_Allreduce(&lmin, &gmin, 1, MPI_DOUBLE, MPI_MIN, NV_DATA_P(num)->GetCommunicator());
#else
    gmin = lmin;
#endif
    return gmin;
}

/*
 * -----------------------------------------------------------------
 * fused vector operations
 * -----------------------------------------------------------------
 */

int N_VLinearCombination_Inmost(int nvec, realtype *c, N_Vector *X, N_Vector z) {
    if (nvec < 1) return -1;
    if (nvec == 1) {
        N_VScale_Inmost(c[0], X[0], z);
        return 0;
    }
    if (nvec == 2){
        N_VLinearSum_Inmost(c[0], X[0], c[1], X[1], z);
        return 0;
    }
    auto zd = NV_DATA_P(z), xd = NV_DATA_P(X[0]);
    unsigned int beg, to;
    xd->GetInterval(beg, to);
    if (z == X[0]){
        for (auto it =xd->Begin(); it != xd->End(); ++it) (*it) *= c[0];
    } else
        for (auto itx = xd->Begin(), itz = zd->Begin(); itx != xd->End() && itz != zd->End(); ++itx, ++itz) (*itz) = c[0] * (*itx);
    for (int i = 1; i < nvec; ++i) {
        xd = NV_DATA_P(X[i]);
        for (auto j = beg; j < to; ++j) {
            zd->operator[](j) += c[i] * xd->operator[](j);
        }
    }
    return 0;
}

int N_VDotProdMulti_Inmost(int nvec, N_Vector x, N_Vector *Y, realtype *dotprods) {
    /* invalid number of vectors */
    if (nvec < 1) return(-1);

    /* should have called N_VDotProd */
    if (nvec == 1) {
        dotprods[0] = N_VDotProd_Inmost(x, Y[0]);
        return(0);
    }

    auto xd   = NV_DATA_P(x), yd = NV_DATA_P(Y[0]);
    unsigned int beg, to;
    xd->GetInterval(beg, to);

    /* compute multiple dot products */
    for (int i = 0; i < nvec; i++) {
        yd = NV_DATA_P(Y[i]);
        dotprods[i] = RCONST(0);
        for (auto j = beg; j < to; j++) {
            dotprods[i] += xd->operator[](j) * yd->operator[](j);
        }
    }
#ifdef USE_MPI
    auto retval = MPI_Allreduce(MPI_IN_PLACE, dotprods, nvec, MPI_DOUBLE, MPI_SUM, xd->GetCommunicator());
    return retval == MPI_SUCCESS ? 0 : -1;
#else
    return 0;
#endif    
}

N_VectorContent_Inmost N_VContent_Inmost(N_Vector x) {
    return x ? NV_CONTENT_P(x) : NULL;
}

int  SUNMatInitOps_Inmost(SUNMatrix A){
    if (A == NULL) return SUNMAT_MEM_FAIL;

    /* Attach operations */
    A->ops->getid     = SUNMatGetID_Inmost;
    A->ops->clone     = SUNMatClone_Inmost;
    A->ops->destroy   = SUNMatDestroy_Inmost;
    A->ops->zero      = SUNMatZero_Inmost;
    A->ops->copy      = SUNMatCopy_Inmost;
    A->ops->scaleadd  = NULL;
    A->ops->scaleaddi = NULL;
    A->ops->matvec    = SUNMatMatvec_Inmost;
    A->ops->space     = SUNMatSpace_Inmost;

    return SUNMAT_SUCCESS;
}

SUNMatrix SUNMatNewEmpty_Inmost(SUN_TYPED_CTX(ctx)) {
    SUNMatrix m;
    SUNMatrixContent_Inmost content;

    m = NULL;
    m = SUNMatNewEmpty(SUN_CTX(ctx));
    if (m == NULL) return(NULL);

    SUNMatInitOps_Inmost(m);

    /* Create content */
    content = NULL;
    content = (SUNMatrixContent_Inmost) malloc(sizeof *content);
    if (content == NULL) { SUNMatFreeEmpty(m); return(NULL); }
    /* Attach content */
    m->content = content;
    content->own_data = false;
    content->data = NULL;
    content->overlap = 0;

    return m;
}

SUNMatrix SUNMatInmost(SUN_TYPED_CTX_COMMA(ctx) INMOST::Sparse::Matrix *data, int overlap) {
    SUNMatrix v = SUNMatNewEmpty_Inmost(SUN_CTX(ctx));
    auto content = static_cast<SUNMatrixContent_Inmost>(v->content);

    content->data = data;
    content->own_data = false;
    content->overlap = overlap;

    return v;
}

SUNMatrix SUNMatInmost(SUN_TYPED_CTX_COMMA(ctx) std::string name) {
    SUNMatrix v = SUNMatNewEmpty_Inmost(SUN_CTX(ctx));
    auto content = static_cast<SUNMatrixContent_Inmost>(v->content);

    content->data = new INMOST::Sparse::Matrix(name);
    content->own_data = true;
    content->overlap = 0;

    return v;
}

SUNMatrix_ID SUNMatGetID_Inmost(SUNMatrix A) {
    (void) A; return SUNMATRIX_CUSTOM;
}

void SUNMatDestroy_Inmost(SUNMatrix m) {
    if (m == NULL) return;
    if (m->content != NULL){
        auto content = static_cast<SUNMatrixContent_Inmost>(m->content);
        if (content->data != NULL){
            if (content->own_data){
                delete content->data;
                content->own_data = false;
            }
            content->data = NULL;
        } else content->own_data = false;
        free(m->content); m->content = NULL;
    }
    SUNMatFreeEmpty(m);
}

SUNMatrix SUNMatCloneEmpty_Inmost(SUNMatrix A) {
    SUNMatrix m;
    SUNMatrixContent_Inmost content;

    if (A == NULL) return(NULL);

    /* Create matrix */
    m = NULL;
    m = SUNMatNewEmpty(SUN_CTX(A->sunctx));
    if (m == NULL) return(NULL);

    /* Attach operations */
    if (SUNMatCopyOps(A, m)) { SUNMatFreeEmpty(m); return(NULL); }

    /* Create content */
    content = NULL;
    content = (SUNMatrixContent_Inmost) malloc(sizeof *content);
    if (content == NULL) { SUNMatFreeEmpty(m); return(NULL); }

    /* Attach content */
    m->content = content;

    /* Initialize content */
    content->own_data      = false;
    content->data          = NULL;
    content->overlap       = 0;

    return(m);
}

SUNMatrix SUNMatClone_Inmost(SUNMatrix A) {
    SUNMatrix v = SUNMatCloneEmpty_Inmost(A); if (v == NULL) return(NULL);
    auto wdat = SM_DATA_P(A);
    auto content = SM_CONTENT_P(v);
    static int id = 0;
    if (wdat){
        std::string wname = wdat->GetName();
        std::regex re("_cln[0-9]+");
        std::smatch m;
        std::string name;
        if (std::regex_search(wname, m, re)){
            auto it1 = m[0].first;
            it1 += 4; //strlen("_cln");
            // auto snum = wname.substr(it1 - wname.cbegin(), m[0].second - wname.cbegin());
            // unsigned num = atoi(snum.c_str());
            name = m.prefix().str() + "_cln" + std::to_string(++id) + m.suffix().str();
        } else {
            name = wname + "_cln" + std::to_string(++id);
        }
        content->data = new INMOST::Sparse::Matrix(name, wdat->GetFirstIndex(), wdat->GetLastIndex(), wdat->GetCommunicator());
        content->own_data = true;
        content->overlap = SM_CONTENT_P(A)->overlap;
    }
    return v;
}

int SUNMatSpace_Inmost(SUNMatrix A, long *lenrw, long *leniw) {
    auto ad = SM_DATA_P(A);
    int nEntry = ad->Nonzeros(), gnEntry = 0;
    int npes = 1;
#ifdef USE_MPI
    MPI_Comm comm = SM_DATA_P(A)->GetCommunicator();
    MPI_Comm_size(comm, &npes);
    MPI_Allreduce(&nEntry, &gnEntry, 1, MPI_INT, MPI_SUM, ad->GetCommunicator());
#else
    gnEntry = nEntry;
#endif        
    *lenrw = gnEntry;
    *leniw = (2 /*in interval*/ + 1 /*in name*/)*npes + gnEntry;

    return SUNMAT_SUCCESS;
}

int SUNMatZero_Inmost(SUNMatrix A) {
    auto ad = SM_DATA_P(A);
    for(auto it = ad->Begin(); it != ad->End(); ++it) it->Clear();
    return SUNMAT_SUCCESS;
}

int SUNMatCopy_Inmost(SUNMatrix A, SUNMatrix B) {
    auto ad = SM_DATA_P(A), bd = SM_DATA_P(B);
    *bd = *ad;
    return SUNMAT_SUCCESS;
}

int SUNMatMatvec_Inmost(SUNMatrix A, N_Vector x, N_Vector y) {
    INMOST::Sparse::Matrix& m = *SM_DATA_P(A);
    INMOST::Sparse::Vector& xv = *NV_DATA_P(x), yv = *NV_DATA_P(y);
#if defined(USE_MPI)
    INMOST::Solver::OrderInfo info;
    info.PrepareMatrix(m, SM_CONTENT_P(A)->overlap);
    info.PrepareVector(xv); info.Update(xv);
#endif
    m.MatVec(1, xv, 0, yv);
#if defined(USE_MPI)
    info.RestoreVector(xv);
    info.RestoreMatrix(m);
#endif
    return SUNMAT_SUCCESS;
}

SUNMatrixContent_Inmost SUNMatContent_Inmost(SUNMatrix A) {
    return A ? SM_CONTENT_P(A) : NULL;
}

int SUNLinSolInitOps_Inmost(SUNLinearSolver LS) {
    if (LS == NULL) return SUNLS_MEM_NULL;

    // Attach operations
    LS->ops->gettype    = SUNLinSolGetType_Inmost;
    LS->ops->getid      = SUNLinSolGetID_Inmost;
    LS->ops->initialize = SUNLinSolInitialize_Inmost;
    LS->ops->setup      = SUNLinSolSetup_Inmost;
    LS->ops->solve      = SUNLinSolSolve_Inmost;
    LS->ops->numiters   = SUNLinSolNumIters_Inmost;
    LS->ops->resnorm    = SUNLinSolResNorm_Inmost;
    LS->ops->free       = SUNLinSolFree_Inmost;

    return SUNLS_SUCCESS;
}

SUNLinearSolver SUNLinSolNewEmpty_Inmost(SUN_TYPED_CTX(ctx)) {
    SUNLinearSolver S;
    SUNLinearSolverContent_Inmost content;

    if (!(S = SUNLinSolNewEmpty(SUN_CTX(ctx)))) return NULL;

    SUNLinSolInitOps_Inmost(S);

    content = (SUNLinearSolverContent_Inmost) malloc(sizeof *content);
    if (content == NULL) { SUNLinSolFreeEmpty(S); return(NULL); }
    /* Attach content */
    S->content = content;
    content->own_data = false;
    content->data = NULL;
    content->updated_mat = 0;
    content->verbosity = 0;

    return S;
}

SUNLinearSolver SUNLinSolInmost(SUN_TYPED_CTX_COMMA(ctx) INMOST::Solver *data) {
    SUNLinearSolver S = SUNLinSolNewEmpty_Inmost(SUN_CTX(ctx));
    auto content = LS_CONTENT_P(S);

    content->data = data;
    content->own_data = false;

    return S;
}

int SUNLinSolFree_Inmost(SUNLinearSolver S) {
    if (S == NULL) return SUNLS_SUCCESS;
    if (S->content != NULL){
        auto content =  LS_CONTENT_P(S);
        if (content->data != NULL){
            if (content->own_data){
                delete content->data;
                content->own_data = false;
            }
            content->data = NULL;
        } else content->own_data = false;
        free(S->content); S->content = NULL;
    }
    SUNLinSolFreeEmpty(S);
    return SUNLS_SUCCESS;
}

SUNLinearSolver_Type SUNLinSolGetType_Inmost(SUNLinearSolver S){ (void)S; return SUNLINEARSOLVER_MATRIX_ITERATIVE; }
SUNLinearSolver_ID   SUNLinSolGetID_Inmost(SUNLinearSolver S) { (void)S; return SUNLINEARSOLVER_CUSTOM; }

int SUNLinSolInitialize_Inmost(SUNLinearSolver S) {
    /* ensure valid options */
    if (S == NULL) return (SUNLS_MEM_NULL);
    auto cont =  LS_CONTENT_P(S);
    if (cont == NULL || cont->data == NULL) return (SUNLS_MEM_NULL);

    return SUNLS_SUCCESS;
}

int SUNLinSolSetup_Inmost(SUNLinearSolver S, SUNMatrix A) {
    auto Sd = LS_DATA_P(S);
    auto Ad = SM_DATA_P(A);
    int pRank = 0;
#if defined(USE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
#endif
    auto verb = LS_CONTENT_P(S)->verbosity;
    if (verb >= 3 || ((verb == 2) && (pRank== 0))) std::cout << "r " << pRank << ": Start precondition" << std::endl;
    //TODO: add opportunity to set custom Setup parameters
    Sd->SetMatrix(*Ad, true, false);
    if (verb >= 3 || ((verb == 2) && (pRank== 0))) std::cout << "r " << pRank << ": End precondition: prec_time = " << Sd->PreconditionerTime() <<  std::endl;
    LS_CONTENT_P(S)->updated_mat = 1;
    return SUNLS_SUCCESS;
}

int SUNLinSolSolve_Inmost(SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, realtype tol) {
    auto Sd = LS_DATA_P(S);
    auto Ad = SM_DATA_P(A);
    auto verb = LS_CONTENT_P(S)->verbosity;
    int pRank = 0;
#if defined(USE_MPI)
    MPI_Comm_rank(MPI_COMM_WORLD, &pRank);
#endif
    if (LS_CONTENT_P(S)->updated_mat != 1){
        //TODO: add opportunity to set custom SolveSetup function
        if (verb >= 3 || ((verb == 2) && (pRank== 0))) std::cout << "r " << pRank << ": Start precondition" << std::endl;
        Sd->SetMatrix(*Ad, false, true);
        if (verb >= 3 || ((verb == 2) && (pRank== 0))) std::cout << "r " << pRank << ": End precondition: prec_time = " << Sd->PreconditionerTime() <<  std::endl;
        LS_CONTENT_P(S)->updated_mat = 0;
    }
    Sd->SetParameterReal("absolute_tolerance", tol);
    double t_solve = Timer();
    if (verb >= 3 || ((verb == 2) && (pRank== 0))) std::cout << "r " << pRank << ": Start solve with tol = "
        << tol << " | x: " << NV_DATA_P(x)->Size() << "  b: " << NV_DATA_P(b)->Size()
        << "  mtx: " << SM_DATA_P(A)->Size() << " x " << SM_DATA_P(A)->Size() << std::endl;
    auto flag = Sd->Solve(*NV_DATA_P(b), *NV_DATA_P(x));
    if (verb >= 3 || ((verb == 2) && (pRank== 0))) std::cout << "r " << pRank << ": End solve: solve_time = " << Sd->IterationsTime() << " prec_time = " << Sd->PreconditionerTime() <<  std::endl;
    t_solve = Timer() - t_solve;

    if (!flag){
        realtype init_resid = std::sqrt(N_VDotProd_Inmost(b, b));
        auto resid = Sd->Residual();
        auto its = Sd->Iterations();
        if (verb > 0) {
            std::cout << "r " << pRank << ": solution_failed Iterations " << its << " Residual " << resid
                      << " (init_resid = " << init_resid << "). Time = " << t_solve << std::endl;
            std::cout << "Reason: " << Sd->GetReason() << std::endl;
        }
        // bool max_its = (static_cast<int>(its) >= atoi(Sd->GetParameter("maximum_iterations").c_str()));
        bool reduce_resid = (init_resid >= resid);
        bool achieve_tol = (resid <= tol);
        if (reduce_resid && achieve_tol) return SUNLS_PACKAGE_FAIL_UNREC;
        if (reduce_resid && !achieve_tol) return SUNLS_RES_REDUCED;
        if (!reduce_resid && achieve_tol) return SUNLS_PACKAGE_FAIL_UNREC;
        if (!reduce_resid && !achieve_tol) return SUNLS_CONV_FAIL;
    } else {
        auto resid = Sd->Residual();
        if (verb >= 1 && pRank == 0)
            std::cout<<"solved_succesful iterations "<<Sd->Iterations()<<" Residual "<<Sd->Residual()<<
                     ". iter_time = " << t_solve << " prec_time = " << Sd->PreconditionerTime() <<  std::endl;
        if (resid <= tol) return SUNLS_SUCCESS;
        else {
            realtype init_resid = std::sqrt(N_VDotProd_Inmost(b, b));
            if (init_resid >= resid) return SUNLS_CONV_FAIL;
            else return SUNLS_RES_REDUCED;
        }
    }

    return SUNLS_MEM_FAIL;
}

int SUNLinSolNumIters_Inmost(SUNLinearSolver LS) {
    return LS_DATA_P(LS)->Iterations();
}
realtype SUNLinSolResNorm_Inmost(SUNLinearSolver LS) {
    return LS_DATA_P(LS)->Residual();
}
SUNLinearSolverContent_Inmost SUNLinSolContent_Inmost(SUNLinearSolver LS) {
    return (LS) ? LS_CONTENT_P(LS) : NULL;
}

#if SUNDIALS_VERSION_MAJOR >= 6
#define SUNNLS_CTX m_ctx 
#define SUNNLS_CTX_COMMA m_ctx,
#else  
#define SUNNLS_CTX  
#define SUNNLS_CTX_COMMA      
#endif

SUNNonlinearSolver::SUNNonlinearSolver(void* comm) { 
#if SUNDIALS_VERSION_MAJOR >= 6
    m_ctx = sundials::Context(comm);
#else
    (void) comm;    
#endif
    kin = KINCreate(SUNNLS_CTX);
    if (!kin) throw std::runtime_error("Error in initialization of KINCreate"); 
}

#if SUNDIALS_VERSION_MAJOR >= 6
SUNNonlinearSolver::SUNNonlinearSolver(sundials::Context&& ctx): m_ctx(std::move(ctx)) {
        kin = KINCreate(SUNNLS_CTX);
        if (!kin) throw std::runtime_error("Error in initialization of KINCreate");
    }
#endif 

SUNNonlinearSolver::SUNNonlinearSolver(SUNNonlinearSolver::Solver &s, SUNNonlinearSolver::Matrix &m,
                                       SUNNonlinearSolver::Vector &x) {
#if SUNDIALS_VERSION_MAJOR >= 6
    m_ctx = sundials::Context(choose_void_internals(m.GetCommunicator()));
#endif
    auto& m_x = SUNNonlinearSolver::x;
    LS = SUNLinSolInmost(SUNNLS_CTX_COMMA &s); if (!LS ) throw std::runtime_error("Error in initialization of SUNLinSolInmost");
    A   = SUNMatInmost(SUNNLS_CTX_COMMA &m);   if (!A  ) throw std::runtime_error("Error in initialization of SUNMatInmost");
    m_x = N_VNew_Inmost(SUNNLS_CTX_COMMA &x);  if (!m_x) throw std::runtime_error("Error in initialization of N_VNew_Inmost for x");

    kin = KINCreate(SUNNLS_CTX); if (!kin) throw std::runtime_error("Error in initialization of KINCreate");
}

SUNNonlinearSolver::~SUNNonlinearSolver() {
    KINFree(&kin);
    SUNLinSolFree_Inmost(LS); LS = nullptr;
    SUNMatDestroy_Inmost(A); A = nullptr;
    N_VDestroy_Inmost(x); x = nullptr;
    N_VDestroy_Inmost(u_scale); u_scale = nullptr;
}

SUNNonlinearSolver &SUNNonlinearSolver::Init() {
    int ierr = 0;
#define DO(X, STR) if ((ierr = X) != KIN_SUCCESS) throw std::runtime_error(STR + std::string(": ") + std::to_string(ierr))
    DO(KINInit(kin, assmRHS_interface, x), "Error in KINInit");

    if (!u_scale) {
        u_scale = N_VClone_Inmost(x); if (!u_scale) throw std::runtime_error( "Error in nvclone");
        N_VConst_Inmost(1.0, u_scale);
    }
    DO(KINSetLinearSolver(kin, LS, A), "Error in KINSetLinearSolver");
    DO(KINSetJacFn(kin, assmMAT_interface), "Error in KINSetJacFn");
    DO(KINSetUserData(kin, this), "Error in KINSetUserData");
#undef DO
    return *this;
}

bool SUNNonlinearSolver::Solve(SUNNonlinearSolver::SolveStrategy s) {
    N_Vector lu_scale, lf_scale;
    lu_scale = u_scale;
    lf_scale = u_scale;
    _gmat_assm_time = _grhs_assm_time = _lmat_assm_time = _lrhs_assm_time = 0;
    reason_flag = KINSol(kin, x, s, lu_scale, lf_scale);
    return (reason_flag >= 0);
}

std::string SUNNonlinearSolver::GetReason() {
    switch (reason_flag) {
        case KIN_SUCCESS:
            return  "Solution success";
        case KIN_INITIAL_GUESS_OK:
            return  "The guess satisfied the system within the tolerance specified";
        case KIN_STEP_LT_STPTOL:
            return  "Solving stopped based on scaled step length. This means that the current iterate may\n"
                    "be an approximate solution of the given nonlinear system, but it is also quite possible\n"
                    "that the algorithm is “stalled” (making insufficient progress) near an invalid solution,\n"
                    "or that the scalar scsteptol is too large (see KINSetScaledStepTol in to\n"
                    "change scsteptol from its default value).";
        case KIN_MEM_NULL:
            return  "The kinsol memory block pointer was NULL.";
        case KIN_ILL_INPUT:
            return  "Initialization of SUNNonlinearSolver is incomplete.";
        case KIN_NO_MALLOC:
            return  "The kinsol memory was not allocated by a call to KINCreate.";
        case KIN_MEM_FAIL:
            return  "A memory allocation failed.";
        case KIN_LINESEARCH_NONCONV:
            return  "The line search algorithm was unable to find an iterate sufficiently distinct from the\n"
                    "current iterate, or could not find an iterate satisfying the sufficient decrease condition.\n"
                    "Failure to satisfy the sufficient decrease condition could mean the current iterate\n"
                    "is “close” to an approximate solution of the given nonlinear system, the difference\n"
                    "approximation of the matrix-vector product J(u)v is inaccurate, or the real scalar\n"
                    "scsteptol is too large.";
        case KIN_MAXITER_REACHED:
            return  "The maximum number of nonlinear iterations has been reached.";
        case KIN_MXNEWT_5X_EXCEEDED:
            return  "Five consecutive steps have been taken that satisfy the inequality \n"
                    "||D_u*p||_{L2} > 0.99*mxnewtstep, where p denotes the current step and mxnewtstep is a scalar\n"
                    "upper bound on the scaled step length. Such a failure may mean that ||D_F*F(u)||_{L2} \n"
                    "asymptotes from above to a positive value, or the real scalar mxnewtstep is too small.\n"
                    "By default D_u = D_F = I.";
        case KIN_LINESEARCH_BCFAIL:
            return  "The line search algorithm was unable to satisfy the “beta-condition” for MXNBCF +1\n"
                    "nonlinear iterations (not necessarily consecutive), which may indicate the algorithm\n"
                    "is making poor progress.";
        case KIN_LINSOLV_NO_RECOVERY:
            return  "The linear solver's solve function failed recoverably, but the Jacobian data is already current.";
        case KIN_LINIT_FAIL:
            return  "The KINLinearSolver initialization routine (linit) encountered an error.";
        case KIN_LSETUP_FAIL:
            return  "The KINLinearSolver setup routine (lsetup) encountered an error; e.g., the user-supplied routine\n"
                    "pset (used to set up the preconditioner data) encountered an unrecoverable error.";
        case KIN_LSOLVE_FAIL:
            return  "The KINLinearSolver solve routine (lsolve) encountered an error.";
        case KIN_SYSFUNC_FAIL:
            return  "Unrecoverable error in user-supplied functions.";
        case KIN_FIRST_SYSFUNC_ERR:
            return  "User-supplied functions failed recoverably at the first call.";
        case KIN_REPTD_SYSFUNC_ERR:
            return  "User-supplied functions had repeated recoverable errors. No recovery is possible.";
        default:
            return  "Encountered unknown error";
    }
}

SUNNonlinearSolver &SUNNonlinearSolver::SetMatrix(SUNNonlinearSolver::Matrix &A) {
    auto& m_A = SUNNonlinearSolver::A;
    SUNMatDestroy_Inmost(m_A); m_A = SUNMatInmost(SUNNLS_CTX_COMMA &A);
    if (!m_A) throw std::runtime_error("Error in initialization of SUNMatInmost");
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetMatrix(std::string name) {
    SUNMatDestroy_Inmost(A); A = SUNMatInmost(SUNNLS_CTX_COMMA name);
    if (!A) throw std::runtime_error("Error in initialization of SUNMatInmost");
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetInitialGuess(SUNNonlinearSolver::Vector &x) {
    auto& m_x = SUNNonlinearSolver::x;
    N_VDestroy_Inmost(m_x); m_x = N_VNew_Inmost(SUNNLS_CTX_COMMA &x);
    if (!m_x) throw std::runtime_error("Error in initialization of N_VNew_Inmost for x");
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetLinearSolver(SUNNonlinearSolver::Solver &s) {
    SUNLinSolFree_Inmost(LS); LS = SUNLinSolInmost(SUNNLS_CTX_COMMA &s);
    if (!LS) throw std::runtime_error("Error in initialization of SUNLinSolInmost");
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetAssembler(std::function<int(Vector & , Matrix & , Vector & )> assm) {
    return SUNNonlinearSolver::assm = std::move(assm), *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetAssemblerRHS(std::function<int(Vector & , Vector & )> assm) {
    return assmRHS = std::move(assm), *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetAssemblerMAT(std::function<int(Vector & , Matrix & )> assm) {
    return assmMAT = std::move(assm), *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetErrHandlerFn(SUNNonlinearSolver::ErrorHandler eh) {
    ehand = std::move(eh);
    KINSetErrHandlerFn(kin, errorHandler, this);
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetInfoHandlerFn(SUNNonlinearSolver::InfoHandler ih) {
    ihand = std::move(ih);
    KINSetInfoHandlerFn(kin, infoHandler, this);
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetErrHandlerFile(std::ostream &out) {
    return SetErrHandlerFn([&out](int error_code, const char *module, const char *function, char *msg){
        out << "\n"
               "[" << module << " " << ((error_code == KIN_WARNING) ? "WARNING": "ERROR") << "] " << function << "\n"
                                                                                                                 "  " << msg << "\n" << std::endl;
    });
}

SUNNonlinearSolver &SUNNonlinearSolver::SetInfoHandlerFile(std::ostream &out) {
    return SetInfoHandlerFn([&out](const char *module, const char *function, char *msg){
        out << "\n[" << module << "] " << function << "\n  " << msg << std::endl;
    });
}

SUNNonlinearSolver &SUNNonlinearSolver::SetVerbosityLevel(int lvl) {
    if (lvl <= 0) KINSetPrintLevel(kin, 0);
    else if (lvl >= 3) KINSetPrintLevel(kin, 3);
    else KINSetPrintLevel(kin, lvl);
    return *this;
}

SUNNonlinearSolver &SUNNonlinearSolver::SetConstraints(SUNNonlinearSolver::Vector &constraints) {
    auto constrs = N_VNew_Inmost(SUNNLS_CTX_COMMA &constraints);
    KINSetConstraints(kin, constrs);
    N_VDestroy_Inmost(constrs);
    return *this;
}

std::vector<SUNNonlinearSolver::ParamDiscr> SUNNonlinearSolver::GetAvaliableParametrs() {
    static std::vector<ParamDiscr> res = {
            {"MaxIters", "Max. number of nonlinear iterations", "200", INTEGER},
            {"UseInitSetup", "Does make initial matrix setup?", "true", BOOLEAN},
            {"UseResMon", "Does perform residual monitoring?", "true", BOOLEAN},
            {"MaxSetupCalls", "Max. iterations without matrix setup", "10", INTEGER},
            // MaxSetupCalls % MaxSubSetupCalls must be equal 0!
            {"MaxSubSetupCalls", "Max. iterations between checks by the residual monitoring algorithm", "5", INTEGER},
            {"EtaForm", "Form of η coefficient", "1", ETACHOISEENUM},
            {"EtaConstValue", "Constant value of η for constant eta choice", "0.1", REAL},
            {"EtaGamma", "Values of γ for ETECHOICE2", "0.9", REAL},
            {"EtaAlpha", "Values of α for ETECHOICE2", "2.0", REAL},
            {"ResMonWmin", "Value of ω_{min}", "1e-5", REAL},
            {"ResMonWmax", "Value of ω_{max}", "0.9", REAL},
            {"ResMonConst", "Constant value of ω^∗", "0.9", REAL},
            {"UseMinEps", "Do set lower bound on epsilon?", "true", BOOLEAN},
            {"MaxNewtonStep", "Max. scaled length of Newton step", "1000||D_u*u_0||_2", REAL},
            {"MaxBetaFails", "Max. number nonlinear its with β-condition failures", "10", INTEGER},
            {"FuncNormTol", "Function-norm stopping tolerance", "(DBL_EPSILON)^(1/3)", REAL},
            {"ScaledSteptol", "Scaled-step stopping tolerance", "(DBL_EPSILON)^(2/3)", REAL},
            // MAA should be setted before KINInit and should be less, than MaxIters
            {"MAA", "Anderson Acceleration subspace size", "0", INTEGER},
            {"DampingAA", "Anderson Acceleration damping parameter beta (0 < beta ≤ 1.0)", "1", REAL}
    };
    return res;
}

bool SUNNonlinearSolver::SetParameterReal(std::string param, double val) {
    if (param == "EtaConstValue") return KINSetEtaConstValue(kin, val), true;
    if (param == "EtaGamma") {
        _egamma = val;
        return KINSetEtaParams(kin, _egamma, _ealpha), true;
    }
    if (param == "EtaAlpha") {
        _ealpha = val;
        return KINSetEtaParams(kin, _egamma, _ealpha), true;
    }
    if (param == "ResMonWmin") {
        _wmin = val;
        return KINSetResMonParams(kin, _wmin, _wmax), true;
    }
    if (param == "ResMonWmax") {
        _wmax = val;
        return KINSetResMonParams(kin, _wmin, _wmax), true;
    }
    if (param == "ResMonConst") return KINSetResMonConstValue(kin, val), true;
    if (param == "MaxNewtonStep") return KINSetMaxNewtonStep(kin, val), true;
    if (param == "FuncNormTol") return KINSetFuncNormTol(kin, val), true;
    if (param == "ScaledSteptol") return KINSetScaledStepTol(kin, val), true;
    if (param == "DampingAA") {
        if (val > 1 || val < 0) return false;
        return KINSetDampingAA(kin, val), true;
    }
    return false;
}

bool SUNNonlinearSolver::SetParameterIntegral(std::string param, int val) {
    if (param == "MaxIters") return KINSetNumMaxIters(kin, val), true;
    if (param == "UseInitSetup") return KINSetNoInitSetup(kin, !val), true;
    if (param == "UseResMon") return KINSetNoResMon(kin, !val), true;
    if (param == "MaxSetupCalls") return KINSetMaxSetupCalls(kin, val), true;
    if (param == "MaxSubSetupCalls") return KINSetMaxSubSetupCalls(kin, val), true;
    if (param == "EtaForm") {
        if (val < 1 || val > 3) return false;
        return KINSetEtaForm(kin, val), true;
    }
    if (param == "UseMinEps") return KINSetNoMinEps(kin, !val), true;
    if (param == "MaxBetaFails") return KINSetMaxBetaFails(kin, val), true;
    if (param == "MAA") return KINSetMAA(kin, val), true;
    return false;
}

bool SUNNonlinearSolver::SetParameter(std::string param, std::string val) {
    static std::map<std::string, ParamType> prms = {
            {"MaxIters", INTEGER},
            {"UseInitSetup", BOOLEAN},
            {"UseResMon", BOOLEAN},
            {"MaxSetupCalls", INTEGER},
            {"MaxSubSetupCalls", INTEGER},
            {"EtaForm", ETACHOISEENUM},
            {"EtaConstValue",  REAL},
            {"EtaGamma", REAL},
            {"EtaAlpha", REAL},
            {"ResMonWmin", REAL},
            {"ResMonWmax", REAL},
            {"ResMonConst", REAL},
            {"UseMinEps", BOOLEAN},
            {"MaxNewtonStep", REAL},
            {"MaxBetaFails", INTEGER},
            {"FuncNormTol", REAL},
            {"ScaledSteptol", REAL},
            {"MAA", INTEGER},
            {"DampingAA", REAL}
    };
    auto it = prms.find(param);
    if (it == prms.end()) return false;
    auto type = it->second;
    switch (type) {
        case REAL: return SetParameterReal(param, atof(val.c_str()));
        case INTEGER: return SetParameterIntegral(param, atoi(val.c_str()));
        case BOOLEAN: {
            std::string tmp = val;
            std::transform(val.begin(), val.end(), tmp.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (tmp == "true") return SetParameterIntegral(param, 1);
            if (tmp == "false") return SetParameterIntegral(param, 0);
            return SetParameterIntegral(param, atoi(val.c_str()));
        }
        case ETACHOISEENUM: return SetParameterIntegral(param, atoi(val.c_str()));
    }
    return false;
}

int SUNNonlinearSolver::assmRHS_interface(N_Vector x, N_Vector b, void *this_obj) {
    SUNNonlinearSolver& sns = *static_cast<SUNNonlinearSolver*>(this_obj);
    if (!sns.assmRHS) return KIN_SYSFUNC_FAIL;
    sns._lrhs_assm_time -= Timer();
    int stat = sns.assmRHS(*NV_DATA_P(x), *NV_DATA_P(b));
    sns._lrhs_assm_time += Timer();
    double dat[2] = {static_cast<double>(stat), -sns._lrhs_assm_time};
#ifdef USE_MPI    
    MPI_Allreduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_MIN, NV_DATA_P(x)->GetCommunicator());
#endif    
    stat = static_cast<int>(dat[0]); sns._grhs_assm_time = -dat[1];
    if (stat < 0) return 1;
    return KIN_SUCCESS;
}

int
SUNNonlinearSolver::assmMAT_interface(N_Vector x, N_Vector b, SUNMatrix J, void *this_obj, N_Vector tmp1, N_Vector tmp2) {
    (void) b; (void) tmp2;
    SUNNonlinearSolver& sns = *static_cast<SUNNonlinearSolver*>(this_obj);
    if (sns.assmMAT) {
        sns._lmat_assm_time -= Timer();
        int stat = sns.assmMAT(*NV_DATA_P(x), *SM_DATA_P(J));
        sns._lmat_assm_time += Timer();
        double dat[2] = {static_cast<double>(stat), -sns._lmat_assm_time};
    #ifdef USE_MPI    
        MPI_Allreduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_MIN, NV_DATA_P(x)->GetCommunicator());
    #endif    
        stat = static_cast<int>(dat[0]); sns._gmat_assm_time = -dat[1];
        if (stat < 0) return 1;
        return KIN_SUCCESS;
    } else if (sns.assm) {
        sns._lmat_assm_time -= Timer();
        int stat = sns.assm(*NV_DATA_P(x), *SM_DATA_P(J), *NV_DATA_P(tmp1));
        sns._lmat_assm_time += Timer();
        double dat[2] = {static_cast<double>(stat), -sns._lmat_assm_time};
    #ifdef USE_MPI    
        MPI_Allreduce(MPI_IN_PLACE, dat, 2, MPI_DOUBLE, MPI_MIN, NV_DATA_P(x)->GetCommunicator());
    #endif    
        stat = static_cast<int>(dat[0]); sns._gmat_assm_time = -dat[1];
        if (stat < 0) return 1;
        return KIN_SUCCESS;
    }
    return KIN_SYSFUNC_FAIL;
}

void
SUNNonlinearSolver::errorHandler(int error_code, const char *module, const char *function, char *msg, void *user_data) {
    SUNNonlinearSolver& sns = *static_cast<SUNNonlinearSolver*>(user_data);
    if (sns.ehand) sns.ehand(error_code, module, function, msg);
}

void SUNNonlinearSolver::infoHandler(const char *module, const char *function, char *msg, void *user_data) {
    SUNNonlinearSolver& sns = *static_cast<SUNNonlinearSolver*>(user_data);
    if (sns.ihand) sns.ihand(module, function, msg);
}

double SUNNonlinearSolver::GetMatAssembleTime() const {
    return _gmat_assm_time;
}
double SUNNonlinearSolver::GetRHSAssembleTime() const {
    return _grhs_assm_time;
}

std::ostream& operator<<(std::ostream& out, SUNNonlinearSolver::ParamType t){
    switch (t)
    {
        case SUNNonlinearSolver::ParamType::REAL: return out << "REAL";
        case SUNNonlinearSolver::ParamType::INTEGER: return out << "INTEGER";
        case SUNNonlinearSolver::ParamType::BOOLEAN: return out << "BOOLEAN";
        case SUNNonlinearSolver::ParamType::ETACHOISEENUM: return out << "ETACHOISEENUM";
    }
    return out;
}