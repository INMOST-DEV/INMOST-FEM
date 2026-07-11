#include "complement_fem_space.h"
#include "equivariant_complement_builder.h"
#include "quadrature_formulas.h"
#include "operations/operations.h"
#include "spaces/spaces.h"
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <array>

namespace Ani {
namespace {

using DofT::uchar;

constexpr double kDefaultTol = 1e-12;

void setup_animem(PlainMemoryX<>& pmx, const BaseFemSpace& space, uint nquad,
    OpMemoryRequirements(BaseFemSpace::*req)(uint, uint) const = &BaseFemSpace::memIDEN){
    auto r = (space.*req)(nquad, 1);
    pmx.dSize += r.Usz + r.extraRsz;
    pmx.iSize += r.extraIsz + 2*(r.mtx_parts + 2);
    pmx.mSize += r.mtx_parts;
}

void init_animem(AniMemoryX<>& mem, uint nquad){
    mem.q = nquad;
    mem.f = 1;
    mem.busy_mtx_parts = 0;
    mem.U.Init(nullptr, 0);
    mem.extraR.Init(nullptr, 0);
    mem.extraI.Init(nullptr, 0);
    mem.MTX.Init(nullptr, 0);
    mem.MTXI_COL.Init(nullptr, 0);
    mem.MTXI_ROW.Init(nullptr, 0);
}

void alloc_animem_from_pmx(AniMemoryX<>& mem, PlainMemoryX<>& pmx, const BaseFemSpace& space, uint nquad,
    OpMemoryRequirements(BaseFemSpace::*req)(uint, uint) const = &BaseFemSpace::memIDEN){
    auto r = (space.*req)(nquad, 1);
    mem.U.Init(pmx.ddata, r.Usz); pmx.ddata += r.Usz; pmx.dSize -= r.Usz;
    mem.extraR.Init(pmx.ddata, r.extraRsz); pmx.ddata += r.extraRsz; pmx.dSize -= r.extraRsz;
    mem.extraI.Init(pmx.idata, r.extraIsz); pmx.idata += r.extraIsz; pmx.iSize -= r.extraIsz;
    mem.MTXI_COL.Init(pmx.idata, r.mtx_parts+2); pmx.idata += r.mtx_parts+2; pmx.iSize -= r.mtx_parts+2;
    mem.MTXI_ROW.Init(pmx.idata, r.mtx_parts+2); pmx.idata += r.mtx_parts+2; pmx.iSize -= r.mtx_parts+2;
    mem.MTX.Init(pmx.mdata, r.mtx_parts); pmx.mdata += r.mtx_parts; pmx.mSize -= r.mtx_parts;
}

void matmul(const double* A, int n, int m, const double* B, int p, double* C){
    std::fill(C, C + n*p, 0.0);
    for (int j = 0; j < p; ++j)
        for (int k = 0; k < m; ++k)
            for (int i = 0; i < n; ++i)
                C[i + j*n] += A[i + k*n] * B[k + j*m];
}

template<typename Scalar>
int svd_rank_t(const Scalar* s, int k, Scalar tol){
    int r = 0;
    for (int i = 0; i < k; ++i)
        if (s[i] > tol) ++r;
    return r;
}

/// Right nullspace of A (n x m). Prefer long double for poorly scaled energy constraints.
std::vector<std::vector<double>> nullspace_right(const double* A, int n, int m, double tol, bool use_ld){
    const int k = std::min(n, m);
    auto extract = [&](auto* V, int rank){
        const int nd = m - rank;
        std::vector<std::vector<double>> basis(nd, std::vector<double>(m, 0.0));
        for (int b = 0; b < nd; ++b)
            for (int i = 0; i < m; ++i)
                basis[b][i] = static_cast<double>(V[i + (rank + b) * m]);
        return basis;
    };
    if (use_ld){
        std::vector<long double> Ald(static_cast<std::size_t>(n) * m);
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < n; ++i)
                Ald[i + j * n] = static_cast<long double>(A[i + j * n]);
        std::vector<long double> U(static_cast<std::size_t>(n) * n), s(k), V(static_cast<std::size_t>(m) * m), mem(2 * k * k + k);
        if (jacobiSVD(Ald.data(), n, m, U.data(), s.data(), V.data(), mem.data()))
            throw std::runtime_error("jacobiSVD failed in ComplementFemSpace::make");
        return extract(V.data(), svd_rank_t(s.data(), k, static_cast<long double>(tol)));
    }
    std::vector<double> U(n * n), s(k), V(m * m), mem(2 * k * k + k);
    if (jacobiSVD(A, n, m, U.data(), s.data(), V.data(), mem.data()))
        throw std::runtime_error("jacobiSVD failed in ComplementFemSpace::make");
    return extract(V.data(), svd_rank_t(s.data(), k, tol));
}

double svd_max_sigma(const double* S, int n0, int n1, bool use_ld){
    const int k = std::min(n0, n1);
    if (k <= 0) return 0.0;
    if (use_ld){
        std::vector<long double> Sld(static_cast<std::size_t>(n0) * n1);
        for (int j = 0; j < n1; ++j)
            for (int i = 0; i < n0; ++i)
                Sld[i + j * n0] = static_cast<long double>(S[i + j * n0]);
        std::vector<long double> s(k), mem(2 * k * k + k);
        if (jacobiSVD(Sld.data(), n0, n1, static_cast<long double*>(nullptr), s.data(),
                      static_cast<long double*>(nullptr), mem.data()))
            throw std::runtime_error("jacobiSVD failed in ComplementFemSpace::make");
        return static_cast<double>(s[0]);
    }
    std::vector<double> s(k), mem(2 * k * k + k);
    if (jacobiSVD(S, n0, n1, static_cast<double*>(nullptr), s.data(),
                  static_cast<double*>(nullptr), mem.data()))
        throw std::runtime_error("jacobiSVD failed in ComplementFemSpace::make");
    return s[0];
}

void evaluate_basis_at_quad(const BaseFemSpace& space, AniMemoryX<>& mem,
                            const double* xyl, const double* wg, uint nquad,
                            double* phi){
    const uint nfa = space.m_order.NumDofOnTet();
    const uint gdim = space.dim();
    std::fill(phi, phi + static_cast<std::size_t>(nfa) * nquad * gdim, 0.0);
    for (uint q = 0; q < nquad; ++q){
        mem.XYL.Init(const_cast<double*>(xyl) + 4*q, 4);
        mem.WG.Init(const_cast<double*>(&wg[q]), 1);
        mem.q = 1;
        mem.busy_mtx_parts = 0;
        auto res = space.applyIDEN(mem, mem.U);
        for (std::size_t p = 0; p < res.nparts; ++p)
        for (int d = res.stRow[p]; d < res.stRow[p+1]; ++d)
        for (int i = res.stCol[p]; i < res.stCol[p+1]; ++i)
            if (d >= 0 && d < static_cast<int>(gdim) && i >= 0 && i < static_cast<int>(nfa))
                phi[q + nquad * i + nfa * nquad * d] = res.data[p](d - res.stRow[p], i - res.stCol[p]);
    }
}

void build_mass_matrix(const BaseFemSpace& V1, uint nquad, const double* xyl, const double* wg,
                       AniMemoryX<>& mem, double* M1, int n1){
    const uint gdim = V1.dim();
    std::vector<double> phi(static_cast<std::size_t>(n1) * nquad * gdim);
    evaluate_basis_at_quad(V1, mem, xyl, wg, nquad, phi.data());
    for (int i = 0; i < n1; ++i)
        for (int j = i; j < n1; ++j){
            double s = 0;
            for (uint q = 0; q < nquad; ++q)
                for (uint d = 0; d < gdim; ++d)
                    s += wg[q] * phi[q + nquad * i + n1 * nquad * d]
                                * phi[q + nquad * j + n1 * nquad * d];
            M1[i + j*n1] = M1[j + i*n1] = s;
        }
}

#ifndef NDEBUG
void build_cross_mass_matrix(const BaseFemSpace& V0, const BaseFemSpace& V1,
                             uint nquad, const double* xyl, const double* wg,
                             AniMemoryX<>& mem_v0, AniMemoryX<>& mem_v1,
                             double* M01, int n0, int n1){
    const uint gdim = V1.dim();
    std::vector<double> phi0(static_cast<std::size_t>(n0) * nquad * gdim);
    std::vector<double> phi1(static_cast<std::size_t>(n1) * nquad * gdim);
    evaluate_basis_at_quad(V0, mem_v0, xyl, wg, nquad, phi0.data());
    evaluate_basis_at_quad(V1, mem_v1, xyl, wg, nquad, phi1.data());
    for (int i = 0; i < n0; ++i)
        for (int j = 0; j < n1; ++j){
            double s = 0;
            for (uint q = 0; q < nquad; ++q)
                for (uint d = 0; d < gdim; ++d)
                    s += wg[q] * phi0[q + nquad * i + n0 * nquad * d]
                                * phi1[q + nquad * j + n1 * nquad * d];
            M01[i + j * n0] = s;
        }
}

bool is_V0_subspace_of_V1(const double* C, const double* M1, const double* M01,
                          int n0, int n1, double tol){
    if (n0 <= 0 || n1 < n0)
        return false;
    std::vector<double> CM1(static_cast<std::size_t>(n0) * n1, 0.0);
    matmul(C, n0, n1, M1, n1, CM1.data());
    double res2 = 0, ref2 = 0;
    for (int k = 0; k < n0 * n1; ++k){
        const double d = M01[k] - CM1[k];
        res2 += d * d;
        ref2 += M01[k] * M01[k];
    }
    const double rel = std::sqrt(res2) / std::max(1.0, std::sqrt(ref2));
    if (rel > tol)
        return false;
    const int k = std::min(n0, n1);
    std::vector<double> U(n0 * n0), s(k), V(n1 * n1), mem(2 * k * k + k);
    if (jacobiSVD(C, n0, n1, U.data(), s.data(), V.data(), mem.data()))
        return false;
    const double smax = k > 0 ? s[0] : 0.0;
    const double rank_tol = std::max(tol, tol * std::max(1.0, smax));
    return svd_rank_t(s.data(), k, rank_tol) == n0;
}
#endif

struct BasisFunctorData {
    const BaseFemSpace* space;
    AniMemoryX<>* mem;
    int idof;
};

int eval_basis_functor(const std::array<double, 3>& X, double* res, uint dim, void* user_data){
    auto& d = *static_cast<BasisFunctorData*>(user_data);
    auto req = d.space->memIDEN(1, 1);
    PlainMemoryX<> pmx;
    pmx.dSize = req.Usz + req.extraRsz;
    pmx.iSize = req.extraIsz + 2*(req.mtx_parts + 2);
    pmx.mSize = req.mtx_parts;
    std::vector<char> buf(pmx.enoughRawSize());
    pmx.allocateFromRaw(buf.data(), buf.size());
    AniMemoryX<> mem;
    mem.q = 1;
    mem.f = 1;
    mem.busy_mtx_parts = 0;
    mem.XYP = d.mem->XYP;
    mem.PSI = d.mem->PSI;
    mem.DET = d.mem->DET;
    alloc_animem_from_pmx(mem, pmx, *d.space, 1);
    mem.XYG.Init(const_cast<double*>(X.data()), 3);
    std::array<double, 4> xyl;
    xyl[0] = 1 - (X[0] + X[1] + X[2]);
    for (int k = 0; k < 3; ++k) xyl[1+k] = X[k];
    mem.XYL.Init(xyl.data(), 4);
    auto r = d.space->applyIDEN(mem, mem.U);
    std::fill(res, res + dim, 0.0);
    for (std::size_t p = 0; p < r.nparts; ++p)
    for (int row = r.stRow[p]; row < r.stRow[p+1]; ++row)
    for (int col = r.stCol[p]; col < r.stCol[p+1]; ++col)
        if (col == d.idof && row >= 0 && row < static_cast<int>(dim))
            res[row] = r.data[p](row - r.stRow[p], col - r.stCol[p]);
    return 0;
}

void build_embedding_matrix(const BaseFemSpace& V0, const BaseFemSpace& V1, const Tetra<const double>& XYZ,
                            AniMemoryX<>& mem_v0, PlainMemoryX<>& pmx,
                            double* C, int n0, int n1, uint max_quad_order){
    BasisFunctorData fd{&V0, &mem_v0, 0};
    for (int i = 0; i < n0; ++i){
        fd.idof = i;
        for (int j = 0; j < n1; ++j){
            auto lpmx = V1.interpolateOnDOF_mem_req(j, 1, max_quad_order);
            pmx.dSize = std::max(pmx.dSize, lpmx.dSize);
            pmx.iSize = std::max(pmx.iSize, lpmx.iSize);
            pmx.mSize = std::max(pmx.mSize, lpmx.mSize);
        }
    }
    auto pmx_sz = pmx.enoughRawSize();
    std::vector<char> wmem(pmx_sz);
    pmx.allocateFromRaw(wmem.data(), pmx_sz);
    double* d0 = pmx.ddata;
    int* i0 = pmx.idata;
    DenseMatrix<>* m0 = pmx.mdata;
    const std::size_t ds = pmx.dSize, is = pmx.iSize, ms = pmx.mSize;
    std::vector<double> val(n1, 0.0);
    for (int i = 0; i < n0; ++i){
        fd.idof = i;
        for (int j = 0; j < n1; ++j){
            std::fill(val.begin(), val.end(), 0.0);
            pmx.ddata = d0; pmx.idata = i0; pmx.mdata = m0;
            pmx.dSize = ds; pmx.iSize = is; pmx.mSize = ms;
            V1.interpolateOnDOF(XYZ, eval_basis_functor, ArrayView<>(val.data(), n1), j, pmx, &fd, max_quad_order);
            C[i + j*n0] = val[j];
        }
    }
}

void build_energy_matrix(const BaseFemSpace& V1, uint nquad, const double* xyl, const double* wg,
                         AniMemoryX<>& mem, const double* D, int gdim3,
                         double* A, int n1){
    std::vector<double> G(static_cast<std::size_t>(n1) * nquad * gdim3, 0.0);
    for (uint q = 0; q < nquad; ++q){
        mem.XYL.Init(const_cast<double*>(xyl) + 4*q, 4);
        mem.WG.Init(const_cast<double*>(&wg[q]), 1);
        mem.q = 1;
        mem.busy_mtx_parts = 0;
        auto res = V1.applyGRAD(mem, mem.U);
        for (std::size_t p = 0; p < res.nparts; ++p)
        for (int c = res.stRow[p]; c < res.stRow[p+1]; ++c)
        for (int i = res.stCol[p]; i < res.stCol[p+1]; ++i)
            if (c >= 0 && c < gdim3 && i >= 0 && i < n1)
                G[q + nquad * i + n1 * nquad * c] = res.data[p](c - res.stRow[p], i - res.stCol[p]);
    }
    std::fill(A, A + n1 * n1, 0.0);
    std::vector<double> Dg(gdim3);
    for (int i = 0; i < n1; ++i)
        for (int j = i; j < n1; ++j){
            double s = 0;
            for (uint q = 0; q < nquad; ++q){
                std::fill(Dg.begin(), Dg.end(), 0.0);
                for (int a = 0; a < gdim3; ++a)
                    for (int b = 0; b < gdim3; ++b)
                        Dg[a] += D[a + b * gdim3] * G[q + nquad * j + n1 * nquad * b];
                double loc = 0;
                for (int a = 0; a < gdim3; ++a)
                    loc += G[q + nquad * i + n1 * nquad * a] * Dg[a];
                s += wg[q] * loc;
            }
            A[i + j * n1] = A[j + i * n1] = s;
        }
}

void build_mean_rows(const BaseFemSpace& V1, uint nquad, const double* xyl, const double* wg,
                     AniMemoryX<>& mem, double* Mconst, int n1, int gdim){
    std::vector<double> phi(static_cast<std::size_t>(n1) * nquad * gdim);
    evaluate_basis_at_quad(V1, mem, xyl, wg, nquad, phi.data());
    std::fill(Mconst, Mconst + gdim * n1, 0.0);
    for (int j = 0; j < n1; ++j)
        for (int d = 0; d < gdim; ++d){
            double s = 0;
            for (uint q = 0; q < nquad; ++q)
                s += wg[q] * phi[q + nquad * j + n1 * nquad * d];
            Mconst[d + j * gdim] = s;
        }
}

bool is_identity_D(const double* D, int gdim3, double tol = 1e-14){
    if (!D) return true;
    for (int j = 0; j < gdim3; ++j)
        for (int i = 0; i < gdim3; ++i){
            const double expect = (i == j) ? 1.0 : 0.0;
            if (std::abs(D[i + j * gdim3] - expect) > tol)
                return false;
        }
    return true;
}

std::shared_ptr<BaseFemSpace> flatten_to_scalar_complex(const std::shared_ptr<BaseFemSpace>& V){
    if (!V) return V;
    if (V->gatherType() == BaseFemSpace::BaseTypes::VectorType){
        auto vv = std::static_pointer_cast<VectorFemSpace>(V);
        if (vv && vv->m_base && vv->m_base->dim() == 1){
            std::vector<std::shared_ptr<BaseFemSpace>> parts(vv->vec_dim(), vv->m_base);
            return std::make_shared<ComplexFemSpace>(std::move(parts));
        }
        return V;
    }
    if (V->gatherType() == BaseFemSpace::BaseTypes::ComplexType){
        auto cv = std::static_pointer_cast<ComplexFemSpace>(V);
        std::vector<std::shared_ptr<BaseFemSpace>> parts;
        for (const auto& p0 : cv->m_spaces){
            auto p = flatten_to_scalar_complex(p0);
            if (p && p->gatherType() == BaseFemSpace::BaseTypes::ComplexType){
                auto cp = std::static_pointer_cast<ComplexFemSpace>(p);
                parts.insert(parts.end(), cp->m_spaces.begin(), cp->m_spaces.end());
            } else {
                parts.push_back(p);
            }
        }
        return std::make_shared<ComplexFemSpace>(std::move(parts));
    }
    return V;
}

#ifndef NDEBUG
bool is_symmetric_D(const double* D, int gdim3, double tol = 1e-10){
    for (int j = 0; j < gdim3; ++j)
        for (int i = j + 1; i < gdim3; ++i)
            if (std::abs(D[i + j * gdim3] - D[j + i * gdim3]) > tol)
                return false;
    return true;
}

void assert_V0_subspace(const BaseFemSpace& V0, const BaseFemSpace& V1,
                        uint nquad, const double* xyl, const double* wg,
                        AniMemoryX<>& mem_v0, AniMemoryX<>& mem_v1,
                        const double* C, const double* M1, int n0, int n1){
    std::vector<double> M01(static_cast<std::size_t>(n0) * n1, 0.0);
    build_cross_mass_matrix(V0, V1, nquad, xyl, wg, mem_v0, mem_v1, M01.data(), n0, n1);
    assert(is_V0_subspace_of_V1(C, M1, M01.data(), n0, n1, 1e-8)
        && "ComplementFemSpace::make: V0 is not a subspace of V1");
}

void assert_P0_in_V0(const BaseFemSpace& V0, uint nquad, const double* xyl, const double* wg,
                     AniMemoryX<>& mem_v0, const Tetra<const double>& XYZ, uint quad_order){
    auto p0 = std::make_shared<P0Space>();
    std::shared_ptr<BaseFemSpace> P0 = (V0.dim() == 1)
        ? std::shared_ptr<BaseFemSpace>(p0)
        : std::make_shared<VectorFemSpace>(static_cast<int>(V0.dim()), p0);
    const int n_p0 = static_cast<int>(P0->m_order.NumDofOnTet());
    const int n0 = static_cast<int>(V0.m_order.NumDofOnTet());
    PlainMemoryX<> pmx_p0;
    setup_animem(pmx_p0, *P0, nquad);
    std::vector<char> buf(pmx_p0.enoughRawSize());
    pmx_p0.allocateFromRaw(buf.data(), buf.size());
    AniMemoryX<> mem_p0;
    init_animem(mem_p0, nquad);
    alloc_animem_from_pmx(mem_p0, pmx_p0, *P0, nquad);
    mem_p0.XYP = mem_v0.XYP;
    mem_p0.PSI = mem_v0.PSI;
    mem_p0.DET = mem_v0.DET;

    std::vector<double> C(n_p0 * n0), M0(n0 * n0), M01(n_p0 * n0);
    PlainMemoryX<> pmx_interp;
    build_embedding_matrix(*P0, V0, XYZ, mem_p0, pmx_interp, C.data(), n_p0, n0, quad_order);
    build_mass_matrix(V0, nquad, xyl, wg, mem_v0, M0.data(), n0);
    build_cross_mass_matrix(*P0, V0, nquad, xyl, wg, mem_p0, mem_v0, M01.data(), n_p0, n0);
    assert(is_V0_subspace_of_V1(C.data(), M0.data(), M01.data(), n_p0, n0, 1e-8)
        && "ComplementFemSpace::make: piecewise constants (P0) are not in V0");
}

void assert_A_S4_invariant(const BaseFemSpace& V1, const double* A, int n1){
    std::array<std::array<uchar, 4>, 24> perms;
    DofT::S4::all_permutations(perms);
    std::vector<double> P(static_cast<std::size_t>(n1) * n1), tmp(n1 * n1), PtAP(n1 * n1);
    double a_nrm2 = 0;
    for (int i = 0; i < n1 * n1; ++i) a_nrm2 += A[i] * A[i];
    double max_rel = 0;
    for (int p = 0; p < 24; ++p){
        DofT::S4::build_dof_permutation(*V1.m_order.base(), perms[p].data(), P.data(), n1);
        matmul(A, n1, n1, P.data(), n1, tmp.data());
        for (int j = 0; j < n1; ++j)
            for (int i = 0; i < n1; ++i){
                double s = 0;
                for (int k = 0; k < n1; ++k)
                    s += P[k + i * n1] * tmp[k + j * n1];
                PtAP[i + j * n1] = s;
            }
        double d2 = 0;
        for (int i = 0; i < n1 * n1; ++i){
            const double d = PtAP[i] - A[i];
            d2 += d * d;
        }
        max_rel = std::max(max_rel, std::sqrt(d2) / std::max(1e-30, std::sqrt(a_nrm2)));
    }
    assert(max_rel <= 1e-8
        && "ComplementFemSpace::make: energy matrix A is not S4-invariant");
}
#endif

template<typename ApplyFn>
BandDenseMatrixX<> apply_op_internal(const ComplementFemSpace& sp, AniMemoryX<>& mem, ArrayView<>& U,
    ApplyFn&& apl, const OpMemoryRequirements& lreq){
    const int n1 = static_cast<int>(sp.m_V1->m_order.NumDofOnTet());
    const int n10 = static_cast<int>(sp.m_order.NumDofOnTet());
    if (mem.extraR.size < lreq.Usz)
        throw std::runtime_error("ComplementFemSpace: not enough extraR for V1 apply buffer");
    const std::size_t saved_extra = mem.extraR.size;
    mem.extraR.size -= lreq.Usz;
    ArrayView<> lU(mem.extraR.data + mem.extraR.size, lreq.Usz);
    BandDenseMatrixX<> Vx = apl(mem, lU);
    uint dimop = 0;
    for (std::size_t p = 0; p < Vx.nparts; ++p)
        dimop = std::max(dimop, static_cast<uint>(Vx.stRow[p+1]));
    const std::size_t need_full = static_cast<std::size_t>(dimop) * mem.q * mem.f * n1;
    if (mem.extraR.size < need_full){
        mem.extraR.size = saved_extra;
        throw std::runtime_error("ComplementFemSpace: not enough extraR for V1 dense buffer");
    }
    DenseMatrix<> lUfull(mem.extraR.data, dimop * mem.q, mem.f * n1, mem.extraR.size);
    lUfull.SetZero();
    for (std::size_t p = 0; p < Vx.nparts; ++p){
        int nCol = Vx.stCol[p+1] - Vx.stCol[p], nRow = Vx.stRow[p+1] - Vx.stRow[p];
        for (std::size_t r = 0; r < mem.f; ++r)
        for (int j = 0; j < nCol; ++j)
        for (std::size_t n = 0; n < mem.q; ++n)
        for (int k = 0; k < nRow; ++k)
            lUfull(k + Vx.stRow[p] + dimop*n, j + Vx.stCol[p] + n1*r) = Vx.data[p](k + nRow*n, j + nCol*r);
    }
    DenseMatrix<> llU(U.data, dimop * mem.q, mem.f * n10, U.size);
    llU.SetZero();
    for (std::size_t r = 0; r < mem.f; ++r)
    for (int i = 0; i < n10; ++i)
    for (int j = 0; j < n1; ++j)
    for (std::size_t n = 0; n < mem.q; ++n)
    for (uint k = 0; k < dimop; ++k)
        llU(k + dimop*n, i + n10*r) += sp.m_basis_coefs[j + n1*i] * lUfull(k + dimop*n, j + n1*r);
    mem.extraR.size = saved_extra;
    uint st_busy = mem.busy_mtx_parts;
    uint mtxishift = st_busy > 0 ? 1 : 0;
    BandDenseMatrixX<> bres(1, mem.MTX.data + st_busy, mem.MTXI_ROW.data + st_busy + mtxishift, mem.MTXI_COL.data + st_busy + mtxishift);
    mem.busy_mtx_parts = st_busy + 1;
    bres.data[0] = llU;
    bres.stRow[0] = 0; bres.stRow[1] = dimop;
    bres.stCol[0] = 0; bres.stCol[1] = n10;
    return bres;
}

OpMemoryRequirements complement_mem_op(const ComplementFemSpace& sp, uint nquadpoints, uint fusion,
    OpMemoryRequirements(BaseFemSpace::*req)(uint, uint) const, uint out_dim){
    auto v1 = (sp.m_V1.get()->*req)(nquadpoints, fusion);
    OpMemoryRequirements out = v1;
    const std::size_t n1 = sp.m_V1->m_order.NumDofOnTet();
    const std::size_t n10 = sp.m_order.NumDofOnTet();
    out.Usz = static_cast<std::size_t>(out_dim) * nquadpoints * fusion * n10;
    out.extraRsz = v1.extraRsz + v1.Usz + static_cast<std::size_t>(out_dim) * nquadpoints * fusion * n1;
    out.extraIsz = v1.extraIsz;
    out.mtx_parts = v1.mtx_parts + 1;
    return out;
}

std::shared_ptr<BaseFemSpace> complement_base_opt(
    std::shared_ptr<BaseFemSpace> V1,
    std::shared_ptr<BaseFemSpace> V0,
    ComplementFemSpace::Orthogonality orth,
    const double* D,
    uint quad_order);

std::vector<std::shared_ptr<BaseFemSpace>> collect_product_factors(const std::shared_ptr<BaseFemSpace>& V){
    std::vector<std::shared_ptr<BaseFemSpace>> leaves;
    if (!V) return leaves;
    if (V->gatherType() == BaseFemSpace::BaseTypes::ComplexType){
        auto cv = std::static_pointer_cast<ComplexFemSpace>(V);
        for (const auto& p : cv->m_spaces){
            auto sub = collect_product_factors(p);
            leaves.insert(leaves.end(), sub.begin(), sub.end());
        }
        return leaves;
    }
    leaves.push_back(V);
    return leaves;
}

std::shared_ptr<BaseFemSpace> try_dim_matched_factor(
    const std::shared_ptr<BaseFemSpace>& V1,
    const std::shared_ptr<BaseFemSpace>& V0,
    ComplementFemSpace::Orthogonality orth,
    uint quad_order)
{
    auto leaves1 = collect_product_factors(V1);
    auto leaves0 = collect_product_factors(V0);
    if (leaves1.size() <= 1 && leaves0.size() <= 1)
        return nullptr;

    std::vector<std::shared_ptr<BaseFemSpace>> parts;
    std::size_t i1 = 0, i0 = 0;
    while (i1 < leaves1.size() && i0 < leaves0.size()){
        const uint d1 = leaves1[i1]->dim();
        const uint d0 = leaves0[i0]->dim();
        if (d1 == 0 || d0 == 0)
            return nullptr;

        if (d1 == d0){
            parts.push_back(complement_base_opt(leaves1[i1], leaves0[i0], orth, nullptr, quad_order));
            ++i1; ++i0;
            continue;
        }

        if (d1 > d0){
            if (d1 % d0 != 0) return nullptr;
            const uint k = d1 / d0;
            if (i0 + k > leaves0.size()) return nullptr;
            for (uint t = 1; t < k; ++t)
                if (leaves0[i0 + t]->dim() != d0 || !(*leaves0[i0 + t] == *leaves0[i0]))
                    return nullptr;
            if (leaves1[i1]->gatherType() == BaseFemSpace::BaseTypes::VectorType){
                auto& rv = *std::static_pointer_cast<VectorFemSpace>(leaves1[i1]);
                if (rv.vec_dim() == k && rv.m_base && rv.m_base->dim() == d0){
                    parts.push_back(std::make_shared<VectorFemSpace>(
                        static_cast<int>(k), complement_base_opt(rv.m_base, leaves0[i0], orth, nullptr, quad_order)));
                    ++i1;
                    i0 += k;
                    continue;
                }
            }
            auto L0k = (k == 1) ? leaves0[i0]
                : std::shared_ptr<BaseFemSpace>(std::make_shared<VectorFemSpace>(static_cast<int>(k), leaves0[i0]));
            parts.push_back(complement_base_opt(leaves1[i1], L0k, orth, nullptr, quad_order));
            ++i1;
            i0 += k;
            continue;
        }

        if (d0 % d1 != 0) return nullptr;
        const uint k = d0 / d1;
        if (i1 + k > leaves1.size()) return nullptr;
        for (uint t = 1; t < k; ++t)
            if (leaves1[i1 + t]->dim() != d1 || !(*leaves1[i1 + t] == *leaves1[i1]))
                return nullptr;
        if (leaves0[i0]->gatherType() == BaseFemSpace::BaseTypes::VectorType){
            auto& rv = *std::static_pointer_cast<VectorFemSpace>(leaves0[i0]);
            if (rv.vec_dim() == k && rv.m_base && rv.m_base->dim() == d1){
                parts.push_back(std::make_shared<VectorFemSpace>(
                    static_cast<int>(k), complement_base_opt(leaves1[i1], rv.m_base, orth, nullptr, quad_order)));
                i1 += k;
                ++i0;
                continue;
            }
        }
        auto R1k = std::make_shared<VectorFemSpace>(static_cast<int>(k), leaves1[i1]);
        parts.push_back(complement_base_opt(R1k, leaves0[i0], orth, nullptr, quad_order));
        i1 += k;
        ++i0;
    }
    if (i1 != leaves1.size() || i0 != leaves0.size())
        return nullptr;
    if (parts.size() == 1)
        return parts[0];
    return std::make_shared<ComplexFemSpace>(std::move(parts));
}

std::shared_ptr<BaseFemSpace> try_complex_factor(
    const std::shared_ptr<BaseFemSpace>& V1,
    const std::shared_ptr<BaseFemSpace>& V0,
    ComplementFemSpace::Orthogonality orth,
    uint quad_order)
{
    if (V1->gatherType() != BaseFemSpace::BaseTypes::ComplexType
        || V0->gatherType() != BaseFemSpace::BaseTypes::ComplexType)
        return nullptr;
    auto& a = *std::static_pointer_cast<ComplexFemSpace>(V1);
    auto& b = *std::static_pointer_cast<ComplexFemSpace>(V0);
    if (a.m_spaces.size() != b.m_spaces.size())
        return nullptr;
    std::vector<std::shared_ptr<BaseFemSpace>> parts;
    parts.reserve(a.m_spaces.size());
    for (std::size_t i = 0; i < a.m_spaces.size(); ++i) {
        if (a.m_spaces[i]->dim() != b.m_spaces[i]->dim())
            return nullptr;
        parts.push_back(complement_base_opt(a.m_spaces[i], b.m_spaces[i], orth, nullptr, quad_order));
    }
    return std::make_shared<ComplexFemSpace>(std::move(parts));
}

} // namespace

ComplementFemSpace ComplementFemSpace::make(
    std::shared_ptr<BaseFemSpace> V1,
    std::shared_ptr<BaseFemSpace> V0,
    Orthogonality orth,
    const double* D_in,
    uint quad_order)
{
    if (!V1 || !V0)
        throw std::runtime_error("ComplementFemSpace::make: invalid input spaces");
    if (V1->dim() != V0->dim())
        throw std::runtime_error("ComplementFemSpace::make: spaces must have same dimension");
    V1 = flatten_to_scalar_complex(V1);
    V0 = flatten_to_scalar_complex(V0);
    const int n1 = static_cast<int>(V1->m_order.NumDofOnTet());
    const int n0 = static_cast<int>(V0->m_order.NumDofOnTet());
    if (n1 < n0)
        throw std::runtime_error("ComplementFemSpace::make: V1 must have no fewer dofs than V0");
    const int n10 = n1 - n0;
    if (n10 == 0)
        throw std::runtime_error("ComplementFemSpace::make: complement dimension is zero");

    const bool energy = (orth == EnergyComplement);
    const int gdim = static_cast<int>(V1->dim());
    const int gdim3 = 3 * gdim;
    std::vector<double> D;
    if (energy){
        D.assign(static_cast<std::size_t>(gdim3) * gdim3, 0.0);
        if (D_in){
            std::copy(D_in, D_in + gdim3 * gdim3, D.data());
#ifndef NDEBUG
            assert(is_symmetric_D(D.data(), gdim3)
                && "ComplementFemSpace::make: D must be symmetric");
#endif
        } else {
            for (int i = 0; i < gdim3; ++i)
                D[i + i * gdim3] = 1.0;
        }
    }

    if (quad_order == 0){
        uint o = V1->order();
        quad_order = (o < std::numeric_limits<uint>::max()) ? (2*o + 1) : 5;
    }
    auto formula = tetrahedron_quadrature_formulas(static_cast<int>(quad_order));
    const uint nquad = static_cast<uint>(formula.GetNumPoints());
    std::vector<double> xyl(4*nquad), wg(nquad);
    auto pp = formula.GetPointData();
    auto wp = formula.GetWeightData();
    std::copy(pp, pp + 4*nquad, xyl.data());
    std::copy(wp, wp + nquad, wg.data());

    PlainMemoryX<> pmx_v1_iden, pmx_v0_iden, pmx_v1_grad;
    setup_animem(pmx_v1_iden, *V1, nquad, &BaseFemSpace::memIDEN);
    setup_animem(pmx_v0_iden, *V0, nquad, &BaseFemSpace::memIDEN);
    if (energy)
        setup_animem(pmx_v1_grad, *V1, nquad, &BaseFemSpace::memGRAD);
    const auto sz1 = pmx_v1_iden.enoughRawSize();
    const auto sz0 = pmx_v0_iden.enoughRawSize();
    const auto szg = energy ? pmx_v1_grad.enoughRawSize() : 0;
    std::vector<char> buf(sz1 + sz0 + szg);
    pmx_v1_iden.allocateFromRaw(buf.data(), sz1);
    pmx_v0_iden.allocateFromRaw(buf.data() + sz1, sz0);
    if (energy)
        pmx_v1_grad.allocateFromRaw(buf.data() + sz1 + sz0, szg);

    AniMemoryX<> mem_v1, mem_v0, mem_grad;
    init_animem(mem_v1, nquad);
    init_animem(mem_v0, nquad);
    alloc_animem_from_pmx(mem_v1, pmx_v1_iden, *V1, nquad, &BaseFemSpace::memIDEN);
    alloc_animem_from_pmx(mem_v0, pmx_v0_iden, *V0, nquad, &BaseFemSpace::memIDEN);
    if (energy){
        init_animem(mem_grad, nquad);
        alloc_animem_from_pmx(mem_grad, pmx_v1_grad, *V1, nquad, &BaseFemSpace::memGRAD);
    }

    double XYZa_corner[12]{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    double one = 1;
    Tetra<const double> XYZ(XYZa_corner+0, XYZa_corner+3, XYZa_corner+6, XYZa_corner+9);
    mem_v1.XYP.Init(XYZa_corner, 12);
    mem_v1.PSI.Init(XYZa_corner+3, 9);
    mem_v1.DET.Init(&one, 1);
    mem_v0.XYP = mem_v1.XYP;
    mem_v0.PSI = mem_v1.PSI;
    mem_v0.DET = mem_v1.DET;

    // Must outlive mem_grad usage (AniMemory holds non-owning views).
    const double s3 = std::sqrt(3.0), s6 = std::sqrt(6.0);
    double XYZa_reg[12]{
        0, 0, 0,
        1, 0, 0,
        0.5, s3 / 2, 0,
        0.5, s3 / 6, s6 / 3
    };
    double PSI_reg[9], DET_reg = 1;
    if (energy){
        DET_reg = inverse3x3(XYZa_reg + 3, PSI_reg);
        mem_grad.XYP.Init(XYZa_reg, 12);
        mem_grad.PSI.Init(PSI_reg, 9);
        mem_grad.DET.Init(&DET_reg, 1);
    }

    std::vector<double> M1(n1*n1), C(n0*n1);
    build_mass_matrix(*V1, nquad, xyl.data(), wg.data(), mem_v1, M1.data(), n1);
    PlainMemoryX<> pmx_interp;
    build_embedding_matrix(*V0, *V1, XYZ, mem_v0, pmx_interp, C.data(), n0, n1, quad_order);

    std::vector<double> S;
    int nS = n0;
    const double* dual_metric = nullptr;

    if (energy){
        std::vector<double> A(n1*n1), Mconst(gdim * n1);
        build_energy_matrix(*V1, nquad, xyl.data(), wg.data(), mem_grad, D.data(), gdim3, A.data(), n1);
        build_mean_rows(*V1, nquad, xyl.data(), wg.data(), mem_v1, Mconst.data(), n1, gdim);

#ifndef NDEBUG
        assert_A_S4_invariant(*V1, A.data(), n1);
        assert_V0_subspace(*V0, *V1, nquad, xyl.data(), wg.data(), mem_v0, mem_v1, C.data(), M1.data(), n0, n1);
        assert_P0_in_V0(*V0, nquad, xyl.data(), wg.data(), mem_v0, XYZ, quad_order);
#endif

        nS = n0 + gdim;
        std::vector<double> CA(n0 * n1);
        S.assign(static_cast<std::size_t>(nS) * n1, 0.0);
        matmul(C.data(), n0, n1, A.data(), n1, CA.data());
        for (int j = 0; j < n1; ++j){
            for (int i = 0; i < n0; ++i)
                S[i + j * nS] = CA[i + j * n0];
            for (int d = 0; d < gdim; ++d)
                S[n0 + d + j * nS] = Mconst[d + j * gdim];
        }
        for (int i = 0; i < nS; ++i){
            double nrm = 0;
            for (int j = 0; j < n1; ++j)
                nrm += S[i + j * nS] * S[i + j * nS];
            nrm = std::sqrt(nrm);
            if (nrm <= 0) continue;
            const double inv = 1.0 / nrm;
            for (int j = 0; j < n1; ++j)
                S[i + j * nS] *= inv;
        }
        dual_metric = M1.data();
    } else {
#ifndef NDEBUG
        assert_V0_subspace(*V0, *V1, nquad, xyl.data(), wg.data(), mem_v0, mem_v1, C.data(), M1.data(), n0, n1);
#endif
        S.assign(static_cast<std::size_t>(n0) * n1, 0.0);
        matmul(C.data(), n0, n1, M1.data(), n1, S.data());
    }

    const double smax = svd_max_sigma(S.data(), nS, n1, energy);
    const double svd_tol = std::max(kDefaultTol, kDefaultTol * std::max(1.0, smax));
    auto ker_basis = nullspace_right(S.data(), nS, n1, svd_tol, energy);
    if (static_cast<int>(ker_basis.size()) != n10)
        throw std::runtime_error("ComplementFemSpace::make: kernel of S has wrong dimension");

    auto built = build_equivariant_complement_basis(
        *V1, *V0, C.data(), n0, n1, ker_basis, M1.data(), n10, svd_tol,
        "ComplementFemSpace::make", dual_metric);

    ComplementFemSpace res;
    res.m_orth = orth;
    res.m_V1 = std::move(V1);
    res.m_V0 = std::move(V0);
    res.m_basis_coefs = std::move(built.basis_coefs);
    res.m_dual_coefs = std::move(built.dual_coefs);
    res.m_dof_tags = std::move(built.dof_tags);
    res.m_order = DofT::DofMap(std::make_shared<DofT::UniteDofMap>(std::move(built.dof_map)));
    if (energy && !is_identity_D(D_in, gdim3))
        res.m_D = std::move(D);
    return res;
}

namespace {

std::shared_ptr<BaseFemSpace> complement_base_opt(
    std::shared_ptr<BaseFemSpace> V1,
    std::shared_ptr<BaseFemSpace> V0,
    ComplementFemSpace::Orthogonality orth,
    const double* D,
    uint quad_order)
{
    if (!V1 || !V0)
        throw std::runtime_error("ComplementFemSpace::make: invalid input spaces");
    if (V1->dim() != V0->dim())
        throw std::runtime_error("ComplementFemSpace::make: spaces must have same field dimension");

    const bool energy = (orth == ComplementFemSpace::EnergyComplement);
    const int gdim3 = static_cast<int>(3 * V1->dim());
    const bool factorable = !energy || is_identity_D(D, gdim3);

    if (factorable){
        const auto t1 = V1->gatherType();
        const auto t0 = V0->gatherType();
        if (t1 == BaseFemSpace::BaseTypes::VectorType && t0 == BaseFemSpace::BaseTypes::VectorType) {
            auto& a = *std::static_pointer_cast<VectorFemSpace>(V1);
            auto& b = *std::static_pointer_cast<VectorFemSpace>(V0);
            if (a.vec_dim() == b.vec_dim())
                return std::make_shared<VectorFemSpace>(
                    static_cast<int>(a.vec_dim()),
                    complement_base_opt(a.m_base, b.m_base, orth, nullptr, quad_order));
        }
        if (auto factored = try_complex_factor(V1, V0, orth, quad_order))
            return factored;
        if (auto factored = try_dim_matched_factor(V1, V0, orth, quad_order))
            return factored;

        auto f1 = flatten_to_scalar_complex(V1);
        auto f0 = flatten_to_scalar_complex(V0);
        if (f1.get() != V1.get() || f0.get() != V0.get()) {
            if (auto factored = try_complex_factor(f1, f0, orth, quad_order))
                return factored;
            if (auto factored = try_dim_matched_factor(f1, f0, orth, quad_order))
                return factored;
        }
        return std::make_shared<ComplementFemSpace>(
            ComplementFemSpace::make(std::move(f1), std::move(f0), orth, nullptr, quad_order));
    }

    return std::make_shared<ComplementFemSpace>(
        ComplementFemSpace::make(std::move(V1), std::move(V0), orth, D, quad_order));
}

} // namespace

FemSpace ComplementFemSpace::make(const FemSpace& V1, const FemSpace& V0,
    Orthogonality orth, const double* D, uint quad_order)
{
    if (!V1.isValid() || !V0.isValid())
        throw std::runtime_error("ComplementFemSpace::make: invalid FemSpace operands");
    return FemSpace(complement_base_opt(V1.base(), V0.base(), orth, D, quad_order));
}

std::shared_ptr<BaseFemSpace> ComplementFemSpace::subSpace(const int* ext_dims, int ndims) const {
    (void)ext_dims; (void)ndims;
    return nullptr;
}

std::shared_ptr<BaseFemSpace> ComplementFemSpace::copy() const {
    return std::make_shared<ComplementFemSpace>(*this);
}

std::string ComplementFemSpace::typeName() const {
    return "Complement(" + (m_V1 ? m_V1->typeName() : "?") + " - " + (m_V0 ? m_V0->typeName() : "?") + ")";
}

bool ComplementFemSpace::operator==(const BaseFemSpace& other) const {
    if (gatherType() != other.gatherType()) return false;
    auto& a = *static_cast<const ComplementFemSpace*>(&other);
    if (m_orth != a.m_orth) return false;
    if (m_V1.get() == a.m_V1.get() && m_V0.get() == a.m_V0.get() && m_D == a.m_D) return true;
    return m_V1 && a.m_V1 && *m_V1 == *a.m_V1 && m_V0 && a.m_V0 && *m_V0 == *a.m_V0
        && m_basis_coefs == a.m_basis_coefs && m_dual_coefs == a.m_dual_coefs && m_D == a.m_D;
}

void ComplementFemSpace::evalBasisFunctions(const Expr& lmb, const Expr& grad_lmb, Expr* phi) const {
    const int n1 = static_cast<int>(m_V1->m_order.NumDofOnTet());
    const int n10 = static_cast<int>(m_order.NumDofOnTet());
    std::vector<Expr> phi1(n1);
    m_V1->evalBasisFunctions(lmb, grad_lmb, phi1.data());
    for (int i = 0; i < n10; ++i)
        phi[i] = FT::scalsum(phi1.data(), phi1.data() + n1, m_basis_coefs.data() + i*n1);
}

PlainMemoryX<> ComplementFemSpace::interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const {
    (void)idof_on_tet;
    PlainMemoryX<> req;
    const int n1 = static_cast<int>(m_V1->m_order.NumDofOnTet());
    for (int j = 0; j < n1; ++j){
        auto lreq = m_V1->interpolateOnDOF_mem_req(j, fusion, max_quad_order);
        req.dSize = std::max(req.dSize, lreq.dSize);
        req.iSize = std::max(req.iSize, lreq.iSize);
        req.mSize = std::max(req.mSize, lreq.mSize);
    }
    req.dSize += fusion;
    return req;
}

void ComplementFemSpace::interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data, uint max_quad_order) const {
    const int n1 = static_cast<int>(m_V1->m_order.NumDofOnTet());
    std::vector<ArrayView<>> v1_udofs(fusion);
    std::vector<std::vector<double>> val(fusion, std::vector<double>(n1, 0.0));
    for (int r = 0; r < fusion; ++r){
        v1_udofs[r] = ArrayView<>(val[r].data(), n1);
        udofs[r].data[idof_on_tet] = 0.0;
    }
    double* d0 = mem.ddata;
    int* i0 = mem.idata;
    DenseMatrix<>* m0 = mem.mdata;
    for (int j = 0; j < n1; ++j){
        for (int r = 0; r < fusion; ++r)
            std::fill(val[r].begin(), val[r].end(), 0.0);
        mem.ddata = d0;
        mem.idata = i0;
        mem.mdata = m0;
        m_V1->interpolateOnDOF(XYZ, f, v1_udofs.data(), j, fusion, mem, user_data, max_quad_order);
        for (int r = 0; r < fusion; ++r)
            udofs[r].data[idof_on_tet] += m_dual_coefs[idof_on_tet*n1 + j] * val[r][j];
    }
}

BandDenseMatrixX<> ComplementFemSpace::applyIDEN(AniMemoryX<>& mem, ArrayView<>& U) const {
    return apply_op_internal(*this, mem, U,
        [this](AniMemoryX<>& m, ArrayView<>& u){ return m_V1->applyIDEN(m, u); },
        m_V1->memIDEN(mem.q, mem.f));
}

OpMemoryRequirements ComplementFemSpace::memIDEN(uint nquadpoints, uint fusion) const {
    return complement_mem_op(*this, nquadpoints, fusion, &BaseFemSpace::memIDEN, dim());
}

BandDenseMatrixX<> ComplementFemSpace::applyGRAD(AniMemoryX<>& mem, ArrayView<>& U) const {
    return apply_op_internal(*this, mem, U,
        [this](AniMemoryX<>& m, ArrayView<>& u){ return m_V1->applyGRAD(m, u); },
        m_V1->memGRAD(mem.q, mem.f));
}

OpMemoryRequirements ComplementFemSpace::memGRAD(uint nquadpoints, uint fusion) const {
    return complement_mem_op(*this, nquadpoints, fusion, &BaseFemSpace::memGRAD, dimGRAD());
}

BandDenseMatrixX<> ComplementFemSpace::applyDUDX(AniMemoryX<>& mem, ArrayView<>& U, uchar k) const {
    return apply_op_internal(*this, mem, U,
        [this, k](AniMemoryX<>& m, ArrayView<>& u){ return m_V1->applyDUDX(m, u, k); },
        m_V1->memDUDX(mem.q, mem.f, k));
}

OpMemoryRequirements ComplementFemSpace::memDUDX(uint nquadpoints, uint fusion, uchar k) const {
    auto v1 = m_V1->memDUDX(nquadpoints, fusion, k);
    OpMemoryRequirements out = v1;
    const std::size_t n1 = m_V1->m_order.NumDofOnTet();
    const std::size_t n10 = m_order.NumDofOnTet();
    out.Usz = static_cast<std::size_t>(dim()) * nquadpoints * fusion * n10;
    out.extraRsz = v1.extraRsz + v1.Usz + static_cast<std::size_t>(dim()) * nquadpoints * fusion * n1;
    out.mtx_parts = v1.mtx_parts + 1;
    return out;
}

FemSpace operator-(const FemSpace& V1, const FemSpace& V0){
    if (!V1.isValid() || !V0.isValid())
        throw std::runtime_error("operator-: invalid FemSpace operands");
    return FemSpace(complement_base_opt(V1.base(), V0.base(),
        ComplementFemSpace::L2Complement, nullptr, 0));
}

} // namespace Ani
