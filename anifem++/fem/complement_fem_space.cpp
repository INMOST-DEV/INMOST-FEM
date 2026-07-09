#include "complement_fem_space.h"
#include "quadrature_formulas.h"
#include "operations/operations.h"
#include <cassert>
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <map>
#include <limits>

namespace Ani {
namespace {

using DofT::uchar;

constexpr double kDefaultTol = 1e-12;

void setup_animem(PlainMemoryX<>& pmx, const BaseFemSpace& space, uint nquad){
    auto req = space.memIDEN(nquad, 1);
    pmx.dSize += req.Usz + req.extraRsz;
    pmx.iSize += req.extraIsz + 2*(req.mtx_parts + 2);
    pmx.mSize += req.mtx_parts;
}

void init_animem(AniMemoryX<>& mem, PlainMemoryX<>& pmx, uint nquad){
    mem.q = nquad;
    mem.f = 1;
    mem.busy_mtx_parts = 0;
    mem.U.Init(pmx.ddata, 0);
    mem.extraR.Init(nullptr, 0);
    mem.extraI.Init(nullptr, 0);
    mem.MTX.Init(nullptr, 0);
    mem.MTXI_COL.Init(nullptr, 0);
    mem.MTXI_ROW.Init(nullptr, 0);
}

void alloc_animem_from_pmx(AniMemoryX<>& mem, PlainMemoryX<>& pmx, const BaseFemSpace& space, uint nquad){
    auto req = space.memIDEN(nquad, 1);
    mem.U.Init(pmx.ddata, req.Usz); pmx.ddata += req.Usz; pmx.dSize -= req.Usz;
    mem.extraR.Init(pmx.ddata, req.extraRsz); pmx.ddata += req.extraRsz; pmx.dSize -= req.extraRsz;
    mem.extraI.Init(pmx.idata, req.extraIsz); pmx.idata += req.extraIsz; pmx.iSize -= req.extraIsz;
    mem.MTXI_COL.Init(pmx.idata, req.mtx_parts+2); pmx.idata += req.mtx_parts+2; pmx.iSize -= req.mtx_parts+2;
    mem.MTXI_ROW.Init(pmx.idata, req.mtx_parts+2); pmx.idata += req.mtx_parts+2; pmx.iSize -= req.mtx_parts+2;
    mem.MTX.Init(pmx.mdata, req.mtx_parts); pmx.mdata += req.mtx_parts; pmx.mSize -= req.mtx_parts;
}

void matmul(const double* A, int n, int m, const double* B, int p, double* C){
    std::fill(C, C + n*p, 0.0);
    for (int j = 0; j < p; ++j)
        for (int k = 0; k < m; ++k)
            for (int i = 0; i < n; ++i)
                C[i + j*n] += A[i + k*n] * B[k + j*m];
}

int svd_rank(const double* s, int k, double tol){
    int r = 0;
    for (int i = 0; i < k; ++i)
        if (s[i] > tol) ++r;
    return r;
}

std::vector<std::vector<double>> nullspace_right(const double* A, int n, int m, double tol){
    const int k = std::min(n, m);
    std::vector<double> U(n*n), s(k), V(m*m);
    std::vector<double> mem(2*k*k + k);
    if (jacobiSVD(A, n, m, U.data(), s.data(), V.data(), mem.data()))
        throw std::runtime_error("jacobiSVD failed in ComplementFemSpace::make");
    int rank = svd_rank(s.data(), k, tol);
    int nd = m - rank;
    std::vector<std::vector<double>> basis(nd, std::vector<double>(m, 0.0));
    for (int b = 0; b < nd; ++b)
        for (int i = 0; i < m; ++i)
            basis[b][i] = V[i + (rank + b)*m];
    return basis;
}

/// Evaluate basis at quadrature: phi[q + nquad*i + nfa*nquad*d] = component d of basis i.
void evaluate_basis_at_quad(const BaseFemSpace& space, AniMemoryX<>& mem,
                            const double* xyl, const double* wg, uint nquad,
                            double* phi /* nfa * nquad * gdim */){
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

/// Cross mass M01[i,j] = ∫ φ0_i · φ1_j  (column-major n0 × n1).
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

/// True iff V0 ⊂ V1: embedding C satisfies M01 ≈ C M1 and rank(C) = n0.
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
    return svd_rank(s.data(), k, rank_tol) == n0;
}

struct BasisFunctorData {
    const BaseFemSpace* space;
    AniMemoryX<>* mem;
    int idof;
    int nfa;
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
                            AniMemoryX<>& mem_v0, AniMemoryX<>& mem_v1, PlainMemoryX<>& pmx,
                            double* C, int n0, int n1, uint max_quad_order){
    BasisFunctorData fd{&V0, &mem_v0, 0, static_cast<int>(V0.m_order.NumDofOnTet())};
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

/// Infer DofSymmetries from tags; remap incomplete symmetry volumes to stype 0.
/// Rewrite tags (leid/stype/lsid/gid) to match UniteDofMap ordering.
DofT::UniteDofMap complement_dof_map_from_tags(std::vector<DofT::LocalOrder>& tags){
    DofT::DofSymmetries sym;
    sym.set({0}, {0, 0}, {0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0, 0, 0});

    std::array<uint, DofT::NGEOM_TYPES> dofs_per_elem{};
    std::array<std::array<uint, 8>, DofT::NGEOM_TYPES> stype_hist{};
    std::array<bool, DofT::NGEOM_TYPES> seen{};
    for (const auto& lo : tags){
        auto t = DofT::GeomTypeToNum(lo.etype);
        if (t < 0) continue;
        uint cnt = 0;
        std::array<uint, 8> hist{};
        for (const auto& o : tags){
            if (o.etype == lo.etype && o.nelem == lo.nelem){
                ++cnt;
                if (o.stype < 8) ++hist[o.stype];
            }
        }
        if (!seen[t] || cnt > dofs_per_elem[t]){
            dofs_per_elem[t] = cnt;
            stype_hist[t] = hist;
            seen[t] = true;
        }
    }

    for (int t = 0; t < DofT::NGEOM_TYPES; ++t){
        if (dofs_per_elem[t] == 0) continue;
        const uchar etype = DofT::NumToGeomType(t);
        const uchar nsym = DofT::DofSymmetries::symmetries_amount(etype);
        bool ok = true;
        uint reconstructed = 0;
        std::array<uint, 8> counts{};
        for (uchar s = 0; s < nsym; ++s){
            const uchar vol = DofT::DofSymmetries::symmetry_volume(etype, s);
            if (stype_hist[t][s] % vol != 0){ ok = false; break; }
            counts[s] = stype_hist[t][s] / vol;
            reconstructed += counts[s] * vol;
        }
        if (!ok || reconstructed != dofs_per_elem[t]){
            for (uchar s = 0; s < nsym; ++s) counts[s] = 0;
            counts[0] = dofs_per_elem[t];
        }
        for (uchar s = 0; s < nsym; ++s)
            if (counts[s] > 0)
                sym.add(etype, s, counts[s]);
    }

    DofT::UniteDofMap map(sym);

    // Assign leid within each geometric element in a stable order, then set stype/lsid/gid from the map.
    std::vector<int> order(tags.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){
        const auto& ta = tags[a];
        const auto& tb = tags[b];
        if (ta.etype != tb.etype) return DofT::GeomTypeToNum(ta.etype) < DofT::GeomTypeToNum(tb.etype);
        if (ta.nelem != tb.nelem) return ta.nelem < tb.nelem;
        if (ta.stype != tb.stype) return ta.stype < tb.stype;
        if (ta.lsid != tb.lsid) return ta.lsid < tb.lsid;
        return a < b;
    });
    std::map<std::pair<uchar, uchar>, uint> local_leid;
    for (int idx : order){
        auto& lo = tags[idx];
        uint leid = local_leid[{lo.etype, lo.nelem}]++;
        auto lso = sym.GetLocSymOrder(lo.etype, leid);
        lo.leid = leid;
        lo.stype = lso.stype;
        lo.lsid = lso.lsid;
        lo.gid = map.TetDofID(lo.getGeomOrder());
    }
    return map;
}

void sort_dof_tags(std::vector<DofT::LocalOrder>& tags, std::vector<std::vector<double>>& coefs){
    std::vector<int> order(tags.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){
        if (tags[a].gid != tags[b].gid) return tags[a].gid < tags[b].gid;
        return a < b;
    });
    std::vector<DofT::LocalOrder> st(tags.size());
    std::vector<std::vector<double>> sc(coefs.size());
    for (std::size_t i = 0; i < order.size(); ++i){
        st[i] = tags[order[i]];
        sc[i] = std::move(coefs[order[i]]);
    }
    tags = std::move(st);
    coefs = std::move(sc);
}

double m1_inner(const double* a, const double* b, const double* M1, int n1){
    double s = 0;
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n1; ++j)
            s += a[i] * M1[i + j*n1] * b[j];
    return s;
}

void m1_orthogonalize(std::vector<double>& w, const std::vector<std::vector<double>>& accepted,
    const double* M1, int n1, double tol){
    for (const auto& u : accepted){
        double un2 = m1_inner(u.data(), u.data(), M1, n1);
        if (un2 <= tol*tol) continue;
        double c = m1_inner(w.data(), u.data(), M1, n1) / un2;
        for (int i = 0; i < n1; ++i)
            w[i] -= c * u[i];
    }
}

bool same_geom_dof(const DofT::LocalOrder& a, const DofT::LocalOrder& b){
    return a.etype == b.etype && a.nelem == b.nelem && a.stype == b.stype;
}

bool dof_geom_in_V0(const BaseFemSpace& V0, const DofT::LocalOrder& lo, int n0){
    for (int i = 0; i < n0; ++i){
        auto lo0 = V0.m_order.LocalOrderOnTet(DofT::TetOrder(i));
        if (same_geom_dof(lo0, lo)) return true;
    }
    return false;
}

struct SymTypeKey {
    uchar etype = 0;
    bool operator<(const SymTypeKey& o) const {
        return DofT::GeomTypeToNum(etype) < DofT::GeomTypeToNum(o.etype);
    }
};

void apply_perm(const double* P, int n, const double* x, double* y){
    std::fill(y, y + n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            y[i] += P[i + j * n] * x[j];
}

int find_perm_index(const std::array<std::array<uchar, 4>, 24>& perms, const std::array<uchar, 4>& sigma){
    for (int i = 0; i < 24; ++i)
        if (perms[i] == sigma) return i;
    throw std::runtime_error("ComplementFemSpace::make: permutation not found in S4 table");
}

std::array<uchar, 4> compose_perm(const std::array<uchar, 4>& a, const std::array<uchar, 4>& b){
    return {a[b[0]], a[b[1]], a[b[2]], a[b[3]]};
}

std::vector<int> close_stabilizer(const std::vector<std::array<uchar, 4>>& generators,
    const std::array<std::array<uchar, 4>, 24>& all_perms){
    std::vector<std::array<uchar, 4>> group;
    group.push_back({0, 1, 2, 3});
    for (const auto& g : generators)
        group.push_back(g);
    bool changed = true;
    while (changed){
        changed = false;
        const int n = static_cast<int>(group.size());
        for (int i = 0; i < n && static_cast<int>(group.size()) < 24; ++i)
            for (int j = 0; j < n && static_cast<int>(group.size()) < 24; ++j){
                auto h = compose_perm(group[i], group[j]);
                bool found = false;
                for (const auto& x : group)
                    if (x == h){ found = true; break; }
                if (!found){ group.push_back(h); changed = true; }
            }
    }
    std::vector<int> idx;
    idx.reserve(group.size());
    for (const auto& g : group)
        idx.push_back(find_perm_index(all_perms, g));
    return idx;
}

void project_onto_ker(const std::vector<std::vector<double>>& ker, int j, std::vector<double>& w){
    const int n1 = static_cast<int>(ker[0].size());
    w.assign(n1, 0.0);
    for (const auto& row : ker){
        double c = row[j];
        for (int i = 0; i < n1; ++i)
            w[i] += c * row[i];
    }
}

uchar stype_for_count(uchar etype, uint n_comp){
    const uchar nsym = DofT::DofSymmetries::symmetries_amount(etype);
    for (uchar s = 0; s < nsym; ++s)
        if (DofT::DofSymmetries::symmetry_volume(etype, s) == n_comp)
            return s;
    return 0;
}

/// Local representation matrix Q(g) for volume-n_comp stype.
void local_rep_matrix(uchar etype, uchar nelem, uchar stype, int n_comp,
    const uchar g[4], double* Q){
    std::fill(Q, Q + n_comp * n_comp, 0.0);
    for (int ell = 0; ell < n_comp; ++ell){
        uchar new_lsid = DofT::DofSymmetries::index_on_reorderd_elem(
            etype, nelem, stype, static_cast<uchar>(ell), g);
        if (new_lsid < static_cast<uchar>(n_comp))
            Q[new_lsid + ell * n_comp] = 1.0;
        else
            Q[ell + ell * n_comp] = 1.0;
    }
}

/// Matrix Reynolds: W <- (1/|G|) sum_g P_g W Q(g)^T.
void matrix_reynolds(std::vector<std::vector<double>>& W, int n1, int n_comp,
    const std::vector<int>& stab_idx, const std::vector<std::vector<double>>& P_cache,
    uchar etype, uchar nelem, uchar stype,
    const std::array<std::array<uchar, 4>, 24>& all_perms){
    if (W.empty() || n_comp == 0) return;
    std::vector<std::vector<double>> acc(n_comp, std::vector<double>(n1, 0.0));
    std::vector<double> Q(n_comp * n_comp), tmp(n1);
    const double inv = 1.0 / static_cast<double>(stab_idx.size());
    for (int pi : stab_idx){
        local_rep_matrix(etype, nelem, stype, n_comp, all_perms[pi].data(), Q.data());
        for (int ell = 0; ell < n_comp; ++ell){
            apply_perm(P_cache[pi].data(), n1, W[ell].data(), tmp.data());
            for (int neu = 0; neu < n_comp; ++neu){
                double q = Q[neu + ell * n_comp];
                if (std::abs(q) < 1e-15) continue;
                for (int i = 0; i < n1; ++i)
                    acc[neu][i] += inv * q * tmp[i];
            }
        }
    }
    W = std::move(acc);
}

/// Equivariant M1-orthonormalization: W <- W * G^{-1/2} with the unique PD square root.
/// Using U S^{-1/2} (eigenbasis) would break Stab-equivariance; U S^{-1/2} U^T preserves it
/// whenever G already commutes with the local representation Q(g).
void equivariant_m1_orthonormalize(std::vector<std::vector<double>>& W, int n1, int n_comp,
    const double* M1, double tol){
    if (n_comp <= 0 || W.empty()) return;
    const int ncol = static_cast<int>(W.size());
    std::vector<double> G(ncol * ncol, 0.0);
    for (int j = 0; j < ncol; ++j)
        for (int i = 0; i < ncol; ++i)
            G[i + j * ncol] = m1_inner(W[i].data(), W[j].data(), M1, n1);

    std::vector<double> U(ncol * ncol), s(ncol), V(ncol * ncol), mem(2 * ncol * ncol + ncol);
    if (jacobiSVD(G.data(), ncol, ncol, U.data(), s.data(), V.data(), mem.data()))
        throw std::runtime_error("ComplementFemSpace::make: fiber Gram SVD failed");

    // G^{-1/2} = U diag(s_a^{-1/2}) U^T on the positive eigenspace.
    std::vector<double> Ginvsqrt(ncol * ncol, 0.0);
    int npos = 0;
    for (int a = 0; a < ncol; ++a){
        if (s[a] <= tol * tol)
            continue;
        ++npos;
        const double inv = 1.0 / std::sqrt(s[a]);
        for (int i = 0; i < ncol; ++i)
            for (int j = 0; j < ncol; ++j)
                Ginvsqrt[i + j * ncol] += inv * U[i + a * ncol] * U[j + a * ncol];
    }
    if (npos < n_comp)
        throw std::runtime_error("ComplementFemSpace::make: fiber Gram is rank-deficient");

    std::vector<std::vector<double>> Wnew(n_comp, std::vector<double>(n1, 0.0));
    for (int j = 0; j < n_comp; ++j)
        for (int b = 0; b < ncol; ++b){
            const double c = Ginvsqrt[b + j * ncol];
            if (std::abs(c) < 1e-15) continue;
            for (int i = 0; i < n1; ++i)
                Wnew[j][i] += c * W[b][i];
        }
    W = std::move(Wnew);
}

double svd_max_sigma(const double* S, int n0, int n1){
    const int k = std::min(n0, n1);
    std::vector<double> U(n0*n0), s(k), V(n1*n1), mem(2*k*k + k);
    if (jacobiSVD(S, n0, n1, U.data(), s.data(), V.data(), mem.data()))
        throw std::runtime_error("jacobiSVD failed in ComplementFemSpace::make");
    return k > 0 ? s[0] : 0.0;
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

} // namespace

ComplementFemSpace ComplementFemSpace::make(std::shared_ptr<BaseFemSpace> V1, std::shared_ptr<BaseFemSpace> V0, uint quad_order){
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

    PlainMemoryX<> pmx_v1, pmx_v0;
    setup_animem(pmx_v1, *V1, nquad);
    setup_animem(pmx_v0, *V0, nquad);
    const auto sz1 = pmx_v1.enoughRawSize();
    const auto sz2 = pmx_v0.enoughRawSize();
    std::vector<char> buf(sz1 + sz2);
    pmx_v1.allocateFromRaw(buf.data(), sz1);
    pmx_v0.allocateFromRaw(buf.data() + sz1, sz2);

    AniMemoryX<> mem_v1, mem_v0;
    init_animem(mem_v1, pmx_v1, nquad);
    init_animem(mem_v0, pmx_v0, nquad);
    alloc_animem_from_pmx(mem_v1, pmx_v1, *V1, nquad);
    alloc_animem_from_pmx(mem_v0, pmx_v0, *V0, nquad);

    double XYZa[12]{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    double one = 1;
    Tetra<const double> XYZ(XYZa+0, XYZa+3, XYZa+6, XYZa+9);
    mem_v1.XYP.Init(XYZa, 12);
    mem_v1.PSI.Init(XYZa+3, 9);
    mem_v1.DET.Init(&one, 1);
    mem_v0.XYP = mem_v1.XYP;
    mem_v0.PSI = mem_v1.PSI;
    mem_v0.DET = mem_v1.DET;
    mem_v0.XYL = mem_v1.XYL;
    mem_v0.WG = mem_v1.WG;

    std::vector<double> M1(n1*n1), C(n0*n1), S(n0*n1);
    build_mass_matrix(*V1, nquad, xyl.data(), wg.data(), mem_v1, M1.data(), n1);

    PlainMemoryX<> pmx_interp;
    build_embedding_matrix(*V0, *V1, XYZ, mem_v0, mem_v1, pmx_interp, C.data(), n0, n1, quad_order);

#ifndef NDEBUG
    {
        std::vector<double> M01(static_cast<std::size_t>(n0) * n1, 0.0);
        build_cross_mass_matrix(*V0, *V1, nquad, xyl.data(), wg.data(), mem_v0, mem_v1, M01.data(), n0, n1);
        assert(is_V0_subspace_of_V1(C.data(), M1.data(), M01.data(), n0, n1, 1e-8)
            && "ComplementFemSpace::make: V0 is not a subspace of V1");
    }
#endif

    matmul(C.data(), n0, n1, M1.data(), n1, S.data());
    const double smax = svd_max_sigma(S.data(), n0, n1);
    const double svd_tol = std::max(kDefaultTol, kDefaultTol * std::max(1.0, smax));

    auto ker_basis = nullspace_right(S.data(), n0, n1, svd_tol);
    if (static_cast<int>(ker_basis.size()) != n10)
        throw std::runtime_error("ComplementFemSpace::make: kernel of S has wrong dimension");

    // Cache all 24 S4 permutation matrices on V1 dofs.
    std::array<std::array<uchar, 4>, 24> all_perms;
    DofT::S4::all_permutations(all_perms);
    std::vector<std::vector<double>> P_cache(24, std::vector<double>(static_cast<std::size_t>(n1) * n1, 0.0));
    for (int p = 0; p < 24; ++p)
        DofT::S4::build_dof_permutation(*V1->m_order.base(), all_perms[p].data(), P_cache[p].data(), n1);

    // Unique geometric etypes present in V1 but not fully covered by V0.
    std::map<SymTypeKey, DofT::LocalOrder> type_repr;
    for (int j = 0; j < n1; ++j){
        auto lo = V1->m_order.LocalOrderOnTet(DofT::TetOrder(j));
        if (dof_geom_in_V0(*V0, lo, n0))
            continue;
        SymTypeKey key{lo.etype};
        auto it = type_repr.find(key);
        if (it == type_repr.end())
            type_repr.emplace(key, lo);
        else if (lo.nelem < it->second.nelem || (lo.nelem == it->second.nelem && lo.lsid < it->second.lsid))
            it->second = lo;
    }

    std::vector<std::vector<double>> collected_coefs;
    std::vector<DofT::LocalOrder> collected_tags;
    collected_coefs.reserve(n10);
    collected_tags.reserve(n10);
    std::vector<std::vector<double>> accepted_for_orth;

    for (const auto& kv : type_repr){
        const DofT::LocalOrder& repr0 = kv.second;
        const uchar etype = repr0.etype;
        const uint n_v1 = V1->m_order.NumDof(etype);
        const uint n_v0 = V0->m_order.NumDof(etype);
        if (n_v1 <= n_v0)
            continue;
        const int n_comp = static_cast<int>(n_v1 - n_v0);
        const uchar out_stype = stype_for_count(etype, static_cast<uint>(n_comp));

        DofT::LocalOrder repr = repr0;
        repr.nelem = 0;
        repr.stype = out_stype;
        auto stab = DofT::S4::stabilizer_data(repr);
        auto stab_idx = close_stabilizer(stab.generators, all_perms);

        std::vector<std::vector<double>> kept;

        if (n_comp == 1){
            // Scalar character: Reynolds-average ker-projections of seeds on element 0.
            std::vector<int> seed_dofs;
            for (int j = 0; j < n1; ++j){
                auto lo = V1->m_order.LocalOrderOnTet(DofT::TetOrder(j));
                if (lo.etype != etype || lo.nelem != 0)
                    continue;
                seed_dofs.push_back(j);
            }
            int used_flip = -1;
            for (int flip : {0, 1}){ // prefer sign isotype first (P3-P2 edge), then trivial
                for (int j : seed_dofs){
                    std::vector<double> seed, w(n1, 0.0), tmp(n1);
                    project_onto_ker(ker_basis, j, seed);
                    const double inv = 1.0 / static_cast<double>(stab_idx.size());
                    for (int pi : stab_idx){
                        int chi = 1;
                        if (flip == 0){
                            DofT::LocalOrder orient = repr;
                            if (repr.etype == DofT::EDGE_UNORIENT) orient.etype = DofT::EDGE_ORIENT;
                            if (repr.etype == DofT::FACE_UNORIENT) orient.etype = DofT::FACE_ORIENT;
                            chi = DofT::S4::stabilizer_character(all_perms[pi].data(), orient);
                        }
                        apply_perm(P_cache[pi].data(), n1, seed.data(), tmp.data());
                        for (int i = 0; i < n1; ++i)
                            w[i] += static_cast<double>(chi) * inv * tmp[i];
                    }
                    m1_orthogonalize(w, accepted_for_orth, M1.data(), n1, svd_tol);
                    m1_orthogonalize(w, kept, M1.data(), n1, svd_tol);
                    // Re-average to restore character equivariance after projection.
                    {
                        std::vector<double> w2(n1, 0.0), tmp(n1);
                        const double inv = 1.0 / static_cast<double>(stab_idx.size());
                        for (int pi : stab_idx){
                            int chi = 1;
                            if (flip == 0){
                                DofT::LocalOrder orient = repr;
                                if (repr.etype == DofT::EDGE_UNORIENT) orient.etype = DofT::EDGE_ORIENT;
                                if (repr.etype == DofT::FACE_UNORIENT) orient.etype = DofT::FACE_ORIENT;
                                chi = DofT::S4::stabilizer_character(all_perms[pi].data(), orient);
                            }
                            apply_perm(P_cache[pi].data(), n1, w.data(), tmp.data());
                            for (int i = 0; i < n1; ++i)
                                w2[i] += static_cast<double>(chi) * inv * tmp[i];
                        }
                        w = std::move(w2);
                    }
                    const double n2 = m1_inner(w.data(), w.data(), M1.data(), n1);
                    if (n2 <= svd_tol * svd_tol)
                        continue;
                    const double invn = 1.0 / std::sqrt(n2);
                    for (double& x : w) x *= invn;
                    kept.push_back(std::move(w));
                    used_flip = flip;
                    break;
                }
                if (!kept.empty())
                    break;
            }
            // Sign isotype → oriented geom type so DofMap P carries ±1.
            if (used_flip == 0){
                if (etype == DofT::EDGE_UNORIENT) repr.etype = DofT::EDGE_ORIENT;
                if (etype == DofT::FACE_UNORIENT) repr.etype = DofT::FACE_ORIENT;
            }
        } else {
            // Volume-n_comp fiber: seed from V1 dofs on element 0, matrix Reynolds.
            std::vector<std::vector<double>> W;
            for (int j = 0; j < n1; ++j){
                auto lo = V1->m_order.LocalOrderOnTet(DofT::TetOrder(j));
                if (lo.etype != etype || lo.nelem != 0)
                    continue;
                if (dof_geom_in_V0(*V0, lo, n0))
                    continue;
                std::vector<double> seed;
                project_onto_ker(ker_basis, j, seed);
                W.push_back(std::move(seed));
            }
            // If stype mismatch made dof_geom_in_V0 miss V0 edge dofs, fall back to all dofs on elem.
            if (W.empty()){
                for (int j = 0; j < n1; ++j){
                    auto lo = V1->m_order.LocalOrderOnTet(DofT::TetOrder(j));
                    if (lo.etype != etype || lo.nelem != 0)
                        continue;
                    std::vector<double> seed;
                    project_onto_ker(ker_basis, j, seed);
                    W.push_back(std::move(seed));
                }
            }
            if (static_cast<int>(W.size()) < n_comp){
                std::ostringstream oss;
                oss << "ComplementFemSpace::make: etype=" << int(etype)
                    << " not enough seeds (" << W.size() << " < " << n_comp << ")";
                throw std::runtime_error(oss.str());
            }
            if (static_cast<int>(W.size()) > n_comp){
                std::vector<std::pair<double, int>> score(W.size());
                for (std::size_t i = 0; i < W.size(); ++i)
                    score[i] = {m1_inner(W[i].data(), W[i].data(), M1.data(), n1), static_cast<int>(i)};
                std::sort(score.begin(), score.end(), [](auto& a, auto& b){ return a.first > b.first; });
                std::vector<std::vector<double>> W2;
                for (int k = 0; k < n_comp; ++k)
                    W2.push_back(std::move(W[score[k].second]));
                W = std::move(W2);
            }
            matrix_reynolds(W, n1, n_comp, stab_idx, P_cache, etype, 0, out_stype, all_perms);
            // Project fiber out of previously accepted modes (same coeffs for all columns breaks equivariance;
            // project each seed before Reynolds instead — here project after Reynolds with shared Gram against accepted).
            for (auto& w : W)
                m1_orthogonalize(w, accepted_for_orth, M1.data(), n1, svd_tol);
            // Re-average to restore equivariance after non-equivariant projection.
            matrix_reynolds(W, n1, n_comp, stab_idx, P_cache, etype, 0, out_stype, all_perms);
            equivariant_m1_orthonormalize(W, n1, n_comp, M1.data(), svd_tol);
            kept = std::move(W);
        }

        if (static_cast<int>(kept.size()) != n_comp){
            std::ostringstream oss;
            oss << "ComplementFemSpace::make: etype=" << int(etype)
                << " got " << kept.size() << " reference modes, expected " << n_comp;
            throw std::runtime_error(oss.str());
        }

        const uchar n_elems = DofT::GeomTypeTetElems(etype);
        const std::size_t orbit_begin = collected_coefs.size();
        for (uchar e = 0; e < n_elems; ++e){
            DofT::LocalOrder target = repr;
            target.nelem = e;
            auto sigma = DofT::S4::perm_vertex_to_repr(repr, target);
            int pi = find_perm_index(all_perms, sigma);
            const double* P = P_cache[pi].data();

            // Equivariant transport: W_e = P_sigma * W_ref * Q(sigma)^T
            std::vector<double> Q(n_comp * n_comp, 0.0);
            if (n_comp > 1)
                local_rep_matrix(etype, 0, out_stype, n_comp, sigma.data(), Q.data());
            else
                Q[0] = 1.0;

            std::vector<std::vector<double>> We(n_comp, std::vector<double>(n1, 0.0));
            std::vector<double> tmp(n1);
            for (int ell = 0; ell < n_comp; ++ell){
                apply_perm(P, n1, kept[ell].data(), tmp.data());
                for (int neu = 0; neu < n_comp; ++neu){
                    double q = Q[ell + neu * n_comp]; // Q^T[neu, ell] = Q[ell, neu]
                    if (std::abs(q) < 1e-15) continue;
                    for (int i = 0; i < n1; ++i)
                        We[neu][i] += q * tmp[i];
                }
            }

            for (int ell = 0; ell < n_comp; ++ell){
                DofT::LocalOrder tag;
                tag.etype = repr.etype;
                tag.nelem = e;
                tag.stype = out_stype;
                tag.lsid = static_cast<uchar>(ell);
                tag.leid = static_cast<uint>(ell);
                tag.gid = uint(-1);
                collected_coefs.push_back(std::move(We[ell]));
                collected_tags.push_back(tag);
            }
        }
        // Full S4-orbit must enter the accepted subspace so later types project
        // against an invariant space (reference modes alone break Stab-equivariance).
        for (std::size_t i = orbit_begin; i < collected_coefs.size(); ++i)
            accepted_for_orth.push_back(collected_coefs[i]);
    }

    if (static_cast<int>(collected_coefs.size()) != n10){
        std::ostringstream oss;
        oss << "ComplementFemSpace::make: equivariant basis has wrong dimension: got "
            << collected_coefs.size() << ", expected " << n10
            << " (types=" << type_repr.size() << ")";
        throw std::runtime_error(oss.str());
    }

    auto dof_map = complement_dof_map_from_tags(collected_tags);
    if (static_cast<int>(dof_map.NumDofOnTet()) != n10){
        std::ostringstream oss;
        oss << "ComplementFemSpace::make: DofMap size mismatch: map has "
            << dof_map.NumDofOnTet() << ", expected " << n10;
        throw std::runtime_error(oss.str());
    }
    sort_dof_tags(collected_tags, collected_coefs);

    std::vector<double> Psi(n10*n1);
    for (int k = 0; k < n10; ++k)
        std::copy(collected_coefs[k].begin(), collected_coefs[k].end(), Psi.data() + k*n1);

    std::vector<double> Big(n1*n1), dual(n10*n1), imem(2*n1);
    std::vector<double> lu_mem(2*n1*n1);
    for (int k = 0; k < n10; ++k)
        for (int j = 0; j < n1; ++j)
            Big[k + j*n1] = Psi[k*n1 + j];
    for (int k = 0; k < n0; ++k)
        for (int j = 0; j < n1; ++j)
            Big[n10 + k + j*n1] = C[k + j*n0];
    std::vector<double> BigInv(n1*n1);
    fullPivLU_inverse(Big.data(), BigInv.data(), n1, lu_mem.data(), reinterpret_cast<int*>(imem.data()));

    double inv_err = 0;
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n1; ++j){
            double s = 0;
            for (int k = 0; k < n1; ++k)
                s += Big[i + k*n1] * BigInv[k + j*n1];
            s -= (i == j) ? 1.0 : 0.0;
            inv_err += s*s;
        }
    if (inv_err > 1e-8)
        throw std::runtime_error("ComplementFemSpace::make: Big matrix is ill-conditioned");

    for (int k = 0; k < n10; ++k)
        for (int j = 0; j < n1; ++j)
            dual[k*n1 + j] = BigInv[j + k*n1];

    double bio_err = 0;
    for (int k = 0; k < n10; ++k)
        for (int l = 0; l < n10; ++l){
            double s = 0;
            for (int j = 0; j < n1; ++j)
                s += Psi[k*n1 + j] * dual[l*n1 + j];
            s -= (k == l) ? 1.0 : 0.0;
            bio_err += s*s;
        }
    if (bio_err > 1e-8)
        throw std::runtime_error("ComplementFemSpace::make: biorthogonality check failed");

    ComplementFemSpace res;
    res.m_V1 = std::move(V1);
    res.m_V0 = std::move(V0);
    res.m_basis_coefs = std::move(Psi);
    res.m_dual_coefs = std::move(dual);
    res.m_dof_tags = std::move(collected_tags);
    res.m_order = DofT::DofMap(std::make_shared<DofT::UniteDofMap>(dof_map));
    return res;
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
    if (m_V1.get() == a.m_V1.get() && m_V0.get() == a.m_V0.get()) return true;
    return m_V1 && a.m_V1 && *m_V1 == *a.m_V1 && m_V0 && a.m_V0 && *m_V0 == *a.m_V0
        && m_basis_coefs == a.m_basis_coefs && m_dual_coefs == a.m_dual_coefs;
}

namespace {

BandDenseMatrixX<> apply_op_internal(const ComplementFemSpace& sp, AniMemoryX<>& mem, ArrayView<>& U,
    BandDenseMatrixX<>(BaseFemSpace::*apl)(AniMemoryX<>&, ArrayView<>&) const,
    OpMemoryRequirements(BaseFemSpace::*req)(uint, uint) const){
    const int n1 = static_cast<int>(sp.m_V1->m_order.NumDofOnTet());
    const int n10 = static_cast<int>(sp.m_order.NumDofOnTet());
    auto lreq = (sp.m_V1.get()->*req)(mem.q, mem.f);
    if (mem.extraR.size < lreq.Usz)
        throw std::runtime_error("ComplementFemSpace: not enough extraR for V1 apply buffer");
    const std::size_t saved_extra = mem.extraR.size;
    mem.extraR.size -= lreq.Usz;
    ArrayView<> lU(mem.extraR.data + mem.extraR.size, lreq.Usz);
    BandDenseMatrixX<> Vx = (sp.m_V1.get()->*apl)(mem, lU);
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

} // namespace

std::shared_ptr<BaseFemSpace> complement_base_opt(
    std::shared_ptr<BaseFemSpace> V1,
    std::shared_ptr<BaseFemSpace> V0);

namespace {

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
    const std::shared_ptr<BaseFemSpace>& V0)
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
            parts.push_back(complement_base_opt(leaves1[i1], leaves0[i0]));
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
                        static_cast<int>(k), complement_base_opt(rv.m_base, leaves0[i0])));
                    ++i1;
                    i0 += k;
                    continue;
                }
            }
            auto L0k = (k == 1) ? leaves0[i0]
                : std::shared_ptr<BaseFemSpace>(std::make_shared<VectorFemSpace>(static_cast<int>(k), leaves0[i0]));
            parts.push_back(complement_base_opt(leaves1[i1], L0k));
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
                    static_cast<int>(k), complement_base_opt(leaves1[i1], rv.m_base)));
                i1 += k;
                ++i0;
                continue;
            }
        }
        auto R1k = std::make_shared<VectorFemSpace>(static_cast<int>(k), leaves1[i1]);
        parts.push_back(complement_base_opt(R1k, leaves0[i0]));
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
    const std::shared_ptr<BaseFemSpace>& V0)
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
        parts.push_back(complement_base_opt(a.m_spaces[i], b.m_spaces[i]));
    }
    return std::make_shared<ComplexFemSpace>(std::move(parts));
}

} // namespace

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
    return apply_op_internal(*this, mem, U, &BaseFemSpace::applyIDEN, &BaseFemSpace::memIDEN);
}

OpMemoryRequirements ComplementFemSpace::memIDEN(uint nquadpoints, uint fusion) const {
    return complement_mem_op(*this, nquadpoints, fusion, &BaseFemSpace::memIDEN, dim());
}

BandDenseMatrixX<> ComplementFemSpace::applyGRAD(AniMemoryX<>& mem, ArrayView<>& U) const {
    return apply_op_internal(*this, mem, U, &BaseFemSpace::applyGRAD, &BaseFemSpace::memGRAD);
}

OpMemoryRequirements ComplementFemSpace::memGRAD(uint nquadpoints, uint fusion) const {
    return complement_mem_op(*this, nquadpoints, fusion, &BaseFemSpace::memGRAD, dimGRAD());
}

BandDenseMatrixX<> ComplementFemSpace::applyDUDX(AniMemoryX<>& mem, ArrayView<>& U, uchar k) const {
    const int n1 = static_cast<int>(m_V1->m_order.NumDofOnTet());
    const int n10 = static_cast<int>(m_order.NumDofOnTet());
    auto lreq = m_V1->memDUDX(mem.q, mem.f, k);
    if (mem.extraR.size < lreq.Usz)
        throw std::runtime_error("ComplementFemSpace::applyDUDX: not enough extraR");
    const std::size_t saved_extra = mem.extraR.size;
    mem.extraR.size -= lreq.Usz;
    ArrayView<> lU(mem.extraR.data + mem.extraR.size, lreq.Usz);
    BandDenseMatrixX<> Vx = m_V1->applyDUDX(mem, lU, k);
    uint dimop = 0;
    for (std::size_t p = 0; p < Vx.nparts; ++p)
        dimop = std::max(dimop, static_cast<uint>(Vx.stRow[p+1]));
    const std::size_t need_full = static_cast<std::size_t>(dimop) * mem.q * mem.f * n1;
    if (mem.extraR.size < need_full){
        mem.extraR.size = saved_extra;
        throw std::runtime_error("ComplementFemSpace::applyDUDX: not enough extraR for dense buffer");
    }
    DenseMatrix<> lUfull(mem.extraR.data, dimop * mem.q, mem.f * n1, mem.extraR.size);
    lUfull.SetZero();
    for (std::size_t p = 0; p < Vx.nparts; ++p){
        int nCol = Vx.stCol[p+1] - Vx.stCol[p], nRow = Vx.stRow[p+1] - Vx.stRow[p];
        for (std::size_t r = 0; r < mem.f; ++r)
        for (int j = 0; j < nCol; ++j)
        for (std::size_t n = 0; n < mem.q; ++n)
        for (int row = 0; row < nRow; ++row)
            lUfull(row + Vx.stRow[p] + dimop*n, j + Vx.stCol[p] + n1*r) = Vx.data[p](row + nRow*n, j + nCol*r);
    }
    DenseMatrix<> llU(U.data, dimop * mem.q, mem.f * n10, U.size);
    llU.SetZero();
    for (std::size_t r = 0; r < mem.f; ++r)
    for (int i = 0; i < n10; ++i)
    for (int j = 0; j < n1; ++j)
    for (std::size_t n = 0; n < mem.q; ++n)
    for (uint d = 0; d < dimop; ++d)
        llU(d + dimop*n, i + n10*r) += m_basis_coefs[j + n1*i] * lUfull(d + dimop*n, j + n1*r);
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

std::shared_ptr<BaseFemSpace> complement_base_opt(
    std::shared_ptr<BaseFemSpace> V1,
    std::shared_ptr<BaseFemSpace> V0)
{
    if (!V1 || !V0)
        throw std::runtime_error("operator-: invalid FemSpace operands");
    if (V1->dim() != V0->dim())
        throw std::runtime_error("operator-: spaces must have same field dimension");

    const auto t1 = V1->gatherType();
    const auto t0 = V0->gatherType();

    if (t1 == BaseFemSpace::BaseTypes::VectorType && t0 == BaseFemSpace::BaseTypes::VectorType) {
        auto& a = *std::static_pointer_cast<VectorFemSpace>(V1);
        auto& b = *std::static_pointer_cast<VectorFemSpace>(V0);
        if (a.vec_dim() == b.vec_dim())
            return std::make_shared<VectorFemSpace>(
                static_cast<int>(a.vec_dim()),
                complement_base_opt(a.m_base, b.m_base));
    }

    if (auto factored = try_complex_factor(V1, V0))
        return factored;
    if (auto factored = try_dim_matched_factor(V1, V0))
        return factored;

    auto f1 = flatten_to_scalar_complex(V1);
    auto f0 = flatten_to_scalar_complex(V0);
    if (f1.get() != V1.get() || f0.get() != V0.get()) {
        if (auto factored = try_complex_factor(f1, f0))
            return factored;
        if (auto factored = try_dim_matched_factor(f1, f0))
            return factored;
    }

    return std::make_shared<ComplementFemSpace>(ComplementFemSpace::make(std::move(f1), std::move(f0)));
}

FemSpace operator-(const FemSpace& V1, const FemSpace& V0){
    if (!V1.isValid() || !V0.isValid())
        throw std::runtime_error("operator-: invalid FemSpace operands");
    return FemSpace(complement_base_opt(V1.base(), V0.base()));
}

} // namespace Ani
