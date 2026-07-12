#include "adapt_mesh.h"

#include "anifem++/fem/geometry.h"
#include "anifem++/fem/mutex_type.h"
#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/spaces/spaces.h"
#include "anifem++/inmost_interface/ordering.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace INMOST;
using namespace Ani;

namespace {

constexpr double kEigenEps = 1e-300;

/// Temporary CELL tag: |H| in Voigt (6) + singular values ascending (3) + degenerate flag (1).
constexpr int kCellHSize = 10;

// ---- 3x3 symmetric matrix helpers (col-major dense / Voigt XX YY ZZ XY YZ XZ) ----

inline void voigt_to_dense(const double v[6], double A[9]) {
    A[0] = v[0]; A[4] = v[1]; A[8] = v[2];
    A[1] = A[3] = v[3];
    A[5] = A[7] = v[4];
    A[2] = A[6] = v[5];
}

inline void dense_to_voigt(const double A[9], double v[6]) {
    v[0] = A[0];
    v[1] = A[4];
    v[2] = A[8];
    v[3] = 0.5 * (A[1] + A[3]);
    v[4] = 0.5 * (A[5] + A[7]);
    v[5] = 0.5 * (A[2] + A[6]);
}

inline void dense_set_zero(double A[9]) {
    std::fill(A, A + 9, 0.0);
}

inline void dense_set_scaled_eye(double A[9], double s) {
    dense_set_zero(A);
    A[0] = A[4] = A[8] = s;
}

inline void dense_add(double A[9], const double B[9], double scale = 1.0) {
    for (int i = 0; i < 9; ++i)
        A[i] += scale * B[i];
}

inline void dense_scale(double A[9], double s) {
    for (int i = 0; i < 9; ++i)
        A[i] *= s;
}

inline void dense_copy(double dst[9], const double src[9]) {
    std::copy(src, src + 9, dst);
}

// ---- Symmetric eigen-decomposition ----

struct SpectrumInfo {
    double eval[3]{}; ///< ascending
    double Q[9]{};    ///< eigenvectors as columns
    double lmin = 0, lmax = 0, ratio = 0, det = 0;
};

inline void sort_eigen_asc(double eval[3], double Q[9]) {
    for (int i = 0; i < 3; ++i) {
        int best = i;
        for (int j = i + 1; j < 3; ++j)
            if (eval[j] < eval[best])
                best = j;
        if (best == i)
            continue;
        std::swap(eval[i], eval[best]);
        for (int k = 0; k < 3; ++k)
            std::swap(Q[k + 3 * i], Q[k + 3 * best]);
    }
}

inline void reconstruct_from_eigen(const double eval[3], const double Q[9], double A[9]) {
    dense_set_zero(A);
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i)
                A[i + 3 * j] += Q[i + 3 * k] * eval[k] * Q[j + 3 * k];
}

inline void finish_spectrum(SpectrumInfo& s) {
    s.lmin = s.eval[0];
    s.lmax = s.eval[2];
    s.det = s.eval[0] * s.eval[1] * s.eval[2];
    s.ratio = (std::fabs(s.lmin) > kEigenEps) ? (s.lmax / s.lmin)
                                              : std::numeric_limits<double>::infinity();
}

inline int sym_eigen3(const double A_in[9], double eval[3], double Q[9]) {
    double sym[9];
    dense_copy(sym, A_in);
    sym[1] = sym[3] = 0.5 * (sym[1] + sym[3]);
    sym[2] = sym[6] = 0.5 * (sym[2] + sym[6]);
    sym[5] = sym[7] = 0.5 * (sym[5] + sym[7]);

#ifdef WITH_EIGEN
    using namespace Eigen;
    Map<Matrix3d> Sm(sym);
    SelfAdjointEigenSolver<Matrix3d> es(Sm);
    if (es.info() != Success)
        return 1;
    Map<Vector3d> ev(eval);
    Map<Matrix3d> Qm(Q);
    ev = es.eigenvalues();
    Qm = es.eigenvectors();
    return 0;
#else
    return detail::jacobi_symmetric_eigen(sym, 3, eval, Q);
#endif
}

inline SpectrumInfo analyze_sym(const double A[9]) {
    SpectrumInfo s;
    if (sym_eigen3(A, s.eval, s.Q)) {
        s.eval[0] = A[0];
        s.eval[1] = A[4];
        s.eval[2] = A[8];
        dense_set_scaled_eye(s.Q, 1.0);
    }
    sort_eigen_asc(s.eval, s.Q);
    finish_spectrum(s);
    return s;
}

/// Spectral absolute value |H| = Q diag(|lambda|) Q^T.
inline void spectral_abs(const double H[9], double AbsH[9], SpectrumInfo* info = nullptr) {
    SpectrumInfo s = analyze_sym(H);
    for (int i = 0; i < 3; ++i)
        s.eval[i] = std::fabs(s.eval[i]);
    sort_eigen_asc(s.eval, s.Q);
    finish_spectrum(s);
    reconstruct_from_eigen(s.eval, s.Q, AbsH);
    if (info)
        *info = s;
}

inline void matrix_log_spd(const double M[9], double Ln[9]) {
    SpectrumInfo s = analyze_sym(M);
    for (int i = 0; i < 3; ++i)
        s.eval[i] = std::log(std::max(s.eval[i], kEigenEps));
    reconstruct_from_eigen(s.eval, s.Q, Ln);
}

inline void matrix_exp_sym(const double S[9], double Exp[9]) {
    SpectrumInfo s = analyze_sym(S);
    for (int i = 0; i < 3; ++i)
        s.eval[i] = std::exp(s.eval[i]);
    reconstruct_from_eigen(s.eval, s.Q, Exp);
}

/// Absolute bounds + relative log-compression of eigenvalues toward geometric mean.
inline bool clamp_evals(SpectrumInfo& s, const MetricsConstructTraits& tr) {
    bool changed = false;
    for (int i = 0; i < 3; ++i) {
        double v = s.eval[i];
        if (v < tr.M_lambda_min) {
            v = tr.M_lambda_min;
            changed = true;
        }
        if (v > tr.M_lambda_max) {
            v = tr.M_lambda_max;
            changed = true;
        }
        s.eval[i] = v;
    }
    sort_eigen_asc(s.eval, s.Q);

    const double R = std::max(tr.M_lambda_max_rel, 1.0);
    if (s.eval[0] > kEigenEps && s.eval[2] / s.eval[0] > R) {
        const double g = std::cbrt(s.eval[0] * s.eval[1] * s.eval[2]);
        const double lo = 1.0 / std::sqrt(R);
        const double hi = std::sqrt(R);
        for (int i = 0; i < 3; ++i) {
            double r = s.eval[i] / g;
            r = std::min(hi, std::max(lo, r));
            s.eval[i] = g * r;
        }
        for (int i = 0; i < 3; ++i)
            s.eval[i] = std::min(tr.M_lambda_max, std::max(tr.M_lambda_min, s.eval[i]));
        sort_eigen_asc(s.eval, s.Q);
        changed = true;
    }
    finish_spectrum(s);
    return changed;
}

inline bool clamp_spectrum(double A[9], const MetricsConstructTraits& tr) {
    SpectrumInfo s = analyze_sym(A);
    const bool changed = clamp_evals(s, tr);
    reconstruct_from_eigen(s.eval, s.Q, A);
    return changed;
}

inline bool h2_is_degenerate(const SpectrumInfo& abs_info, double H_lambda_max_rel) {
    if (!(abs_info.lmin > kEigenEps))
        return true;
    return abs_info.ratio > H_lambda_max_rel;
}

/**
 * Cell-average Hessian via divergence theorem:
 *   H_ij = 1/|T| sum_f int_f (d_j v) n_i dS.
 * On a planar face n is constant, so the face contribution is n outer int_f grad(v) dS.
 * One fem3Dface per face with test space (P0)^3 assembles int grad(phi_k).
 */
inline void compute_H2_from_v2(const Tetra<const double>& XYZ, const ApplyOpBase& grad_op,
                               const DenseMatrix<>& v2, double vol, int quad_order, DynMem<>& mem,
                               double* Adata, double H[9]) {
    dense_set_zero(H);
    if (!(vol > 0.0))
        return;

    const unsigned nfa = static_cast<unsigned>(v2.nRow);
    DenseMatrix<> A(Adata, 3, nfa);
    ApplyOpFromTemplate<IDEN, FemVec<3, FEM_P0>> iden_p0_vec;

    const double* P[4] = {XYZ.XY0.data, XYZ.XY1.data, XYZ.XY2.data, XYZ.XY3.data};
    static const int fv[4][4] = {
        {0, 1, 2, 3},
        {1, 2, 3, 0},
        {2, 3, 0, 1},
        {3, 0, 1, 2},
    };

    for (int face = 0; face < 4; ++face) {
        double n[3];
        face_normal(P[fv[face][0]], P[fv[face][1]], P[fv[face][2]], P[fv[face][3]], n);

        A.SetZero();
        fem3Dface<DfuncTraits<TENSOR_NULL, true>>(XYZ, face, grad_op, iden_p0_vec, TensorNull<>, A, mem,
                                                  quad_order);

        // I = int_f grad(v) dS = A * v2
        double I[3] = {0.0, 0.0, 0.0};
        for (unsigned k = 0; k < nfa; ++k) {
            const double vk = v2(k, 0);
            I[0] += A(0, k) * vk;
            I[1] += A(1, k) * vk;
            I[2] += A(2, k) * vk;
        }
        // H += n outer I
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                H[i + 3 * j] += n[i] * I[j];
    }
    dense_scale(H, 1.0 / vol);
    H[1] = H[3] = 0.5 * (H[1] + H[3]);
    H[2] = H[6] = 0.5 * (H[2] + H[6]);
    H[5] = H[7] = 0.5 * (H[5] + H[7]);
}

/// M = (det |H|)^{-1/5} |H|  (tetrahedral metric scaling).
inline void build_M_from_spectrum(const SpectrumInfo& s, const double AbsH[9], double M[9],
                                  double isotropic_fallback) {
    if (!(s.det > kEigenEps) || !std::isfinite(s.det)) {
        dense_set_scaled_eye(M, isotropic_fallback);
        return;
    }
    const double scale = std::pow(s.det, -0.2);
    dense_copy(M, AbsH);
    dense_scale(M, scale);
}

/// Clamp |H| spectrum then form M. One eigen-decomposition.
inline void absH_to_M_clamped(const double AbsH_in[9], double M[9], const MetricsConstructTraits& tr) {
    SpectrumInfo s = analyze_sym(AbsH_in);
    clamp_evals(s, tr);
    double AbsH[9];
    reconstruct_from_eigen(s.eval, s.Q, AbsH);
    build_M_from_spectrum(s, AbsH, M, tr.isotropic_fallback);
}

// ---- Cell |H| storage ----

struct CellHView {
    double AbsH[9]{};
    double s[3]{};
    bool degenerate = true;
    double vol = 0.0;
};

inline CellHView load_cell_H(Cell c, Tag cell_H) {
    CellHView v;
    auto arr = c.RealArray(cell_H);
    voigt_to_dense(arr.data(), v.AbsH);
    v.s[0] = arr[6];
    v.s[1] = arr[7];
    v.s[2] = arr[8];
    // CreateTag zero-fills: arr[9]==0 would look "non-degenerate" with AbsH=0.
    // Treat a vanishing spectrum as degenerate regardless of the stored flag.
    v.degenerate = (arr[9] > 0.5) || !(v.s[0] > kEigenEps);
    v.vol = c.Volume();
    return v;
}

inline void store_cell_H(Cell c, Tag cell_H, const double AbsH[9], const double s[3], bool degenerate) {
    auto arr = c.RealArray(cell_H);
    dense_to_voigt(AbsH, arr.data());
    arr[6] = s[0];
    arr[7] = s[1];
    arr[8] = s[2];
    arr[9] = degenerate ? 1.0 : 0.0;
}

inline int count_svals_above(const double s[3], double thr) {
    int n = 0;
    for (int i = 0; i < 3; ++i)
        if (s[i] > thr)
            ++n;
    return n;
}

inline double product_svals_above(const double s[3], double thr) {
    double p = 1.0;
    bool any = false;
    for (int i = 0; i < 3; ++i)
        if (s[i] > thr) {
            p *= s[i];
            any = true;
        }
    return any ? p : 0.0;
}

inline double min_sval_above(const double s[3], double thr) {
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 3; ++i)
        if (s[i] > thr)
            m = std::min(m, s[i]);
    return m;
}

/// Replace singular values <= thr by the smallest value > thr (keeps eigenvectors).
inline void lift_zero_svals(double AbsH[9], double s[3], double thr) {
    SpectrumInfo info = analyze_sym(AbsH);
    const double smin_nz = min_sval_above(info.eval, thr);
    if (!(smin_nz < std::numeric_limits<double>::infinity()))
        return;
    for (int i = 0; i < 3; ++i)
        if (!(info.eval[i] > thr))
            info.eval[i] = smin_nz;
    sort_eigen_asc(info.eval, info.Q);
    reconstruct_from_eigen(info.eval, info.Q, AbsH);
    std::copy(info.eval, info.eval + 3, s);
}

/// Edge / DOF weights alpha for v2 = -1/2 sum alpha_k phi_k.
inline void fill_alpha_from_d(const DenseMatrix<>& d, DenseMatrix<>& alpha, bool heuristics,
                              const double* Bdata, unsigned nfa, double sum_abs) {
    if (!heuristics) {
        for (unsigned k = 0; k < nfa; ++k)
            alpha(k, 0) = d(k, 0);
        return;
    }
    if (!(sum_abs > 0.0)) {
        for (unsigned k = 0; k < nfa; ++k)
            alpha(k, 0) = 0.0;
        return;
    }
    double Bd_d = 0.0;
    for (unsigned i = 0; i < nfa; ++i) {
        double Bi = 0.0;
        for (unsigned j = 0; j < nfa; ++j)
            Bi += Bdata[i + nfa * j] * d(j, 0);
        Bd_d += d(i, 0) * Bi;
    }
    Bd_d = std::fabs(Bd_d);
    for (unsigned k = 0; k < nfa; ++k)
        alpha(k, 0) = std::fabs(d(k, 0)) * Bd_d / sum_abs;
}

// ---- Vertex projection helpers ----

inline void pick_max_det_M(const std::vector<CellHView>& patch, bool only_nonsing,
                           const MetricsConstructTraits& tr, double M[9]) {
    double best_det = -std::numeric_limits<double>::infinity();
    const CellHView* best = nullptr;
    for (const auto& v : patch) {
        if (only_nonsing && v.degenerate)
            continue;
        const double det = v.s[0] * v.s[1] * v.s[2];
        if (!best || det > best_det) {
            best_det = det;
            best = &v;
        }
    }
    if (!best) {
        dense_set_scaled_eye(M, tr.isotropic_fallback);
        return;
    }
    absH_to_M_clamped(best->AbsH, M, tr);
}

inline void geom_mean_M(const std::vector<CellHView>& patch, bool only_nonsing,
                        const MetricsConstructTraits& tr, double M[9]) {
    double SumLn[9];
    dense_set_zero(SumLn);
    int n_used = 0;
    for (const auto& v : patch) {
        if (only_nonsing && v.degenerate)
            continue;
        double Mc[9], Ln[9];
        absH_to_M_clamped(v.AbsH, Mc, tr);
        matrix_log_spd(Mc, Ln);
        dense_add(SumLn, Ln);
        ++n_used;
    }
    if (n_used == 0) {
        dense_set_scaled_eye(M, tr.isotropic_fallback);
        return;
    }
    dense_scale(SumLn, 1.0 / static_cast<double>(n_used));
    matrix_exp_sym(SumLn, M);
}

/// All-degenerate neighbourhood: lift best |H|, or isotropic / volume-weighted fallback.
inline void project_all_degenerate(const std::vector<CellHView>& patch, bool any_nonzero_H,
                                   const MetricsConstructTraits& tr, double M[9]) {
    const double thr = tr.M_lambda_min;
    int best_nnz = -1;
    double best_prod = -1.0;
    const CellHView* best = nullptr;
    for (const auto& v : patch) {
        const int nnz = count_svals_above(v.s, thr);
        const double prod = product_svals_above(v.s, thr);
        if (!best || nnz > best_nnz || (nnz == best_nnz && prod > best_prod)) {
            best_nnz = nnz;
            best_prod = prod;
            best = &v;
        }
    }

    if (best && best_nnz > 0) {
        double AbsH[9], s[3];
        dense_copy(AbsH, best->AbsH);
        std::copy(best->s, best->s + 3, s);
        lift_zero_svals(AbsH, s, thr);
        absH_to_M_clamped(AbsH, M, tr);
        return;
    }

    if (!any_nonzero_H) {
        dense_set_scaled_eye(M, tr.isotropic_fallback);
        return;
    }

    // Pure-zero patch on a mesh that has some nonzero |H|: volume-weighted max singular value.
    double wsum = 0.0, csum = 0.0;
    for (const auto& v : patch) {
        if (v.s[2] > kEigenEps) {
            csum += v.vol * v.s[2];
            wsum += v.vol;
        }
    }
    dense_set_scaled_eye(M, (wsum > 0.0) ? (csum / wsum) : tr.isotropic_fallback);
}

// ---- MPI reductions for verbosity (mesh communicator) ----

#if defined(USE_MPI)
inline void reduce_sum(Mesh* mesh, long long& v) {
    long long g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_LONG_LONG, MPI_SUM, mesh->GetCommunicator());
    v = g;
}
inline void reduce_sum(Mesh* mesh, double& v) {
    double g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, mesh->GetCommunicator());
    v = g;
}
inline void reduce_min(Mesh* mesh, double& v) {
    double g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MIN, mesh->GetCommunicator());
    v = g;
}
inline void reduce_max(Mesh* mesh, double& v) {
    double g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, mesh->GetCommunicator());
    v = g;
}
inline void reduce_max(Mesh* mesh, long long& v) {
    long long g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_LONG_LONG, MPI_MAX, mesh->GetCommunicator());
    v = g;
}
#else
inline void reduce_sum(Mesh*, long long&) {}
inline void reduce_sum(Mesh*, double&) {}
inline void reduce_min(Mesh*, double&) {}
inline void reduce_max(Mesh*, double&) {}
inline void reduce_max(Mesh*, long long&) {}
#endif

struct CellLoopStats {
    long long n_cells = 0;
    long long n_degenerate = 0;
    long long n_d_boost = 0;
    long long n_zero_dof = 0;
    long long any_nonzero_H = 0;
    double sum_s[3] = {0.0, 0.0, 0.0};
    double min_s[3] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::infinity()};
    double max_s[3] = {0.0, 0.0, 0.0};

    void merge(const CellLoopStats& o) {
        n_cells += o.n_cells;
        n_degenerate += o.n_degenerate;
        n_d_boost += o.n_d_boost;
        n_zero_dof += o.n_zero_dof;
        any_nonzero_H = std::max(any_nonzero_H, o.any_nonzero_H);
        for (int i = 0; i < 3; ++i) {
            sum_s[i] += o.sum_s[i];
            min_s[i] = std::min(min_s[i], o.min_s[i]);
            max_s[i] = std::max(max_s[i], o.max_s[i]);
        }
    }
};

struct NodeLoopStats {
    long long n_nodes = 0;
    long long n_node_clamp = 0;
    double sum_log_det13 = 0.0;
    double min_det13 = std::numeric_limits<double>::infinity();
    double max_det13 = 0.0;
    double max_node_ratio = 0.0;
    double min_node_ratio = std::numeric_limits<double>::infinity();

    void merge(const NodeLoopStats& o) {
        n_nodes += o.n_nodes;
        n_node_clamp += o.n_node_clamp;
        sum_log_det13 += o.sum_log_det13;
        min_det13 = std::min(min_det13, o.min_det13);
        max_det13 = std::max(max_det13, o.max_det13);
        max_node_ratio = std::max(max_node_ratio, o.max_node_ratio);
        min_node_ratio = std::min(min_node_ratio, o.min_node_ratio);
    }
};

} // namespace

void construct_metrics(FemSpace fem, Tag var, Tag& metrics, MetricsConstructTraits traits) {
    if (!metrics.isValid())
        throw std::runtime_error("construct_metrics: metrics tag is invalid");
    if (!var.isValid())
        throw std::runtime_error("construct_metrics: var tag is invalid");
    if (fem.dim() != 1)
        throw std::runtime_error("construct_metrics: expected scalar FemSpace (dim == 1)");
    if (traits.vertex_projection_strategy != 0 && traits.vertex_projection_strategy != 1)
        throw std::runtime_error("construct_metrics: vertex_projection_strategy must be 0 or 1");

    Mesh* m = var.GetMeshLink();
    if (m == nullptr)
        m = metrics.GetMeshLink();
    if (m == nullptr)
        throw std::runtime_error("construct_metrics: cannot obtain mesh from tags");

    if (!metrics.isDefined(NODE) || (metrics.GetSize() != ENUMUNDEF && metrics.GetSize() < 6))
        metrics = m->CreateTag(metrics.GetTagName(), DATA_REAL, NODE, NONE, 6);
    if (!metrics.isDefined(NODE) || (metrics.GetSize() != ENUMUNDEF && metrics.GetSize() < 6)) {
        throw std::runtime_error("construct_metrics: failed to obtain a NODE tag of size >= 6 named \"" +
                                 metrics.GetTagName() + "\"");
    }
    if (metrics.GetSize() == ENUMUNDEF) {
        for (auto it = m->BeginNode(); it != m->EndNode(); ++it) {
            auto arr = it->RealArrayDV(metrics);
            if (arr.size() < 6)
                arr.resize(6, 0.0);
        }
    }

    const unsigned nfa = fem.dofMap().NumDofOnTet();
    if (nfa == 0)
        throw std::runtime_error("construct_metrics: FemSpace has zero DOFs on tetrahedron");

    const auto& dmap = *fem.dofMap().target<>();
    const auto grad_op = fem.getOP(GRAD);
    const int quad_order = std::max(1, traits.max_quad_order);
    const bool prep_ef = dmap.GetGeomMask() & (DofT::EDGE | DofT::FACE);
    const bool need_node_perm = dmap.GetGeomMask() & (DofT::EDGE_ORIENT | DofT::FACE_ORIENT);

    {
        const char* tmp_name = "___tmp_cell_absH_svals";
        if (m->HaveTag(tmp_name))
            m->DeleteTag(m->GetTag(tmp_name));
    }
    Tag cell_H = m->CreateTag("___tmp_cell_absH_svals", DATA_REAL, CELL, NONE, kCellHSize);

#ifdef WITH_OPENMP
    const int nthreads = ThreadPar::get_num_threads<ThreadPar::Type::OMP>(-1);
#else
    const int nthreads = 1;
#endif

    // ======================== Cell loop: build |H| ========================
    {
        std::vector<DynMem<>> pools(static_cast<std::size_t>(nthreads));
        std::vector<std::vector<double>> d_bufs(static_cast<std::size_t>(nthreads), std::vector<double>(nfa));
        std::vector<std::vector<double>> alpha_bufs(static_cast<std::size_t>(nthreads), std::vector<double>(nfa));
        std::vector<std::vector<double>> v2_bufs(static_cast<std::size_t>(nthreads), std::vector<double>(nfa));
        std::vector<std::vector<double>> B_bufs(static_cast<std::size_t>(nthreads),
                                                std::vector<double>(static_cast<std::size_t>(nfa) * nfa));
        std::vector<std::vector<double>> A_bufs(static_cast<std::size_t>(nthreads),
                                                std::vector<double>(static_cast<std::size_t>(3) * nfa));
        std::vector<CellLoopStats> local_stats(static_cast<std::size_t>(nthreads));

        auto cell_body = [&](Storage::integer lid, int nthread) {
            Cell cell = m->CellByLocalID(lid);
            if (!cell.isValid() || cell.Hidden() || cell.GetStatus() == Element::Ghost)
                return;

            auto& mem = pools[static_cast<std::size_t>(nthread)];
            auto& st = local_stats[static_cast<std::size_t>(nthread)];
            auto& d_buf = d_bufs[static_cast<std::size_t>(nthread)];
            auto& alpha_buf = alpha_bufs[static_cast<std::size_t>(nthread)];
            auto& v2_buf = v2_bufs[static_cast<std::size_t>(nthread)];
            auto& B_buf = B_bufs[static_cast<std::size_t>(nthread)];
            auto& A_buf = A_bufs[static_cast<std::size_t>(nthread)];

            ++st.n_cells;

            std::array<HandleType, 4> nds{}, fcs{};
            std::array<HandleType, 6> eds{};
            collectConnectivityInfo(cell, nds.data(), eds.data(), fcs.data(), true, prep_ef);

            std::array<unsigned char, 4> canonical_node_indexes{0, 1, 2, 3};
            if (need_node_perm) {
                std::array<long, 4> gni;
                for (int i = 0; i < 4; ++i)
                    gni[i] = Node(m, nds[i]).GlobalID();
                canonical_node_indexes = createOrderPermutation(gni.data());
            }

            double XY[12]{};
            for (int n = 0; n < 4; ++n)
                for (int k = 0; k < 3; ++k)
                    XY[3 * n + k] = Node(m, nds[n]).Coords()[k];
            Tetra<const double> XYZ(XY + 0, XY + 3, XY + 6, XY + 9);
            const double vol = cell.Volume();

            DenseMatrix<> d(d_buf.data(), nfa, 1);
            DenseMatrix<> alpha(alpha_buf.data(), nfa, 1);
            DenseMatrix<> v2(v2_buf.data(), nfa, 1);
            d.SetZero();
            GatherDataOnElement(var, dmap, m, cell.GetHandle(), fcs.data(), eds.data(), nds.data(),
                                canonical_node_indexes.data(), d.data, nullptr, 0);

            double sum_abs = 0.0;
            for (unsigned k = 0; k < nfa; ++k)
                sum_abs += std::fabs(d(k, 0));

            double AbsH[9];
            double svals[3] = {0.0, 0.0, 0.0};
            bool degenerate = true;

            if (!(sum_abs > 0.0)) {
                ++st.n_zero_dof;
                dense_set_zero(AbsH);
            } else {
                if (traits.distrib_heuristics) {
                    DenseMatrix<> B(B_buf.data(), nfa, nfa);
                    B.SetZero();
                    fem3Dtet<DfuncTraits<TENSOR_NULL, true>>(XYZ, grad_op, grad_op, TensorNull<>, B, mem,
                                                            quad_order);
                    if (vol > 0.0) {
                        const double inv_vol = 1.0 / vol;
                        for (unsigned i = 0; i < nfa * nfa; ++i)
                            B_buf[i] *= inv_vol;
                    }
                }

                auto recompute_alpha = [&]() {
                    fill_alpha_from_d(d, alpha, traits.distrib_heuristics, B_buf.data(), nfa, sum_abs);
                };
                recompute_alpha();

                auto eval_absH = [&](SpectrumInfo& info) {
                    for (unsigned k = 0; k < nfa; ++k)
                        v2(k, 0) = -0.5 * alpha(k, 0);
                    double H[9];
                    compute_H2_from_v2(XYZ, grad_op, v2, vol, quad_order, mem, A_buf.data(), H);
                    spectral_abs(H, AbsH, &info);
                };

                SpectrumInfo info;
                eval_absH(info);
                std::copy(info.eval, info.eval + 3, svals);
                degenerate = h2_is_degenerate(info, traits.H_lambda_max_rel);

                // One attempt to boost the largest |d_k|; keep original |H| if still degenerate.
                if (degenerate) {
                    double AbsH0[9], s0[3];
                    dense_copy(AbsH0, AbsH);
                    std::copy(svals, svals + 3, s0);

                    unsigned imax = 0;
                    double amax = std::fabs(d(0, 0));
                    for (unsigned k = 1; k < nfa; ++k) {
                        const double ak = std::fabs(d(k, 0));
                        if (ak > amax) {
                            amax = ak;
                            imax = k;
                        }
                    }
                    d(imax, 0) *= traits.alpha_boost;
                    sum_abs = 0.0;
                    for (unsigned k = 0; k < nfa; ++k)
                        sum_abs += std::fabs(d(k, 0));
                    recompute_alpha();
                    eval_absH(info);
                    ++st.n_d_boost;

                    if (h2_is_degenerate(info, traits.H_lambda_max_rel)) {
                        dense_copy(AbsH, AbsH0);
                        std::copy(s0, s0 + 3, svals);
                        degenerate = true;
                    } else {
                        std::copy(info.eval, info.eval + 3, svals);
                        degenerate = false;
                    }
                }
            }

            if (degenerate)
                ++st.n_degenerate;
            if (svals[2] > kEigenEps)
                st.any_nonzero_H = 1;

            for (int i = 0; i < 3; ++i) {
                st.sum_s[i] += svals[i];
                st.min_s[i] = std::min(st.min_s[i], svals[i]);
                st.max_s[i] = std::max(st.max_s[i], svals[i]);
            }

            store_cell_H(cell, cell_H, AbsH, svals, degenerate);
            mem.defragment();
        };

#ifdef WITH_OPENMP
        if (nthreads > 1)
            ThreadPar::parallel_for<ThreadPar::Type::OMP>(nthreads, cell_body, m->FirstLocalID(CELL),
                                                          m->CellLastLocalID());
        else
#endif
            ThreadPar::parallel_for<ThreadPar::Type::NONE>(1, cell_body, m->FirstLocalID(CELL),
                                                           m->CellLastLocalID());

        CellLoopStats cell_stats;
        for (int t = 0; t < nthreads; ++t)
            cell_stats.merge(local_stats[static_cast<std::size_t>(t)]);

        m->ExchangeData(cell_H, CELL);
        reduce_max(m, cell_stats.any_nonzero_H);

        // ======================== Node loop: |H| -> M ========================
        if (metrics.GetSize() != ENUMUNDEF && metrics.GetSize() < 6)
            throw std::runtime_error("construct_metrics: node metrics tag size < 6");

        std::vector<std::vector<CellHView>> patches(static_cast<std::size_t>(nthreads));
        std::vector<NodeLoopStats> node_local(static_cast<std::size_t>(nthreads));
        const bool any_nonzero_H = (cell_stats.any_nonzero_H != 0);

        auto node_body = [&](Storage::integer lid, int nthread) {
            Node node = m->NodeByLocalID(lid);
            if (!node.isValid() || node.Hidden() || node.GetStatus() == Element::Ghost)
                return;

            auto& st = node_local[static_cast<std::size_t>(nthread)];
            auto& patch = patches[static_cast<std::size_t>(nthread)];
            ++st.n_nodes;

            auto cells = node.getCells();
            double M[9];
            dense_set_zero(M);

            patch.clear();
            patch.reserve(cells.size());
            int n_nonsing = 0;
            for (auto it = cells.begin(); it != cells.end(); ++it) {
                Cell c(m, *it);
                // Hidden cells are skipped in the cell loop and keep zero-filled tag data.
                if (!c.isValid() || c.Hidden())
                    continue;
                patch.push_back(load_cell_H(c, cell_H));
                if (!patch.back().degenerate)
                    ++n_nonsing;
            }

            if (patch.empty()) {
                dense_set_scaled_eye(M, traits.isotropic_fallback);
            } else if (n_nonsing > 0) {
                // (1) all non-degenerate or (2) mixed: use non-degenerate subset only.
                if (traits.vertex_projection_strategy == 0)
                    pick_max_det_M(patch, true, traits, M);
                else
                    geom_mean_M(patch, true, traits, M);
            } else {
                project_all_degenerate(patch, any_nonzero_H, traits, M);
            }

            if (clamp_spectrum(M, traits))
                ++st.n_node_clamp;

            auto marr = node.RealArray(metrics);
            dense_to_voigt(M, marr.data());

            if (traits.verbosity) {
                SpectrumInfo ns = analyze_sym(M);
                const double det13 = std::cbrt(std::max(ns.det, 0.0));
                st.sum_log_det13 += (det13 > 0.0) ? std::log(det13) : 0.0;
                st.min_det13 = std::min(st.min_det13, det13);
                st.max_det13 = std::max(st.max_det13, det13);
                if (std::isfinite(ns.ratio)) {
                    st.max_node_ratio = std::max(st.max_node_ratio, ns.ratio);
                    st.min_node_ratio = std::min(st.min_node_ratio, ns.ratio);
                }
            }
        };

#ifdef WITH_OPENMP
        if (nthreads > 1)
            ThreadPar::parallel_for<ThreadPar::Type::OMP>(nthreads, node_body, m->FirstLocalID(NODE),
                                                          m->NodeLastLocalID());
        else
#endif
            ThreadPar::parallel_for<ThreadPar::Type::NONE>(1, node_body, m->FirstLocalID(NODE),
                                                           m->NodeLastLocalID());

        NodeLoopStats node_stats;
        for (int t = 0; t < nthreads; ++t)
            node_stats.merge(node_local[static_cast<std::size_t>(t)]);

        m->ExchangeData(metrics, NODE);
        m->DeleteTag(cell_H);

        if (!traits.verbosity)
            return;

        reduce_sum(m, cell_stats.n_cells);
        reduce_sum(m, cell_stats.n_degenerate);
        reduce_sum(m, cell_stats.n_d_boost);
        reduce_sum(m, cell_stats.n_zero_dof);
        for (int i = 0; i < 3; ++i) {
            reduce_sum(m, cell_stats.sum_s[i]);
            reduce_min(m, cell_stats.min_s[i]);
            reduce_max(m, cell_stats.max_s[i]);
        }
        reduce_sum(m, node_stats.n_nodes);
        reduce_sum(m, node_stats.n_node_clamp);
        reduce_sum(m, node_stats.sum_log_det13);
        reduce_min(m, node_stats.min_det13);
        reduce_max(m, node_stats.max_det13);
        reduce_max(m, node_stats.max_node_ratio);
        reduce_min(m, node_stats.min_node_ratio);

        if (m->GetProcessorRank() != 0)
            return;

        const double inv_c =
            (cell_stats.n_cells > 0) ? (1.0 / static_cast<double>(cell_stats.n_cells)) : 0.0;
        const double geom_mean_det13 =
            (node_stats.n_nodes > 0)
                ? std::exp(node_stats.sum_log_det13 / static_cast<double>(node_stats.n_nodes))
                : 0.0;
        std::cout << "construct_metrics:\n"
                  << "  cells=" << cell_stats.n_cells << " nodes=" << node_stats.n_nodes
                  << " zero_dof_cells=" << cell_stats.n_zero_dof << "\n"
                  << "  degenerate_|H|=" << cell_stats.n_degenerate << " ("
                  << (100.0 * cell_stats.n_degenerate * inv_c) << "%)"
                  << " d_boost_attempts=" << cell_stats.n_d_boost << "\n"
                  << "  singular values of |H| (asc):\n"
                  << "    s0: mean=" << cell_stats.sum_s[0] * inv_c << " min=" << cell_stats.min_s[0]
                  << " max=" << cell_stats.max_s[0] << "\n"
                  << "    s1: mean=" << cell_stats.sum_s[1] * inv_c << " min=" << cell_stats.min_s[1]
                  << " max=" << cell_stats.max_s[1] << "\n"
                  << "    s2: mean=" << cell_stats.sum_s[2] * inv_c << " min=" << cell_stats.min_s[2]
                  << " max=" << cell_stats.max_s[2] << "\n"
                  << "  node_spectrum_clamps=" << node_stats.n_node_clamp << "\n"
                  << "  det(M)^{1/3}: min=" << node_stats.min_det13 << " max=" << node_stats.max_det13
                  << " geom_mean=" << geom_mean_det13 << "\n"
                  << "  node anisotropy lambda_max/lambda_min: min=" << node_stats.min_node_ratio
                  << " max=" << node_stats.max_node_ratio << std::endl;
    }
}
