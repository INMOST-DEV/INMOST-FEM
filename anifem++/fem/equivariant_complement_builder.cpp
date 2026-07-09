//
// Shared S4-equivariant complement basis builder.
//

#include "equivariant_complement_builder.h"
#include "geometry.h"
#include <stdexcept>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <map>
#include <array>

namespace Ani {
namespace {

using DofT::uchar;

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
    throw std::runtime_error("equivariant complement: permutation not found in S4 table");
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
/// G is SPD: compute its eigen-decomposition in long double (avoid forming G*G^T).
void equivariant_m1_orthonormalize(std::vector<std::vector<double>>& W, int n1, int n_comp,
    const double* M1, double tol){
    if (n_comp <= 0 || W.empty()) return;
    const int ncol = static_cast<int>(W.size());
    std::vector<long double> G(static_cast<std::size_t>(ncol) * ncol, 0.0L);
    for (int j = 0; j < ncol; ++j)
        for (int i = 0; i < ncol; ++i)
            G[i + j * ncol] = static_cast<long double>(m1_inner(W[i].data(), W[j].data(), M1, n1));

    std::vector<long double> U(static_cast<std::size_t>(ncol) * ncol), s(ncol), V(static_cast<std::size_t>(ncol) * ncol);
    std::vector<long double> mem(2 * ncol * ncol + ncol);
    if (jacobiSVD(G.data(), ncol, ncol, U.data(), s.data(), V.data(), mem.data()))
        throw std::runtime_error("equivariant complement: fiber Gram SVD failed");

    // G^{-1/2} = U diag(s_a^{-1/2}) U^T on the positive eigenspace.
    std::vector<long double> Ginvsqrt(static_cast<std::size_t>(ncol) * ncol, 0.0L);
    int npos = 0;
    const long double cut = static_cast<long double>(tol) * static_cast<long double>(tol);
    for (int a = 0; a < ncol; ++a){
        if (s[a] <= cut)
            continue;
        ++npos;
        const long double inv = 1.0L / std::sqrt(s[a]);
        for (int i = 0; i < ncol; ++i)
            for (int j = 0; j < ncol; ++j)
                Ginvsqrt[i + j * ncol] += inv * U[i + a * ncol] * U[j + a * ncol];
    }
    if (npos < n_comp)
        throw std::runtime_error("equivariant complement: fiber Gram is rank-deficient");

    std::vector<std::vector<double>> Wnew(n_comp, std::vector<double>(n1, 0.0));
    for (int j = 0; j < n_comp; ++j)
        for (int b = 0; b < ncol; ++b){
            const double c = static_cast<double>(Ginvsqrt[b + j * ncol]);
            if (std::abs(c) < 1e-15) continue;
            for (int i = 0; i < n1; ++i)
                Wnew[j][i] += c * W[b][i];
        }
    W = std::move(Wnew);
}

} // namespace

EquivariantComplementBasis build_equivariant_complement_basis(
    const BaseFemSpace& V1,
    const BaseFemSpace& V0,
    const double* C,
    int n0,
    int n1,
    const std::vector<std::vector<double>>& ker_basis,
    const double* orth_metric,
    int n10,
    double svd_tol,
    const std::string& err_prefix,
    const double* dual_metric)
{
    if (static_cast<int>(ker_basis.size()) != n10)
        throw std::runtime_error(err_prefix + ": kernel has wrong dimension");

    // Cache all 24 S4 permutation matrices on V1 dofs.
    std::array<std::array<uchar, 4>, 24> all_perms;
    DofT::S4::all_permutations(all_perms);
    std::vector<std::vector<double>> P_cache(24, std::vector<double>(static_cast<std::size_t>(n1) * n1, 0.0));
    for (int p = 0; p < 24; ++p)
    DofT::S4::build_dof_permutation(*V1.m_order.base(), all_perms[p].data(), P_cache[p].data(), n1);

    // Unique geometric etypes present in V1 but not fully covered by V0.
    std::map<SymTypeKey, DofT::LocalOrder> type_repr;
    for (int j = 0; j < n1; ++j){
    auto lo = V1.m_order.LocalOrderOnTet(DofT::TetOrder(j));
    if (dof_geom_in_V0(V0, lo, n0))
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
    const uint n_v1 = V1.m_order.NumDof(etype);
    const uint n_v0 = V0.m_order.NumDof(etype);
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
        // Scalar character: Reynolds-average seeds into Stab-invariants, then orbit.
        // Prefer ker_basis vectors as seeds: for energy complements, projections of
        // geometric EDGE dofs may span only a proper submodule (orbit rank < 6),
        // while a generic vector in ker yields a full Ind(χ) frame (same as L2).
        std::vector<std::vector<double>> seeds = ker_basis;
        for (int j = 0; j < n1; ++j){
            auto lo = V1.m_order.LocalOrderOnTet(DofT::TetOrder(j));
            if (lo.etype != etype || lo.nelem != 0)
                continue;
            std::vector<double> seed;
            project_onto_ker(ker_basis, j, seed);
            seeds.push_back(std::move(seed));
        }
        int used_flip = -1;
        for (int flip : {0, 1}){ // prefer sign isotype first (P3-P2 edge), then trivial
            for (const auto& seed : seeds){
                std::vector<double> w(n1, 0.0), tmp(n1);
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
                m1_orthogonalize(w, accepted_for_orth, orth_metric, n1, svd_tol);
                m1_orthogonalize(w, kept, orth_metric, n1, svd_tol);
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
                const double n2 = m1_inner(w.data(), w.data(), orth_metric, n1);
                if (n2 <= svd_tol * svd_tol)
                    continue;
                const double invn = 1.0 / std::sqrt(n2);
                for (double& x : w) x *= invn;
                // Reject Stab-invariants whose S4-orbit is rank-deficient (proper submodule).
                {
                    std::vector<std::vector<double>> orb;
                    const uchar n_elems = DofT::GeomTypeTetElems(etype);
                    DofT::LocalOrder src = repr;
                    if (flip == 0){
                        if (etype == DofT::EDGE_UNORIENT) src.etype = DofT::EDGE_ORIENT;
                        if (etype == DofT::FACE_UNORIENT) src.etype = DofT::FACE_ORIENT;
                    }
                    for (uchar e = 0; e < n_elems; ++e){
                        DofT::LocalOrder target = src; target.nelem = e;
                        auto sigma = DofT::S4::perm_vertex_to_repr(src, target);
                        int pi = find_perm_index(all_perms, sigma);
                        std::vector<double> we(n1);
                        apply_perm(P_cache[pi].data(), n1, w.data(), we.data());
                        orb.push_back(std::move(we));
                    }
                    const int nm = static_cast<int>(orb.size());
                    std::vector<long double> G(static_cast<std::size_t>(nm) * nm, 0.0L);
                    for (int j = 0; j < nm; ++j)
                        for (int i = 0; i < nm; ++i)
                            G[i + j * nm] = static_cast<long double>(
                                m1_inner(orb[i].data(), orb[j].data(), orth_metric, n1));
                    std::vector<long double> U(nm * nm), s(nm), V(nm * nm), mem(2 * nm * nm + nm);
                    int orank = 0;
                    if (jacobiSVD(G.data(), nm, nm, U.data(), s.data(), V.data(), mem.data()) == 0){
                        const long double cut = std::max(1e-10L * (s.empty() ? 0.0L : s[0]),
                            static_cast<long double>(svd_tol) * svd_tol);
                        for (int i = 0; i < nm; ++i)
                            if (s[i] > cut) ++orank;
                    }
                    if (orank < nm)
                        continue;
                }
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
            auto lo = V1.m_order.LocalOrderOnTet(DofT::TetOrder(j));
            if (lo.etype != etype || lo.nelem != 0)
                continue;
            if (dof_geom_in_V0(V0, lo, n0))
                continue;
            std::vector<double> seed;
            project_onto_ker(ker_basis, j, seed);
            W.push_back(std::move(seed));
        }
        // If stype mismatch made dof_geom_in_V0 miss V0 edge dofs, fall back to all dofs on elem.
        if (W.empty()){
            for (int j = 0; j < n1; ++j){
                auto lo = V1.m_order.LocalOrderOnTet(DofT::TetOrder(j));
                if (lo.etype != etype || lo.nelem != 0)
                    continue;
                std::vector<double> seed;
                project_onto_ker(ker_basis, j, seed);
                W.push_back(std::move(seed));
            }
        }
        if (static_cast<int>(W.size()) < n_comp){
            std::ostringstream oss;
            oss << err_prefix << ": etype=" << int(etype)
                << " not enough seeds (" << W.size() << " < " << n_comp << ")";
            throw std::runtime_error(oss.str());
        }
        if (static_cast<int>(W.size()) > n_comp){
            std::vector<std::pair<double, int>> score(W.size());
            for (std::size_t i = 0; i < W.size(); ++i)
                score[i] = {m1_inner(W[i].data(), W[i].data(), orth_metric, n1), static_cast<int>(i)};
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
            m1_orthogonalize(w, accepted_for_orth, orth_metric, n1, svd_tol);
        // Re-average to restore equivariance after non-equivariant projection.
        matrix_reynolds(W, n1, n_comp, stab_idx, P_cache, etype, 0, out_stype, all_perms);
        equivariant_m1_orthonormalize(W, n1, n_comp, orth_metric, svd_tol);
        kept = std::move(W);
    }

    if (static_cast<int>(kept.size()) != n_comp){
        std::ostringstream oss;
        oss << err_prefix << ": etype=" << int(etype)
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
    oss << err_prefix << ": equivariant basis has wrong dimension: got "
        << collected_coefs.size() << ", expected " << n10
        << " (types=" << type_repr.size() << ")";
    throw std::runtime_error(oss.str());
    }

    auto dof_map = complement_dof_map_from_tags(collected_tags);
    if (static_cast<int>(dof_map.NumDofOnTet()) != n10){
    std::ostringstream oss;
    oss << err_prefix << ": DofMap size mismatch: map has "
        << dof_map.NumDofOnTet() << ", expected " << n10;
    throw std::runtime_error(oss.str());
    }
    sort_dof_tags(collected_tags, collected_coefs);

    std::vector<double> Psi(n10*n1);
    for (int k = 0; k < n10; ++k)
        std::copy(collected_coefs[k].begin(), collected_coefs[k].end(), Psi.data() + k*n1);

    std::vector<double> dual(n10*n1, 0.0);
    if (dual_metric){
        // Gram Gij = <Psi_i, Psi_j>_{dual_metric} is SPD; solve G X = Psi·G_metric via Cholesky
        // in long double (avoid LU of G and avoid forming G*G^T).
        std::vector<long double> Gram(static_cast<std::size_t>(n10) * n10, 0.0L);
        for (int j = 0; j < n10; ++j)
            for (int i = 0; i < n10; ++i)
                Gram[i + j * n10] = static_cast<long double>(
                    m1_inner(Psi.data() + i * n1, Psi.data() + j * n1, dual_metric, n1));
        std::vector<long double> PsiG(static_cast<std::size_t>(n10) * n1, 0.0L);
        for (int i = 0; i < n10; ++i)
            for (int j = 0; j < n1; ++j){
                long double s = 0;
                for (int k = 0; k < n1; ++k)
                    s += static_cast<long double>(Psi[i * n1 + k])
                       * static_cast<long double>(dual_metric[k + j * n1]);
                PsiG[i + j * n10] = s;
            }
        // Diagnose Gram before Cholesky: silent LLT failure on singular G yields dual≈0 → bio_err≈n10.
        {
            std::vector<long double> U(static_cast<std::size_t>(n10) * n10), s(n10), V(static_cast<std::size_t>(n10) * n10);
            std::vector<long double> mem(2 * static_cast<std::size_t>(n10) * n10 + n10);
            std::vector<long double> Gcopy = Gram;
            if (jacobiSVD(Gcopy.data(), n10, n10, U.data(), s.data(), V.data(), mem.data()) == 0){
                int grank = 0;
                const long double gmax = s.empty() ? 0.0L : s[0];
                const long double gcut = std::max(static_cast<long double>(svd_tol) * svd_tol,
                    (gmax > 0 ? gmax : 1.0L) * 1e-14L);
                for (int i = 0; i < n10; ++i)
                    if (s[i] > gcut) ++grank;
                if (grank < n10){
                    std::ostringstream oss;
                    oss << err_prefix << ": Psi Gram (dual_metric) rank-deficient: rank=" << grank
                        << " n10=" << n10 << " sigma=[";
                    for (int i = 0; i < n10; ++i){
                        if (i) oss << ",";
                        oss << static_cast<double>(s[i]);
                    }
                    oss << "]";
                    throw std::runtime_error(oss.str());
                }
            }
        }
        std::vector<long double> chol_mem(static_cast<std::size_t>(n10) * (n10 + 1) / 2);
        cholesky_solve(Gram.data(), PsiG.data(), n10, n1, PsiG.data(), chol_mem.data());
        for (int i = 0; i < n10; ++i)
            for (int j = 0; j < n1; ++j)
                dual[i * n1 + j] = static_cast<double>(PsiG[i + j * n10]);
    } else {
        std::vector<double> Big(n1*n1), imem(2*n1), lu_mem(2*n1*n1);
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
            throw std::runtime_error(err_prefix + ": Big matrix is ill-conditioned");

        for (int k = 0; k < n10; ++k)
            for (int j = 0; j < n1; ++j)
                dual[k*n1 + j] = BigInv[j + k*n1];
    }

    // Biorthogonality: Euclidean Psi·B^T = I for [Psi;C] duals;
    // M1-duals satisfy B = (Psi M Psi^T)^{-1} Psi M, so check Psi·B^T in the same layout.
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
        throw std::runtime_error(err_prefix + ": biorthogonality check failed (err=" + std::to_string(bio_err) + ")");

    EquivariantComplementBasis out;
    out.basis_coefs = std::move(Psi);
    out.dual_coefs = std::move(dual);
    out.dof_tags = std::move(collected_tags);
    out.dof_map = std::move(dof_map);
    return out;
}


} // namespace Ani
