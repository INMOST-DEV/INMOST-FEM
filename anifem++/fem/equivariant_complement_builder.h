//
// Shared S4-equivariant complement basis builder (L2 / energy).
//

#ifndef CARNUM_ANI_EQUIVARIANT_COMPLEMENT_BUILDER_H
#define CARNUM_ANI_EQUIVARIANT_COMPLEMENT_BUILDER_H

#include "fem_space.h"
#include <string>
#include <vector>

namespace Ani {

struct EquivariantComplementBasis {
    std::vector<double> basis_coefs; ///< Psi, row-major n10 x n1
    std::vector<double> dual_coefs;  ///< B,   row-major n10 x n1
    std::vector<DofT::LocalOrder> dof_tags;
    DofT::UniteDofMap dof_map;
};

/// Build an S4-equivariant frame of ker_basis with fiber orthonormalization in
/// the SPD metric `orth_metric` (n1 x n1, column-major).
/// Duals: if `dual_metric` is null, from [Psi; C]^{-1}; otherwise B = (Psi G Psi^T)^{-1} Psi G
/// with G = dual_metric (n1 x n1). Use the latter when span(Psi) ⊕ V0 is not a direct
/// algebraic split of R^{n1} (e.g. energy complements).
/// `ker_basis` must have size n10.
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
    const double* dual_metric = nullptr);

} // namespace Ani

#endif // CARNUM_ANI_EQUIVARIANT_COMPLEMENT_BUILDER_H
