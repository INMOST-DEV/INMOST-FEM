//
// ComplementFemSpace: V1 - V0 = V1 ∩ V0^{perp_L2}
//

#ifndef CARNUM_ANI_COMPLEMENT_FEMSPACE_H
#define CARNUM_ANI_COMPLEMENT_FEMSPACE_H

#include "fem_space.h"
#include "geometry.h"

namespace Ani {

struct ComplementFemSpace : public BaseFemSpace {
    std::shared_ptr<BaseFemSpace> m_V1;
    std::shared_ptr<BaseFemSpace> m_V0;
    std::vector<double> m_basis_coefs; ///< Psi, row-major n10 x n1
    std::vector<double> m_dual_coefs;  ///< B,   row-major n10 x n1
    std::vector<DofT::LocalOrder> m_dof_tags;

    ComplementFemSpace() = default;

    static ComplementFemSpace make(
        std::shared_ptr<BaseFemSpace> V1,
        std::shared_ptr<BaseFemSpace> V0,
        uint quad_order = 0);

    BaseTypes gatherType() const override { return BaseTypes::ComplementType; }
    std::shared_ptr<BaseFemSpace> subSpace(const int* ext_dims, int ndims) const override;
    std::shared_ptr<BaseFemSpace> copy() const override;
    std::string typeName() const override;
    bool operator==(const BaseFemSpace& other) const override;

    uint dim() const override { return m_V1 ? m_V1->dim() : 0; }
    uint order() const override { return m_V1 ? m_V1->order() : std::numeric_limits<uint>::max(); }

    void evalBasisFunctions(const Expr& lmb, const Expr& grad_lmb, Expr* phi) const override;
    void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const override;
    PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const override;

    BandDenseMatrixX<> applyIDEN(AniMemoryX<>& mem, ArrayView<>& U) const override;
    OpMemoryRequirements memIDEN(uint nquadpoints, uint fusion = 1) const override;
    BandDenseMatrixX<> applyGRAD(AniMemoryX<>& mem, ArrayView<>& U) const override;
    OpMemoryRequirements memGRAD(uint nquadpoints, uint fusion = 1) const override;
    BandDenseMatrixX<> applyDUDX(AniMemoryX<>& mem, ArrayView<>& U, uchar k) const override;
    OpMemoryRequirements memDUDX(uint nquadpoints, uint fusion, uchar k) const override;
};

FemSpace operator-(const FemSpace& V1, const FemSpace& V0);

} // namespace Ani

#endif // CARNUM_ANI_COMPLEMENT_FEMSPACE_H
