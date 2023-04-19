//
// Created by Liogky Alexey on 05.04.2022.
//

#include "for_assembler.h"
#include <stdexcept>

using namespace Ani;

std::shared_ptr<FemExprDescr::SimpleDiscrSpace> HelperFactory<FemFix<FEM_P0>>::build() {
    std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
    NumDofs[FemExprDescr::CELL] = 1;
    auto res = std::make_shared<FemExprDescr::SimpleDiscrSpace>(NumDofs, 0, FEM_P0);
    res->coordAt = Ani::DOF_coord<FemFix<FEM_P0>>::template at<double>;
    return res;
}

std::shared_ptr<FemExprDescr::SimpleDiscrSpace> HelperFactory<Ani::FemFix<Ani::FEM_P1>>::build() {
    std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
    NumDofs[FemExprDescr::NODE] = 1;
    auto res = std::make_shared<FemExprDescr::SimpleDiscrSpace>(NumDofs, 0, Ani::FEM_P1);
    res->coordAt = Ani::DOF_coord<FemFix<FEM_P1>>::template at<double>;
    return res;
}

std::shared_ptr<FemExprDescr::SimpleDiscrSpace> HelperFactory<Ani::FemFix<Ani::FEM_P2>>::build() {
    std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
    NumDofs[FemExprDescr::NODE] = 1, NumDofs[FemExprDescr::EDGE] = 1;
    auto res = std::make_shared<FemExprDescr::SimpleDiscrSpace>(NumDofs, 0, Ani::FEM_P2);
    res->coordAt = Ani::DOF_coord<FemFix<FEM_P2>>::template at<double>;
    return res;
}

std::shared_ptr<FemExprDescr::SimpleDiscrSpace> HelperFactory<Ani::FemFix<Ani::FEM_P3>>::build() {
    std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
    NumDofs[FemExprDescr::NODE] = 1, NumDofs[FemExprDescr::EDGE] = 2, NumDofs[FemExprDescr::FACE] = 1;
    auto res = std::make_shared<FemExprDescr::SimpleDiscrSpace>(NumDofs, 0, Ani::FEM_P3);
    res->coordAt = Ani::DOF_coord<FemFix<FEM_P3>>::template at<double>;
    return res;
}

std::shared_ptr<FemExprDescr::SimpleDiscrSpace> HelperFactory<Ani::FemFix<Ani::FEM_CR1>>::build() {
    std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
    NumDofs[FemExprDescr::FACE] = 1;
    auto res = std::make_shared<FemExprDescr::SimpleDiscrSpace>(NumDofs, 0, Ani::FEM_CR1);
    res->coordAt = Ani::DOF_coord<FemFix<FEM_CR1>>::template at<double>;
    return res;
}

std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper> Ani::GenerateHelper(const DofT::BaseDofMap& m, uint dim, bool internal_save_dim, unsigned int driver_id, unsigned int fem_id){
    auto t = m.ActualType();
    using namespace DofT;
    switch(t){
        case static_cast<uint>(BaseDofMap::BaseTypes::UniteType):{
            auto& lm = *static_cast<const UniteDofMap*>(&m);
            auto utm_dofs = lm.NumDofs();
            unsigned char lookup[] = {0, 1, 3, 5, 2, 4};
            std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
            for (int i = 0; i < static_cast<int>(FemExprDescr::NGEOM_TYPES); ++i)
                NumDofs[i] = utm_dofs[lookup[i]];
            if (dim == 1 && internal_save_dim)
                return std::make_shared<FemExprDescr::SimpleDiscrSpace>(NumDofs, driver_id, fem_id);
            else
                return std::make_shared<FemExprDescr::UniteDiscrSpace>(internal_save_dim ? dim : 0U, NumDofs, driver_id, fem_id);    
        }
        case static_cast<uint>(BaseDofMap::BaseTypes::VectorType):
        case static_cast<uint>(BaseDofMap::BaseTypes::VectorTemplateType):{
            uint vdim = m.NestedDim();
            int vdimi = vdim;
            return std::make_shared<FemExprDescr::SimpleVectorDiscrSpace>(vdim, GenerateHelper(*(m.GetSubDofMap(&vdimi, 1)), dim / vdim, internal_save_dim, driver_id, fem_id));
        }
        case static_cast<uint>(BaseDofMap::BaseTypes::ComplexType):
        case static_cast<uint>(BaseDofMap::BaseTypes::ComplexTemplateType):{
            uint vdim = m.NestedDim();
            if (!internal_save_dim || dim == vdim){
                std::vector<std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper>> bds(vdim);
                for (int i = 0; i < static_cast<int>(vdim); ++i)
                    bds[i] = GenerateHelper(*(m.GetSubDofMap(&i, 1)), 1, internal_save_dim, driver_id, fem_id);
                return std::make_shared<FemExprDescr::ComplexSpaceHelper>(std::move(bds));    
            } else 
                throw std::runtime_error("Can't compute physical dimesions for complex type"); 
        }
        default:
            throw std::runtime_error("Don't known rules for converting of BaseDofMap with gather type = " + std::to_string(static_cast<int>(t)));
    }
    return nullptr;
}
std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper> Ani::GenerateHelper(const BaseFemSpace& fs, bool internal_save_dim){
    auto t = fs.gatherType();
    switch (t){
        case BaseFemSpace::BaseTypes::UniteType:{
            auto& lfs = *static_cast<const UniteFemSpace*>(&fs);
            return GenerateHelper(*lfs.m_order.target<>(), lfs.dim(), internal_save_dim, 0U, lfs.familyType());
        }
        case BaseFemSpace::BaseTypes::UnionType:{
            auto& lfs = *static_cast<const UnionFemSpace*>(&fs);
            std::vector<std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper>> bds(lfs.m_subsets.size());
            for (uint i = 0; i < lfs.m_subsets.size(); ++i)
                bds[i] = GenerateHelper(*lfs.m_subsets[i], internal_save_dim && (i == 0));
            return std::make_shared<FemExprDescr::ComplexSpaceHelper>(std::move(bds));
        }
        case BaseFemSpace::BaseTypes::VectorType:{
            auto& lfs = *static_cast<const VectorFemSpace*>(&fs);
            return std::make_shared<FemExprDescr::SimpleVectorDiscrSpace>(lfs.vec_dim(), GenerateHelper(*lfs.m_base, internal_save_dim));
        }
        case BaseFemSpace::BaseTypes::ComplexType:{
            auto& lfs = *static_cast<const ComplexFemSpace*>(&fs);
            std::vector<std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper>> bds(lfs.m_spaces.size());
            for (uint i = 0; i < lfs.m_spaces.size(); ++i)
                bds[i] = GenerateHelper(*lfs.m_spaces[i], internal_save_dim);
            return std::make_shared<FemExprDescr::ComplexSpaceHelper>(std::move(bds));
        }
        default:
            throw std::runtime_error("Don't known rules for converting of BaseFemSpace with gather type = " + std::to_string(static_cast<int>(t)));
    }
    return nullptr;
}