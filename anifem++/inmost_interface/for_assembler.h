//
// Created by Liogky Alexey on 05.04.2022.
//

#ifndef CARNUM_FORASSEMBLER_H
#define CARNUM_FORASSEMBLER_H

#include "anifem++/fem/operators.h"
#include "anifem++/fem/spaces/spaces.h"
#include "elemental_assembler.h"
#include <memory>
#include <array>
#include <vector>
#include <tuple>
#include <utility>
#include <functional>

namespace Ani {
    inline INMOST::ElementType AniGeomMaskToInmostElementType(DofT::uint ani_geom_mask);
    inline std::array<uint, 4> DofTNumDofsToGeomNumDofs(std::array<DofT::uint, DofT::NGEOM_TYPES> num_dofs);
    template<typename FemType>
    struct HelperFactory;

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_P0>> {
        static std::shared_ptr<FemExprDescr::SimpleDiscrSpace> build();
    };

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_P1>> {
        static std::shared_ptr<FemExprDescr::SimpleDiscrSpace> build();
    };

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_P2>> {
        static std::shared_ptr<FemExprDescr::SimpleDiscrSpace> build();
    };

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_P3>> {
        static std::shared_ptr<FemExprDescr::SimpleDiscrSpace> build();
    };

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_CR1>> {
        static std::shared_ptr<FemExprDescr::SimpleDiscrSpace> build();
    };

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_RT0>> {
        static std::shared_ptr<FemExprDescr::UniteDiscrSpace> build() {
            std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
            NumDofs[5] = 1;
            return std::make_shared<FemExprDescr::UniteDiscrSpace>(3, NumDofs, 0, Ani::FEM_RT0);
        }
    };

    template<>
    struct HelperFactory<Ani::FemFix<Ani::FEM_ND0>> {
        static std::shared_ptr<FemExprDescr::UniteDiscrSpace> build() {
            std::array<int, FemExprDescr::NGEOM_TYPES> NumDofs{0};
            NumDofs[4] = 1;
            return std::make_shared<FemExprDescr::UniteDiscrSpace>(3, NumDofs, 0, Ani::FEM_ND0);
        }
    };

    template<int DIM, int OP>
    struct HelperFactory<Ani::FemVec<DIM, OP>> {
        static std::shared_ptr<FemExprDescr::SimpleVectorDiscrSpace> build() {
            return std::make_shared<FemExprDescr::SimpleVectorDiscrSpace>(DIM,
                                                                          HelperFactory<typename Ani::FemVec<DIM, OP>::Base>::build());
        }
    };

    template<int DIM, typename T>
    struct HelperFactory<Ani::FemVecT<DIM, T>> {
        static std::shared_ptr<FemExprDescr::SimpleVectorDiscrSpace> build() {
            return std::make_shared<FemExprDescr::SimpleVectorDiscrSpace>(DIM,
                                                                          HelperFactory<typename Ani::FemVecT<DIM, T>::Base>::build());
        }
    };

    namespace FemComDetails {
        template<size_t I, size_t N, typename T>
        struct ForEach {
            static void item(std::vector<std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper>> &v) {
                v[I] = HelperFactory<std::tuple_element<I, typename T::Base>>::build();
                ForEach<I + 1, N, T>::item();
            }
        };

        template<size_t N, typename T>
        struct ForEach<N, N, T> {
            static void item(std::vector<std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper>> &v) {}
        };
    };


    template<typename ...T>
    struct HelperFactory<Ani::FemCom<T...>> {
        static std::shared_ptr<FemExprDescr::ComplexSpaceHelper> build() {
            auto nVar = std::tuple_size<typename Ani::FemCom<T...>::Base>::value;
            std::vector<std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper>> v(nVar);
            FemComDetails::ForEach<0, nVar, Ani::FemCom<T...>>::item(v);
            return std::make_shared<FemExprDescr::ComplexSpaceHelper>(std::move(v));
        }
    };

    ///Create FEM variable description
    template<typename FemType>
    std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper> GenerateHelper() {
        return HelperFactory<FemType>::build();
    }
    std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper> GenerateHelper(const DofT::BaseDofMap& m, uint dim, bool internal_save_dim = true, unsigned int driver_id = UINT_MAX, unsigned int fem_id = UINT_MAX);
    std::shared_ptr<FemExprDescr::BaseDiscrSpaceHelper> GenerateHelper(const BaseFemSpace& fs, bool internal_save_dim = true);

    ///Wrap some variants of elemental matrix and rhs evaluators
    template<typename Real = double >
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F)> f, size_t nRow, size_t nCol);
    template<typename Real = double>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, void* user_data)> f, size_t nRow, size_t nCol);
    template<typename Real = double, typename Int = int>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data)> f,
                                                    size_t nRow, size_t nCol, size_t nw, size_t niw);
    template<typename Real = double, typename Int = int>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw)> f,
                                                    size_t nRow, size_t nCol, size_t nw, size_t niw);

    ///Wrap some variants of elemental matrix evaluators
    template<typename Real = double>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A)> f, size_t nRow, size_t nCol);
    template<typename Real = double>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, void* user_data)> f, size_t nRow, size_t nCol);
    template<typename Real = double, typename Int = int>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data)> f,
                                                    size_t nRow, size_t nCol, size_t nw, size_t niw);
    template<typename Real = double, typename Int = int>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw)> f,
                                                 size_t nRow, size_t nCol, size_t nw, size_t niw);

    ///Wrap some variants of elemental rhs evaluators
    template<typename Real = double>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F)> f, size_t nRow);
    template<typename Real = double>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, void* user_data)> f, size_t nRow);
    template<typename Real = double, typename Int = int>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw, void* user_data)> f,
                                                    size_t nRow, size_t nw, size_t niw);
    template<typename Real = double, typename Int = int>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw)> f,
                                                 size_t nRow, size_t nw, size_t niw);
}

#include "for_assembler.inl"

#endif //CARNUM_FORASSEMBLER_H
