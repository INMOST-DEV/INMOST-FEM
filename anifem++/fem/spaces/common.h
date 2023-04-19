//
// Created by Liogky Alexey on 10.02.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_SPACES_COMMON_H
#define CARNUM_FEM_ANIINTERFACE_SPACES_COMMON_H

#include "anifem++/fem/operators.h"
#include "anifem++/fem/fem_space.h"
#include "anifem++/fem/quadrature_formulas.h"
#include <map>
#include <string>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <array>
#include <algorithm>
#include <cassert>

namespace Ani{
    enum FemFamily{
        Polynomial = 1,
        Nedelec = 2,
        RaviantThomas = 3,
        CrouzeixRaviant = 4,
        Bubble = 5,
    };
    static std::string FemFamilyName(uint family_id){
        std::map<uint, std::string> names{
            {Polynomial, "FEM_P"},
            {Nedelec, "FEM_ND"},
            {RaviantThomas, "FEM_RT"},
            {CrouzeixRaviant, "FEM_CR"},
            {Bubble, "FEM_B"},
        };
        auto it = names.find(family_id);
        return it != names.end() ? it->second : "UNKNOWN";
    }

    struct FemTraits: public UniteFemSpace{
        virtual uint polyOrder() const = 0;
        bool operator==(const BaseFemSpace& other) const override{
            if (other.gatherType() != gatherType()) return false;
            auto& a = *static_cast<const UniteFemSpace*>(&other);
            if (a.familyType() != familyType()) return false;
            auto& b = *static_cast<const FemTraits*>(&other);
            return b.polyOrder() == polyOrder();
        }
    };

    template<int FAMILY, int POLY_ORDER>
    struct FemTraitsT: public FemTraits{
        uint familyType() const override { return FAMILY; }
        uint polyOrder() const override { return POLY_ORDER; }
        std::string typeName() const override { return FemFamilyName(FAMILY) + std::to_string(POLY_ORDER); };
    };

    template<typename FEMTYPE>
    struct Basis{
        //should define static method eval() 
    };

    template<int FAMILY, int FEM_TYPE>
    struct GenSpace: public FemTraitsT<FAMILY, Operator<IDEN, FemFix<FEM_TYPE>>::Order::value>{
        using uchar = BaseFemSpace::uchar;
        using uint = BaseFemSpace::uint;
        using DofMap = BaseFemSpace::DofMap;
        using TetGeomSparsity = BaseFemSpace::TetGeomSparsity;
        using EvalFunc = BaseFemSpace::EvalFunc;
        using EvalFunctor = BaseFemSpace::EvalFunctor;
        using BaseTypes = BaseFemSpace::BaseTypes;

        GenSpace() { BaseFemSpace::m_order = DofT::DofMap(Dof<FemFix<FEM_TYPE>>::Map()); }
        uint dim() const override { return Operator<IDEN, FemFix<FEM_TYPE>>::Dim::value; }
        uint order() const override { return Operator<IDEN, FemFix<FEM_TYPE>>::Order::value; }
        void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const override { Basis<FemFix<FEM_TYPE>>::eval(lmb, grad_lmb, phi); } 
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const override { 
            (void)idof_on_tet, (void) max_quad_order;
            PlainMemoryX<> res;
            res.dSize = fusion*Operator<IDEN, FemFix<FEM_TYPE>>::Dim::value;
            return res;
        }
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> wmem, void* user_data = nullptr, uint max_quad_order = 5) const{
            assert(wmem.ge(interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order)) && "Not enough of work memory");
            double* mem = wmem.ddata; //[fusion*Operator<IDEN, FemFix<FEM_TYPE>>::Dim::value]
            Dof<FemFix<FEM_TYPE>>::interpolate(XYZ, f, mem, udofs, idof_on_tet, fusion, user_data, max_quad_order);
        }
        std::shared_ptr<BaseFemSpace> copy() const override { return std::make_shared<GenSpace<FAMILY, FEM_TYPE>>(*this); }
    private:
        template<int DIM, bool is3Davail = true>
        struct InternalOp{
            template<OperatorType OP>
            using OrderT = typename Operator<OP, FemFix<FEM_TYPE>>::Order;
            template<OperatorType OP>
            static BandDenseMatrixX<> appl(AniMemoryX<> &mem, ArrayView<> &U){
                DenseMatrix<> lU = Operator<OP, FemFix<FEM_TYPE>>::template apply<double, int>(mem, U);
                uint bshift = mem.busy_mtx_parts > 0 ? 1 : 0;
                BandDenseMatrixX<> res(1, mem.MTX.data + mem.busy_mtx_parts, mem.MTXI_ROW.data + mem.busy_mtx_parts + bshift, mem.MTXI_COL.data + mem.busy_mtx_parts + bshift);
                res.data[0] = lU;
                res.stRow[0] = 0; res.stRow[1] = Operator<OP, FemFix<FEM_TYPE>>::Dim::value;
                res.stCol[0] = 0; res.stCol[1] = Operator<OP, FemFix<FEM_TYPE>>::Nfa::value;
                ++mem.busy_mtx_parts;
                return res;
            }
            template<OperatorType OP>
            static OpMemoryRequirements memreq(uint nquadpoints, uint fusion){
                OpMemoryRequirements res;
                res.mtx_parts = 1;
                Operator<OP, FemFix<FEM_TYPE>>::template memoryRequirements<double, int>(fusion, nquadpoints, res.Usz, res.extraRsz, res.extraIsz);
                return res;
            }
        };
        template<int DIM>
        struct InternalOp<DIM, false>{
            template<OperatorType OP>
            using OrderT = typename std::integral_constant<int, 0>;
            template<OperatorType OP>
            static BandDenseMatrixX<> appl(AniMemoryX<> &mem, ArrayView<> &U){
                throw std::runtime_error("This operator defined only for 3D fem spaces");
                (void) mem; (void) U;
                return BandDenseMatrixX<>();
            }
            template<OperatorType OP>
            static OpMemoryRequirements memreq(uint nquadpoints, uint fusion){
                throw std::runtime_error("This operator defined only for 3D fem spaces");
                (void) nquadpoints; (void) fusion;
                return OpMemoryRequirements();
            }
        };
        template<bool is3Drequired = false>
        using ImplOP = InternalOp<Operator<IDEN, FemFix<FEM_TYPE>>::Dim::value, !is3Drequired || Operator<IDEN, FemFix<FEM_TYPE>>::Dim::value==3 >;
        
    public:
        BandDenseMatrixX<> applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const override{ return ImplOP<false>::template appl<IDEN>(mem, U); }
        OpMemoryRequirements memIDEN(uint nquadpoints, uint fusion = 1) const override{ return ImplOP<false>::template memreq<IDEN>(nquadpoints, fusion); }
        BandDenseMatrixX<> applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const override{ return ImplOP<false>::template appl<GRAD>(mem, U); }
        OpMemoryRequirements memGRAD(uint nquadpoints, uint fusion = 1) const override{ return ImplOP<false>::template memreq<GRAD>(nquadpoints, fusion); }
        BandDenseMatrixX<> applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const override{ return ImplOP<true>::template appl<DIV>(mem, U); }
        OpMemoryRequirements memDIV(uint nquadpoints, uint fusion = 1) const override{ return ImplOP<true>::template memreq<DIV>(nquadpoints, fusion); }
        BandDenseMatrixX<> applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const override{ return ImplOP<true>::template appl<CURL>(mem, U); }
        OpMemoryRequirements memCURL(uint nquadpoints, uint fusion = 1) const override{ return ImplOP<true>::template memreq<CURL>(nquadpoints, fusion); }
        BandDenseMatrixX<> applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, unsigned char k) const override{ 
            switch (k) {
                case 0: return ImplOP<false>::template appl<DUDX>(mem, U); 
                case 1: return ImplOP<false>::template appl<DUDY>(mem, U);
                case 2: return ImplOP<false>::template appl<DUDZ>(mem, U);
            } 
            return BandDenseMatrixX<>();
        }
        OpMemoryRequirements memDUDX(uint nquadpoints, uint fusion, unsigned char k) const override{ 
            switch(k){
                case 0: return ImplOP<false>::template memreq<DUDX>(nquadpoints, fusion); 
                case 1: return ImplOP<false>::template memreq<DUDY>(nquadpoints, fusion); 
                case 2: return ImplOP<false>::template memreq<DUDZ>(nquadpoints, fusion); 
            }
            return OpMemoryRequirements(); 
        }
        
        uint orderDIV() const override { return ImplOP<true>::template OrderT<DIV>::value; }
        uint orderCURL() const override { return ImplOP<true>::template OrderT<CURL>::value; }
    };
};

#endif //CARNUM_FEM_ANIINTERFACE_SPACES_COMMON_H