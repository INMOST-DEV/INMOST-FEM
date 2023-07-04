//
// Created by Liogky Alexey on 10.01.2023.
//

#ifndef CARNUM_ANI_FEMSPACE_H
#define CARNUM_ANI_FEMSPACE_H

#include "expr_interface.h"
#include "tetdofmap.h"
#include "fem_memory.h"
#include "operators.h"
#include "diff_tensor.h"
#include <initializer_list>
#include <array>
#include <vector>
#include <functional>
#include <numeric>
#include <memory>
#include <cmath>
#include <cassert>
#include <type_traits>
#include <limits>
#include <string>
#include <utility>
#include <algorithm>

/**
* Vector of d.o.f's of physical variable should met to the following rules
*
*      A) First, basis function associated with vertices are enumerated
*         in the same order as the vertices 1,2,3,4;
*
*      B) Second, basis function associated with edges are enumerated
*         in the same order as egdes 12,13,14,23,24 and 34;
*   
*        B1) On the edge, the d.o.f.s are ordered so that first there are d.o.f.s 
*            from the symmetry group S1, and then pairs of d.o.f.s from the symmetry group S2
*
*      C) Third, basis function associated with faces are enumerated
*         in the same order as faces 123,234,341 and 412;
*
*        C1) On the face, degrees of freedom are enumerated as group of symmteries S1, S3, S6
*            where for every Sk (k = 1, 3, 6) d.o.f's enumerated by k d.o.f.'s belonging to the same Sk group
*
*      D) Fourth, basis function associated with cells
*
*        D1) On the cell, degrees of freedom are enumerated as group of symmteries S1, S4, S6, S12, S24
*            where for every Sk (k = 1, 4, 6, 12, 24) d.o.f's enumerated by k d.o.f.'s belonging to the same Sk group
*
*      E) The vector basis functions with several degrees of freedom per
*         a mesh object (vertex, edge, face) are enumerated first by the
*         corresponding mesh objects (vertex, edge, face) and then by space coordinates, x,y,z;
*
*      F) The vector with several physical variables are numerated first by
*         basis functions of the variables and than by the physical variables order in the vector.
*
*      G) Compound variables discretized by mixture of different types of FEM basis spaces 
*         (e.g. MINI1 = P1 + Bubble) are enumerated first by basis functions of the basis space 
*         and then by basis space order in mixture
**/

namespace Ani{
    struct ApplyOpFromSpaceView;
    struct BaseFemSpace{
        using uchar = unsigned char;
        using uint = unsigned;
        using DofMap = DofT::DofMap;
        using TetGeomSparsity = DofT::TetGeomSparsity;
        using EvalFunc = int(*)(const std::array<double, 3>& X, double* res, uint dim, void* user_data);
        using EvalFunctor = std::function<int(const std::array<double, 3>& X, double* res, uint dim, void* user_data)>;
        enum class BaseTypes{
            UniteType = 0,  ///< contains set of actions of basis functions
            UnionType = 1,  ///< union of several sets of basis functions into one variable (e.g. FEM_P1 + FEM_B, i.e. linear polynomial space + bubble function)
            VectorType = 2, ///< form vector-variable from space for element, perform cartesian power operation, e.g. FEM_P1^3
            ComplexType = 3,///< form vector-variable from vector of spaces for elements, perform cartesian multiplication of spaces, e.g. FEM_P2 x FEM_P1
            NoType = 4,
        };
        
        DofMap m_order;
        DofT::BaseDofMap& dofMap(){ return *m_order.base(); }
        const DofT::BaseDofMap& dofMap() const { return const_cast<BaseFemSpace*>(this)->dofMap(); }

        virtual BaseTypes gatherType() const = 0;
        /// @brief Get space type name 
        virtual std::string typeName() const = 0;
        virtual bool operator==(const BaseFemSpace& other) const = 0;

        /// @return dimension of space elements
        virtual uint dim() const  = 0;
        /// @return polynomial order of basis functions of the space or MAX_UINT if order is infinity
        virtual uint order() const { return std::numeric_limits<uint>::max(); }
        /// @brief Evaluate basis function on tetra in specified point
        /// @note designed specially for using with symbolic expressions, for numerical evaluation applyIDEN(...) is prefered 
        /// @param[in] l is baricentric coordinate on tetrahedron
        /// @param[out] phi are space basis functions on tetra 
        void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, std::vector<Expr>& phi) const;
        /// @brief Same as previous specialization but don't allocate memory for phi vector
        /// @param[in,out] phi are memory to save basis functions on tetra
        /// @see evalBasisFunctions
        virtual void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const = 0;
        /// @brief Set interpolation of the vector-function on idof_on_tet-th degree of freedom
        /// @param XYZ is tetrahedron
        /// @param f is vector-function to be interpolated, return [f_0, f_1, .., f_{r-1}], r = fusion
        /// @param udofs[fusion] are vectors to save result of interpolation
        /// @param idof_on_tet is index of d.o.f.
        /// @param user_data is some additional data for function f
        /// @param max_quad_order is maximal quadrature order will be used (if required)
        virtual void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const = 0;
        virtual PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const { (void)idof_on_tet, (void) fusion, (void) max_quad_order;  return PlainMemoryX<>(); }
        /// @brief Set interpolation of the function on idof_on_tet-th degree of freedom
        /// @param XYZ is tetrahedron
        /// @param f is function to be interpolated
        /// @param udofs is vector to save result of interpolation
        /// @param idof_on_tet is index of d.o.f.
        /// @param user_data is some additional data for function f
        /// @param max_quad_order is maximal quadrature order will be used (if required)
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const{ interpolateOnDOF(XYZ, f, &udofs, idof_on_tet, 1, mem, user_data, max_quad_order); }
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, uint max_quad_order) const { return interpolateOnDOF_mem_req(idof_on_tet, 1, max_quad_order); }
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const;
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const{ interpolateOnDOF(XYZ, f, &udofs, idof_on_tet, 1, wmem, user_data, max_quad_order); }
        
        /// For operator A = Identity save matrix U: U[k + DimOp * (n + Npnts * ( i + NumDofOnTet*r ))] = (A phi^i)_k ( x_{nr} ) 
        /// @param[in] mem is memory for all operations
        /// @param[in,out] U memory to store result
        /// @return view on assembled local matrix 
        virtual BandDenseMatrixX<> applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const = 0;
        virtual OpMemoryRequirements memIDEN(uint nquadpoints, uint fusion = 1) const = 0;
        inline uint dimIDEN() const { return dim(); }
        inline uint orderIDEN() const { return order(); }
        /// For operator A = Gradient save matrix U
        /// @see  applyIDEN for parameter descrition
        virtual BandDenseMatrixX<> applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const = 0;
        virtual OpMemoryRequirements memGRAD(uint nquadpoints, uint fusion = 1) const = 0;
        inline uint dimGRAD() const { return dim()*3; }
        inline uint orderGRAD() const { return orderDiff(); }
        /// For operator A = DIV save matrix U
        /// @see  applyIDEN for parameter descrition
        virtual BandDenseMatrixX<> applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const;
        virtual OpMemoryRequirements memDIV(uint nquadpoints, uint fusion = 1) const;
        inline uint dimDIV() const { return dim()/3; }
        virtual uint orderDIV() const { return orderDiff(); }
        /// For operator A = CURL save matrix U
        /// @see  applyIDEN for parameter descrition
        virtual BandDenseMatrixX<> applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const;
        virtual OpMemoryRequirements memCURL(uint nquadpoints, uint fusion = 1) const;
        inline uint dimCURL() const { return dim(); }
        virtual uint orderCURL() const { return orderDiff(); }
        /// For operator A = d/dx_k save matrix U
        /// @param k is index of space variable, 0 for x, 1 for y and 2 for z
        /// @see  applyIDEN for other parameter descrition
        virtual BandDenseMatrixX<> applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const;
        virtual OpMemoryRequirements memDUDX(uint nquadpoints, uint fusion, uchar k) const;
        inline uint dimDUDX() const { return dim(); }
        virtual uint orderDUDX(uchar k) const { (void) k; return orderDiff(); }

        /// For operator op assemble matrix U: U[k + DimOp * (n + Npnts * ( i + NumDofOnTet*r ))] = (A phi^i)_k ( x_{nr} ) 
        /// @param[in] op is linear differential operator (including IDEN) 
        /// @param[in] mem is memory for all operations
        /// @param[in,out] U memory to store result
        /// @return view on assembled local matrix Dim
        BandDenseMatrixX<> applyOP(OperatorType op, AniMemoryX<> &mem, ArrayView<> &U) const;
        /// @brief Get memory requirements for specific operation
        /// @param op is linear differential operator
        /// @param nquadpoints is number points to be used to compute matrix (A phi^i)_k ( x_{nr} ) (dimension of n index)
        /// @param fusion is number tetra to be used to compute matrix (A phi^i)_k ( x_{nr} ) (dimension of r index)
        /// @return memory requirements for specific operation
        OpMemoryRequirements memOP(OperatorType op, uint nquadpoints, uint fusion = 1) const;
        /// @brief Get dimension of result of applying operator on element of FEM space 
        uint dimOP(OperatorType op) const;
        /// @brief Get polynomial order of applying operator
        uint orderOP(OperatorType op) const;
        /// @return operator applier  
        ApplyOpFromSpaceView getOP(OperatorType op) const;
    protected:
        template<typename FUNCTOR>
        struct FunctorContainer{
            const FUNCTOR& f;
            void* user_data;

            static int eval_functor(const std::array<double, 3>& X, double* res, uint dim, void* user_data);
        };
    public:
        /// @brief Interpolate function f on d.o.f.'s belongs to region defined by sp
        /// @see interpolateOnDOF for parameters descrition
        virtual void interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc &f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const;
        virtual PlainMemoryX<> interpolateByDOFs_mem_req(uint max_quad_order = 5) const;
        void interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc &f, ArrayView<> udofs, const TetGeomSparsity& sp, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const;
        
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const;
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<>* udofs, int idof_on_tet, int fusion, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const;
        
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type = true>
        inline void interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, int idof_on_tet, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const;
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, BaseFemSpace::EvalFunc>::value, bool>::type = true>
        inline void interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, int idof_on_tet, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const;
        
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const;
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const;
        
        template<typename EVAL_FUNCTOR>
        inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const;
        template<typename EVAL_FUNCTOR>
        inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map, DynMem<>& wmem, void* user_data = nullptr, uint max_quad_order = 5) const;
        
        /// @brief Set value val on d.o.f.'s belongs to region defined by sp
        inline void interpolateConstant(double val, ArrayView<> udofs, const TetGeomSparsity& sp) const;
        inline void interpolateConstant(double val, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map) const;
        /// Perform interpolation of func on d.o.f.'s by solving the equation
        /// \f[ 
        ///   \int_{\Omega_s} [\phi_i \phi_j] u_j dx = \int_{\Omega_s} f \phi_i dx
        /// \f]
        /// where \Omega_s is selected geometrical element
        /// @param XYZ is coords of tetrahedron
        /// @param f is the functor
        /// @param udofs is vector of d.o.f's
        /// @param elem_type is type of element, see DofT namespace
        /// @param ielem is number of element of specified type
        /// @param user_data user-supplied data to be used by functor
        /// @param max_quad_order is maximal quadrature order to be used
        void interpolateOnRegion(const Tetra<const double>& XYZ, const EvalFunctor& f, ArrayView<> udofs, uchar elem_type, int ielem, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const;
        PlainMemoryX<> interpolateOnRegion_mem_req(uchar elem_type, uint max_quad_order) const;
        virtual std::shared_ptr<BaseFemSpace> copy() const = 0;
        virtual std::shared_ptr<BaseFemSpace> subSpace(const int* ext_dims, int ndims) const = 0;
    protected:
        inline uint orderDiff() const;
    }; 
    /// Apply operator from space from space view (without space ownship) 
    struct ApplyOpFromSpaceView: public ApplyOpBase{
        const BaseFemSpace* m_space = nullptr;
        OperatorType m_op = IDEN; 
        uint m_dim = 0;
        uint m_nfa = 0;
        uint m_order = std::numeric_limits<uint>::max();

        BandDenseMatrixX<> operator()(AniMemoryX<> &mem, ArrayView<> &U) const override { return m_space->applyOP(m_op, mem, U); }
        OpMemoryRequirements getMemoryRequirements(uint nquadpoints, uint fusion = 1) const override { return m_space->memOP(m_op, nquadpoints, fusion); }
        uint Nfa() const { return m_nfa; }
        uint Dim() const { return m_dim; }
        uint Order() const { return m_order; }
        uint ActualType() const { return 2; }
        bool operator==(const ApplyOpBase& otherOp) const { return otherOp.ActualType() == ActualType() && m_space == static_cast<const ApplyOpFromSpaceView&>(otherOp).m_space && m_op == static_cast<const ApplyOpFromSpaceView&>(otherOp).m_op; }
        std::shared_ptr<ApplyOpBase> Copy() const { return std::make_shared<ApplyOpFromSpaceView>(*this); } 

        ApplyOpFromSpaceView() = default;
        ApplyOpFromSpaceView(OperatorType op, const BaseFemSpace* sp): m_space{sp}, m_op{op}, m_dim{sp->dimOP(op)}, m_nfa{sp->dofMap().NumDofOnTet()}, m_order{sp->orderOP(op)} {}
        ApplyOpFromSpaceView& Init(OperatorType op, const BaseFemSpace* sp){ m_space = sp, m_op = op, m_dim = sp->dimOP(op),  m_nfa = sp->dofMap().NumDofOnTet(), m_order = sp->orderOP(op); return *this; }
        bool isValid() const { return m_space != nullptr; }
    protected:   
        friend class BaseFemSpace;
    };

    struct UniteFemSpace: public BaseFemSpace{
        BaseTypes gatherType() const final { return BaseFemSpace::BaseTypes::UniteType; }
        std::shared_ptr<BaseFemSpace> subSpace(const int* ext_dims, int ndims) const final{ (void) ext_dims; (void) ndims; return nullptr; }
        /// @return unique identificator of the type
        virtual uint familyType() const = 0;
    };
    struct VectorFemSpace: public BaseFemSpace{
        std::shared_ptr<BaseFemSpace> m_base;
        VectorFemSpace() = default;
        VectorFemSpace(int dim, std::shared_ptr<BaseFemSpace> space): m_base(std::move(space)){ m_order = pow(m_base->m_order, dim); }
        VectorFemSpace& Init(int dim, std::shared_ptr<BaseFemSpace> space);
        BaseTypes gatherType() const final { return BaseFemSpace::BaseTypes::VectorType; }
        std::shared_ptr<BaseFemSpace> subSpace(const int* ext_dims, int ndims) const final;
        std::shared_ptr<BaseFemSpace> copy() const override { return std::make_shared<VectorFemSpace>(*this); }
        std::string typeName() const override;
        bool operator==(const BaseFemSpace& other) const override;

        uint vec_dim() const { return m_order.target<DofT::VectorDofMap>()->m_dim; }
        uint dim() const override { return m_base->dim() * m_order.target<DofT::VectorDofMap>()->m_dim; };
        uint order() const override { return m_base->order(); }
        void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const override;    
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const override;
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const override; 
        void interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const override;
        PlainMemoryX<> interpolateByDOFs_mem_req(uint max_quad_order = 5) const override;

        BandDenseMatrixX<> applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memIDEN(uint nquadpoints, uint fusion = 1) const override;
        BandDenseMatrixX<> applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memGRAD(uint nquadpoints, uint fusion = 1) const override;
        BandDenseMatrixX<> applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memDIV(uint nquadpoints, uint fusion = 1) const override;
        uint orderDIV() const override{ return m_base->orderDIV(); }
        BandDenseMatrixX<> applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memCURL(uint nquadpoints, uint fusion = 1) const override;
        uint orderCURL() const override{ return m_base->orderCURL(); }
        BandDenseMatrixX<> applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const override;
        OpMemoryRequirements memDUDX(uint nquadpoints, uint fusion, uchar k) const override;
        uint orderDUDX(uchar k) const override { return m_base->orderDUDX(k); }
    private:
        static void realloc_rep_mtx(AniMemoryX<> &mem, BandDenseMatrixX<>& mtx, uint nrep);
    };
    struct ComplexFemSpace: public BaseFemSpace{
        std::vector<std::shared_ptr<BaseFemSpace>> m_spaces;

        ComplexFemSpace() = default;
        explicit ComplexFemSpace(std::vector<std::shared_ptr<BaseFemSpace>> spaces): m_spaces(std::move(spaces)){ setup(); }
        ComplexFemSpace& Init(std::vector<std::shared_ptr<BaseFemSpace>> spaces);
        BaseTypes gatherType() const final { return BaseFemSpace::BaseTypes::ComplexType; }
        std::shared_ptr<BaseFemSpace> subSpace(const int* ext_dims, int ndims) const final;
        std::shared_ptr<BaseFemSpace> copy() const override { return std::make_shared<ComplexFemSpace>(*this); }
        std::string typeName() const override;
        bool operator==(const BaseFemSpace& other) const override;
        uint dim() const override { return m_dim; };
        uint order() const override { return m_poly_order; }
        void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const override;
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const override;
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const override;

        BandDenseMatrixX<> applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memIDEN(uint nquadpoints, uint fusion = 1) const override;
        BandDenseMatrixX<> applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memGRAD(uint nquadpoints, uint fusion = 1) const override;
        BandDenseMatrixX<> applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const override;
        OpMemoryRequirements memDUDX(uint nquadpoints, uint fusion, uchar k) const override;
        uint orderDUDX(uchar k) const override { return m_poly_order_DUDX[k]; }
        BandDenseMatrixX<> applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const override;
        OpMemoryRequirements memDIV(uint nquadpoints, uint fusion = 1) const override;
        uint orderDIV() const override { return m_poly_order_DIV; }
        //default applyCURL and memCURL
        uint orderCURL() const override { return m_poly_order_CURL; }

    private:
        void setup();
        BandDenseMatrixX<> applyDIV_1x1x1(AniMemoryX<> &mem, ArrayView<> &U) const;
        BandDenseMatrixX<> applyDIV_1x2(AniMemoryX<> &mem, ArrayView<> &U) const;
        template<typename ...Targs>
        OpMemoryRequirements memOP_internal(uint nquadpoints, uint fusion, OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const;
        template<typename ...Targs>
        BandDenseMatrixX<> applyOP_internal(AniMemoryX<> &mem, ArrayView<> &U, 
            BandDenseMatrixX<>(Ani::BaseFemSpace::* apl)(AniMemoryX<> &, ArrayView<> &, Targs...) const,
            OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const;

        uint m_dim = 0;
        uint m_poly_order = 0;
        uint m_poly_order_DIV = 0;
        uint m_poly_order_CURL = 0;
        std::array<uint, 3> m_poly_order_DUDX = { 0, 0, 0 };
    };

    struct UnionFemSpace: public BaseFemSpace{
        std::vector<std::shared_ptr<BaseFemSpace>> m_subsets;

        UnionFemSpace() = default;
        explicit UnionFemSpace(std::vector<std::shared_ptr<BaseFemSpace>> subsets): m_subsets(std::move(subsets)){ setup(); }
        UnionFemSpace& Init(std::vector<std::shared_ptr<BaseFemSpace>> subsets);
        BaseTypes gatherType() const final { return BaseFemSpace::BaseTypes::UnionType; }
        std::shared_ptr<BaseFemSpace> subSpace(const int* ext_dims, int ndims) const;
        std::shared_ptr<BaseFemSpace> copy() const override { return std::make_shared<UnionFemSpace>(*this); }
        std::string typeName() const override;
        bool operator==(const BaseFemSpace& other) const override;
        uint dim() const override { return m_subsets.empty() ? 0 : m_subsets[0]->dim(); };     
        uint order() const override { return internal_take_max_val<>(&BaseFemSpace::order); }
        uint orderDIV() const override { return internal_take_max_val<>(&BaseFemSpace::orderDIV); }
        uint orderCURL() const override { return internal_take_max_val<>(&BaseFemSpace::orderCURL); }
        uint orderDUDX(uchar k) const override { return internal_take_max_val<uchar>(&BaseFemSpace::orderDUDX, k); }
        void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const override;
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const override;
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const override;

        OpMemoryRequirements memIDEN(uint nquadpoints, uint fusion = 1) const override;
        OpMemoryRequirements memGRAD(uint nquadpoints, uint fusion = 1) const override;
        OpMemoryRequirements memDIV(uint nquadpoints, uint fusion = 1) const override;
        OpMemoryRequirements memCURL(uint nquadpoints, uint fusion = 1) const override;
        OpMemoryRequirements memDUDX(uint nquadpoints, uint fusion, uchar k) const override;
        BandDenseMatrixX<> applyIDEN(AniMemoryX<> &mem, ArrayView<> &U) const override;
        BandDenseMatrixX<> applyGRAD(AniMemoryX<> &mem, ArrayView<> &U) const override;
        BandDenseMatrixX<> applyDIV(AniMemoryX<> &mem, ArrayView<> &U) const override;
        BandDenseMatrixX<> applyCURL(AniMemoryX<> &mem, ArrayView<> &U) const override;
        BandDenseMatrixX<> applyDUDX(AniMemoryX<> &mem, ArrayView<> &U, uchar k) const override;

        DenseMatrix<const double> GetOrthBasisShiftMatrix() const; 
    private:
        std::vector<double> m_orth_coefs;
        void orthogonalize(uint max_quad_order = 5);
        void setup(); 
        template<typename ...Targs>
        OpMemoryRequirements memOP_internal(uint gdim, uint nquadpoints, uint fusion, OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const;
        template<typename ...Targs>
        BandDenseMatrixX<> applyOP_internal(uint gdim, AniMemoryX<> &mem, ArrayView<> &U, 
            BandDenseMatrixX<>(Ani::BaseFemSpace::* apl)(AniMemoryX<> &, ArrayView<> &, Targs...) const,
            OpMemoryRequirements(Ani::BaseFemSpace::* req)(uint , uint, Targs...) const, Targs... ts) const; 
        template<typename ...Targs>
        inline uint internal_take_max_val(uint(Ani::BaseFemSpace::* take_val)(Targs...) const, Targs... ts) const;  
    };

    struct ApplyOpFromSpace;
    struct FemSpace{
        using uchar = BaseFemSpace::uchar;
        using uint = BaseFemSpace::uint;
        using DofMap = BaseFemSpace::DofMap;
        using TetGeomSparsity = BaseFemSpace::TetGeomSparsity;
        using EvalFunc = BaseFemSpace::EvalFunc;
        using EvalFunctor = BaseFemSpace::EvalFunctor;
        using BaseTypes = BaseFemSpace::BaseTypes;

        std::shared_ptr<BaseFemSpace> m_invoker;

        template<typename FemSpaceT>
        explicit FemSpace(const FemSpaceT& f, typename std::enable_if<std::is_base_of<BaseFemSpace, FemSpaceT>::value>::type* = 0): m_invoker{new FemSpaceT(f)} {}
        template<typename FemSpaceT>
        explicit FemSpace(FemSpaceT&& f, typename std::enable_if<std::is_base_of<BaseFemSpace, FemSpaceT>::value>::type* = 0): m_invoker{new FemSpaceT(std::move(f))} {}
        FemSpace(const FemSpace &) = default;
        FemSpace(FemSpace &&) = default;
        FemSpace() = default;
        explicit FemSpace(std::shared_ptr<BaseFemSpace> fem_space): m_invoker(std::move(fem_space)) {}
        FemSpace& operator=(const FemSpace &f){ return m_invoker = f.m_invoker, *this; }
        FemSpace& operator=(FemSpace &&f){ return m_invoker = std::move(f.m_invoker), *this; }
        
        template<typename FemSpaceT = BaseFemSpace>
        FemSpaceT* target() { return static_cast<FemSpaceT *>(m_invoker.get()); }
        template<typename FemSpaceT = BaseFemSpace>
        const FemSpaceT* target() const { return static_cast<const FemSpaceT *>(m_invoker.get()); }
        
        /// @brief Get stored fem space
        std::shared_ptr<BaseFemSpace> base() const { return m_invoker; }
        /// @brief Get d.o.f. map of the space
        DofMap& dofMap(){ return m_invoker->m_order; }
        const DofMap& dofMap() const { return m_invoker->m_order; }

        /// @brief Get gather type of the space
        /// @see BaseFemSpace::BaseTypes
        BaseTypes gatherType() const { return m_invoker ? m_invoker->gatherType() : BaseTypes::NoType; }
        /// @brief Get name of the space
        std::string typeName() const { return m_invoker ? m_invoker->typeName() : ""; }
        bool operator==(const FemSpace& other) const { return m_invoker.get() == other.m_invoker.get() || (m_invoker && other.m_invoker && *m_invoker == *other.m_invoker); }
        
        /// @brief Check wheather the object store some actual fem space 
        bool isValid() const { return m_invoker != nullptr; }
        /// @brief Get dimesion of the fem space
        uint dim() const { return m_invoker->dim(); }
        /// @brief Get polynomial order of the fem space
        uint order() const { return m_invoker->order(); }
        /// @brief Evaluate numericaly or symbolically basis functions of the space
        /// @see BaseFemSpace::evalBasisFunctions
        void evalBasisFunctions(const Expr& lmb /*4x1*/, const Expr& grad_lmb/*4x3*/, Expr* phi) const { m_invoker->evalBasisFunctions(lmb, grad_lmb, phi); }
        /// @brief Set interpolation of the vector-function on idof_on_tet-th degree of freedom
        /// @param XYZ is tetrahedron
        /// @param f is vector-function to be interpolated, return [f_0, f_1, .., f_{r-1}], r = fusion
        /// @param udofs[fusion] are vectors to save result of interpolation
        /// @param idof_on_tet is index of d.o.f.
        /// @param user_data is some additional data for function f
        /// @param max_quad_order is maximal quadrature order will be used (if required)
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const { 
            m_invoker->interpolateOnDOF(XYZ, f, udofs, idof_on_tet, fusion, mem, user_data, max_quad_order); 
        }
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, int fusion, uint max_quad_order) const { 
            return m_invoker->interpolateOnDOF_mem_req(idof_on_tet, fusion, max_quad_order);
        }
        /// @brief Set interpolation of the function on idof_on_tet-th degree of freedom
        /// @param XYZ is tetrahedron
        /// @param f is function to be interpolated
        /// @param udofs is vector to save result of interpolation
        /// @param idof_on_tet is index of d.o.f.
        /// @param user_data is some additional data for function f
        /// @param max_quad_order is maximal quadrature order will be used (if required)
        void interpolateOnDOF(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, int idof_on_tet, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const { 
            m_invoker->interpolateOnDOF(XYZ, f, udofs, idof_on_tet, mem, user_data, max_quad_order); 
        }
        PlainMemoryX<> interpolateOnDOF_mem_req(int idof_on_tet, uint max_quad_order) const { 
            return m_invoker->interpolateOnDOF_mem_req(idof_on_tet, max_quad_order);
        }
        /// For operator op assemble matrix U: U[k + DimOp * (n + Npnts * ( i + NumDofOnTet*r ))] = (A phi^i)_k ( x_{nr} ) 
        /// @param[in] op is linear differential operator (including IDEN) 
        /// @param[in] mem is memory for all operations
        /// @param[in,out] U memory to store result
        /// @return view on assembled local matrix Dim
        BandDenseMatrixX<> applyOP(OperatorType op, AniMemoryX<> &mem, ArrayView<> &U) const { return m_invoker->applyOP(op, mem, U); }
        /// @brief Get memory requirements for specific operation
        /// @param op is linear differential operator
        /// @param nquadpoints is number points to be used to compute matrix (A phi^i)_k ( x_{nr} ) (dimension of n index)
        /// @param fusion is number tetra to be used to compute matrix (A phi^i)_k ( x_{nr} ) (dimension of r index)
        /// @return memory requirements for specific operation
        OpMemoryRequirements memOP(OperatorType op, uint nquadpoints, uint fusion = 1) const { return m_invoker->memOP(op, nquadpoints, fusion); }
        /// @brief Get dimension of result of applying operator on element of FEM space
        uint dimOP(OperatorType op) const { return m_invoker->dimOP(op); }
        /// @brief Get polynomial order of applying operator
        uint orderOP(OperatorType op) const { return m_invoker->orderOP(op); }
        /// @return operator applier 
        ApplyOpFromSpaceView getOP(OperatorType op) const { return m_invoker->getOP(op); }
        ApplyOpFromSpace getOP_own(OperatorType op) const;
        
        /// @brief Interpolate function f on d.o.f.'s belongs to region defined by sp
        /// @see BaseFemSpace::interpolateByDOFs
        void interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc&f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const {
            m_invoker->interpolateByDOFs(XYZ, f, udofs, sp, mem, user_data, max_quad_order);
        }
        PlainMemoryX<> interpolateByDOFs_mem_req(uint max_quad_order = 5) const{
            return m_invoker->interpolateByDOFs_mem_req(max_quad_order);
        }
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<>* udofs, int idof_on_tet, int fusion, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const {
            m_invoker->template interpolateOnDOF<>(XYZ, f, udofs, idof_on_tet, fusion, mem, user_data, max_quad_order);
        }
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateOnDOF(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, int idof_on_tet, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const {
            m_invoker->template interpolateOnDOF<>(XYZ, f, udofs, idof_on_tet, mem, user_data, max_quad_order);
        }
        template<typename EVAL_FUNCTOR, typename std::enable_if<!std::is_same<EVAL_FUNCTOR, EvalFunc>::value, bool>::type = true>
        inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const {
            m_invoker->template interpolateByDOFs<>(XYZ, f, udofs, sp, mem, user_data, max_quad_order);
        }
        template<typename EVAL_FUNCTOR>
        inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EVAL_FUNCTOR& f, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const {
            m_invoker->template interpolateByDOFs<>(XYZ, f, udofs, sp, sub_map, mem, user_data, max_quad_order);
        }

        /// @brief Set value val on d.o.f.'s belongs to region defined by sp
        inline void interpolateConstant(double val, ArrayView<> udofs, const TetGeomSparsity& sp) const { m_invoker->interpolateConstant(val, udofs, sp); }
        inline void interpolateConstant(double val, ArrayView<> udofs, const TetGeomSparsity& sp, const DofT::NestedDofMapView& sub_map) const {
            m_invoker->interpolateConstant(val, udofs, sp, sub_map);
        }

        /// Perform interpolation of func on d.o.f.'s by solving the equation
        /// \f[ 
        ///   \int_{\Omega_s} [\phi_i \phi_j] u_j dx = \int_{\Omega_s} f \phi_i dx
        /// \f]
        /// where \Omega_s is selected geometrical element
        /// @see BaseFemSpace::interpolateOnRegion
        void interpolateOnRegion(const Tetra<const double>& XYZ, const EvalFunctor& f, ArrayView<> udofs, uchar elem_type, int ielem, PlainMemoryX<> mem, void* user_data = nullptr, uint max_quad_order = 5) const{
            m_invoker->interpolateOnRegion(XYZ, f, udofs, elem_type, ielem, mem, user_data, max_quad_order);
        } 
        PlainMemoryX<> interpolateOnRegion_mem_req(uchar elem_type, uint max_quad_order) const {
            return m_invoker->interpolateOnRegion_mem_req(elem_type, max_quad_order);
        }
        /// @brief Get stored fem subspace
        /// @param ext_dims[ndims] is array of size ndims of subcomponent numbers
        /// @param ndims is size of ext_dims array
        /// @return a subspace 
        /// e.g. for space Complex : (Unite, Unite, (Vector[3]: Unite) )
        /// ext_dims = 2, 1 defines Complex -> 2-Vector[] -> 1-Unite[1]=Unite
        inline FemSpace subSpace(const int* ext_dims = nullptr, int ndims = 0) const {
            return (ndims > 0) ? FemSpace(m_invoker->subSpace(ext_dims, ndims)) : (ndims == 0 ? *this : FemSpace());
        }
        inline FemSpace subSpace(const std::initializer_list<int>& ext_dims) const { return subSpace(ext_dims.begin(), ext_dims.size()); }
        inline FemSpace subSpace(const int ncomp) const { return subSpace(&ncomp, 1); }
        inline FemSpace copy() const { return FemSpace(m_invoker->copy()); }
        /// @brief Create UnionSpace(a, b), a+b
        friend FemSpace operator+(const FemSpace& a, const FemSpace& b);
        /// @brief Create ComplexSpace(a, b), a x b
        friend FemSpace operator*(const FemSpace& a, const FemSpace& b);
        /// @brief Create VectorSpace(k, a), a^k
        friend FemSpace operator^(const FemSpace& a, int k);
    };

    FemSpace operator+(const FemSpace& a, const FemSpace& b);
    FemSpace operator*(const FemSpace& a, const FemSpace& b);
    FemSpace operator^(const FemSpace& a, int k);

    inline static FemSpace make_union_raw(const std::vector<FemSpace>& spaces);
    inline static FemSpace make_union_with_simplification(const std::vector<FemSpace>& spaces);
    inline static FemSpace make_complex_raw(const std::vector<FemSpace>& spaces);
    inline static FemSpace make_complex_with_simplification(const std::vector<FemSpace>& spaces);
    inline static FemSpace pow(const FemSpace& d, int k) { return d^k; } 

    /// Apply operator from space with space ownship 
    struct ApplyOpFromSpace: public ApplyOpBase{
        FemSpace m_space;
        OperatorType m_op = IDEN; 
        uint m_dim = 0;
        uint m_nfa = 0;
        uint m_order = std::numeric_limits<uint>::max();

        BandDenseMatrixX<> operator()(AniMemoryX<> &mem, ArrayView<> &U) const override { return m_space.applyOP(m_op, mem, U); }
        OpMemoryRequirements getMemoryRequirements(uint nquadpoints, uint fusion = 1) const override { return m_space.memOP(m_op, nquadpoints, fusion); }
        uint Nfa() const override { return m_nfa; }
        uint Dim() const override { return m_dim; }
        uint Order() const override { return m_order; }
        uint ActualType() const override { return 3; }
        bool operator==(const ApplyOpBase& otherOp) const override { return otherOp.ActualType() == ActualType() && m_space == static_cast<const ApplyOpFromSpace&>(otherOp).m_space && m_op == static_cast<const ApplyOpFromSpace&>(otherOp).m_op; }
        std::shared_ptr<ApplyOpBase> Copy() const { return std::make_shared<ApplyOpFromSpace>(*this); } 
        ApplyOpFromSpace(OperatorType op, const FemSpace& sp): m_space{sp}, m_op{op}, m_dim{sp.dimOP(op)}, m_nfa{sp.dofMap().NumDofOnTet()}, m_order{sp.orderOP(op)} {}
        ApplyOpFromSpace(const ApplyOpFromSpace&) = default;
        ApplyOpFromSpace(ApplyOpFromSpace&&) = default;
        ApplyOpFromSpace& operator=(const ApplyOpFromSpace&) = default;
        ApplyOpFromSpace& operator=(ApplyOpFromSpace&&) = default;
        ApplyOpFromSpace() {}
        ApplyOpFromSpace& Init(OperatorType op, const FemSpace& sp){ m_space = sp, m_op = op, m_dim = sp.dimOP(op), m_nfa = sp.dofMap().NumDofOnTet(), m_order = sp.orderOP(op); return *this; }
        bool isValid() const { return m_space.base().get() != nullptr; }
    };
};

#include "fem_space.inl"

#endif //CARNUM_ANI_FEMSPACE_H