//
// Created by Liogky Alexey on 6.02.2023.
//

#ifndef CARNUM_FEM_ANIINTERFACE_OPERATIONS_INTERPOLATE_H
#define CARNUM_FEM_ANIINTERFACE_OPERATIONS_INTERPOLATE_H
#include "core.h"
#include "int_pnt.h"
#include <algorithm>
#include <stdexcept>
#include <array>

namespace Ani{
    /// Interpolate function f on degrees of freedom of fem-variable of FEMTYPE associated to geometrical parts of tetra defined by sp. 
    /// Save result into vector of d.o.f's udofs in corresponding places. Interpolation is performed sequential by d.o.f.s
    /// @tparam FEMTYPE is type of FEM space of variable
    /// @tparam EvalFunc is type of functor with signature: 
    /// \code {C++}
    /// int(const std::array<double, 3>& X, double* res, uint dim, void* user_data)
    /// \endcode
    /// where X is point in 3D space, res is memory to save result of func evaluation, dim is expected dimension of func result, user_data is additional user-supplied data
    /// @param XYZ is coords of tetrahedron
    /// @param f is the functor
    /// @param udofs is vector of d.o.f's
    /// @param sp is is selection of parts of the tetrahedron
    /// @param user_data user-supplied data to be used by functor
    /// @param max_quad_order is maximal quadrature order to be used
    template<typename FEMTYPE, typename EvalFunc>
    static inline void interpolateByDOFs(const Tetra<const double>& XYZ, const EvalFunc& f, ArrayView<> udofs, const DofT::TetGeomSparsity& sp, void* user_data = nullptr, uint max_quad_order = 5);
    
    /// @brief Interpolate constants on selection of the tetrahedron
    /// @tparam FEMTYPE is type of FEM space of variable
    /// @param val is constant to be interpolated
    /// @param udofs is vector of d.o.f's
    /// @param sp is is selection of parts of the tetrahedron
    template<typename FEMTYPE>
    static inline void interpolateConstant(double val, ArrayView<> udofs, const DofT::TetGeomSparsity& sp);
    
    /// @brief Perform interpolation of func on d.o.f.'s by solving the equation
    /// \f[ 
    ///   \int_{\Omega_s} [\phi_i \phi_j] u_j dx = \int_{\Omega_s} f \phi_i dx
    /// \f]
    /// where \Omega_s is selected geometrical element
    /// @param elem_type is type of element, see DofT namespace
    /// @param ielem is number of element of specified type
    /// @see interpolateByDOFs for description of the others parameters 
    template<typename FEMTYPE, typename EvalFunc, typename ScalarType = double, typename IndexType = int>
    static inline void interpolateByRegion(const Tetra<const ScalarType>& XYZ, const EvalFunc& func, ArrayView<ScalarType> udofs, unsigned char elem_type, uint ielem, 
                    PlainMemory<ScalarType, IndexType> plainMemory, void* user_data = nullptr, uint max_quad_order = 5);
    template<typename FEMTYPE, typename ScalarType = double, typename IndexType = int>
    PlainMemory <ScalarType, IndexType> interpolateByRegion_memory_requirements(unsigned char elem_type, uint max_quad_order = 5);            
    template<typename FEMTYPE, typename EvalFunc, typename ScalarType = double, typename IndexType = int, uint STATIC_MEMORY_SIZE = 256*1024>
    static inline void interpolateByRegion(const Tetra<const ScalarType>& XYZ, const EvalFunc& func, ArrayView<ScalarType> udofs, unsigned char elem_type, uint ielem, void* user_data = nullptr, uint max_quad_order = 5);
};

#include "interpolate.inl"

#endif //CARNUM_FEM_ANIINTERFACE_OPERATIONS_INTERPOLATE_H