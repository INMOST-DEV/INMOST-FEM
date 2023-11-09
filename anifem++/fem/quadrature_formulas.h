//
// Created by Liogky Alexey on 02.03.2022.
//

#ifndef CARNUM_FEM_QUADRATURE_FORMULAS_H
#define CARNUM_FEM_QUADRATURE_FORMULAS_H
#include <utility>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <array>

#ifdef WITH_EIGEN
#include "eigen3/Eigen/Eigen"
#endif

#ifndef QUADRATURE_FORMULAS_NUM_TYPE
typedef double qReal;
#else
typedef QUADRATURE_FORMULAS_NUM_TYPE qReal;
#endif

///Container for 1D quadrature formula parameters
struct SegmentQuadFormula{
    template<typename Real>
    struct PW{
        std::array<Real, 2> p;
        Real w;
    };
#ifdef WITH_EIGEN
    using WView = Eigen::Map<const Eigen::Matrix<qReal, 1, Eigen::Dynamic>, Eigen::Aligned128>;
    using PView = Eigen::Map<const Eigen::Matrix<qReal, 2, Eigen::Dynamic>, Eigen::Aligned128>;
#endif
    /// Number of quadrature points
    int num_points = 0;
    /// Number of points of different symmetry types: #s1, #s2
    /// first #s1 points have s1 symmetry and next #s2 pair of points have s2 symmetry
    /// if quadrature formula isn't symmetric then have negative values
    std::array<int, 2> sym = {-1, -1};
    /// Array of barycentric coordinates on segment (p0, p1), supposed p1 = 1 - p0
    ///  p consists data as
    ///  p0_0, p1_0 - barycoords for first quadrature point
    ///  p0_1, p1_1- barycoords for second quadrature point
    ///  ... and etc
    const qReal* p = nullptr;
    /// Array of quadrature weights
    ///  w consists data as
    ///  w0 - weight of first quadrature point
    ///  w1 - weight of  second quadrature point
    ///  ... and etc
    const qReal* w = nullptr;

    SegmentQuadFormula() = default;
    SegmentQuadFormula(int num_points, const qReal* p, const qReal* w, std::array<int, 2> symmetry = {-1, -1}): num_points{num_points}, sym{symmetry}, p{p}, w{w} {}
    int GetNumPoints() const{ return num_points; }
    const qReal* GetPointData() const { return p; }
    const qReal* GetWeightData() const { return w; }
    bool IsSymmetric() const { return sym[0] >= 0 && sym[1] >= 0; }
    std::array<int, 2> GetSymmetryPartition() const { return sym; }
    ///@return quadrature barycentric coords and weight of the point
    ///@param ipnt is number of quadrature point
    template<typename Real = qReal>
    PW<Real> GetPointWeight(int ipnt) const {
        assert(ipnt < num_points && "Wrong quadrature point");
        PW<Real> pw;
        pw.p[0] = p[2*ipnt+0];
        pw.p[1] = 1 - pw.p[0]; //to more numerical stability
        pw.w = w[ipnt];
        return pw;
    }

#ifdef WITH_EIGEN
    ///@return view on quadrature weights as vector [w1, w2, ..., wn]
    WView WeightView() const{
        return {GetWeightData(), GetNumPoints()};
    }
    ///@return view on quadrature points as matrix [P1, P2, ..., Pn], P = [p0, p1, p2]^T
    PView BaryView() const{
        return {GetPointData(), 2, GetNumPoints()};
    }
#endif
};
///Container for triangle quadrature formula parameters
struct TriangleQuadFormula{
    template<typename Real>
    struct PW{
        std::array<Real, 3> p;
        Real w;
    };
#ifdef WITH_EIGEN
    using WView = Eigen::Map<const Eigen::Matrix<qReal, 1, Eigen::Dynamic>, Eigen::Aligned128>;
    using PView = Eigen::Map<const Eigen::Matrix<qReal, 3, Eigen::Dynamic>, Eigen::Aligned128>;
#endif
    /// Number of quadrature points
    int num_points = 0;
    /// Number of points of different symmetry types: #s1, #s3, #s6
    /// full number of points = 1 * #s1 + 3 * #s3 + 6 * #s6
    /// if quadrature formula isn't symmetric then have negative values
    std::array<int, 3> sym = {-1, -1, -1};
    /// Array of barycentric coordinates on triangle (p0, p1, p2), supposed p2 = 1 - (p0 + p1)
    ///  p consists data as
    ///  p0_0, p1_0, p2_0 - barycoords for first quadrature point
    ///  p0_1, p1_1, p2_1 - barycoords for second quadrature point
    ///  ... and etc
    const qReal* p = nullptr;
    /// Array of quadrature weights
    ///  w consists data as
    ///  w0 - weight of first quadrature point
    ///  w1 - weight of  second quadrature point
    ///  ... and etc
    const qReal* w = nullptr;

    TriangleQuadFormula() = default;
    TriangleQuadFormula(int num_points, const qReal* p, const qReal* w, std::array<int, 3> symmetry = {-1, -1, -1}): num_points{num_points}, sym{symmetry}, p{p}, w{w} {}
    int GetNumPoints() const{ return num_points; }
    const qReal* GetPointData() const { return p; }
    const qReal* GetWeightData() const { return w; }
    bool IsSymmetric() const { return sym[0] >= 0 && sym[1] >= 0 && sym[2] >= 0; }
    std::array<int, 3> GetSymmetryPartition() const { return sym; }
    ///@return quadrature barycentric coords and weight of the point
    ///@param ipnt is number of quadrature point
    template<typename Real = qReal>
    PW<Real> GetPointWeight(int ipnt) const {
        assert(ipnt < num_points && "Wrong quadrature point");
        PW<Real> pw;
        pw.p[0] = p[3*ipnt+0], pw.p[1] = p[3*ipnt+1];
//        pw.p[2] = p[3*ipnt+2];
        pw.p[2] = 1 - pw.p[0] - pw.p[1]; //to more numerical stability
        pw.w = w[ipnt];
        return pw;
    }

#ifdef WITH_EIGEN
    ///@return view on quadrature weights as vector [w1, w2, ..., wn]
    WView WeightView() const{
        return {GetWeightData(), GetNumPoints()};
    }
    ///@return view on quadrature points as matrix [P1, P2, ..., Pn], P = [p0, p1, p2]^T
    PView BaryView() const{
        return {GetPointData(), 3, GetNumPoints()};
    }
#endif
};

///Container for tetrahedron quadrature formula parameters
struct TetraQuadFormula{
    template<typename Real>
    struct PW{
        std::array<Real, 4> p;
        Real w;
    };
#ifdef WITH_EIGEN
    using WView = Eigen::Map<const Eigen::Matrix<qReal, 1, Eigen::Dynamic>, Eigen::Aligned128>;
    using PView = Eigen::Map<const Eigen::Matrix<qReal, 4, Eigen::Dynamic>, Eigen::Aligned128>;
#endif
    /// Number of quadrature points
    int num_points = 0;
    /// Number of points of different symmetry types: #s1, #s4, #s6, #s12, #s24
    /// full number of points = 1 * #s1 + 4 * #s4 + 6 * #s6 + 12 * #s12 + 24 * #s24
    /// if quadrature formula isn't symmetric then have negative values
    std::array<int, 5> sym = {-1, -1, -1, -1, -1};
    /// Array of barycentric coordinates on tetrahedron (p0, p1, p2, p3) of points and weights (w), supposed p3 = 1 - (p0 + p1 + p2)
    ///  p consists data as
    ///  p0_0, p1_0, p2_0, p3_0 - barycoords for first quadrature point
    ///  p0_1, p1_1, p2_1, p3_1 - barycoords for second quadrature point
    ///  ... and etc
    const qReal* p = nullptr;
    /// Array of quadrature weights
    ///  w consists data as
    ///  w0 - weight of first quadrature point
    ///  w1 - weight of  second quadrature point
    ///  ... and etc
    const qReal* w = nullptr;

    TetraQuadFormula() = default;
    TetraQuadFormula(int num_points, const qReal* p, const qReal* w, std::array<int, 5> sym = {-1, -1, -1, -1, -1}): num_points{num_points}, sym{sym}, p{p}, w{w} {}
    int GetNumPoints() const{ return num_points; }
    const qReal* GetPointData() const { return p; }
    const qReal* GetWeightData() const { return w; }
    bool IsSymmetric() const { return sym[0] >= 0 && sym[1] >= 0 && sym[2] >= 0 && sym[3] >= 0 && sym[4] >= 0; }
    std::array<int, 5> GetSymmetryPartition() const { return sym; }
    ///@return quadrature barycentric coords and weight of the point
    ///@param ipnt is number of quadrature point
    template<typename Real = qReal>
    PW<Real> GetPointWeight(int ipnt) const {
        assert(ipnt < num_points && "Wrong quadrature point");
        PW<Real> pw;
        pw.p[0] = p[4*ipnt+0], pw.p[1] = p[4*ipnt+1], pw.p[2] = p[4*ipnt+2];
//        pw.p[3] = p[4*ipnt+3];
        pw.p[3] = 1 - pw.p[0] - pw.p[1] - pw.p[2]; //to more numerical stability
        pw.w = w[ipnt];
        return pw;
    }

#ifdef WITH_EIGEN
    ///@return view on quadrature weights as vector [w1, w2, ..., wn]
    WView WeightView() const{
        return {GetWeightData(), GetNumPoints()};
    }
    ///@return view on quadrature points as matrix [P1, P2, ..., Pn], P = [p0, p1, p2, p3]^T
    PView BaryView() const{
        return {GetPointData(), 4, GetNumPoints()};
    }
#endif
};

/// @return tetrahedron quadrature formula of chosen approximation order,
/// it's guaranteed the formula has symmetric template and all points are located
/// in internal part of tetrahedron
/// @note this function has zero runtime overhead (all return arrays are static)
/// @param order is minimal approximation order of quadrature ( 1 <= order <= 20 )
TetraQuadFormula tetrahedron_quadrature_formulas(int order);

/// @return triangle quadrature formula of chosen approximation order
/// it's guaranteed the formula has symmetric template and all points are located
/// in internal part of triangle
/// @note this function has zero runtime overhead (all return arrays are static)
/// @param order is minimal approximation order of quadrature ( 1 <= order <= 7 )
TriangleQuadFormula triangle_quadrature_formulas(int order);

/// @return segment quadrature formula of chosen approximation order
/// it's guaranteed the formula has symmetric template and all points are located
/// in internal part of segment
/// @note this function has zero runtime overhead (all return arrays are static)
/// @param order is minimal approximation order of quadrature ( 1 <= order <= 20 )
SegmentQuadFormula segment_quadrature_formulas(int order);

#endif //CARNUM_FEM_QUADRATURE_FORMULAS_H
