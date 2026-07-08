//
// Created by Liogky Alexey on 08.07.2026.
//

#include <gtest/gtest.h>
#include "anifem++/fem/geometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr double kTol = 1e-10;

void expect_vec_near(const double* a, const double* b, int n, double tol = kTol) {
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(a[i], b[i], tol) << "component " << i;
}

void expect_mat_near_colmajor(const double* a, const double* b, int n, double tol = kTol) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(a[i + j * n], b[i + j * n], tol) << "at (" << i << "," << j << ")";
}

std::vector<double> mat_mul_colmajor(const double* a, const double* b, int n) {
    std::vector<double> c(n * n, 0.0);
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                c[i + j * n] += a[i + k * n] * b[k + j * n];
    return c;
}

std::vector<double> identity_colmajor(int n) {
    std::vector<double> id(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        id[i + i * n] = 1.0;
    return id;
}

double vec_dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

double vec_norm(const double* a, int n) {
    return std::sqrt(vec_dot(a, a, n));
}

std::vector<double> mat_vec_colmajor(const double* a, const double* x, int n, int m) {
    std::vector<double> b(n * m, 0.0);
    for (int col = 0; col < m; ++col)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                b[i + n * col] += a[i + j * n] * x[j + n * col];
    return b;
}

} // namespace

TEST(Geometry, Inverse3x3) {
    using namespace Ani;

    const auto expect_inverse = [](const double* m, double expected_det, const char* label) {
        double inv[9];
        const double det = inverse3x3(m, inv);
        SCOPED_TRACE(label);
        EXPECT_NEAR(det, expected_det, kTol);
        const auto prod = mat_mul_colmajor(m, inv, 3);
        expect_mat_near_colmajor(prod.data(), identity_colmajor(3).data(), 3);
    };

    const double identity_m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    expect_inverse(identity_m, 1.0, "identity");

    const double diagonal_m[9] = {2, 0, 0, 0, 3, 0, 0, 0, 4};
    expect_inverse(diagonal_m, 24.0, "diagonal");

    // [[1, 2, 0], [0, 1, 3], [4, 0, 1]] in column-major layout
    const double general_m[9] = {1, 0, 4, 2, 1, 0, 0, 3, 1};
    expect_inverse(general_m, 25.0, "general");

    // 90-degree rotation around z-axis
    const double rotation_m[9] = {0, -1, 0, 1, 0, 0, 0, 0, 1};
    expect_inverse(rotation_m, 1.0, "rotation");
}

TEST(Geometry, CrossProduct) {
    using namespace Ani;

    const auto expect_cross_pair = [](const double* a, const double* b, const double* expected, const char* label) {
        double axb[3];
        SCOPED_TRACE(label);
        cross(a, b, axb);
        expect_vec_near(axb, expected, 3);

        double bxa[3];
        cross(b, a, bxa);
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(axb[i], -bxa[i], kTol);
    };

    const double ex[3] = {1, 0, 0};
    const double ey[3] = {0, 1, 0};
    const double ez[3] = {0, 0, 1};
    expect_cross_pair(ex, ey, ez, "basis ex x ey");

    const double a[3] = {2, -1, 3};
    const double b[3] = {-4, 5, 2};
    const double expected_ab[3] = {-17, -16, 6};
    expect_cross_pair(a, b, expected_ab, "general vectors");

    const double p0[3] = {1, 0, 0};
    const double p1[3] = {0, 1, 0};
    const double p2[3] = {0, 0, 0};
    double from_points[3];
    cross(p0, p1, p2, from_points);
    expect_vec_near(from_points, ez, 3);

    double from_vectors[3];
    const double pa[3] = {p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]};
    const double pb[3] = {p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
    cross(pa, pb, from_vectors);
    expect_vec_near(from_points, from_vectors, 3);

    double parallel[3];
    cross(ex, ex, parallel);
    expect_vec_near(parallel, std::array<double, 3>{0, 0, 0}.data(), 3);
}

TEST(Geometry, TriArea) {
    using namespace Ani;
    const double p0[3] = {0, 0, 0};
    const double p1[3] = {1, 0, 0};
    const double p2[3] = {0, 1, 0};
    EXPECT_NEAR(tri_area(p0, p1, p2), 0.5, kTol);

    const double q0[3] = {1, 1, 1};
    const double q1[3] = {2, 1, 1};
    const double q2[3] = {1, 2, 1};
    EXPECT_NEAR(tri_area(q0, q1, q2), 0.5, kTol);

    const double r0[3] = {0, 0, 0};
    const double r1[3] = {3, 0, 0};
    const double r2[3] = {1.5, 2, 1};
    EXPECT_NEAR(tri_area(r0, r1, r2), 0.5 * std::sqrt(45.0), kTol);

    const double s0[3] = {0, 0, 0};
    const double s1[3] = {2, 0, 0};
    const double s2[3] = {4, 0, 0};
    EXPECT_NEAR(tri_area(s0, s1, s2), 0.0, kTol);
}

TEST(Geometry, FaceNormal) {
    using namespace Ani;

    const auto expect_outward_unit_normal = [](const double* p0, const double* p1, const double* p2, const double* p3,
                                               double expected_area, const char* label) {
        double normal[3];
        SCOPED_TRACE(label);
        const double area = face_normal(p0, p1, p2, p3, normal);
        EXPECT_NEAR(area, expected_area, kTol);
        EXPECT_NEAR(vec_norm(normal, 3), 1.0, kTol);

        double to_opposite[3] = {p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
        EXPECT_LT(vec_dot(normal, to_opposite, 3), 0.0);

        const double tri_a = tri_area(p0, p1, p2);
        EXPECT_NEAR(tri_a, area, kTol);
    };

    const double c0[3] = {0, 0, 0};
    const double c1[3] = {1, 0, 0};
    const double c2[3] = {0, 1, 0};
    const double c3[3] = {0, 0, 1};
    double n_canonical[3];
    face_normal(c0, c1, c2, c3, n_canonical);
    expect_vec_near(n_canonical, std::array<double, 3>{0, 0, -1}.data(), 3);
    expect_outward_unit_normal(c0, c1, c2, c3, 0.5, "canonical tetrahedron");

    const double t0[3] = {1, 1, 1};
    const double t1[3] = {2, 1, 1};
    const double t2[3] = {1, 2, 1};
    const double t3[3] = {1, 1, 2};
    expect_outward_unit_normal(t0, t1, t2, t3, 0.5, "translated canonical tetrahedron");

    const double g0[3] = {0, 0, 0};
    const double g1[3] = {2, 0, 0};
    const double g2[3] = {1, 2, 0};
    const double g3[3] = {1, 1, 2};
    expect_outward_unit_normal(g0, g1, g2, g3, 2.0, "general tetrahedron bottom face");

    // opposite vertex inside: normal should flip sign relative to raw cross product
    const double h0[3] = {0, 0, 0};
    const double h1[3] = {1, 0, 0};
    const double h2[3] = {0, 1, 0};
    const double h3[3] = {0, 0, -1};
    double n_below[3];
    face_normal(h0, h1, h2, h3, n_below);
    expect_vec_near(n_below, std::array<double, 3>{0, 0, 1}.data(), 3);
}

TEST(Geometry, TetraAndMakeTetras) {
    using namespace Ani;

    struct TetraVerts {
        std::array<double, 3> p0;
        std::array<double, 3> p1;
        std::array<double, 3> p2;
        std::array<double, 3> p3;
    };

    const auto expect_tetra_geometry = [](const TetraVerts& v, const double* expected_centroid,
                                          double expected_diameter, const char* label) {
        Tetra<const double> tet(v.p0.data(), v.p1.data(), v.p2.data(), v.p3.data());
        SCOPED_TRACE(label);

        const auto c = tet.centroid();
        expect_vec_near(c.data(), expected_centroid, 3);
        EXPECT_NEAR(tet.diameter(), expected_diameter, kTol);

        const double* pts[4] = {v.p0.data(), v.p1.data(), v.p2.data(), v.p3.data()};
        for (unsigned char iopp = 0; iopp < 4; ++iopp) {
            const auto n = tet.normal(iopp);
            EXPECT_NEAR(vec_norm(n.data(), 3), 1.0, kTol) << "face opposite node " << static_cast<int>(iopp);

            const double* p0 = pts[(iopp + 1) % 4];
            const double* p3 = pts[iopp];
            double to_opposite[3] = {p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]};
            EXPECT_LT(vec_dot(n.data(), to_opposite, 3), 0.0) << "face opposite node " << static_cast<int>(iopp);
        }
    };

    const TetraVerts canonical{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    expect_tetra_geometry(canonical, std::array<double, 3>{0.25, 0.25, 0.25}.data(), std::sqrt(2.0),
                          "canonical tetrahedron");

    const TetraVerts translated{{1, 1, 1}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
    expect_tetra_geometry(translated, std::array<double, 3>{1.25, 1.25, 1.25}.data(), std::sqrt(2.0),
                          "translated canonical tetrahedron");

    const TetraVerts general{{0, 0, 0}, {2, 0, 0}, {1, 2, 0}, {1, 1, 2}};
    expect_tetra_geometry(general, std::array<double, 3>{1.0, 0.75, 0.5}.data(), std::sqrt(6.0),
                          "general tetrahedron");

    auto single = make_tetras(canonical.p0.data(), canonical.p1.data(), canonical.p2.data(), canonical.p3.data(), 1);
    EXPECT_EQ(single.fusion, 1);
    EXPECT_EQ(single.XY0.nRow, 3u);
    EXPECT_EQ(single.XY0.nCol, 1u);
    EXPECT_NEAR(single.XY1[0], 1.0, kTol);

    double xy0[6] = {0, 1, 0, 0, 0, 0};
    double xy1[6] = {1, 2, 0, 1, 0, 0};
    double xy2[6] = {0, 1, 0, 0, 1, 0};
    double xy3[6] = {0, 0, 1, 0, 0, 1};
    auto batch = make_tetras(xy0, xy1, xy2, xy3, 2);
    EXPECT_EQ(batch.fusion, 2);
    EXPECT_NEAR(batch.XY0[3], 0.0, kTol);
    EXPECT_NEAR(batch.XY1[3], 1.0, kTol);
    EXPECT_NEAR(batch.XY3[2], 1.0, kTol);
    EXPECT_NEAR(batch.XY3[5], 1.0, kTol);

    double owned_p0[3], owned_p1[3], owned_p2[3], owned_p3[3];
    std::copy(canonical.p0.begin(), canonical.p0.end(), owned_p0);
    std::copy(canonical.p1.begin(), canonical.p1.end(), owned_p1);
    std::copy(canonical.p2.begin(), canonical.p2.end(), owned_p2);
    std::copy(canonical.p3.begin(), canonical.p3.end(), owned_p3);
    Tetra<> owned_tet(owned_p0, owned_p1, owned_p2, owned_p3);
    const auto owned_centroid = owned_tet.centroid();
    expect_vec_near(owned_centroid.data(), std::array<double, 3>{0.25, 0.25, 0.25}.data(), 3);
}

TEST(Geometry, CholeskySolveAndInverse) {
    using namespace Ani;

    const auto expect_system = [](const double* a, const double* expected_x, int n, int m, const char* label) {
        const auto b = mat_vec_colmajor(a, expected_x, n, m);
        std::vector<double> x(n * m);
        std::vector<double> mem(n * (n + 1) / 2);
        SCOPED_TRACE(label);
        cholesky_solve(a, b.data(), n, m, x.data(), mem.data());
        expect_vec_near(x.data(), expected_x, n * m);

        std::vector<double> inv(n * n);
        cholesky_inverse(a, inv.data(), n, mem.data());
        const auto prod = mat_mul_colmajor(a, inv.data(), n);
        expect_mat_near_colmajor(prod.data(), identity_colmajor(n).data(), n);
    };

    const double a2[4] = {2, 1, 1, 2};
    const double x2[2] = {1, 1};
    expect_system(a2, x2, 2, 1, "2x2 SPD");

    const double a3[9] = {4, 1, 0, 1, 3, 1, 0, 1, 2};
    const double x3[3] = {1, 2, 1};
    expect_system(a3, x3, 3, 1, "3x3 SPD");

    const double x3_multi[6] = {1, 2, 1, 1, 1, 1};
    expect_system(a3, x3_multi, 3, 2, "3x3 SPD multiple RHS");
}

TEST(Geometry, FullPivLU) {
    using namespace Ani;

    const auto expect_system = [](const double* a, const double* expected_x, int n, int m, const char* label) {
        const auto b = mat_vec_colmajor(a, expected_x, n, m);
        std::vector<double> x(n * m);
        std::vector<double> mem_solve(n * (n + m));
        std::vector<double> mem_inv(2 * n * n);
        std::vector<int> imem(2 * n);
        SCOPED_TRACE(label);

        fullPivLU_solve(a, b.data(), n, m, x.data(), mem_solve.data(), imem.data());
        expect_vec_near(x.data(), expected_x, n * m);

        std::vector<double> lu(n * n);
        std::memcpy(lu.data(), a, n * n * sizeof(double));
        std::vector<int> p(n), q(n);
        for (int i = 0; i < n; ++i)
            p[i] = q[i] = i;
        fullPivLU(lu.data(), n, p.data(), q.data());

        std::vector<double> b_work(b);
        std::vector<double> x_lu(n * m);
        LU_solve(lu.data(), n, p.data(), q.data(), b_work.data(), x_lu.data(), m);
        expect_vec_near(x_lu.data(), expected_x, n * m);

        std::vector<double> inv(n * n);
        fullPivLU_inverse(a, inv.data(), n, mem_inv.data(), imem.data());
        const auto prod = mat_mul_colmajor(a, inv.data(), n);
        expect_mat_near_colmajor(prod.data(), identity_colmajor(n).data(), n);
    };

    const double a2[4] = {1, 3, 2, 4};
    const double x2[2] = {1, 2};
    expect_system(a2, x2, 2, 1, "2x2 general");

    const double a3[9] = {2, 1, 0, 1, 2, 1, 0, 1, 2};
    const double x3[3] = {1, 0, 1};
    expect_system(a3, x3, 3, 1, "3x3 general");

    const double a_pivot[4] = {1e-12, 1, 1, 1};
    const double x_pivot[2] = {0, 2};
    expect_system(a_pivot, x_pivot, 2, 1, "2x2 needs pivoting");

    const double x3_multi[6] = {1, 0, 1, 2, 0, 1};
    expect_system(a3, x3_multi, 3, 2, "3x3 multiple RHS");
}
