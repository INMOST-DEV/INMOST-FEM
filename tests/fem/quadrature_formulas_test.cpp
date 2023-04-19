//
// Created by Liogky Alexey on 02.03.2022.
//
#include <gtest/gtest.h>
#include "anifem++/fem/quadrature_formulas.h"

TEST(QuadratureFormulas, TetraFormulas){
    ///Canonical tetrahedron with points
    /// P0 = {0, 0, 0}, P1 = {1, 0, 0}, P2 = {0, 1, 0}, P3 = {0, 0, 1}
    for (int ord = 1; ord <= 20; ++ord) {
        auto f = [ord](double x, double y, double z) {
            long double a = 1, c = 10, e = 1000;
            int b = 1, d = ord > 1 ? ord - 1 : 1, q = ord;
            return a * std::pow(x, b) + c * std::pow(y, d) + e * std::pow(z, q);
        };
        auto exact_intf = [ord]() {
            long double a = 1, c = 10, e = 1000;
            int b = 1, d = ord > 1 ? ord - 1 : 1, q = ord;
            return a / ((b + 1) * (b + 2) * (b + 3)) + c / ((d + 1) * (d + 2) * (d + 3)) +
                   e / ((q + 1) * (q + 2) * (q + 3));
        };
        auto q = tetrahedron_quadrature_formulas(ord);
        long double eval_intf = 0;
        for (int i = 0; i < q.GetNumPoints(); ++i) {
            auto p_w = q.GetPointWeight<qReal>(i);
            double x = 0 * p_w.p[0] + 1 * p_w.p[1] + 0 * p_w.p[2] + 0 * p_w.p[3];
            double y = 0 * p_w.p[0] + 0 * p_w.p[1] + 1 * p_w.p[2] + 0 * p_w.p[3];
            double z = 0 * p_w.p[0] + 0 * p_w.p[1] + 0 * p_w.p[2] + 1 * p_w.p[3];
            eval_intf += p_w.w * f(x, y, z); //mean value
        }
        long double volume = static_cast<long double>(1) / 6;
        eval_intf *= volume; // integral
        long double exact_intf_ = exact_intf();
        long double err = std::abs(exact_intf_ - eval_intf);
        long double rel = err / exact_intf_;
        EXPECT_FALSE(err > 1e-10 && rel > 1e-15) << "error in " << ord << "-th order quadrature formula: "
                                                 << std::scientific << std::setprecision(20) << "int_exact = "
                                                 << exact_intf_ << " int_eval = " << eval_intf
                                                 << " |int_exact - int_eval| = " << err <<
                                                 " relative_error = " << rel << std::endl;
    }

}

TEST(QuadratureFormulas, TriangleFormulas){
    ///Canonical triangle with points
    /// P0 = {0, 0}, P1 = {1, 0}, P2 = {0, 1}
    for (int ord = 1; ord <= 20; ++ord) {
        auto f = [ord](double x, double y) {
            long double a = 1, c = 10, e = 1000;
            int b = 1, d = ord > 1 ? ord - 1 : 1, q = ord;
            return a * std::pow(x, b) + c * std::pow(x, d) + e * std::pow(y, q);
        };
        auto exact_intf = [ord]() {
            long double a = 1, c = 10, e = 1000;
            int b = 1, d = ord > 1 ? ord - 1 : 1, q = ord;
            return a / ((b + 1) * (b + 2)) + c / ((d + 1) * (d + 2)) +
                   e / ((q + 1) * (q + 2));
        };
        auto q = triangle_quadrature_formulas(ord);
        long double eval_intf = 0;
        for (int i = 0; i < q.GetNumPoints(); ++i) {
            auto p_w = q.GetPointWeight<qReal>(i);
            double x = 0 * p_w.p[0] + 1 * p_w.p[1] + 0 * p_w.p[2];
            double y = 0 * p_w.p[0] + 0 * p_w.p[1] + 1 * p_w.p[2];
            eval_intf += p_w.w * f(x, y); //mean value
        }
        long double area = static_cast<long double>(1) / 2;
        eval_intf *= area; // integral
        long double exact_intf_ = exact_intf();
        long double err = std::abs(exact_intf_ - eval_intf);
        long double rel = err / exact_intf_;
        EXPECT_FALSE(err > 1e-10 && rel > 1e-15) << "error in " << ord << "-th order quadrature formula: "
                                                 << std::scientific << std::setprecision(20) << "int_exact = "
                                                 << exact_intf_ << " int_eval = " << eval_intf
                                                 << " |int_exact - int_eval| = " << err <<
                                                 " relative_error = " << rel << std::endl;
    }

}

TEST(QuadratureFormulas, SegmentFormulas){
    ///Canonical segment [0, 1]
    for (int ord = 1; ord <= 21; ++ord) {
        auto f = [ord](double x) {
            long double a = 1, c = 10, e = 1000;
            int b = 1, d = ord > 1 ? ord - 1 : 1, q = ord;
            return a * std::pow(x, b) + c * std::pow(x, d) + e * std::pow(x, q);
        };
        auto exact_intf = [ord]() {
            long double a = 1, c = 10, e = 1000;
            int b = 1, d = ord > 1 ? ord - 1 : 1, q = ord;
            return a / (b + 1) + c / (d + 1) + e / (q + 1);
        };
        auto q = segment_quadrature_formulas(ord);
        long double eval_intf = 0;
        for (int i = 0; i < q.GetNumPoints(); ++i) {
            auto p_w = q.GetPointWeight<qReal>(i);
            double x = 0 * p_w.p[0] + 1 * p_w.p[1];
            eval_intf += p_w.w * f(x); //mean value
        }
        long double measure = 1;
        eval_intf *= measure; // integral
        long double exact_intf_ = exact_intf();
        long double err = std::abs(exact_intf_ - eval_intf);
        long double rel = err / exact_intf_;
        EXPECT_FALSE(err > 1e-10 && rel > 1e-15) << "error in " << ord << "-th order quadrature formula: "
                                                 << std::scientific << std::setprecision(20) << "int_exact = "
                                                 << exact_intf_ << " int_eval = " << eval_intf
                                                 << " |int_exact - int_eval| = " << err <<
                                                 " relative_error = " << rel << std::endl;
    }
}

