//
// Created by Liogky Alexey on 15.03.2023.
//

#ifndef CARNUM_ANI_NUMERICEXPR_H
#define CARNUM_ANI_NUMERICEXPR_H

#include "expr_interface.h"
#include <vector>
#include <cstddef>
#include <memory>
#include <functional>
#include <ostream>

#include <stdexcept>
#include <utility>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace Ani{
    template<typename REAL = double>
    struct NumericExpr: public BaseExpr{
        std::vector<REAL> m_v = 0;
        std::size_t m_nrows = 1, m_ncols = 1;

        static Expr CreateExpr(REAL a = 0, std::size_t sz1 = 1, std::size_t sz2 = 1) { return Expr(std::make_shared<NumericExpr<REAL>>(a, sz1, sz2)); }
        NumericExpr<REAL>() = default; 
        explicit NumericExpr<REAL>(REAL v, std::size_t sz1 = 1, std::size_t sz2 = 1): m_v(sz1*sz2, v), m_nrows{sz1}, m_ncols{sz2} {}
        explicit NumericExpr<REAL>(REAL* vp, std::size_t sz1 = 1, std::size_t sz2 = 1): m_nrows{sz1}, m_ncols{sz2} {
            m_v.resize(sz1*sz2);
            std::copy(vp, vp + sz1*sz2, m_v.data());
        }
        REAL& operator()(std::size_t i, std::size_t j) { return m_v[i + m_nrows*j]; }
        REAL operator()(std::size_t i, std::size_t j) const { return m_v[i + m_nrows*j]; }

        std::shared_ptr<BaseExpr> makeExpr(std::size_t sz1 = 1, std::size_t sz2 = 1) const override { return std::make_shared<NumericExpr<REAL>>(0, sz1, sz2); }
        std::shared_ptr<BaseExpr> makeCopy() const override { return std::make_shared<NumericExpr<REAL>>(*this); }
        std::size_t size1() const override { return m_nrows; }
        std::size_t size2() const override { return m_ncols; }
        NumericExpr<REAL>& reshape(std::size_t sz1, std::size_t sz2);
        NumericExpr<REAL>& set_val(double val) override{ 
            for (auto& i: m_v) i = val;
            return *this;
        }
        NumericExpr<REAL>& set_val(double* val) override { std::copy(val, val + m_v.size(), m_v.data()); return *this; }
        NumericExpr<REAL>& set_val(const BaseExpr& val) override { return *this = *static_cast<const NumericExpr<REAL>*>(&val); }
        NumericExpr<REAL>& set_val(const BaseExpr& val, std::size_t i, std::size_t j) override;
        NumericExpr<REAL>& set_sub_matrix(const BaseExpr& val, const std::size_t* i, std::size_t i_arr_sz, const std::size_t* j, long j_arr_sz) override;
        NumericExpr<REAL>& set_sub_matrix(const BaseExpr& val, std::size_t i_start, long i_inc, std::size_t j_start, long j_inc) override ;
        NumericExpr<REAL>& set_horzcat(const BaseExpr** vals_begin, const BaseExpr** vals_end) override;
        NumericExpr<REAL>& set_vertcat(const BaseExpr** vals_begin, const BaseExpr** vals_end) override;
        NumericExpr<REAL>& set_row(const BaseExpr& val, std::size_t i) override;
        NumericExpr<REAL>& set_col(const BaseExpr& val, std::size_t j) override;
        NumericExpr<REAL>& set_scal_sum(const BaseExpr** vals_begin, const BaseExpr** vals_end, const double* coefs) override;
        NumericExpr<REAL>& set_sum(const BaseExpr** vals_begin, const BaseExpr** vals_end) override;
        NumericExpr<REAL>& set_mul(const BaseExpr** vals_begin, const BaseExpr** vals_end) override;
        NumericExpr<REAL>& set_min(const BaseExpr** vals_begin, const BaseExpr** vals_end) override;
        NumericExpr<REAL>& set_max(const BaseExpr** vals_begin, const BaseExpr** vals_end) override;
        NumericExpr<REAL>& set_sum(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, std::plus<REAL>()); }
        NumericExpr<REAL>& set_mul(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, std::multiplies<REAL>()); }
        NumericExpr<REAL>& set_sub(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, std::minus<REAL>()); }
        NumericExpr<REAL>& set_div(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, std::divides<REAL>()); }
        NumericExpr<REAL>& set_swap_sign(const BaseExpr& a) override { return set_op(a, std::negate<REAL>()); }
        NumericExpr<REAL>& set_abs(const BaseExpr& a) override { return set_op(a, [](REAL a){ return abs(a); }); }
        NumericExpr<REAL>& set_exp(const BaseExpr& a) override { return set_op(a, [](REAL a){ return exp(a); }); }
        NumericExpr<REAL>& set_expm1(const BaseExpr& a) override { return set_op(a, [](REAL a){ return expm1(a); }); }
        NumericExpr<REAL>& set_log(const BaseExpr& a) override { return set_op(a, [](REAL a){ return log(a); }); }
        NumericExpr<REAL>& set_log1p(const BaseExpr& a) override { return set_op(a, [](REAL a){ return log1p(a); }); }
        NumericExpr<REAL>& set_sqrt(const BaseExpr& a) override { return set_op(a, [](REAL a){ return sqrt(a); }); }
        NumericExpr<REAL>& set_cbrt(const BaseExpr& a) override { return set_op(a, [](REAL a){ return cbrt(a); }); }
        NumericExpr<REAL>& set_sin(const BaseExpr& a) override { return set_op(a, [](REAL a){ return sin(a); }); }
        NumericExpr<REAL>& set_cos(const BaseExpr& a) override { return set_op(a, [](REAL a){ return cos(a); }); }
        NumericExpr<REAL>& set_tan(const BaseExpr& a) override { return set_op(a, [](REAL a){ return tan(a); }); }
        NumericExpr<REAL>& set_asin(const BaseExpr& a) override { return set_op(a, [](REAL a){ return asin(a); }); }
        NumericExpr<REAL>& set_acos(const BaseExpr& a) override { return set_op(a, [](REAL a){ return acos(a); }); }
        NumericExpr<REAL>& set_atan(const BaseExpr& a) override { return set_op(a, [](REAL a){ return atan(a); }); }
        NumericExpr<REAL>& set_sinh(const BaseExpr& a) override { return set_op(a, [](REAL a){ return sinh(a); }); }
        NumericExpr<REAL>& set_cosh(const BaseExpr& a) override { return set_op(a, [](REAL a){ return cosh(a); }); }
        NumericExpr<REAL>& set_tanh(const BaseExpr& a) override { return set_op(a, [](REAL a){ return tanh(a); }); }
        NumericExpr<REAL>& set_erf(const BaseExpr& a) override { return set_op(a, [](REAL a){ return erf(a); }); }
        NumericExpr<REAL>& set_erfc(const BaseExpr& a) override { return set_op(a, [](REAL a){ return erfc(a); }); }
        NumericExpr<REAL>& set_tgamma(const BaseExpr& a) override { return set_op(a, [](REAL a){ return tgamma(a); }); }
        NumericExpr<REAL>& set_lgamma(const BaseExpr& a) override { return set_op(a, [](REAL a){ return lgamma(a); }); }
        NumericExpr<REAL>& set_floor(const BaseExpr& a) override { return set_op(a, [](REAL a){ return floor(a); }); }
        NumericExpr<REAL>& set_ceil(const BaseExpr& a) override { return set_op(a, [](REAL a){ return ceil(a); }); }
        NumericExpr<REAL>& set_sign(const BaseExpr& a) override { return set_op(a, [](REAL a){ return ((a > 0) ? 1 : 0) - ((a < 0) ? 1 : 0); }); }
        NumericExpr<REAL>& set_sq(const BaseExpr& a) override { return set_op(a, [](REAL a){ return a*a; }); }
        NumericExpr<REAL>& set_pow(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return pow(a, b); });}
        NumericExpr<REAL>& set_atan2(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return atan2(a, b); });}
        NumericExpr<REAL>& set_fmod(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return fmod(a, b); });}
        NumericExpr<REAL>& set_hypot(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return hypot(a, b); });}
        NumericExpr<REAL>& set_min(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return min(a, b); });}
        NumericExpr<REAL>& set_max(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return max(a, b); });}

        NumericExpr<REAL>& set_fma(const BaseExpr& a, const BaseExpr& b, const BaseExpr& c) override { return set_op(a, b, c, [](REAL a, REAL b, REAL c){ return fma(a, b, c); }); }
        NumericExpr<REAL>& set_ifelse(const BaseExpr& a, const BaseExpr& b, const BaseExpr& c) override { return set_op(a, b, c, [](REAL a, REAL b, REAL c){ return (a > 0.5) ? b : c; }); }
        NumericExpr<REAL>& set_not(const BaseExpr& a) override { return set_op(a, [](REAL a){ return a > 0.5 ? 0.0 : 1.0; }); }
        NumericExpr<REAL>& set_lt(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a < b) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_eq(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a == b) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_ne(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a != b) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_le(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a <= b) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_gt(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a > b) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_ge(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a >= b) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_and(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a > 0.5 && b > 0.5) ? 1.0 : 0.0; });}
        NumericExpr<REAL>& set_or(const BaseExpr& a, const BaseExpr& b) override { return set_op(a, b, [](REAL a, REAL b){ return (a > 0.5 || b > 0.5) ? 1.0 : 0.0; });}
        
        std::ostream& operator<<(std::ostream& out) const override;
        NumericExpr<REAL>& set_transpose(const BaseExpr& a) override;
        NumericExpr<REAL>& set_dot(const BaseExpr& a, const BaseExpr& b) override;
        NumericExpr<REAL>& set_cross(const BaseExpr& a, const BaseExpr& b) override;
        NumericExpr<REAL>& set_mtimes(const BaseExpr** m_begin, const BaseExpr** m_end) override;
    private:
        template<typename ONEARG_FUNCTOR>
        NumericExpr<REAL>& set_op(const BaseExpr& a, const ONEARG_FUNCTOR& transform);
        template<typename TWOARG_FUNCTOR>
        NumericExpr<REAL>& set_op(const BaseExpr& a, const BaseExpr& b, const TWOARG_FUNCTOR& op);
        template<typename THREEARG_FUNCTOR>
        NumericExpr<REAL>& set_op(const BaseExpr& a, const BaseExpr& b, const BaseExpr& c, const THREEARG_FUNCTOR& op);
    };
}

#include "numeric_expr.inl"

#endif //CARNUM_ANI_NUMERICEXPR_H