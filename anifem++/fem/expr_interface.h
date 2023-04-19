//
// Created by Liogky Alexey on 10.01.2023.
//

#ifndef CARNUM_ANI_FLOATABLE_H
#define CARNUM_ANI_FLOATABLE_H

#include <numeric>
#include <memory>
#include <cmath>
#include <ostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <string>
#include <initializer_list>

namespace Ani{
    struct BaseExpr{
        virtual std::shared_ptr<BaseExpr> makeExpr(std::size_t sz1 = 1, std::size_t sz2 = 1) const = 0;
        virtual std::shared_ptr<BaseExpr> makeCopy() const = 0;
        virtual std::size_t size1() const = 0;
        virtual std::size_t size2() const = 0;
        virtual BaseExpr& reshape(std::size_t sz1, std::size_t sz2) = 0;
        virtual BaseExpr& set_val(double val) = 0;
        virtual BaseExpr& set_val(double* val) = 0;
        virtual BaseExpr& set_val(const BaseExpr& val) = 0;
        virtual BaseExpr& set_val(const BaseExpr& val, std::size_t i, std::size_t j) { return set_sub_matrix(val, &i, 1, &j, 1);  }
        virtual BaseExpr& set_sub_matrix(const BaseExpr& val, const std::size_t* i, std::size_t i_arr_sz, const std::size_t* j, long j_arr_sz) = 0;
        virtual BaseExpr& set_sub_matrix(const BaseExpr& val, std::size_t i_start, long i_inc, std::size_t j_start, long j_inc) = 0;
        virtual BaseExpr& set_horzcat(const BaseExpr** vals_begin, const BaseExpr** vals_end) = 0;
        virtual BaseExpr& set_vertcat(const BaseExpr** vals_begin, const BaseExpr** vals_end) = 0;
        virtual BaseExpr& set_row(const BaseExpr& val, std::size_t i) { return set_sub_matrix(val, i, size1(), 0, 1); }
        virtual BaseExpr& set_col(const BaseExpr& val, std::size_t j) { return set_sub_matrix(val, 0, 1, j, size2()); }
        virtual BaseExpr& set_scal_sum(const BaseExpr** vals_begin, const BaseExpr** vals_end, const double* coefs) = 0;
        virtual BaseExpr& set_sum(const BaseExpr** vals_begin, const BaseExpr** vals_end) = 0;
        virtual BaseExpr& set_sum(const BaseExpr& a, const BaseExpr& b){ const BaseExpr* arr[2]{&a, &b}; return set_sum(arr, arr+2); }
        virtual BaseExpr& set_mul(const BaseExpr** vals_begin, const BaseExpr** vals_end) = 0;
        virtual BaseExpr& set_mul(const BaseExpr& a, const BaseExpr& b){ const BaseExpr* arr[2]{&a, &b}; return set_mul(arr, arr+2); }
        virtual BaseExpr& set_sub(const BaseExpr& a, const BaseExpr& b) = 0;
        virtual BaseExpr& set_swap_sign(const BaseExpr& a) = 0;// = -a
        virtual BaseExpr& set_div(const BaseExpr& a, const BaseExpr& b) = 0;

        virtual BaseExpr& set_abs(const BaseExpr& a) = 0;
        virtual BaseExpr& set_exp(const BaseExpr& a) = 0;
        virtual BaseExpr& set_expm1(const BaseExpr& a) = 0;
        virtual BaseExpr& set_log(const BaseExpr& a) = 0;
        virtual BaseExpr& set_log1p(const BaseExpr& a) = 0;
        virtual BaseExpr& set_sqrt(const BaseExpr& a) = 0;
        virtual BaseExpr& set_cbrt(const BaseExpr& a) = 0;
        virtual BaseExpr& set_sin(const BaseExpr& a) = 0;
        virtual BaseExpr& set_cos(const BaseExpr& a) = 0;
        virtual BaseExpr& set_tan(const BaseExpr& a) = 0;
        virtual BaseExpr& set_asin(const BaseExpr& a) = 0;
        virtual BaseExpr& set_acos(const BaseExpr& a) = 0;
        virtual BaseExpr& set_atan(const BaseExpr& a) = 0;
        virtual BaseExpr& set_sinh(const BaseExpr& a) = 0;
        virtual BaseExpr& set_cosh(const BaseExpr& a) = 0;
        virtual BaseExpr& set_tanh(const BaseExpr& a) = 0;
        virtual BaseExpr& set_erf(const BaseExpr& a) = 0;
        virtual BaseExpr& set_erfc(const BaseExpr& a) = 0;
        virtual BaseExpr& set_tgamma(const BaseExpr& a) = 0;
        virtual BaseExpr& set_lgamma(const BaseExpr& a) = 0;
        virtual BaseExpr& set_floor(const BaseExpr& a) = 0;
        virtual BaseExpr& set_ceil(const BaseExpr& a) = 0;
        virtual BaseExpr& set_sign(const BaseExpr& a) = 0;
        virtual BaseExpr& set_sq(const BaseExpr& x) { return set_mul(x, x); }

        virtual BaseExpr& set_pow(const BaseExpr& a, const BaseExpr& b) = 0;
        virtual BaseExpr& set_atan2(const BaseExpr& a, const BaseExpr& b) = 0;
        virtual BaseExpr& set_fmod(const BaseExpr& a, const BaseExpr& b) = 0;
        virtual BaseExpr& set_hypot(const BaseExpr& a, const BaseExpr& b) = 0;
        virtual BaseExpr& set_min(const BaseExpr** vals_begin, const BaseExpr** vals_end) = 0;
        virtual BaseExpr& set_max(const BaseExpr** vals_begin, const BaseExpr** vals_end) = 0;
        virtual BaseExpr& set_min(const BaseExpr& a, const BaseExpr& b){ const BaseExpr* arr[2]{&a, &b}; return set_min(arr, arr+2); }
        virtual BaseExpr& set_max(const BaseExpr& a, const BaseExpr& b){ const BaseExpr* arr[2]{&a, &b}; return set_max(arr, arr+2); }

        virtual BaseExpr& set_fma(const BaseExpr& a, const BaseExpr& b, const BaseExpr& c) = 0; //a*b+c

        virtual BaseExpr& set_ifelse(const BaseExpr& x, const BaseExpr& y, const BaseExpr& z) = 0;
        virtual BaseExpr& set_not(const BaseExpr& x) = 0;
        virtual BaseExpr& set_lt(const BaseExpr& x, const BaseExpr& y) = 0;
        virtual BaseExpr& set_eq(const BaseExpr& x, const BaseExpr& y) = 0;
        virtual BaseExpr& set_and(const BaseExpr& x, const BaseExpr& y) = 0;
        virtual BaseExpr& set_or(const BaseExpr& x, const BaseExpr& y) = 0;
        virtual BaseExpr& set_ne(const BaseExpr& x, const BaseExpr& y) { return set_not(set_eq(x, y)); }
        virtual BaseExpr& set_le(const BaseExpr& x, const BaseExpr& y) { return set_not(set_lt(y, x)); }
        virtual BaseExpr& set_gt(const BaseExpr& x, const BaseExpr& y) { return set_lt(y, x); }
        virtual BaseExpr& set_ge(const BaseExpr& x, const BaseExpr& y) { return set_not(set_lt(x, y)); }

        virtual std::ostream& operator<<(std::ostream& out) const  { throw std::runtime_error("operator<< is not implemented for this Expr"); return out;  }

        virtual BaseExpr& set_transpose(const BaseExpr& x) = 0;
        virtual BaseExpr& set_cross(const BaseExpr& x, const BaseExpr& y) = 0;
        virtual BaseExpr& set_dot(const BaseExpr& x, const BaseExpr& y) = 0;
        virtual BaseExpr& set_norm(const BaseExpr& x) { return set_sqrt(set_dot(x, x)); }
        virtual BaseExpr& set_mtimes(const BaseExpr** m_begin, const BaseExpr** m_end) = 0;
        virtual BaseExpr& set_mtimes(const BaseExpr& a, const BaseExpr& b){ const BaseExpr* arr[2]{&a, &b}; return set_mtimes(arr, arr+2); }

        std::size_t nrows() const { return size1(); }
        std::size_t ncols() const { return size2(); }
    };

    struct Expr{
        std::shared_ptr<BaseExpr> m_invoker;

        template<typename ExprT>
        explicit Expr(const ExprT& f, typename std::enable_if<std::is_base_of<BaseExpr, ExprT>::value>::type* = 0): m_invoker{new ExprT(f)} {}
        template<typename ExprT>
        explicit Expr(ExprT&& f, typename std::enable_if<std::is_base_of<BaseExpr, ExprT>::value>::type* = 0): m_invoker{new ExprT(std::move(f))} {}
        Expr(const Expr &other): m_invoker(other.m_invoker->makeCopy()) {} 
        Expr(Expr &&) = default;
        Expr() = default;
        explicit Expr(const std::shared_ptr<BaseExpr>& expr): m_invoker(std::move(expr->makeCopy())) {}
        explicit Expr(std::shared_ptr<BaseExpr>&& expr): m_invoker(std::move(expr)) {}
        Expr& operator=(const Expr &f){ return m_invoker = f.m_invoker->makeCopy(), *this; }
        Expr& operator=(Expr &&f){ return m_invoker = std::move(f.m_invoker), *this; }
        template<typename ExprT, typename std::enable_if<std::is_base_of<BaseExpr, ExprT>::value>::type* = 0>
        Expr& operator=(const ExprT& x){ 
            if (m_invoker)
                m_invoker->set_val(x);
            else    
                m_invoker = std::move(x.makeCopy()); 
            return *this; 
        }
        template<typename ExprT, typename std::enable_if<std::is_base_of<BaseExpr, ExprT>::value>::type* = 0>
        Expr& operator=(std::shared_ptr<ExprT>&& x){ m_invoker = std::move(x); return *this; }
        template<typename ExprT, typename std::enable_if<std::is_base_of<BaseExpr, ExprT>::value>::type* = 0>
        Expr& operator=(const std::shared_ptr<ExprT>& x){ m_invoker = std::move(x->makeCopy()); return *this; }
        
        Expr& operator=(double x){ 
            if (!m_invoker)
                throw std::runtime_error("Assignment of value is available only for initialized expressions");
            return m_invoker->set_val(x), *this;    
        }
        Expr& operator=(double* x){ 
            if (!m_invoker) 
                throw std::runtime_error("Assignment of value is available only for initialized expressions");
            return m_invoker->set_val(x), *this;    
        }
        Expr& reshape(std::size_t sz1, std::size_t sz2){
            if (!m_invoker) 
                throw std::runtime_error("Expressions must be initialized to apply operations");
            return m_invoker->reshape(sz1, sz2), *this;   
        }
        std::size_t nrows() const { return m_invoker ? m_invoker->size1() : 0; }
        std::size_t ncols() const { return m_invoker ? m_invoker->size2() : 0; }
        Expr zeroes(std::size_t sz1 = 1, std::size_t sz2 = 1) const {
            if (!m_invoker) 
                throw std::runtime_error("Expressions must be initialized to apply operations");
            return Expr(m_invoker->makeExpr(sz1, sz2));   
        }
        Expr& set(const Expr& other, std::size_t sz1, std::size_t sz2){
            if (other.ncols() != 1 || other.nrows() != 1)
                throw std::runtime_error("This operation is available only for 1x1 input");
            if (m_invoker)
                m_invoker->set_val(*other.m_invoker, sz1, sz2); 
            else  
                (*this = other).m_invoker->set_val(*other.m_invoker, sz1, sz2);
            return *this;          
        }
        Expr operator()(std::size_t i, std::size_t j) const {
            if (i > nrows() || j > ncols())
                throw std::runtime_error("Sizes of expression: " + std::to_string(nrows()) + "x" + std::to_string(ncols()) 
                    + ", but requested value at (" + std::to_string(i) + ", " + std::to_string(j) + ")");
            Expr res = zeroes();
            return res.m_invoker->set_val(*m_invoker, i, j), res;
        }
        Expr row(std::size_t i) const {
            Expr res = zeroes();
            return res.m_invoker->set_row(*m_invoker, i), res;
        }
        Expr col(std::size_t j) const {
            Expr res = zeroes();
            return res.m_invoker->set_col(*m_invoker, j), res;
        }
        Expr T() const {
            Expr res = zeroes();
            return res.m_invoker->set_transpose(*m_invoker), res;
        }

        template<typename ExprT = BaseExpr>
        ExprT* target() { return static_cast<ExprT *>(m_invoker.get()); }
        template<typename ExprT = BaseExpr>
        const ExprT* target() const { return static_cast<const ExprT *>(m_invoker.get()); }
        
        /// @brief Get stored data
        std::shared_ptr<BaseExpr> base() const { return m_invoker; }
        virtual Expr& operator+=(const Expr& y)  { return m_invoker->set_sum(*m_invoker, *y.m_invoker), *this; }
        virtual Expr& operator-=(const Expr& y)  { return m_invoker->set_sub(*m_invoker, *y.m_invoker), *this; }
        virtual Expr& operator*=(const Expr& y)  { return m_invoker->set_mul(*m_invoker, *y.m_invoker), *this; } 
        virtual Expr& operator/=(const Expr& y)  { return m_invoker->set_div(*m_invoker, *y.m_invoker), *this; }
    };
    inline std::ostream& operator<<(std::ostream& out, const Expr& a);
    inline Expr operator+(const Expr& m1, const Expr& m2);
    inline Expr operator-(const Expr& m1, const Expr& m2);
    inline Expr operator*(const Expr& m1, const Expr& m2);
    inline Expr operator/(const Expr& m1, const Expr& m2);
    inline Expr operator-(const Expr& m);

    inline Expr operator!(const Expr& m);
    inline Expr operator<(const Expr& m1, const Expr& m2);
    inline Expr operator<=(const Expr& m1, const Expr& m2);
    inline Expr operator>(const Expr& m1, const Expr& m2);
    inline Expr operator>=(const Expr& m1, const Expr& m2);
    inline Expr operator==(const Expr& m1, const Expr& m2);
    inline Expr operator!=(const Expr& m1, const Expr& m2);
    inline Expr operator&&(const Expr& m1, const Expr& m2);
    inline Expr operator||(const Expr& m1, const Expr& m2);

    namespace FT{
        inline Expr scalsum(const Expr* st, const Expr* end, const double* coefs);
        inline Expr submatrix(const Expr& m, const std::size_t* i, std::size_t i_arr_sz, const std::size_t* j, long j_arr_sz);
        inline Expr submatrix(const Expr& m, std::initializer_list<std::size_t> i, std::initializer_list<std::size_t> j) { return submatrix(m, i.begin(), i.size(), j.begin(), j.size()); }
        inline Expr submatrix(const Expr& m, std::size_t i_start, long i_inc, std::size_t j_start, long j_inc); 
        inline Expr vertcat(const Expr* st, const Expr* end);
        inline Expr vertcat(std::initializer_list<Expr> a){ return vertcat(a.begin(), a.end()); }
        inline Expr horzcat(const Expr* st, const Expr* end);
        inline Expr horzcat(std::initializer_list<Expr> a){ return horzcat(a.begin(), a.end()); }
        inline Expr abs(const Expr& m);
        inline Expr exp(const Expr& m);
        inline Expr expm1(const Expr& m);
        inline Expr log(const Expr& m);
        inline Expr log1p(const Expr& m);
        inline Expr sqrt(const Expr& m);
        inline Expr cbrt(const Expr& m);
        inline Expr sin(const Expr& m);
        inline Expr cos(const Expr& m);
        inline Expr tan(const Expr& m);
        inline Expr asin(const Expr& m);
        inline Expr acos(const Expr& m);
        inline Expr atan(const Expr& m);
        inline Expr sinh(const Expr& m);
        inline Expr cosh(const Expr& m);
        inline Expr tanh(const Expr& m);
        inline Expr erf(const Expr& m);
        inline Expr erfc(const Expr& m);
        inline Expr tgamma(const Expr& m);
        inline Expr lgamma(const Expr& m);
        inline Expr floor(const Expr& m);
        inline Expr ceil(const Expr& m);
        inline Expr sign(const Expr& m);
        inline Expr sq(const Expr& m);
        inline Expr norm(const Expr& m);
        inline Expr pow(const Expr& m1, const Expr& m2);
        inline Expr atan2(const Expr& m1, const Expr& m2);
        inline Expr fmod(const Expr& m1, const Expr& m2);
        inline Expr min(const Expr& m1, const Expr& m2);
        inline Expr max(const Expr& m1, const Expr& m2);
        inline Expr hypot(const Expr& m1, const Expr& m2);
        inline Expr dot(const Expr& m1, const Expr& m2);
        inline Expr cross(const Expr& m1, const Expr& m2);
        inline Expr mtimes(const Expr& m1, const Expr& m2);
        inline Expr fma(const Expr& x, const Expr& y, const Expr& z);
        inline Expr ifelse(const Expr& condotion, const Expr& on_true, const Expr& on_false);
        inline Expr min(const Expr* st, const Expr* end);
        inline Expr max(const Expr* st, const Expr* end);
        inline Expr mtimes(const Expr* st, const Expr* end);
        inline Expr det3x3(const Expr& m);
    };
}

#include "expr_interface.inl"

#endif //CARNUM_ANI_FLOATABLE_H