//
// Created by Liogky Alexey on 10.01.2024.
//

#include "physical_tensors.h"
#include <cmath>

#ifndef ANIFEM_AUTODIFF_H
#define ANIFEM_AUTODIFF_H

namespace Ani{

struct ADExpr{};
template<bool _NotTrivialValue, bool _NotTrivialDeriv, bool _NotTrivialSecondDeriv>
struct ADState{
    static constexpr bool NotTrivialValue = _NotTrivialValue;
    static constexpr bool NotTrivialDeriv = _NotTrivialDeriv;
    static constexpr bool NotTrivialSecondDeriv = _NotTrivialSecondDeriv;
};
template<typename FT = double>
struct FTMathOperations{
    static inline FT pow(FT a, FT power) { return ::std::pow(a, power); }
    static inline FT atan2(FT y, FT x) { return ::std::atan2(y, x); }
    static inline FT hypot(FT x, FT y) { return ::std::hypot(x, y); }
    static inline FT exp(FT a) { return ::std::exp(a); }
    static inline FT expm1(FT a) { return ::std::expm1(a); }
    static inline FT log(FT a) { return ::std::log(a); }
    static inline FT log1p(FT a) { return ::std::log1p(a); }
    static inline FT sqrt(FT a) { return ::std::sqrt(a); }
    static inline FT cbrt(FT a) { return ::std::cbrt(a); }
    static inline FT sin(FT a) { return ::std::sin(a); }
    static inline FT cos(FT a) { return ::std::cos(a); }
    static inline FT tan(FT a) { return ::std::tan(a); }
    static inline FT atan(FT a) { return ::std::atan(a); }
    static inline FT asin(FT a) { return ::std::asin(a); }
    static inline FT acos(FT a) { return ::std::acos(a); }
    static inline FT sinh(FT a) { return ::std::sinh(a); }
    static inline FT cosh(FT a) { return ::std::cosh(a); }
    static inline FT tanh(FT a) { return ::std::tanh(a); }
};
template<typename ValueT, typename GradientT, typename HessianT, typename MathOpsT = FTMathOperations<ValueT>>
struct ADStorageSet{
    using ValueType = ValueT;
    using GradientType = GradientT;
    using HessianType = HessianT;
    using MathOps = MathOpsT;
};
template<std::size_t N, typename FT = double>
using PhysSymTensorStorageSet = ADStorageSet<FT, SymMtx<N, FT>, BiSymTensor4Rank<N, FT>>;

template<std::size_t N, typename FT = double>
using PhysTensorStorageSet = ADStorageSet<FT, PhysMtx<N, FT>, SymTensor4Rank<N, FT>>;

template<typename StateT = ADState<true, true, true>, typename StorageT = PhysSymTensorStorageSet<3, double> >
struct ADVal: public ADExpr{
    using State = StateT;
    using Storage = StorageT;
    using VT = typename Storage::ValueType;
    using GT = typename Storage::GradientType;
    using HT = typename Storage::HessianType;

    HT m_dd = HT();
    GT m_d = GT();
    VT m_v = VT();
    unsigned char m_dif = -1;

    ADVal() = default;
    ADVal(int numdif, VT val, GT d, HT dd): m_dd{std::move(dd)}, m_d{std::move(d)}, m_v{std::move(val)}, m_dif{static_cast<unsigned char>(numdif)}  {}

    VT operator()() const { return m_v; }
    const GT& D() const { return m_d; }
    const HT& DD() const { return m_dd; }

    void Init(int numdif, VT val, GT d, HT dd){
        m_v = std::move(val);
        if (numdif >= 1) m_d = std::move(d);
        if (numdif >= 2) m_dd = std::move(dd);
        m_dif = numdif;
    }
};

template<typename StorageT = PhysSymTensorStorageSet<3, double> >
struct Param: public ADExpr{
    using State = ADState<true, false, false>;
    using Storage = StorageT;
    using VT = typename Storage::ValueType;
    using GT = typename Storage::GradientType;
    using HT = typename Storage::HessianType;

    VT m_v = 0;
    const unsigned char m_dif = 0;

    Param() = default;
    explicit Param(VT val): m_v{std::move(val)} {}
    void Init(VT val) { m_v = std::move(val); }

    VT operator()() const { return m_v; }
    GT D() const { return GT(); }
    HT DD() const { return HT(); }
};

template<typename EX1, typename EX2, typename std::enable_if<std::is_same<typename EX1::Storage, typename EX2::Storage>::value>::type* = nullptr>
using UnionVal = ADVal< ADState<
                    EX1::State::NotTrivialValue || EX2::State::NotTrivialValue, 
                    EX1::State::NotTrivialDeriv || EX2::State::NotTrivialDeriv,
                    EX1::State::NotTrivialSecondDeriv || EX2::State::NotTrivialSecondDeriv>, 
                typename EX1::Storage
                >;
template<typename EX1, typename EX2, typename std::enable_if<std::is_same<typename EX1::Storage, typename EX2::Storage>::value>::type* = nullptr>
using MulVal = ADVal< ADState<
                    EX1::State::NotTrivialValue || EX2::State::NotTrivialValue, 
                    EX1::State::NotTrivialDeriv || EX2::State::NotTrivialDeriv,
                    EX1::State::NotTrivialSecondDeriv || EX2::State::NotTrivialSecondDeriv || 
                    (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialDeriv)>,
                typename EX1::Storage
                >;  
template<typename EX1>
using CopVal = ADVal< typename EX1::State, typename EX1::Storage >;
template<typename EX1>
using FuncVal = ADVal< ADState<true, EX1::State::NotTrivialDeriv, EX1::State::NotTrivialDeriv>, typename EX1::Storage>;
template<typename EX1>
using ADCheck1 = std::enable_if<std::is_base_of<ADExpr, EX1>::value, void>; 
template<typename EX1, typename EX2>
using ADCheck2 = std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value && std::is_same<typename EX1::Storage, typename EX2::Storage>::value, void>;                            

template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> make_ADVal(EX1 a);
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
ADVal<ADState<true, true, true>, typename EX1::Storage> make_shell(EX1 a);

template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator-(EX1 a);
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
UnionVal<EX1, EX2> operator+(EX1 a, EX2 b);
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
UnionVal<EX1, EX2> operator-(EX1 a, EX2 b);
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
MulVal<EX1, EX2> operator*(EX1 a, EX2 b);
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
MulVal<EX1, EX2> operator/(EX1 a, EX2 b);

template<typename EX1>
using _EXParam = Param<typename EX1::Storage>;
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator+(EX1 a, typename EX1::Storage::ValueType b) { return a + _EXParam<EX1>(b); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator+(typename EX1::Storage::ValueType b, EX1 a) { return _EXParam<EX1>(b) + a; }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator-(EX1 a, typename EX1::Storage::ValueType b) { return a - _EXParam<EX1>(b); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator-(typename EX1::Storage::ValueType b, EX1 a) { return _EXParam<EX1>(b) - a; }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator*(EX1 a, typename EX1::Storage::ValueType b) { return a * _EXParam<EX1>(b); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator*(typename EX1::Storage::ValueType b, EX1 a) { return _EXParam<EX1>(b) * a; }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator/(EX1 a, typename EX1::Storage::ValueType b) { return a / _EXParam<EX1>(b); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
CopVal<EX1> operator/(typename EX1::Storage::ValueType b, EX1 a) { return _EXParam<EX1>(b) / a; }

template<typename EX1, typename ADCheck1<EX1>::type* = nullptr> 
FuncVal<EX1> pow(EX1 a, typename EX1::Storage::ValueType q);
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr> 
FuncVal<EX1> pow(EX1 a, long q);
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr> 
FuncVal<EX1> pow(EX1 a, int q) { return pow(a, long(q)); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
auto pow(EX1 a, EX2 b);
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
FuncVal<UnionVal<EX1, EX2>> atan2(EX1 a, EX2 b);
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> hypot(const EX1* vec, ::std::size_t nvec);
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> hypot(::std::initializer_list<EX1> vec){ return hypot(vec.begin(), vec.size()); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr> 
FuncVal<EX1> pow(typename EX1::Storage::ValueType a, EX1 q) { return pow(_EXParam<EX1>(a), q); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> sq(EX1 a);
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
auto cube(EX1 a) { return pow(a, 3L); }
template<typename EX1, typename OP, typename ADCheck1<EX1>::type* = nullptr>

FuncVal<EX1> unary_operation(EX1 a, OP op = OP());
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> exp(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned , VT a){ auto v = EX1::Storage::MathOps::exp(a); return ::std::array<VT, 3>{v, v, v}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> expm1(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned , VT a){ auto v = EX1::Storage::MathOps::expm1(a); return ::std::array<VT, 3>{v, v+1, v+1}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> log(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned , VT a){ auto v = EX1::Storage::MathOps::log(a); return ::std::array<VT, 3>{v, 1/a, -1/(a*a)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> log1p(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned , VT a){ auto v = EX1::Storage::MathOps::log1p(a); return ::std::array<VT, 3>{v, 1/(a+1), -1/((a+1)*(a+1))}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> sqrt(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::sqrt(a); return ::std::array<VT, 3>{v, (dif >= 1 ? (1/(2*v)) : 0.), (dif >= 2 ? -1/(4*v*a) : 0)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> cbrt(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::cbrt(a); return ::std::array<VT, 3>{v, (dif >= 1 ? (1/(3*v*v)) : 0.), (dif >= 2 ? -2/(9*v*v*a) : 0)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> sin(EX1 a) {  using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::sin(a); return ::std::array<VT, 3>{v, (dif >= 1 ? EX1::Storage::MathOps::cos(a) : 0.), -v}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> cos(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::cos(a); return ::std::array<VT, 3>{v, (dif >= 1 ? -EX1::Storage::MathOps::sin(a) : 0.), -v}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> tan(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::tan(a), q = (dif >= 1) ? EX1::Storage::MathOps::cos(a) : 1.; VT t = q*q; return ::std::array<VT, 3>{v, (dif >= 1 ? 1/t : 0.), (dif >= 2 ? 2*v/t : 0.)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> atan(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::atan(a); return ::std::array<VT, 3>{v, (dif >= 1 ? 1./(1 + a*a) : 0.), (dif >= 2 ? -2*a/((1 + a*a)*(1 + a*a)) : 0.)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> asin(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::asin(a), t = ((dif >= 1) ? EX1::Storage::MathOps::sqrt(1-a*a) : 1.); return ::std::array<VT, 3>{v, (dif >= 1 ? 1/t : 0.), (dif >= 2 ? a / ((1 - a*a)*t): 0.)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> acos(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::acos(a), t = ((dif >= 1) ? EX1::Storage::MathOps::sqrt(1-a*a) : 1.); return ::std::array<VT, 3>{v, (dif >= 1 ? -1/t : 0.), (dif >=2 ? -a / ((1 - a*a)*t): 0.)}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> sinh(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::sinh(a); return ::std::array<VT, 3>{v, (dif >= 1 ? EX1::Storage::MathOps::cosh(a) : 0.), v}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> cosh(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::cosh(a); return ::std::array<VT, 3>{v, (dif >= 1 ? EX1::Storage::MathOps::sinh(a) : 0.), v}; }); }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
FuncVal<EX1> tanh(EX1 a) { using VT = typename EX1::Storage::ValueType; return unary_operation(a, [](unsigned dif, VT a){ auto v = EX1::Storage::MathOps::tanh(a), q = (dif >= 1) ? EX1::Storage::MathOps::cosh(a) : 1.; VT t = q * q; return ::std::array<VT, 3>{v, (dif >= 1 ? 1 / t : 0.), (dif >= 2 ? -2*v/t : 0.)}; }); }

template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
auto abs(EX1 a) { return a() >= typename EX1::Storage::ValueType(0) ? make_ADVal(a) : -a; }
template<typename EX1, typename ADCheck1<EX1>::type* = nullptr>
ADVal<ADState<true, false, false>, typename EX1::Storage> sign(EX1 a) { using VT = typename EX1::Storage::ValueType; auto v = a(); return {a.m_dif, VT((v > VT(0) ? 1 : 0) - (v < VT(0) ? 1 : 0)), typename EX1::Storage::GradientType(), typename EX1::Storage::HessianType()}; }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
typename std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value, bool>::type operator<(EX1 a, EX2 b) { return a() < b(); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
typename std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value, bool>::type operator>(EX1 a, EX2 b) { return a() > b(); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
typename std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value, bool>::type operator<=(EX1 a, EX2 b) { return a() <= b(); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
typename std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value, bool>::type operator>=(EX1 a, EX2 b) { return a() >= b(); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
typename std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value, bool>::type operator==(EX1 a, EX2 b) { return a() == b(); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
typename std::enable_if<std::is_base_of<ADExpr, EX1>::value && std::is_base_of<ADExpr, EX2>::value, bool>::type operator!=(EX1 a, EX2 b) { return a() != b(); }
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
UnionVal<EX1, EX2> min(EX1 a, EX2 b);
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* = nullptr>
UnionVal<EX1, EX2> max(EX1 a, EX2 b);

}

#include "autodiff.inl"

#endif //ANIFEM_AUTODIFF_H


