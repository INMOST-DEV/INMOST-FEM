//
// Created by Liogky Alexey on 10.01.2024.
//

#ifndef ANIFEM_AUTODIFF_INL
#define ANIFEM_AUTODIFF_INL

#include "autodiff.h"

namespace Ani{
#if __cplusplus >= 201703L                    
#define CONSTIF if constexpr
#else 
#define CONSTIF if
#endif

template<typename EX1, typename ADCheck1<EX1>::type* Dummy>
CopVal<EX1> make_ADVal(EX1 a){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto v = a();
    GT m;
    CONSTIF (EX1::State::NotTrivialDeriv)
        m = a.D();
    HT r;
    CONSTIF (EX1::State::NotTrivialSecondDeriv)
        r = a.DD();
    return {a.m_dif, v, m, r};         
}

template<typename EX1, typename ADCheck1<EX1>::type* Dummy>
ADVal<ADState<true, true, true>, typename EX1::Storage>  make_shell(EX1 a){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto v = a();
    auto dif = a.m_dif;
    GT m;
    HT r;
    CONSTIF (EX1::State::NotTrivialDeriv){
        if (dif >= 1){
            m = a.D();
            CONSTIF (EX1::State::NotTrivialSecondDeriv){
                if (dif >= 2)
                    r = a.DD();   
            } else 
                dif = 1;
        }
    } else  
        dif = 0;
        
    return {a.m_dif, v, m, r};     
}
template<typename EX1, typename ADCheck1<EX1>::type* Dummy>
CopVal<EX1> operator-(EX1 a){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto v = -a();
    unsigned char dif = a.m_dif;
    GT m;
    CONSTIF (EX1::State::NotTrivialDeriv)
        if (dif >= 1)
            m = -a.D();
    HT r;
    CONSTIF (EX1::State::NotTrivialSecondDeriv)
        if (dif >= 2)
            r = -a.DD();
    return {dif, v, m, r};                  
}
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
UnionVal<EX1, EX2> operator+(EX1 a, EX2 b){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto v = a() + b();
    unsigned char dif = std::max(a.m_dif, b.m_dif);
    GT m;
    if ((EX1::State::NotTrivialDeriv || EX2::State::NotTrivialValue) && dif >= 1){
        CONSTIF (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue) m = a.D() + b.D();
        else CONSTIF (EX1::State::NotTrivialDeriv) m = a.D();
        else CONSTIF (EX2::State::NotTrivialDeriv) m = b.D();
    }
    HT r;
    if ((EX1::State::NotTrivialSecondDeriv || EX2::State::NotTrivialSecondDeriv) && dif >= 2){
        CONSTIF (EX1::State::NotTrivialSecondDeriv && EX2::State::NotTrivialSecondDeriv) r = a.DD() + b.DD();
        else CONSTIF (EX1::State::NotTrivialSecondDeriv) r = a.DD();
        else CONSTIF (EX2::State::NotTrivialSecondDeriv) r = b.DD();
    }
    return UnionVal<EX1, EX2>(dif, v, m, r);
}
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
UnionVal<EX1, EX2> operator-(EX1 a, EX2 b){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto v = a() - b();
    unsigned char dif = std::max(a.m_dif, b.m_dif);
    GT m;
    if ((EX1::State::NotTrivialDeriv || EX2::State::NotTrivialValue) && dif >= 1){
        CONSTIF (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue) m = a.D() - b.D();
        else CONSTIF (EX1::State::NotTrivialDeriv) m = a.D();
        else CONSTIF (EX2::State::NotTrivialDeriv) m -= b.D();
    }
    HT r;
    if ((EX1::State::NotTrivialSecondDeriv || EX2::State::NotTrivialSecondDeriv) && dif >= 2){
        CONSTIF (EX1::State::NotTrivialSecondDeriv && EX2::State::NotTrivialSecondDeriv) r = a.DD() - b.DD();
        else CONSTIF (EX1::State::NotTrivialSecondDeriv) r = a.DD();
        else CONSTIF (EX2::State::NotTrivialSecondDeriv) r -= b.DD();
    }
    return UnionVal<EX1, EX2>(dif, v, m, r);
}
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
MulVal<EX1, EX2> operator*(EX1 a, EX2 b){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto av = a(), bv = b();
    auto v = av * bv;
    unsigned char dif = std::max(a.m_dif, b.m_dif);
    GT am, bm, m;
    if ((EX1::State::NotTrivialDeriv || EX2::State::NotTrivialValue) && dif >= 1){
        CONSTIF (EX1::State::NotTrivialDeriv) {
            am = a.D();
            m += bv * am;
        }
        CONSTIF (EX2::State::NotTrivialDeriv) {
            bm = b.D();
            m += av * bm;
        }
    }
    HT r;
    if ((EX1::State::NotTrivialSecondDeriv || EX2::State::NotTrivialSecondDeriv || 
        (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue)) 
        && dif >= 2){
        CONSTIF (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue)
            r += HT::TensorSymMul2(am, bm);
        CONSTIF (EX1::State::NotTrivialSecondDeriv ) r += bv * a.DD();
        CONSTIF (EX2::State::NotTrivialSecondDeriv ) r += av * b.DD();
    }
    return {dif, v, m, r};
}
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
MulVal<EX1, EX2> operator/(EX1 a, EX2 b){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    auto av = a(), bv = b();
    auto v = av / bv;
    unsigned char dif = std::max(a.m_dif, b.m_dif);
    GT am, bm, m;
    if ((EX1::State::NotTrivialDeriv || EX2::State::NotTrivialValue) && dif >= 1){
        CONSTIF (EX1::State::NotTrivialDeriv) {
            am = a.D();
            m += am / bv;
        }
        CONSTIF (EX2::State::NotTrivialDeriv) {
            bm = b.D();
            m -= av / (bv * bv) * bm;
        }
    }
    HT r;
    if ((EX1::State::NotTrivialSecondDeriv || EX2::State::NotTrivialSecondDeriv || 
        (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue)) 
        && dif >= 2){
        CONSTIF (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue)
            r -= (1/(bv*bv))*HT::TensorSymMul2(am, bm);
        CONSTIF (EX1::State::NotTrivialSecondDeriv ) r += a.DD() / bv;
        CONSTIF (EX2::State::NotTrivialSecondDeriv ) {
            auto c = av / (bv * bv);
            auto rb = b.DD();
            r += c * ( (2./bv) * HT::TensorSquare(bm) - rb );
        }
    }
    return {dif, v, m, r};
}

template<typename EX1, typename ADCheck1<EX1>::type* Dummy> 
FuncVal<EX1> pow(EX1 a, typename EX1::Storage::ValueType q){
    using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    if (q == 0) return {0, VT(1), GT(), HT()};
    unsigned char dif = a.m_dif;
    auto av = a();
    auto v = EX1::Storage::MathOps::pow(av, q);
    CONSTIF (!EX1::State::NotTrivialDeriv)
        return {dif, v, GT(), HT()};
    GT am, m;
    CONSTIF (EX1::State::NotTrivialDeriv)
        if (dif >= 1){
            am = a.D();
            m = (q * v / av) * am;
        }    
    HT r;
    if (dif >= 2){
        r += (av != VT(0) ? q * (q - 1) * v / (av*av) : VT(0)) * HT::TensorSquare(am);
        CONSTIF (EX1::State::NotTrivialSecondDeriv )
            r += (av != VT(0) ? q * v / av : VT(0)) * a.DD();    
    } 
    return {dif, v, m, r};
}
template<typename EX1, typename ADCheck1<EX1>::type* Dummy> 
FuncVal<EX1> pow(EX1 a, long q){
    using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    if (q == 0) return {0, VT(1), GT(), HT()};
    unsigned char dif = a.m_dif;
    auto av = a();
    auto pv = (q > 0) ? av : (1. / av);
    VT v = pv, vm1 = VT(1), vm2 = VT(0);  //< q == 1
    if (q != 1) v = pv*pv, vm1 = pv, vm2 = VT(1);
    long qp = (q > 0) ? (q-2) : (-q);
    for (long i = 0; i < qp; ++i){
        vm2 = vm1;
        vm1 = v;
        v *= pv;
    }
    if (q < 0) ::std::swap(v, vm2);
    CONSTIF (!EX1::State::NotTrivialDeriv)
        return {dif, v, GT(), HT()};
    GT am, m;

    CONSTIF (EX1::State::NotTrivialDeriv)
        if (dif >= 1){
            am = a.D();
            m = (q == 1) ? am : ((q * vm1) * am);
        } 
    HT r;
    if (dif >= 2){
        if (q != 1)
            r += (q * (q - 1) * vm2) * HT::TensorSquare(am);
        CONSTIF (EX1::State::NotTrivialSecondDeriv )
            r += (q * vm1) * a.DD();    
    } 
    return {dif, v, m, r};
}

template<typename EX1, typename ADCheck1<EX1>::type* Dummy>
FuncVal<EX1> sq(EX1 a){
    // using VT = typename EX1::Storage::ValueType; 
    using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    unsigned char dif = a.m_dif;
    auto av = a();
    auto v = av * av;
    CONSTIF (!EX1::State::NotTrivialDeriv)
        return {dif, v, GT(), HT()};
    GT am, m;
    CONSTIF (EX1::State::NotTrivialDeriv)
        if (dif >= 1){
            am = a.D();
            m = (2 * av) * am;
        }
    HT r;
    if (dif >= 2){ 
        r += 2 * HT::TensorSquare(am);
        CONSTIF (EX1::State::NotTrivialSecondDeriv )
            r += (2 * av) * a.DD();    
    }
    return {dif, v, m, r};      
}

namespace internal{
    template<typename EX1, typename EX2, int CASE = ((!EX1::State::NotTrivialDeriv) ? 1 : 0) + ((!EX2::State::NotTrivialDeriv) ? 2 : 0)>
    struct EvalPow{
        static ADVal<ADState<true, true, true>, typename EX1::Storage> eval(EX1 a, EX2 b){
            using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
            unsigned char dif = std::max(a.m_dif, b.m_dif);
            auto av = a(), bv = b();
            auto v = EX1::Storage::MathOps::pow(av, bv);
            if (dif == 0) return {0, v, GT(), HT()}; 
            GT am = a.D(), bm = b.D();
            auto ln_a = EX1::Storage::MathOps::log(av);
            GT m = v * (ln_a * bm + (bv / av) * am);
            HT r;
            if (dif >= 2){
                HT ar, br;
                CONSTIF ( EX1::State::NotTrivialSecondDeriv ) ar = a.DD();
                CONSTIF ( EX2::State::NotTrivialSecondDeriv ) br = b.DD();
                VT c1 = v*ln_a*ln_a, c2 = v*(bv * ln_a + 1)/av, c3 = v*bv*(bv-1) / (av*av), c4 = v*ln_a, c5 = v*bv/av;
                // r = c1 * HT::TensorSquare(bm) + c2 * HT::TensorSymMul2(am, bm) + c3 * HT::TensorSquare(am) + c4 * br + c5 * ar;
                for (auto it = r.begin(); it != r.end(); ++it){
                    auto q = it.index();
                    std::array<std::size_t, r.rank()/2> id1, id2;
                    for (std::size_t rr = 0; rr < r.rank()/2; ++rr) id1[rr] = q[rr], id2[rr] = q[rr + r.rank()/2];
                    *it = c1 * bm(id1) * bm(id2) + c2 * ( bm(id1) * am(id2) + am(id1) * bm(id2) ) + c3 * am(id1) * am(id2) + c4 * br[q] + c5 * ar[q];
                }
            }
            return {dif, v, m, r};
        }
        
    };
    template<typename EX1, typename EX2>
    struct EvalPow<EX1, EX2, 3>{
        static auto eval(EX1 a, EX2 b){ return _EXParam<EX1>(EX1::Storage::MathOps::pow(a(), b())); }
    };
    template<typename EX1, typename EX2>
    struct EvalPow<EX1, EX2, 2>{
        static auto eval(EX1 a, EX2 b){ return pow(a, b()); }
    };
    template<typename EX1, typename EX2>
    struct EvalPow<EX1, EX2, 1>{
        static auto eval(EX1 a, EX2 b){ return exp(EX1::Storage::MathOps::log(a()) * b); }
    };
}
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
auto pow(EX1 a, EX2 b){ return internal::EvalPow<EX1, EX2>::eval(a, b); }

template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
FuncVal<UnionVal<EX1, EX2>> atan2(EX1 y, EX2 x){
    using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    unsigned char dif = std::max(y.m_dif, x.m_dif);
    auto vy = y(), vx = x();
    auto v = EX1::Storage::MathOps::atan2(vy, vx);
    GT m;
    GT mx, my;
    VT r2 = vy*vy + vx*vx;
    VT c1 = vx / r2, c2 = vy / r2;
    if ((EX1::State::NotTrivialDeriv || EX2::State::NotTrivialValue) && dif >= 1){
        CONSTIF (EX1::State::NotTrivialDeriv) my = y.D();
        CONSTIF (EX2::State::NotTrivialDeriv) mx = x.D();
        CONSTIF (EX1::State::NotTrivialDeriv && EX2::State::NotTrivialValue) m = c1*my - c2*mx;
        else CONSTIF (EX1::State::NotTrivialDeriv) m =  c1*my;
        else CONSTIF (EX2::State::NotTrivialDeriv) m = -c2*mx;
    }
    HT r;
    if (dif >= 2){
        HT ry, rx;
        CONSTIF (EX1::State::NotTrivialSecondDeriv ) ry = y.DD();
        CONSTIF (EX2::State::NotTrivialSecondDeriv ) rx = x.DD();
        // r = c1*ry - c2*rx + ( (vy*vy - vx*vx)*HT::TensorSymMul2(mx, my) - 2*vx*vy*(HT::TensorSquare(my) - HT::TensorSquare(mx)) ) / (r2*r2);
        auto c3 = (vy*vy - vx*vx) / (r2*r2), c4 = 2*vx*vy / (r2*r2);
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            std::array<std::size_t, r.rank()/2> id1, id2;
            for (std::size_t rr = 0; rr < r.rank()/2; ++rr) id1[rr] = q[rr], id2[rr] = q[rr + r.rank()/2];
            *it = c1*ry[q] - c2*rx[q] + c3 * (mx(id1)*my(id2) + my(id1)*mx(id2)) - c4 * (my(id1)*my(id2) - mx(id1)*mx(id2));
            // *it = c1*ry[q] - c2*rx[q] + ( (vy*vy - vx*vx)*(mx(id1)*my(id2) + my(id1)*mx(id2)) - 2*vx*vy*(my(id1)*my(id2) - mx(id1)*mx(id2)) ) / (r2*r2);
        }
        // for (auto it = r.begin(); it != r.end(); ++it){
        //     Sym4Tensor::Index q = it.index();
        //     auto i = q.i, j = q.j, k = q.k, l = q.l;
        //     *it = c1*ry[q] - c2*rx[q] + ((my(i,j)*mx(k,l) - my(k,l)*mx(i,j)) - 2 * (c1*my(i,j) - c2*mx(i,j))*(vx*mx(k,l) + vy*my(k,l))) / r2;
        // }
    }

    return FuncVal<UnionVal<EX1, EX2>>(dif, v, m, r);
}

template<typename EX1, typename ADCheck1<EX1>::type* Dummy>
FuncVal<EX1> hypot(const EX1* vec, ::std::size_t nvec){
    using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    if (nvec == 0) return FuncVal<EX1>();
    if (nvec == 1) return FuncVal<EX1>(vec[0].m_dif, vec[0](), vec[0].D(), vec[0].DD());
    VT v = VT(0), v2 = VT(0);
    if (nvec == 2) v = EX1::Storage::MathOps::hypot(vec[0](), vec[1]()), v2 = vec[0]()*vec[0]() + vec[1]()*vec[1]();
    else {
        v2 = std::accumulate(vec, vec+nvec, 0.0, [](VT sum, const EX1& ex) { return sum + ex()*ex(); });
        v = EX1::Storage::MathOps::sqrt(v2);
    }
    unsigned char dif = vec[0].m_dif;
    GT m;
    HT r;
    CONSTIF (EX1::State::NotTrivialDeriv){
        if (dif >= 1)
            m = std::accumulate(vec, vec+nvec, GT(), [](GT sum, const EX1& ex) { return sum + ex()*ex.D(); }) / v;
        if (dif >= 2){
            CONSTIF (EX1::State::NotTrivialSecondDeriv)   
                r = std::accumulate(vec, vec+nvec, HT(), [](HT sum, const EX1& ex) { return sum + ex()*ex.DD(); });  
            r += std::accumulate(vec, vec+nvec, HT(), [](auto sum, const EX1& ex) { return sum + HT::TensorSquare(ex.D()); });
            r -= HT::TensorSquare(m);
            r = r / v;        
        }
    }
    return FuncVal<EX1>(dif, v, m, r);
}

template<typename EX1, typename OP, typename ADCheck1<EX1>::type* Dummy>
FuncVal<EX1> unary_operation(EX1 a, OP op){
    using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    unsigned char dif = a.m_dif;
    auto av = a();
    ::std::array<VT, 3> vs = op(dif, av);
    VT v = vs[0];
    CONSTIF (!EX1::State::NotTrivialDeriv)
        return {dif, v, GT(), HT()};
    GT am, m;
    if (dif >= 1){
        am = a.D();
        m = vs[1] * am;
    }
    HT r;
    if (dif >= 2){ 
        r = vs[2] * HT::TensorSquare(am);
        CONSTIF (EX1::State::NotTrivialSecondDeriv)
            r += vs[1] * a.DD();        
    } 
    return {dif, v, m, r};   
}

template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
UnionVal<EX1, EX2> min(EX1 a, EX2 b){
    // using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    UnionVal<EX1, EX2> r;
    auto av = a(), bv = b();
    if (av <= bv){
        r.m_dif = a.m_dif;
        r.m_v = av;
        CONSTIF (EX1::State::NotTrivialDeriv)
            if (r.m_dif >= 1)
                r.m_d = a.D();
        CONSTIF (EX1::State::NotTrivialSecondDeriv)
            if (r.m_dif >= 2)
                r.m_dd = a.DD();        
    } else {
        r.m_dif = b.m_dif;
        r.m_v = bv;
        CONSTIF (EX2::State::NotTrivialDeriv)
            if (r.m_dif >= 1)
                r.m_d = b.D();
        CONSTIF (EX2::State::NotTrivialSecondDeriv)
            if (r.m_dif >= 2)
                r.m_dd = b.DD(); 
    }
    return r;
}
template<typename EX1, typename EX2, typename ADCheck2<EX1, EX2>::type* Dummy>
UnionVal<EX1, EX2> max(EX1 a, EX2 b){
    // using VT = typename EX1::Storage::ValueType; using GT = typename EX1::Storage::GradientType; using HT = typename EX1::Storage::HessianType;
    UnionVal<EX1, EX2> r;
    auto av = a(), bv = b();
    if (av >= bv){
        r.m_dif = a.m_dif;
        r.m_v = av;
        CONSTIF (EX1::State::NotTrivialDeriv)
            if (r.m_dif >= 1)
                r.m_d = a.D();
        CONSTIF (EX1::State::NotTrivialSecondDeriv)
            if (r.m_dif >= 2)
                r.m_dd = a.DD();        
    } else {
        r.m_dif = b.m_dif;
        r.m_v = bv;
        CONSTIF (EX2::State::NotTrivialDeriv)
            if (r.m_dif >= 1)
                r.m_d = b.D();
        CONSTIF (EX2::State::NotTrivialSecondDeriv)
            if (r.m_dif >= 2)
                r.m_dd = b.DD(); 
    }
    return r;
}


#undef  CONSTIF
}

#endif  //ANIFEM_AUTODIFF_INL