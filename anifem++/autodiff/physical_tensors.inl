//
// Created by Liogky Alexey on 10.01.2024.
//

#ifndef CARNUM_PHYSICAL_TENSORS_INL
#define CARNUM_PHYSICAL_TENSORS_INL

#include "physical_tensors.h"
#include "../fem/geometry.h"

namespace Ani{
    template<std::size_t N, typename FT>
    PhysMtx<N, FT>::PhysMtx(const FT* src, bool row_major) { 
        if (row_major)
            std::copy(src, src+N*N, m_dat.data());
        else    
            for (unsigned i = 0; i < N; ++i)
            for (unsigned j = 0; j < N; ++j)
                m_dat[N*i + j] = src[i + N*j];    
    }
    template<std::size_t N, typename FT>
    PhysMtx<N, FT>::PhysMtx(const SymMtx<N, FT>& r){ 
        for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j)
            m_dat[N*i + j] = m_dat[N*j + i] = r(i, j);
    }
    namespace internal{
        template <std::size_t N, typename MTX, typename FT>
        FT Trace(const MTX& m){
            FT v = FT(); 
            for (std::size_t i = 0; i < N; ++i) 
                v += m(i, i); 
            return v;
        }
        template <typename ARR, typename FT>
        FT SquareFrobNorm(const ARR& m){
            FT v = FT();
            for (std::size_t i = 0; i < m.continuous_size(); ++i) 
                v += m[i]*m[i]*m.index_duplication(i);
            return v;    
        }
        template <typename ARR, typename FT>
        FT Dot(const ARR& a, const ARR& b){
            FT v = FT();
            for (std::size_t i = 0; i < a.continuous_size(); ++i) 
                v += a[i]*b[i]*a.index_duplication(i);
            return v;    
        }
        template <std::size_t N, typename MTX, typename ARR, typename FT>
        ARR Mul(const MTX& m, const ARR& v){
            ARR r{};
            for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = 0; j < N; ++j)
                r[i] += m(i, j) * v[j];
            return r;    
        }

        template<typename T1, typename T2, typename U = void>
        struct Tensor_convert{
            static_assert("Not allowed type conversion");
        };
        template<typename T>
        struct Tensor_convert<T, T>{
            static T convert(const T& v){ return v; }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysMtx<N, FT>, SymMtx<N, FT>>{
            static SymMtx<N, FT> convert(const PhysMtx<N, FT>& v){
                SymMtx<N, FT> r;
                for (auto it = r.begin(); it != r.end(); ++it){
                    auto q = it.index();
                    *it = v(q.i, q.j);
                }
                return r;
            }
            static PhysMtx<N, FT> convert(const SymMtx<N, FT>& v){
                PhysMtx<N, FT> r;
                for (auto it = v.cbegin(); it != v.cend(); ++it){
                    auto q = it.index();
                    r(q.i, q.j) = r(q.j, q.i) = *it;
                }
                return r;
            }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<SymMtx<N, FT>, PhysMtx<N, FT>>: public Tensor_convert<PhysMtx<N, FT>, SymMtx<N, FT>> {};
        
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysMtx<N, FT>, PhysArr<N*N, FT>>{
            static PhysMtx<N, FT> convert(const PhysArr<N*N, FT>& v){ return PhysMtx<N, FT>(v); }
            static PhysArr<N*N, FT> convert(const PhysMtx<N, FT>& v){ return PhysArr<N*N, FT>(v.m_dat); }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysArr<N*N, FT>, PhysMtx<N, FT>>: public Tensor_convert<PhysMtx<N, FT>, PhysArr<N*N, FT>> {};

        template<std::size_t N, typename FT>
        struct Tensor_convert<SymMtx<N, FT>, PhysArr<N*N, FT>>{
            static SymMtx<N, FT> convert(const PhysArr<N*N, FT>& v){ 
                SymMtx<N, FT> r;
                for (auto it = r.begin(); it != r.end(); ++it){
                    auto q = it.index();
                    *it = v[q.j + q.i * N];
                }
                return r;
            }
            static PhysArr<N*N, FT> convert(const SymMtx<N, FT>& v){ 
                PhysArr<N*N, FT> r;
                for (auto it = v.cbegin(); it != v.cend(); ++it){
                    auto q = it.index();
                    r[q.i + N*q.j] = r[q.j + N*q.i] = *it;
                }
                return r; 
            }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysArr<N*N, FT>, SymMtx<N, FT>>: public Tensor_convert<SymMtx<N, FT>, PhysArr<N*N, FT>> {};

        template<std::size_t N, typename FT>
        struct Tensor_convert<SymMtx<N, FT>, PhysArr<N*(N+1)/2, FT>, typename std::enable_if<(N>1)>::type>{
            static SymMtx<N, FT> convert(const PhysArr<N*(N+1)/2, FT>& v){ return SymMtx<N, FT>(v); }
            static PhysArr<N*(N+1)/2, FT> convert(const SymMtx<N, FT>& v){ return PhysArr<N*(N+1)/2, FT>(v.m_dat); }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysArr<N*(N+1)/2, FT>, SymMtx<N, FT>, typename std::enable_if<(N>1)>::type>: public Tensor_convert<SymMtx<N, FT>, PhysArr<N*(N+1)/2, FT>> {};
        
        template<std::size_t N, typename FT>
        struct Tensor_convert<Tensor4Rank<N, FT>, SymTensor4Rank<N, FT>>{
            static Tensor4Rank<N, FT> convert(const SymTensor4Rank<N, FT>& v){
                Tensor4Rank<N, FT> r;
                for (auto it = v.cbegin(); it != v.cend(); ++it){
                    auto q = it.index();
                    r(q.i, q.j, q.k, q.l) = r(q.k, q.l, q.i, q.j) = *it; 
                }
                return r;
            }
            static SymTensor4Rank<N, FT> convert(const Tensor4Rank<N, FT>& r){
                SymTensor4Rank<N, FT> v;
                for (auto it = v.begin(); it != v.end(); ++it){
                    auto q = it.index();
                    *it = r(q.i, q.j, q.k, q.l);
                }
                return v;
            }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<SymTensor4Rank<N, FT>, Tensor4Rank<N, FT>>: public Tensor_convert<Tensor4Rank<N, FT>, SymTensor4Rank<N, FT>> {};
        template<std::size_t N, typename FT>
        struct Tensor_convert<Tensor4Rank<N, FT>, BiSymTensor4Rank<N, FT>>{
            static Tensor4Rank<N, FT> convert(const BiSymTensor4Rank<N, FT>& v){
                Tensor4Rank<N, FT> r;
                for (auto it = v.cbegin(); it != v.cend(); ++it){
                    auto q = it.index();
                    r(q.i, q.j, q.k, q.l) = r(q.k, q.l, q.i, q.j) = *it; 
                    r(q.j, q.i, q.k, q.l) = r(q.k, q.l, q.j, q.i) = *it;
                    r(q.i, q.j, q.l, q.k) = r(q.l, q.k, q.i, q.j) = *it; 
                    r(q.j, q.i, q.l, q.k) = r(q.l, q.k, q.j, q.i) = *it;
                }
                return r;
            }
            static BiSymTensor4Rank<N, FT> convert(const Tensor4Rank<N, FT>& r){
                BiSymTensor4Rank<N, FT> v;
                for (auto it = v.begin(); it != v.end(); ++it){
                    auto q = it.index();
                    *it = r(q.i, q.j, q.k, q.l);
                }
                return v;
            }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<BiSymTensor4Rank<N, FT>, Tensor4Rank<N, FT>>: public Tensor_convert<Tensor4Rank<N, FT>, BiSymTensor4Rank<N, FT>> {};
        template<std::size_t N, typename FT>
        struct Tensor_convert<SymTensor4Rank<N, FT>, BiSymTensor4Rank<N, FT>>{
            static SymTensor4Rank<N, FT> convert(const BiSymTensor4Rank<N, FT>& v){
                SymTensor4Rank<N, FT> r;
                for (auto it = v.cbegin(); it != v.cend(); ++it){
                    auto q = it.index();
                    r(q.i, q.j, q.k, q.l) = *it; 
                    r(q.j, q.i, q.k, q.l) = *it;
                    r(q.i, q.j, q.l, q.k) = *it; 
                    r(q.j, q.i, q.l, q.k) = *it;
                }
                return r;
            }
            static BiSymTensor4Rank<N, FT> convert(const SymTensor4Rank<N, FT>& r){
                BiSymTensor4Rank<N, FT> v;
                for (auto it = v.begin(); it != v.end(); ++it){
                    auto q = it.index();
                    *it = r(q.i, q.j, q.k, q.l);
                }
                return v;
            }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<BiSymTensor4Rank<N, FT>, SymTensor4Rank<N, FT>>: public Tensor_convert<SymTensor4Rank<N, FT>, BiSymTensor4Rank<N, FT>> {};

        template<std::size_t N, typename FT>
        struct Tensor_convert<Tensor4Rank<N, FT>, PhysArr<N*N*N*N, FT>, typename std::enable_if<(N>1)>::type>{
            static Tensor4Rank<N, FT> convert(const PhysArr<N*N*N*N, FT>& v){ return Tensor4Rank<N, FT>(v); }
            static PhysArr<N*N*N*N, FT> convert(const Tensor4Rank<N, FT>& v){ return PhysArr<N*N*N*N, FT>(v.m_dat); }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysArr<N*N*N*N, FT>, Tensor4Rank<N, FT>, typename std::enable_if<(N>1)>::type>: public Tensor_convert<Tensor4Rank<N, FT>, PhysArr<N*N*N*N, FT>> {};

        template<std::size_t N, typename FT>
        struct Tensor_convert<SymTensor4Rank<N, FT>, PhysArr<N*N*(N*N+1)/2, FT>, typename std::enable_if<(N>1)>::type>{
            static SymTensor4Rank<N, FT> convert(const PhysArr<N*N*(N*N+1)/2, FT>& v){ return SymTensor4Rank<N, FT>(v); }
            static PhysArr<N*N*(N*N+1)/2, FT> convert(const SymTensor4Rank<N, FT>& v){ return PhysArr<N*N*(N*N+1)/2, FT>(v.m_dat); }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysArr<N*N*(N*N+1)/2, FT>, SymTensor4Rank<N, FT>, typename std::enable_if<(N>1)>::type>: public Tensor_convert<SymTensor4Rank<N, FT>, PhysArr<N*N*(N*N+1)/2, FT>> {};

        template<std::size_t N, typename FT>
        struct Tensor_convert<BiSymTensor4Rank<N, FT>, PhysArr<N*(N+1)/2 * (N*(N+1)/2 + 1)/2, FT>, typename std::enable_if<(N>1)>::type>{
            static BiSymTensor4Rank<N, FT> convert(const PhysArr<N*(N+1)/2 * (N*(N+1)/2 + 1)/2, FT>& v){ return BiSymTensor4Rank<N, FT>(v); }
            static PhysArr<N*(N+1)/2 * (N*(N+1)/2 + 1)/2, FT> convert(const BiSymTensor4Rank<N, FT>& v){ return PhysArr<N*(N+1)/2 * (N*(N+1)/2 + 1)/2, FT>(v.m_dat); }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<PhysArr<N*(N+1)/2 * (N*(N+1)/2 + 1)/2, FT>, BiSymTensor4Rank<N, FT>, typename std::enable_if<(N>1)>::type>: public Tensor_convert<BiSymTensor4Rank<N, FT>, PhysArr<N*(N+1)/2 * (N*(N+1)/2 + 1)/2, FT>> {};

        template<std::size_t N, typename FT>
        struct Tensor_convert<BiSymTensor4Rank<N, FT>, SymMtx<N*(N+1)/2, FT>>{
            static BiSymTensor4Rank<N, FT> convert(const SymMtx<N*(N+1)/2, FT>& v){ return BiSymTensor4Rank<N, FT>(v.m_dat); }
            static SymMtx<N*(N+1)/2, FT> convert(const BiSymTensor4Rank<N, FT>& v){ return SymMtx<N*(N+1)/2, FT>(v.m_dat); }
        };
        template<std::size_t N, typename FT>
        struct Tensor_convert<SymMtx<N*(N+1)/2, FT>, BiSymTensor4Rank<N, FT>>: public Tensor_convert<BiSymTensor4Rank<N, FT>, SymMtx<N*(N+1)/2, FT>> {};
    }
    template<typename TYPE_TO, typename TYPE_FROM>
    TYPE_TO tensor_convert(const TYPE_FROM& v) { return internal::Tensor_convert<TYPE_FROM, TYPE_TO>::convert(v); }

    template<std::size_t N, typename FT>
    FT PhysArr<N, FT>::SquareFrobNorm() const { return internal::SquareFrobNorm<PhysArr<N, FT>, FT>(*this); }
    template<std::size_t N, typename FT>
    FT PhysMtx<N, FT>::SquareFrobNorm() const { return internal::SquareFrobNorm<PhysMtx<N, FT>, FT>(*this); }
    template<std::size_t N, typename FT>
    FT SymMtx<N, FT>::SquareFrobNorm() const { return internal::SquareFrobNorm<SymMtx<N, FT>, FT>(*this); }
    template<std::size_t N, typename FT>
    FT PhysMtx<N, FT>::Dot(const PhysMtx<N, FT>& b) const { return internal::Dot<PhysMtx<N, FT>, FT>(*this, b); }
    template<std::size_t N, typename FT>
    FT SymMtx<N, FT>::Dot(const SymMtx<N, FT>& b) const { return internal::Dot<SymMtx<N, FT>, FT>(*this, b); }

    template<std::size_t N, typename FT>
    FT PhysMtx<N, FT>::Trace() const { return internal::Trace<N, PhysMtx<N, FT>, FT>(*this); }
    template<std::size_t N, typename FT>
    FT SymMtx<N, FT>::Trace() const { return internal::Trace<N, SymMtx<N, FT>, FT>(*this); }
    template<std::size_t N, typename FT>
    std::array<FT, N> PhysMtx<N, FT>::Mul(const std::array<FT, N>& v) const { return internal::Mul<N, PhysMtx<N, FT>, std::array<FT, N>, FT>(*this, v); }
    template<std::size_t N, typename FT>
    PhysArr<N, FT> PhysMtx<N, FT>::Mul(const PhysArr<N, FT>& v) const { return internal::Mul<N, PhysMtx<N, FT>, PhysArr<N, FT>, FT>(*this, v); }
    template<std::size_t N, typename FT>
    std::array<FT, N> SymMtx<N, FT>::Mul(const std::array<FT, N>& v) const { return internal::Mul<N, SymMtx<N, FT>, std::array<FT, N>, FT>(*this, v); }
    template<std::size_t N, typename FT>
    PhysArr<N, FT> SymMtx<N, FT>::Mul(const PhysArr<N, FT>& v) const { return internal::Mul<N, SymMtx<N, FT>, PhysArr<N, FT>, FT>(*this, v); }
    
    template<std::size_t N, typename FT>
    FT PhysMtx<N, FT>::Dot(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s) const { 
        FT r{};
        for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            r += operator()(i, j)*f[i]*s[j];
        return r;    
    }
    template<std::size_t N, typename FT>
    FT SymMtx<N, FT>::Dot(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s) const { 
        FT r{};
        for (auto it = cbegin(); it != cend(); ++it){
            auto q = it.index();
            r += (*it) * ( q.i!=q.j ? (f[q.i]*s[q.j] + f[q.j]*s[q.i]) : f[q.i]*s[q.i] ); 
        }
        return r;    
    }
    template<std::size_t N, typename FT>
    PhysMtx<N, FT> PhysMtx<N, FT>::operator*(const PhysMtx<N, FT>& a){
        PhysMtx<N, FT> r{};
        for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
        for (std::size_t k = 0; k < N; ++k)
            r(i, j) += operator()(i, k) * a(k, j);
        return r;    
    }
    template<std::size_t N, typename FT>
    SymMtx<N, FT> SymMtx<N, FT>::operator*(const SymMtx<N, FT>& a){
        SymMtx<N, FT> r{};
        for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; j < N; ++j)
        for (std::size_t k = 0; k < N; ++k)
            r(i, j) += operator()(i, k) * a(j, k);
        return r;    
    }

    template<std::size_t N, typename FT>
    PhysMtx<N, FT> PhysMtx<N, FT>::Inv() const {
        PhysMtx<N, FT> r;
        std::array<FT, 2*N*N> wmem;
        std::array<int, 2*N> imem;
        fullPivLU_inverse(m_dat.data(), r.m_dat.data(), N, wmem.data(), imem.data());
        return r;
    }
    template<std::size_t N, typename FT>
    SymMtx<N, FT> SymMtx<N, FT>::Inv() const { return tensor_convert<SymMtx<N, FT>>(PhysMtx<N, FT>(*this).Inv()); }

    template<std::size_t N, typename FT>
    FT PhysMtx<N, FT>::Det() const {
        FT max_val = abs(*std::max_element(cbegin(), cend(), [](const auto& a, const auto& b){ return abs(a) < abs(b); }));
        const FT eps = 100 * max_val * std::numeric_limits<FT>::epsilon();

        PhysMtx<N, FT> A(*this);
        std::array<std::size_t, N> row_visited;
        const auto NOT_VISITED = std::numeric_limits<std::size_t>::max();
        std::fill(row_visited.begin(), row_visited.end(), NOT_VISITED);
        FT ret(1);
        for (std::size_t d = 0; d < N; ++d) {
            std::size_t r = 0;
            int sign = 1;
            // find unvisited row with non-zero value in column d
            while(row_visited[r] != NOT_VISITED || abs(A(r, d)) < eps){
                if (r == N-1) return FT();
                if (row_visited[r] == NOT_VISITED) sign *= -1;
                ++r;
            }
            row_visited[r] = d;
            ret *= sign * A(r, d);
            for (std::size_t i = 0; i < N; ++i) if(row_visited[i] == NOT_VISITED) {
                FT coef = A(i, d) / A(r, d);
                if(abs(coef) > eps)
                    for (std::size_t j = 0; j < N; ++j)
                        A(i, j) = A(i, j) - coef * A(r, j);
            }
        }
        return ret;
    }

    template<std::size_t N, typename FT>
    FT SymMtx<N, FT>::Det() const {
        FT max_val = abs(*std::max_element(cbegin(), cend(), [](const auto& a, const auto& b){ return abs(a) < abs(b); }));
        const FT eps = 100 * max_val * std::numeric_limits<FT>::epsilon();

        SymMtx<N, FT> A(*this);
        std::array<std::size_t, N> row_visited;
        const auto NOT_VISITED = std::numeric_limits<std::size_t>::max();
        std::fill(row_visited.begin(), row_visited.end(), NOT_VISITED);
        FT ret(1);
        for (std::size_t d = 0; d < N; ++d) {
            std::size_t r = 0;
            int sign = 1;
            // find unvisited row with non-zero value in column d
            while(row_visited[r] != NOT_VISITED || abs(A(r, d)) < eps){
                if (r == N-1) return FT();
                if (row_visited[r] == NOT_VISITED) sign *= -1;
                ++r;
            }
            row_visited[r] = d;
            ret *= sign * A(r, d);
            for (std::size_t i = 0; i < N; ++i) if(row_visited[i] == NOT_VISITED) {
                FT coef = A(i, d) / A(r, d);
                if(abs(coef) > eps)
                    for (std::size_t j = i; j < N; ++j)
                        A(i, j) = A(i, j) - coef * A(r, j);
            }
        }
        return ret;
    }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> PhysMtx<N, FT>::TensorSymMul2(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s){
        PhysMtx<N, FT> r;
        for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; i < N; ++i)
            r(i, j) = r(j, i) = f[i]*s[j] + f[j]*s[i];
        return r;
    }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> PhysMtx<N, FT>::TensorSquare(const PhysArr<N, FT>& s){
        PhysMtx<N, FT> r;
        for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i; i < N; ++i)
            r(i, j) = r(j, i) = s[i]*s[j];
        return r;
    }

    template<std::size_t N, typename FT>
    inline SymMtx<N, FT> SymMtx<N, FT>::TensorSymMul2(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s){
        SymMtx<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = f[q.i]*s[q.j] + f[q.j]*s[q.i];
        }
        return r;
    }
    template<std::size_t N, typename FT>
    inline SymMtx<N, FT> SymMtx<N, FT>::TensorSquare(const PhysArr<N, FT>& s){
        SymMtx<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = s[q.i]*s[q.j];
        }
        return r;
    }

    
    template<std::size_t N, typename FT>
    FT Tensor4Rank<N, FT>::SquareFrobNorm() const { 
        FT v = FT();
        for (std::size_t i = 0; i < continuous_size(); ++i) 
            v += m_dat[i] * m_dat[i];
        return v;         
    }
    template<std::size_t N, typename FT>
    FT Tensor4Rank<N, FT>::Dot(const Tensor4Rank<N, FT>& b) const { return internal::Dot<Tensor4Rank<N, FT>, FT>(*this, b); }
    template<std::size_t N, typename FT>
    Tensor4Rank<N, FT> Tensor4Rank<N, FT>::TensorSquare(const PhysMtx<N, FT>& s){
        Tensor4Rank<N, FT> r;
        for (std::size_t I = 0; I < N*N; ++I)
        for (std::size_t J = I; J < N*N; ++J)
            r[J + N*N*I] = r[I + N*N*J] = s[I]*s[J];
        return r;    
    }
    template<std::size_t N, typename FT>
    inline Tensor4Rank<N, FT> Tensor4Rank<N, FT>::TensorSymMul2(const PhysMtx<N, FT>& f, const PhysMtx<N, FT>& s){
        Tensor4Rank<N, FT> r;
        for (std::size_t I = 0; I < N*N; ++I)
        for (std::size_t J = I; J < N*N; ++J)
            r[J + N*N*I] = r[I + N*N*J] =  f[I]*s[J] + s[I]*f[J];
        return r;
    }
    template<std::size_t N, typename FT>
    inline Tensor4Rank<N, FT> Tensor4Rank<N, FT>::TensorSquare(const SymMtx<N, FT>& s){
        Tensor4Rank<N, FT> r;
        for (std::size_t I = 0; I < N*N; ++I)
        for (std::size_t J = I; J < N*N; ++J)
            r[J + N*N*I] = r[I + N*N*J] = s(I/N, I%N)*s(J/N, J%N);
        return r;    
    }
    template<std::size_t N, typename FT>
    inline Tensor4Rank<N, FT> Tensor4Rank<N, FT>::TensorSymMul2(const SymMtx<N, FT>& f, const SymMtx<N, FT>& s){
        Tensor4Rank<N, FT> r;
        for (std::size_t I = 0; I < N*N; ++I)
        for (std::size_t J = I; J < N*N; ++J)
            r[J + N*N*I] = r[I + N*N*J] = f(I/N, I%N)*s(J/N, J%N) + s(I/N, I%N)*f(J/N, J%N);
        return r;    
    }

    template<std::size_t N, typename FT>
    FT SymTensor4Rank<N, FT>::SquareFrobNorm() const { 
        FT v = FT();
        static const typename SymMtx<N*N, FT>::IndexMap a{};
        for (std::size_t i = 0; i < continuous_size(); ++i)
            v += (a.imap[i] != a.jmap[i] ? 2 : 1)*m_dat[i]*m_dat[i];
            
        return v; 
    }
    template<std::size_t N, typename FT>
    FT SymTensor4Rank<N, FT>::Dot(const SymTensor4Rank<N, FT>& b) const { return internal::Dot<SymTensor4Rank<N, FT>, FT>(*this, b); }
    template<std::size_t N, typename FT>
    SymTensor4Rank<N, FT> SymTensor4Rank<N, FT>::TensorSquare(const PhysMtx<N, FT>& s){
        SymTensor4Rank<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = s(q.i, q.j) * s(q.k, q.l);
        }
        return r;    
    }
    template<std::size_t N, typename FT>
    inline SymTensor4Rank<N, FT> SymTensor4Rank<N, FT>::TensorSymMul2(const PhysMtx<N, FT>& f, const PhysMtx<N, FT>& s){
        SymTensor4Rank<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = f(q.i, q.j) * s(q.k, q.l) + s(q.i, q.j) * f(q.k, q.l);
        }
        return r;
    }
    template<std::size_t N, typename FT>
    SymTensor4Rank<N, FT> SymTensor4Rank<N, FT>::TensorSquare(const SymMtx<N, FT>& s){
        SymTensor4Rank<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = s(q.i, q.j) * s(q.k, q.l);
        }
        return r;    
    }
    template<std::size_t N, typename FT>
    inline SymTensor4Rank<N, FT> SymTensor4Rank<N, FT>::TensorSymMul2(const SymMtx<N, FT>& f, const SymMtx<N, FT>& s){
        SymTensor4Rank<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = f(q.i, q.j) * s(q.k, q.l) + s(q.i, q.j) * f(q.k, q.l);
        }
        return r;
    }

    template<std::size_t N, typename FT>
    FT BiSymTensor4Rank<N, FT>::SquareFrobNorm() const { return internal::SquareFrobNorm<BiSymTensor4Rank<N, FT>, FT>(*this); }
    template<std::size_t N, typename FT>
    FT BiSymTensor4Rank<N, FT>::Dot(const BiSymTensor4Rank<N, FT>& b) const { return internal::Dot<BiSymTensor4Rank<N, FT>, FT>(*this, b); }
    template<std::size_t N, typename FT>
    BiSymTensor4Rank<N, FT> BiSymTensor4Rank<N, FT>::TensorSquare(const SymMtx<N, FT>& s){
        BiSymTensor4Rank<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = s(q.i, q.j) * s(q.k, q.l);
        }
        return r;  
    }
    template<std::size_t N, typename FT>
    BiSymTensor4Rank<N, FT> BiSymTensor4Rank<N, FT>::TensorSymMul2(const SymMtx<N, FT>& f, const SymMtx<N, FT>& s){
        BiSymTensor4Rank<N, FT> r;
        for (auto it = r.begin(); it != r.end(); ++it){
            auto q = it.index();
            *it = f(q.i, q.j) * s(q.k, q.l) + s(q.i, q.j) * f(q.k, q.l);
        }
        return r;
    }

    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> operator*(const PhysMtx<N, FT>& a, const SymMtx<N, FT>& b){
        PhysMtx<N, FT> r;
        for(std::size_t i = 0; i < N; ++i)
        for(std::size_t j = 0; j < N; ++j)
        for(std::size_t k = 0; k < N; ++k)
            r(i, j) += a(i,k)*b(k,j);
        return r;
    }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> operator*(const SymMtx<N, FT>& a, const PhysMtx<N, FT>& b){
        PhysMtx<N, FT> r;
        for(std::size_t i = 0; i < N; ++i)
        for(std::size_t j = 0; j < N; ++j)
        for(std::size_t k = 0; k < N; ++k)
            r(i, j) += a(i,k)*b(k,j);
        return r;
    }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> operator+(const PhysMtx<N, FT>& a, const SymMtx<N, FT>& b){
        PhysMtx<N, FT> r;
        for (unsigned q = 0; q < b.continuous_size(); ++q){
            auto i = SymMtx<N, FT>::index(q).i, j = SymMtx<N, FT>::index(q).j;
            r(i, j) = a(i, j) + b[q];
            if (j != i) r(j, i) = a(j, i) + b[q];
        }
        return r;
    }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> operator+(const SymMtx<N, FT>& a, const PhysMtx<N, FT>& b){ return b + a; }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> operator-(const PhysMtx<N, FT>& a, const SymMtx<N, FT>& b){
        PhysMtx<N, FT> r;
        for (unsigned q = 0; q < b.continuous_size(); ++q){
            auto i = SymMtx<N, FT>::index(q).i, j = SymMtx<N, FT>::index(q).j;
            r(i, j) = a(i, j) - b[q];
            if (j != i) r(j, i) = a(j, i) - b[q];
        }
        return r;
    }
    template<std::size_t N, typename FT>
    inline PhysMtx<N, FT> operator-(const SymMtx<N, FT>& a, const PhysMtx<N, FT>& b){
        PhysMtx<N, FT> r;
        for (unsigned q = 0; q < a.continuous_size(); ++q){
            auto i = SymMtx<N, FT>::index(q).i, j = SymMtx<N, FT>::index(q).j;
            r(i, j) = a[q] - b(i, j);
            if (j != i) r(j, i) = a[q] - b(j, i);
        }
        return r;
    }

    namespace internal{
        template<std::size_t N, typename MTX>
        inline ::std::ostream& print_mtx(::std::ostream& out, const MTX& m, const ::std::string& val_sep = " ", const ::std::string& row_sep = "\n"){
            //out << ::std::setprecision(::std::numeric_limits<double>::digits10) << ::std::scientific;
            auto sign_shift = [](auto x) { return (x >= 0) ? " " : ""; };
            for (::std::size_t i = 0; i < N; ++i){
                for (::std::size_t j = 0; j < N-1; ++j){
                    const auto& val = m(i, j);
                    out << sign_shift(val) << val << val_sep;
                }
                const auto& val = m(i, N-1);
                out << sign_shift(val) << val << row_sep;
            }
            return out;
        }
        template<std::size_t N, typename TENSOR4>
        inline ::std::ostream& print_4tensor(::std::ostream& out, const TENSOR4& m, const ::std::string& val_sep = " ", const ::std::string& row_sep = "\n"){
            //out << ::std::setprecision(::std::numeric_limits<float>::digits10) << ::std::scientific;
            auto sign_shift = [](auto x) { return (x >= 0) ? " " : ""; };
            for (::std::size_t i = 0; i < N*N; ++i){
                if (i != 0 && i % N == 0) out << "\n";
                for (::std::size_t j = 0; j < N*N-1; ++j){
                    if (j != 0 && j % N == 0) out << " ";
                    const auto& val = m(i%N, i/N, j%N, j/N);
                    out << sign_shift(val) << val << val_sep;
                }
                const auto& val = m(i%N, i/N, N-1, N-1);
                out << sign_shift(val) << val << row_sep;
            }
            return out;
        }
    }
    template<std::size_t N, typename FT>
    inline std::ostream& operator<<(std::ostream& out, const PhysMtx<N, FT>& m) { return internal::print_mtx<N>(out, m); }
    template<std::size_t N, typename FT>
    inline std::ostream& operator<<(std::ostream& out, const SymMtx<N, FT>& m) { return internal::print_mtx<N>(out, m); }
    template<std::size_t N, typename FT>
    inline std::ostream& operator<<(std::ostream& out, const Tensor4Rank<N, FT>& m) { return internal::print_4tensor<N>(out, m); }
    template<std::size_t N, typename FT>
    inline std::ostream& operator<<(std::ostream& out, const SymTensor4Rank<N, FT>& m) { return internal::print_4tensor<N>(out, m); }
    template<std::size_t N, typename FT>
    inline std::ostream& operator<<(std::ostream& out, const BiSymTensor4Rank<N, FT>& m) { return internal::print_4tensor<N>(out, m); }

}

#endif //CARNUM_PHYSICAL_TENSORS_INL