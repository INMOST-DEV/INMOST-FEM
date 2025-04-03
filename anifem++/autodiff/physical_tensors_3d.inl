//
// Created by Liogky Alexey on 10.01.2024.
//

#ifndef CARNUM_PHYSICAL_TENSORS_3D_INL
#define CARNUM_PHYSICAL_TENSORS_3D_INL

#include "physical_tensors.h"

namespace Ani{

/// Structure to store symmetrical 3x3 matrices. 
/// Storage numbering is follow:
///  a0 a1 a2
///  a1 a3 a4
///  a2 a4 a5
template<typename FT>
struct SymMtx<3, FT>{
    union Index{
        struct {std::size_t i, j; };
        std::size_t id[2];
        Index(): i{0}, j{0} {}
        Index(std::size_t _i, std::size_t _j): i{_i}, j{_j} { if (i > j) std::swap(i, j); }
        bool operator==(Index a) const { return i == a.i && j == a.j; }
        std::size_t operator[](std::size_t n) const { return id[n]; }
    };
    
    static Index index(unsigned char continuous_index){
        static const std::array<Index, 6> ids = { 
            Index{0, 0}, Index{0, 1}, Index{0, 2}, 
                         Index{1, 1}, Index{1, 2}, 
                                      Index{2, 2} 
        };
        return ids[continuous_index];
    }
    static std::size_t continuous_index(Index id){ return id.i+id.j+(id.i != 0 ? 1 : 0); }

    std::array<FT, 6> m_dat = {FT(0)};
    constexpr std::size_t rank() const { return 2U; }
    constexpr std::size_t continuous_size() const { return 6; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { 
        constexpr std::array<std::size_t, 6> dup{1, 2, 2, 1, 2, 1};
        return dup[continuous_index];
    }

    SymMtx() = default;
    explicit SymMtx(std::array<FT, 3*(3+1)/2> arr): m_dat{std::move(arr)} {}

    template <typename DataType = FT>
    struct mtx_iterator{
    protected:   
        DataType* val;
        std::size_t contiguous_index;
        mtx_iterator() = default;
        mtx_iterator(DataType* dat, std::size_t id): val{dat}, contiguous_index{id} {} 
    public:
        typedef std::random_access_iterator_tag  iterator_category;
        typedef DataType value_type;
        typedef long difference_type;
        typedef DataType* pointer;
        typedef DataType& reference;
        
        mtx_iterator& operator ++() { ++contiguous_index; return *this; }
        mtx_iterator  operator ++(int) {mtx_iterator ret(*this); operator++(); return ret;}
        mtx_iterator& operator --() { --contiguous_index; return *this; }
        mtx_iterator  operator --(int) {mtx_iterator ret(*this); operator--(); return ret;}
        bool operator ==(const mtx_iterator & b) const { return contiguous_index == b.contiguous_index; }
        bool operator !=(const  mtx_iterator & other) const { return !operator==(other); }
        reference operator*() const { return val[contiguous_index]; }
        pointer operator ->() const { return val + contiguous_index; }
        reference operator[](difference_type n) const { return val[static_cast<char>(contiguous_index) + n]; }
        difference_type operator-(const mtx_iterator & other) const { return static_cast<char>(contiguous_index) - static_cast<char>(other.contiguous_index); }
        mtx_iterator& operator+=(difference_type n) { contiguous_index += n; return *this; } 
        mtx_iterator& operator-=(difference_type n) { contiguous_index -= n; return *this; }
        mtx_iterator  operator+ (difference_type n) const { mtx_iterator other = *this; other += n; return other; }
        mtx_iterator  operator- (difference_type n) const { mtx_iterator other = *this; other -= n; return other; }
        friend mtx_iterator  operator+(difference_type n, const mtx_iterator& a) { return a+n; }
        bool operator< (const  mtx_iterator & other) const { return contiguous_index <  other.contiguous_index; }
        bool operator> (const  mtx_iterator & other) const { return contiguous_index >  other.contiguous_index; }
        bool operator<=(const  mtx_iterator & other) const { return contiguous_index <= other.contiguous_index; }
        bool operator>=(const  mtx_iterator & other) const { return contiguous_index >= other.contiguous_index; }

        std::size_t continuous_index() const { return contiguous_index; }
        Index index() const { return SymMtx<3, FT>::index(contiguous_index); }

        friend class SymMtx<3, FT>;    
    };
    using iterator = mtx_iterator<FT>;
    using const_iterator = mtx_iterator<const FT>;

    iterator begin()              { return iterator(m_dat.data(), 0);       }
    iterator end()                { return iterator(m_dat.data(), continuous_size());       }
    const_iterator cbegin() const { return const_iterator(m_dat.data(), 0); }
    const_iterator cend()   const { return const_iterator(m_dat.data(), continuous_size()); }
    const_iterator begin()  const { return cbegin();                        }
    const_iterator end()    const { return cend();                          } 

    inline FT& operator[](Index i){ return m_dat[continuous_index(i)]; }
    inline FT operator[](Index i) const { return m_dat[continuous_index(i)]; }
    inline FT& operator[](std::size_t i){ return m_dat[i]; }
    inline FT operator[](std::size_t i) const { return m_dat[i]; } 
    
    inline FT& operator()(std::size_t i, std::size_t j) { return m_dat[i+j+(std::min(i, j)!=0 ? 1 : 0)]; }
    inline FT operator()(std::size_t i, std::size_t j) const { return m_dat[i+j+(std::min(i, j)!=0 ? 1 : 0)]; }
    inline FT& operator()(std::array<std::size_t, 1> i, std::array<std::size_t, 1> j) { return operator()(i[0], j[0]); }
    inline FT operator()(std::array<std::size_t, 1> i, std::array<std::size_t, 1> j) const { return operator()(i[0], j[0]); }
    inline FT& operator()(std::array<std::size_t, 2> i) { return operator()(i[0], i[1]); }
    inline FT operator()(std::array<std::size_t, 2> i) const { return operator()(i[0], i[1]); }

    inline FT Det() const; 
    inline FT Trace() const;
    inline SymMtx<3, FT> Adj() const;
    inline FT SquareFrobNorm() const;
    inline SymMtx<3, FT> Inv() const;
    inline std::array<FT, 3> Mul(const std::array<FT, 3>& v) const;
    inline PhysArr<3, FT> Mul(const PhysArr<3, FT>& v) const;
    inline FT Dot(const SymMtx<3, FT>& b) const;
    inline FT Dot(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s) const;
    static SymMtx<3, FT> Identity(FT val = FT(1)) { SymMtx<3, FT> r; r[0] = r[3] = r[5] = val; return r; }
    /// @brief Matrix product
    /// @return m = this * a
    inline SymMtx<3, FT> operator*(const SymMtx<3, FT>& a);
    SymMtx<3, FT> Transpose() const { return *this; }
    
    inline SymMtx<3, FT>& operator+=(const SymMtx<3, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline SymMtx<3, FT>& operator-=(const SymMtx<3, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline SymMtx<3, FT> operator+(const SymMtx<3, FT>& a)const { SymMtx<3, FT> r(*this); return (r += a);}
    inline SymMtx<3, FT> operator-(const SymMtx<3, FT>& a) const { SymMtx<3, FT> r(*this); return (r -= a);}
    inline SymMtx<3, FT> operator-() const { SymMtx<3, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline SymMtx<3, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline SymMtx<3, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline SymMtx<3, FT> operator*(FT a) const { SymMtx<3, FT> r(*this); return (r *= a);}
    friend SymMtx<3, FT> operator*(FT c, const SymMtx<3, FT>& a){ return a.operator*(c); }
    inline SymMtx<3, FT> operator/(FT a) const { SymMtx<3, FT> r(*this); return (r /= a); }

    /// @return a x a
    static inline SymMtx<3, FT> TensorSquare(const PhysArr<3, FT>& s);
    /// @return f x s + s x f
    static inline SymMtx<3, FT> TensorSymMul2(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s);
};

template<typename FT>
inline FT SymMtx<3, FT>::Det() const { 
    auto& a = m_dat; 
    return a[0]*a[3]*a[5] + 2 * a[1]*a[2]*a[4] - (a[1]*a[1]*a[5] + a[2]*a[2]*a[3] + a[4]*a[4]*a[0]); 
} 
template<typename FT>
inline FT SymMtx<3, FT>::Trace() const {
    auto& a = m_dat; 
    return a[0]+a[3]+a[5]; 
}
template<typename FT>
inline FT SymMtx<3, FT>::SquareFrobNorm() const { 
    auto& a = *this; 
    return a[0]*a[0]+a[3]*a[3]+a[5]*a[5] + 2 * (a[1]*a[1] + a[2]*a[2] + a[4]*a[4]); 
}
template<typename FT>
inline SymMtx<3, FT> SymMtx<3, FT>::Adj() const {
    auto& m = *this;
    SymMtx<3, FT> r;
    r(0, 0) = m(1, 1) * m(2, 2) - m(1, 2) * m(1, 2);
    r(0, 1) = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
    r(0, 2) = m(1, 0) * m(1, 2) - m(1, 1) * m(2, 0);
    r(1, 1) = m(2, 2) * m(0, 0) - m(2, 0) * m(2, 0);
    r(1, 2) = m(2, 0) * m(1, 0) - m(1, 2) * m(0, 0);
    r(2, 2) = m(0, 0) * m(1, 1) - m(0, 1) * m(0, 1);
    return r;
}
template<typename FT>
inline SymMtx<3, FT> SymMtx<3, FT>::Inv() const { 
    auto& m = *this;
    auto r = Adj();
    FT det = m(0, 0) * r(0, 0) + m(0, 1) * r(0, 1) + m(0, 2) * r(0, 2);
    for (std::size_t i = 0; i < m_dat.size(); ++i) r[i] /= det;
    return r;
}
template<typename FT>
inline std::array<FT, 3> SymMtx<3, FT>::Mul(const std::array<FT, 3>& v) const{
    auto& m = *this;
    return {m(0, 0)*v[0] + m(0, 1)*v[1] + m(0, 2)*v[2],
            m(1, 0)*v[0] + m(1, 1)*v[1] + m(1, 2)*v[2],
            m(2, 0)*v[0] + m(2, 1)*v[1] + m(2, 2)*v[2] };         
}
template<typename FT>
inline PhysArr<3, FT> SymMtx<3, FT>::Mul(const PhysArr<3, FT>& v) const{
    auto& m = *this;
    return PhysArr<3, FT>(std::array<FT, 3>{
            m(0, 0)*v[0] + m(0, 1)*v[1] + m(0, 2)*v[2],
            m(1, 0)*v[0] + m(1, 1)*v[1] + m(1, 2)*v[2],
            m(2, 0)*v[0] + m(2, 1)*v[1] + m(2, 2)*v[2] });
}
template<typename FT>
inline FT SymMtx<3, FT>::Dot(const SymMtx<3, FT>& b) const{ 
    auto& a = *this; 
    return a[0]*b[0]+a[3]*b[3]+a[5]*b[5] + 2 * (a[1]*b[1] + a[2]*b[2] + a[4]*b[4]); 
}
template<typename FT>
inline FT SymMtx<3, FT>::Dot(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s) const { 
    auto& a = *this; 
    return a[0]*f[0]*s[0] + a[3]*f[1]*s[1] + a[5]*f[2]*s[2] 
            + a[1]*(f[1]*s[0] + f[0]*s[1]) + a[2]*(f[2]*s[0] + f[0]*s[2]) + a[4]*(f[2]*s[1] + f[1]*s[2]);
}
template<typename FT>
inline SymMtx<3, FT> SymMtx<3, FT>::operator*(const SymMtx<3, FT>& a){
    auto& b = *this;
    SymMtx<3, FT> r;
    r(0,0) = b(0,0)*a(0,0) + b(0,1)*a(1,0) + b(0,2)*a(2,0);
    r(1,0) = b(1,0)*a(0,0) + b(1,1)*a(1,0) + b(1,2)*a(2,0);
    r(1,1) = b(1,0)*a(0,1) + b(1,1)*a(1,1) + b(1,2)*a(2,1);
    r(2,0) = b(2,0)*a(0,0) + b(2,1)*a(1,0) + b(2,2)*a(2,0);
    r(2,1) = b(2,0)*a(0,1) + b(2,1)*a(1,1) + b(2,2)*a(2,1);
    r(2,2) = b(2,0)*a(0,2) + b(2,1)*a(1,2) + b(2,2)*a(2,2);

    return r;            
}
template<typename FT>
inline SymMtx<3, FT> SymMtx<3, FT>::TensorSymMul2(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s){
    SymMtx<3, FT> r;
    r[0] = 2*f[0]*s[0]; r[1] = f[1]*s[0] + f[0]*s[1];   r[2] = f[2]*s[0] + f[0]*s[2];
                        r[3] = 2*f[1]*s[1];             r[4] = f[1]*s[2] + f[2]*s[1];
                                                        r[5] = 2*f[2]*s[2];
    return r;
}
template<typename FT>
inline SymMtx<3, FT> SymMtx<3, FT>::TensorSquare(const PhysArr<3, FT>& s){
    SymMtx<3, FT> r;
    for (auto it = r.begin(); it != r.end(); ++it){
        auto q = it.index();
        *it = s[q.i]*s[q.j];
    }
    return r;
}

/// Structure to store general 3x3 tensors. 
/// Data stored in row major order
///  a0 a1 a2
///  a3 a4 a5
///  a6 a7 a8
template<typename FT>
struct PhysMtx<3, FT>{
    std::array<FT, 3*3> m_dat = {FT(0)};

    constexpr std::size_t rank() const { return 2U; }
    constexpr std::size_t continuous_size() const { return 3*3; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { (void) continuous_index; return 1; }

    PhysMtx() = default;
    explicit PhysMtx(const SymMtx<3, FT>& r);
    PhysMtx(const FT* src, bool row_major = true);
    PhysMtx(const std::array<FT, 3*3>& src, bool row_major = true): PhysMtx(src.data(), row_major) {}
    inline FT& operator[](std::size_t i) { return m_dat[i]; }
    inline FT operator[](std::size_t i) const { return m_dat[i]; }
    inline FT& operator()(std::size_t i, std::size_t j) { return m_dat[3*i + j]; }
    inline FT operator()(std::size_t i, std::size_t j) const { return m_dat[3*i + j]; }
    
    inline FT& operator()(std::array<std::size_t, 1> i, std::array<std::size_t, 1> j) { return operator()(i[0], j[0]); }
    inline FT operator()(std::array<std::size_t, 1> i, std::array<std::size_t, 1> j) const { return operator()(i[0], j[0]); }
    inline FT& operator()(std::array<std::size_t, 2> i) { return operator()(i[0], i[1]); }
    inline FT operator()(std::array<std::size_t, 2> i) const { return operator()(i[0], i[1]); }

    auto begin() { return m_dat.begin(); }
    auto end() { return m_dat.end(); }
    auto cbegin() const { return m_dat.cbegin(); }
    auto cend() const { return m_dat.cend(); }

    inline FT Trace() const;
    inline FT SquareFrobNorm() const;
    inline PhysMtx<3, FT> Adj() const;
    inline PhysMtx<3, FT> Inv() const;
    inline FT Det() const;
    inline std::array<FT, 3> Mul(const std::array<FT, 3>& v) const;
    inline PhysArr<3, FT> Mul(const PhysArr<3, FT>& v) const;
    inline FT Dot(const PhysMtx<3, FT>& b) const;
    inline FT Dot(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s) const;
    inline PhysMtx<3, FT> operator*(const PhysMtx<3, FT>& a);
    static PhysMtx<3, FT> Identity(FT val = FT(1)) { PhysMtx<3, FT> r; for (std::size_t i = 0; i < 3; ++i) r(i, i) = val; return r;}
    PhysMtx<3, FT> Transpose() const { return PhysMtx<3, FT>(m_dat.data(), false); }
    
    inline PhysMtx<3, FT>& operator+=(const PhysMtx<3, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline PhysMtx<3, FT>& operator-=(const PhysMtx<3, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline PhysMtx<3, FT> operator+(const PhysMtx<3, FT>& a) const { PhysMtx<3, FT> r(*this); return (r += a);}
    inline PhysMtx<3, FT> operator-(const PhysMtx<3, FT>& a) const { PhysMtx<3, FT> r(*this); return (r -= a);}
    inline PhysMtx<3, FT> operator-() const { PhysMtx<3, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline PhysMtx<3, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline PhysMtx<3, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline PhysMtx<3, FT> operator*(FT a) const { PhysMtx<3, FT> r(*this); return (r *= a);}
    friend PhysMtx<3, FT> operator*(FT c, const PhysMtx<3, FT>& a){ return a.operator*(c); }
    inline PhysMtx<3, FT> operator/(FT a) const { PhysMtx<3, FT> r(*this); return (r /= a); }

    inline SymMtx<3, FT> Sym() const;

    /// @return a x a
    static inline PhysMtx<3, FT> TensorSquare(const PhysArr<3, FT>& s);
    /// @return f x s + s x f
    static inline PhysMtx<3, FT> TensorSymMul2(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s);
};


template<typename FT>
PhysMtx<3, FT>::PhysMtx(const FT* src, bool row_major) { 
    if (row_major)
        std::copy(src, src+3*3, m_dat.data());
    else    
        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            m_dat[3*i + j] = src[i + 3*j];    
}
template<typename FT>
PhysMtx<3, FT>::PhysMtx(const SymMtx<3, FT>& r){ 
    for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = i; j < 3; ++j)
        m_dat[3*i + j] = m_dat[3*j + i] = r(i, j);
}
template<typename FT>
inline SymMtx<3, FT> PhysMtx<3, FT>::Sym() const {
    SymMtx<3, FT> A;
        for (auto it = A.begin(); it != A.end(); ++it){
            auto q = it.index();
            *it = (operator()(q.i, q.j) + operator()(q.j, q.i)) / 2;
        }
    return A;
}
template<typename FT>
FT PhysMtx<3, FT>::Trace() const { return operator()(0,0) + operator()(1, 1) + operator()(2,2); }
template<typename FT>
inline FT PhysMtx<3, FT>::Det() const { 
    const auto& m = *this;
    auto da = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    auto db = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
    auto dc = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    return m(0, 0) * da + m(0, 1) * db + m(0, 2) * dc;
} 
template<typename FT>
inline FT PhysMtx<3, FT>::SquareFrobNorm() const { 
    FT r = 0;
    for (unsigned i = 0; i < continuous_size(); ++i)
        r += m_dat[i]*m_dat[i];
    return r;    
}
template<typename FT>
inline PhysMtx<3, FT> PhysMtx<3, FT>::Adj() const {
    const auto& m = *this;
    PhysMtx<3, FT> adj;
    for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
        adj(j, i) = (m((i + 1) % 3, (j + 1) % 3) * m((i + 2) % 3, (j + 2) % 3) -
                        m((i + 1) % 3, (j + 2) % 3) * m((i + 2) % 3, (j + 1) % 3));
    return adj;
}
template<typename FT>
inline PhysMtx<3, FT> PhysMtx<3, FT>::Inv() const {
    const auto& m = *this;
    auto da = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    auto db = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
    auto dc = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
    auto det = m(0, 0) * da + m(0, 1) * db + m(0, 2) * dc;
    PhysMtx<3, FT> inv;
    inv(0, 0) = da / det; inv(1, 0) = db / det; inv(2, 0) = dc / det;
    for(int i = 1; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
        inv(j, i) = (m((i + 1) % 3, (j + 1) % 3) * m((i + 2) % 3, (j + 2) % 3) -
                        m((i + 1) % 3, (j + 2) % 3) * m((i + 2) % 3, (j + 1) % 3)) / det;
    return inv;                 
}
template<typename FT>
inline std::array<FT, 3> PhysMtx<3, FT>::Mul(const std::array<FT, 3>& v) const{
    const auto& m = *this;
    std::array<FT, 3> r;
    for(int i = 0; i < 3; ++i)
        r[i] = m(i, 0) * v[0] + m(i, 1) * v[1] + m(i, 2) * v[2];
    return r;    
}
template<typename FT>
inline PhysArr<3, FT> PhysMtx<3, FT>::Mul(const PhysArr<3, FT>& v) const {
    const auto& m = *this;
    PhysArr<3, FT> r;
    for(int i = 0; i < 3; ++i)
        r[i] = m(i, 0) * v[0] + m(i, 1) * v[1] + m(i, 2) * v[2];
    return r;    
}
template<typename FT>
inline FT PhysMtx<3, FT>::Dot(const PhysMtx<3, FT>& b) const {
    double r = 0;
    for(unsigned i = 0; i < 9; ++i)
        r += m_dat[i] * b.m_dat[i];
    return r;    
}
template<typename FT>
inline FT PhysMtx<3, FT>::Dot(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s) const {
    double r = 0;
    for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
        r += f[i]*s[j] * operator()(i,j);
    return r;    
}
template<typename FT>
inline PhysMtx<3, FT> PhysMtx<3, FT>::operator*(const PhysMtx<3, FT>& a){
    const auto& m = *this;
    PhysMtx<3, FT> r;
    for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 3; ++j)
        r(i, j) = m(i,0)*a(0,j) + m(i,1)*a(1,j) + m(i,2)*a(2,j);
    return r;         
}
template<typename FT>
inline PhysMtx<3, FT> PhysMtx<3, FT>::TensorSquare(const PhysArr<3, FT>& s){
    PhysMtx<3, FT> r;
    for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = i; j < 3; ++j)
        r(i, j) = r(j, i) = s[i]*s[j];
    return r;
}
template<typename FT>
inline PhysMtx<3, FT> PhysMtx<3, FT>::TensorSymMul2(const PhysArr<3, FT>& f, const PhysArr<3, FT>& s){
    PhysMtx<3, FT> r;
    for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = i; j < 3; ++j)
        r(i, j) = r(j, i) = f[i]*s[j] + f[j]*s[i];
    return r;
}

/// Structure to store bisymmetrical 3x3x3x3 tensors, i.e. such tensors that R_ijkl = R_klij = R_jikl
template<typename FT>
struct BiSymTensor4Rank<3, FT>{
    union Index{
        struct {std::size_t i, j, k, l; };
        std::size_t id[4];
        Index(): i{0}, j{0}, k{0}, l{0} {}
        Index(std::size_t _i, std::size_t _j, std::size_t _k, std::size_t _l): i{_i}, j{_j}, k{_k}, l{_l} 
        { 
            if (i > j) std::swap(i, j);
            if (k > l) std::swap(k, l);
            if (i > k || (i == k && j > l)) std::swap(i, k), std::swap(j, l);
        }
        bool operator==(Index a) const { return i == a.i && j == a.j && k == a.k && l == a.l; }
        std::size_t operator[](std::size_t n) const { return id[n]; }
    };
    static Index index(unsigned char continuous_index){
        static const Index ids[] = {
            Index{0, 0, 0, 0}, Index{0, 0, 0, 1}, Index{0, 0, 0, 2}, Index{0, 0, 1, 1}, Index{0, 0, 1, 2}, Index{0, 0, 2, 2},
                               Index{0, 1, 0, 1}, Index{0, 1, 0, 2}, Index{0, 1, 1, 1}, Index{0, 1, 1, 2}, Index{0, 1, 2, 2},
                                                  Index{0, 2, 0, 2}, Index{0, 2, 1, 1}, Index{0, 2, 1, 2}, Index{0, 2, 2, 2},
                                                                     Index{1, 1, 1, 1}, Index{1, 1, 1, 2}, Index{1, 1, 2, 2},
                                                                                        Index{1, 2, 1, 2}, Index{1, 2, 2, 2},
                                                                                                           Index{2, 2, 2, 2}
            };
        return ids[continuous_index];
    }
    static std::size_t continuous_index(Index id){ 
        unsigned char I = id.i + id.j + (id.i!=0 ? 1 : 0), J = id.k + id.l + (id.k!=0 ? 1 : 0); return (2*6-1-I)*I/2 + J;
    }

    std::array<FT, 21> m_dat = {0};

    BiSymTensor4Rank() = default;
    BiSymTensor4Rank(std::array<FT, 21> dat): m_dat(dat) {}

    constexpr std::size_t rank() const { return 4U; }
    constexpr std::size_t continuous_size() const { return 21; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { 
        constexpr const unsigned char dup[] = {
            1, 4, 4, 2, 4, 2,
               4, 8, 4, 8, 4,
                  4, 4, 8, 4,
                     1, 4, 2,
                        4, 4,
                           1
        };
        return dup[continuous_index];
    }

    template <typename DataType = FT>
    struct mtx_iterator{
    protected:   
        DataType* val;
        std::size_t contiguous_index;
        mtx_iterator() = default;
        mtx_iterator(DataType* dat, std::size_t id): val{dat}, contiguous_index{id} {} 
    public:
        typedef std::random_access_iterator_tag  iterator_category;
        typedef DataType value_type;
        typedef long difference_type;
        typedef DataType* pointer;
        typedef DataType& reference;
        
        mtx_iterator& operator ++() { ++contiguous_index; return *this; }
        mtx_iterator  operator ++(int) {mtx_iterator ret(*this); operator++(); return ret;}
        mtx_iterator& operator --() { --contiguous_index; return *this; }
        mtx_iterator  operator --(int) {mtx_iterator ret(*this); operator--(); return ret;}
        bool operator ==(const mtx_iterator & b) const { return contiguous_index == b.contiguous_index; }
        bool operator !=(const  mtx_iterator & other) const { return !operator==(other); }
        reference operator*() const { return val[contiguous_index]; }
        pointer operator ->() const { return val + contiguous_index; }
        reference operator[](difference_type n) const { return val[static_cast<char>(contiguous_index) + n]; }
        difference_type operator-(const mtx_iterator & other) const { return static_cast<char>(contiguous_index) - static_cast<char>(other.contiguous_index); }
        mtx_iterator& operator+=(difference_type n) { contiguous_index += n; return *this; } 
        mtx_iterator& operator-=(difference_type n) { contiguous_index -= n; return *this; }
        mtx_iterator  operator+ (difference_type n) const { mtx_iterator other = *this; other += n; return other; }
        mtx_iterator  operator- (difference_type n) const { mtx_iterator other = *this; other -= n; return other; }
        friend mtx_iterator  operator+(difference_type n, const mtx_iterator& a) { return a+n; }
        bool operator< (const  mtx_iterator & other) const { return contiguous_index <  other.contiguous_index; }
        bool operator> (const  mtx_iterator & other) const { return contiguous_index >  other.contiguous_index; }
        bool operator<=(const  mtx_iterator & other) const { return contiguous_index <= other.contiguous_index; }
        bool operator>=(const  mtx_iterator & other) const { return contiguous_index >= other.contiguous_index; }

        std::size_t continuous_index() const { return contiguous_index; }
        Index index() const { return BiSymTensor4Rank<3, FT>::index(contiguous_index); }

        friend class BiSymTensor4Rank<3, FT>;    
    };
    using iterator = mtx_iterator<FT>;
    using const_iterator = mtx_iterator<const FT>;

    iterator begin()              { return iterator(m_dat.data(), 0);       }
    iterator end()                { return iterator(m_dat.data(), continuous_size());       }
    const_iterator cbegin() const { return const_iterator(m_dat.data(), 0); }
    const_iterator cend()   const { return const_iterator(m_dat.data(), continuous_size()); }
    const_iterator begin()  const { return cbegin();                        }
    const_iterator end()    const { return cend();                          }

    inline FT& operator[](Index i){ return m_dat[continuous_index(i)]; }
    inline FT operator[](Index i) const { return m_dat[continuous_index(i)]; }
    inline FT& operator[](std::size_t i) { return m_dat[i]; }
    inline FT operator[](std::size_t i) const { return m_dat[i]; }
    inline FT& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) { return m_dat[continuous_index(Index(i, j, k, l))]; }
    inline FT operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const { return m_dat[continuous_index(Index(i, j, k, l))]; }
    
    inline FT& operator()(std::array<std::size_t, 2> i, std::array<std::size_t, 2> j) { return operator()(i[0], i[1], j[0], j[1]); }
    inline FT operator()(std::array<std::size_t, 2> i, std::array<std::size_t, 2> j) const { return operator()(i[0], i[1], j[0], j[1]); }
    inline FT& operator()(std::array<std::size_t, 4> i) { return operator()(i[0], i[1], i[2], i[3]); }
    inline FT operator()(std::array<std::size_t, 4> i) const { return operator()(i[0], i[1], i[2], i[3]); }

    inline FT SquareFrobNorm() const;
    inline FT Dot(const BiSymTensor4Rank<3, FT>& b) const;
    static inline BiSymTensor4Rank<3, FT> TensorSquare(const SymMtx<3, FT>& s);
    static inline BiSymTensor4Rank<3, FT> TensorSymMul2(const SymMtx<3, FT>& f, const SymMtx<3, FT>& s);
    static inline BiSymTensor4Rank<3, FT> Identity(FT val = FT(1));

    inline BiSymTensor4Rank<3, FT>& operator+=(const BiSymTensor4Rank<3, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline BiSymTensor4Rank<3, FT>& operator-=(const BiSymTensor4Rank<3, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline BiSymTensor4Rank<3, FT> operator+(const BiSymTensor4Rank<3, FT>& a) const { BiSymTensor4Rank<3, FT> r(*this); return (r += a);}
    inline BiSymTensor4Rank<3, FT> operator-(const BiSymTensor4Rank<3, FT>& a) const { BiSymTensor4Rank<3, FT> r(*this); return (r -= a);}
    inline BiSymTensor4Rank<3, FT> operator-() const { BiSymTensor4Rank<3, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline BiSymTensor4Rank<3, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline BiSymTensor4Rank<3, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline BiSymTensor4Rank<3, FT> operator*(FT a) const { BiSymTensor4Rank<3, FT> r(*this); return (r *= a);}
    friend BiSymTensor4Rank<3, FT> operator*(FT c, const BiSymTensor4Rank<3, FT>& a){ return a.operator*(c); }
    inline BiSymTensor4Rank<3, FT> operator/(FT a) const { BiSymTensor4Rank<3, FT> r(*this); return (r /= a); }
};

template<typename FT>
FT BiSymTensor4Rank<3, FT>::SquareFrobNorm() const { 
    FT v = FT();
    for (std::size_t i = 0; i < continuous_size(); ++i) 
        v += operator[](i)*operator[](i)*index_duplication(i);
    return v; 
}
template<typename FT>
FT BiSymTensor4Rank<3, FT>::Dot(const BiSymTensor4Rank<3, FT>& b) const { 
    FT v = FT();
    for (std::size_t i = 0; i < continuous_size(); ++i) 
        v += operator[](i)*b[i]*index_duplication(i);
    return v;  
}
template<typename FT>
BiSymTensor4Rank<3, FT> BiSymTensor4Rank<3, FT>::TensorSquare(const SymMtx<3, FT>& s){
    BiSymTensor4Rank<3, FT> r;
    for (auto it = r.begin(); it != r.end(); ++it){
        auto q = it.index();
        *it = s(q.i, q.j) * s(q.k, q.l);
    }
    return r;  
}
template<typename FT>
BiSymTensor4Rank<3, FT> BiSymTensor4Rank<3, FT>::TensorSymMul2(const SymMtx<3, FT>& f, const SymMtx<3, FT>& s){
    BiSymTensor4Rank<3, FT> r;
    for (auto it = r.begin(); it != r.end(); ++it){
        auto q = it.index();
        *it = f(q.i, q.j) * s(q.k, q.l) + s(q.i, q.j) * f(q.k, q.l);
    }
    return r;
}
// I_ijkl = (\delta_ik \delta_jl + \delta_il \delta_jk) / 2
template<typename FT>
BiSymTensor4Rank<3, FT> BiSymTensor4Rank<3, FT>::Identity(FT val){
    BiSymTensor4Rank<3, FT> r;
       r(0, 0, 0, 0) = r(1, 1, 1, 1) = r(2, 2, 2, 2) = val    ;
       r(0, 1, 0, 1) = r(0, 2, 0, 2) = r(1, 2, 1, 2) = val / 2;
    return r;
}

}

#endif //CARNUM_PHYSICAL_TENSORS_3D_INL