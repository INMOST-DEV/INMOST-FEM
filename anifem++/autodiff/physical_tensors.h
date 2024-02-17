//
// Created by Liogky Alexey on 10.01.2024.
//

#include <numeric>
#include <array> 
#include <iterator>
#include <utility>
#include <sys/stat.h>
#include <ostream>

#ifndef ANIFEM_AUTODIFF_PHYSICAL_TENSORS_H
#define ANIFEM_AUTODIFF_PHYSICAL_TENSORS_H

namespace Ani{

template<std::size_t N, typename FT = double>
struct PhysArr: public std::array<FT, N>{
    PhysArr(): std::array<FT, N>{FT()} {};
    PhysArr(const std::array<FT, N>& a): std::array<FT, N>(a) {}
    PhysArr(std::array<FT, N>&& a): std::array<FT, N>(std::move(a)) {}

    constexpr std::size_t rank() const { return 1U; }
    constexpr std::size_t continuous_size() const { return N; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { (void) continuous_index; return 1; }

    inline FT SquareFrobNorm() const;
};

/// Structure to store symmetrical NxN matrices. 
/// Data stored in row major order
///  a0 a1 a2
///  a3 a4 a5
///  a6 a7 a8
template<std::size_t N, typename FT = double>
struct SymMtx{
    union Index{
        struct {std::size_t i, j; };
        std::size_t id[2];
        Index(): i{0}, j{0} {}
        Index(std::size_t _i, std::size_t _j): i{_i}, j{_j} { if (i > j) std::swap(i, j); }
        bool operator==(Index a) const { return i == a.i && j == a.j; }
        std::size_t operator[](std::size_t n) const { return id[n]; }
    };
    struct IndexMap {
        constexpr IndexMap() : imap(), jmap() {
            for (std::size_t i = 0; i < N; ++i)
            for (std::size_t j = i; j < N; ++j){
                imap[(2*N-1-i)*i/2 + j] = i;
                jmap[(2*N-1-i)*i/2 + j] = j;
            }
        }
        std::size_t imap[N*(N+1)/2];
        std::size_t jmap[N*(N+1)/2];
    };
    static Index index(std::size_t continuous_index) { static const IndexMap a{}; return Index{a.imap[continuous_index], a.jmap[continuous_index]}; }
    static std::size_t continuous_index(Index id){ return (2*N-1-id.i)*id.i/2 + id.j; }

    std::array<FT, N*(N+1)/2> m_dat = {FT(0)};
    constexpr std::size_t rank() const { return 2U; }
    constexpr std::size_t continuous_size() const { return N*(N+1)/2; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const {constexpr IndexMap a{}; return (a.imap[continuous_index] != a.jmap[continuous_index] ? 2 : 1);  }
    
    SymMtx() = default;
    explicit SymMtx(std::array<FT, N*(N+1)/2> arr): m_dat{std::move(arr)} {}

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
        Index index() const { return SymMtx<N, FT>::index(contiguous_index); }

        friend class SymMtx<N, FT>;    
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
    
    inline FT& operator()(std::size_t i, std::size_t j) { return m_dat[Index(i, j)]; }
    inline FT operator()(std::size_t i, std::size_t j) const { return m_dat[Index(i, j)]; }
    inline FT& operator()(std::array<std::size_t, 1> i, std::array<std::size_t, 1> j) { return operator()(i[0], j[0]); }
    inline FT operator()(std::array<std::size_t, 1> i, std::array<std::size_t, 1> j) const { return operator()(i[0], j[0]); }
    inline FT& operator()(std::array<std::size_t, 2> i) { return operator()(i[0], i[1]); }
    inline FT operator()(std::array<std::size_t, 2> i) const { return operator()(i[0], i[1]); }

    inline FT Det() const; 
    inline FT Trace() const;
    inline FT SquareFrobNorm() const;
    inline SymMtx<N, FT> Inv() const;
    inline std::array<FT, N> Mul(const std::array<FT, N>& v) const;
    inline PhysArr<N, FT> Mul(const PhysArr<N, FT>& v) const;
    inline FT Dot(const SymMtx<N, FT>& b) const;
    inline FT Dot(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s) const;
    static SymMtx<N, FT> Identity(FT val = FT(1)) { SymMtx<N, FT> r; for (std::size_t i = 0; i < N; ++i) r(i, i) = val; return r;}
    /// @brief Matrix product
    /// @return m = this * a
    inline SymMtx<N, FT> operator*(const SymMtx<N, FT>& a);
    SymMtx<N, FT> Transpose() const { return *this; }
    
    inline SymMtx<N, FT>& operator+=(const SymMtx<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline SymMtx<N, FT>& operator-=(const SymMtx<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline SymMtx<N, FT> operator+(const SymMtx<N, FT>& a) const { SymMtx<N, FT> r(*this); return (r += a);}
    inline SymMtx<N, FT> operator-(const SymMtx<N, FT>& a) const { SymMtx<N, FT> r(*this); return (r -= a);}
    inline SymMtx<N, FT> operator-() const { SymMtx<N, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline SymMtx<N, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline SymMtx<N, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline SymMtx<N, FT> operator*(FT a) const { SymMtx<N, FT> r(*this); return (r *= a);}
    friend SymMtx<N, FT> operator*(FT c, const SymMtx<N, FT>& a){ return a.operator*(c); }
    inline SymMtx<N, FT> operator/(FT a) const { SymMtx<N, FT> r(*this); return (r /= a); }

    /// @return a x a
    static inline SymMtx<N, FT> TensorSquare(const PhysArr<N, FT>& s);
    /// @return f x s + s x f
    static inline SymMtx<N, FT> TensorSymMul2(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s);
};

/// Structure to store general (unsymmetrical) NxN matrices. 
/// Data stored in row major order
///  a0 a1 a2
///  a3 a4 a5
///  a6 a7 a8
template<std::size_t N, typename FT = double>
struct PhysMtx{
    std::array<FT, N*N> m_dat = {FT(0)};

    constexpr std::size_t rank() const { return 2U; }
    constexpr std::size_t continuous_size() const { return N*N; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { (void) continuous_index; return 1; }

    PhysMtx() = default;
    explicit PhysMtx(const SymMtx<N, FT>& r);
    PhysMtx(const FT* src, bool row_major = true);
    PhysMtx(const std::array<FT, N*N>& src, bool row_major = true): PhysMtx(src.data(), row_major) {}
    inline FT& operator[](std::size_t i) { return m_dat[i]; }
    inline FT operator[](std::size_t i) const { return m_dat[i]; }
    inline FT& operator()(std::size_t i, std::size_t j) { return m_dat[N*i + j]; }
    inline FT operator()(std::size_t i, std::size_t j) const { return m_dat[N*i + j]; }
    
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
    inline PhysMtx<N, FT> Inv() const;
    inline FT Det() const;
    inline std::array<FT, N> Mul(const std::array<FT, N>& v) const;
    inline PhysArr<N, FT> Mul(const PhysArr<N, FT>& v) const;
    inline FT Dot(const PhysMtx<N, FT>& b) const;
    inline FT Dot(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s) const;
    inline PhysMtx<N, FT> operator*(const PhysMtx<N, FT>& a);
    static PhysMtx<N, FT> Identity(FT val = FT(1)) { PhysMtx<N, FT> r; for (std::size_t i = 0; i < N; ++i) r(i, i) = val; return r;}
    PhysMtx<N, FT> Transpose() const { return PhysMtx<N, FT>(m_dat.data(), false); }
    
    inline PhysMtx<N, FT>& operator+=(const PhysMtx<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline PhysMtx<N, FT>& operator-=(const PhysMtx<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline PhysMtx<N, FT> operator+(const PhysMtx<N, FT>& a) const { PhysMtx<N, FT> r(*this); return (r += a);}
    inline PhysMtx<N, FT> operator-(const PhysMtx<N, FT>& a) const { PhysMtx<N, FT> r(*this); return (r -= a);}
    inline PhysMtx<N, FT> operator-() const { PhysMtx<N, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline PhysMtx<N, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline PhysMtx<N, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline PhysMtx<N, FT> operator*(FT a) const { PhysMtx<N, FT> r(*this); return (r *= a);}
    friend PhysMtx<N, FT> operator*(FT c, const PhysMtx<N, FT>& a){ return a.operator*(c); }
    inline PhysMtx<N, FT> operator/(FT a) const { PhysMtx<N, FT> r(*this); return (r /= a); }

    /// @return a x a
    static inline PhysMtx<N, FT> TensorSquare(const PhysArr<N, FT>& s);
    /// @return f x s + s x f
    static inline PhysMtx<N, FT> TensorSymMul2(const PhysArr<N, FT>& f, const PhysArr<N, FT>& s);
};

/// Structure to store general (unsymmetrical) NxNxNxN tensors
/// Data stored as R_ijkl <-> m_dat[l + N * (k + N * (j + N * i))]
template<std::size_t N, typename FT = double>
struct Tensor4Rank{
    std::array<FT, N*N*N*N> m_dat = {FT(0)};

    Tensor4Rank() = default;
    Tensor4Rank(std::array<FT, N*N*N*N> dat): m_dat(dat) {}

    constexpr std::size_t rank() const { return 4U; }
    constexpr std::size_t continuous_size() const { return N*N*N*N; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { (void) continuous_index; return 1; }

    inline FT& operator[](std::size_t i) { return m_dat[i]; }
    inline FT operator[](std::size_t i) const { return m_dat[i]; }
    inline FT& operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) { return m_dat[l + N * (k + N * (j + N * i))]; }
    inline FT operator()(std::size_t i, std::size_t j, std::size_t k, std::size_t l) const { return m_dat[l + N * (k + N * (j + N * i))]; }
    
    inline FT& operator()(std::array<std::size_t, 2> i, std::array<std::size_t, 2> j) { return operator()(i[0], i[1], j[0], j[1]); }
    inline FT operator()(std::array<std::size_t, 2> i, std::array<std::size_t, 2> j) const { return operator()(i[0], i[1], j[0], j[1]); }
    inline FT& operator()(std::array<std::size_t, 4> i) { return operator()(i[0], i[1], i[2], i[3]); }
    inline FT operator()(std::array<std::size_t, 4> i) const { return operator()(i[0], i[1], i[2], i[3]); }

    auto begin() { return m_dat.begin(); }
    auto end() { return m_dat.end(); }
    auto cbegin() const { return m_dat.cbegin(); }
    auto cend() const { return m_dat.cend(); }
    auto begin() const { return m_dat.cbegin(); }
    auto end() const { return m_dat.cend(); }

    inline FT SquareFrobNorm() const;
    inline FT Dot(const Tensor4Rank<N, FT>& b) const;
    static inline Tensor4Rank<N, FT> TensorSquare(const PhysMtx<N, FT>& s);
    static inline Tensor4Rank<N, FT> TensorSymMul2(const PhysMtx<N, FT>& f, const PhysMtx<N, FT>& s);
    static inline Tensor4Rank<N, FT> TensorSquare(const SymMtx<N, FT>& s);
    static inline Tensor4Rank<N, FT> TensorSymMul2(const SymMtx<N, FT>& f, const SymMtx<N, FT>& s);

    inline Tensor4Rank<N, FT>& operator+=(const Tensor4Rank<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline Tensor4Rank<N, FT>& operator-=(const Tensor4Rank<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline Tensor4Rank<N, FT> operator+(const Tensor4Rank<N, FT>& a) const { Tensor4Rank<N, FT> r(*this); return (r += a);}
    inline Tensor4Rank<N, FT> operator-(const Tensor4Rank<N, FT>& a) const { Tensor4Rank<N, FT> r(*this); return (r -= a);}
    inline Tensor4Rank<N, FT> operator-() const { Tensor4Rank<N, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline Tensor4Rank<N, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline Tensor4Rank<N, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline Tensor4Rank<N, FT> operator*(FT a) const { Tensor4Rank<N, FT> r(*this); return (r *= a);}
    friend Tensor4Rank<N, FT> operator*(FT c, const Tensor4Rank<N, FT>& a){ return a.operator*(c); }
    inline Tensor4Rank<N, FT> operator/(FT a) const { Tensor4Rank<N, FT> r(*this); return (r /= a); }
};

/// Structure to store symmetrical NxNxNxN tensors, i.e. such tensors that R_ijkl = R_klij
/// Data for R_ijkl stored as for SymMatrix<N*N, FT> with M_(ij)(kl) = R_ijkl
template<std::size_t N, typename FT = double>
struct SymTensor4Rank{
    union Index{
        struct {std::size_t i, j, k, l; };
        std::size_t id[4];
        Index(): i{0}, j{0}, k{0}, l{0} {}
        Index(std::size_t _i, std::size_t _j, std::size_t _k, std::size_t _l): i{_i}, j{_j}, k{_k}, l{_l} { if (j + i*N > l + k*N) std::swap(i, k), std::swap(j, l); }
        bool operator==(Index a) const { return i == a.i && j == a.j && k == a.k && l == a.l; }
        std::size_t operator[](std::size_t n) const { return id[n]; }
    };
    static Index index(std::size_t continuous_index){ auto q = SymMtx<N*N, FT>::index(continuous_index); return Index(q.i/N, q.i%N, q.j/N, q.j%N); }
    static std::size_t continuous_index(Index id){ return SymMtx<N*N, FT>::continuous_index(typename SymMtx<N*N, FT>::Index{id.j + N*id.i, id.l + N*id.k}); }

    std::array<FT, N*N*(N*N+1)/2> m_dat = {FT(0)};

    SymTensor4Rank() = default;
    SymTensor4Rank(std::array<FT, N*N*(N*N+1)/2> dat): m_dat(dat) {}

    constexpr std::size_t rank() const { return 4U; }
    constexpr std::size_t continuous_size() const { return N*N*(N*N+1)/2; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { constexpr typename SymMtx<N*N, FT>::IndexMap a{}; return (a.imap[continuous_index] != a.jmap[continuous_index] ? 2 : 1);}

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
        Index index() const { return SymTensor4Rank<N, FT>::index(contiguous_index); }

        friend class SymTensor4Rank<N, FT>;    
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
    inline FT Dot(const SymTensor4Rank<N, FT>& b) const;
    static inline SymTensor4Rank<N, FT> TensorSquare(const PhysMtx<N, FT>& s);
    static inline SymTensor4Rank<N, FT> TensorSymMul2(const PhysMtx<N, FT>& f, const PhysMtx<N, FT>& s);
    static inline SymTensor4Rank<N, FT> TensorSquare(const SymMtx<N, FT>& s);
    static inline SymTensor4Rank<N, FT> TensorSymMul2(const SymMtx<N, FT>& f, const SymMtx<N, FT>& s);

    inline SymTensor4Rank<N, FT>& operator+=(const SymTensor4Rank<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline SymTensor4Rank<N, FT>& operator-=(const SymTensor4Rank<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline SymTensor4Rank<N, FT> operator+(const SymTensor4Rank<N, FT>& a) const { SymTensor4Rank<N, FT> r(*this); return (r += a);}
    inline SymTensor4Rank<N, FT> operator-(const SymTensor4Rank<N, FT>& a) const { SymTensor4Rank<N, FT> r(*this); return (r -= a);}
    inline SymTensor4Rank<N, FT> operator-() const { SymTensor4Rank<N, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline SymTensor4Rank<N, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline SymTensor4Rank<N, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline SymTensor4Rank<N, FT> operator*(FT a) const { SymTensor4Rank<N, FT> r(*this); return (r *= a);}
    friend SymTensor4Rank<N, FT> operator*(FT c, const SymTensor4Rank<N, FT>& a){ return a.operator*(c); }
    inline SymTensor4Rank<N, FT> operator/(FT a) const { SymTensor4Rank<N, FT> r(*this); return (r /= a); }
};

/// Structure to store bisymmetrical NxNxNxN tensors, i.e. such tensors that R_ijkl = R_klij = R_jikl
template<std::size_t N, typename FT = double>
struct BiSymTensor4Rank{
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
    // struct IndexMap {
    //     constexpr IndexMap() : imap(), jmap(), kmap(), lmap() {
    //         auto SZ = N*(N+1)/2;
    //         for (std::size_t I = 0; I < N*(N+1)/2; ++I)
    //         for (std::size_t J = I; J < N*(N+1)/2; ++J){
    //             auto q = SymMtx<N, FT>::index(I), t = SymMtx<N, FT>::index(J);
    //             auto id = (2*SZ-1-I)*I/2 + J;
    //             imap[id] = q.i; jmap[id] = q.j; kmap[id] = q.k; lmap[id] = q.l;
    //         } 
    //     }
    //     std::size_t imap[N*(N+1)/2 * (N*(N+1)/2 + 1)/2];
    //     std::size_t jmap[N*(N+1)/2 * (N*(N+1)/2 + 1)/2];
    //     std::size_t kmap[N*(N+1)/2 * (N*(N+1)/2 + 1)/2];
    //     std::size_t lmap[N*(N+1)/2 * (N*(N+1)/2 + 1)/2];
    // };
    static Index index(std::size_t continuous_index) { 
        auto q = SymMtx<N*(N+1)/2, FT>::index(continuous_index); 
        auto ti = SymMtx<N, FT>::index(q.i), tj = SymMtx<N, FT>::index(q.j);
        return Index(ti.i, ti.j, tj.i, tj.i); 
    }
    static std::size_t continuous_index(Index id){ 
        auto I = SymMtx<N, FT>::continuous_index(SymMtx<N, FT>::Index(id.i, id.j)), 
             J = SymMtx<N, FT>::continuous_index(SymMtx<N, FT>::Index(id.k, id.l));
        return SymMtx<N*(N+1)/2, FT>::continuous_index(SymMtx<N*(N+1)/2, FT>::Index(I, J));
    }

    std::array<FT, N*(N+1)/2 * (N*(N+1)/2 + 1) / 2> m_dat = {0};

    BiSymTensor4Rank() = default;
    BiSymTensor4Rank(std::array<FT, N*(N+1)/2 * (N*(N+1)/2 + 1) / 2> dat): m_dat(dat) {}

    constexpr std::size_t rank() const { return 4U; }
    constexpr std::size_t continuous_size() const { return N*(N+1)/2 * (N*(N+1)/2 + 1) / 2; }
    constexpr std::size_t index_duplication(std::size_t continuous_index) const { 
        constexpr typename SymMtx<N*(N+1)/2, FT>::IndexMap a{}; 
        constexpr typename SymMtx<N, FT>::IndexMap b{};
        auto q = typename SymMtx<N*(N+1)/2, FT>::Index{a.imap[continuous_index], a.jmap[continuous_index]};
        return (a.imap[continuous_index] != a.jmap[continuous_index] ? 2 : 1) * (b.imap[q.i] != b.jmap[q.i] ? 2 : 1) * (b.imap[q.j] != b.jmap[q.j] ? 2 : 1);
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
        Index index() const { return BiSymTensor4Rank<N, FT>::index(contiguous_index); }

        friend class BiSymTensor4Rank<N, FT>;    
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
    inline FT Dot(const BiSymTensor4Rank<N, FT>& b) const;
    static inline BiSymTensor4Rank<N, FT> TensorSquare(const SymMtx<N, FT>& s);
    static inline BiSymTensor4Rank<N, FT> TensorSymMul2(const SymMtx<N, FT>& f, const SymMtx<N, FT>& s);

    inline BiSymTensor4Rank<N, FT>& operator+=(const BiSymTensor4Rank<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] += a[i]; return *this; }
    inline BiSymTensor4Rank<N, FT>& operator-=(const BiSymTensor4Rank<N, FT>& a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] -= a[i]; return *this; }
    inline BiSymTensor4Rank<N, FT> operator+(const BiSymTensor4Rank<N, FT>& a) const { BiSymTensor4Rank<N, FT> r(*this); return (r += a);}
    inline BiSymTensor4Rank<N, FT> operator-(const BiSymTensor4Rank<N, FT>& a) const { BiSymTensor4Rank<N, FT> r(*this); return (r -= a);}
    inline BiSymTensor4Rank<N, FT> operator-() const { BiSymTensor4Rank<N, FT> r; for (std::size_t i = 0; i < continuous_size(); ++i) r[i] = -m_dat[i]; return r;}
    inline BiSymTensor4Rank<N, FT>& operator*=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] *= a; return *this; }
    inline BiSymTensor4Rank<N, FT>& operator/=(FT a){ for (std::size_t i = 0; i < continuous_size(); ++i) m_dat[i] /= a; return *this; }
    inline BiSymTensor4Rank<N, FT> operator*(FT a) const { BiSymTensor4Rank<N, FT> r(*this); return (r *= a);}
    friend BiSymTensor4Rank<N, FT> operator*(FT c, const BiSymTensor4Rank<N, FT>& a){ return a.operator*(c); }
    inline BiSymTensor4Rank<N, FT> operator/(FT a) const { BiSymTensor4Rank<N, FT> r(*this); return (r /= a); }
};

template<std::size_t N, typename FT>
inline PhysMtx<N, FT> operator*(const PhysMtx<N, FT>& a, const SymMtx<N, FT>& b);
template<std::size_t N, typename FT>
inline PhysMtx<N, FT> operator*(const SymMtx<N, FT>& a, const PhysMtx<N, FT>& b);
template<std::size_t N, typename FT>
inline PhysMtx<N, FT> operator+(const PhysMtx<N, FT>& a, const SymMtx<N, FT>& b);
template<std::size_t N, typename FT>
inline PhysMtx<N, FT> operator+(const SymMtx<N, FT>& a, const PhysMtx<N, FT>& b);
template<std::size_t N, typename FT>
inline PhysMtx<N, FT> operator-(const PhysMtx<N, FT>& a, const SymMtx<N, FT>& b);
template<std::size_t N, typename FT>
inline PhysMtx<N, FT> operator-(const SymMtx<N, FT>& a, const PhysMtx<N, FT>& b);

template<typename TYPE_TO, typename TYPE_FROM>
TYPE_TO tensor_convert(const TYPE_FROM& v);

template<std::size_t N, typename FT>
inline std::ostream& operator<<(std::ostream& out, const PhysMtx<N, FT>& m);
template<std::size_t N, typename FT>
inline std::ostream& operator<<(std::ostream& out, const SymMtx<N, FT>& m);
template<std::size_t N, typename FT>
inline std::ostream& operator<<(std::ostream& out, const Tensor4Rank<N, FT>& m);
template<std::size_t N, typename FT>
inline std::ostream& operator<<(std::ostream& out, const SymTensor4Rank<N, FT>& m);
template<std::size_t N, typename FT>
inline std::ostream& operator<<(std::ostream& out, const BiSymTensor4Rank<N, FT>& m);

}


#include "physical_tensors_3d.inl"
#include "physical_tensors.inl"


#endif //ANIFEM_AUTODIFF_PHYSICAL_TENSORS_H