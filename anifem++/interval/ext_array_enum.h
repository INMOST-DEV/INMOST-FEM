//
// Created by Liogky Alexey on 01.08.2023.
//

#ifndef CARNUM_EXT_ARRAY_ENUM_H
#define CARNUM_EXT_ARRAY_ENUM_H

#include <vector>
#include <map>
#include <set> 
#include <array>
#include <algorithm>
#include <iterator> 
#include <iostream>
#include <cassert>
#include "interval_enum.h"

namespace Ani{

template<typename UInt = std::size_t, typename GInt = std::size_t>   
struct vec_range_func{
    std::vector<GInt> ranges;
    vec_range_func() = default;
    vec_range_func(std::vector<GInt> ranges): ranges{std::move(ranges)} {}
    GInt operator()(UInt v) const { return ranges[v]; }
};
template<unsigned N, typename UInt = std::size_t, typename GInt = std::size_t>   
struct arr_range_func{
    std::array<GInt, N> ranges;
    arr_range_func() = default;
    arr_range_func(std::array<GInt, N> ranges): ranges{std::move(ranges)} {}
    void set(std::array<GInt, N> ranges) { ranges = std::move(ranges); }
    GInt operator()(UInt v) const { return ranges[v]; }
};
/// Construct lexicographical enumerator for set B of (N+1)-dimensional arrays: { b[0], b[1], ..., b[N] },
/// where set B obtained from set A of N-dimensional arrays: { a[0], a[1], ..., a[N-1] }
/// by inserting in every a from A at position n every value from range [0, rg(a[m])), with function rg is known,
/// i.e. every b from B obtained from element {b[0], ..., b[n-1], b[n+1], ..., b[N]]} from A by inserting 
/// value b[n] belongs the range [0, rg(b[m + (m>=n)?1:0]))     
/// @note The structure is designed under the assumption that |A| is not large and 
/// A-set can be stored entirely in memory, but |B| may be too large to store B-set directly in storage
template<unsigned N = 4, typename RangeFunc = vec_range_func<std::size_t, std::size_t>, typename UInt = std::size_t, typename GInt = std::size_t>
class ext_range_array_enum{
public:
    /// Enumerator of initial array set
    std::vector<std::array<UInt, N>> m_ord;
    /// Inserted rangable value
    RangeFunc m_range_func;
    UInt insert_pos = 0, depends_on = 0;
    /// Additional internal indexes
    std::vector<std::pair<std::array<UInt, N>, GInt>> m_r, m_w, m_p, m_q;
    //for computing reverse
    std::vector<std::pair< std::array<UInt, N>, std::pair<std::size_t, std::size_t> >> m_qr;
    std::vector<std::pair<GInt, GInt>> m_qrb; 
    GInt m_size = 0;

    struct iterator{
    protected:
        using h_iter = typename std::vector<std::array<UInt, N>>::const_iterator;
        using x_iter = typename std::vector<std::pair<std::array<UInt, N>, GInt>>::const_iterator;
        using q_iter = typename std::vector<std::pair< std::array<UInt, N>, std::pair<std::size_t, std::size_t> >>::const_iterator;
        union hq_iter{ h_iter h; q_iter q; hq_iter(): h() {} };
        
        std::pair<GInt, std::array<UInt, N+1>> id;
        const ext_range_array_enum* ptr;

        h_iter kst; 
        h_iter ked;
        x_iter rit; 
        x_iter wit;
        h_iter pit; 
        hq_iter qit; 
    public:   

        typedef std::bidirectional_iterator_tag  iterator_category;
        typedef std::pair<GInt, std::array<UInt, N+1>> value_type;
        typedef GInt difference_type;
        typedef const std::pair<GInt, std::array<UInt, N+1>>*  pointer;
        typedef const std::pair<GInt, std::array<UInt, N+1>>& reference;

        bool operator ==(const iterator & other) const { return id.first == other.id.first; }
        bool operator !=(const iterator & other) const { return !operator==(other); }
        bool operator < (const iterator & other) const { return id.first < other.id.first; }
        bool operator > (const iterator & other) const { return id.first > other.id.first; }
        bool operator <=(const iterator & other) const { return id.first <= other.id.first; }
        bool operator >=(const iterator & other) const { return id.first < other.id.first; }
        reference operator*() const { return id; }
        pointer operator ->() const { return &id; }
        iterator& operator ++() { return *this = std::move(ptr->next_it(*this)); }
        iterator  operator ++(int) { iterator ret(*this); operator++(); return ret;}
        iterator& operator --() { return *this = std::move(ptr->prev_it(*this)); }
        iterator  operator --(int) { iterator ret(*this); operator--(); return ret;}
        iterator& operator +=(difference_type n) { return *this = std::move(ptr->begin(id.first+n)); }
        iterator& operator -=(difference_type n) { return *this = std::move(ptr->begin(id.first+n)); }
        friend iterator operator+(const iterator& a, difference_type n) { iterator b(a); return b+=n; }
        friend iterator operator+(difference_type n, const iterator& a) { return a + n; }
        friend iterator operator-(const iterator& a, difference_type n) { iterator b(a); return b-=n; }
        friend iterator operator-(difference_type n, const iterator& a) { return a - n; }

        friend class ext_range_array_enum;
    };

    ext_range_array_enum() = default;
    ext_range_array_enum(std::vector<std::array<UInt, N>> from_set, RangeFunc rgf, UInt insert_pos, UInt depends_on):
        m_range_func{std::move(rgf)}, insert_pos{insert_pos}, depends_on{depends_on} { _setBaseSet(std::move(from_set)); }
    ext_range_array_enum& setBaseSet(std::vector<std::array<UInt, N>> from_set){ _setBaseSet(std::move(from_set)); return *this; }
    ext_range_array_enum& setRangedExtension(RangeFunc rgf, UInt insert_position, UInt depends_on_num) { return m_range_func = std::move(rgf), insert_pos = insert_position, depends_on = depends_on_num, *this; }
    /// @brief Prepare internal quick indexes
    ext_range_array_enum& setup();

    GInt size() const { return m_size; }
    bool empty() const { return size() == 0; }
    bool contains(const std::array<UInt, N+1>& i) const;
    std::size_t count(const std::array<UInt, N+1>& i) const { return contains(i) ? 1 : 0; }
    iterator begin(GInt i = 0) const;
    iterator end() const{ iterator x; x.id.first = m_size; return x; }
    iterator find(GInt i) const { return i < size() ? begin(i) : end(); }
    iterator find(const std::array<UInt, N+1>& i) const { return contains(i) ? begin(this->operator[](i)) : end(); }
    /// Returns an iterator to the first element not less than the given key
    iterator lower_bound(const std::array<UInt, N+1>& i) const;
    /// Returns an iterator to the first element greater than the given key
    iterator upper_bound(const std::array<UInt, N+1>& i) const { auto j = i; ++j[N]; return lower_bound(j); }
    ///Straight mapping
    GInt operator[](const std::array<UInt, N+1>& i) const;
    ///Reverse mapping
    std::array<UInt, N+1> operator[](GInt i) const;
    void debug_print_state(std::ostream& out = std::cout) const;
    void clear() { m_ord.clear(); insert_pos = 0, depends_on = 0; m_r.clear(); m_w.clear(); m_p.clear(); m_q.clear(); m_qr.clear(); m_qrb.clear(); m_size = 0; }
private:
    void _setBaseSet(std::vector<std::array<UInt, N>> from_set);
    GInt rg_size(UInt v) const { return m_range_func(v); }
    iterator next_it(iterator a) const;
    iterator prev_it(iterator a) const { return begin(a.id.first-1); }
};

/// Construct enumerator for set B of (N+1)-dimensional arrays: { b[0], b[1], ..., b[N] }
/// where set B obtained from set A of N-dimensional arrays: { a[0], a[1], ..., a[N-1] }
/// by inserting in every a from A at position n every value from range of values M_{a[m]}( [0, rg(a[m])) ), with function rg is known,
/// i.e. every b from B obtained from element {b[0], ..., b[n-1], b[n+1], ..., b[N]]} from A by inserting 
/// value b[n] belongs the set M_{b[m + (m>=n)?1:0]}( [0, rg(b[m + (m>=n)?1:0])) ). 
/// Known that a[m] takes values from 0 to NV.
/// M_m is known mapping from contiguous range [0, rg(m)) to numbers, such that M_m(i) < M_m(j) <=> i < j
/// Comparasion in B such that if b1 < b2 <=> 
/// \exist k in range [0, N): if (k != n) b1[k] < b2[k] else M_{b1[m + (m>=n)?1:0]}^{-1}(b1[k]) < M_{b2[m + (m>=n)?1:0]}^{-1}(b2[k])
/// and forall l > k -> if (k != n) b1[k] == b2[k] else M_{b1[m + (m>=n)?1:0]}^{-1}(b1[l]) == M_{b2[m + (m>=n)?1:0]}^{-1}(b2[l]) 
template <unsigned N, unsigned NV, typename RangeFunc, typename UInt, typename GInt, typename Mem>
class ext_array_enum{
public:
    using ExtRangeArrayEnum = ext_range_array_enum<N, RangeFunc, UInt, GInt>;
    using IntervalEnum = interval_enum<Mem>;
    ExtRangeArrayEnum m_arr_enum;
    std::array<IntervalEnum, NV> m_int_enum;

    struct iterator{
    protected:
        const ext_array_enum* ptr;     
        typename IntervalEnum::const_iterator it1;
        typename ExtRangeArrayEnum::iterator it2;
        std::pair<GInt, std::array<UInt, N+1>> val;

        iterator(const ext_array_enum* ptr, typename IntervalEnum::const_iterator it1, typename ExtRangeArrayEnum::iterator it2): ptr{ptr}, it1{it1}, it2{it2} { if (it2 != ptr->m_arr_enum.end()){ val = *it2; val.second[ptr->m_arr_enum.insert_pos] = it1->second; } }
    public:
        typedef std::bidirectional_iterator_tag  iterator_category;
        typedef std::pair<GInt, std::array<UInt, N+1>> value_type;
        typedef GInt difference_type;
        typedef const std::pair<GInt, std::array<UInt, N+1>>*  pointer;
        typedef const std::pair<GInt, std::array<UInt, N+1>>& reference;

        bool operator ==(const iterator & other) const { return it2== other.it2; }
        bool operator !=(const iterator & other) const { return !operator==(other); }
        bool operator < (const iterator & other) const { return it2 < other.it2; }
        bool operator > (const iterator & other) const { return it2 > other.it2; }
        bool operator <=(const iterator & other) const { return it2 <= other.it2; }
        bool operator >=(const iterator & other) const { return it2 < other.it2; }
        reference operator*() const { return val; }
        pointer operator ->() const { return &val; }
        iterator& operator ++();
        iterator& operator --();
        iterator  operator ++(int) { iterator ret(*this); operator++(); return ret;}
        iterator  operator --(int) { iterator ret(*this); operator--(); return ret;}
        iterator& operator +=(difference_type n);
        iterator& operator -=(difference_type n);
        friend iterator operator+(const iterator& a, difference_type n) { iterator b(a); return b+=n; }
        friend iterator operator+(difference_type n, const iterator& a) { return a + n; }
        friend iterator operator-(const iterator& a, difference_type n) { iterator b(a); return b-=n; }
        friend iterator operator-(difference_type n, const iterator& a) { return a - n; }

        friend class ext_array_enum<N, NV, RangeFunc, UInt, GInt, Mem>;
    };


    ext_array_enum() = default;
    ext_array_enum(ExtRangeArrayEnum arr_enum, std::array<IntervalEnum, NV> int_enum): m_arr_enum{arr_enum}, m_int_enum{int_enum} {}
    GInt size() const { return m_arr_enum.size(); }
    bool empty() const { return m_arr_enum.empty(); }
    bool contains(const std::array<UInt, N+1>& i) const;
    std::size_t count(const std::array<UInt, N+1>& i) const { return contains(i) ? 1 : 0; }
    iterator begin(GInt i = 0) const;
    iterator end() const { return iterator(this, m_int_enum[0].end(), m_arr_enum.end()); }
    iterator find(GInt i) const;
    iterator find(const std::array<UInt, N+1>& i) const ;
    iterator lower_bound(const std::array<UInt, N+1>& i) const;
    iterator upper_bound(const std::array<UInt, N+1>& i) const { auto j = i; ++j[N]; return lower_bound(j); }
    // ///Straight mapping
    GInt operator[](const std::array<UInt, N+1>& i) const;
    // ///Reverse mapping
    std::array<UInt, N+1> operator[](GInt i) const;
    void clear() { m_arr_enum.clear(); m_int_enum.clear(); }
};

}

#include "ext_array_enum.inl"

#endif //CARNUM_EXT_ARRAY_ENUM_H