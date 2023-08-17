//
// Created by Liogky Alexey on 01.09.2023.
//

#ifndef CARNUM_INTERVAL_ENUM_H
#define CARNUM_INTERVAL_ENUM_H

#include <vector>
#include <map>
#include <set> 
#include <array>
#include <algorithm>
#include <iterator> 
#include <iostream>
#include <cassert>

namespace Ani{ 

/// Store enumerator of values. Suppose values clusterization in intervals, 
/// storage represent array of pairs: number of first elem in interval and first elem in interval 
/// e.g. for values 1,2,3, 5,6, 8 storage is {0, 1}, {3, 5}, {5, 8}, {6, 9}
template<typename TEnumInt, typename TMapInt>
struct interval_own_memory{
    using EnumInt = TEnumInt;
    using MapInt = TMapInt;
    using const_iterator = typename std::vector<std::pair<EnumInt, MapInt>>::const_iterator;

    std::vector<std::pair<EnumInt, MapInt>> m_dat;
    auto begin() { return m_dat.begin(); }
    auto begin() const { return m_dat.begin(); }
    auto end() { return m_dat.end(); }
    auto end() const { return m_dat.end(); }
    const_iterator cbegin() const { return m_dat.cbegin(); }
    const_iterator cend() const { return m_dat.cend(); }
    bool empty() const { return m_dat.empty(); }
    void clear() { m_dat.clear(); }
};

/// Store enumerator of values suppose values clusterization in intervals, 
/// storage represent array of pairs: number of first elem in interval and first elem in interval 
/// e.g. for values 1,2,3, 5,6, 8 storage is  0,1,  3,5,  5,8,  6,9
template<typename Int>
struct interval_external_memory{
    using EnumInt = Int;
    using MapInt = Int;

    Int* m_dat_st = nullptr;
    Int* m_dat_end = nullptr;

    struct const_iterator{
    protected:
        Int* m_p = nullptr;
        const_iterator(Int* dat): m_p{dat} {}
    public:
        typedef std::random_access_iterator_tag   iterator_category;
        typedef std::pair<Int, Int> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef const value_type*  pointer;
        typedef value_type reference;
        mutable value_type m_val;

        const_iterator() = default;
        const_iterator(const const_iterator& other) = default;
        const_iterator& operator= (const const_iterator & other) = default;
        const_iterator& operator++() { m_p += 2; return *this; }
        const_iterator  operator++(int) {const_iterator ret(*this); operator++(); return ret;}
        const_iterator& operator--() { m_p -= 2; return *this; }
        const_iterator  operator--(int) {const_iterator ret(*this); operator--(); return ret;}
        const_iterator& operator+=(difference_type n) { m_p += 2*n; return *this; }
        const_iterator& operator-=(difference_type n) { m_p -= 2*n; return *this; }
        difference_type operator- (const const_iterator& b) { return (m_p - b.m_p)/2; }
        bool operator==(const const_iterator & other) const { return m_p == other.m_p; }
        bool operator!=(const const_iterator & other) const { return !operator==(other); }
        bool operator< (const const_iterator & other) const { return m_p < other.m_p; }
        bool operator> (const const_iterator & other) const { return m_p > other.m_p; }
        bool operator<=(const const_iterator & other) const { return m_p <= other.m_p; }
        bool operator>=(const const_iterator & other) const { return m_p < other.m_p; }
        reference operator*() const { m_val.first = *m_p, m_val.second = *(m_p+1); return m_val; }
        pointer operator ->() const { m_val.first = *m_p, m_val.second = *(m_p+1); return &m_val; }
        value_type operator[](difference_type n) const {  return {*(m_p+2*n), *(m_p+2*n+1) }; }
        friend const_iterator operator+(const_iterator& a, difference_type n) { const_iterator b(a); return b+=n; }
        friend const_iterator operator+(difference_type n, const_iterator& a) { return a + n; }
        friend const_iterator operator-(const_iterator& a, difference_type n) { const_iterator b(a); return b-=n; }

        friend class interval_external_memory<Int>;
    };
    const_iterator begin() const{ return const_iterator(m_dat_st); }
    const_iterator end() const{ return const_iterator(m_dat_end); }
    auto cbegin() const { return begin(); }
    auto cend() const { return end(); }
    bool empty() const { return m_dat_st == nullptr || m_dat_end == nullptr || m_dat_st == m_dat_end; }
    void clear() { m_dat_st = m_dat_end = nullptr; }
};

/// Class for work with enumerated sequential values stored by intervals 
template<typename Mem>
struct interval_enum{
    using EnumInt = typename Mem::EnumInt;
    using MapInt = typename Mem::MapInt;

    Mem m_mem;

    struct const_iterator{
    private:
        typename Mem::const_iterator m_it, m_end;
        EnumInt m_shift;
        const_iterator(typename Mem::const_iterator it, typename Mem::const_iterator end, EnumInt shift): m_it(it), m_end(end), m_shift(shift) {}
    public:
        typedef std::bidirectional_iterator_tag   iterator_category;
        typedef std::pair<EnumInt, MapInt> value_type;
        typedef std::ptrdiff_t difference_type;
        typedef const value_type*  pointer;
        typedef value_type reference;
        mutable value_type m_val;

        const_iterator() = default;
        const_iterator(const const_iterator& other) = default;
        const_iterator& operator= (const const_iterator & other) = default;
        const_iterator& operator++();
        const_iterator& operator--();
        const_iterator  operator++(int) {const_iterator ret(*this); operator++(); return ret;}
        const_iterator  operator--(int) {const_iterator ret(*this); operator--(); return ret;}
        difference_type operator- (const const_iterator& b) { return (m_it->first - b.m_it->first) + m_shift - b.m_shift; }
        reference operator*() const { m_val = *m_it; m_val.first += m_shift, m_val.second += m_shift; return m_val; }
        pointer operator ->() const { m_val = *m_it; m_val.first += m_shift, m_val.second += m_shift; return &m_val; }
        bool operator==(const const_iterator & other) const { return m_it == other.m_it && m_shift == other.m_shift; }
        bool operator!=(const const_iterator & other) const { return !operator==(other); }
        bool operator< (const const_iterator & other) const { return m_it < other.m_it || (m_it == other.m_it && m_shift < other.m_shift); }
        bool operator> (const const_iterator & other) const { return m_it > other.m_it || (m_it == other.m_it && m_shift > other.m_shift); }
        bool operator<=(const const_iterator & other) const { return !operator>(other); }
        bool operator>=(const const_iterator & other) const { return !operator<(other); }

        friend class interval_enum<Mem>;
    };

    interval_enum() = default;
    interval_enum(Mem m): m_mem{std::move(m)} {}
    interval_enum& setMem(Mem m) { return m_mem = std::move(m), *this; }
    const Mem& data() const { return m_mem; }
    Mem& data() { return m_mem; }

    std::size_t size() const { return m_mem.empty() ? std::size_t(0) : std::size_t(end() - begin()); }
    bool empty() const { return size() == 0; }
    bool contains_left(EnumInt left) const { return left >= 0 && left < size(); }
    bool contains_right(MapInt right) const { return find_right(right) != end(); }
    const_iterator begin() const { return (!m_mem.empty()) ? const_iterator(m_mem.cbegin(), std::prev(m_mem.cend()), 0) : const_iterator(m_mem.cend(), m_mem.cend(), 0); }
    const_iterator end() const {  return (!m_mem.empty()) ? const_iterator(std::prev(m_mem.cend()), std::prev(m_mem.cend()), 0) : const_iterator(m_mem.cend(), m_mem.cend(), 0); }
    const_iterator find_left(EnumInt left) const;
    const_iterator find_right(MapInt right) const;
    const_iterator lower_bound_left(EnumInt left) const{ return left < 0 ? begin() : find_left(left); }
    const_iterator upper_bound_left(EnumInt left) const{ return lower_bound_left(++left); }
    const_iterator lower_bound_right(MapInt right) const;
    const_iterator upper_bound_right(MapInt right) const{ return lower_bound_right(++right); }
    void clear() { m_mem.clear(); }
};

}

#include "interval_enum.inl"


#endif //CARNUM_INTERVAL_ENUM_H