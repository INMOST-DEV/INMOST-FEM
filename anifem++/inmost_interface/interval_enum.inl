//
// Created by Liogky Alexey on 01.09.2023.
//

#ifndef CARNUM_INTERVAL_ENUM_INL
#define CARNUM_INTERVAL_ENUM_INL

#include "interval_enum.h"

namespace Ani{

template<typename Mem>
typename interval_enum<Mem>::const_iterator& interval_enum<Mem>::const_iterator::operator++() {
    if (m_it != m_end){
        auto next_it = std::next(m_it);
        if (m_shift + 1 < next_it->first - m_it->first) ++m_shift;
        else {
            m_shift = 0;
            m_it = next_it;
        }
    } else {
        ++m_shift;
    }
    return *this;
}

template<typename Mem>
typename interval_enum<Mem>::const_iterator& interval_enum<Mem>::const_iterator::operator--() {
    if (m_shift > 0) --m_shift;
    else {
        auto prev_it = std::prev(m_it);
        m_shift = m_it->first - 1 - prev_it->first;
        m_it = prev_it;
    }
    return *this;
}

template<typename Mem>
typename interval_enum<Mem>::const_iterator interval_enum<Mem>::find_left(EnumInt left) const{
    if (m_mem.empty()) return end();
    auto it = std::upper_bound(m_mem.cbegin(), m_mem.cend(), left, [](EnumInt left, std::pair<EnumInt, MapInt> a){ return left < a.first; });
    if (it == m_mem.cbegin() || it == m_mem.cend())
        return end();
    --it;

    return const_iterator(it, std::prev(m_mem.cend()), left - it->first);   
}
template<typename Mem>
typename interval_enum<Mem>::const_iterator interval_enum<Mem>::find_right(MapInt right) const{
    if (m_mem.empty()) return end();
    auto it = std::upper_bound(m_mem.cbegin(), m_mem.cend(), right, [](auto right, std::pair<EnumInt, MapInt> a){ return right < a.second; });
    if (it == m_mem.cbegin() || it == m_mem.cend())
        return end();
    auto next_v = *it;            
    --it;
    auto shift = right - it->second;
    if (it->first + shift >= next_v.first)
        return end();

    return const_iterator(it, std::prev(m_mem.cend()), shift);   
}
template<typename Mem>
typename interval_enum<Mem>::const_iterator interval_enum<Mem>::lower_bound_right(MapInt right) const{
    if (m_mem.empty()) return end();
    auto it = std::upper_bound(m_mem.cbegin(), m_mem.cend(), right, [](auto right, std::pair<EnumInt, MapInt> a){ return right < a.second; });
    if (it == m_mem.cbegin()) return begin();
    if (it == m_mem.cend()) return end();
    auto next_v = *it;
    auto next_it = it;
    --it;
    auto shift = right - it->second;
    return (it->first + shift >= next_v.first) ? const_iterator(next_it, std::prev(m_mem.cend()), 0) : const_iterator(it, std::prev(m_mem.cend()), shift);
}

}

#endif //CARNUM_INTERVAL_ENUM_INL