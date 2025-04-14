//
// Created by Liogky Alexey on 01.04.2025.
//

#ifndef CARNUM_INTERVAL_INTERVAL_MAP_INL
#define CARNUM_INTERVAL_INTERVAL_MAP_INL

#include "interval_map.h"

namespace Ani{
    template<typename Key, typename T, typename Compare, typename Allocator>
    template<typename Dummy, bool isOk>
    std::ostream& interval_map<Key, T, Compare, Allocator>::PrintState<Dummy, isOk>::print(std::ostream&, const interval_map<Key, T, Compare, Allocator>&, const ParseIvlMapTraits&){
        return std::cout << "Print interval_map<" << typeid(key_type).name() << ", " << typeid(mapped_type).name() << "> unsupported";
    }
    
    template<typename Key, typename T, typename Compare, typename Allocator>
    template<typename Dummy>
    std::ostream& interval_map<Key, T, Compare, Allocator>::PrintState<Dummy, true>::print(std::ostream& out, const interval_map<Key, T, Compare, Allocator>& r, const ParseIvlMapTraits& traits){
        if (r.m_map_list.empty()) return out;
        std::string space;
        if (traits.ignore_symbols.find(" ") != std::string::npos)
            space = " ";
        auto &colon = traits.interval_delimiter, &arrow = traits.map_delimiter, &semicolon = traits.elem_delimiter;
        if (r.m_map_list.size() == 1){
            auto v = r.m_map_list.front();
            if ((r.boundary_status & 1) && (r.boundary_status & 2) && r.m_comp(v.first.m_end, v.first.m_start))
                return out << v.first.m_start << space << colon << space << v.first.m_end << space << arrow << space << v.second;
            return out  
                << ((r.boundary_status & 1) ? std::string() : std::to_string(v.first.m_start)) << space << colon << space 
                << ((r.boundary_status & 2) ? std::string() : std::to_string(v.first.m_end)) << space << arrow << space << v.second;
        }
        auto it_st = r.m_map_list.begin();
        auto it_end = std::prev(r.m_map_list.end());
        auto it = it_st;
        out << ((r.boundary_status & 1) ? std::string() : std::to_string(it->first.m_start)) << space << colon << space << it->first.m_end << space << arrow << space << it->second;
        for (auto it = std::next(it_st); it < it_end; ++it)
            out << semicolon << space << it->first.m_start << space << colon << space << it->first.m_end << space << arrow << space << it->second;
        it = it_end;
        out << semicolon << space << it->first.m_start << space << colon << space << ((r.boundary_status & 2) ? std::string() : std::to_string(it->first.m_end)) << space << arrow << space << it->second;
        return out;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::ostream& interval_map<Key, T, Compare, Allocator>::debug_print_state(std::ostream& out) const { 
        out << "interval_map[" << m_map_list.size() << "] = { ";
        PrintState<>::print(out, *this, ParseIvlMapTraits()); 
        out << " }";
        return out;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::ostream& interval_map<Key, T, Compare, Allocator>::print(std::ostream& out, const ParseIvlMapTraits& traits) const { 
        PrintState<>::print(out, *this, traits); 
        return out;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    bool interval_map<Key, T, Compare, Allocator>::contains(const key_type& key) const {
        if (m_map_list.empty()) return false;
        auto it_ed = m_map_list.end();
        if (boundary_status & 2){
            it_ed = std::prev(m_map_list.end());
            if (!m_comp(key, it_ed->first.m_start)) return true;
        }
        auto it = std::upper_bound(m_map_list.begin(), it_ed, key, [&cmp = m_comp](auto a, auto b){ return cmp(a, b.first.m_end); });
        return (it != it_ed) && (((boundary_status & 1) && it == m_map_list.begin()) || !m_comp(key, it->first.m_start));
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    const T& interval_map<Key, T, Compare, Allocator>::at( const Key& key ) const {
        auto it = internal_find(key);
        if (it == m_map_list.end()) 
            throw std::out_of_range("Map does not contain the key");
        return it->second;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::pair<typename interval_map<Key, T, Compare, Allocator>::iterator, bool> 
        interval_map<Key, T, Compare, Allocator>::insert(key_type start_interval, key_type end_interval, const mapped_type& value){ 
        return wrap_insert(start_interval, end_interval, std::move(value), 0);
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::pair<typename interval_map<Key, T, Compare, Allocator>::iterator, bool> 
        interval_map<Key, T, Compare, Allocator>::insert(key_type start_interval, infinite_t, const mapped_type& value){
        // key_type end_interval = start_interval + 1;
        key_type end_interval = start_interval;
        if (!m_map_list.empty()){
            auto v = m_map_list.back().first.m_end;
            if (m_comp(end_interval, v))
                end_interval = v;
        }
        auto r = wrap_insert(start_interval, end_interval, std::move(value), 2);
        if (r.second) boundary_status |= 2;
        return r;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::pair<typename interval_map<Key, T, Compare, Allocator>::iterator, bool> 
        interval_map<Key, T, Compare, Allocator>::insert(infinite_t, key_type end_interval, const mapped_type& value){
        // key_type start_interval = end_interval - 1;
        key_type start_interval = end_interval;
        if (!m_map_list.empty()){
            auto v = m_map_list.front().first.m_start;
            if (m_comp(v, start_interval))
                start_interval = v;
        }
        auto r = wrap_insert(start_interval, end_interval, std::move(value), 1);
        if (r.second) boundary_status |= 1;
        return r;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::pair<typename interval_map<Key, T, Compare, Allocator>::iterator, bool> 
        interval_map<Key, T, Compare, Allocator>::insert(infinite_t, infinite_t, const mapped_type& value){
        for (auto it = m_map_list.begin(); it != m_map_list.end(); ++it) if (it->second != value)
            return {iterator{it, this}, false};
        m_map_list.resize(1);
        m_map_list[0].second = std::move(value);
        m_map_list[0].first.m_start = key_type(0);
        // m_map_list[0].first.m_end = key_type(0) + 1;
        m_map_list[0].first.m_end = key_type(0);
        boundary_status = 1|2;
        return {iterator{m_map_list.begin(), this}, true};
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    bool interval_map<Key, T, Compare, Allocator>::contains(const std::pair<interval<key_type>, mapped_type>* it, const key_type& key) const {
        if (m_map_list.empty()) return false;
        // if ((boundary_status & 1) && (boundary_status & 2) && m_map_list.size() == 1)
        //     return true;
        if ((boundary_status & 2) && it == &(*std::prev(m_map_list.end())))
            return !m_comp(key, it->first.m_start);
        if ((boundary_status & 1) && it == &(*m_map_list.begin()))
            return m_comp(key, it->first.m_end);
        return !m_comp(key, it->first.m_start) && m_comp(key, it->first.m_end);
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    typename interval_map<Key, T, Compare, Allocator>::const_vec_iterator 
        interval_map<Key, T, Compare, Allocator>::internal_find(const key_type& key) const {
        if (m_map_list.empty()) return m_map_list.end();
        auto it_ed = m_map_list.end();
        if (boundary_status & 2){
            it_ed = std::prev(m_map_list.end());
            if (!m_comp(key, it_ed->first.m_start)) return it_ed;
        }
        auto it = std::upper_bound(m_map_list.begin(), it_ed, key, [&cmp = m_comp](auto a, auto b){ return cmp(a, b.first.m_end); }); 
        if ((it != it_ed) && (((boundary_status & 1) && it == m_map_list.begin()) || !m_comp(key, it->first.m_start)))
            return it;
        return m_map_list.end();
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    typename interval_map<Key, T, Compare, Allocator>::const_vec_iterator 
        interval_map<Key, T, Compare, Allocator>::internal_lower_bound( const Key& key ) const {
        if (m_map_list.empty()) return m_map_list.end();
        auto it_ed = m_map_list.end();
        if (boundary_status & 2){
            auto it = std::prev(m_map_list.end());
            if (!comp(key, it_ed->first.m_start)) return it_ed;
            it_ed = it;
        }
        auto it = std::upper_bound(m_map_list.begin(), it_ed, key, [&cmp = m_comp](auto a, auto b){ return cmp(a, b.first.m_end); });
        return it;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    typename interval_map<Key, T, Compare, Allocator>::const_vec_iterator 
        interval_map<Key, T, Compare, Allocator>::internal_upper_bound( const Key& key ) const {
        if (m_map_list.empty()) return m_map_list.end();
        auto it_ed = m_map_list.end();
        if (boundary_status & 2){
            auto it = std::prev(m_map_list.end());
            if (!comp(key, it_ed->first.m_start)) return it_ed;
            it_ed = it;
        }
        auto it = std::upper_bound(m_map_list.begin(), it_ed, key, [&cmp = m_comp](auto a, auto b){ return cmp(a, b.first.m_end); });
        if (it == it_ed) return it_ed;
        if (std::max(it->first.m_end, key) - std::min(it->first.m_end, key) == 1) 
            it = std::next(it);
        return it;
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::pair<typename interval_map<Key, T, Compare, Allocator>::iterator, bool> 
        interval_map<Key, T, Compare, Allocator>::wrap_insert(key_type start_interval, key_type end_interval, const mapped_type& value, unsigned char bnd_type){
        auto r = insert(start_interval, end_interval, value, bnd_type);
        return std::pair<iterator, bool>{iterator{r.first, this}, r.second};
    }

    template<typename Key, typename T, typename Compare, typename Allocator>
    std::pair<typename interval_map<Key, T, Compare, Allocator>::vec_iterator, bool> 
        interval_map<Key, T, Compare, Allocator>::insert(key_type start_interval, key_type end_interval, const mapped_type& value, unsigned char bnd_type){
        if (m_comp(end_interval, start_interval) || (bnd_type == 0 && start_interval == end_interval)) 
            return {m_map_list.end(), false};
        // if (!m_comp(start_interval, end_interval)) return {m_map_list.end(), false};
        if (boundary_status & 1)
            m_map_list.front().first.m_start = std::min(m_map_list.front().first.m_start, start_interval, m_comp);
        if (boundary_status & 2)
            m_map_list.back().first.m_end = std::max(m_map_list.back().first.m_end, end_interval, m_comp);
        
        auto it_st = std::upper_bound(m_map_list.begin(), m_map_list.end(), start_interval, [&cmp = m_comp](auto a, auto b){ return cmp(a, b.first.m_end); });
        if (bnd_type == 1 && start_interval == end_interval && it_st != m_map_list.begin()){
            auto it = std::prev(it_st);
            if (it->first.m_end == start_interval)
                it_st = it;
        }
        if ((boundary_status & 2) && it_st == m_map_list.end()){ //< extra logic for case m_start = m_end if right-infinity
            auto b = m_map_list.back().first;
            if (b.m_end == b.m_start && b.m_end == start_interval && !(bnd_type == 1 && start_interval == end_interval))
                it_st = std::prev(it_st);
        }
        auto it_st0 = it_st;
        if (it_st != m_map_list.begin()){
            auto it_prev = std::prev(it_st);
            if (it_prev->first.m_end == start_interval && it_prev->second == value)
                it_st = it_prev;     
        }
        auto it_end = (it_st0 != m_map_list.end()) ? 
                    std::lower_bound(it_st0, m_map_list.end(), end_interval, [&cmp = m_comp](auto a, auto b){ return cmp(a.first.m_start, b); }) : m_map_list.end();
        if ((boundary_status & 1) && it_end == m_map_list.begin()){ //< extra logic for case m_start = m_end if left infinity
            auto b = m_map_list.front().first;
            if (b.m_start == b.m_end && b.m_start == end_interval)
                it_end = std::next(it_end);
        }
        auto it_end0 = it_end;
        if (it_end != m_map_list.end() && it_end->first.m_start == end_interval && it_end->second == value)
            ++it_end;
        
        for (auto it = it_st0; it != it_end0; ++it) if (it->second != value)
            return {it, false};
        
        auto it = it_st;
        if (std::distance(it_st, it_end) == 0){
            it = m_map_list.emplace(it_end, interval<key_type>{start_interval, end_interval}, std::move(value));
        } else {
            it->first.m_start = std::min(it_st->first.m_start, start_interval, m_comp);
            it->first.m_end = std::max(std::prev(it_end)->first.m_end, end_interval, m_comp);
            m_map_list.erase(std::next(it), it_end);
        }
        return {it, true};
    }

    template<typename T>
    T details::DefaultParser<T>::operator()(const std::string& input) const {
        std::istringstream oss(input);
        T res;
        oss >> res;
        if (!oss.eof())
            throw std::runtime_error("Wrong value");
        return res;
    }

    template<typename T1, typename T2, typename ParseT1, typename ParseT2>
    interval_map<T1, T2> parse_interval_map(const std::string& input, std::size_t pos, std::size_t count, const ParseT1& parse_interval_num, const ParseT2& parse_value, const ParseIvlMapTraits& traits){ 
        // ( ((+|-)?\d+)?\s*:((+|-)?\d+)?\s*->((+|-)?\d+) ;)*
        if (traits.interval_delimiter.empty() || traits.map_delimiter.empty())
            throw std::runtime_error("Wrong parser traits");
        Ani::interval_map<T1, T2> m;
        if (count == 0) 
            return m;
        
        auto& spaces = traits.ignore_symbols;
        auto end = input.find_last_not_of(spaces, count - 1);
        if (end == std::string::npos || end < pos) 
            return m;
        auto st = input.find_first_not_of(spaces, pos);
        pos = st;
        while (pos < end){
            auto lend = input.find(traits.elem_delimiter, pos);
            auto colon_pos = input.find(traits.interval_delimiter, pos);
            if (colon_pos == std::string::npos || colon_pos > lend)
                throw std::invalid_argument("Missing \"" + traits.interval_delimiter + "\" separator");
            auto arrow_pos = input.find(traits.map_delimiter, colon_pos+1);
            if (arrow_pos == std::string::npos || arrow_pos > lend)
                throw std::invalid_argument("Missing \"" + traits.map_delimiter + "\" operator");
            
            auto left_st = pos, left_ed = pos;
            left_st = input.find_first_not_of(spaces, pos);
            if (left_st < colon_pos)
                left_ed = input.find_last_not_of(spaces, colon_pos - 1);
            else
                left_st = std::max(colon_pos, left_ed+1);
    
            auto right_st = colon_pos+1, right_ed = colon_pos;
            right_st = input.find_first_not_of(spaces, colon_pos+1);
            if (right_st < arrow_pos)
                right_ed = input.find_last_not_of(spaces, arrow_pos - 1);
            
            auto value_st = arrow_pos+2, value_ed = arrow_pos+1;
            value_st = input.find_first_not_of(spaces, arrow_pos+2);
            if (value_st == std::string::npos || value_st >= lend)
                throw std::invalid_argument("Missing mapping value");
            value_ed = input.find_last_not_of(spaces, lend - 1);
            
            unsigned char status = (left_st > left_ed ? 1 : 0) | (right_st > right_ed ? 2 : 0);
            std::string left = left_st > left_ed ? std::string() : input.substr(left_st, left_ed - left_st + 1);
            std::string right = right_st > right_ed ? std::string() : input.substr(right_st, right_ed - right_st + 1);
            std::string value = input.substr(value_st, value_ed - value_st + 1);
            switch (status){
                case 0: m.insert(parse_interval_num(left), parse_interval_num(right), parse_value(value)); break;
                case 1: m.insert(decltype(m)::tag_inf, parse_interval_num(right), parse_value(value)); break;
                case 2: m.insert(parse_interval_num(left), decltype(m)::tag_inf, parse_value(value)); break;
                case 3: m.insert(decltype(m)::tag_inf, decltype(m)::tag_inf, parse_value(value)); break;
                default:
                    throw std::runtime_error("Reached unreacheable code");
            }
            pos = (lend == std::string::npos) ? std::string::npos : input.find_first_not_of(spaces, lend+1);
        }
    
        return m;
    }
}

#endif //CARNUM_INTERVAL_INTERVAL_MAP_INL