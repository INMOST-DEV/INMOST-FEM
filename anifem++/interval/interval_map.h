//
// Created by Liogky Alexey on 01.04.2025.
//

#ifndef CARNUM_INTERVAL_INTERVAL_MAP_H
#define CARNUM_INTERVAL_INTERVAL_MAP_H

#include <vector>
#include <set>
#include <type_traits>
#include <algorithm>

namespace Ani{
    struct ParseIvlMapTraits{
        std::string interval_delimiter = ":";
        std::string map_delimiter = "->";
        std::string elem_delimiter = ";";
        std::string ignore_symbols = " \t";
        ParseIvlMapTraits() = default;
        ParseIvlMapTraits(std::string interval_delim, std::string map_delim, std::string elem_delim, std::string ignore_symbols = " \t"): interval_delimiter{std::move(interval_delim)}, map_delimiter{std::move(map_delim)}, elem_delimiter{elem_delim}, ignore_symbols{ignore_symbols} {}
        ParseIvlMapTraits& setIntervalDelim(std::string delim){ return interval_delimiter = std::move(delim), *this; }
        ParseIvlMapTraits& setMapDelim(std::string delim){ return map_delimiter = std::move(delim), *this; }
        ParseIvlMapTraits& setElemDelim(std::string delim){ return elem_delimiter = std::move(delim), *this; }
        ParseIvlMapTraits& setIgnoreSymbols(std::string symbols){ return ignore_symbols = std::move(symbols), *this; }
    };
    /// @brief Represent half-open interval [start, end)
    template<typename Key>
    struct interval{
        using key_type = Key;

        key_type m_start;
        key_type m_end;

        bool operator==(const interval<key_type>& other) const { return m_start == other.m_start && m_end == other.m_end; } 
    };
    struct infinite_t {};

    /// This is an associative container that maps non-overlapping intervals to values: [key_start, key_end) -> value. 
    /// It provides a reduced subset of the standard key -> value mapping functionality, ensuring unique keys. 
    /// This serves as an efficient alternative to std::map in scenarios 
    /// where many (or even infinitely many) consecutive adjacent keys are mapped to the same value.
    template<typename Key, typename T, typename Compare = std::less<Key>, typename Allocator = std::allocator<std::pair<interval<Key>, T>>>
    struct interval_map{
        using key_type = Key;
        using mapped_type = T;
        using key_compare = Compare;
        using allocator_type = Allocator;
        static constexpr infinite_t tag_inf{};

        // template<typename BaseIterator = vec_iterator, typename ObjType = interval_map<Key, T, Compare, Allocator>>
        template<typename BaseIterator, typename ObjType>
        struct map_iterator: public BaseIterator{
        protected:
            ObjType* m_self = nullptr;

            map_iterator<BaseIterator, ObjType>(BaseIterator it, ObjType* self): BaseIterator(it), m_self{self} { }
        public: 
            bool contains(const key_type& key) const { return m_self->contains(&(this->operator*()), key); }
            bool is_left_infinite() const { return (m_self->boundary_status & 1) && &(this->operator*()) == &(*(m_self->m_map_list.begin())); }
            bool is_right_infinite() const { return (m_self->boundary_status & 2) && &(this->operator*()) == &(*(std::prev(m_self->m_map_list.end()))); }

            friend class interval_map<Key, T, Compare, Allocator>;
        };
        using iterator = map_iterator<typename std::vector<std::pair<interval<key_type>, mapped_type>, allocator_type>::iterator, interval_map<Key, T, Compare, Allocator>>;
        using const_iterator = map_iterator<typename std::vector<std::pair<interval<key_type>, mapped_type>, allocator_type>::const_iterator, const interval_map<Key, T, Compare, Allocator>>;

        iterator begin() { return iterator(m_map_list.begin(), this); }
        iterator end() { return iterator(m_map_list.end(), this); }
        const_iterator cbegin() const { return const_iterator(m_map_list.cbegin(), this); }
        const_iterator cend() const { return const_iterator(m_map_list.cend(), this); }
        const_iterator begin() const { return cbegin(); }
        const_iterator end() const { return cend(); }
        
        void clear() { m_map_list.clear(); boundary_status = 0; }
        bool empty() const { return m_map_list.empty(); }
        std::size_t size() const { return m_map_list.size(); }
        void swap(interval_map<Key, T, Compare, Allocator>& other){ m_map_list.swap(other.m_map_list); std::swap(boundary_status, other.boundary_status); }
        bool contains(const key_type& key) const; 
        std::size_t count(const key_type& key) const { return contains(key) ? 1 : 0; }
        iterator find(const key_type& key) { return iterator(std::next(m_map_list.begin(), std::distance(m_map_list.cbegin(), internal_find(key))), this); }
        const_iterator find(const key_type& key) const { return const_iterator(internal_find(key), this); }
        const T& operator[]( const Key& key ) const { return find(key)->second; }
        const T& at( const Key& key ) const;
        std::pair<iterator, bool> insert(key_type start_interval, key_type end_interval, const mapped_type& value);
        std::pair<iterator, bool> insert(key_type start_interval, infinite_t, const mapped_type& value);
        std::pair<iterator, bool> insert(infinite_t, key_type end_interval, const mapped_type& value);
        std::pair<iterator, bool> insert(infinite_t, infinite_t, const mapped_type& value);

        iterator lower_bound( const Key& key ) { return iterator(std::next(m_map_list.begin(), std::distance(m_map_list.cbegin(), internal_lower_bound(key))), this); }
        const_iterator lower_bound( const Key& key ) const { return const_iterator(internal_lower_bound(key), this); }
        /// @warning Within this function it is assumed that the largest element smaller than m_end is equal to m_end-1 in each interval. 
        /// This assumption may not be consistent with comparator m_comp.
        iterator upper_bound( const Key& key ) { return iterator(std::next(m_map_list.begin(), std::distance(m_map_list.cbegin(), internal_upper_bound(key))), this); }
        const_iterator upper_bound( const Key& key ) const { return const_iterator(internal_upper_bound(key), this); }

        std::ostream& print(std::ostream& out = std::cout, const ParseIvlMapTraits& traits = ParseIvlMapTraits()) const;
        std::ostream& debug_print_state(std::ostream& out = std::cout) const;
    protected:
        key_compare m_comp;

    private:
        using vec_iterator = typename std::vector<std::pair<interval<key_type>, mapped_type>, allocator_type>::iterator;
        using const_vec_iterator = typename std::vector<std::pair<interval<key_type>, mapped_type>, allocator_type>::const_iterator;

        std::vector<std::pair<interval<key_type>, mapped_type>, allocator_type> m_map_list;
        unsigned char boundary_status = 0; // 0 - finite, 1 - left infinite, 2 - right infinite, 3 - left-right infinite
        
        bool contains(const std::pair<interval<key_type>, mapped_type>* it, const key_type& key) const;
        const_vec_iterator internal_find(const key_type& key) const;
        const_vec_iterator internal_lower_bound( const Key& key ) const;
        const_vec_iterator internal_upper_bound( const Key& key ) const;
        std::pair<iterator, bool> wrap_insert(key_type start_interval, key_type end_interval, const mapped_type& value, unsigned char bnd_type);
        std::pair<vec_iterator, bool> insert(key_type start_interval, key_type end_interval, const mapped_type& value, unsigned char bnd_type);

        template <typename TT>
        class has_ostream_operator {
            template <typename U>
            static auto test(U&& u) -> decltype(std::declval<std::ostream&>() << u, std::true_type{});
            static std::false_type test(...);
        public:
            static constexpr bool value = decltype(test(std::declval<T>()))::value;
        };
        template<typename Dummy = void, bool isOk = has_ostream_operator<key_type>::value && has_ostream_operator<mapped_type>::value>
        struct PrintState{
            static std::ostream& print(std::ostream&, const interval_map&, const ParseIvlMapTraits&);
        };
        template<typename Dummy>
        struct PrintState<Dummy, true>{
            static std::ostream& print(std::ostream& out, const interval_map& r, const ParseIvlMapTraits&);
        };
    };

    namespace details{
        template<typename T>
        struct DefaultParser{
            T operator()(const std::string& input) const;
        };
    }
    template<typename T1, typename T2, typename ParseT1 = details::DefaultParser<T1>, typename ParseT2 = details::DefaultParser<T2>>
    interval_map<T1, T2> parse_interval_map(const std::string& input, std::size_t pos = 0, std::size_t count = std::string::npos, const ParseT1& parse_interval_num = details::DefaultParser<T1>(), const ParseT2& parse_value = details::DefaultParser<T2>(), const ParseIvlMapTraits& traits = ParseIvlMapTraits());
}

#include "interval_map.inl"

#endif //CARNUM_INTERVAL_INTERVAL_MAP_H