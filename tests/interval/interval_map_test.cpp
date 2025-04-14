//
// Created by Liogky Alexey on 01.04.2025.
//
#include <gtest/gtest.h>
#include "anifem++/interval/interval_map.h"
#include <string>
#include <sstream>

using namespace Ani;

template<typename T1, typename T2>
struct ToTest{
    unsigned char status = 0;
    std::vector<T1> intervals;
    std::vector<T2> values;
    ToTest(unsigned char status, std::vector<T1> intervals, std::vector<T2> values): status{status}, intervals{intervals}, values{values} {}

    bool test(interval_map<T1, T2>& m) const { return is_same_as(*this, m); } 
    static bool is_same_as(const ToTest& t, const interval_map<T1, T2>& m){
        if (m.empty() || t.values.empty()) return m.empty() && t.values.empty();
        if (m.size() != t.values.size()) return false;
        auto it_beg = m.begin();
        if (m.size() == 1 && it_beg.is_left_infinite() && it_beg.is_right_infinite())
            return t.values[0] == it_beg->second;
        std::size_t off = 0;
        if (it_beg.is_left_infinite()) {
            if (!(t.status & 1)) return false;
            if (t.intervals[0] != it_beg->first.m_end || t.values[0] != it_beg->second) return false;
            ++it_beg;
            ++off;
        }
        auto it_ed = std::prev(m.end());
        if (it_ed.is_right_infinite()) {
            if (!(t.status & 2)) return false;
            if (t.intervals.back() != it_ed->first.m_start || t.values.back() != it_ed->second) return false;
        } else {
            ++it_ed;
        }
        for (auto it = it_beg; it != it_ed; ++it, off += 2){
            if (t.intervals[off + 0] != it->first.m_start || t.intervals[off + 1] != it->first.m_end || t.values[off/2 + off%2] != it->second)
                return false;
        }
        return true;
    }
};

TEST(Interval, IntervalMap){
    interval_map<long, long> m;
    auto Inf = interval_map<long, long>::tag_inf;
    
    auto print_state = [](auto& m) -> std::string {
        std::stringstream oss;
        for (auto it = m.begin(); it != m.end(); ++it)
            oss << (it.is_left_infinite() ? std::string() : std::to_string(it->first.m_start)) << " : " 
                << (it.is_right_infinite() ? std::string() : std::to_string(it->first.m_end)) << " -> " << it->second << "; ";
        return oss.str();
    };

    m.insert(0, 10, 1);     EXPECT_TRUE((ToTest<long, long>(0, {0, 10}, {1}).test(m)));
    m.insert(10, 20, 2);    EXPECT_TRUE((ToTest<long, long>(0, {0, 10, 10, 20}, {1, 2}).test(m)));
    m.insert(20, Inf, 2);   EXPECT_TRUE((ToTest<long, long>(2, {0, 10, 10}, {1, 2}).test(m)));
    m.insert(15, Inf, 2);   EXPECT_TRUE((ToTest<long, long>(2, {0, 10, 10}, {1, 2}).test(m)));
    EXPECT_TRUE(print_state(m) == std::string("0 : 10 -> 1; 10 :  -> 2; "));

    m.clear();
    m.insert(0, 10, 1);     EXPECT_TRUE((ToTest<long, long>(0, {0, 10}, {1}).test(m)));
    m.insert(10, 20, 2);    EXPECT_TRUE((ToTest<long, long>(0, {0, 10, 10, 20}, {1, 2}).test(m)));
    m.insert(20, Inf, 3);   EXPECT_TRUE((ToTest<long, long>(2, {0, 10, 10, 20, 20}, {1, 2, 3}).test(m)));

    m.clear();
    m.insert(0, 10, 1);   //EXPECT_TRUE((ToTest<long, long>(0, {0, 10}, {1}).test(m)));
    m.insert(10, 20, 2);  //EXPECT_TRUE((ToTest<long, long>(0, {0, 10, 10, 20}, {1, 2}).test(m)));
    m.insert(Inf, 0, 2);    EXPECT_TRUE((ToTest<long, long>(1, {0, 0, 10, 10, 20}, {2, 1, 2}).test(m)));

    m.clear();
    m.insert(0, 10, 1);
    m.insert(10, 20, 2);
    m.insert(Inf, 0, 1);     EXPECT_TRUE((ToTest<long, long>(1, {10, 10, 20}, {1, 2}).test(m)));

    m.clear();
    m.insert(0, 3, 1);      EXPECT_TRUE((ToTest<long, long>(0, {0, 3}, {1}).test(m)));
    m.insert(5, 8, 2);      EXPECT_TRUE((ToTest<long, long>(0, {0, 3, 5, 8}, {1, 2}).test(m)));
    m.insert(10, 11, 2);    EXPECT_TRUE((ToTest<long, long>(0, {0, 3, 5, 8, 10, 11}, {1, 2, 2}).test(m)));
    m.insert(4, 12, 2);     EXPECT_TRUE((ToTest<long, long>(0, {0, 3, 4, 12}, {1, 2}).test(m)));
    m.insert(13, Inf, 2);   EXPECT_TRUE((ToTest<long, long>(2, {0, 3, 4, 12, 13}, {1, 2, 2}).test(m)));
    m.insert(5, Inf, 2);    EXPECT_TRUE((ToTest<long, long>(2, {0, 3, 4}, {1, 2}).test(m)));

    m.clear();
    m.insert(0, Inf, 1);    EXPECT_TRUE((ToTest<long, long>(2, {0}, {1}).test(m)));
    m.insert(Inf, 0, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m.insert(Inf, 0, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));

    m.clear();
    m.insert(Inf, 0, 1);    EXPECT_TRUE((ToTest<long, long>(1, {0}, {1}).test(m)));
    m.insert(Inf, 0, 1);    EXPECT_TRUE((ToTest<long, long>(1, {0}, {1}).test(m)));
    m.insert(0, Inf, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m.insert(Inf, 0, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m.insert(0, Inf, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m.insert(Inf, 1, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m.insert(1, Inf, 1);    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    

    m.clear();
    m.insert(0, 3, 1);      EXPECT_TRUE((ToTest<long, long>(0, {0, 3}, {1}).test(m)));
    m.insert(5, 8, 2);      EXPECT_TRUE((ToTest<long, long>(0, {0, 3, 5, 8}, {1, 2}).test(m)));
    m.insert(10, 11, 2);    EXPECT_TRUE((ToTest<long, long>(0, {0, 3, 5, 8, 10, 11}, {1, 2, 2}).test(m)));
    m.insert(4, 12, 2);     EXPECT_TRUE((ToTest<long, long>(0, {0, 3, 4, 12}, {1, 2}).test(m)));
    
    m.insert(13, Inf, 2);   EXPECT_TRUE((ToTest<long, long>(2, {0, 3, 4, 12, 13}, {1, 2, 2}).test(m)));
    EXPECT_TRUE(!m.contains(-1));
    EXPECT_TRUE(m.contains(0));     EXPECT_TRUE(m[0] == 1); EXPECT_TRUE(m.count(0) == 1);
    EXPECT_TRUE(!m.contains(3));                            EXPECT_TRUE(m.count(3) == 0);
    EXPECT_TRUE(m.contains(4));     EXPECT_TRUE(m[4] == 2);
    EXPECT_TRUE(m.contains(6));     EXPECT_TRUE(m[6] == 2);
    EXPECT_TRUE(!m.contains(12));
    EXPECT_TRUE(m.contains(13));    EXPECT_TRUE(m[13] == 2);
    EXPECT_TRUE(m.contains(15));    EXPECT_TRUE(m[15] == 2);

    m.insert(Inf, -3, 5);
    EXPECT_TRUE(!m.contains(-2));
    EXPECT_TRUE(!m.contains(-3));
    EXPECT_TRUE(m.contains(-4));   EXPECT_TRUE(m[-4] == 5);

    struct Point{ double x, y; };
    interval_map<long, Point> mp;
    EXPECT_TRUE(mp.empty());

    m = parse_interval_map<long, long>(":->1");
    // m.print(std::cout) << std::endl;
    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m = parse_interval_map<long, long>(" : -> 1 ");
    // m.print(std::cout) << std::endl;
    EXPECT_TRUE((ToTest<long, long>(3, {}, {1}).test(m)));
    m = parse_interval_map<long, long>("1 : -> 1 ");
    // m.print(std::cout) << std::endl;
    EXPECT_TRUE((ToTest<long, long>(2, {1}, {1}).test(m)));
    m = parse_interval_map<long, long>(" : 1 -> 1 ");
    // m.print(std::cout) << std::endl;
    EXPECT_TRUE((ToTest<long, long>(1, {1}, {1}).test(m)));
    m = parse_interval_map<long, long>("0 : 3 -> 1; 5 : 8 -> 2; 10 : 11 -> 3; 13 : -> 5; ");
    // m.print(std::cout) << std::endl;
    EXPECT_TRUE((ToTest<long, long>(2, {0, 3, 5, 8, 10, 11, 13}, {1, 2, 3, 5}).test(m)));
}