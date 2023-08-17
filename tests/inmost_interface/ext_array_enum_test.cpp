//
// Created by Liogky Alexey on 01.08.2023.
//
#include <gtest/gtest.h>
#include "anifem++/inmost_interface/ext_array_enum.h"

using namespace Ani;

template<unsigned N=4>
bool test_ext_array(const std::vector<std::array<unsigned, N>>& vals, bool with_shuffles = false, int insert_pos = -1, int depends_on = -1){
    struct const_range{
        unsigned operator()(unsigned v) const { return v%3 + v%2 + 1; }
    };
    std::array<unsigned, N> p;
    for(unsigned i = 0; i < N; ++i) 
        p[i] = i;
    bool repeat_cycle = with_shuffles;
    std::vector<std::array<unsigned, N>> vv(vals.size());
    do{
        for (unsigned i = 0; i < vals.size(); ++i){
            std::array<unsigned, N> v;
            for(unsigned j = 0; j < N; ++j)
                v[j] = vals[i][p[j]];
            vv[i] = v;    
        }
        int n_st = insert_pos < 0 ? 0 : insert_pos,
            n_ed = (insert_pos < 0 ? N : insert_pos) + 1,
            m_st = depends_on < 0 ? 0 : depends_on,
            m_ed = depends_on < 0 ? N : (depends_on+1);
        for (int n = n_st; n < n_ed; ++n)
        for (int m = m_st; m < m_ed; ++m){
            std::vector<std::array<unsigned, N+1>> evals;
            for (auto& i: vv){
                std::array<unsigned, N+1> r;
                std::copy(i.data(), i.data() + n, r.data());
                std::copy(i.data() + n, i.data() + i.size(), r.data()+n+1);
                auto rg = const_range{}(i[m]);
                for (int s = 0; s < static_cast<int>(rg); ++s){
                    r[n] = s;
                    evals.push_back(r);
                }
            }
            std::sort(evals.begin(), evals.end(), [](const auto& a, const auto& b) {return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); });
            ext_range_array_enum<N, const_range, unsigned, unsigned> emap(vv, const_range{}, n, m);
            emap.setup();
            if (evals.size() != emap.size())
                return false;
            int have_error = 0;
            auto it = emap.begin();
            for (std::size_t i = 0; i < evals.size(); ++i){
                auto v = evals[i];
                std::size_t ei = emap[v];
                auto vv = emap[i]; 
                auto vp = it->second;
                if (ei != i || v != vv || v != vp){
                    have_error++;
                }
                ++it;
            }
            if (have_error > 0)
                return false;
        }
        if (with_shuffles){
            int j = static_cast<int>(N) - 2;
            while (j != -1 && p[j] >= p[j + 1]) j--;
            if (j == -1){
                repeat_cycle = false;
                break;
            }
            int k = static_cast<int>(N) - 1;
            while (p[j] >= p[k]) k--;
            std::swap(p[j], p[k]);
            int l = j + 1, r = static_cast<int>(N) - 1;
            while (l<r)
                std::swap(p[l++], p[r--]);         
        }
    } while(repeat_cycle);
    return true;   
}

TEST(IInterfaceHelpers, ArrayEnumerator){
    std::vector<std::array<unsigned, 1>> vals1{
        {1}, {0}, {3}, {5}, {7}, {2}  
    };
    std::vector<std::array<unsigned, 2>> vals2{
        {1, 0}, {1, 1}, {1, 2}, {0, 2}, {0, 1}, {3, 1}, {5, 2}, {7, 8}, {2, 6}  
    };
    std::vector<std::array<unsigned, 2>> vals2e{
        {0, 0}, {0, 1}, {0, 2}, {1, 0}, 
        {2, 4}, {2, 3}, 
        {2, 0}, {2, 1},
        {1, 1}, 
        {8, 1}, {8, 2}, {8, 3}, {8, 5},
        {6, 0}, {6, 2}, {6, 4}, {6, 7},
    };
    std::vector<std::array<unsigned, 3>> vals3{
        {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, 
        {1, 2, 4}, {1, 2, 3}, 
        {0, 2, 0}, {0, 2, 1},
        {0, 1, 0},
        {3, 1, 1}, 
        {5, 2, 3}, 
        {7, 8, 1}, {7, 8, 2}, {7, 8, 3}, {7, 8, 5},
        {2, 6, 0}, {2, 6, 2}, {2, 6, 4}, {2, 6, 7},
    };
    std::vector<std::array<unsigned, 4>> vals4{
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0, 2}, {0, 0, 0, 3}, {0, 0, 0, 4}, {0, 0, 0, 5}, {0, 0, 0, 6}, {0, 0, 0, 7},
        {0, 0, 2, 0}, {0, 0, 2, 1}, {0, 0, 2, 2}, {0, 0, 2, 3}, {0, 0, 2, 4}, {0, 0, 2, 5},
        {0, 0, 3, 0},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 0, 2}, {0, 1, 0, 3}, {0, 1, 0, 4}, {0, 1, 0, 5}, {0, 1, 0, 6}, {0, 1, 0, 7},
        {0, 1, 2, 0}, {0, 1, 2, 1}, {0, 1, 2, 2}, {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 2, 5},
        {0, 1, 3, 0},
        {0, 2, 0, 0}, {0, 2, 0, 1}, {0, 2, 0, 2}, {0, 2, 0, 3}, {0, 2, 0, 4}, {0, 2, 0, 5}, {0, 2, 0, 6}, {0, 2, 0, 7},
        {0, 2, 2, 0}, {0, 2, 2, 1}, {0, 2, 2, 2}, {0, 2, 2, 3}, {0, 2, 2, 4}, {0, 2, 2, 5},
        {0, 2, 3, 0},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 0, 2},
        {1, 0, 1, 0},
        {1, 0, 2, 0},
    };
    // EXPECT_TRUE(test_ext_array<1>(vals1, true)) << "Error at N = 1" << std::endl;
    // EXPECT_TRUE(test_ext_array<2>(vals2, true)) << "Error at N = 2" << std::endl;
    // EXPECT_TRUE(test_ext_array<2>(vals2, true)) << "Error at N = 2 spec" << std::endl;
    EXPECT_TRUE(test_ext_array<3>(vals3, true)) << "Error at N = 3" << std::endl;
    EXPECT_TRUE(test_ext_array<4>(vals4, true)) << "Error at N = 4" << std::endl;
}

TEST(IInterfaceHelpers, IntervalEnumerator){
    interval_external_memory<int> mem;
    std::map<int, int> cmap = {
        {0, 1}, {1, 2}, {2, 5}, {3, 6}, {4, 7}, {5, 100}
    };
    std::vector<int> vals;
    vals.push_back(cmap.begin()->first);
    vals.push_back(cmap.begin()->second);  
    int prev_val = vals.back(); 
    for (auto it = std::next(cmap.begin()); it != cmap.end(); ++it){
        if (prev_val + 1 == it->second) {++prev_val; continue;}
        auto cnt = prev_val - vals.back() + vals[vals.size()-2] + 1;
        vals.push_back(cnt);
        vals.push_back(it->second);
        prev_val = it->second;
    }
    auto lcnt = prev_val - vals.back() + 1;
    auto cnt = lcnt + vals[vals.size()-2];
    vals.push_back(cnt);
    vals.push_back(vals[vals.size()-2]+lcnt);
    // for (auto v: vals)
    //     std::cout << v << ", ";
    // std::cout << std::endl; 

    interval_enum<interval_external_memory<int>> rmap;
    mem.m_dat_st = vals.data(), mem.m_dat_end = vals.data() + vals.size();
    rmap.setMem(std::move(mem));
    int have_error = 0;
    auto itt = rmap.begin();
    auto sz = rmap.size();
    for (unsigned i = 0; i < sz; ++i){
        auto l = rmap.find_left(i)->second;
        auto cl = cmap[i];
        if (l != cl || static_cast<int>(i) != itt->first || l != itt->second)
            have_error++;
        // std::cout << i << " " <<  itt->first << " " << l << " " << cl << " " << itt->second << std::endl;   
        ++itt;
    }
    for (unsigned i = sz; i > 0; --i){
        --itt;
        auto l = rmap.find_left(i-1)->second;
        auto cl = cmap[i-1];
        if (l != cl || static_cast<int>(i-1) != itt->first || l != itt->second)
            have_error++;
        // std::cout << i << " " <<  itt->first << " " << l << " " << cl << " " << itt->second << std::endl;   
    }

    EXPECT_TRUE(have_error == 0) << "NERRORS: " << have_error << std::endl;   
}

TEST(IInterfaceHelpers, RemapedArrayEnumerator){
    std::vector<std::array<unsigned, 3>> vv{
        {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, 
        {1, 2, 4}, {1, 2, 3}, 
        {0, 2, 0}, {0, 2, 1},
        {0, 1, 0},
        {3, 1, 1}, 
        {5, 2, 3}, 
        {7, 3, 1}, {7, 3, 2}, {7, 3, 3}, {7, 3, 4},
        {2, 2, 0}, {2, 2, 2}, {2, 2, 4}, {2, 2, 5},
    };
    //0, 1, 2,  3, 4, 5
    //nmem = 4
    const int depends_on = 1;
    const int insert_pos = 0;
    struct const_range{
        unsigned operator()(unsigned v) const { (void) v; return 6; }
    };
    std::array<interval_own_memory<int, int>, 4> mem;
    std::vector<std::pair<int, int>> vals0 = {{0,0}, {3,5},  {4,7}, {6,9}};
    std::vector<std::pair<int, int>> vals1 = {{0,0}, {6,6}};
    std::vector<std::pair<int, int>> vals2 = {{0,4}, {6,10}};
    std::vector<std::pair<int, int>> vals3 = {{0,4}, {3,8}, {6,11}};
    mem[0].m_dat = vals0;
    mem[1].m_dat = vals1;
    mem[2].m_dat = vals2;
    mem[3].m_dat = vals3;
    std::array<interval_enum<interval_own_memory<int, int>>, 4> iemap;
    for (int i = 0; i < 4; ++i) iemap[i].setMem(std::move(mem[i]));
    ext_range_array_enum<3, const_range, unsigned, unsigned> _emap(vv, const_range{}, insert_pos, depends_on);
    ext_array_enum<3, 4, const_range, unsigned, unsigned, interval_own_memory<int, int>> emap;
    emap.m_arr_enum = std::move(_emap);
    emap.m_arr_enum.setup();
    emap.m_int_enum = std::move(iemap);
    auto it = emap.begin();
    int have_error = 0;
    // auto print_arr = [](const auto& a) {
    //     std::cout << "{" << a[0];
    //     for (unsigned i = 1; i < a.size(); ++i) 
    //         std::cout << ", " << a[i];
    //     std::cout << "}";     
    // };
    for (unsigned i = 0; i < emap.size(); ++i){
        auto v = *it;
        auto vv = emap[i];
        auto vi = emap[vv];
        if (v.first != vi || v.second != vv){
            have_error++;
        }
        // std::cout << i << " - " << v.first << " - " << vi << " - "; print_arr(v.second);  std::cout << " - "; print_arr(vv); std::cout << std::endl;
        ++it;
    }

    EXPECT_TRUE(have_error == 0) << "NERRORS: " << have_error << std::endl;
}