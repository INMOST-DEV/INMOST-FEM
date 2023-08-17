//
// Created by Liogky Alexey on 01.08.2023.
//

#ifndef CARNUM_EXT_ARRAY_ENUM_INL
#define CARNUM_EXT_ARRAY_ENUM_INL

#include "ext_array_enum.h"

namespace Ani{

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
ext_range_array_enum<N, RangeFunc, UInt, GInt>& ext_range_array_enum<N, RangeFunc, UInt, GInt>::setup() {
    auto make_check = [](UInt t){
        return [t](const std::array<UInt, N>& a, const std::array<UInt, N>& b){
            for (unsigned i = 0; i < t; ++i)
                if (a[i] != b[i]) return false;
            return true;    
        };
    };
    auto sum_part_ids = [make_check](UInt st, UInt ed, UInt at, const std::map<std::array<UInt, N>, GInt>& rw){
        std::map<std::array<UInt, N>, GInt> w1;
        auto check_st = make_check(st);
        auto check_ed = make_check(ed);
        
        std::map<UInt, GInt> s_prev;
        std::array<UInt, N> a_ext = rw.begin()->first;
        {
            s_prev.emplace(rw.begin()->first[at], rw.begin()->second);
            w1.emplace(rw.begin()->first, 0);
        }
        for (auto it = std::next(rw.begin()); it != rw.end(); ++it){
            bool ced = check_ed(a_ext, it->first);
            if (ced){
                auto jt = s_prev.find(it->first[at]);
                if (jt != s_prev.end())
                    jt->second += it->second;
                else{
                    s_prev.emplace(it->first[at], it->second);
                    w1.emplace(it->first, 0);
                }
                continue;        
            } 
            bool cst = check_st(a_ext, it->first);
            if (!cst){
                s_prev.clear();
                {
                    s_prev.emplace(it->first[at], it->second);
                    w1.emplace(it->first, 0);
                }
                a_ext = it->first;
                continue;
            }
            a_ext = it->first;
            auto a_cur = a_ext;
            std::fill(a_cur.data() + ed, a_cur.data() + a_cur.size(), 0);
            for (const auto& v: s_prev){
                auto b = a_cur;
                b[at] = v.first;
                w1.emplace(b, v.second);
            }
            auto jt = s_prev.find(it->first[at]);
            if (jt != s_prev.end())
                jt->second += it->second;
            else{
                s_prev.emplace(it->first[at], it->second);
                w1.emplace(it->first, 0);
            }
        }
        return w1;
    };
    auto compare_arr = [](const auto& a, const auto& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
    UInt n = insert_pos, m = depends_on;
    
    std::map<std::array<UInt, N>, GInt> _r, _w, _p, _q;
    std::map<std::array<UInt, N>, std::map<GInt, GInt>> _qr; 

    std::map<std::array<UInt, N>, GInt> &r = _r;
    for (auto it = m_ord.begin(), ed = m_ord.end(); it != ed; ){
        auto a = *it;
        for (unsigned i = m+1; i < N; ++i) a[i] = 0;
        auto b = a;
        a[m]++;
        auto it1 = std::lower_bound(std::next(it), ed, a, compare_arr);
        GInt lsize = it1 - it;
        r.emplace(b, lsize);
        it = it1;
    }

    if (m > 0){
        std::array<std::map<std::array<UInt, N>, GInt>, N> ww;
        ww[m] = r;
        for (int i = m-2; i >= -1; --i){
            for (auto it = ww[i+2].begin(); it != ww[i+2].end(); ++it){
                auto a = it->first;
                a[i+1] = 0;
                auto jt = ww[i+1].find(a);
                if (jt != ww[i+1].end())
                    jt->second += it->second;
                else
                    ww[i+1].emplace(a, it->second);
            }
        }
        _q = (n < m) ? ww[n] : r;
        if (n < m){
            _p = sum_part_ids(n, m, m, ww[m]);
        }
        
        auto l = std::min(m, n);
        if (l > 0){
            auto x = sum_part_ids(0, l, m, ww[l]);
            for (auto it = x.begin(); it != x.end(); ++it){
                auto v = it->first;
                auto rg = rg_size(v[m]);
                auto val = rg * it->second;
                v[m] = 0;
                auto jt = _w.find(v);
                if (jt != _w.end())
                    jt->second += val;
                else
                    _w.emplace(v, val);
            }
        }
    } else {
        _q = r;
    }

    m_size = 0;
    for (auto& v: _q)
        m_size += rg_size(v.first[m]) * v.second;
    
    if (n <= m && !_q.empty()){
        if (n > 0){
            std::map<std::array<UInt, N>, decltype(_q.begin())> st;
            auto check = make_check(n);
            auto v = _q.begin()->first; v[m] = 0;
            st.emplace(v, _q.begin());
            for (auto it = _q.begin(); it != _q.end(); ++it){
                if (!check(v, it->first)){
                    v = it->first;
                    st.emplace(v, it);
                }
            }
            v[n-1]++;
            st.emplace(v, _q.end());
            for (auto it = st.begin(), jt = std::next(st.begin()); jt != st.end(); jt++){
                std::set<GInt> rg;
                for (auto kt = it->second; kt != jt->second; ++kt)
                    rg.insert(rg_size(kt->first[m]));
                std::map<GInt, GInt> beta_sz;    
                for (auto v: rg){
                    GInt sum = 0;
                    for (auto kt = it->second; kt != jt->second; ++kt){
                        auto lrg = rg_size(kt->first[m]);
                        sum += std::min(lrg, v) * kt->second;
                    }
                    beta_sz.emplace(v, sum);
                }
                auto lid = it->first;
                std::fill(lid.data() + n, lid.data() + lid.size(), 0);    
                _qr.emplace(lid, std::move(beta_sz));   

                it = jt;   
            }
        } else {
            std::set<GInt> rg;
            for (auto kt = _q.begin(); kt != _q.end(); ++kt)
                rg.insert(rg_size(kt->first[m]));
            std::map<GInt, GInt> beta_sz;    
            for (auto v: rg){
                GInt sum = 0;
                for (auto kt = _q.begin(); kt != _q.end(); ++kt){
                    auto lrg = rg_size(kt->first[m]);
                    sum += std::min(lrg, v) * kt->second;
                }
                beta_sz.emplace(v, sum);
            } 
            std::array<UInt, N> id{0};   
            _qr.emplace(id, std::move(beta_sz));
        }
    }  

    m_r.resize(0); m_w.resize(0); m_p.resize(0); m_q.resize(0);
    m_qr.resize(0); m_qrb.resize(0);
    m_r.reserve(_r.size()); m_w.reserve(_w.size()); m_p.reserve(_p.size()); m_q.reserve(_q.size());
    m_qr.reserve(_qr.size());
    std::size_t _qrb_sz = 0;
    for (auto& v: _qr) _qrb_sz += v.second.size();
    m_qrb.reserve(_qrb_sz);

    for (auto it = _r.begin(); it != _r.end(); ++it)  m_r.push_back(*it);
    for (auto it = _w.begin(); it != _w.end(); ++it)  m_w.push_back(*it);
    for (auto it = _p.begin(); it != _p.end(); ++it)  m_p.push_back(*it);
    for (auto it = _q.begin(); it != _q.end(); ++it)  m_q.push_back(*it);
    for (auto it = _qr.begin(); it != _qr.end(); ++it)  {
        auto st = m_qrb.size();
        for (auto jt = it->second.begin(); jt != it->second.end(); ++jt)
            m_qrb.push_back(*jt);
        auto ed = m_qrb.size();   
        m_qr.push_back({it->first, std::pair<std::size_t, std::size_t>{st, ed}});
    }

    return *this;
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
GInt ext_range_array_enum<N, RangeFunc, UInt, GInt>::operator[](const std::array<UInt, N+1>& i) const {
    std::array<UInt, N> id;
    std::copy(i.data(), i.data() + insert_pos, id.data());
    std::copy(i.data() + insert_pos+1, i.data() + i.size(), id.data()+insert_pos);
    GInt s = i[insert_pos];
    UInt n = insert_pos, m = depends_on;
    assert(s < rg_size(id[m]) && "Wrong ext index");
    assert(std::find(m_ord.begin(), m_ord.end(), id) != m_ord.end() && "Wrong base index");

    auto make_check = [](UInt t){
        return [t](const std::array<UInt, N>& a, const std::array<UInt, N>& b){
            for (unsigned i = 0; i < t; ++i)
                if (a[i] != b[i]) return false;
            return true;    
        };
    };
    auto compare_arr = [](const auto& a, const auto& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
    auto compare_arr_key = [compare_arr](const auto& a, const auto& b) { return compare_arr(a.first, b); };

    std::array<UInt, N> k_id = id;
    auto it_id = std::find(m_ord.begin(), m_ord.end(), id);
    auto it_kid = it_id;
    GInt kn = 0;
    if (n != id.size()){
        std::fill(k_id.data() + ((n > m) ? n : m+1), k_id.data() + k_id.size(), 0);
        it_kid = std::lower_bound(m_ord.begin(), std::next(it_id), k_id, compare_arr);
        kn = it_id - it_kid;
    }

    GInt pn = 0;
    if (n > m){
        std::array<UInt, N> p_id = k_id; p_id[n-1]++;
        auto it_pid = std::lower_bound(std::next(it_id), m_ord.end(), p_id, compare_arr);
        pn = s * (it_pid - it_kid);
    } else if (n + 1 <= m) {
        std::array<UInt, N> p_id = k_id; p_id[m] = 0;
        auto check = make_check(m);
        auto it_p = std::lower_bound(m_p.begin(), m_p.end(), p_id, compare_arr_key);
        for (auto it = it_p; it != m_p.end() && check(it->first, p_id); ++it)
            pn += (s < rg_size(it->first[m])) ? it->second : 0;
    }

    GInt qn = 0;
    if (n > m + 1){
        std::array<UInt, N> q_id{0};
        std::copy(k_id.data(), k_id.data() + m+1, q_id.data());
        auto rg = rg_size(id[m]);
        auto it_qid = std::lower_bound(m_ord.begin(), std::next(it_kid), q_id, compare_arr);

        qn = rg * (it_kid - it_qid); 
    } else if (n <= m && s != 0) {
        std::array<UInt, N> q_id{0};
        std::copy(k_id.data(), k_id.data() + n, q_id.data());
        auto check = make_check(m);
        auto it_q = std::lower_bound(m_q.begin(), m_q.end(), q_id, compare_arr_key); 
        for(auto it = it_q; it != m_q.end() && check(it->first, q_id); ++it){
            auto rg = rg_size(it->first[m]);
            GInt scl = std::min(s, rg);
            
            qn += scl * it->second;
        }
    }

    GInt rn = 0;
    auto l = std::min(m, n);
    std::array<UInt, N> r_id{0};
    std::copy(k_id.data(), k_id.data() + m, r_id.data());
    auto check = make_check(m);
    auto it_r = std::lower_bound(m_r.begin(), m_r.end(), r_id, compare_arr_key); 
    for(auto rit_kid = it_r; rit_kid != m_r.end() && check(rit_kid->first, k_id) && rit_kid->first[m] < id[m]; ++rit_kid){
        auto rg = rg_size(rit_kid->first[m]);
        auto scl = rg;
        if (n <= m) 
            scl = (s < rg) ? 1 : 0;
        rn += scl * rit_kid->second;
    }
    
    GInt wn = 0;
    if (l > 0){
        std::array<UInt, N> w_id{0};
        std::copy(k_id.data(), k_id.data() + l, w_id.data());
        auto it_w = std::lower_bound(m_w.begin(), m_w.end(), w_id, compare_arr_key);
        wn = (it_w != m_w.end() && it_w->first == w_id) ? it_w->second : 0;
    }

    return wn + rn + qn + pn + kn;
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
typename ext_range_array_enum<N, RangeFunc, UInt, GInt>::iterator 
    ext_range_array_enum<N, RangeFunc, UInt, GInt>::begin(GInt i) const{
    auto make_check = [](UInt t){
        return [t](const std::array<UInt, N>& a, const std::array<UInt, N>& b){
            for (unsigned i = 0; i < t; ++i)
                if (a[i] != b[i]) return false;
            return true;    
        };
    };
    auto compare_arr = [](const auto& a, const auto& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
    auto compare_arr_key = [compare_arr](const auto& a, const auto& b) { return compare_arr(a.first, b); };

    iterator x;
    x.ptr = this;
    x.id.first = i;
    if (i >= m_size)
        return x;

    UInt n = insert_pos, m = depends_on;
    auto l = std::min(n, m);
    std::array<UInt, N> res{0}; UInt beta = 0;
    if (l > 0){
        auto wit = std::lower_bound(m_w.rbegin(), m_w.rend(), i, [](const auto& a, GInt i) { return a.second > i; });
        std::array<UInt, N> v;
        // if (wit == m_w.rend()){
        //     v = m_ord.begin()->first;
        //     throw std::runtime_error("Something wrong");
        // } else 
        {
            v = wit->first;
            i -= wit->second;
            x.wit = std::prev(wit.base());
        }
        std::copy(v.data(), v.data() + l, res.data());    
    }
    if (n > m){
        auto check = make_check(m);
        auto sum = 0;
        auto it_r = std::lower_bound(m_r.begin(), m_r.end(), res, compare_arr_key);
        for (auto it = it_r; it != m_r.end() && check(it->first, res); ++it){
            auto rg = rg_size(it->first[m]);
            if (sum + rg*it->second > i){
                i -= sum;
                res[m] = it->first[m];
                x.rit = std::next(it);
                break;
            } else {
                sum += rg*it->second;
                res[m] = it->first[m];
            }    
        }
        auto it = std::lower_bound(m_ord.begin(), m_ord.end(), res, compare_arr);
        x.qit.h = it;
        auto tid = res; tid[m]++;
        auto jt = std::lower_bound(std::next(it), m_ord.end(), tid, compare_arr);
        if (n > m + 1){
            auto lrg = rg_size((*it)[m]);
            auto _pit = it + i/lrg;
            auto v = *_pit;
            std::fill(v.data() + n, v.data() + v.size(), 0);
            auto pit = std::lower_bound(it, _pit, v, compare_arr);
            i -= lrg*(pit - it);
            std::copy(v.data() + m + 1, v.data() + n, res.data() + m+1);
            it = pit; 
            tid = res; tid[n-1]++;
            jt = std::lower_bound(std::next(it), jt, tid, compare_arr);
            x.pit = jt;
        } 
        auto sh = jt - it;
        beta = i / sh;
        i = i%sh;
        x.kst = it;
        auto kit = std::next(it, i);
        x.ked = kit;
        auto v = *kit;
        std::copy(v.data() + n, v.data() + v.size(), res.data() + n);
    } else {
        auto m_qrit = std::lower_bound(m_qr.begin(), m_qr.end(), res, compare_arr_key);
        x.qit.q = m_qrit;
        auto qr_rbegin = std::make_reverse_iterator(m_qrb.begin() + m_qrit->second.second);
        auto qr_rend = std::make_reverse_iterator(m_qrb.begin() + m_qrit->second.first);
        auto qr_it = std::lower_bound(qr_rbegin, qr_rend, i, [](auto a, GInt i){ return a.second > i; });
        auto nqr_it = std::prev(qr_it);
        auto dif = nqr_it->second - 0;
        auto rdif = nqr_it->first - 0;
        if (qr_it != qr_rend){
            beta += qr_it->first;
            i -= qr_it->second;
            dif -= qr_it->second;
            rdif -= qr_it->first;
        }
        auto w = dif / rdif;
        beta += i / w;
        i = i%w;

        GInt pn = 0;
        if (n + 1 <= m) {
            std::array<UInt, N> p_id = res;
            auto check_st = make_check(n);
            auto check_ed = make_check(m);
            auto it_pid = std::lower_bound(m_p.begin(), m_p.end(), p_id, compare_arr_key);
            p_id = it_pid->first;
            std::array<UInt, N> prev_p_id = p_id;
            GInt sum = (beta < rg_size(it_pid->first[m])) ? it_pid->second : 0; 
            GInt prev_sum = 0;
            bool finded = false;
            auto change_it = it_pid, change_it_prev = it_pid;
            auto it = std::next(it_pid);
            for (; it != m_p.end() && check_st(it->first, p_id); ++it){
                bool same_ed = check_ed(p_id, it->first);                    
                if (same_ed){
                    sum += (beta < rg_size(it->first[m])) ? it->second : 0;
                } else {
                    prev_sum = sum;
                    sum = (beta < rg_size(it->first[m])) ? it->second : 0;
                    prev_p_id = p_id;
                    p_id = it->first;
                    change_it_prev = change_it;
                    change_it = it;
                }
                if (sum > i){
                    std::copy(prev_p_id.data() + n, prev_p_id.data() + m, res.data() + n);
                    pn = prev_sum;
                    finded = true;
                    break;
                }
            }
            if (!finded){
                std::copy(p_id.data() + n, p_id.data() + m, res.data() + n);
                pn = sum;
            }
        }
        i -= pn;

        auto check = make_check(m);
        auto sum = 0;
        auto it_r = std::lower_bound(m_r.begin(), m_r.end(), res, compare_arr_key);
        for (auto it = it_r; it != m_r.end() && check(it->first, res); ++it){
            auto rg = (beta < rg_size(it->first[m])) ? 1 : 0;
            if (sum + rg*it->second > i){
                i -= sum;
                res[m] = it->first[m];
                x.rit = std::next(it);
                break;
            } else {
                sum += rg*it->second;
                res[m] = it->first[m];
            }    
        }
        if (m + 1 < N){
            auto it = std::lower_bound(m_ord.begin(), m_ord.end(), res, compare_arr);
            auto tid = res; tid[m]++;
            // auto jt = std::lower_bound(std::next(it), m_ord.end(), res, compare_arr);
            // auto sh = jt - it;
            auto kit = std::next(it, i);
            auto v = *kit;
            x.kst = it; x.ked = kit;
            std::copy(std::next(v.begin(), m+1), v.end(), std::next(res.begin(),m+1));
        } else {
            x.ked = x.kst = std::lower_bound(m_ord.begin(), m_ord.end(), res, compare_arr);
        }
    }

    std::array<UInt, N+1> id;
    std::copy(res.data(), res.data() + n, id.data());
    std::copy(res.data()+n, res.data() + res.size(), id.data()+n+1);
    id[n] = beta;
    x.id.second = id;

    return x;
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
std::array<UInt, N+1> ext_range_array_enum<N, RangeFunc, UInt, GInt>::operator[](GInt i) const{ 
    assert(i < m_size && "Index out of range");
    return begin(i).id.second; 
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
void ext_range_array_enum<N, RangeFunc, UInt, GInt>::debug_print_state(std::ostream& out) const {
    auto print_arr = [&out](auto a) mutable -> std::ostream&{ 
        out << "[" << a.size() << "]{";
        if (!a.empty())
            out << a[0];
        for (std::size_t i = 1; i < a.size(); ++i)
            out << ", " << a[i];    
        out << "}";
        return out;
    };
    std::set<UInt> depends_var;
    for (auto& a: m_ord)
        depends_var.insert(a[depends_on]);

    out << "internal_order: \n";
    for (auto& a: m_ord)
        {out << "\t";  print_arr(a) << " : " << (&a-m_ord.data()) << "\n";}
    out << "\n";
    out << "range_func:" << " insert_pos(n) = " << insert_pos << " depends_on(m) = " << depends_on << "\n\t";
    for (auto a: depends_var)
        out << a << " -> " << rg_size(a) << "; "; 
    out << "\n";
    out << "\n";

    out << "r[";
    for (int i = 0; i < depends_on+1; ++i)
        out << i << ((i+1 != depends_on + 1) ? ", " : "");
    out << "]";       
    out << ":\n";
    for (auto& a: m_r)
        {out << "\t";  print_arr(a.first) << " : " << a.second << "\n"; }
    out << "\n";  

    auto l = std::min(insert_pos, depends_on);   
    out << "w[";
    if (l > 0)
        for (unsigned i = 0; i < l; ++i)
            out << i << ((i+1 != l) ? ", " : "");
    out << "]";        
    out << ":\n";
    for (auto& a: m_w)
        {out << "\t";  print_arr(a.first) << " : " << a.second << "\n"; }
    out << "\n";

    if (insert_pos <= depends_on){
        if (insert_pos < depends_on){
            out << "p[";
            for (int i = 0; i < depends_on+1; ++i)
                out << i << ((i+1 != depends_on+1) ? ", " : "");
            out << "]";       
            out << ":\n";
            for (auto& a: m_p)
                {out << "\t";  print_arr(a.first) << " : " << a.second << "\n"; }
            out << "\n";
        } 

        out << "q[";
        if (insert_pos > 0)
            for (unsigned i = 0; i < insert_pos; ++i)
                out << i << ", ";
        if (depends_on != insert_pos) out << depends_on; 
        out << "]";       
        out << ":\n";
        for (auto& a: m_q)
            {out << "\t";  print_arr(a.first) << " : " << a.second << "\n"; }
        out << "\n";

        out << "m_qr[";
        if (insert_pos > 0)
            for (unsigned i = 0; i < insert_pos; ++i)
                out << i << ", ";
        out << "]:\n";
        for (const auto& a: m_qr){
            out << "\t";  print_arr(a.first);
            out << " <-> {";
            for (auto vi = a.second.first; vi != a.second.second; ++vi)
                out << m_qrb[vi].first << " : " << m_qrb[vi].second << ", ";
            out << "}";
        }
        out << "\n";
    } 

    out << std::endl;              
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
typename ext_range_array_enum<N, RangeFunc, UInt, GInt>::iterator 
    ext_range_array_enum<N, RangeFunc, UInt, GInt>::next_it(iterator a) const {
    auto make_check = [](UInt t){
        return [t](const std::array<UInt, N>& a, const std::array<UInt, N>& b){
            for (UInt i = 0; i < t; ++i)
                if (a[i] != b[i]) return false;
            return true;    
        };
    };
    auto compare_arr = [](const auto& a, const auto& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
    auto compare_arr_key = [compare_arr](const auto& a, const auto& b) { return compare_arr(a.first, b); };

    iterator x = a;
    x.id.first = a.id.first+1;
    if (x.id.first >= m_size)
        return x;
    auto& res = x.id.second;
    GInt beta = x.id.second[insert_pos];
    std::array<UInt, N> id; 
    std::copy(res.data(), res.data() + insert_pos, id.data());
    std::copy(res.data() + insert_pos+1, res.data() + res.size(), id.data()+insert_pos);

    UInt n = insert_pos, m = depends_on;
    auto t = std::max(n, m+1);
    if (t < N){
        auto kit_next = std::next(a.ked);
        if (kit_next != m_ord.end() && make_check(t)(id, *kit_next)){
            x.ked = kit_next;
            auto v = *kit_next;
            std::copy(v.data() + t, v.data()+v.size(), res.data() + t+1);
            return x;
        }
    }
    if (n > m){
        auto lrg = rg_size(id[m]);
        if (beta + 1 < lrg){
            res[n]++;
            x.ked = x.kst;
            auto v = *x.kst;
            std::copy(v.data() + n, v.data() + v.size(), res.data() + n + 1);
            return x;
        }
        res[n] = 0;
        if (n > m + 1){
            auto pit_next = x.pit;
            if (pit_next != m_ord.end() && make_check(m+1)(id, *pit_next)){
                auto pid_next = *pit_next;
                pid_next[n-1]++;
                std::fill(pid_next.data()+n, pid_next.data()+pid_next.size(), 0);
                x.pit = std::lower_bound(pit_next, m_ord.end(), pid_next, compare_arr);
                x.kst = x.ked = pit_next;
                auto v = *x.kst;
                std::copy(std::next(v.begin(), m+1), std::next(v.begin(), n), res.data() + m+1);
                std::copy(v.data()+n, v.data()+v.size(), res.data() + n+1);
                return x;
            }
        }
        auto rit = x.rit;
        if (rit != m_r.end() && make_check(m)(id, rit->first)){
            x.rit = std::next(rit);
            auto v = id; 
            v[m] = rit->first[m];
            res[m] = rit->first[m];
            if (m + 1 < v.size())
                std::fill(v.data() + m + 1, v.data() + v.size(), 0);
            auto st = std::lower_bound(x.ked, m_ord.end(), v, compare_arr);
            auto vv = *st;
            std::copy(vv.data() + m + 1, vv.data() + n, res.data() + m + 1);
            std::copy(vv.data() + n, vv.data() + vv.size(), res.data() + n + 1);
            x.ked = x.kst = x.qit.h = st;
            if (n > m + 1){
                v = vv; v[n-1]++;
                std::fill(v.data() + n, v.data() + v.size(), 0);
                x.pit = std::lower_bound(std::next(x.ked), m_ord.end(), v, compare_arr);
            }
            return x;
        }
        if (m > 0){
            auto wit = x.wit;
            if (wit != m_w.end()){
                x.wit = std::next(wit);
                auto v = x.wit->first;
                std::fill(v.data() + m, v.data() + v.size(), 0);
                auto st = std::lower_bound(std::next(x.ked), m_ord.end(), v, compare_arr);
                v[m] = (*st)[m];
                x.rit = std::next(std::lower_bound(x.rit, m_r.end(), v, compare_arr_key));
                auto vv = *st;
                std::copy(vv.data() + 0, vv.data() + n, res.data() + 0);
                std::copy(vv.data() + n, vv.data() + vv.size(), res.data() + n + 1);
                x.ked = x.kst = x.qit.h = st;
                if (n > m + 1){
                    v = vv; v[n-1]++;
                    std::fill(v.data() + n, v.data() + v.size(), 0);
                    x.pit = std::lower_bound(std::next(x.ked), m_ord.end(), v, compare_arr);
                }
                return x;
            }
        }
    } else {
        auto check_st = make_check(n);
        // auto check_ed = make_check(m);
        for (auto rit = x.rit; rit != m_r.end() && check_st(id, rit->first); ++rit){
            auto lrg = rg_size(rit->first[m]);
            if (beta < lrg){
                auto v = rit->first;
                x.rit = std::next(rit);
                std::copy(v.data() + n, v.data() + m+1, res.data() + n+1);
                std::fill(v.data() + m+1, v.data() + v.size(), 0);
                auto st = std::lower_bound(std::next(x.ked), m_ord.end(), v, compare_arr);
                auto vv = *st;
                std::copy(vv.data() + m + 1, vv.data() + vv.size(), res.data() + m + 2);
                x.ked = x.kst = st;
                return x;
            }
        }
        if (x.qit.q != m_qr.end() && x.qit.q->second.first != x.qit.q->second.second 
                && ((beta+1) < m_qrb[x.qit.q->second.second-1].first)){
            res[n]++;
            beta++;
            auto v = id;
            std::fill(v.data() + n, v.data() + v.size(), 0);
            auto rit = std::lower_bound(m_r.begin(), m_r.end(), v, compare_arr_key);
            for (; rit != m_r.end() && check_st(v, rit->first) && (beta >= rg_size(rit->first[m])); ++rit);
            std::copy(rit->first.data() + n, rit->first.data() + m + 1, v.data() + n);
            x.rit = std::next(rit);
            auto st = std::lower_bound(m_ord.begin(), m_ord.end(), v, compare_arr);
            v = *st;
            x.ked = x.kst = st;
            std::copy(v.data() + n, v.data() + v.size(), res.data() + n + 1);
            return x;
        }
        if (n > 0){
            auto wit = x.wit;
            if (wit != m_w.end()){
                x.wit = std::next(wit);
                auto v = x.wit->first;
                std::fill(v.data() + n, v.data() + v.size(), 0);
                x.qit.q = std::lower_bound(m_qr.begin(), m_qr.end(), v, compare_arr_key);
                auto st = std::lower_bound(x.ked, m_ord.end(), v, compare_arr);
                x.ked = x.kst = st;
                v = *st;
                std::copy(v.data(), v.data() + n, res.data());
                res[n] = 0;
                std::copy(v.data()+n, v.data() + v.size(), res.data()+n+1);
                std::fill(v.data() + n, v.data() + v.size(), 0);
                x.rit = std::next(std::lower_bound(m_r.begin(), m_r.end(), v, compare_arr_key));
                return x;
            }
        }
    }

    assert("Reached unreacheable code");
    return x; 
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
void ext_range_array_enum<N, RangeFunc, UInt, GInt>::_setBaseSet(std::vector<std::array<UInt, N>> from_set){
    std::sort(from_set.begin(), from_set.end(), [](const auto& a, const auto& b){ return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); });
    from_set.erase( std::unique( from_set.begin(), from_set.end() ), from_set.end() );
    m_ord = std::move(from_set);  
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
bool ext_range_array_enum<N, RangeFunc, UInt, GInt>::contains(const std::array<UInt, N+1>& i) const{
    auto compare_arr = [](const auto& a, const auto& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
    std::array<UInt, N> id;
    std::copy(i.data(), i.data() + insert_pos, id.data());
    std::copy(i.data() + insert_pos+1, i.data() + i.size(), id.data()+insert_pos);
    GInt beta = i[insert_pos];
    if (!(beta < rg_size(id[depends_on])))
        return false;
    auto it = std::lower_bound(m_ord.begin(), m_ord.end(), id, compare_arr);
    return it != m_ord.end() && *it == id; 
}

template<unsigned N, typename RangeFunc, typename UInt, typename GInt>
typename ext_range_array_enum<N, RangeFunc, UInt, GInt>::iterator ext_range_array_enum<N, RangeFunc, UInt, GInt>::lower_bound(const std::array<UInt, N+1>& i) const{
    auto make_check = [](UInt t){
        return [t](const std::array<UInt, N>& a, const std::array<UInt, N>& b){
            for (unsigned i = 0; i < t; ++i)
                if (a[i] != b[i]) return false;
            return true;    
        };
    };
    auto compare_arr = [](const auto& a, const auto& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); };
    auto compare_arr_key = [compare_arr](const auto& a, const auto& b) { return compare_arr(a.first, b); };

    auto res = i;
    auto beta = [&id = res, n = insert_pos]() -> UInt& { return id[n]; };
    auto alpha = [&id = res, n = insert_pos](UInt k) -> UInt& { return id[k + (k >= n ? 1 : 0)]; };
    UInt n = insert_pos, m = depends_on;
    std::array<UInt, N> id;
    std::copy(i.data(), i.data() + insert_pos, id.data());
    std::copy(i.data() + insert_pos+1, i.data() + i.size(), id.data()+insert_pos);
    
    auto jt = m_ord.begin();
    if (n > 0){
        auto jd = id;
        std::fill(jd.begin() + n, jd.end(), 0);
        jt = std::lower_bound(m_ord.begin(), m_ord.end(), jd, compare_arr);
        if (jt == m_ord.end())
            return end(); 
        if (!make_check(n)(id, *jt)){  
            std::copy(jt->begin(), jt->begin()+n, res.begin()); 
            beta() = 0;
            std::copy(jt->begin()+n, jt->end(), res.begin()+n+1);
            return begin(this->operator[](res));
        }      
    }
    if (n > m){
        if (beta() >= rg_size(id[m])){
            auto jd = id;
            std::fill(jd.begin() + n, jd.end(), 0);
            jd[n-1]++;
            jt = std::lower_bound(jt, m_ord.end(), jd, compare_arr);
            if (jt == m_ord.end())
                return end();
            beta() = 0;
            for (unsigned k = 0; k < N; ++k)
                alpha(k) = (*jt)[k];
            return begin(this->operator[](res));
        }
        auto it = std::lower_bound(jt, m_ord.end(), id, compare_arr);
        if (it != m_ord.end() && make_check(n)(id, *it)){
            std::copy(it->begin() + n, it->end(), res.begin() + n + 1);
            return begin(this->operator[](res));
        } else if (beta()+1 < rg_size(id[m])){
            std::copy(jt->begin() + n, jt->end(), res.begin() + n + 1);
            ++beta();
            return begin(this->operator[](res));
        } else {
            if (it == m_ord.end()) return end();
            std::copy(it->begin(), it->begin()+n, res.begin()); 
            beta() = 0;
            std::copy(it->begin()+n, it->end(), res.begin()+n+1);
            return begin(this->operator[](res));
        }
    } else {
        auto set_next_start = [&](){
            if (n == 0) return end();
            auto iid = id;
            std::fill(iid.begin() + n, iid.end(), 0);
            iid[n-1]++;
            auto iit = std::lower_bound(jt, m_ord.end(), iid, compare_arr);
            if (iit == m_ord.end())
                return end();
            beta() = 0;  
            for (unsigned k = 0; k < N; ++k)
                alpha(k) = (*iit)[k];
            return begin(this->operator[](res));  
        };

        auto jd = id;
        std::fill(jd.begin() + n, jd.end(), 0);
        auto it_qr = std::lower_bound(m_qr.begin(), m_qr.end(), jd, compare_arr_key);
        auto b = m_qrb[it_qr->second.second - 1].first;
        
        if (beta() >= b)
            return set_next_start();

        auto it_r = std::lower_bound(m_r.begin(), m_r.end(), jd, compare_arr_key);
        auto check_st = make_check(n);
        auto check_ed = make_check(m+1);
        auto it_add = m_r.end();
        jd = id; 
        std::fill(jd.begin() + m+1, jd.end(), 0);
        for (auto it = it_r; it != m_r.end() && check_st(it->first, jd); ++it){
            if (beta() >= rg_size(it->first[m])) continue;
            if (!compare_arr(it->first, jd)){
                auto kt = std::lower_bound(jt, m_ord.end(), (it->first == jd) ? id : it->first, compare_arr);
                if(check_ed(*kt, it->first)){
                    for (unsigned k = 0; k < N; ++k)
                        alpha(k) = (*kt)[k];
                    return begin(this->operator[](res));    
                }
            }
            if (it_add == m_r.end() && beta()+1 < rg_size(it->first[m]))
                it_add = it;
        }
        if (it_add != m_r.end()){
            beta()++;
            auto kt = std::lower_bound(jt, m_ord.end(), it_add->first, compare_arr);
            for (unsigned k = 0; k < N; ++k)
                alpha(k) = (*kt)[k];
            return begin(this->operator[](res));
        }

        return set_next_start();
    }
}

template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
typename ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator& ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator::operator ++() {
    auto n = ptr->m_arr_enum.insert_pos;
    auto m = ptr->m_arr_enum.depends_on + ((ptr->m_arr_enum.insert_pos <= ptr->m_arr_enum.depends_on) ? 1 : 0);
    auto x = it2->second[m];
    ++it2;
    if (it2 == ptr->m_arr_enum.end())
        return *this; 
    
    auto v2 = *it2;
    auto v1 = *it1;
    if (v2.second[m] == x && v2.second[n] == static_cast<UInt>(v1.first + 1))
        ++it1;
    else if (v2.second[m] != x || v2.second[n] != static_cast<UInt>(v1.first))
        it1 = ptr->m_int_enum[v2.second[m]].find_left(v2.second[n]);
    val = std::move(v2);
    val.second[n] = it1->second;
    return *this;
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
typename ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator& ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator::operator --(){
    auto n = ptr->m_arr_enum.insert_pos;
    auto m = ptr->m_arr_enum.depends_on + ((ptr->m_arr_enum.insert_pos <= ptr->m_arr_enum.depends_on) ? 1 : 0);
    auto x = (it2 != m_arr_enum.end()) ? it2->second[m] : UInt(-1);
    --it2;
    auto v2 = *it2;
    auto v1 = *it1;
    if (v2.second[m] == x && v2.second[n] + 1 == static_cast<UInt>(v1.first))
        --it1;
    else if (v2.second[m] != x || v2.second[n] != static_cast<UInt>(v1.first))
        it1 = ptr->m_int_enum[v2.second[m]].find_left(v2.second[n]);
    val = std::move(v2);
    val.second[n] = it1->second;
    return *this;   
}
template <unsigned N, unsigned NV, typename T1, typename T2, typename T3, typename T4>
typename ext_array_enum<N, NV, T1, T2, T3, T4>::iterator& ext_array_enum<N, NV, T1, T2, T3, T4>::iterator::operator +=(difference_type n) { 
    auto l = ptr->m_arr_enum.insert_pos;
    auto m = ptr->m_arr_enum.depends_on + ((ptr->m_arr_enum.insert_pos <= ptr->m_arr_enum.depends_on) ? 1 : 0);
    it2 += n;
    auto v2 = *it2;
    it1 = ptr->m_int_enum[v2.second[m]].find_left(v2.second[l]);
    val = std::move(v2);
    val.second[l] = it1->second;
    return *this;
}
template <unsigned N, unsigned NV, typename T1, typename T2, typename T3, typename T4>
typename ext_array_enum<N, NV, T1, T2, T3, T4>::iterator& ext_array_enum<N, NV, T1, T2, T3, T4>::iterator::operator -=(difference_type n) { 
    auto l = ptr->m_arr_enum.insert_pos;
    auto m = ptr->m_arr_enum.depends_on + ((ptr->m_arr_enum.insert_pos <= ptr->m_arr_enum.depends_on) ? 1 : 0);
    it2 -= n;
    auto v2 = *it2;
    it1 = ptr->m_int_enum[v2.second[m]].find_left(v2.second[l]);
    val = std::move(v2);
    val.second[l] = it1->second;
    return *this;
}

template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
bool ext_array_enum<N, NV, T1, UInt, GInt, T4>::contains(const std::array<UInt, N+1>& i) const { 
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);
    if (i[m] >= NV) return false;
    auto it1 = m_int_enum[i[m]].find_right(i[n]); 
    if (it1 == m_int_enum[i[m]].end()) return false;
    std::array<UInt, N+1> j;
    j[n] = it1->first;
    return m_arr_enum.contains(j);
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
typename ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator ext_array_enum<N, NV, T1, UInt, GInt, T4>::begin(GInt i) const {
    if (empty()) return end();
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);

    auto it2 = m_arr_enum.begin(i);
    auto v2 = *it2;
    typename IntervalEnum::const_iterator it1;
    if (i == 0 && v2.second[n] == 0)
        it1 = m_int_enum[v2.second[m]].begin();
    else    
        it1 = m_int_enum[v2.second[m]].find_left(v2.second[n]); 
    return iterator(this, it1, it2);      
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
typename ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator ext_array_enum<N, NV, T1, UInt, GInt, T4>::find(GInt i) const { 
    auto it2 = m_arr_enum.find(i);
    if (it2 == m_arr_enum.end()) return end();
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);
    auto v2 = *it2;
    auto it1 = m_int_enum[v2.second[m]].find_left(v2.second[n]);
    return iterator(this, it1, it2);  
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
typename ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator ext_array_enum<N, NV, T1, UInt, GInt, T4>::find(const std::array<UInt, N+1>& i) const { 
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);
    if (i[m] >= NV) return end();
    auto it1 = m_int_enum[i[m]].find_right(i[n]);
    if (it1 == m_int_enum[i[m]].end()) return end();
    auto j = i;
    j[n] = it1->first;
    return iterator(this, it1, m_arr_enum.find(j));
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
typename ext_array_enum<N, NV, T1, UInt, GInt, T4>::iterator ext_array_enum<N, NV, T1, UInt, GInt, T4>::lower_bound(const std::array<UInt, N+1>& i) const{
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);
    auto j = i;
    if (m < n){
        auto it1 = m_int_enum[i[m]].lower_bound_right(i[n]);
        j[n] = (it1 == m_int_enum.end()) ? std::numeric_limits<UInt>::max() : UInt(it1->first);
    } else {
        int l = std::numeric_limits<int>::max();
        typename IntervalEnum::const_iterator it1;
        for (unsigned r = 0; r < NV; ++r){
            auto lit = m_int_enum[r].lower_bound_right(i[n]);
            if (lit != m_int_enum[r].end() && m_int_enum[r]->first < l){
                l = m_int_enum[r]->first;
                it1 = lit;
            }
        }
        j[n] = (l != std::numeric_limits<int>::max()) ? l : std::numeric_limits<UInt>::max();
    }
    auto it2 = m_arr_enum.lower_bound(j);
    if (it2 == m_arr_enum.end()) return end();
    auto _it1 = m_int_enum[it2->second[m]].find_left(it2->second[n]);
    return iterator(this, _it1, it2);
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
GInt ext_array_enum<N, NV, T1, UInt, GInt, T4>::operator[](const std::array<UInt, N+1>& i) const { 
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);
    assert(i[m] < NV && "Depensive index out of range");
    auto it1 = m_int_enum[i[m]].find_right(i[n]);
    assert(it1 != m_int_enum[i[m]].end() && "Index out of range");
    auto j = i;
    j[n] = it1->first;
    return m_arr_enum[j];
}
template <unsigned N, unsigned NV, typename T1, typename UInt, typename GInt, typename T4>
std::array<UInt, N+1> ext_array_enum<N, NV, T1, UInt, GInt, T4>::operator[](GInt i) const{
    auto n = m_arr_enum.insert_pos;
    auto m = m_arr_enum.depends_on + ((m_arr_enum.insert_pos <= m_arr_enum.depends_on) ? 1 : 0);

    auto j = m_arr_enum[i];
    auto it1 = m_int_enum[j[m]].find_left(j[n]);
    assert(it1 != m_int_enum[j[m]].end() && "Index out of range");
    j[n] = it1->second;
    return j;
}


}

#endif //CARNUM_EXT_ARRAY_ENUM_INL