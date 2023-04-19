namespace Ani{
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::reshape(std::size_t sz1, std::size_t sz2) {
        if (sz1 * sz2 != m_nrows * m_ncols)
            throw std::runtime_error("Reshaping can't be performed for such sizes");
        m_nrows = sz1,  m_ncols = sz2; 
        return *this;  
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_val(const BaseExpr& val, std::size_t i, std::size_t j)  { 
        auto& o = *static_cast<const NumericExpr<REAL>*>(&val);
        m_v.resize(1);
        m_nrows = 1, m_ncols = 1;
        m_v[0] = o(i, j);
        return *this;  
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_sub_matrix(const BaseExpr& val, const std::size_t* i, std::size_t i_arr_sz, const std::size_t* j, long j_arr_sz)  {
        auto& o = *static_cast<const NumericExpr<REAL>*>(&val);
        std::vector<REAL> tmp;
        std::vector<REAL>* work = (&val == this) ? &tmp : &m_v;
        work->resize(i_arr_sz*j_arr_sz);
        for (int jk = 0; jk < j_arr_sz; ++jk)
        for (int ik = 0; ik < i_arr_sz; ++ik)
            (*work)[ik + jk*j_arr_sz] = o(i[ik], j[jk]);
        m_v = std::move(*work);    
        m_nrows = i_arr_sz, m_ncols = j_arr_sz;    
        return *this;    
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_sub_matrix(const BaseExpr& val, std::size_t i_start, long i_inc, std::size_t j_start, long j_inc)  {
        auto& o = *static_cast<const NumericExpr<REAL>*>(&val);
        if (i_inc == 0) i_inc = o.size1();
        if (j_inc == 0) j_inc = o.size2();
        std::vector<REAL> tmp;
        std::vector<REAL>* work = (&val == this) ? &tmp : &m_v;
        std::size_t t_nrows = 0, t_ncols = 0;
        if (i_inc > 0)
            t_nrows = (o.size1() - i_start) / i_inc + (((o.size1() - i_start) % i_inc > 0) ? 1 : 0);
        else 
            t_nrows = (i_start + 1) / (-i_inc) + (((i_start + 1) % (-i_inc) > 0) ? 1 : 0);
        if (j_inc > 0)
            t_ncols = (o.size2() - j_start) / j_inc + (((o.size2() - j_start) % j_inc > 0) ? 1 : 0);
        else 
            t_ncols = (j_start + 1) / (-j_inc) + (((j_start + 1) % (-j_inc) > 0) ? 1 : 0);
        work->resize(t_nrows*t_ncols);
        
        std::size_t ik = 0, jk = 0;
        while(j_start < o.size2() && j_start >= 0){
            ik = 0;
            while(i_start < o.size1() && i_start >= 0){
                (*work)[ik + jk*t_nrows] = o(i_start, j_start);
                i_start += i_inc;
                ++ik;
            } 
            j_start += j_inc;
            ++jk; 
        } 
        m_v = std::move(*work);
        m_nrows = t_nrows, m_ncols = t_ncols;
        return *this;     
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_horzcat(const BaseExpr** vals_begin, const BaseExpr** vals_end){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        bool overlap = (vals_begin[0] == this);
        auto t_nrow = vals_begin[0]->nrows(), t_ncol = vals_begin[0]->ncols();
        for(auto p = vals_begin+1; p < vals_end; ++p){
            if ((*p)->nrows() != t_nrow)
                throw std::runtime_error("All expression must have same number of rows for this operation");
            if (*p == this) 
                overlap = true; 
            t_ncol += (*p)->ncols();      
        }    
        std::vector<REAL> tmp;
        std::vector<REAL>* work = overlap ? &tmp : &m_v;
        work->resize(t_nrow * t_ncol);
        t_ncol = 0;
        for (auto p = vals_begin; p < vals_end; ++p){
            auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
            for (int j = 0; j < (*p)->ncols(); ++j)
            for (int i = 0; i < t_nrow; ++i)
                (*work)[i + (j + t_ncol) * t_nrow] = o(i, j);
            t_ncol += (*p)->ncols(); 
        }
        m_v = std::move(*work);
        m_nrows = t_nrow, m_ncols = t_ncol;
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_vertcat(const BaseExpr** vals_begin, const BaseExpr** vals_end){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        bool overlap = (vals_begin[0] == this);
        auto t_nrow = vals_begin[0]->nrows(), t_ncol = vals_begin[0]->ncols();
        for(auto p = vals_begin+1; p < vals_end; ++p){
            if ((*p)->ncols() != t_ncol)
                throw std::runtime_error("All expression must have same number of cols for this operation");
            if (*p == this) 
                overlap = true; 
            t_nrow += (*p)->nrows();      
        }    
        std::vector<REAL> tmp;
        std::vector<REAL>* work = overlap ? &tmp : &m_v;
        work->resize(t_nrow * t_ncol);
        std::size_t lt_nrow = 0;
        for (auto p = vals_begin; p < vals_end; ++p){
            auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
            for (int j = 0; j < (*p)->ncols(); ++j)
            for (int i = 0; i < t_nrow; ++i)
                (*work)[(lt_nrow+i) + j * t_nrow] = o(i, j);
            lt_nrow += (*p)->ncols(); 
        }
        m_v = std::move(*work);
        m_nrows = t_nrow, m_ncols = t_ncol;
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_row(const BaseExpr& val, std::size_t i)  { 
        auto& o = *static_cast<const NumericExpr<REAL>*>(&val);
        m_nrows = 1, m_ncols = o.size2();
        m_v.resize(m_nrows*m_ncols);
        for (int j = 0; j < m_ncols; ++j)
            m_v[j] = o(i, j);
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_col(const BaseExpr& val, std::size_t j)  { 
        auto& o = *static_cast<const NumericExpr<REAL>*>(&val);
        m_nrows = o.size1(), m_ncols = 1;
        m_v.resize(m_nrows*m_ncols);
        for (int i = 0; i < m_nrows; ++i)
            m_v[i] = o(i, j);
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& set_scal_sum(const BaseExpr** vals_begin, const BaseExpr** vals_end, const double* coefs){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        bool mem_overlap = false;    
        for (auto p = vals_begin; p < vals_end; ++p){ 
            if (p == this){
                mem_overlap = true;
                break;
            }
        }
        std::vector<double> tmp;
        std::vector<double>* work = mem_overlap ? &tmp : &m_v;   
        uint t_nrows = 0, t_ncols = 0; 
        {    
            auto& o = *static_cast<const NumericExpr<REAL>*>(vals_begin[0]); 
            t_nrows = o.nrows(), t_ncols = o.ncols();
            work->resize(m_nrows*m_ncols);  
        } 
        if (vals_begin < vals_end){
            for (auto p = vals_begin+1; p < vals_end; ++p){
                auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                if (o.nrows() != t_nrows || o.ncols() != t_ncols)
                    throw std::runtime_error("Dimension inconsistency");
            }
            for (std::size_t j = 0; j < t_ncols; ++j)
            for (std::size_t i = 0; i < t_nrows; ++i)
                for (auto p = vals_begin; p < vals_end; ++p){
                    auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                    (*work)[i + j*m_nrows] += coefs[p] * o(i, j);
                }
        }
        m_v = std::move(*work);
        m_nrows = t_nrows, m_ncols = t_ncols;
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_sum(const BaseExpr** vals_begin, const BaseExpr** vals_end){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        bool mem_overlap = false;    
        for (auto p = vals_begin; p < vals_end; ++p){ 
            if (p == this){
                mem_overlap = true;
                break;
            }
        }
        std::vector<double> tmp;
        std::vector<double>* work = mem_overlap ? &tmp : &m_v;   
        uint t_nrows = 0, t_ncols = 0; 
        {    
            auto& o = *static_cast<const NumericExpr<REAL>*>(vals_begin[0]); 
            t_nrows = o.nrows(), t_ncols = o.ncols();
            work->resize(m_nrows*m_ncols); 
            std::copy(o.m_v.data(), o.m_v.data() + m_nrows*m_ncols, work->data()); 
        } 
        if (vals_begin+1 < vals_end){
            for (auto p = vals_begin+1; p < vals_end; ++p){
                auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                if (o.nrows() != t_nrows || o.ncols() != t_ncols)
                    throw std::runtime_error("Dimension inconsistency");
            }
            for (std::size_t j = 0; j < t_ncols; ++j)
            for (std::size_t i = 0; i < t_nrows; ++i)
                for (auto p = vals_begin+1; p < vals_end; ++p){
                    auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                    (*work)[i + j*m_nrows] += o(i, j);
                }
        }
        m_v = std::move(*work);
        m_nrows = t_nrows, m_ncols = t_ncols;
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_mul(const BaseExpr** vals_begin, const BaseExpr** vals_end){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        bool mem_overlap = false;    
        for (auto p = vals_begin; p < vals_end; ++p){ 
            if (p == this){
                mem_overlap = true;
                break;
            }
        }
        std::vector<double> tmp;
        std::vector<double>* work = mem_overlap ? &tmp : &m_v;   
        uint t_nrows = 0, t_ncols = 0; 
        {    
            auto& o = *static_cast<const NumericExpr<REAL>*>(vals_begin[0]); 
            t_nrows = o.nrows(), t_ncols = o.ncols();
            work->resize(m_nrows*m_ncols); 
            std::copy(o.m_v.data(), o.m_v.data() + m_nrows*m_ncols, work->data()); 
        } 
        if (vals_begin+1 < vals_end){
            for (auto p = vals_begin+1; p < vals_end; ++p){
                auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                if (o.nrows() != t_nrows || o.ncols() != t_ncols)
                    throw std::runtime_error("Dimension inconsistency");
            }
            for (std::size_t j = 0; j < t_ncols; ++j)
            for (std::size_t i = 0; i < t_nrows; ++i)
                for (auto p = vals_begin+1; p < vals_end; ++p){
                    auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                    (*work)[i + j*m_nrows] *= o(i, j);
                }
        }
        m_v = std::move(*work);
        m_nrows = t_nrows, m_ncols = t_ncols;
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_min(const BaseExpr** vals_begin, const BaseExpr** vals_end){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        {    
            auto& o = *static_cast<const NumericExpr<REAL>*>(vals_begin[0]); 
            m_nrows = o.nrows(), m_ncols = o.ncols();
            m_v.resize(m_nrows*m_ncols); 
            std::copy(o.m_v.data(), o.m_v.data() + m_nrows*m_ncols, m_v.data()); 
        } 
        if (vals_begin+1 < vals_end){
            for (auto p = vals_begin+1; p < vals_end; ++p){
                auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                if (o.nrows() != m_nrows || o.ncols() != m_ncols)
                    throw std::runtime_error("Dimension inconsistency");
            }
            for (std::size_t j = 0; j < m_ncols; ++j)
            for (std::size_t i = 0; i < m_nrows; ++i)
                for (auto p = vals_begin+1; p < vals_end; ++p){
                    auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                    m_v[i + j*m_nrows] = min(m_v[i + j*m_nrows], o(i, j));
                }
        }
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_max(const BaseExpr** vals_begin, const BaseExpr** vals_end){
        if (vals_begin = nullptr || vals_end == nullptr || vals_begin >= vals_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        {    
            auto& o = *static_cast<const NumericExpr<REAL>*>(vals_begin[0]); 
            m_nrows = o.nrows(), m_ncols = o.ncols();
            m_v.resize(m_nrows*m_ncols); 
            std::copy(o.m_v.data(), o.m_v.data() + m_nrows*m_ncols, m_v.data()); 
        } 
        if (vals_begin+1 < vals_end){
            for (auto p = vals_begin+1; p < vals_end; ++p){
                auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                if (o.nrows() != m_nrows || o.ncols() != m_ncols)
                    throw std::runtime_error("Dimension inconsistency");
            }
            for (std::size_t j = 0; j < m_ncols; ++j)
            for (std::size_t i = 0; i < m_nrows; ++i)
                for (auto p = vals_begin+1; p < vals_end; ++p){
                    auto& o = *static_cast<const NumericExpr<REAL>*>(*p);
                    m_v[i + j*m_nrows] = max(m_v[i + j*m_nrows], o(i, j));
                }
        }
        return *this;
    }
    template<typename REAL>
    std::ostream& NumericExpr<REAL>::operator<<(std::ostream& out) const {
        out << std::scientific << std::setprecision(8);
        for (int i = 0; i < m_nrows; ++i){
            for (int j = 0; j < m_ncols; ++j)
                out << operator()(i, j) << "  ";
            out << "\n";
        }
        return out;   
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_transpose(const BaseExpr& a)  {
        auto& o = *static_cast<const NumericExpr<REAL>*>(&a);
        std::size_t t_nrows = o.nrows(), t_ncols = o.ncols();
        if (&a == this){
            std::vector<REAL> res(t_nrows*t_ncols);
            for (std::size_t j = 0; j < t_ncols; ++j)
                for (std::size_t i = 0; i < t_nrows; ++i)
                    res(j + i*t_ncols) = o(i, j);
            m_v = std::move(res);
        } else {
            m_v.resize(t_nrows*t_ncols);
            for (std::size_t j = 0; j < t_ncols; ++j)
                for (std::size_t i = 0; i < t_nrows; ++i)
                    m_v(j + i*t_ncols) = o(i, j);
        }
        m_nrows = t_ncols, m_ncols = t_nrows;
        return *this;       
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_dot(const BaseExpr& a, const BaseExpr& b)  {
        auto& o1 = *static_cast<const NumericExpr<REAL>*>(&a);
        auto& o2 = *static_cast<const NumericExpr<REAL>*>(&b);
        if (o1.nrows() != o2.nrows() || o1.ncols() != o2.ncols())
            throw std::runtime_error("Dimension inconsistency");
        std::size_t sz = o1.nrows() * o1.ncols();   
        REAL r = std::inner_product(o1.m_v.data(), o1.m_v.data() + sz, o2.m_v.data(), REAL(0.0), std::plus<REAL>(), std::multiplies<REAL>()); 
        m_v.resize(1); m_v[0] = r;
        m_nrows = m_ncols = 1;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_cross(const BaseExpr& a, const BaseExpr& b) {
        if (a.nrows() != 3 || b.nrows() != 3 || a.ncols() != 1 || b.ncols() != 1)
            throw std::runtime_error("Cross product defined only for 3x1 vectors");
        auto& v1 = *static_cast<const NumericExpr<REAL>*>(&a)->m_v;
        auto& v2 = *static_cast<const NumericExpr<REAL>*>(&b)->m_v;    
        std::array<REAL, 3> res{v1[1]*v2[2] - v2[1]*v1[2], -v1[0]*v2[2] + v2[0]*v1[2], v1[0]*v2[1] - v2[0]*v1[1]};
        m_v.resize(3);
        std::copy(res.data(), res.data() + 3, m_v.data());
        m_nrows = 3, m_ncols = 1;
        return *this;
    }
    template<typename REAL>
    NumericExpr<REAL>& NumericExpr<REAL>::set_mtimes(const BaseExpr** m_begin, const BaseExpr** m_end){
        if (m_begin = nullptr || m_end == nullptr || m_begin >= m_end)
            throw std::runtime_error("Function supported only for nonempty ranges");
        if (m_end - m_begin == 1){
            auto& o = *static_cast<const NumericExpr<REAL>*>(m_begin[0]);
            m_nrows = o.nrows(), m_ncols = o.ncols();
            m_v.resize(m_nrows*m_ncols); 
            std::copy(o.m_v.data(), o.m_v.data() + m_nrows*m_ncols, m_v.data()); 
            return *this;
        } else if (m_end - m_begin == 2){
            auto& o1 = *static_cast<const NumericExpr<REAL>*>(m_begin[0]);
            auto& o2 = *static_cast<const NumericExpr<REAL>*>(m_begin[1]);
            if (m_begin[0] != this && m_begin[1] != this){
                m_nrows = o1.nrows(), m_ncols = o2.ncols();
                m_v.resize(m_nrows*m_ncols);
                if (o1.ncols() != o2.nrows())
                        throw std::runtime_error("Dimension inconsistency");
                for (int j = 0; j < o2.ncols(); ++j)
                for (int i = 0; i < o1.nrows(); ++i){
                        REAL s = 0;
                        for (int k = 0; k < o1.ncols(); ++k) 
                            s += o1(i, k) * o2(k, j);
                        m_v[i + m_nrows*j] = s;    
                }
                return *this;    
            } else {
                std::vector<REAL> tmp;
                std::size_t t_nrows = o1.nrows(), t_ncols = o2.ncols();
                tmp.resize(t_nrows*t_ncols);
                if (o1.ncols() != o2.nrows())
                    throw std::runtime_error("Dimension inconsistency");
                for (int j = 0; j < o2.ncols(); ++j)
                for (int i = 0; i < o1.nrows(); ++i){
                        REAL s = 0;
                        for (int k = 0; k < o1.ncols(); ++k) 
                            s += o1(i, k) * o2(k, j);
                        tmp[i + t_nrows*j] = s;    
                }  
                m_nrows = t_nrows, m_ncols = t_ncols;
                m_v = std::move(tmp);
                return *this;  
            }
        } else {
            std::vector<REAL> tmp, tmp1;
            bool mem_overlap = false;
            for (auto p = m_begin; p < m_end; ++p)
                if (*p == this) {
                    mem_overlap = true;
                    break;
                }
            std::vector<REAL>* t0 = &tmp, *t1 = mem_overlap ? &tmp1 : &m_v;
            for (std::size_t r = 0; r < m_end - m_begin - 1; ++r){
                auto& o1 = *static_cast<const NumericExpr<REAL>*>(m_begin[r]);
                auto& o2 = *static_cast<const NumericExpr<REAL>*>(m_begin[r+1]);
                if (o1.ncols() != o2.nrows())
                        throw std::runtime_error("Dimension inconsistency");
            }
            std::size_t t_nrows = 0, t_ncols = 0;
            {
                auto& o1 = *static_cast<const NumericExpr<REAL>*>(m_begin[0]);
                t0->resize(o1.size1()*o1.size2());
                std::copy(o1.m_v.data(), o1.m_v.data() + o1.m_v.size(), t0->data());
                t_nrows = o1.size1(); t_ncols = o1.size2();
            }
            for (std::size_t r = 0; r < m_end - m_begin - 1; ++r){

                auto& o1 = *static_cast<const NumericExpr<REAL>*>(m_begin[r]);
                auto& o2 = *static_cast<const NumericExpr<REAL>*>(m_begin[r+1]);
                std::size_t ttncols = o2.ncols();
                t1->resize(t_nrows*t_ncols);
                for (int j = 0; j < o2.ncols(); ++j)
                for (int i = 0; i < o1.nrows(); ++i){
                        REAL s = 0;
                        for (int k = 0; k < o1.ncols(); ++k) 
                            s += (*t0)[i + k * t_nrows] * o2.m_v[k + j*o2.nrows()];
                        (*t1)[i + t_nrows*j] = s;    
                }
                t_ncols = ttncols;
                std::swap(t1, t0);
            }
            m_v = std::move(*t1);
            m_nrows = t_nrows; m_ncols = t_ncols;
            return *this;
        }
    }
    template<typename REAL>
    template<typename ONEARG_FUNCTOR>
    NumericExpr<REAL>& NumericExpr<REAL>::set_op(const BaseExpr& a, const ONEARG_FUNCTOR& transform){
        auto& o = *static_cast<const NumericExpr<REAL>*>(&a);
        m_nrows = o.nrows(), m_ncols = o.ncols();
        m_v.resize(m_nrows*m_ncols);
        for (std::size_t j = 0; j < m_ncols; ++j)
            for (std::size_t i = 0; i < m_nrows; ++i)
                m_v[i + j*m_nrows] = transform(o(i, j));
        return *this;
    }
    template<typename REAL>
    template<typename TWOARG_FUNCTOR>
    NumericExpr<REAL>& NumericExpr<REAL>::set_op(const BaseExpr& a, const BaseExpr& b, const TWOARG_FUNCTOR& op){
        auto& o1 = *static_cast<const NumericExpr<REAL>*>(&a);
        auto& o2 = *static_cast<const NumericExpr<REAL>*>(&b);
        m_nrows = o1.nrows(), m_ncols = o1.ncols();
        m_v.resize(m_nrows*m_ncols);
        if (o1.nrows() != o2.nrows() || o1.ncols() != o2.ncols())
            throw std::runtime_error("Dimension inconsistency");
        for (std::size_t j = 0; j < m_ncols; ++j)
            for (std::size_t i = 0; i < m_nrows; ++i)
                m_v[i + j*m_nrows] = op(o1(i, j), o2(i, j));
        return *this;        
    }
    template<typename REAL>
    template<typename THREEARG_FUNCTOR>
    NumericExpr<REAL>& NumericExpr<REAL>::set_op(const BaseExpr& a, const BaseExpr& b, const BaseExpr& c, const THREEARG_FUNCTOR& op){
        auto& o1 = *static_cast<const NumericExpr<REAL>*>(&a);
        auto& o2 = *static_cast<const NumericExpr<REAL>*>(&b);
        auto& o3 = *static_cast<const NumericExpr<REAL>*>(&c);
        m_nrows = o1.nrows(), m_ncols = o1.ncols();
        m_v.resize(m_nrows*m_ncols);
        if (o1.nrows() != o2.nrows() || o1.ncols() != o2.ncols() ||
            o1.nrows() != o3.nrows() || o1.ncols() != o3.ncols())
            throw std::runtime_error("Dimension inconsistency");
        for (std::size_t j = 0; j < m_ncols; ++j)
            for (std::size_t i = 0; i < m_nrows; ++i)
                m_v[i + j*m_nrows] = op(o1(i, j), o2(i, j), o3(i, j));
        return *this;        
    }
}