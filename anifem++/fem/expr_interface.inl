namespace Ani{
    inline std::ostream& operator<<(std::ostream& out, const Expr& a) {
        if (!a.m_invoker)
            out << "Expr[0, 0]\n";
        else {
            out << "Expr[" << a.nrows() << ", " << a.ncols() << "] = \n";
            a.m_invoker->operator<<(out);
        }
        return out;    
    }

    #define APPL_OP_FULL1(OPNAME, SETNAME)\
    inline Expr OPNAME (const Expr& m){ \
        Expr res = m.zeroes(); \
        return res.m_invoker->set_##SETNAME(*m.m_invoker), res; \
    }

    #define APPL_OP_FULL2(OPNAME, SETNAME)\
    inline Expr OPNAME (const Expr& m1, const Expr& m2){ \
        Expr res = m1.zeroes(); \
        return res.m_invoker->set_##SETNAME(*m1.m_invoker, *m2.m_invoker), res; \
    }\
    inline Expr OPNAME (const Expr& m1, double m2){ \
        Expr res = m1.zeroes(), tmp = m1.zeroes(m1.nrows(), m1.ncols()); \
        tmp = m2;\
        return res.m_invoker->set_##SETNAME(*m1.m_invoker, *tmp.m_invoker), res; \
    }\
    inline Expr OPNAME (double m1, const Expr& m2){ \
        Expr res = m2.zeroes(), tmp = m2.zeroes(m2.nrows(), m2.ncols()); \
        tmp = m1;\
        return res.m_invoker->set_##SETNAME(*tmp.m_invoker, *m2.m_invoker), res; \
    }
    
    #define APPL_OP_FULL3(OPNAME, SETNAME)\
    inline Expr OPNAME (const Expr& m1, const Expr& m2, const Expr& m3){ \
        Expr res = m1.zeroes(); \
        return res.m_invoker->set_##SETNAME(*m1.m_invoker, *m2.m_invoker, *m3.m_invoker), res; \
    }\
    inline Expr OPNAME (double m1, const Expr& m2, const Expr& m3){ \
        Expr res = m2.zeroes(); \
        auto tmp = (m2.zeroes(m2.nrows(), m2.ncols()) = m1);\
        return res.m_invoker->set_##SETNAME(*tmp.m_invoker, *m2.m_invoker, *m3.m_invoker), res; \
    }\
    inline Expr OPNAME (const Expr& m1, double m2, const Expr& m3){ \
        Expr res = m1.zeroes(); \
        auto tmp = (m1.zeroes(m1.nrows(), m1.ncols()) = m2);\
        return res.m_invoker->set_##SETNAME(*m1.m_invoker, *tmp.m_invoker, *m3.m_invoker), res; \
    }\
    inline Expr OPNAME (const Expr& m1, const Expr& m2, double m3){ \
        Expr res = m1.zeroes(); \
        auto tmp = (m1.zeroes(m1.nrows(), m1.ncols()) = m3);\
        return res.m_invoker->set_##SETNAME(*m1.m_invoker, *m2.m_invoker, *tmp.m_invoker), res; \
    }\
    inline Expr OPNAME (double m1, double m2, const Expr& m3){ \
        Expr res = m3.zeroes(); \
        auto t1 = (m3.zeroes(m3.nrows(), m3.ncols()) = m1);\
        auto t2 = (m3.zeroes(m3.nrows(), m3.ncols()) = m2);\
        return res.m_invoker->set_##SETNAME(*t1.m_invoker, *t2.m_invoker, *m3.m_invoker), res; \
    }\
    inline Expr OPNAME (const Expr& m1, double m2, double m3){ \
        Expr res = m1.zeroes(); \
        auto t2 = (m1.zeroes(m1.nrows(), m1.ncols()) = m2);\
        auto t3 = (m1.zeroes(m1.nrows(), m1.ncols()) = m3);\
        return res.m_invoker->set_##SETNAME(*m1.m_invoker, *t2.m_invoker, *t3.m_invoker), res; \
    }\
    inline Expr OPNAME (double m1, const Expr& m2, double m3){ \
        Expr res = m2.zeroes(); \
        auto t1 = (m2.zeroes(m2.nrows(), m2.ncols()) = m1);\
        auto t3 = (m2.zeroes(m2.nrows(), m2.ncols()) = m3);\
        return res.m_invoker->set_##SETNAME(*t1.m_invoker, *m2.m_invoker, *t3.m_invoker), res; \
    }

    #define APPL_OP1(CUROP) APPL_OP_FULL1(CUROP, CUROP)
    #define APPL_OP2(CUROP) APPL_OP_FULL2(CUROP, CUROP)
    #define APPL_OP3(CUROP) APPL_OP_FULL3(CUROP, CUROP)

    #define APPL_GATHER(OPNAME, SETNAME)\
    inline Expr OPNAME(const Expr* st, const Expr* end) {\
        long sz = end - st;\
        if (sz <= 0)\
            throw std::runtime_error("Function supported only for nonempty ranges");\
        std::vector<const BaseExpr*> be(sz);\
        for (long i = 0; i < sz; ++i){\
            be[i] = st[i].m_invoker.get();\
            if (be[i] == nullptr)\
                throw std::runtime_error("Faced uninitialized expresion");\
        }\
        Expr res = st->zeroes();\
        return res.m_invoker->set_##SETNAME(be.data(), be.data() + sz), res;\
    }

    APPL_OP_FULL2(operator+, sum)
    APPL_OP_FULL2(operator-, sub)
    APPL_OP_FULL2(operator*, mul)
    APPL_OP_FULL2(operator/, div)
    APPL_OP_FULL1(operator-, swap_sign)

    APPL_OP_FULL1(operator!, not)
    APPL_OP_FULL2(operator<, lt)
    APPL_OP_FULL2(operator<=, le)
    APPL_OP_FULL2(operator>, gt)
    APPL_OP_FULL2(operator>=, le)
    APPL_OP_FULL2(operator==, eq)
    APPL_OP_FULL2(operator!=, ne)
    APPL_OP_FULL2(operator&&, and)
    APPL_OP_FULL2(operator||, or)

    namespace FT{
        inline Expr submatrix(const Expr& m, const std::size_t* i, std::size_t i_arr_sz, const std::size_t* j, long j_arr_sz){
            Expr res = m.zeroes();
            return res.m_invoker->set_sub_matrix(*m.m_invoker, i, i_arr_sz, j, j_arr_sz), res;
        }
        inline Expr submatrix(const Expr& m, std::size_t i_start, long i_inc, std::size_t j_start, long j_inc){
            Expr res = m.zeroes();
            return res.m_invoker->set_sub_matrix(*m.m_invoker, i_start, i_inc, j_start, j_inc), res;
        }

        APPL_OP1(abs   )
        APPL_OP1(exp   )
        APPL_OP1(expm1 )
        APPL_OP1(log   )
        APPL_OP1(log1p )
        APPL_OP1(sqrt  )
        APPL_OP1(cbrt  )
        APPL_OP1(sin   )
        APPL_OP1(cos   )
        APPL_OP1(tan   )
        APPL_OP1(asin  )
        APPL_OP1(acos  )
        APPL_OP1(atan  )
        APPL_OP1(sinh  )
        APPL_OP1(cosh  )
        APPL_OP1(tanh  )
        APPL_OP1(erf   )
        APPL_OP1(erfc  )
        APPL_OP1(tgamma)
        APPL_OP1(lgamma)
        APPL_OP1(floor )
        APPL_OP1(ceil  )
        APPL_OP1(sign  )
        APPL_OP1(sq    )
        APPL_OP1(norm  )

        APPL_OP2(pow   )
        APPL_OP2(atan2 )
        APPL_OP2(fmod  )
        APPL_OP2(min   )
        APPL_OP2(max   )
        APPL_OP2(hypot )
        APPL_OP2(dot   )
        APPL_OP2(cross )
        APPL_OP2(mtimes)

        APPL_OP3(fma   )
        APPL_OP3(ifelse)

        APPL_GATHER(min, min)
        APPL_GATHER(max, max)
        APPL_GATHER(mtimes, mtimes)
        APPL_GATHER(vertcat, vertcat)
        APPL_GATHER(horzcat, horzcat)
        inline Expr det3x3(const Expr& m){
            if (m.nrows() != 3 || m.ncols() != 3)
                throw std::runtime_error("Function det3x3 defined only for 3x3 matrices");
            return dot(m.row(0), cross(m.row(1), m.row(2)));    
        }
        inline Expr scalsum(const Expr* st, const Expr* end, const double* coefs) {
            long sz = end - st;
            if (sz <= 0)
                throw std::runtime_error("Function supported only for nonempty ranges");
            std::vector<const BaseExpr*> be(sz);
            for (long i = 0; i < sz; ++i){
                be[i] = st[i].m_invoker.get();
                if (be[i] == nullptr)
                    throw std::runtime_error("Faced uninitialized expresion");
            }
            Expr res = st->zeroes();
            return res.m_invoker->set_scal_sum(be.data(), be.data() + sz, coefs), res;
        }
    }

    #undef APPL_OP1
    #undef APPL_OP2
    #undef APPL_OP3
    #undef APPL_OP_FULL1
    #undef APPL_OP_FULL2
    #undef APPL_OP_FULL3
    #undef APPL_GATHER
}