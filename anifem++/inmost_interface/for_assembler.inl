//
// Created by Liogky Alexey on 07.04.2022.
//

#ifndef CARNUM_FORASSEMBLER_INL
#define CARNUM_FORASSEMBLER_INL

namespace Ani{
    inline INMOST::ElementType AniGeomMaskToInmostElementType(DofT::uint ani_geom_mask){
        using namespace INMOST;
        static constexpr ElementType lookup[] = {NONE, NODE, EDGE, EDGE, FACE, FACE, CELL};
        ElementType r = NONE;
        for (int i = 0; i < DofT::NGEOM_TYPES; ++i) if (ani_geom_mask & (1 << i))
            r |= lookup[i+1];
        return r;    
    }
    inline std::array<uint, 4> DofTNumDofsToGeomNumDofs(std::array<DofT::uint, DofT::NGEOM_TYPES> num_dofs){
        return {num_dofs[0], num_dofs[1] + num_dofs[2], num_dofs[3] + num_dofs[4], num_dofs[5]};
    }
    template<typename Real>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F)> f, size_t nRow, size_t nCol){
        struct AniMatRhs0: public ElemMatEval{
            std::function<void(const Real** XY, Real* A, Real* F)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t n_out() const override { return 2; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); return (res_id==0) ? nCol : 1; }
            bool is_user_data_required() const override{ return false; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void)w; (void)iw; (void)user_data;
                m_f(args, res[0], res[1]);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniMatRhs0(decltype(m_f) f, size_t nRow, size_t nCol): m_f{f}, nRow(nRow), nCol(nCol) {}
        };
        return std::make_shared<AniMatRhs0>(std::move(f), nRow, nCol);
    }

    template<typename Real>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, void* user_data)> f, size_t nRow, size_t nCol){
        struct AniMatRhs1: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, void* user_data)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t n_out() const override { return 2; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); return (res_id==0) ? nCol : 1; }
            bool is_user_data_required() const override{ return true; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void)w; (void)iw;
                m_f(args, res[0], res[1], user_data);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void)res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void)res_id; return 1; }
            AniMatRhs1(decltype(m_f) f, size_t nRow, size_t nCol): m_f{f}, nRow(nRow), nCol(nCol) {}
        };
        return std::make_shared<AniMatRhs1>(std::move(f), nRow, nCol);
    }

    template<typename Real, typename Int>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data)> f,
                                                    size_t nRow, size_t nCol, size_t nw, size_t niw){
        struct AniMatRhs2: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw, void* user_data)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t nw = 0, niw = 0;
            size_t n_out() const override { return 2; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void)res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); return (res_id==0) ? nCol : 1; }
            void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
                sz_args = 4; sz_res = 2; sz_w = nw; sz_iw = niw;
            }
            bool is_user_data_required() const override{ return true; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                m_f(args, res[0], res[1], w, iw, user_data);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void)res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void)res_id; return 1; }
            AniMatRhs2(decltype(m_f) f, size_t nRow, size_t nCol, size_t nw, size_t niw):
                m_f{f}, nRow(nRow), nCol(nCol), nw(nw), niw(niw) {}
        };
        return std::make_shared<AniMatRhs2>(std::move(f), nRow, nCol, nw, niw);
    }

    template<typename Real, typename Int>
    std::shared_ptr<ElemMatEval> GenerateElemMatRhs(std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw)> f,
                                                    size_t nRow, size_t nCol, size_t nw, size_t niw){
        struct AniMatRhs3: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* F, Real* w, Int* iw)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t nw = 0, niw = 0;
            size_t n_out() const override { return 2; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); return (res_id==0) ? nCol : 1; }
            void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
                sz_args = 4; sz_res = 2; sz_w = nw; sz_iw = niw;
            }
            bool is_user_data_required() const override{ return false; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void)user_data;
                m_f(args, res[0], res[1], w, iw);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniMatRhs3(decltype(m_f) f, size_t nRow, size_t nCol, size_t nw, size_t niw):
                    m_f{f}, nRow(nRow), nCol(nCol), nw(nw), niw(niw) {}
        };
        return std::make_shared<AniMatRhs3>(std::move(f), nRow, nCol, nw, niw);
    }

    template<typename Real>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A)> f, size_t nRow, size_t nCol){
        struct AniMat0: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nCol; }
            bool is_user_data_required() const override{ return false; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void) w; (void) iw; (void) user_data;
                m_f(args, res[0]);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniMat0(decltype(m_f) f, size_t nRow, size_t nCol):
                    m_f{f}, nRow(nRow), nCol(nCol) {}
        };
        return std::make_shared<AniMat0>(std::move(f), nRow, nCol);
    }

    template<typename Real>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, void* user_data)> f, size_t nRow, size_t nCol){
        struct AniMat1: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, void* user_data)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nCol; }
            bool is_user_data_required() const override{ return true; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void) w; (void) iw;
                m_f(args, res[0], user_data);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniMat1(decltype(m_f) f, size_t nRow, size_t nCol):
                    m_f{f}, nRow(nRow), nCol(nCol) {}
        };
        return std::make_shared<AniMat1>(std::move(f), nRow, nCol);
    }

    template<typename Real, typename Int>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data)> f,
                                                 size_t nRow, size_t nCol, size_t nw, size_t niw){
        struct AniMat2: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t nw = 0, niw = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nCol; }
            void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
                sz_args = 4; sz_res = 1; sz_w = nw; sz_iw = niw;
            }
            bool is_user_data_required() const override{ return true; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                m_f(args, res[0], w, iw, user_data);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniMat2(decltype(m_f) f, size_t nRow, size_t nCol, size_t nw, size_t niw):
                    m_f{f}, nRow(nRow), nCol(nCol), nw(nw), niw(niw) {}
        };
        return std::make_shared<AniMat2>(std::move(f), nRow, nCol, nw, niw);
    }

    template<typename Real, typename Int>
    std::shared_ptr<ElemMatEval> GenerateElemMat(std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw)> f,
                                                 size_t nRow, size_t nCol, size_t nw, size_t niw){
        struct AniMat3: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw)> m_f;
            size_t nRow = 0, nCol = 0;
            size_t nw = 0, niw = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nCol; }
            void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
                sz_args = 4; sz_res = 1; sz_w = nw; sz_iw = niw;
            }
            bool is_user_data_required() const override{ return false; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void) user_data; 
                m_f(args, res[0], w, iw);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniMat3(decltype(m_f) f, size_t nRow, size_t nCol, size_t nw, size_t niw):
                    m_f{f}, nRow(nRow), nCol(nCol), nw(nw), niw(niw) {}
        };
        return std::make_shared<AniMat3>(std::move(f), nRow, nCol, nw, niw);
    }

    template<typename Real>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F)> f, size_t nRow){
        struct AniRhs0: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A)> m_f;
            size_t nRow = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return 1; }
            bool is_user_data_required() const override{ return false; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void) w; (void) iw; (void) user_data;
                m_f(args, res[0]);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniRhs0(decltype(m_f) f, size_t nRow):
                    m_f{f}, nRow(nRow) {}
        };
        return std::make_shared<AniRhs0>(std::move(f), nRow);
    }

    template<typename Real>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, void* user_data)> f, size_t nRow){
        struct AniRhs1: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, void* user_data)> m_f;
            size_t nRow = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<int>(n_out())); (void) res_id; return 1; }
            bool is_user_data_required() const override{ return true; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void) w; (void) iw;
                m_f(args, res[0], user_data);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniRhs1(decltype(m_f) f, size_t nRow):
                    m_f{f}, nRow(nRow) {}
        };
        return std::make_shared<AniRhs1>(std::move(f), nRow);
    }

    template<typename Real, typename Int>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw, void* user_data)> f,
                                                 size_t nRow, size_t nw, size_t niw){
        struct AniRhs2: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw, void* user_data)> m_f;
            size_t nRow = 0;
            size_t nw = 0, niw = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return 1; }
            void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
                sz_args = 4; sz_res = 1; sz_w = nw; sz_iw = niw;
            }
            bool is_user_data_required() const override{ return true; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                m_f(args, res[0], w, iw, user_data);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniRhs2(decltype(m_f) f, size_t nRow, size_t nw, size_t niw):
                    m_f{f}, nRow(nRow), nw(nw), niw(niw) {}
        };
        return std::make_shared<AniRhs2>(std::move(f), nRow, nw, niw);
    }

    template<typename Real, typename Int>
    std::shared_ptr<ElemMatEval> GenerateElemRhs(std::function<void(const Real** XY/*[4]*/, Real* F, Real* w, Int* iw)> f,
                                                 size_t nRow, size_t nw, size_t niw){
        struct AniRhs3: public ElemMatEval{
            std::function<void(const Real** XY/*[4]*/, Real* A, Real* w, Int* iw)> m_f;
            size_t nRow = 0;
            size_t nw = 0, niw = 0;
            size_t n_out() const override { return 1; }
            Int out_size1(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return nRow; }
            Int out_size2(Int res_id) const override{ assert(res_id < static_cast<Int>(n_out())); (void) res_id; return 1; }
            void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
                sz_args = 4; sz_res = 1; sz_w = nw; sz_iw = niw;
            }
            bool is_user_data_required() const override{ return false; }
            void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
                (void) user_data;
                m_f(args, res[0], w, iw);
            }
            size_t n_in() const override { return 4; }
            Int in_size1(Int res_id) const override{ (void) res_id; return 3; }
            Int in_size2(Int res_id) const override{ (void) res_id; return 1; }
            AniRhs3(decltype(m_f) f, size_t nRow, size_t nw, size_t niw):
                    m_f{f}, nRow(nRow), nw(nw), niw(niw) {}
        };
        return std::make_shared<AniRhs3>(std::move(f), nRow, nw, niw);
    }
};

#endif //CARNUM_FORASSEMBLER_INL
