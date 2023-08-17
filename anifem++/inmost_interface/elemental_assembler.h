//
// Created by Liogky Alexey on 28.03.2022.
//

#ifndef CARNUM_ELEMENTALASSEMBLER_H
#define CARNUM_ELEMENTALASSEMBLER_H

#include "anifem++/fem/tetdofmap.h"
#include "ordering.h"
#include "func_wrap.h"
#include "timer_wrap.h"
#include <vector>
#include <array>
#include <set>
#include <iostream>

namespace Ani{
    struct FemVarContainer: public DofT::ComplexDofMap{
        std::vector<std::string> names;

        using DofMap = DofT::DofMap;
        struct FemVarView{
            const FemVarContainer* back_ptr = nullptr;
            uint iVar = 0;
            const std::string* name = nullptr;
            DofT::NestedDofMapView dofmap;
        };
        struct Iterator{
        protected:
            const FemVarContainer* back_ptr = nullptr;
            uint iVar = 0;
            mutable FemVarView view;
            Iterator() = default;
            Iterator(const FemVarContainer* self, uint iVar = 0): back_ptr{self}, iVar{iVar} {}
        public:
            Iterator(const Iterator&) = default;
            typedef std::random_access_iterator_tag  iterator_category;
            typedef FemVarView value_type;
            typedef int difference_type;
            typedef const FemVarView* pointer;
            typedef const FemVarView reference;

            Iterator& operator ++() { ++iVar; return *this; }
            Iterator  operator ++(int) {Iterator ret(*this); operator++(); return ret;}
            Iterator& operator --() { --iVar; return *this; }
            Iterator  operator --(int) {Iterator ret(*this); operator--(); return ret;}
            bool operator ==(const Iterator & b) const { return iVar == b.iVar && back_ptr == b.back_ptr; }
            bool operator !=(const  Iterator & other) const { return !operator==(other); }
            reference operator*() const { set_state(); return view; }
            pointer operator ->() const { set_state(); return &view; }
            reference operator[](difference_type n) const { return *(*this + n); }
            difference_type operator-(const  Iterator & other) const { return iVar - other.iVar; }
            Iterator& operator+=(difference_type n) { iVar += n; return *this; } 
            Iterator& operator-=(difference_type n) { iVar -= n; return *this; }
            Iterator  operator+ (difference_type n) const { Iterator other = *this; other += n; return other; }
            Iterator  operator- (difference_type n) const { Iterator other = *this; other -= n; return other; }
            friend Iterator  operator+(difference_type n, const Iterator& a) { return a+n; }
            bool operator< (const  Iterator & other) const { return back_ptr == other.back_ptr ? iVar <  other.iVar : false; }
            bool operator> (const  Iterator & other) const { return back_ptr == other.back_ptr ? iVar >  other.iVar : false; }
            bool operator<=(const  Iterator & other) const { return back_ptr == other.back_ptr ? iVar <= other.iVar : false; }
            bool operator>=(const  Iterator & other) const { return back_ptr == other.back_ptr ? iVar >= other.iVar : false; }
        private:
            inline void set_state() const;
            friend FemVarContainer;
            // friend Iterator FemVarContainer::Begin() const;
            // friend Iterator FemVarContainer::End() const;
        };
        Iterator Begin() const { return Iterator(this, 0); }
        Iterator End() const { return Iterator(this, m_spaces.size()); }
        FemVarView operator[](int id) const { return Iterator(this, 0)[id];}

        auto Clear() { 
            m_spaces.clear();
            for (auto& v: m_spaceNumDofTet) v.clear();
            for (auto& v: m_spaceNumDof) v.clear();
            m_spaceNumDofsTet.clear();
            names.clear();
        } 
        void PushBack(std::shared_ptr<DofT::BaseDofMap> space) override { 
            static int unique_var = 0;
            DofT::ComplexDofMap::PushBack(std::move(space));
            names.push_back("anonymous_" + std::to_string(unique_var++));
        }
        void PushVar(DofT::DofMap odf, const std::string& name){ 
            DofT::ComplexDofMap::PushBack(std::move(odf.m_invoker));
            names.push_back(name);
        }
        int NumVars() const { return m_spaces.size(); }
    };
    inline void FemVarContainer::Iterator::set_state() const {
        const int idx = iVar;
        view = FemVarView{back_ptr, iVar, back_ptr->names.data() + iVar, back_ptr->GetNestedDofMapView(&idx, 1)};
    }

    /// @brief Hold description about local fem enumeration, d.o.f.s loci and other description of FEM variables
    struct FemExprDescr{
        using DofMap = DofT::DofMap;

        FemVarContainer trial_funcs;
        FemVarContainer test_funcs;

        void Clear() { trial_funcs.Clear(), test_funcs.Clear(); }
        /// Add variable to physical vector (add trial space dof map)
        void PushTrialFunc(DofMap trial_map, const std::string& name) { trial_funcs.PushVar(trial_map, name); }
        /// Add test func space to vector of test functional spaces (add test space dof map)
        void PushTestFunc(DofMap test_map, const std::string& name) { test_funcs.PushVar(test_map, name); }
        /// Add fem pair of conjugate spaces: trial-test spaces
        void PushFemPair(DofMap trial_map, const std::string& trial_var_name, DofMap test_map, const std::string& test_var_name){
            trial_funcs.PushVar(trial_map, trial_var_name);
            test_funcs.PushVar(test_map, test_var_name);
        }
        int NumVars() const { return trial_funcs.NumVars(); }
        const FemVarContainer& TrialFuncs() const { return trial_funcs; }
        const FemVarContainer& TestFuncs() const { return test_funcs; }
    };

    /// Wrapper for internal global assembler information to give user flexible interface to run elemental matrix assembling
    class ElementalAssembler {
        using Mesh = INMOST::Mesh;
        using uchar = unsigned char;

        template<typename Storage>
        using ElementArray = INMOST::ElementArray<Storage>;
        using Node = INMOST::Node;
        using Edge = INMOST::Edge;
        using Face = INMOST::Face;
        using Cell = INMOST::Cell;
        using Real = MatFuncWrap<>::Real;
        using Int = MatFuncWrap<>::Int;

    public:
    #ifndef NO_ASSEMBLER_TIMERS
        struct TimeMessures{
            TimerWrap* m_timer;
            double *m_time_init_user_handler,
                    *m_time_comp_func;
        };
    #endif
        struct VarsHelper{
            FemExprDescr* descr; ///< vector of descriptions of physical variables
            std::vector<double> initValues; ///< vector of current values of d.o.f.s (important for nonlinear problems)
            bool same_template = false;

            /// Number of physical variables
            int NumVars() const { return descr->NumVars(); }
            /// Is Base FEM space equal to Test FEM space ?
            bool IsSameTemplate() const { return same_template; }
            /// Iterator of current values of d.o.f.s for specific number of variable in vector of physical variables
            double* begin(int iVar) { return initValues.data() + descr->TrialFuncs().m_spaceNumDofsTet[iVar  ]; }
            double* end(int iVar)   { return initValues.data() + descr->TrialFuncs().m_spaceNumDofsTet[iVar+1]; }
            void Clear() { initValues.clear(); descr = nullptr; }
        }; 
        enum UsageType{
            RHS = 0,    ///< run to compute elemental rhs
            MAT = 1,    ///< run to compute elemental matrix
            MAT_RHS = 2,///< run to compute elemental matrix and rhs simultaneously (common case for linear problems)
            NUSAGETYPES
        };

        MatFuncWrap<>* func = nullptr;             ///< elemental matrix and rhs evaluator
        UsageType f_type = NUSAGETYPES;            ///< type of running
        SparsedData<> loc_m, loc_rhs;              ///< memory to save elemental matrix and rhs 
        MatFuncWrap<>::Memory mem;                 ///< some additional memory for evaluator
        std::vector<bool>* loc_mb = nullptr;       ///< indicator of zeros (including structured) in elemental matrix
        VarsHelper* vars = nullptr;                ///< description of FEM spaces of variables and initial guess
        const Mesh * m = nullptr;
        const ElementArray<Node>* nodes = nullptr; ///< nodes of current tetrahedron in positive order
        const ElementArray<Edge>* edges = nullptr; ///< edges of current tetrahedron in consistent with nodes order
        const ElementArray<Face>* faces = nullptr; ///< faces of current tetrahedron in consistent with nodes order
        const Cell* cell = nullptr;                ///< current tetrahedron
        /// Permutation of {0, 1, 2, 3} showing remapping between current element or it's parts 
        /// and d.o.f.'s on particular geometrical element when some d.o.f.'s distributed 
        /// by not trivial group of symmetry (not s1) on the element; 
        /// Actually show relation of global indexes by its values, e.g. for nodes with global indexes {523, 11, 977, 143} will be {2, 0, 3, 1}  
        /// @warning if used spaces have only trivially distributed d.o.f.'s may be set to {0, 1, 2, 3} instead of proper value
        const uchar* node_permutation = nullptr; 
        DynMem<Real, Int>* pool = nullptr;    ///< memory allocator
    #ifndef NO_ASSEMBLER_TIMERS
    private:
        TimeMessures m_tmes;
    public:    
    #endif
        ///for backward compatibility
        // static constexpr int local_node_index[4]{0, 1, 2, 3};  
        // static constexpr int local_edge_index[6]{0, 1, 2, 3, 4, 5};
        // static constexpr int local_face_index[4]{0, 1, 2, 3};  

        ElementalAssembler() = default;  
        void init(MatFuncWrap<>* _f, UsageType _f_type,
                SparsedData<> _loc_m, SparsedData<> _loc_rhs, MatFuncWrap<>::Memory _mem, std::vector<bool> * _loc_mb,
                VarsHelper* _vars, const Mesh * const _m,
                const ElementArray<Node>* _nodes, const ElementArray<Edge>* _edges, const ElementArray<Face>* _faces, const Cell* _cell,
                const uchar* _node_permutation,
    #ifndef NO_ASSEMBLER_TIMERS
                TimeMessures _tmes,
    #endif
                DynMem<Real, Int>* _pool = nullptr
                ); 
        ElementalAssembler(MatFuncWrap<>* f, UsageType f_type,
                SparsedData<> loc_m, SparsedData<> loc_rhs, MatFuncWrap<>::Memory mem, std::vector<bool> *loc_mb,
                VarsHelper* vars, const Mesh * const m,
                const ElementArray<Node>* nodes, const ElementArray<Edge>* edges, const ElementArray<Face>* faces,
                const Cell* cell,
                const uchar* _node_permutation,
    #ifndef NO_ASSEMBLER_TIMERS
                TimeMessures tmes,
    #endif
                DynMem<Real, Int>* _pool = nullptr
                );
        ///Update some internal data and set zeros to loc_m and loc_rhs
        void update();

        /// Run evaluator
        ///@param args are input parameters to be delivered into evaluator
        void compute(const Real** args);
        /// Run evaluator
        ///@param args are input parameters to be delivered into evaluator
        ///@param user_data is user specific data to be delivered into evaluator
        void compute(const Real** args, void* user_data);
        ///Return data for 3x4 col-major matrix of coordinates of nodes of tetrahedron
        double* get_nodes();
        const double* get_nodes() const;
        //sepX and sepY is set of numbers columns and rows before thats we set separator = "|"
        std::ostream& print_matrix_and_rhs_arbitrarily(std::ostream& out = std::cout, const std::set<int>& sepX = {}, const std::set<int>& sepY = {}) const;
        void print_input(const double** args) const;

    private:
        Real _nn_p[4][3];
        void make_result_buffer(); 
        void densify_result(); 

    public:
        template<class RandomIt>
        static void GatherDataOnElement(const INMOST::Tag& from, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp); 
        template<class RandomIt>
        static void GatherDataOnElement(const INMOST::Tag* var_tags, const std::size_t nvar_tags, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp);
        template<class RandomIt>
        static void GatherDataOnElement(const std::vector<INMOST::Tag>& var_tags, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp);
        template<class RandomIt>
        static void GatherDataOnElement(std::initializer_list<INMOST::Tag> var_tags, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp);
        template<class RandomIt>
        static void GatherDataOnElement(const INMOST::Tag& from, const ElementalAssembler& p, RandomIt out, std::initializer_list<int> components);
        template<class RandomIt>
        static void GatherDataOnElement(std::vector<INMOST::Tag>& var_tags, const ElementalAssembler& p, RandomIt out, std::initializer_list<int> components);
        template<class RandomIt>
        static void GatherDataOnElement(std::initializer_list<INMOST::Tag> var_tags, const ElementalAssembler& p, RandomIt out, std::initializer_list<int> components);
        template<class RandomIt>
        static void GatherDataOnElement(const INMOST::Tag& from, const ElementalAssembler& p, RandomIt out, unsigned int physVar);
        template<class RandomIt>
        static void GatherDataOnElement(std::vector<INMOST::Tag>& var_tags, const ElementalAssembler& p, RandomIt out, unsigned int physVar);
        template<class RandomIt>
        static void GatherDataOnElement(std::initializer_list<INMOST::Tag> var_tags, const ElementalAssembler& p, RandomIt out, unsigned int physVar);
    };
}

#include "elemental_assembler.inl"

#endif //CARNUM_ELEMENTALASSEMBLER_H