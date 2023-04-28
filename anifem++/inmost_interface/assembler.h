//
// Created by Liogky Alexey on 29.03.2022.
//

#ifndef CARNUM_ASSEMBLER_H
#define CARNUM_ASSEMBLER_H

#include "inmost.h"
#include "elemental_assembler.h"
#include <iterator>
#include <array>
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <functional>

/// Base class for any type of fem global matrix and rhs enumeration
class ProblemGlobEnumeration{
public:
    struct VectorIndex{
        static const long UnSet = -2;
        static const long UnValid = -1;

        long id = UnSet;
        bool isValid() { return id >= 0; }
        VectorIndex() = default;
        explicit VectorIndex(long id): id{id} {}
    };
    struct GeomIndex{
        INMOST::Element elem;           ///< geometrical element

        int var_id = INT_MIN;           ///< number of the variable in physical vector of variables
        int var_elem_dof_id = INT_MIN;  ///< variable element-local d.o.f. index (if there is no other variables on the element it's index of d.o.f on the element)

        GeomIndex() = default;
        GeomIndex(INMOST::Element elem, int var_id, int loc_elem_dof_id):
            elem(elem), var_id(var_id), var_elem_dof_id(loc_elem_dof_id) {}
    };
    struct GeomIndexExt: public GeomIndex{
        int var_vector_elem_dof_shift = 0;  ///< memory shift on the element if there are other variables on the element

        GeomIndexExt() = default;
        GeomIndexExt(INMOST::Element elem, int var_id, int loc_elem_dof_id, int var_vector_elem_dof_shift = 0):
                GeomIndex(elem, var_id, loc_elem_dof_id), var_vector_elem_dof_shift(var_vector_elem_dof_shift) {}
        /// Get index of d.o.f on the element
        int GetElemDofInd(){ return var_elem_dof_id + var_vector_elem_dof_shift; }
    };

    struct iteratorByVector;
    struct VecIterVal{
    protected:
        VectorIndex vid;
        GeomIndexExt gid;
        const ProblemGlobEnumeration* enumeration;
        VecIterVal(const ProblemGlobEnumeration* enumeration, VectorIndex vid): vid{vid}, gid(), enumeration(enumeration) {}
    public:
        VectorIndex GetVecInd(){ return vid; }
        GeomIndexExt GetGeomInd(){
            if (!gid.elem.isValid())
                gid = enumeration->operator()(vid);
            return gid;
        }
        friend struct iteratorByVector;
        friend bool operator==(const VecIterVal& a, const VecIterVal& b) { return a.vid.id == b.vid.id; }
    };

    struct SliceIteratorData{
        enum Status{
            NONE = 0,
            SAME_VAR = 1,
            SAME_LOC_ELEM_IDS = 2
        };
        const std::array<std::vector<int>, 4>* slice_elem_ids;
        int slice_id = 0;
        Status status = NONE;

        INMOST::Mesh::iteratorElement it;
        int var_id;
        int var_elem_dof_id;
        int var_elem_shift = 0;
        SliceIteratorData(INMOST::Mesh::iteratorElement it, int var_id, int var_elem_dof_id, int var_elem_shift, Status status = NONE, const std::array<std::vector<int>, 4>* slice_elem_ids = nullptr):
            slice_elem_ids(slice_elem_ids), status(status), it(it), var_id(var_id), var_elem_dof_id(var_elem_dof_id), var_elem_shift(var_elem_shift) {}

        GeomIndexExt GetGeomInd() { return {it->getAsElement(), var_id, var_elem_dof_id, var_elem_shift}; }
    };

    struct SliceIterator;
    struct SliceIteratorValue{
    protected:
        SliceIteratorData data;
        long id = VectorIndex::UnSet;
        const ProblemGlobEnumeration* enumeration;
        SliceIteratorValue(const ProblemGlobEnumeration* enumeration, SliceIteratorData data): data(std::move(data)), enumeration(enumeration) {}
    public:
        VectorIndex GetVecInd() {
            if (id == VectorIndex::UnSet)
                id = enumeration->operator()(data.GetGeomInd()).id;
            return VectorIndex(id);
        }
        GeomIndexExt GetGeomInd() { return data.GetGeomInd(); }

        friend bool operator==(const SliceIteratorValue& a, const SliceIteratorValue& b) {
            return a.enumeration == b.enumeration && a.data.it == b.data.it && a.data.var_id == b.data.var_id && a.data.var_elem_dof_id == b.data.var_elem_dof_id;
        }
        friend struct SliceIterator;
    };
    /// Specify order of rows in global matrix
    /// Must be computable on all elements excluding GHOST Elements
    inline VectorIndex OrderR(const GeomIndex& physIndex) const { return operator()(std::move(physIndex)); };
    /// Specify order of vector of unknowns
    /// Must be computable on all elements including GHOST Elements
    virtual VectorIndex OrderC(const GeomIndex& physIndex) const = 0;
    /// Map function from position in vector to position on the mesh
    virtual GeomIndexExt operator()(VectorIndex vectorIndex) const  = 0;
    /// Map function from position on mesh to position on the vector
    virtual VectorIndex operator()(const GeomIndex& geomIndex) const  = 0;
    /// Sets the iterator pointer to the next position, should skip GHOST elements
    virtual void Clear();
    /// Iterate by incrementing SliceIteratorData index according specific policies:
    /// 1. Iterate over all no Ghost geometrical elements according to input INMOST::ElementType mask (internally specified in INMOST::Mesh::iteratorElement it)
    /// 2. If (status & SAME_VAR) iterate only over d.o.f.s of variable with var_id number else iterate over all variables
    /// 3.a) If slice_elem_ids specified and (status & SAME_LOC_ELEM_IDS):
    ///     - if (status & SAME_VAR) iterate only over d.o.f.s of variable with var_id number and with var_elem_dof_id that taken from slice_elem_ids
    ///       else iterate over all indexes of d.o.f (var_elem_shift + var_elem_dof_id) on the element that taken from slice_elem_ids
    /// 3.b) otherwise:
    ///     - if (status & SAME_LOC_ELEM_IDS) iterate only over d.o.f.s that have same var_elem_dof_id value
    ///       else iterate over all available values of var_elem_dof_id
    virtual void IncrementSliceIter(SliceIteratorData& it) const;

    iteratorByVector beginByVector() const {
        iteratorByVector iter(BegInd, this);
        return iter;
    }
    iteratorByVector endByVector() const {
        iteratorByVector iter(EndInd, this);
        return iter;
    }
    SliceIterator beginByGeom(INMOST::ElementType mask = INMOST::NODE|INMOST::EDGE|INMOST::FACE|INMOST::CELL, int var_id = 0, int var_elem_dof_id = 0, SliceIteratorData::Status status = SliceIteratorData::NONE) const;
    SliceIterator beginByGeom(const std::array<std::vector<int>, 4>& slice_elem_dof, INMOST::ElementType mask = INMOST::NODE|INMOST::EDGE|INMOST::FACE|INMOST::CELL, int var_id = 0, int var_elem_dof_id = 0, SliceIteratorData::Status status = SliceIteratorData::NONE) const;
    SliceIterator endByGeom() const { return {this, mesh->EndElement()}; }
    INMOST::ElementType GetGeomMask() const;

    struct SliceIterator{
    protected:
        const ProblemGlobEnumeration* enumeration;
        std::shared_ptr<std::array<std::vector<int>, 4>> slice_elem_ids;
        SliceIteratorData data;
        SliceIterator(const ProblemGlobEnumeration* enumeration, INMOST::Mesh::iteratorElement it, int var_id = 0, int var_elem_dof_id = 0, SliceIteratorData::Status status = SliceIteratorData::NONE):
                enumeration(enumeration), data(it, var_id, var_elem_dof_id, 0, status, nullptr) {}
        SliceIterator(const std::array<std::vector<int>, 4>& slice_elem_ids, const ProblemGlobEnumeration* enumeration, INMOST::Mesh::iteratorElement it, int var_id = 0, int var_elem_dof_id = 0, SliceIteratorData::Status status = SliceIteratorData::NONE):
                SliceIterator(enumeration, it, var_id, var_elem_dof_id, status)
        {
            SliceIterator::slice_elem_ids = std::make_shared<std::array<std::vector<int>, 4>>(slice_elem_ids);
            data.slice_elem_ids = SliceIterator::slice_elem_ids.get();
        }
    public:
        typedef std::input_iterator_tag  iterator_category; //forward_iterator_tag but do not satisfy reference requirement
        typedef SliceIteratorValue value_type;
        typedef void difference_type;
        typedef SliceIteratorValue*  pointer;
        typedef SliceIteratorValue reference;

        SliceIterator& operator ++() { enumeration->IncrementSliceIter(data); return *this; }
        SliceIterator  operator ++(int) {SliceIterator ret(*this); operator++(); return ret;}
        bool operator ==(const SliceIterator & b) const { return data.it == b.data.it && ( (data.it == enumeration->mesh->EndElement()) || (data.var_id == b.data.var_id && data.var_elem_dof_id == b.data.var_elem_dof_id)); }
        bool operator !=(const SliceIterator & other) const { return !operator==(other); }
        reference operator*(){ SliceIteratorValue ind(enumeration, data); return ind; }
        reference operator ->() { return operator*(); }
        friend class ProblemGlobEnumeration;
    };

    /// Iterate by incrementing vector index
    struct iteratorByVector{
    protected:
        unsigned long id;
        const ProblemGlobEnumeration* enumeration;
        iteratorByVector(unsigned long id, const ProblemGlobEnumeration* enumeration): id{id}, enumeration{enumeration} {}
    public:
        typedef std::input_iterator_tag  iterator_category; //random_access_iterator_tag but do not satisfy reference requirement
        typedef VecIterVal value_type;
        typedef long difference_type;
        typedef VecIterVal*  pointer;
        typedef VecIterVal reference;

        iteratorByVector& operator ++() { ++id; return *this; }
        iteratorByVector  operator ++(int) {iteratorByVector ret(*this); operator++(); return ret;}
        iteratorByVector& operator --() { --id; return *this; }
        iteratorByVector  operator --(int) {iteratorByVector ret(*this); operator--(); return ret;}
        iteratorByVector(const iteratorByVector& other) = default;
        iteratorByVector& operator = (const iteratorByVector & other) { if (this != &other) {id = other.id, enumeration = other.enumeration;} return *this; }
        bool operator ==(const iteratorByVector & other) const { return id == other.id; }
        bool operator !=(const iteratorByVector & other) const { return !operator==(other); }
        bool operator < (const iteratorByVector & other) const { return id < other.id; }
        bool operator > (const iteratorByVector & other) const { return id > other.id; }
        bool operator <= (const iteratorByVector & other) const { return id <= other.id; }
        bool operator >= (const iteratorByVector & other) const { return id < other.id; }
        iteratorByVector& operator += (difference_type n) { id += n; return *this; }
        iteratorByVector& operator -= (difference_type n) { id -= n; return *this; }
        friend iteratorByVector operator+(iteratorByVector& a, difference_type n) { iteratorByVector b(a); return b+=n; }
        friend iteratorByVector operator+(difference_type n, iteratorByVector& a) { return a + n; }
        friend iteratorByVector operator-(iteratorByVector& a, difference_type n) { iteratorByVector b(a); return b-=n; }
        friend iteratorByVector operator-(difference_type n, iteratorByVector& a) { return a - n; }
        friend difference_type operator-(const iteratorByVector& a, const iteratorByVector& b) { return static_cast<difference_type>(a.id) - b.id; }
        reference operator*(){ VecIterVal ind(enumeration, VectorIndex(id)); return ind; }
        reference operator ->() { return operator*(); }
        reference operator[](difference_type n) { VecIterVal ind(enumeration, VectorIndex(id+n)); return ind; }
        friend iteratorByVector ProblemGlobEnumeration::beginByVector() const;
        friend iteratorByVector ProblemGlobEnumeration::endByVector() const;
    };


    long NumDof[4] = {0, 0, 0, 0},
         NumElem[4] = {0, 0, 0, 0},
         MinElem[4] = {LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX},
         MaxElem[4] = {-1, -1, -1, -1};
    long MatrSize = -1;
    long BegInd = LONG_MAX,
         EndInd = -1;

    std::shared_ptr<FemExprDescr::ComplexSpaceHelper> unite_var;
    INMOST::Mesh* mesh;
};


struct DefaultEnumeratorFactory{
    enum ASSEMBLING_TYPE {
        MINIBLOCKS,
        ANITYPE
    };
    enum ORDER_TYPE {
        STRAIGHT,
        REVERSE
    };
    INMOST::Mesh* m = nullptr;
    const FemExprDescr* info = nullptr;
    ASSEMBLING_TYPE aType = ANITYPE;
    ORDER_TYPE oType = STRAIGHT;

    DefaultEnumeratorFactory(INMOST::Mesh& mesh, const FemExprDescr& descr, ASSEMBLING_TYPE aType = ANITYPE, ORDER_TYPE oType = STRAIGHT):
        m{&mesh}, info{&descr}, aType(aType), oType(oType) { }

    ///Produce problem enumerator
    ///@param enum_prefix is prefix will be used for enumeration tags
    ///@param is_global_id_updated should be set on true if the mesh was modified from last call of this function
    std::shared_ptr<ProblemGlobEnumeration> build(std::string enum_prefix, bool is_global_id_updated = false);
};


class Assembler {
    using Real = double;
    using Int = int;
    struct WorkMem{
        //internal working arrays
        std::vector<Real> m_A, m_F, m_w;
        std::vector<Int> colindA, colindF, rowA, rowF, m_iw;
        std::vector<Real*> m_args, m_res;
        std::vector<Int> m_indexesR, m_indexesC;
        std::vector<bool> m_Ab; //m_Ab[i] is true if m_A[i] is nonzero according template
        void Clear(){
            m_A.clear(), m_F.clear(), m_w.clear();
            colindA.clear(), colindF.clear(), rowA.clear(), rowF.clear(), m_iw.clear();
            m_args.clear(), m_res.clear();
            m_indexesR.clear(), m_indexesC.clear();
            m_Ab.clear();
        }
    };
    struct OrderTempl{
        int var_id;
        INMOST::ElementType etype;
        int nelem;
        int loc_elem_dof_id;
    };
    WorkMem m_w;
    ElementalAssembler::VarsHelper m_helper;
    std::vector<OrderTempl> orderC;
    std::vector<OrderTempl> orderR;
public:
    struct AssembleTraits{
        bool reorder_nodes = true;
        bool prepare_edges = true;
        bool prepare_faces = true;
    };

    ///\warning Next three functions must have same input parameters
    std::shared_ptr<ElemMatEval> mat_func;  ///< evaluates elemental matrix only
    std::shared_ptr<ElemMatEval> rhs_func;  ///< evaluates elemental right hand-side only
    std::shared_ptr<ElemMatEval> mat_rhs_func;///< evaluates elemental matrix and rhs together

    AssembleTraits m_assm_traits;
    FemExprDescr m_info;                    ///< fem problem description
    std::function<void(ElementalAssembler& p)> m_prob_handler;  ///< set parameters required by elemental evaluator
    std::function<void(ElementalAssembler& p)> initial_value_setter; ///< set initial value for fem-variables (important for nonlinear problems)
    std::shared_ptr<ProblemGlobEnumeration> m_enum; ///< matrix and rhs enumerator
    INMOST::Mesh* m_mesh = nullptr;

    Assembler() = default;
    explicit Assembler(INMOST::Mesh *m): m_mesh(m) {}
    explicit Assembler(INMOST::Mesh& m): Assembler(&m) {}
    ///Allocate internal memory for the problem ( must be called after all Set* calls )
    void PrepareProblem();

    Assembler& SetMesh(INMOST::Mesh& m) { m_mesh = &m; return *this; }
    Assembler& SetEnumerator(std::shared_ptr<ProblemGlobEnumeration> enumeration) { m_enum = std::move(enumeration); return *this; }
    Assembler& SetDataGatherer(std::function<void(ElementalAssembler& p)> problem_handler) { m_prob_handler = std::move(problem_handler); return *this; }
    Assembler& SetProbDescr(FemExprDescr info) { m_info = std::move(info); return *this; }
    Assembler& SetMatFunc (std::shared_ptr<ElemMatEval> func){ mat_func = std::move(func); return *this; }
    Assembler& SetRHSFunc (std::shared_ptr<ElemMatEval> func){ rhs_func = std::move(func); return *this; }
    Assembler& SetMatRHSFunc (std::shared_ptr<ElemMatEval> func){ mat_rhs_func = std::move(func); return *this; }
    Assembler& SetInitValueSetter(std::function<void(ElementalAssembler& p)> init_value_setter) { initial_value_setter = std::move(init_value_setter); return *this; }
    Assembler& pullInitValFrom(INMOST::Tag  tag_x              ) { initial_value_setter = makeInitValueSetter(tag_x); return *this; }
    Assembler& pullInitValFrom(INMOST::Tag* tag_x              ) { initial_value_setter = makeInitValueSetter(tag_x); return *this; }
    Assembler& pullInitValFrom(std::vector<INMOST::Tag> tag_vec) { initial_value_setter = makeInitValueSetter(tag_vec ); return *this; }

    void SaveSolution(const INMOST::Sparse::Vector& from, INMOST::Tag& to) const;
    void SaveSolution(const INMOST::Sparse::Vector& from, std::vector<INMOST::Tag> to) const ;
    void SaveSolution(const INMOST::Tag& from, INMOST::Sparse::Vector& to) const;
    void SaveSolution(const INMOST::Tag& from, INMOST::Tag& to) const;
    void SaveSolution(const INMOST::Tag& from, std::vector<INMOST::Tag> to) const;
    void SaveSolution(const std::vector<INMOST::Tag>& from, INMOST::Tag& to) const;
    void SaveSolution(const std::vector<INMOST::Tag>& from, INMOST::Sparse::Vector& to) const;
    /// Save variable with number iVar from vector vars to var_tag
    void SaveVar(const INMOST::Sparse::Vector& vars, int iVar, INMOST::Tag& var_tag) const;
    /// Save variable with number iVar from tag of physical variables vars to var_tag
    void SaveVar(const INMOST::Tag& vars, int iVar, INMOST::Tag& var_tag) const;
    /// Save variable with number iVar from var_tag to tag of physical variables vars
    void SaveVar(int iVar, const INMOST::Tag& var_tag, INMOST::Tag& vars) const;
    /// Copy variable with number iVar from var_tag_from to var_tag_to
    void CopyVar(int iVar, const INMOST::Tag& var_tag_from, INMOST::Tag& var_tag_to) const;

    int Assemble(INMOST::Sparse::Matrix &matrix, INMOST::Sparse::Vector &rhs, void* user_data = nullptr, double drp_val = 1e-100);
    int AssembleMatrix(INMOST::Sparse::Matrix &matrix, void* user_data = nullptr, double drp_val = 1e-100);
    int AssembleRHS(INMOST::Sparse::Vector &rhs, void* user_data = nullptr, double drp_val = 1e-100);

    /// Begin index for sparse matrix and vector
    int getBegInd() const { return m_enum->BegInd;}
    /// End index for sparse matrix and vector
    int getEndInd() const { return m_enum->EndInd;}

    void Clear();
    const ElementalAssembler::VarsHelper& GetVarHelper() const { return m_helper; }

    template<class TagContainer, class RandomIt>
    void GatherDataOnElement(const TagContainer& from, const INMOST::Cell& cell, RandomIt out, const int* component/*[ncomp]*/, int ncomp) const {
        auto* m = cell.GetMeshLink();
        INMOST::ElementArray<INMOST::Node> nds(m, 4); 
        INMOST::ElementArray<INMOST::Edge> eds(m, 6); 
        INMOST::ElementArray<INMOST::Face> fcs(m, 4);
        int nds_ord[] = {0, 1, 2, 3}, eds_ord[6], fcs_ord[4];
        collect_connectivity_info(cell, nds, eds, fcs, eds_ord, fcs_ord, true, true);
        
        ElementalAssembler::GatherDataOnElement(from, m_helper, cell, fcs, eds, nds, fcs_ord, eds_ord, nds_ord, out, component, ncomp);
    }
    template<class RandomIt>
    void GatherDataOnElement(const INMOST::Tag& from, const INMOST::Cell& cell, RandomIt out, std::initializer_list<int> components = {}) const { GatherDataOnElement<INMOST::Tag, RandomIt>(from, cell, out, components.begin(), components.size()); }
    template<class RandomIt>
    void GatherDataOnElement(const std::vector<INMOST::Tag>& from, const INMOST::Cell& cell, RandomIt out, std::initializer_list<int> components = {}) const { GatherDataOnElement<std::vector<INMOST::Tag>, RandomIt>(from, cell, out, components.begin(), components.size()); }

    /// Create initial_value_setter that sets all initial values of all variables
    static std::function<void(ElementalAssembler& p)> makeInitValueSetter(double val = 0.0);
    static std::function<void(ElementalAssembler& p)> makeInitValueSetter(std::vector<double> val);
    static std::function<void(ElementalAssembler& p)> makeInitValueSetter(INMOST::Tag* tag_x);
    static std::function<void(ElementalAssembler& p)> makeInitValueSetter(INMOST::Tag tag_x);
    static std::function<void(ElementalAssembler& p)> makeInitValueSetter(std::vector<INMOST::Tag> tag_vec);
    /// Get cell and restore tetrahedron connectivity on the cell
    /// @warning this function doesn't aalocate memory for arrays nodes, edges, faces
    static bool collect_connectivity_info(const INMOST::Cell& cell, INMOST::ElementArray<INMOST::Node>& nodes, INMOST::ElementArray<INMOST::Edge>& edges, INMOST::ElementArray<INMOST::Face>& faces, 
                                            int* loc_edge_ids/*[6]*/, int* loc_face_ids/*[4]*/, bool reorder_nodes = true, bool prepare_edges_and_faces = true);
    double GetTimeInitAssembleData();
    double GetTimeFillMapTemplate();
    double GetTimeInitValSet();
    double GetTimeInitUserHandler();
    double GetTimeEvalLocFunc();
    double GetTimePostProcUserHandler();
    double GetTimeFillGlobalStructs();
    double GetTimeTotal();
private:
    void extend_memory_for_fem_func(ElemMatEval* func);
    bool fill_assemble_templates(int nRows,
                                 INMOST::ElementArray<INMOST::Node>& nodes, INMOST::ElementArray<INMOST::Edge>& edges, INMOST::ElementArray<INMOST::Face>& faces, INMOST::Mesh::iteratorCell it,
                                 std::vector<int>& indexesR, std::vector<int>& indexesC,
                                 int* local_edge_index, int* local_face_index);
    void generate_mat_rhs_func();
    void generate_mat_func();
    void generate_rhs_func();

#ifndef NO_ASSEMBLER_TIMERS
    TimerWrap m_timer, m_timer_ttl;
    double  m_time_init_assemble_dat = 0,
            m_time_fill_map_template = 0,
            m_time_init_val_setter = 0,
            m_time_init_user_handler = 0,
            m_time_comp_func = 0,
            m_time_proc_user_handler = 0,
            m_time_set_elemental_res = 0,
            m_timer_total = 0;
#endif
    void reset_timers(){
#ifndef NO_ASSEMBLER_TIMERS
        m_timer_ttl.reset();
        m_timer.reset();
        m_time_init_assemble_dat = 0,
        m_time_fill_map_template = 0,
        m_time_init_val_setter = 0,
        m_time_init_user_handler = 0,
        m_time_comp_func = 0,
        m_time_proc_user_handler = 0,
        m_time_set_elemental_res = 0,
        m_timer_total = 0;
#endif
    }
#ifndef NO_ASSEMBLER_TIMERS
    ElementalAssembler::TimeMessures getTimeMessures(){
        ElementalAssembler::TimeMessures tm;
        tm.m_timer = &m_timer;
        tm.m_time_init_user_handler = &m_time_init_user_handler;
        tm.m_time_comp_func = &m_time_comp_func;
        return tm;
    };
#endif
};

#endif //CARNUM_ASSEMBLER_H
