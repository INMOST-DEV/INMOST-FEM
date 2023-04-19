//
// Created by Liogky Alexey on 28.03.2022.
//

#ifndef CARNUM_ELEMENTALASSEMBLER_H
#define CARNUM_ELEMENTALASSEMBLER_H

#include "inmost.h"
#include <cstddef>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <array>
#include <vector>
#include <set>
#include <initializer_list>
#include <utility>
#include <iostream>
#include <functional>
#include <memory>

///Interface structure for any function evaluating fem matrix and rhs on the element
///This structure expect that any input and output value is sparse matrix in csc-format
///The function wrapped by this struct is responsible for evaluation of elemental matrix
///
/// @see DiscrSpaceHelper

struct ElemMatEval{
    using Real = double;
    using Int = int;

    ///Make numerical evaluation
    ///@param args is working array of const Real pointers, first n_in() elements is pointers on some input parameters
    ///@param res is working array of Real pointers, first n_out() elements will contain pointers of result matrices
    ///@param iw is working integer data
    ///@param w is working real data
    ///@param user_data is some working user-defined data (usually it is not required)
    virtual void operator()(const Real** args, Real** res, Real* w = nullptr, Int* iw = nullptr, void* user_data = nullptr) = 0;

    ///Set minimal required sizes of corresponding arrays
    virtual void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const {
        sz_args = n_in();
        sz_res = n_out();
        sz_w = sz_iw = 0;
    }
    ///Return true if specific user_data required
    virtual bool is_user_data_required() const { return false; }
    virtual size_t n_in() const { return 0; }; ///< Number of input parameters
    virtual size_t n_out() const = 0; ///< Number of output matrices


    virtual Int out_nnz(Int res_id) const { return out_size1(res_id) * out_size2(res_id); }; ///< Expected count of elements in res[res_id] result data array
    virtual Int out_size1(Int res_id) const = 0; ///< first dimension of res[res_id] matrix
    virtual Int out_size2(Int res_id) const = 0; ///< second dimension of res[res_id] matrix
    ///Set csc indexes in colind, row arrays if they are not NULL
    ///@param[in] res_id is index of result parameter
    ///@param[in] colind, row are arrays with out_size2()+1 and out_nnz(res_id) sizes correspondingly
    ///@param[out] colind, row are arrays with csc indexes
    virtual void out_csc(Int res_id, Int* colind, Int* row) const {
        assert(out_nnz(res_id) == out_size1(res_id) * out_size2(res_id) && "CSC format arrays for result arg is not specified");
        auto sz1 = out_size1(res_id), sz2 = out_size2(res_id);
        for (int i = 0; i < sz2+1; ++i) colind[i] = i*sz1;
        if (sz2 > 0) for (int i = 0; i < sz1; ++i) row[i] = i;
        for (int i = 1; i < sz2; ++i)
            std::copy(row, row + sz1, row + sz1*i);
    };


    ///Next functions are just informative and are not used in computations
    ///But we advise to set them to facilitate possible debugging

    virtual Int in_nnz(Int arg_id) const ///< Expected count of elements in args[arg_id] input data array
        { return in_size1(arg_id) * in_size2(arg_id); };
    virtual Int in_size1(Int arg_id) const ///< first dimension of args[args_id] matrix
        { (void)arg_id; return 0; };
    virtual Int in_size2(Int arg_id) const ///< second dimension of args[args_id] matrix
        { (void)arg_id; return 0; };
    ///Set csc indexes in colind, row arrays if they are not NULL
    ///@param[in] arg_id is index of input parameter
    ///@param[in] colind, row are arrays with in_size2()+1 and in_nnz(arg_id) sizes correspondingly
    ///@param[out] colind, row are arrays with csc indexes
    virtual void in_csc(Int arg_id, Int* colind, Int* row) const {
        assert(in_nnz(arg_id) == in_size1(arg_id) * in_size2(arg_id) && "CSC format arrays for input arg is not specified");
        auto sz1 = in_size1(arg_id), sz2 = in_size2(arg_id);
        for (int i = 0; i < sz2+1; ++i) colind[i] = i*sz1;
        if (sz2 > 0) for (int i = 0; i < sz1; ++i) row[i] = i;
        for (int i = 1; i < sz2; ++i)
            std::copy(row, row + sz1, row + sz1*i);
    };


    std::ostream& print_signature(std::ostream& out = std::cout) const;

    ///Helper memory container
    struct Memory{
        Int   *m_iw = nullptr;
        Real  *m_w = nullptr;
        const Real **m_args = nullptr;
        Real **m_res = nullptr;
        void* user_data = nullptr;
    };
};

/// Specialization of ElemMatEval for case when all input and output parameters are dense matrices
template<int nArgs, int nRes, bool UserDataRequired = false>
struct ElemMatEvalDense: public ElemMatEval{
    using Functor = std::function<void(const Real** args, Real** res, Real* w, Int* iw, void* user_data)>;
    Functor m_f;
    std::array<size_t, nArgs> nArgRow, nArgCol;
    std::array<size_t, nRes> nResRow, nResCol;
    size_t nw = 0, niw = 0;

    size_t n_in() const override { return nArgs; }
    Int in_size1(Int res_id) const override{ return nArgRow[res_id]; }
    Int in_size2(Int res_id) const override{ return nArgCol[res_id]; }
    size_t n_out() const override { return nRes; }
    Int out_size1(Int res_id) const override{ return nResRow[res_id]; }
    Int out_size2(Int res_id) const override{ return nResCol[res_id]; }
    void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override {
        sz_args = nArgs; sz_res = nRes; sz_w = nw; sz_iw = niw;
    }
    bool is_user_data_required() const override{ return UserDataRequired; }
    void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
        m_f(args, res, w, iw, user_data);
    }
    ElemMatEvalDense() = default;
    ElemMatEvalDense(Functor f, const std::array<size_t, nArgs>& nArgRow, const std::array<size_t, nArgs>& nArgCol,
                     const std::array<size_t, nRes>& nResRow, const std::array<size_t, nRes>& nResCol, size_t nw = 0, size_t niw = 0):
                     m_f(std::move(f)), nArgRow(nArgRow), nArgCol(nArgCol), nResRow(nResRow), nResCol(nResCol), nw(nw), niw(niw) {}
    void Init(Functor f, const std::array<size_t, nArgs>& nArgRow, const std::array<size_t, nArgs>& nArgCol,
         const std::array<size_t, nRes>& nResRow, const std::array<size_t, nRes>& nResCol, size_t nw = 0, size_t niw = 0){
        m_f = std::move(f);
        ElemMatEvalDense::nArgRow = nArgRow;
        ElemMatEvalDense::nArgCol = nArgCol;
        ElemMatEvalDense::nResRow = nResRow;
        ElemMatEvalDense::nResCol = nResCol;
        ElemMatEvalDense::nw = nw; ElemMatEvalDense::niw = niw;
    }
};

/**
* @brief Hold description about local fem enumeration, d.o.f.s loci and other description of FEM variables
* Vector of d.o.f of physical variable is met to the following rules
*
*      A) First, basis function associated with vertices are enumerated
*         in the same order as the vertices 1,2,3,4;
*
*      B) Second, basis function associated with edges are enumerated
*         in the same order as egdes 12,13,14,23,24 and 34;
*
*      C) Third, basis function associated with faces are enumerated
*         in the same order as faces 123,234,341 and 412;
*
*      D) The vector basis functions with several degrees of freedom per
*         a mesh object (vertex, edge, face) are enumerated first by the
*         corresponding mesh objects (vertex, edge, face) and then by space coordinates, x,y,z;
*
*      E) The vector with several physical variables are numerated first by
*         basis functions of the variables and than by the physical variables order in the vector.
*
*/
struct FemExprDescr{
    ///The structure help to translate index of element in elemental matrix into it's geometrical type
    ///@see ElemMatEval (order rules)
    enum GeomType{
        NODE = 0,
        EDGE = 1,
        FACE = 2,
        CELL = 3,
//        EDGE_ORIENT = 4, ///< currently not supported
//        FACE_ORIENT = 5, ///< currently not supported
        NGEOM_TYPES
    };
    enum DiscrSpaceTypes{
        SimpleType = 0,
        SimpleVectorType = 1,
        UniteType = 2,
        ComplexType = 3
    };
    struct LocalOrder{
        GeomType etype; ///< type of geometrical element
        int nelem;      ///< number of geometrical element on the tetrahedron (from 0 to 3 for NODE or FACE, from 0 to 6 for EDGE, always 0 for CELL)
        int loc_elem_dof_id; ///< number of d.o.f. on specific geometrical element of tetrahedron

        int gid;        ///< contiguous tetrahedron index
    };
    struct SimpleDiscrSpace;
    struct ShiftedSpaceHelperView;
    struct BaseDiscrSpaceHelper{
        virtual int ActualType() const = 0; ///< @return Child type number (usually one of number from DiscrSpaceTypes)
        virtual int Dim() const = 0;    ///< @return dimension of FEM space
        virtual int NumDof(GeomType t) const = 0;   ///< @return number of d.o.f. on specific geometrical element type
        virtual int NumDof() const; ///< @return NumDof(NODE) + NumDof(EDGE) + NumDof(FACE) + NumDof(CELL)
        virtual int NumDofOnTet(GeomType t) const = 0; ///< @return number of d.o.f. on specific geometrical element type on a tetrahedron
        virtual int NumDofOnTet() const; ///< @return total number of d.o.f.s on tetrahedron
        virtual LocalOrder LocalOrderOnTet(int dof) const = 0; ///< convert tetrahedron contiguous index into LocalOrder index
        virtual LocalOrder LocalOrderOnTet(GeomType t, int ldof) const; ///< convert element-tetrahedron contiguous index (from 0 to NumDofOnTet(t)-1) into LocalOrder index
        virtual GeomType TypeOnTet(int dof) const; ///< get type of geometrical element from tetrahedron contiguous index
        virtual GeomType Type(int dof) const = 0; ///< get type of geometrical element from contiguous index (from 0 to NumDof() - 1)
        virtual int TetDofID(GeomType t, int ldof) const = 0; ///< get tetrahedron contiguous index from element type and element-tetrahedron contiguous index
        virtual bool DefinedOn(GeomType t) const { return NumDof(t) > 0; }; ///< is FEM Space has d.o.f.s on specific element type
        ///@return optionally pair of {driver_id, fem_id}
        /// where driver_id specify origin implementation of DiscrSpace(, for example, from CasAdi library or from Ani library)
        /// and fem_id that specify fem basis function, depends on driver_id
        /// or {UINT_MAX, UINT_MAX} if this values are not specified
        virtual std::pair<unsigned int, unsigned int> OriginFemType() const { return {UINT_MAX, UINT_MAX}; }
        std::array<int, NGEOM_TYPES> NumDofsOnTet() const; ///< @return values NumDofOnTet(t) for all types t
        std::array<bool, NGEOM_TYPES> GetGeomMask() const; ///< @return values DefinedOn(t) for all types t
        /// If part of fem space that define dim-th component of the full FEM variable is some one-dimensional simple FEM space, than it will be returned
        /// otherwise will return nullptr
        /// @warning returned space doesn't takes into account the shift of degrees of freedom from the enclosing space
        /// @see GetNestedComponent
        virtual const SimpleDiscrSpace* GetComponent(int dim) const { (void)dim; return nullptr; }
        /// @param[in] ext_dims[ndims] is array of size ndims of external component numbers
        /// e.g. for space ComplexSpaceHelper : (UniteDiscrSpace, SimpleDiscrSpace, (SimpleVectorDiscrSpace[3]: SimpleDiscrSpace) )
        /// ext_dims = 2, 1 defines ComplexSpaceHelper -> 2-SimpleVectorDiscrSpace[] -> 1-SimpleVectorDiscrSpace[1]=SimpleDiscrSpace
        /// @param ndims is size of ext_dims array
        /// @return a nested subspace that takes into account the shift of degrees of freedom from the enclosing space
        ShiftedSpaceHelperView GetNestedComponent(const int* ext_dims = nullptr, int ndims = 0) const;
        virtual void GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const = 0;
        template<int N>
        ShiftedSpaceHelperView GetNestedComponent(const std::array<int, N>& ext_dims) const { return GetNestedComponent(ext_dims.data(), N); }
    };
    /// Store elementary FEM spaces
    struct UniteDiscrSpace: public BaseDiscrSpaceHelper{
        std::array<int, NGEOM_TYPES+1> m_shiftDof = {0};
        std::array<int, NGEOM_TYPES+1> m_shiftTetDof = {0};

        int m_dim;                                  ///< dimension of discrete space
        unsigned int driver_id = UINT_MAX;          ///< optional, specify origin implementation of DiscrSpace, for example, from CasAdi library or from Ani library
        unsigned int fem_id = UINT_MAX;             ///< optional, specify fem basis function, depends on driver_id

        UniteDiscrSpace(int dim, std::array<int, NGEOM_TYPES> NumDofs, unsigned int driver_id = UINT_MAX, unsigned int fem_id = UINT_MAX);

        int Dim() const override{ return m_dim; }
        int NumDof(GeomType t) const override { return m_shiftDof[t+1] - m_shiftDof[t]; }
        int NumDof() const override { return m_shiftDof[NGEOM_TYPES] - m_shiftDof[0]; }
        GeomType Type(int dof) const override;
        int NumDofOnTet(GeomType t) const override { return m_shiftTetDof[t+1] - m_shiftTetDof[t]; }
        int NumDofOnTet() const override { return m_shiftTetDof[NGEOM_TYPES] - m_shiftTetDof[0]; }
        LocalOrder LocalOrderOnTet(int dof) const override;
        LocalOrder LocalOrderOnTet(GeomType t, int ldof) const override;
        GeomType TypeOnTet(int dof) const override;
        int TetDofID(GeomType t, int ldof) const override { return m_shiftTetDof[t] + ldof; }
        std::pair<unsigned int, unsigned int> OriginFemType() const override { return {driver_id, fem_id}; }
        int ActualType() const override { return UniteType; }
        void GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const override;
    };
    /// Store elementary one-dimensional FEM spaces
    struct SimpleDiscrSpace: public UniteDiscrSpace{
        using CoordFunc = void (*)(int odf, const double* /*[4][3]*/ nodes, double*/*[3]*/ coord);
        CoordFunc coordAt; ///< return coordinate of specific d.o.f. (used by Dirichlet BC setter)
        SimpleDiscrSpace(std::array<int, NGEOM_TYPES> NumDofs, unsigned int driver_id = UINT_MAX, unsigned int fem_id = UINT_MAX):
                UniteDiscrSpace(1, NumDofs, driver_id, fem_id) {}
        int ActualType() const override { return SimpleType; }
        const SimpleDiscrSpace* GetComponent(int dim) const override { return (dim == 0) ? this : nullptr; }
    };
    /// Store vector of same FEM space type
    struct SimpleVectorDiscrSpace: public BaseDiscrSpaceHelper{
        std::shared_ptr<BaseDiscrSpaceHelper> base;
        int m_dim;
        SimpleVectorDiscrSpace(int dim, std::shared_ptr<BaseDiscrSpaceHelper> space): base{std::move(space)}, m_dim(dim) {}
        int Dim() const override{ return m_dim * base->Dim(); }
        int NumDof(GeomType t) const override { return m_dim * base->NumDof(t); }
        int NumDof() const override { return m_dim * base->NumDof(); }
        GeomType Type(int dof) const override { return base->Type(dof % base->NumDof()); }
        int NumDofOnTet(GeomType t) const override { return m_dim * base->NumDofOnTet(t); }
        int NumDofOnTet() const override { return m_dim * base->NumDofOnTet(); }
        GeomType TypeOnTet(int dof) const override { return base->TypeOnTet(dof % base->NumDofOnTet()); }
        int TetDofID(GeomType t, int ldof) const override;
        LocalOrder LocalOrderOnTet(int dof) const override;
        std::pair<unsigned int, unsigned int> OriginFemType() const override { return base->OriginFemType(); }
        int ActualType() const override { return SimpleVectorType; }
        const SimpleDiscrSpace* GetComponent(int dim) const override;
        void GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const override;
    };
    /// Store compound FEM space
    struct ComplexSpaceHelper: public BaseDiscrSpaceHelper{
        std::vector<std::shared_ptr<BaseDiscrSpaceHelper>> m_spaces;
        unsigned int driver_id = UINT_MAX, fem_id = UINT_MAX;
        std::vector<int> m_dimShift;
        std::array<std::vector<int>, NGEOM_TYPES> m_spaceNumDofTet;
        std::array<std::vector<int>, NGEOM_TYPES> m_spaceNumDof;
        std::vector<int> m_spaceNumDofs, m_spaceNumDofsTet;
        explicit ComplexSpaceHelper(std::vector<std::shared_ptr<BaseDiscrSpaceHelper>> spaces, unsigned int driver_id = UINT_MAX, unsigned int fem_id = UINT_MAX);

        int Dim() const override{ return m_dimShift[m_spaces.size()]; }
        int NumDof(GeomType t) const override { return m_spaceNumDof[t][m_spaces.size()] - m_spaceNumDof[t][0]; }
        GeomType Type(int dof) const override;
        int NumDofOnTet(GeomType t) const override { return m_spaceNumDofTet[t][m_spaces.size()] - m_spaceNumDofTet[t][0]; }
        GeomType TypeOnTet(int dof) const override;
        LocalOrder LocalOrderOnTet(int dof) const override;
        int TetDofID(GeomType t, int ldof) const override;
        LocalOrder LocalOrderOnTet(GeomType t, int ldof) const override;
        std::pair<unsigned int, unsigned int> OriginFemType() const override { return {driver_id, fem_id}; }
        int ActualType() const override { return ComplexType; }
        const SimpleDiscrSpace* GetComponent(int dim) const override;
        void GetNestedComponent(const int* ext_dims, int ndims, ShiftedSpaceHelperView& view) const override;
    };
    //A structure for accessing nested components of a fem space, 
    //taking into account the shift of d.o.f.s of this component within the entire space.
    //Designed for easy access to degrees of freedom of nested subspaces
    struct ShiftedSpaceHelperView{
        const BaseDiscrSpaceHelper* base = nullptr;
        std::array<int, NGEOM_TYPES> m_shiftNumDofTet = {0};
        
        int ActualType() const { return base->ActualType(); }
        int Dim() const { return base->Dim(); }
        int NumDof(GeomType t) const { return base->NumDof(t); }
        int NumDof() const { return base->NumDof(); }
        int NumDofOnTet(GeomType t) const { return base->NumDofOnTet(t); }
        int NumDofOnTet() const { return base->NumDofOnTet(); }
        /// convert tetrahedron contiguous index into shifted LocalOrder index
        LocalOrder LocalOrderOnTet(int dof) const;
        /// convert element-tetrahedron contiguous index (from 0 to NumDofOnTet(t)-1) into shifted LocalOrder index
        LocalOrder LocalOrderOnTet(GeomType t, int ldof) const; 
        GeomType TypeOnTet(int dof) const { return base->TypeOnTet(dof); }
        GeomType Type(int dof) const { return base->Type(dof); }
        int TetDofID(GeomType t, int ldof) const { return base->TetDofID(t, ldof); }
        bool DefinedOn(GeomType t) const { return base->DefinedOn(t); }
        std::pair<unsigned int, unsigned int> OriginFemType() const { return base->OriginFemType(); }
        std::array<int, NGEOM_TYPES> NumDofsOnTet() const { return base->NumDofsOnTet(); }
        std::array<bool, NGEOM_TYPES> GetGeomMask() const { return base->GetGeomMask(); }
        void Clear(){ base = nullptr; std::fill(m_shiftNumDofTet.begin(), m_shiftNumDofTet.end(), 0); }
    };
    using DiscrSpaceHelper = std::shared_ptr<BaseDiscrSpaceHelper>;

    struct VarDescr{
        DiscrSpaceHelper odf;           ///< map from matrix order to geometrical d.o.f. type
        std::string name;               ///< optional, name of physical variable
    };

    std::vector<VarDescr> base_funcs;       ///< vector of description of physical variables
    std::vector<VarDescr> test_funcs;       ///< vector of description of test functions

    void Clear() { base_funcs.clear(), test_funcs.clear(); }
    /// Add variable to physical vector
    auto PushVar(DiscrSpaceHelper odf, std::string name){ base_funcs.push_back(VarDescr{std::move(odf), std::move(name)}); }
    /// Add test func space to vector of test functional spaces
    auto PushTestFunc(DiscrSpaceHelper odf, const std::string& name = ""){ test_funcs.push_back(VarDescr{std::move(odf), name}); }
    /// Get geometrical mask of physical vector of variables
    std::array<bool, NGEOM_TYPES> GetBaseGeomMask() const;
    int NumVars() { return base_funcs.size(); }
    /// Get number of d.o.f.s on all geometrical types for physical vector
    std::array<int, NGEOM_TYPES> NumDofs() const;
    /// Get number of d.o.f.s on all geometrical types on tetrahedron for physical vector
    std::array<int, NGEOM_TYPES> NumDofsOnTet() const;
    /// Total number of d.o.f.s on tetrahedron
    int NumDofOnTet() const;
};

///Reorder nodes of tetrahedron in positive order
void reorderNodesOnTetrahedron(INMOST::ElementArray<INMOST::Node>& nodes);

/// Helper structure for measuring the running time of code blocks
struct TimerWrap{
    double time_point = 0;
    void reset(){ time_point = Timer(); }
    double elapsed() const { return Timer() - time_point; }
    double elapsed_and_reset() { double res = elapsed(); reset(); return res; }
};

/// Wrapper for internal global assembler information to give user flexible interface to run elemental matrix assembling
class ElementalAssembler {
public:
#ifndef NO_ASSEMBLER_TIMERS
    struct TimeMessures{
        TimerWrap* m_timer;
        double *m_time_init_user_handler,
                *m_time_comp_func;
    };
#endif
    struct VarsHelper{
        FemExprDescr* descr;     ///< vector of description of physical variables
        std::vector<double> initValues; ///< vector of current values of d.o.f.s (important for nonlinear problems)
        std::vector<int> base_MemOffsets;
        std::vector<int> test_MemOffsets;
        bool same_template = false;

        /// Number of physical variables
        int NumBaseVars() const { return descr->base_funcs.size(); }
        ///@see begin, end
        int BaseOffset(int nVar) const { return base_MemOffsets[nVar]; }
        /// Number of test FEM spaces
        int NumTestVars() const { return descr->test_funcs.size(); }
        int TestOffset(int nVar) const { return test_MemOffsets[nVar]; }
        /// Is Base FEM space equal to Test FEM space ?
        bool IsSameTemplate() const { return same_template; }
        /// Iterator of current values of d.o.f.s for specific number of variable in vector of physical variables
        double * begin(int iVar) { return initValues.data() + BaseOffset(iVar); }
        double * end(int iVar) { return initValues.data() + BaseOffset(iVar+1); }
        void Clear();
    };
    /// Helper structure to store potentially sparse elemental matrices and rhs
    struct SparsedDat{
        ElemMatEval::Real* dat = nullptr;
        ElemMatEval::Int* colind = nullptr;
        ElemMatEval::Int* row = nullptr;
        ElemMatEval::Int sz1 = 0, sz2 = 0, nnz = 0;
    };
    enum Type{
        RHS = 0,    ///< run to compute elemental rhs
        MAT = 1,    ///< run to compute elemental matrix
        MAT_RHS = 2,///< run to compute elemental matrix and rhs together (common case)
        NTYPES
    };
    Type f_type = NTYPES;   ///< type of running
    ElemMatEval* func;      ///< elemental matrix and rhs evaluator
    SparsedDat loc_m, loc_rhs;  ///< memory to save elemental matrix and rhs
    ElemMatEval::Memory mem;    ///< some additional memory for evaluator
    std::vector<bool>* loc_mb;  ///< indicator of zeros (including structured) in elemental matrix
    VarsHelper* vars;           ///< description of FEM spaces of variables and initial guess
    const INMOST::Mesh * m = nullptr;
    const INMOST::ElementArray<INMOST::Node>* nodes;    ///< nodes of current tetrahedron
    const INMOST::ElementArray<INMOST::Edge>* edges;    ///< edges of current tetrahedron, @see local_edge_index
    const INMOST::ElementArray<INMOST::Face>* faces;    ///< faces of current tetrahedron, @see local_face_index
    INMOST::Mesh::iteratorCell cell;                    ///< current tetrahedron
    const int* indexesC = nullptr, *indexesR = nullptr; ///< vectors of global indexes of rows and columns for elemental matrix and rhs
    int* local_edge_index = nullptr;  ///< to get node-consistent (that follows the rules listed above FemExprDescr) k-th edge need to use edges[local_edge_index[k]]
    int* local_face_index = nullptr;  ///< to get node-consistent (that follows the rules listed above FemExprDescr) k-th face need to use faces[local_face_index[k]]

    ElementalAssembler() = default;
    void init(ElemMatEval* _f, Type _f_type,
              SparsedDat _loc_m, SparsedDat _loc_rhs, ElemMatEval::Memory _mem, std::vector<bool> * _loc_mb,
              VarsHelper* _vars, const INMOST::Mesh * const _m,
              const INMOST::ElementArray<INMOST::Node>* _nodes, const INMOST::ElementArray<INMOST::Edge>* edges, const INMOST::ElementArray<INMOST::Face>* faces,
              INMOST::Mesh::iteratorCell _cell,
              const int* _indexesC, const int* indexesR,
              int* _local_edge_index, int* _local_face_index
#ifndef NO_ASSEMBLER_TIMERS
              , TimeMessures _tmes
#endif
              );
    ElementalAssembler(ElemMatEval* f, Type f_type,
            SparsedDat loc_m, SparsedDat loc_rhs, ElemMatEval::Memory _mem, std::vector<bool> *loc_mb,
            VarsHelper* vars, const INMOST::Mesh * const m,
            const INMOST::ElementArray<INMOST::Node>* nodes, const INMOST::ElementArray<INMOST::Edge>* edges, const INMOST::ElementArray<INMOST::Face>* faces,
            INMOST::Mesh::iteratorCell cell,
            const int* indexesC, const int* indexesR,
            int* local_edge_index, int* local_face_index
#ifndef NO_ASSEMBLER_TIMERS
            , TimeMessures tmes
#endif
            );
    /// Run evaluator
    ///@param args are input parameters to be delivered into evaluator
    void compute(const ElemMatEval::Real** args);
    /// Run evaluator
    ///@param args are input parameters to be delivered into evaluator
    ///@param user_data is user specific data to be delivered into evaluator
    void compute(const ElemMatEval::Real** args, void* user_data);
    ///Update some internal data and set zeros to loc_m and loc_rhs
    void update(const INMOST::ElementArray<INMOST::Node>& nodes);
    ///Return data for 3x4 col-major matrix of coordinates of nodes of tetrahedron
    double* get_nodes();
    const double* get_nodes() const;
    //sepX and sepY is set of numbers columns and rows before thats we set separator = "|"
    std::ostream& print_matrix_and_rhs_arbitrarily(std::ostream& out = std::cout, const std::set<int>& sepX = {}, const std::set<int>& sepY = {}) const;
    void print_input(const double** args) const;

private:
    ElemMatEval::Real _nn_p[4][3];
    void make_result_buffer();
    void densify_result();
#ifndef NO_ASSEMBLER_TIMERS
    TimeMessures m_tmes;
#endif

    static std::function<INMOST::Storage::real(
        const INMOST::Tag&, const FemExprDescr::LocalOrder&, 
        const INMOST::Cell&, const INMOST::ElementArray<INMOST::Face>&, const INMOST::ElementArray<INMOST::Edge>&, const INMOST::ElementArray<INMOST::Node>&, 
        const int* /*local_face_index[4]*/, const int* /*local_edge_index[6]*/, const int* /*local_node_index[4]*/ )> 
        GeomTakerDOF(FemExprDescr::GeomType t);

public:
    ///Take specified degree of freedom on cell
    static INMOST::Storage::real TakeElementDOF(const INMOST::Tag& tag, const FemExprDescr::LocalOrder& lo, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const int local_face_index[4], const int local_edge_index[6], const int local_node_index[4]);

    ///Gather all odfs of the variable component on cell from tag into container out 
    template<class RandomIt>
    static void GatherDataOnElement(
        const INMOST::Tag& from, const ElementalAssembler::VarsHelper& vars, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const int local_face_index[4], const int local_edge_index[6], const int local_node_index[4],
        RandomIt out, const int* component/*[ncomp]*/, int ncomp);

    ///Gather all odfs of the variable component on cell from tags into container out
    ///supposed for every physical variable correspond different tag
    template<class RandomIt>
    static void GatherDataOnElement(
        const std::vector<INMOST::Tag>& from, const ElementalAssembler::VarsHelper& vars, 
        const INMOST::Cell& cell, const INMOST::ElementArray<INMOST::Face>& faces, const INMOST::ElementArray<INMOST::Edge>& edges, const INMOST::ElementArray<INMOST::Node>& nodes, 
        const int local_face_index[4], const int local_edge_index[6], const int local_node_index[4],
        RandomIt out, const int* component/*[ncomp]*/, int ncomp);

    template<class RandomIt, int N = 0>
    static void GatherDataOnElement(const INMOST::Tag& from, ElementalAssembler& p, RandomIt out, const std::array<int, N>& components){
        static const int local_node_index[4] = {0, 1, 2, 3};
        GatherDataOnElement<RandomIt>(from, *p.vars, p.cell.operator->(), *p.faces, *p.edges, *p.nodes, p.local_face_index, p.local_edge_index, local_node_index, out, components.data(), N);
    }
    template<class RandomIt>
    static void GatherDataOnElement(const INMOST::Tag& from, ElementalAssembler& p, RandomIt out, const std::initializer_list<int>& components){
        static const int local_node_index[4] = {0, 1, 2, 3};
        GatherDataOnElement<RandomIt>(from, *p.vars, p.cell.operator->(), *p.faces, *p.edges, *p.nodes, p.local_face_index, p.local_edge_index, local_node_index, out, components.begin(), components.size());
    }
    template<class RandomIt>
    static void GatherDataOnElement(const INMOST::Tag& from, ElementalAssembler& p, RandomIt out, int nPhysVar){
        GatherDataOnElement<RandomIt, 1>(from, p, out, std::array<int, 1>{nPhysVar});
    }

    template<class RandomIt, std::size_t N = 0>
    static void GatherDataOnElement(const std::vector<INMOST::Tag>& from, ElementalAssembler& p, RandomIt out, const std::array<int, N>& components){
        static const int local_node_index[4] = {0, 1, 2, 3};
        GatherDataOnElement<RandomIt>(from, *p.vars, p.cell.operator->(), *p.faces, *p.edges, *p.nodes, p.local_face_index, p.local_edge_index, local_node_index, out, components.data(), N);
    }
    template<class RandomIt>
    static void GatherDataOnElement(const std::vector<INMOST::Tag>& from, ElementalAssembler& p, RandomIt out, int nPhysVar){
        GatherDataOnElement<RandomIt, 1>(from, p, out, std::array<int, 1>{nPhysVar});
    }
};

bool operator==(const FemExprDescr::DiscrSpaceHelper& a, const FemExprDescr::DiscrSpaceHelper& b);

#include "elemental_assembler.inl"

#endif //CARNUM_ELEMENTALASSEMBLER_H
