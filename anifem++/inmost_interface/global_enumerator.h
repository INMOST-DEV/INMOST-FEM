//
// Created by Liogky Alexey on 09.08.2023.
//

#ifndef CARNUM_GLOBAL_ENUMERATOR_H
#define CARNUM_GLOBAL_ENUMERATOR_H

#include "inmost.h"
#include "elemental_assembler.h"
#include <map>
#include "anifem++/interval/ext_array_enum.h"

namespace Ani{

/// Base class for any type of fem global matrix and rhs enumeration
class IGlobEnumeration{
public:
    /// Integral index of d.o.f. used to specify its place in global vector of d.o.f.'s 
    struct EnumerateIndex{
        static const long UnSet = -2;
        static const long UnValid = -1;

        long id = UnSet;
        bool isValid() const { return id >= 0; }
        EnumerateIndex() = default;
        explicit EnumerateIndex(long id): id{id} {}
        void clear() { id = UnSet; }
        friend bool operator==(const EnumerateIndex& a, const EnumerateIndex& b) { return a.id == b.id; }
    };

    struct IsoElementIndex{
        DofT::uchar elem_type = DofT::UNDEF;  ///< type of geometrical element (NODE, EDGE_UNORIENT, EDGE_ORIENT, FACE_UNORIENT, FACE_ORIENT, CELL)

        int var_id = INT_MIN;           ///< number of the variable in physical vector of variables
        int dim_id = 0;                 ///< number of dimensional component of the variable for vector-type variables
        int dim_elem_dof_id = INT_MIN;  ///< element-local d.o.f. index for specific dim of the var (if there is no other variables on the element and dim == 0 it's index of d.o.f on the element)
        IsoElementIndex() = default;
        IsoElementIndex(const IsoElementIndex&) = default;
        IsoElementIndex(DofT::uchar elem_type, int var_id, int dim_id, int dim_elem_dof_id):
            elem_type(elem_type), var_id(var_id), dim_id(dim_id), dim_elem_dof_id(dim_elem_dof_id) {}
        bool isValid() const { return elem_type != DofT::UNDEF && var_id >= 0 && dim_elem_dof_id >= 0; }
        void clear() { elem_type = DofT::UNDEF, var_id = INT_MIN, dim_elem_dof_id = INT_MIN, dim_id = 0; } 
        friend bool operator==(const IsoElementIndex& a, const IsoElementIndex& b) { 
            return a.elem_type == b.elem_type && a.var_id == b.var_id && a.dim_id == b.dim_id && a.dim_elem_dof_id == b.dim_elem_dof_id;  
        }   
    };

    /// Store natural, positional, degree of freedom index
    struct NaturalIndex: public IsoElementIndex{
        INMOST::Element elem; ///< geometrical element (NODE, EDGE, FACE, CELL)
        
        NaturalIndex() = default;
        NaturalIndex(INMOST::Element elem, IsoElementIndex iei): IsoElementIndex(iei), elem{elem} {}
        NaturalIndex(INMOST::Element elem, DofT::uchar elem_type, int var_id, int dim_id, int dim_elem_dof_id):
            IsoElementIndex(elem_type, var_id, dim_id, dim_elem_dof_id), elem(elem) {}
        bool isValid() const { return IsoElementIndex::isValid() && INMOST::isValidHandle(elem.GetHandle()); }    
        void clear() { IsoElementIndex::clear(); elem = INMOST::Element(); }
        friend bool operator==(const NaturalIndex& a, const NaturalIndex& b){ 
            bool is_valid = a.elem.isValid();
            return is_valid == b.elem.isValid() && 
                (!is_valid || (static_cast<const IsoElementIndex&>(a) == static_cast<const IsoElementIndex&>(b) && a.elem == b.elem) ); 
        }
    };

    /// Extended index contained some memory shift information for convinient extraction of degrees of freedom from tags using this index
    struct NaturalIndexExt: public NaturalIndex{
        uint elem_type_shift = uint(-1);  ///< comulative shift to specified geometrical type caused by all variables, might be nonzero for EDGE_ORIENT and FACE_ORIENT
        uint elem_var_type_shift = uint(-1); ///< shift to specified geometrical type caused only by reference variable
        uint var_dof_shift = uint(-1);    ///< shift on the mesh element to d.o.f.'s of the variable
        uint dim_dof_shift = uint(-1);    ///< shift after start of variable d.o.f.'s to d.o.f's of specified dimension component

        NaturalIndexExt() = default;
        NaturalIndexExt(NaturalIndex ni, uint elem_type_shift, uint elem_var_type_shift, uint var_dof_shift, uint dim_dof_shift): 
            NaturalIndex(std::move(ni)), elem_type_shift(elem_type_shift), elem_var_type_shift(elem_var_type_shift), var_dof_shift(var_dof_shift), dim_dof_shift(dim_dof_shift) {}
        /// Number of d.o.f. on the INMOST mesh element if element store all variables
        uint GetElemDofId() const { return elem_type_shift + var_dof_shift + dim_dof_shift + dim_elem_dof_id; } 
        /// Number of d.o.f. on the INMOST mesh element if element store only reference variable
        uint GetSingleVarDofId() const { return elem_var_type_shift + dim_dof_shift + dim_elem_dof_id; }
        void clear() { NaturalIndex::clear();  elem_type_shift = var_dof_shift = dim_dof_shift = uint(-1); }
    };

    /// Iterator moving by contiguous index 
    struct iteratorByVector{
    protected:
        iteratorByVector(const IGlobEnumeration* e, EnumerateIndex st): m_val{st, NaturalIndex(), e} {}
        iteratorByVector(const IGlobEnumeration* e, EnumerateIndex st, NaturalIndex nid): m_val{st, nid, e} {}
    public:
        struct value_type{
        protected:
            EnumerateIndex       eid;
            mutable NaturalIndex nid;
            const IGlobEnumeration* enumeration;
            value_type(EnumerateIndex ei, NaturalIndex ni, const IGlobEnumeration* e): eid{ei}, nid{ni}, enumeration{e} {}
        public:    
            EnumerateIndex GetVecInd() const { return eid; }
            NaturalIndex   GetNatInd() const {
                if (!nid.isValid())
                    nid = enumeration->operator()(eid);
                return nid;    
            }
            friend bool operator==(const value_type& a, const value_type& b){ return a.eid == b.eid; }
            friend class iteratorByVector;
        };

        typedef std::random_access_iterator_tag   iterator_category;
        typedef long difference_type;
        typedef const value_type*  pointer;
        typedef const value_type reference;

        value_type m_val;

        iteratorByVector& operator ++() { ++m_val.eid.id; m_val.nid.clear(); return *this; }
        iteratorByVector  operator ++(int) {iteratorByVector ret(*this); operator++(); return ret;}
        iteratorByVector& operator --() { --m_val.eid.id; m_val.nid.clear(); return *this; }
        iteratorByVector  operator --(int) {iteratorByVector ret(*this); operator--(); return ret;}
        iteratorByVector(const iteratorByVector& other) = default;
        iteratorByVector& operator= (const iteratorByVector & other) = default;
        bool operator ==(const iteratorByVector & other) const { return m_val == other.m_val; }
        bool operator !=(const iteratorByVector & other) const { return !operator==(other); }
        bool operator < (const iteratorByVector & other) const { return m_val.eid.id < other.m_val.eid.id; }
        bool operator > (const iteratorByVector & other) const { return m_val.eid.id > other.m_val.eid.id; }
        bool operator <= (const iteratorByVector & other) const { return m_val.eid.id <= other.m_val.eid.id; }
        bool operator >= (const iteratorByVector & other) const { return m_val.eid.id < other.m_val.eid.id; }
        iteratorByVector& operator += (difference_type n) { m_val.eid.id += n; m_val.nid.clear(); return *this; }
        iteratorByVector& operator -= (difference_type n) { m_val.eid.id -= n; m_val.nid.clear(); return *this; }
        friend iteratorByVector operator+(iteratorByVector& a, difference_type n) { iteratorByVector b(a); return b+=n; }
        friend iteratorByVector operator+(difference_type n, iteratorByVector& a) { return a + n; }
        friend iteratorByVector operator-(iteratorByVector& a, difference_type n) { iteratorByVector b(a); return b-=n; }
        difference_type operator-(const iteratorByVector& b) { return static_cast<difference_type>(m_val.eid.id - b.m_val.eid.id); }
        reference operator*() const { return m_val; }
        pointer operator ->() const { return &m_val; }
        value_type operator[](difference_type n) const {  value_type res(m_val); if (n != 0) { res.eid.id += n; res.nid.clear(); } return res; }

        friend class IGlobEnumeration;
    };
    
    /// Iterator moving geometrical position of degree of freedom, 
    /// cache friendly iterator for accessing to degrees of freedom
    struct iteratorByGeom{
        static const int ANY = -1;
        struct SliceIndex{    
            int var_id = ANY;
            int dim_id = ANY;
            int dim_elem_dof_id = ANY;
            DofT::uchar etype = 0xFF;
            long loc_elem_id = ANY; 
        };
        struct value_type{
        protected:
            NaturalIndex nid;
            mutable EnumerateIndex id;
            const IGlobEnumeration* enumeration;
        public:
            NaturalIndex   GetNatInd() const { return nid; }
            EnumerateIndex GetVecInd() const {
                if (!id.isValid())
                    id = enumeration->operator()(nid);
                return id; 
            }  
            const IGlobEnumeration* GetEnum() const { return enumeration; }
            friend bool operator==(const value_type& a, const value_type& b) { return a.nid == b.nid; }
            friend class iteratorByGeom;
        };
        typedef std::forward_iterator_tag  iterator_category;
        typedef void difference_type;
        typedef const value_type*  pointer;
        typedef const value_type& reference;
    protected:
        SliceIndex sid;
        value_type val;

        iteratorByGeom(const IGlobEnumeration* enumeration, bool is_end = false, 
                        int var_id = ANY, int dim_id = ANY, int dim_elem_dof_id = ANY, DofT::uchar etype = 0xFF, long loc_elem_id = ANY){
            val.enumeration = enumeration;
            sid.var_id = var_id, sid.dim_id = dim_id, sid.dim_elem_dof_id = dim_elem_dof_id, sid.etype = etype, sid.loc_elem_id = loc_elem_id;
            if (!is_end) find_begin();
        }
        void find_begin();
        static bool _choose_var(const IGlobEnumeration* p, NaturalIndex& j, uint st_var, uint last_var);
        static bool _choose_element(const IGlobEnumeration* p, const SliceIndex& sid, NaturalIndex& j, INMOST::ElementType et, INMOST::Storage::integer st, INMOST::Storage::integer ed);


    public:
        bool operator==(const iteratorByGeom& b) const { return val == b.val; }
        bool operator !=(const iteratorByGeom & other) const { return !operator==(other); }
        reference operator*() const { return val; }
        pointer operator ->() const { return &val; }
        iteratorByGeom  operator ++(int) {iteratorByGeom ret(*this); operator++(); return ret;}
        iteratorByGeom& operator ++();  

        friend class IGlobEnumeration;
    };

    /// @brief Construct full index by elem and contiguous index of dof on the element supposing on element stored all variables
    /// @param elem is actual mesh element
    /// @param elem_dof_id contiguous index of dof on the element supposing on element stored all variables
    NaturalIndexExt getFullIndex(INMOST::Element elem, int elem_dof_id) const;
    /// Extend the index by memory shifts info
    NaturalIndexExt getFullIndex(const NaturalIndex& ni) const;
    /// @brief  Construct full index by elem and contiguous index of dof on the element supposing on element stored only variable with number var_id
    /// @param elem is actual mesh element
    /// @param var_id is number of reference variable
    /// @param var_loc_elem_dof_id is contiguous index on the element relative to the reference variable
    NaturalIndexExt getFullIndex(INMOST::Element elem, uint var_id, int var_loc_elem_dof_id) const;

    /// Specify order of rows in global matrix
    /// Must be computable on all elements excluding GHOST Elements
    inline EnumerateIndex OrderR(const NaturalIndex& physIndex) const { return operator()(std::move(physIndex)); };
    /// Specify order d.o.f's in global vector of unknowns
    /// Must be computable on all elements including GHOST Elements
    virtual EnumerateIndex OrderC(const NaturalIndex& physIndex) const = 0;
    /// Map function from position in vector to position on the mesh
    /// Must be computable on all elements excluding GHOST Elements
    virtual NaturalIndex operator()(EnumerateIndex vectorIndex) const  = 0;
    /// Map function from position on mesh to position on the vector
    /// Must be computable on all elements excluding GHOST Elements
    virtual EnumerateIndex operator()(const NaturalIndex& geomIndex) const  = 0;
    /// Get global node index affecting on reordering of d.o.f.'s on elemental parts of rhs and mtx
    /// Must be computable on all elements including GHOST Elements
    virtual INMOST::Storage::integer GNodeIndex(const INMOST::Node& n) const = 0;
    virtual void Clear();
    /// Compute internal values based on current settings
    virtual void setup() = 0;
    virtual operator bool() const { return MatrSize >= 0; }

    /// Create iterator which is incrementing by vector contiguous index
    iteratorByVector beginByVector() const { return iteratorByVector(this, EnumerateIndex(BegInd)); }
    iteratorByVector endByVector()   const { return iteratorByVector(this, EnumerateIndex(EndInd)); }

    /// Create iterator which is incrementing by geometrical position of d.o.f. 
    /// The iterator iterates in cache friendly manner for d.o.f. access from mesh storages
    /// @param var_id is fixed index of variable used by iterator or iteratorByGeom::ANY for iterating over all variables
    /// @param dim_id is fixed index of dimensional component used by iterator or iteratorByGeom::ANY for iterating over all possibilities
    /// @param dim_elem_dof_id is fixed index of dof inside dimensional component used by iterator or iteratorByGeom::ANY for iterating over all possibilities
    /// @param etype is geometrical types used in iterations (e.g. DofT::NODE | DofT::EDGE_UNORIENT)
    /// @param loc_elem_id is fixed local index of element used by iterator or iteratorByGeom::ANY for iterating over all possibilities
    iteratorByGeom beginByGeom(int var_id = iteratorByGeom::ANY, int dim_id = iteratorByGeom::ANY, int dim_elem_dof_id = iteratorByGeom::ANY,  
                                DofT::uchar etype = DofT::NODE | DofT::EDGE_UNORIENT | DofT::EDGE_ORIENT | DofT::FACE_UNORIENT | DofT::FACE_ORIENT | DofT::CELL, 
                                long loc_elem_id = iteratorByGeom::ANY) const { return iteratorByGeom(this, false, var_id, dim_id, dim_elem_dof_id, etype, loc_elem_id); }
    iteratorByGeom endByGeom() const { return iteratorByGeom(this, true); }

    IGlobEnumeration& setMesh(INMOST::Mesh* m) { return mesh = m, *this; }
    IGlobEnumeration& setVars(FemVarContainer vc) { return vars = std::move(vc), *this; }
    INMOST::Mesh* getMeshLink() const { return mesh; }
    const FemVarContainer& getVars() const { return vars; }
    const std::array<unsigned, DofT::NGEOM_TYPES>& getNumDofs() const { return NumDof; }
    const std::array<long, 4>& getNumElem() const { return NumElem; }
    const std::array<long, 4>& getBegElemID() const { return BegElemID; }
    const std::array<long, 4>& getEndElemID() const { return EndElemID; }
    long getMatrixSize() const { return MatrSize; }
    long getBegInd() const { return BegInd; }
    long getEndInd() const { return EndInd; }
    std::array<unsigned, 4> getInmostMeshNumDof() const { return toInmostMeshNumDof(getNumDofs()); }
    INMOST::ElementType getInmostVarElementType() const;
    /// Whether all d.o.f.'s of variables belong to the trivial tetrahedral symmetry group (s1)?
    bool areVarsTriviallySymmetric() const;

    void CopyByEnumeration(const INMOST::Sparse::Vector&            from, INMOST::Tag        union_var_tags_to) const;
    void CopyByEnumeration(const INMOST::Sparse::Vector&            from, std::vector<INMOST::Tag> var_tags_to) const;
    void CopyByEnumeration(INMOST::Tag               union_var_tags_from, INMOST::Sparse::Vector&           to) const;
    void CopyByEnumeration(INMOST::Tag               union_var_tags_from, INMOST::Tag        union_var_tags_to) const;
    void CopyByEnumeration(INMOST::Tag               union_var_tags_from, std::vector<INMOST::Tag> var_tags_to) const;
    void CopyByEnumeration(const std::vector<INMOST::Tag>& var_tags_from, INMOST::Tag        union_var_tags_to) const;
    void CopyByEnumeration(const std::vector<INMOST::Tag>& var_tags_from, INMOST::Sparse::Vector&           to) const;
    /// Save variable with number iVar from vector vars to var_tag
    void CopyVarByEnumeration(const INMOST::Sparse::Vector& vars,         INMOST::Tag var_tag, int iVar) const;
    /// Save variable with number iVar from tag of physical variables vars to var_tag
    void CopyVarByEnumeration(INMOST::Tag union_var_tags_from,            INMOST::Tag var_tag, int iVar) const;
    /// Save variable with number iVar from var_tag to tag of physical variables vars
    void CopyVarByEnumeration(const INMOST::Tag var_tag, int iVar,        INMOST::Tag union_var_tags_to) const;
    /// Copy variable with number iVar from var_tag_from to var_tag_to
    void CopyVarByEnumeration(int iVar, const INMOST::Tag var_tag_from, INMOST::Tag var_tag_to) const;

protected:
    std::array<unsigned, DofT::NGEOM_TYPES> 
        NumDof = {0, 0, 0, 0, 0, 0};
    std::array<long, 4>    
        NumElem = {0, 0, 0, 0},
        BegElemID = {LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX},
        EndElemID = {-1, -1, -1, -1};
    long MatrSize = -1;
    long BegInd = LONG_MAX,
         EndInd = -1;    

    FemVarContainer vars;
    INMOST::Mesh* mesh = nullptr;

    void setupIGlobEnumeration(); 
    
public:
    struct GlobToLocIDMap{
        // std::array<interval_enum<interval_external_memory<INMOST::Storage::integer>>, 4> cont_to_loc;
        std::array<INMOST::Tag, 4> glob_to_loc;
    };
    static GlobToLocIDMap createGlobalToLocalIDMap(INMOST::Mesh* m, INMOST::ElementType et, bool forced_update = false);
    static std::string generateUniqueTagNamePrefix(INMOST::Mesh* m = nullptr);
    static std::array<unsigned, 4> toInmostMeshNumDof(const std::array<unsigned, DofT::NGEOM_TYPES>& numdofs){ return DofTNumDofsToInmostNumDofs(numdofs); }
};

struct TagOwned{
    INMOST::Tag tag;
    void Clear() {
        if (tag.isValid())
            tag = tag.GetMeshLink()->DeleteTag(tag);
    }
    TagOwned() = default;
    TagOwned(TagOwned&& t){
        tag = t.tag;
        t.tag = INMOST::Tag();
    }
    TagOwned& operator=(TagOwned&& t) noexcept{
        if (this == &t) return *this;
        tag = t.tag;
        t.tag = INMOST::Tag();
        return *this;
    }
    ~TagOwned() { Clear(); }
};

/// Construct enumerator by lexicographical ordering of vals in NaturalIndex, i.e. var_id, dim_id, elem_type, elem_global_id, dim_elem_dof_id
/// Compute indexes with O(1) time relative mesh size and O(log(MDOF)) time relative maximum number of dofs at specific element (MDOF)
struct OrderedEnumerator: public IGlobEnumeration{
    static const unsigned char VAR = 0;
    static const unsigned char DIM = 1;
    static const unsigned char ELEM_TYPE = 2;
    static const unsigned char ELEM_ID = 3;
    static const unsigned char DOF_ID = 4;

    INMOST::Storage::integer GNodeIndex(const INMOST::Node& n) const override { return n.GlobalID(); }
    EnumerateIndex operator()(const NaturalIndex& physIndex) const override;
    EnumerateIndex OrderC(const NaturalIndex& physIndex) const override;
    EnumerateIndex index_byMap(const NaturalIndex& physIndex) const;
    NaturalIndex operator()(EnumerateIndex vi) const override;
    void Clear() override;

    OrderedEnumerator& setReverse(bool reverse = true) { return OrderedEnumerator::reverse = reverse, *this; }
    OrderedEnumerator& setArrangment(std::array<unsigned char, 5> order = {VAR, DIM, ELEM_TYPE, ELEM_ID, DOF_ID});
    std::array<TagOwned, 4>& getIndexTags() { return index_tags; }
    const std::array<TagOwned, 4>& getIndexTags() const { return index_tags; }
    std::pair<std::array<unsigned char, 5>, bool> getArrangement() const;
    iteratorByVector getLowerOrderIndex(long global_elem_id, IsoElementIndex iei) const;

    OrderedEnumerator() = default;
    OrderedEnumerator(std::string prefix, INMOST::Mesh* m, FemVarContainer vars, std::array<unsigned char, 5> order = {VAR, DIM, ELEM_TYPE, ELEM_ID, DOF_ID}, bool reverse = false)
        { setArrangment(order).setReverse(reverse).setMesh(m).setVars(vars); setup(prefix);  }
    OrderedEnumerator(INMOST::Mesh* m, FemVarContainer vars, std::array<unsigned char, 5> order = {VAR, DIM, ELEM_TYPE, ELEM_ID, DOF_ID}, bool reverse = false)  
        { setArrangment(order).setReverse(reverse).setMesh(m).setVars(vars).setup();  }  

    void setup(std::string prefix);
    void setup() override { setup(generateUniqueTagNamePrefix(mesh)); }

protected:
    std::array<TagOwned, 4> index_tags;
    GlobToLocIDMap back_maps;
    std::array<unsigned char, 5> perm = {0, 1, 2, 3, 4}; /// <-> VAR, DIM, ELEM_TYPE, ELEM_ID, DOF_ID
    ext_range_array_enum<4, arr_range_func<DofT::NGEOM_TYPES>> loc_emap;
    bool reverse = false;
};

/// Construct enumerator which compute indexes with O(1) time relative to mesh size and number of dofs in variables
struct SimpleEnumerator: public IGlobEnumeration{
    enum ASSEMBLING_TYPE {
        ANITYPE,            //ordered as ELEM_TYPE VAR DIM DOF_ID ELEM_ID without distinction of ORDERED and UNORDERED ELEM_TYPEs
        MINIBLOCKS,         //ordered as ELEM_TYPE ELEM_ID VAR DIM DOF_ID without distinction of ORDERED and UNORDERED ELEM_TYPEs
    }; 

    SimpleEnumerator& setAssemblingType(ASSEMBLING_TYPE t){ return m_t = t, *this;} 
    ASSEMBLING_TYPE getAssemblingType() const { return m_t; }
    std::array<TagOwned, 4>& getIndexTags() { return index_tags; }
    const std::array<TagOwned, 4>& getIndexTags() const { return index_tags; }

    INMOST::Storage::integer GNodeIndex(const INMOST::Node& n) const override { return n.GlobalID(); }
    void Clear() override;
    EnumerateIndex OrderC(const NaturalIndex& physIndex) const override;
    EnumerateIndex operator()(const NaturalIndex& physIndex) const override;
    NaturalIndex operator()(EnumerateIndex vi) const override;

    SimpleEnumerator() = default;
    SimpleEnumerator(ASSEMBLING_TYPE t): m_t(t) {}

    void setup() override { setup(generateUniqueTagNamePrefix(mesh)); }
    void setup(std::string prefix);

protected:
    ASSEMBLING_TYPE m_t = ANITYPE;
    std::array<long, 4> InitElemIndex = {-1, -1, -1, -1};
    std::array<TagOwned, 4> index_tags;
    GlobToLocIDMap back_maps;
};

/// Polymorphic holder for global enumerators of any type
struct GlobEnumeration: public IGlobEnumeration{
    enum ASSEMBLING_TYPE {
        ANITYPE,            //ordered as ELEM_TYPE VAR DIM DOF_ID ELEM_ID without distinction of ORDERED and UNORDERED ELEM_TYPEs
        MINIBLOCKS,         //ordered as ELEM_TYPE ELEM_ID VAR DIM DOF_ID without distinction of ORDERED and UNORDERED ELEM_TYPEs
        NATURAL,            //ordered as VAR DIM ELEM_TYPE ELEM_ID DOF_ID
        DIMUNION,           //ordered as VAR ELEM_TYPE ELEM_ID DOF_ID DIM
        BYELEMTYPE,         //ordered as ELEM_TYPE VAR DIM ELEM_ID DOF_ID
        ETDIMBLOCKS,        //ordered as ELEM_TYPE VAR ELEM_ID DOF_ID DIM
        NOSPECIFIED
    };

    std::shared_ptr<IGlobEnumeration> m_invoker;
    ASSEMBLING_TYPE m_t = NATURAL;

    template<typename T = IGlobEnumeration>
    T* base(){ return reinterpret_cast<T*>(m_invoker.get()); }
    template<typename T = IGlobEnumeration>
    const T* base() const { return reinterpret_cast<T*>(m_invoker.get()); }
    GlobEnumeration& setAssemblingType(ASSEMBLING_TYPE t){ if (m_t == NOSPECIFIED) throw std::runtime_error("Set NOSPECIFIED assembling type is not allowed"); return m_t = t, *this;} 
    ASSEMBLING_TYPE getAssemblingType() const { return m_t; }
    GlobEnumeration& setBaseEnumerator(std::shared_ptr<IGlobEnumeration> base){ return m_invoker = std::move(base), *this; }
    template<typename GlobEnumerationT>
    GlobEnumeration& setBaseEnumerator(GlobEnumerationT&& f, typename std::enable_if<std::is_base_of<IGlobEnumeration, GlobEnumerationT>::value>::type* = 0){ return m_invoker = std::make_unique<GlobEnumerationT>(std::move(f)), *this; }
    template<typename GlobEnumerationT>
    GlobEnumeration& setBaseEnumerator(typename std::enable_if<std::is_base_of<IGlobEnumeration, GlobEnumerationT>::value>::type* = 0){ m_t = NOSPECIFIED; return m_invoker = std::make_unique<GlobEnumerationT>(), *this; }

    INMOST::Storage::integer GNodeIndex(const INMOST::Node& n) const override { return m_invoker->GNodeIndex(n); }
    void Clear() override { if (m_invoker) m_invoker->Clear(); m_invoker.reset(); }
    EnumerateIndex OrderC (const NaturalIndex& physIndex) const override { return m_invoker->OrderC(physIndex); }
    EnumerateIndex operator()(const NaturalIndex& physIndex) const override { return m_invoker->operator()(physIndex); }
    NaturalIndex operator()(EnumerateIndex vi) const override { return m_invoker->operator()(vi); }

    GlobEnumeration() = default;
    GlobEnumeration(ASSEMBLING_TYPE t) { setAssemblingType(t); }

    void setup() override;
};

}

#endif //CARNUM_GLOBAL_ENUMERATOR_H