//
// Created by Liogky Alexey on 10.01.2023.
//

#ifndef CARNUM_ANI_TETDOFMAP_H
#define CARNUM_ANI_TETDOFMAP_H

#include <initializer_list>
#include <array>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <type_traits>

namespace Ani{
    namespace DofT{
        using uchar = unsigned char;
        using uint = unsigned;
        /// @brief types of geometrical elements
        static constexpr uchar UNDEF = 0x0;
        static constexpr uchar NODE = 0x1;
        static constexpr uchar EDGE_UNORIENT = 0x2;  ///< if sign of d.o.f. doesn't depends on edge orientation
        static constexpr uchar EDGE_ORIENT = 0x4;    ///< if sign of d.o.f. depends on edge orientation
        static constexpr uchar EDGE = 0x2 | 0x4;
        static constexpr uchar FACE_UNORIENT = 0x8;  ///< if sign of d.o.f. doesn't depends on face orientation
        static constexpr uchar FACE_ORIENT = 0x10;   ///< if sign of d.o.f. depends on face orientation
        static constexpr uchar FACE = 0x8 | 0x10;
        static constexpr uchar CELL = 0x20;
        static constexpr uchar NGEOM_TYPES = 6;
        
        ///Convert primitive geom type, i.e. equals to 2^i into sequential number
        ///@return sequential number of geom element, i.e. return i for 2^i or negative number if error occured
        ///@warning etype must be primitive geom type, i.e. equals to 2^i
        inline int GeomTypeToNum(uchar etype) { return (etype != UNDEF) ? ilogb(etype) : -1; }
        /// Return maximal dimension of specified geometrical type (e.g. for NODE|EDGE will be returned 2), for UNDEF return -1
        inline int GeomTypeDim(uchar etype) {
            static constexpr int lookup[] = {-1, 0, 1, 1, 2, 2, 3}; 
            return lookup[GeomTypeToNum(etype) + 1]; 
        }
        /// Convert sequential number in primitive geometric type
        inline uchar NumToGeomType(int num) { return (num >= 0) ? (1U << num) : UNDEF; }
        /// Convert dimension of geometric element to type of this element
        inline uchar DimToGeomType(int dim) {
            static constexpr uchar lookup[] = {NODE, EDGE, FACE, CELL};
            return (dim >= 0) ? lookup[dim] : UNDEF;
        }
        inline bool GeomTypeIsValid(uchar etype) { return etype < (1 << NGEOM_TYPES); }
        /// Return count of elements of specified primitive type on tetrahedron
        inline uchar GeomTypeTetElems(uchar etype) { 
            static constexpr int sz[] = {0, 4, 6, 4, 1};
            return sz[GeomTypeDim(etype) + 1];
        }
        /// To store local d.o.f. map element
        struct TetOrder{
            uint  gid;  ///< contiguous tetrahedron d.o.f. index
            operator uint () const { return gid; }
            TetOrder(): gid(uint(-1)) {}
            TetOrder(uint id): gid(id) {}
            inline bool isValid() const { return gid != uint(-1); }
            bool operator==(const TetOrder& a) { return gid == a.gid; }
            bool operator!=(const TetOrder& a) { return !(*this == a); }
        };
        struct ComponentTetOrder{
            TetOrder gid;   ///< contigous index of dof inside embracing space
            TetOrder cgid;  ///< contigous index of dof inside component
            uint part_id;   ///< component id, number of component in array of components
        };

        struct LocGeomOrder{
            uint  leid;     ///< number of d.o.f. on specific geometrical element of tetrahedron
            uchar etype;    ///< type of geometrical element
            uchar nelem;    ///< number of geometrical element on the tetrahedron (from 0 to 3 for NODE or FACE, from 0 to 6 for EDGE, always 0 for CELL)
            LocGeomOrder(): leid(uint(-1)), etype(UNDEF), nelem(0) {}
            LocGeomOrder(uchar etype, uchar nelem, uint leid): leid{leid}, etype{etype}, nelem{nelem} {} 
            inline bool isValid() const { return etype != UNDEF && leid != uint(-1) &&  GeomTypeIsValid(etype) && leid < GeomTypeTetElems(etype); }
            bool operator==(const LocGeomOrder& a) { return etype == a.etype && nelem == a.nelem && leid == a.leid; }
            bool operator!=(const LocGeomOrder& a) { return !(*this == a); }
        };
        /// Store map between specific tetrahedron d.o.f. and its geometrical locus on tetrahedron
        struct LocalOrder{
            uint  gid;      ///< contiguous tetrahedron index

            uint  leid;     ///< number of d.o.f. on specific geometrical element of tetrahedron
            uchar etype;    ///< type of geometrical element
            uchar nelem;    ///< number of geometrical element on the tetrahedron (from 0 to 3 for NODE or FACE, from 0 to 6 for EDGE, always 0 for CELL) 
            LocalOrder(): gid(uint(-1)), leid(uint(-1)), etype(UNDEF), nelem(0) {}
            LocalOrder(TetOrder tid, LocGeomOrder lid): gid(tid.gid), leid(lid.leid), etype(lid.etype), nelem(lid.nelem) {} 
            LocalOrder(uint gid, uchar etype, uchar nelem, uint leid): gid(gid), leid(leid), etype(etype), nelem(nelem) {} 
            inline LocGeomOrder getGeomOrder() const { return LocGeomOrder(etype, nelem, leid); }
            inline TetOrder getTetOrder() const { return TetOrder(gid); }
            inline bool isValid() const { return getTetOrder().isValid() && getGeomOrder().isValid(); }
            bool operator==(const LocalOrder& a) const{ return gid == a.gid; } 
            bool operator!=(const LocalOrder& a) { return !(*this == a); }
        };

        /// Efficient handler for convenient selection of geometric parts of the tetrahedron
        struct TetGeomSparsity{
        protected:
            std::array<uchar, 4> elems = {0, 0, 0, 0}; //nodes, edges, faces, cells
        public:
            void clear() { *this = TetGeomSparsity(); }
            bool empty() const { return (elems[0] | elems[1] | elems[2] | elems[3]) == 0; }
            bool empty(uchar elem_dim) const { return elems[elem_dim] == 0; }
            TetGeomSparsity& setCell(bool with_closure = false);
            TetGeomSparsity& unsetCell(bool with_closure = false);
            TetGeomSparsity& setFace(int iface, bool with_closure = false);
            TetGeomSparsity& unsetFace(int iface, bool with_closure = false);
            TetGeomSparsity& setEdge(int iedge, bool with_closure = false);
            TetGeomSparsity& unsetEdge(int iedge, bool with_closure = false);
            TetGeomSparsity& setNode(int inode);
            TetGeomSparsity& unsetNode(int inode);
            /// @brief Turn on some part of tetrahedron
            /// @param elem_dim is dimension of geometric part of tetrahedron to be turned on
            /// @param ielem is number of geomentric element over geomentric elements with same dimension
            /// @param with_closure should we also turn on all less dimension elements belonging to the chosen geomentric part
            TetGeomSparsity& set(uchar elem_dim, int ielem, bool with_closure = false);
            TetGeomSparsity& set(TetGeomSparsity sp) { for (uchar d = 0; d < 4; ++d) elems[d] |= sp.elems[d]; return *this; }
            TetGeomSparsity& unset(TetGeomSparsity sp) { for (uchar d = 0; d < 4; ++d) elems[d] &= ~(sp.elems[d]); return *this; }
            /// Same as set, but turn off parts of tetrahedron @see set
            TetGeomSparsity& unset(uchar elem_dim, int ielem, bool with_closure = false);
            TetGeomSparsity& unset(uchar elem_dim) { return elems[elem_dim] = 0, *this; }
            std::pair<std::array<uchar, 6>, uchar> getElemsIds(uchar elem_dim) const;
            struct Pos {
               uchar elem_dim = 6;
               uchar elem_num = 6; 
               Pos() = default;
               Pos(uchar elem_dim, uchar elem_num): elem_dim{elem_dim}, elem_num{elem_num} {}
               bool isValid() { return elem_dim != 6; }
               bool operator==(const Pos& a) const { return elem_dim == a.elem_dim && elem_num == a.elem_num;}
               bool operator!=(const Pos& a) const { return !(*this == a); }
            };
            Pos endPos() const { return Pos(); }
            Pos beginPos() const;
            Pos beginPos(uchar elem_dim) const;
            Pos nextPos(Pos p) const;
            Pos nextPosOnDim(Pos p) const;

            friend inline TetGeomSparsity operator&(TetGeomSparsity a, TetGeomSparsity b);
            friend inline TetGeomSparsity operator|(TetGeomSparsity a, TetGeomSparsity b);
            friend inline TetGeomSparsity operator^(TetGeomSparsity a, TetGeomSparsity b);
            friend inline TetGeomSparsity operator~(TetGeomSparsity a);
        };

        struct DofIterator;
        struct DofSparsedIterator;
        struct NestedDofMapBase;
        struct NestedDofMap;
        struct NestedDofMapView;
        struct BaseDofMap{
            enum class BaseTypes{
                Unknown = 0,
                UniteType = 1,
                VectorType = 2,
                ComplexType = 3,
                NestedType = 4,
                NestedViewType = 5,
                VectorTemplateType = 6,
                ComplexTemplateType = 7,
            };
            /// @return Child type number
            virtual uint ActualType() const { return static_cast<uint>(BaseTypes::Unknown); } 
            ///@return number of d.o.f. on specific geometrical element type
            ///@warning etype should be primitive geom type, i.e. equals to 2^i
            virtual uint NumDof(uchar etype) const = 0;
            /// @return number of d.o.f. on specific geometrical element type on a tetrahedron
            virtual uint NumDofOnTet(uchar etype) const; 
            /// @return total number of d.o.f.s on tetrahedron  
            virtual uint NumDofOnTet() const; 
            /// @return values NumDof(t) for all types t 
            virtual std::array<uint, NGEOM_TYPES> NumDofs() const;
            /// @return values NumDofOnTet(t) for all types t  
            virtual std::array<uint, NGEOM_TYPES> NumDofsOnTet() const;
            /// Convert tetrahedron contiguous index into d.o.f. index  
            virtual LocalOrder LocalOrderOnTet(TetOrder dof_id) const = 0; 
            /// Convert geometric locus into d.o.f. index 
            LocalOrder LocalOrderOnTet(LocGeomOrder dof_id) const {return LocalOrder(TetDofID(dof_id), dof_id.etype, dof_id.nelem, dof_id.leid); } 
            /// Compute geometrical type of element by its tetrahedron contiguous index
            virtual uint TypeOnTet(uint dof) const { assert(isValidIndex(TetOrder(dof)) && "Wrong index"); return LocalOrderOnTet(TetOrder(dof)).etype; }
            /// Get tetrahedron contiguous index from geometric locus
            virtual uint TetDofID(LocGeomOrder dof_id) const = 0;
            /// Is map has d.o.f.s on specific element types
            virtual bool DefinedOn(uchar etype) const { return NumDof(etype) > 0; };
            /// @return values DefinedOn(t) for all types t
            virtual uint GetGeomMask() const;
            inline bool isValidIndex(TetOrder dof_id) const { return dof_id.gid < NumDofOnTet(); }
            inline bool isValidIndex(LocGeomOrder dof_id) const { return dof_id.etype != UNDEF && GeomTypeIsValid(dof_id.etype) && dof_id.nelem < GeomTypeTetElems(dof_id.etype) && dof_id.leid < NumDof(dof_id.etype); }
            inline bool isValidIndex(LocalOrder dof_id) const { return isValidIndex(dof_id.getTetOrder()) && isValidIndex(dof_id.getGeomOrder()); }
            DofIterator begin() const;
            DofIterator end() const;
            inline LocalOrder operator[](TetOrder dof_id) const { return LocalOrderOnTet(dof_id); }
            inline LocalOrder operator[](LocGeomOrder dof_id) const { return LocalOrderOnTet(dof_id); }
            inline LocalOrder operator[](uint tet_dof_id) const { return operator[](TetOrder(tet_dof_id)); }
            LocalOrder at(TetOrder dof_id) const { if (!isValidIndex(dof_id)) throw std::out_of_range("Not valid index"); return operator[](dof_id); }
            LocalOrder at(LocGeomOrder dof_id) const { if (!isValidIndex(dof_id)) throw std::out_of_range("Not valid index"); return operator[](dof_id); }
            DofSparsedIterator beginBySparsity(const TetGeomSparsity& sp, bool preferGeomOrdering = false) const;
            DofSparsedIterator endBySparsity() const;
            /// @return number of submaps (of first level) stored in current map
            virtual uint NestedDim() const = 0;
            /// @param[in] ext_dims[ndims] is array of size ndims of nested component numbers
            /// e.g. for space Complex : (Unite, Unite, (Vector[3]: Unite) )
            /// ext_dims = 2, 1 defines Complex -> 2-Vector[] -> 1-Unite[1]=Unite
            /// @param ndims is size of ext_dims array
            /// @param[in,out] view is returned result
            /// @return a nested subspace that takes into account the shift of degrees of freedom from the enclosing space
            virtual void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const = 0;
            NestedDofMap GetNestedDofMap(const int* ext_dims, int ndims) const;
            NestedDofMapView GetNestedDofMapView(const int* ext_dims, int ndims) const;
            /// Return stored subspace. Parameters same as for GetNestedComponent
            /// @see GetNestedComponent
            virtual std::shared_ptr<const BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) const {return const_cast<BaseDofMap*>(this)->GetSubDofMap(ext_dims, ndims); }
            virtual std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) = 0;
            virtual bool operator==(const BaseDofMap&) const = 0;
            bool operator!=(const BaseDofMap& other) const { return !(*this == other); }
            virtual std::shared_ptr<BaseDofMap> Copy() const  = 0;
        
            virtual void BeginByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const;
            virtual void EndByGeomSparsity(LocalOrder& lo) const;
            virtual void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const;
            friend class DofSparsedIterator;
            friend class DofIterator;
        };

        struct DofSparsedIterator{
        protected: 
            TetGeomSparsity sparsity;
            LocalOrder ord;
            const BaseDofMap* map = nullptr;
            bool preferGeomOrdering = false;
        public: 
            typedef std::input_iterator_tag  iterator_category;
            typedef LocalOrder value_type;
            typedef void difference_type;
            typedef const LocalOrder*  pointer;
            typedef LocalOrder reference;

            DofSparsedIterator& operator ++() { map->IncrementByGeomSparsity(sparsity, ord, preferGeomOrdering); return *this; }
            DofSparsedIterator  operator ++(int) {DofSparsedIterator ret(*this); operator++(); return ret;}
            bool operator ==(const DofSparsedIterator & b) const { return ord == b.ord; }
            bool operator !=(const  DofSparsedIterator & other) const { return !operator==(other); }
            reference operator*() const { return ord; }
            pointer operator ->() const { return &ord; }
            const BaseDofMap* MapLink() const { return map; }
            TetGeomSparsity Sparsity() const { return sparsity; }
            friend class BaseDofMap; 
        };

        struct DofIterator{
        protected: 
            mutable LocalOrder ord;
            const BaseDofMap* map = nullptr;
            DofIterator() = default;
            DofIterator(const BaseDofMap* map, LocalOrder ord): ord{ord}, map{map} {} 
        public: 
            typedef std::random_access_iterator_tag  iterator_category;
            typedef LocalOrder value_type;
            typedef int difference_type;
            typedef const LocalOrder* pointer;
            typedef LocalOrder reference;

            DofIterator& operator ++() { ++ord.gid; BrokeGeomOrd(); return *this; }
            DofIterator  operator ++(int) {DofIterator ret(*this); operator++(); return ret;}
            DofIterator& operator --() { --ord.gid; BrokeGeomOrd(); return *this; }
            DofIterator  operator --(int) {DofIterator ret(*this); operator--(); return ret;}
            bool operator ==(const DofIterator & b) const { return ord.gid == b.ord.gid; }
            bool operator !=(const  DofIterator & other) const { return !operator==(other); }
            reference operator*() const { if (!isOrdValid()) RepairGeomOrd(); return ord; }
            pointer operator ->() const { if (!isOrdValid()) RepairGeomOrd(); return &ord; }
            reference operator[](difference_type n) const { return *(*this + n); }
            difference_type operator-(const  DofIterator & other) const { return ord.gid - other.ord.gid; }
            DofIterator& operator+=(difference_type n) { ord.gid += n; BrokeGeomOrd(); return *this; } 
            DofIterator& operator-=(difference_type n) { ord.gid -= n; BrokeGeomOrd(); return *this; }
            DofIterator  operator+ (difference_type n) const { DofIterator other = *this; other += n; return other; }
            DofIterator  operator- (difference_type n) const { DofIterator other = *this; other -= n; return other; }
            friend DofIterator  operator+(difference_type n, const DofIterator& a) { return a+n; }
            bool operator< (const  DofIterator & other) const { return ord.gid <  other.ord.gid; }
            bool operator> (const  DofIterator & other) const { return ord.gid >  other.ord.gid; }
            bool operator<=(const  DofIterator & other) const { return ord.gid <= other.ord.gid; }
            bool operator>=(const  DofIterator & other) const { return ord.gid >= other.ord.gid; }
            const BaseDofMap* MapLink() const { return map; }
            friend class BaseDofMap; 
        private:
            void BrokeGeomOrd(){ ord.etype = UNDEF; }
            bool isOrdValid() const { return ord.etype != UNDEF; }
            void RepairGeomOrd() const { assert(map->isValidIndex(ord.getTetOrder()) && "Tet index is not valid"); ord = map->LocalOrderOnTet(ord.getTetOrder()); }
        };

        struct NestedDofMapBase: public BaseDofMap{
            std::array<uint, NGEOM_TYPES> m_shiftNumDof = {0};
            uint m_shiftOnTet = 0;

            virtual const BaseDofMap* base() const = 0;
            virtual void set_base(const std::shared_ptr<BaseDofMap>& new_base) = 0;
            virtual void set_base(const BaseDofMap* new_base) = 0;

            uint NumDof(uchar etype) const override { return base()->NumDof(etype); }
            uint NumDofOnTet(uchar etype) const override { return base()->NumDofOnTet(etype); }
            uint NumDofOnTet() const override { return base()->NumDofOnTet(); }
            std::array<uint, NGEOM_TYPES> NumDofs() const override { return base()->NumDofs(); }
            std::array<uint, NGEOM_TYPES> NumDofsOnTet() const override { return base()->NumDofsOnTet(); }
            /// Convert tetrahedron contiguous subvector index into d.o.f. index in vector
            LocalOrder LocalOrderOnTet(TetOrder dof_id) const override; 
            /// Convert geometric locus of subvector index into d.o.f. index in vector
            LocalOrder LocalOrderOnTet(LocGeomOrder dof_id) const;
            /// Is vector index point to position in subvector?
            inline bool isValidOutputIndex(TetOrder dof_id) const { return base()->isValidIndex(TetOrder(dof_id.gid - m_shiftOnTet)); }
            /// Is vector index point to position in subvector?
            inline bool isValidOutputIndex(LocGeomOrder dof_id) const { return dof_id.etype != UNDEF && GeomTypeIsValid(dof_id.etype) && base()->isValidIndex(LocGeomOrder(dof_id.etype, dof_id.nelem, dof_id.leid - m_shiftNumDof[GeomTypeToNum(dof_id.etype)]));  }
            /// Is vector index point to position in subvector?
            bool isValidOutputIndex(LocalOrder dof_id) const;
            /// Type of subvector index
            uint TypeOnTet(uint dof) const override { return base()->TypeOnTet(dof); }
            /// @return contigous vector index from geometric locus of subvector d.o.f.
            uint TetDofID(LocGeomOrder dof_id) const override { return base()->TetDofID(dof_id) + m_shiftOnTet; }
            /// Is subvector defined on the element of specified type
            bool DefinedOn(uchar etype) const override { return base()->DefinedOn(etype); }
            uint GetGeomMask() const override { return base()->GetGeomMask(); }
            virtual void Clear(){ std::fill(m_shiftNumDof.begin(), m_shiftNumDof.end(), 0); m_shiftOnTet = 0; }
            uint NestedDim() const override { return 1; }
            bool operator==(const BaseDofMap& other) const;

            void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const;
        };

        /// A special representation to work with a subvector of d.o.f.s distributed in enclosing vector of a given subspace 
        struct NestedDofMap: public NestedDofMapBase{
            std::shared_ptr<BaseDofMap> m_base;
            
            NestedDofMap() = default;
            explicit NestedDofMap(std::shared_ptr<BaseDofMap> base, const std::array<uint, NGEOM_TYPES>& shifts_on_elem = std::array<uint, NGEOM_TYPES>{0}, uint full_shift = 0): m_base{base} { m_shiftNumDof = shifts_on_elem, m_shiftOnTet = full_shift; }
            BaseDofMap* base() { return m_base.get(); }
            const BaseDofMap* base() const override { return m_base.get(); }
            void set_base(const std::shared_ptr<BaseDofMap>& new_base) override { m_base = new_base; }
            void set_base(const BaseDofMap* new_base) override { m_base = new_base->Copy(); }
            uint ActualType() const { return static_cast<uint>(BaseTypes::NestedType); }
            void Clear(){ m_base = nullptr; NestedDofMapBase::Clear(); }
            void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const;
            std::shared_ptr<const BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) const;
            std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims);

            std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<NestedDofMap>(*this); }
        };

        struct NestedDofMapView: public NestedDofMapBase{
            const BaseDofMap* m_base;
            
            NestedDofMapView() = default;
            explicit NestedDofMapView(const BaseDofMap* base, const std::array<uint, NGEOM_TYPES>& shifts_on_elem = std::array<uint, NGEOM_TYPES>{0}, uint full_shift = 0): m_base{base} { m_shiftNumDof = shifts_on_elem, m_shiftOnTet = full_shift; }
            const BaseDofMap* base() const override { return m_base; }
            void set_base(const std::shared_ptr<BaseDofMap>& new_base) override { m_base = new_base.get(); }
            void set_base(const BaseDofMap* new_base) override { m_base = new_base; }
            uint ActualType() const { return static_cast<uint>(BaseTypes::NestedViewType); }
            void Clear(){ m_base = nullptr; NestedDofMapBase::Clear(); }
            void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const;
            std::shared_ptr<const BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) const;
            std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims);

            std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<NestedDofMapView>(*this); }
        };

        /// @brief To store BaseDofMap objects
        struct DofMap{
            std::shared_ptr<BaseDofMap> m_invoker;

            template<typename DofMapT>
            explicit DofMap(const DofMapT& f, typename std::enable_if<std::is_base_of<BaseDofMap, DofMapT>::value>::type* = 0): m_invoker{new DofMapT(f)} {}
            template<typename DofMapT>
            explicit DofMap(DofMapT&& f, typename std::enable_if<std::is_base_of<BaseDofMap, DofMapT>::value>::type* = 0): m_invoker{new DofMapT(std::move(f))} {}
            DofMap(const DofMap &) = default;
            DofMap(DofMap &&) = default;
            DofMap() = default;
            explicit DofMap(std::shared_ptr<BaseDofMap> dof_map): m_invoker(std::move(dof_map)) {}
            DofMap& operator=(const DofMap &f){ return m_invoker = f.m_invoker, *this; }
            DofMap& operator=(DofMap &&f){ return m_invoker = std::move(f.m_invoker), *this; }
            
            template<typename DofMapT = BaseDofMap>
            DofMapT* target() { return static_cast<DofMapT *>(m_invoker.get()); }
            template<typename DofMapT = BaseDofMap>
            const DofMapT* target() const { return static_cast<const DofMapT *>(m_invoker.get()); }
            std::shared_ptr<BaseDofMap> base() const { return m_invoker; }

            /// @return Get type id
            inline uint ActualType() const { return m_invoker->ActualType(); }
            /// @return number of d.o.f. on specific geometrical element type
            /// @warning etype shold have primitive type
            inline uint NumDof(uchar etype) const { return m_invoker->NumDof(etype); }
            /// @return number of d.o.f. on specific geometrical element type on a tetrahedron
            inline uint NumDofOnTet(uchar etype) const { return m_invoker->NumDofOnTet(etype); } 
            /// @return total number of d.o.f.s on tetrahedron 
            inline uint NumDofOnTet() const { return m_invoker->NumDofOnTet(); }
            /// @return number of d.o.f. on every geometrical element type
            inline std::array<uint, NGEOM_TYPES> NumDofs() const { return m_invoker->NumDofs(); }
            /// @return total number of d.o.f. on every geometrical element types of tetrahedron
            inline std::array<uint, NGEOM_TYPES> NumDofsOnTet() const { return m_invoker->NumDofsOnTet(); }
            /// @return d.o.f. index by it's tetrahedron contiguous index  
            inline LocalOrder LocalOrderOnTet(TetOrder dof_id) const { return m_invoker->LocalOrderOnTet(dof_id); }
            /// @return d.o.f. index by it's geometrical locus 
            inline LocalOrder LocalOrderOnTet(LocGeomOrder dof_id) const { return m_invoker->LocalOrderOnTet(dof_id); } 
            /// @return geometrical type of element by its tetrahedron contiguous index
            inline uint TypeOnTet(uint dof) const { return m_invoker->TypeOnTet(dof); }
            /// @return tetrahedron contiguous index by it's geometric locus
            inline uint TetDofID(LocGeomOrder dof_id) const { return m_invoker->TetDofID(dof_id); }
            /// Is map has d.o.f.s on specific element types
            inline bool DefinedOn(uchar etype) const { return m_invoker->DefinedOn(etype); } 
            /// @return geometric definedness mask, if mask & etype > 0 then map has d.o.f.s on all geom elemets with types from etype
            inline uint GetGeomMask() const { return m_invoker->GetGeomMask(); }
            inline bool isValidIndex(TetOrder dof_id) const { return m_invoker->isValidIndex(dof_id); }
            inline bool isValidIndex(LocGeomOrder dof_id) const { return m_invoker->isValidIndex(dof_id); }
            inline bool isValidIndex(LocalOrder dof_id) const { return m_invoker->isValidIndex(dof_id); }
            inline LocalOrder operator[](TetOrder dof_id) const { return m_invoker->operator[](dof_id); }
            inline LocalOrder operator[](LocGeomOrder dof_id) const { return m_invoker->operator[](dof_id); }
            inline LocalOrder operator[](uint tet_dof_id) const { return m_invoker->operator[](tet_dof_id); }
            inline LocalOrder at(TetOrder dof_id) const { return m_invoker->at(dof_id); }
            inline LocalOrder at(LocGeomOrder dof_id) const { return m_invoker->at(dof_id); }
            inline DofIterator begin() const { return m_invoker->begin(); }
            inline DofIterator end() const { return m_invoker->end(); }
            /// @brief Get first sparces iterator 
            /// @param sp is geometrical sparsity on that will be taken elements of map
            /// @param preferGeomOrdering : is it desirable to go strictly sequentially through geometric elements?
            /// @return first iterator
            inline DofSparsedIterator beginBySparsity(const TetGeomSparsity& sp, bool preferGeomOrdering = false) const { return m_invoker->beginBySparsity(sp, preferGeomOrdering); }
            inline DofSparsedIterator endBySparsity() const { return m_invoker->endBySparsity(); }
            /// @return number of submaps (of first level) stored in current map
            inline uint NestedDim() const { return m_invoker->NestedDim(); }
            /// @brief Get nested component
            /// @param ext_dims[ndims] is array of size ndims of nested component numbers
            /// @param ndims is size of ext_dims array
            /// @return a nested submap that takes into account the shift of degrees of freedom from the enclosing map.
            /// e.g. for space Complex : (Unite, Unite, (Vector[3]: Unite) )
            /// ext_dims = 2, 1 defines Complex -> 2-Vector[] -> 1-Unite[1]=Unite
            inline NestedDofMap GetNestedDofMap(const int* ext_dims = nullptr, int ndims = 0) const { return (ndims > 0) ? m_invoker->GetNestedDofMap(ext_dims, ndims) : (ndims == 0 ? NestedDofMap(m_invoker) : NestedDofMap()); }
            inline NestedDofMap GetNestedDofMap(const std::initializer_list<int>& ext_dims) const { return GetNestedDofMap(ext_dims.begin(), ext_dims.size()); }
            /// Return stored subspace. Parameters same as for GetNestedComponent
            /// @see GetNestedComponent
            inline const DofMap GetSubDofMap(const int* ext_dims, int ndims) const { return (ndims > 0) ? DofMap(m_invoker->GetSubDofMap(ext_dims, ndims)) : (ndims == 0 ? *this : DofMap()); }
            inline DofMap GetSubDofMap(const int* ext_dims, int ndims) { return (ndims > 0) ? DofMap(m_invoker->GetSubDofMap(ext_dims, ndims)) : (ndims == 0 ? *this : DofMap()); }
            inline DofMap GetSubDofMap(const std::initializer_list<int>& ext_dims) { return GetSubDofMap(ext_dims.begin(), ext_dims.size()); }
            inline const DofMap GetSubDofMap(const std::initializer_list<int>& ext_dims) const { return GetSubDofMap(ext_dims.begin(), ext_dims.size()); }
            
            inline bool operator==(const DofMap& other) const { return m_invoker.get() == other.m_invoker.get() || (m_invoker.get() && other.m_invoker.get() && *m_invoker.get() == *other.m_invoker.get()); }
            inline bool operator!=(const DofMap& other) const { return !(*this == other); }
            /// @brief Create ComplexDofMap(a, b) and simplify result
            DofMap operator*(const DofMap& other) const;
            friend DofMap operator^(const DofMap& d, uint k);
            friend DofMap pow(const DofMap& d, uint k);
        };
        /// @brief Make VectorDofMap(k, d) and simplify result
        DofMap operator^(const DofMap& d, uint k);
        /// @brief Create VectorDofMap(k, d)
        DofMap pow(const DofMap& d, uint k);
        /// @brief Create ComplexDofMap(maps)
        DofMap merge(const std::vector<DofMap>& maps);
        /// @brief Create ComplexDofMap(maps) and simplify result
        DofMap merge_with_simplifications(const std::vector<DofMap>& maps);

        /// Store the simpliest maps
        struct UniteDofMap: public BaseDofMap{
            std::array<uint, NGEOM_TYPES+1> m_shiftDof = {0};
            std::array<uint, NGEOM_TYPES+1> m_shiftTetDof = {0};

            uint ActualType() const override { return static_cast<uint>(BaseTypes::UniteType); }
            UniteDofMap() = default;
            UniteDofMap(std::array<uint, NGEOM_TYPES> NumDofs);
            uint NumDof(uchar etype) const override { auto t = GeomTypeToNum(etype); return m_shiftDof[t+1] - m_shiftDof[t]; }
            uint NumDofOnTet(uchar etype) const override;
            uint NumDofOnTet() const override { return m_shiftTetDof[NGEOM_TYPES] - m_shiftTetDof[0]; }
            std::array<uint, NGEOM_TYPES> NumDofs() const override; 
            std::array<uint, NGEOM_TYPES> NumDofsOnTet() const override;
            LocalOrder LocalOrderOnTet(TetOrder dof) const override;
            uint TypeOnTet(uint dof) const override;
            uint TetDofID(LocGeomOrder dof) const override;
            uint GetGeomMask() const override;
            uint NestedDim() const override { return 0; }
            void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const override{ (void) ext_dims; (void) ndims; view.Clear(); }
            std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) override { (void) ext_dims; (void) ndims; return nullptr; }
            void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const override;
            bool operator==(const BaseDofMap& other) const override;
            std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<UniteDofMap>(*this); }
        };

        /// Store vector maps
        struct VectorDofMap: public BaseDofMap {
            std::shared_ptr<BaseDofMap> base;
            int m_dim = 0;
            VectorDofMap() = default;
            VectorDofMap(int dim, std::shared_ptr<BaseDofMap> space): base{std::move(space)}, m_dim(dim) {}
            
            uint ActualType() const override { return static_cast<uint>(BaseTypes::VectorType); }
            uint NumDof(uchar etype) const override { return m_dim ? m_dim * base->NumDof(etype) : 0; }
            uint NumDofOnTet(uchar etype) const override { return m_dim ? m_dim * base->NumDofOnTet(etype) : 0; }
            uint NumDofOnTet() const override { return m_dim ? m_dim * base->NumDofOnTet() : 0; }
            std::array<uint, NGEOM_TYPES> NumDofs() const override;
            std::array<uint, NGEOM_TYPES> NumDofsOnTet() const override;
            LocalOrder LocalOrderOnTet(TetOrder dof_id) const override;
            uint TypeOnTet(uint dof) const override { return m_dim ? base->TypeOnTet(dof % base->NumDofOnTet()) : UNDEF; }
            uint TetDofID(LocGeomOrder dof_id) const override;
            uint TetDofID(uint part_id, TetOrder lgid) const { return part_id*base->NumDofOnTet() + lgid; }
            bool DefinedOn(uchar etype) const override { return m_dim ? base->DefinedOn(etype) : false; }
            uint GetGeomMask() const { return m_dim ? base->GetGeomMask() : UNDEF; }
            uint NestedDim() const override { return m_dim; }
            void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const;
            std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims);
            bool operator==(const BaseDofMap& other) const override;
            std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<VectorDofMap>(*this); }
            
            void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const;
            ComponentTetOrder ComponentID(TetOrder dof_id) const { return {dof_id, dof_id % base->NumDofOnTet(), dof_id / base->NumDofOnTet()}; }
        };

        namespace FemComDetails{
            template<bool isDofMapTDerivedFromBaseDofMap, typename DofMapT>
            struct VectorDofMapCImpl;

            template<class T, typename... DofMapT>
            struct CheckDofMap: public std::integral_constant<bool, std::is_base_of<BaseDofMap, T>::value && CheckDofMap<DofMapT...>::value> {};
            template<class T>
            struct CheckDofMap<T>: public std::integral_constant<bool, std::is_base_of<BaseDofMap, T>::value> {};

            template<bool isDofMapTDerivedFromBaseDofMap, typename... DofMapT>
            struct ComplexDofMapCImpl;
        };
        template<typename DofMapT>
        using VectorDofMapC = FemComDetails::VectorDofMapCImpl<std::is_base_of<BaseDofMap, DofMapT>::value, DofMapT>;

        template<typename... DofMapT>
        using ComplexDofMapC = FemComDetails::ComplexDofMapCImpl<FemComDetails::CheckDofMap<DofMapT...>::value, DofMapT...>;

        /// Store complex maps
        struct ComplexDofMap: public BaseDofMap{
            std::vector<std::shared_ptr<BaseDofMap>> m_spaces;
            std::array<std::vector<uint>, NGEOM_TYPES> m_spaceNumDofTet;
            std::array<std::vector<uint>, NGEOM_TYPES> m_spaceNumDof;
            std::vector<uint> m_spaceNumDofsTet;

            static ComplexDofMap makeCompressed(const std::vector<std::shared_ptr<BaseDofMap>>& spaces);
            ComplexDofMap() = default;
            explicit ComplexDofMap(std::vector<std::shared_ptr<BaseDofMap>> spaces);
            uint NestedDim() const override { return m_spaces.size(); }
            uint ActualType() const override { return static_cast<uint>(BaseTypes::ComplexType); }
            uint NumDof(uchar etype) const override { auto t = GeomTypeToNum(etype); return m_spaceNumDof[t][m_spaces.size()] - m_spaceNumDof[t][0]; }
            uint NumDofOnTet(uchar etype) const override { auto t = GeomTypeToNum(etype); return m_spaceNumDofTet[t][m_spaces.size()] - m_spaceNumDofTet[t][0]; }
            std::array<uint, NGEOM_TYPES> NumDofs() const override;
            std::array<uint, NGEOM_TYPES> NumDofsOnTet() const override;
            LocalOrder LocalOrderOnTet(TetOrder dof_id) const override;
            uint TypeOnTet(uint dof) const override ;
            uint TetDofID(LocGeomOrder dof_id) const override;
            uint TetDofID(uint part_id, TetOrder lgid) const { return m_spaceNumDofsTet[part_id] + lgid; }
            uint GetGeomMask() const override ;
            void GetNestedComponent(const int* ext_dims, int ndims, NestedDofMapBase& view) const override;
            std::shared_ptr<BaseDofMap> GetSubDofMap(const int* ext_dims, int ndims) override;
            bool operator==(const BaseDofMap& other) const override;
            std::shared_ptr<BaseDofMap> Copy() const override { return std::make_shared<ComplexDofMap>(*this); }
            
            void BeginByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const override;
            void IncrementByGeomSparsity(const TetGeomSparsity& sp, LocalOrder& lo, bool preferGeomOrdering = false) const override;
            ComponentTetOrder ComponentID(TetOrder dof_id) const;
        };

    };
}

#include "operations/dc_on_dof.h"
namespace Ani{
    /// Applies the Dirichlet condition to the matrix 
    /// under the assumption that the trial and test spaces have the same distribution of d.o.f.s over the mesh geometrical elements
    /// @param trial_map is dof map of trial FEM space
    /// @param A is local FEM matrix
    /// @param sp is selection of dirichlet parts of the tetrahedron
    template<typename Scalar>
    inline void applyDirMatrix(const DofT::BaseDofMap& trial_map, DenseMatrix <Scalar>& A, const DofT::TetGeomSparsity& sp);
    /// Applies the Dirichlet condition to the matrix 
    /// under the assumption that the trial and test spaces have the same nfa dimensions
    /// @param trial_map is dof map of trial FEM space
    /// @param test_map is dof map of test FEM space
    /// @param A is local FEM matrix
    /// @param sp is selection of dirichlet parts of the tetrahedron
    template<typename Scalar>
    inline void applyDirMatrix(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix <Scalar>& A, const DofT::TetGeomSparsity& sp);
    /// @brief Set zeros to position of residual corresponding to dirichlet condition
    /// @param test_map is dof map of test FEM space
    /// @param F is local residual
    /// @param sp is selection of dirichlet parts of the tetrahedron
    template<typename Scalar>
    inline void applyDirResidual(const DofT::BaseDofMap& test_map, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp);
    /// @brief Applies the Dirichlet condition to the matrix and set zeros to position of residual corresponding to Dirichlet condition
    /// @see applyDirMatrix, applyDirResidual
    template<typename Scalar>
    inline void applyDirResidual(const DofT::BaseDofMap& trial_map, DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp);
    /// @brief Applies the Dirichlet condition to the matrix and set zeros to position of residual corresponding to Dirichlet condition
    /// @see applyDirMatrix, applyDirResidual
    template<typename Scalar>
    inline void applyDirResidual(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix <Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp);
    /// @brief Applies the constant Dirichlet condition var = bc * I, where dim(u) = dim(I) and I is vector of units (1, 1, ..., 1)
    /// @see applyDirMatrix, applyDirResidual
    template<typename Scalar>
    inline void applyConstantDirByDofs(const DofT::BaseDofMap& trial_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, Scalar bc);
    /// @brief Applies the constant Dirichlet condition var[i-th dof] = dofs[i] where geometrical locus(i) belongs to sp 
    /// @param dofs is array of d.o.f.s that should be considered as value of dirichlet d.o.f.s in indexes defined by sp
    template<typename Scalar>
    inline void applyDirByDofs(const DofT::BaseDofMap& trial_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, const ArrayView<const Scalar>& dofs);
    template<typename Scalar>
    inline void applyConstantDirByDofs(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, Scalar bc);
    template<typename Scalar>
    inline void applyDirByDofs(const DofT::BaseDofMap& trial_map, const DofT::BaseDofMap& test_map, DenseMatrix<Scalar>& A, DenseMatrix <Scalar>& F, const DofT::TetGeomSparsity& sp, const ArrayView<const Scalar>& dofs);
};

#include "tetdofmap.inl"

#endif //CARNUM_ANI_TETDOFMAP_H