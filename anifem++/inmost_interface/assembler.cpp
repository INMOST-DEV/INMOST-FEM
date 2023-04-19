//
// Created by Liogky Alexey on 29.03.2022.
//

#include "assembler.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

using namespace INMOST;

void ProblemGlobEnumeration::Clear() {
    std::fill(NumDof, NumDof + 4, 0);
    std::fill(NumElem, NumElem + 4, 0);
    std::fill(MinElem, MinElem + 4, 0);
    std::fill(MaxElem, MaxElem + 4, 0);
    MatrSize = -1;
    BegInd = LONG_MAX;
    EndInd = -1;
    unite_var.reset();
    mesh = nullptr;
}

void ProblemGlobEnumeration::IncrementSliceIter(ProblemGlobEnumeration::SliceIteratorData &it) const {
    auto t = it.it->GetElementType();
    auto ti = static_cast<FemExprDescr::GeomType>(INMOST::ElementNum(t));
    auto is_same_loc_elem = it.status & SliceIteratorData::SAME_LOC_ELEM_IDS;
    auto is_same_var = it.status & SliceIteratorData::SAME_VAR;
    if (it.slice_elem_ids && is_same_loc_elem) {
        if (is_same_var){
            if (static_cast<unsigned long>(it.slice_id+1) < it.slice_elem_ids->size()){
                it.slice_id++;
                it.var_elem_dof_id = it.slice_elem_ids->operator[](ti)[it.slice_id];
            } else {
                ++it.it;
                while(it.it->isValid() && it.it->GetStatus() == INMOST::Element::Ghost) ++it.it;
                it.slice_id = 0;
                if (it.it->isValid()) {
                    t = it.it->GetElementType();
                    auto tti = static_cast<FemExprDescr::GeomType>(INMOST::ElementNum(t));
                    if (tti != ti)
                        it.var_elem_dof_id = it.slice_elem_ids->operator[](tti)[it.slice_id];
                }
            }
        } else {
            int elem_dof = 0;
            if (static_cast<unsigned long>(it.slice_id+1) < it.slice_elem_ids->size()){
                it.slice_id++;
                elem_dof = it.slice_elem_ids->operator[](ti)[it.slice_id];
                if (unite_var->m_spaceNumDof[ti][it.var_id+1] <= elem_dof) {
                    it.var_id = std::upper_bound(unite_var->m_spaceNumDof[ti].data() + it.var_id+1, unite_var->m_spaceNumDof[ti].data() + unite_var->m_spaceNumDof[ti].size(), elem_dof)  - unite_var->m_spaceNumDof[ti].data() - 1;
                }
                it.var_elem_dof_id = elem_dof - unite_var->m_spaceNumDof[ti][it.var_id];
            } else {
                ++it.it;
                while(it.it->isValid() && it.it->GetStatus() == INMOST::Element::Ghost) ++it.it;
                it.slice_id = 0;
                elem_dof = it.slice_elem_ids->operator[](ti)[it.slice_id];
                it.var_id = std::upper_bound(unite_var->m_spaceNumDof[ti].data(), unite_var->m_spaceNumDof[ti].data() + it.var_id + 1, elem_dof)  - unite_var->m_spaceNumDof[ti].data() - 1;
                if (it.it->isValid()) {
                    t = it.it->GetElementType();
                    auto tti = static_cast<FemExprDescr::GeomType>(INMOST::ElementNum(t));
                    if (tti != ti)
                        it.var_elem_dof_id = elem_dof - unite_var->m_spaceNumDof[tti][it.var_id];
                }
            }
        }
    } else {
        if (!is_same_loc_elem && it.var_elem_dof_id + 1 < unite_var->m_spaces[it.var_id]->NumDof(ti)){
            it.var_elem_dof_id++;
            return;
        }
        if (!is_same_var && static_cast<unsigned long>(it.var_id + 1) < unite_var->m_spaces.size()) {
            bool find_var = false;
            while (static_cast<unsigned long>(it.var_id + 1) < unite_var->m_spaces.size() && !find_var){
                it.var_id++;
                auto num_loc_elems = unite_var->m_spaceNumDof[ti][it.var_id+1] - unite_var->m_spaceNumDof[ti][it.var_id];
                if (num_loc_elems != 0){
                    find_var = (!is_same_loc_elem) ? true : (num_loc_elems > it.var_elem_dof_id);
                }
            }
            if (find_var){
                it.var_elem_shift =  unite_var->m_spaceNumDof[ti][it.var_id];
                if (!is_same_loc_elem) it.var_elem_dof_id = 0;
                return;
            }
        }
        {
            while (it.it->isValid()){
                ++it.it;
                while(it.it->isValid() && it.it->GetStatus() == INMOST::Element::Ghost) ++it.it;
                if (!it.it->isValid()){
                    if (!is_same_var) it.var_id = 0;
                    if (!is_same_loc_elem) it.var_elem_dof_id = 0;
                    return;
                }
                auto tti = static_cast<FemExprDescr::GeomType>(INMOST::ElementNum(it.it->GetElementType()));
                bool find_var = false;
                if (is_same_var){
                    find_var = (unite_var->m_spaces[it.var_id]->NumDof(tti) > 0);
                } else {
                    for (unsigned i = 0; i < unite_var->m_spaces.size() && !find_var; ++i)
                        if (unite_var->m_spaces[i]->NumDof(tti) > 0){
                            it.var_id = i;
                            find_var = true;
                        }
                }
                if (find_var){
                    if (!is_same_loc_elem) it.var_elem_dof_id = 0;
                    it.var_elem_shift =  unite_var->m_spaceNumDof[tti][it.var_id];
                    return;
                }
            }
        }
    }
}

ProblemGlobEnumeration::SliceIterator
    ProblemGlobEnumeration::beginByGeom(INMOST::ElementType mask, int var_id, int var_elem_dof_id,
                                    ProblemGlobEnumeration::SliceIteratorData::Status status) const {
    mask &= GetGeomMask();
    auto it = mesh->BeginElement(mask);
    while (it->isValid() && it->GetStatus() == INMOST::Element::Ghost) ++it;
    SliceIterator si{this, it, var_id, var_elem_dof_id, status};
    if (si.data.it != mesh->EndElement()) {
        auto t = si.data.it->GetElementType();
        auto ti = static_cast<FemExprDescr::GeomType>(INMOST::ElementNum(t));
        si.data.var_elem_shift = unite_var->m_spaceNumDof[ti][si.data.var_id];
    }
    return si;
}

ProblemGlobEnumeration::SliceIterator
    ProblemGlobEnumeration::beginByGeom(const std::array<std::vector<int>, 4> &slice_elem_dof, INMOST::ElementType mask,
                                    int var_id, int var_elem_dof_id,
                                    ProblemGlobEnumeration::SliceIteratorData::Status status) const {
    mask &= GetGeomMask();
    auto it = mesh->BeginElement(mask);
    while (it->isValid() && it->GetStatus() == INMOST::Element::Ghost) ++it;
    SliceIterator si{slice_elem_dof, this, it, var_id, var_elem_dof_id, status};
    if (si.data.it != mesh->EndElement()) {
        auto t = si.data.it->GetElementType();
        auto ti = static_cast<FemExprDescr::GeomType>(INMOST::ElementNum(t));
        si.data.var_elem_shift = unite_var->m_spaceNumDof[ti][si.data.var_id];
    }
    return si;
}

INMOST::ElementType ProblemGlobEnumeration::GetGeomMask() const {
    INMOST::ElementType mask = 0;
    for (int i = 0; i < 4; ++i)
        if (NumDof[i] > 0)
            mask |= (1 << i);
    return mask;
}


std::shared_ptr<ProblemGlobEnumeration> DefaultEnumeratorFactory::build(std::string prefix, bool is_global_id_updated) {
    using namespace INMOST;
    ElementType mask = 0;
    std::array<ElementType, 6> itypes = {NODE, EDGE, FACE, CELL, EDGE, FACE};
    auto _NumDofs = info->NumDofs();
    auto iend = std::min(_NumDofs.size(), itypes.size());
    for (int i = 0; i < static_cast<int>(iend); ++i)
        if (_NumDofs[i] > 0) mask |= itypes[i];
    if (mask == 0) throw std::runtime_error("Try to build enumenator for empty problem");
    bool no_nead_update_glodal_id = !is_global_id_updated;
    for (int i = 0; i < NUM_ELEMENT_TYPS && no_nead_update_glodal_id; ++i)
        if (mask & (1 << i)) no_nead_update_glodal_id &= m->HaveGlobalID(1 << i);
    if (!no_nead_update_glodal_id) m->AssignGlobalID(mask);

    long NumDof[4] = {0, 0, 0, 0},
            NumElem[4] = {0, 0, 0, 0},
            MinElem[4] = {LONG_MAX, LONG_MAX, LONG_MAX, LONG_MAX},
            MaxElem[4] = {-1, -1, -1, -1},
            InitIndex[4] = {0, 0, 0, 0};
    long BegInd = 0,EndInd = 0,MatrSize = 0;
    static const int FastInnerIndex[5] = {0,1,2,-1,3};

    for (int i = 0; i < static_cast<int>(iend); ++i)
        NumDof[FastInnerIndex[itypes[i]/2]] += _NumDofs[i];


    for(Mesh::iteratorElement it = m->BeginElement(mask); it != m->EndElement();it++){
        int j=FastInnerIndex[it->GetElementType()/2];

        int id = it->GlobalID();

        if(it->GetStatus() != Element::Ghost){
            NumElem[j]++;
            if(id<MinElem[j])
                MinElem[j] = id;
            if(id>MaxElem[j])
                MaxElem[j] = id;
        }
    }

    long total_geom_elements[4] = {0, 0, 0, 0};
    for (int i = 0; i < 4; ++i)
        if ((1<<i) & mask) total_geom_elements[i] = m->TotalNumberOf(1<<i);

    for(int i=0;i<4;i++)
        if(NumDof[i] > 0) {
            if (MinElem[i] == LONG_MAX)
                MinElem[i] = total_geom_elements[i];
        }

    for(int i=0;i<4;i++)
        BegInd += NumDof[i] * MinElem[i];
    EndInd = BegInd;
    for(int i=0;i<4;i++)
        EndInd += NumDof[i] * NumElem[i];

    Tag maps[4];
    bool is_new_maps = false;
    for (int i = 0; i < 4; ++i) {
        static const std::array<std::string, 4> postfix{"_n", "_e", "_f", "_c"};
        if (NumDof[i] <= 0) continue;
        std::string name = "GidToLidMap" + postfix[i];
        if (!m->HaveTag(name)){
            maps[i] = m->CreateTag(name, DATA_INTEGER, MESH, NONE, ENUMUNDEF);
            auto arr = m->IntegerArray(m->GetHandle(), maps[i]);
            arr.resize(MaxElem[i] - MinElem[i] + 1);
            is_new_maps = true;
        }
        else
            maps[i] = m->GetTag(name);
    }
    if (is_global_id_updated || is_new_maps){
        auto mHdl = m->GetHandle();
        for(Mesh::iteratorElement it = m->BeginElement(mask); it != m->EndElement();it++){
            int j=FastInnerIndex[it->GetElementType()/2];
            if(it->GetStatus() != Element::Ghost)
                m->IntegerArray(mHdl, maps[j])[it->GlobalID() - MinElem[j]] = it->LocalID();
        }
    }

    MatrSize = 0;
    for (int i = 0; i < 4; ++i) MatrSize += total_geom_elements[i] * NumDof[i];

    switch (oType){
        case STRAIGHT:
            InitIndex[0] = BegInd;
            for(int i=1;i<4;i++)
                InitIndex[i] = InitIndex[i-1] + NumDof[i-1]*NumElem[i-1];
            break;
        case REVERSE:
            InitIndex[3] = BegInd;
            for(int i=2;i>=0;i--)
                InitIndex[i] = InitIndex[i+1] + NumDof[i+1]*NumElem[i+1];
            break;
        default:
            std::cout<<"Wrong type order"<<std::endl;
            abort();
    }
    struct TagWrap{
        Tag tag;
        void Clear() {
            if (tag.isValid())
                tag = tag.GetMeshLink()->DeleteTag(tag);
        }
        TagWrap() = default;
        TagWrap(TagWrap&& t){
            tag = t.tag;
            t.tag = INMOST::Tag();
        }
        TagWrap& operator=(TagWrap&& t) noexcept{
            if (this == &t) return *this;
            tag = t.tag;
            t.tag = INMOST::Tag();
            return *this;
        }
        ~TagWrap() { Clear(); }
    };
    std::array<TagWrap, 4> tags;
    std::array<std::string, 4> postfix{"_n_", "_e_", "_f_", "_c_"};
    for (int i = 0; i < 4; ++i){
        if (mask & (1 << i)) {
            if (!m->HaveTag(prefix + postfix[i])){
                tags[i].tag = m->CreateTag(prefix + postfix[i], DATA_INTEGER, 1<<i, NONE, NumDof[i]);
            } else
                throw std::runtime_error("Tag with name \"" + prefix + postfix[i] + "\" already exists");
            switch (aType) {
                case ANITYPE:{
                    for (auto it = m->BeginElement(1<<i); it != m->EndElement(); ++it){
                        if (it->GetStatus() != Element::Ghost){
                            int num_inner = it->GlobalID();
                            for (int k = 0; k < NumDof[i]; k++) {
                                it->IntegerArray(tags[i].tag)[k] =
                                        (num_inner - MinElem[i]) + InitIndex[i] + k * NumElem[i];
                            }
                        }
                    }
                    break;
                }
                case MINIBLOCKS:{
                    for (auto it = m->BeginElement(1<<i); it != m->EndElement(); ++it){
                        if (it->GetStatus() != Element::Ghost){
                            int num_inner = it->GlobalID();
                            for (int k = 0; k < NumDof[i]; k++) {
                                it->IntegerArray(tags[i].tag)[k] =
                                        (num_inner - MinElem[i])*NumDof[i] + InitIndex[i] + k;
                            }
                        }
                    }
                    break;
                }
            }
            m->ExchangeData(tags[i].tag, 1<<i);
        }
    }

    static_assert(FemExprDescr::GeomType::NGEOM_TYPES == 4 && "Numeration for EDGE_ORIENT and FACE_ORIENT is not implemented");
    std::vector<FemExprDescr::DiscrSpaceHelper> discr_vars;
    for (int i = 0; i < static_cast<int>(info->base_funcs.size()); ++i) discr_vars.emplace_back(info->base_funcs[i].odf);
    auto origin = info->base_funcs[0].odf->OriginFemType();
    auto unite_var = std::make_shared<FemExprDescr::ComplexSpaceHelper>(discr_vars, origin.first, origin.second);

    std::shared_ptr<ProblemGlobEnumeration> res;
    switch (aType) {
        case ANITYPE:{
            struct AniEnumenator: public ProblemGlobEnumeration{
                void Clear() override{
                    for (int i = 0; i < 4; ++i) index_tag[i].Clear();
                    ProblemGlobEnumeration::Clear();
                }
                VectorIndex OrderC(const GeomIndex& geomIndex) const override {
                    VectorIndex res;
                    auto ind = ElementNum(geomIndex.elem.GetElementType());
                    auto iodf = unite_var->m_spaceNumDof[ind][geomIndex.var_id] + geomIndex.var_elem_dof_id;

                    res.id = geomIndex.elem.IntegerArray(index_tag[ind].tag)[iodf];
                    return res;
                }
                VectorIndex operator()(const GeomIndex& geomIndex) const override {
                    VectorIndex res;
                    auto ind = ElementNum(geomIndex.elem.GetElementType());
                    auto iodf = unite_var->m_spaceNumDof[ind][geomIndex.var_id] + geomIndex.var_elem_dof_id;

                    if (geomIndex.elem.GetStatus() != INMOST::Element::Ghost)
                        res.id = (geomIndex.elem.GlobalID() - MinElem[ind]) + InitIndex[ind] + iodf * NumElem[ind];
                    else
                        res.id = VectorIndex::UnValid;
                    return res;
                }
                GeomIndexExt operator()(VectorIndex vectorIndex) const override{
                    assert(vectorIndex.id >= 0 && "Not valid vector index");
                    int ind = 0;
                    if (oType == STRAIGHT){
                        int i = 1;
                        for (i = 1; i < 4 && InitIndex[i] < vectorIndex.id; ++i);
                        ind = i-1;
                    } else {
                        int i = 3;
                        for (i = 2; i >= 0 && InitIndex[i] < vectorIndex.id; --i);
                        ind = i+1;
                    }
                    GeomIndexExt res;
                    int iodf = (vectorIndex.id - InitIndex[ind]) / NumElem[ind];
                    long gid = (vectorIndex.id - InitIndex[ind]) % NumElem[ind];
                    res.elem = mesh->ElementByLocalID(1<<ind, mesh->IntegerArray(mesh->GetHandle(), maps[ind])[gid]);
                    auto& arr = unite_var->m_spaceNumDof[ind];
                    res.var_id = std::upper_bound(arr.data(), arr.data() + arr.size(), iodf) - arr.data() - 1;
                    res.var_elem_dof_id = iodf - unite_var->m_spaceNumDof[ind][res.var_id];
                    res.var_vector_elem_dof_shift = unite_var->m_spaceNumDof[ind][res.var_id];

                    return res;
                }

                long InitIndex[4];
                ORDER_TYPE oType;
                Tag maps[4];
                std::array<TagWrap, 4> index_tag;
            };
            res = std::make_shared<AniEnumenator>();
            auto s = static_cast<AniEnumenator*>(res.get());
            std::copy(InitIndex, InitIndex + 4, s->InitIndex);
            s->oType = oType;
            for (int i = 0; i < 4; ++i) s->maps[i] = maps[i];
            s->index_tag = std::move(tags);
            break;
        }
        case MINIBLOCKS:{
            struct MiniBlockEnumenator: public ProblemGlobEnumeration{
                void Clear() override{
                    for (int i = 0; i < 4; ++i) index_tag[i].Clear();
                    ProblemGlobEnumeration::Clear();
                }
                VectorIndex OrderC(const GeomIndex& geomIndex) const override {
                    VectorIndex res;
                    auto ind = ElementNum(geomIndex.elem.GetElementType());
                    auto iodf = unite_var->m_spaceNumDof[ind][geomIndex.var_id] + geomIndex.var_elem_dof_id;
                    res.id = geomIndex.elem.IntegerArray(index_tag[ind].tag)[iodf];
                    return res;
                }
                VectorIndex operator()(const GeomIndex& geomIndex) const override {
                    VectorIndex res;
                    auto ind = ElementNum(geomIndex.elem.GetElementType());
                    auto iodf = unite_var->m_spaceNumDof[ind][geomIndex.var_id] + geomIndex.var_elem_dof_id;

                    if (geomIndex.elem.GetStatus() != INMOST::Element::Ghost)
                        res.id = (geomIndex.elem.GlobalID() - MinElem[ind])*NumDof[ind] + InitIndex[ind] + iodf;
                    else
                        res.id = VectorIndex::UnValid;
                    return res;
                }
                GeomIndexExt operator()(VectorIndex vectorIndex) const override{
                    int ind = 0;
                    if (oType == STRAIGHT){
                        int i = 1;
                        for (i = 1; i < 4 && InitIndex[i] < vectorIndex.id; ++i);
                        ind = i-1;
                    } else {
                        int i = 3;
                        for (i = 2; i >= 0 && InitIndex[i] < vectorIndex.id; --i);
                        ind = i+1;
                    }
                    GeomIndexExt res;
                    int gid = (vectorIndex.id - InitIndex[ind]) / NumDof[ind];
                    long iodf = (vectorIndex.id - InitIndex[ind]) % NumDof[ind];
                    res.elem = mesh->ElementByLocalID(1<<ind, mesh->IntegerArray(mesh->GetHandle(), maps[ind])[gid]);
                    auto& arr = unite_var->m_spaceNumDof[ind];
                    res.var_id = std::upper_bound(arr.data(), arr.data() + arr.size(), iodf) - arr.data() - 1;
                    res.var_elem_dof_id = iodf - unite_var->m_spaceNumDof[ind][res.var_id];
                    res.var_vector_elem_dof_shift = unite_var->m_spaceNumDof[ind][res.var_id];

                    return res;
                }

                long InitIndex[4];
                ORDER_TYPE oType;
                Tag maps[4];
                std::array<TagWrap, 4> index_tag;
            };
            res = std::make_shared<MiniBlockEnumenator>();
            auto s = static_cast<MiniBlockEnumenator*>(res.get());
            std::copy(InitIndex, InitIndex + 4, s->InitIndex);
            s->oType = oType;
            for (int i = 0; i < 4; ++i) s->maps[i] = maps[i];
            s->index_tag = std::move(tags);
            break;
        }
        default:
            std::cout<<"Wrong assembling type"<<std::endl;
            abort();
    }
    std::copy(NumDof, NumDof + 4, res->NumDof);
    std::copy(NumElem, NumElem + 4, res->NumElem);
    std::copy(MinElem, MinElem + 4, res->MinElem);
    std::copy(MaxElem, MaxElem + 4, res->MaxElem);
    res->BegInd = BegInd, res->EndInd = EndInd, res->MatrSize = MatrSize;
    res->mesh = m;
    res->unite_var = std::move(unite_var);
    return res;
}

std::function<void(ElementalAssembler &p)> Assembler::makeInitValueSetter(double val) {
    auto initValueSetter = [val](ElementalAssembler &p) -> void {
        std::fill(p.vars->initValues.begin(), p.vars->initValues.end(), val);
    };
    return initValueSetter;
}

std::function<void(ElementalAssembler &p)> Assembler::makeInitValueSetter(std::vector<double> vals) {
    auto initValueSetter = [vals](ElementalAssembler &p) -> void {
        assert(static_cast<int>(vals.size()) == p.vars->descr->NumVars() && "Wrong number of initializing vals");
        for (int v = 0; v < p.vars->NumBaseVars(); ++v){
            std::fill(p.vars->begin(v), p.vars->end(v), vals[v]);
        }

    };
    return initValueSetter;
}

static void makeInitValueSetterByTag(ElementalAssembler& p, const INMOST::Tag& tag_x){
    auto chooser = [](int ti) -> std::function<INMOST::Storage::real(const FemExprDescr::LocalOrder&, ElementalAssembler &, const INMOST::Tag&)>{
        switch (ti) {
            case 0: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x){
                    return (*(p.nodes))[lo.nelem].RealArray(tag_x)[lo.loc_elem_dof_id];
                };
            case 1: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x){
                    return (*(p.edges))[p.local_edge_index[lo.nelem]].RealArray(tag_x)[lo.loc_elem_dof_id];
                };
            case 2: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x){
                    return (*(p.faces))[p.local_face_index[lo.nelem]].RealArray(tag_x)[lo.loc_elem_dof_id];
                };
            case 3: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x) -> INMOST::Storage::real{
                    return p.cell->RealArray(tag_x)[lo.loc_elem_dof_id];
                };
        }
        return nullptr;
    };
    for (int ti = 0; ti < 4; ++ti) {
        auto choose = chooser(ti);
        auto t = static_cast<FemExprDescr::GeomType >(ti);
        int loffset = 0;
        for (int v = 0; v < p.vars->NumBaseVars(); ++v) {
            auto initVal = p.vars->begin(v);
            auto& var = p.vars->descr->base_funcs[v].odf;
            auto ndof = var->NumDofOnTet(t);
            for (int ldof = 0; ldof < ndof; ++ldof){
                auto lo = var->LocalOrderOnTet(t, ldof);
                lo.loc_elem_dof_id += loffset;
                initVal[lo.gid] = choose(lo, p, tag_x);
            }
            loffset += ndof;
        }
    }
}

std::function<void(ElementalAssembler& p)> Assembler::makeInitValueSetter(INMOST::Tag* tag_x) {
    auto initValueSetter = [tag_x](ElementalAssembler &p) -> void {
        makeInitValueSetterByTag(p, *tag_x);
    };
    return initValueSetter;
}

std::function<void(ElementalAssembler& p)> Assembler::makeInitValueSetter(INMOST::Tag tag_x) {
    auto initValueSetter = [tag_x](ElementalAssembler &p) -> void {
        makeInitValueSetterByTag(p, tag_x);
    };
    return initValueSetter;
}

std::function<void(ElementalAssembler& p)> Assembler::makeInitValueSetter(std::vector<INMOST::Tag> tag_vec){
    auto initValueSetter = [tag_vec](ElementalAssembler &p) -> void {
        assert(static_cast<int>(tag_vec.size()) == p.vars->NumBaseVars() && "Wrong number of initializing tags");
        auto chooser = [](int ti) -> std::function<INMOST::Storage::real(const FemExprDescr::LocalOrder&, ElementalAssembler &, const INMOST::Tag&)>{
            switch (ti) {
                case 0: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x){
                        return (*(p.nodes))[lo.nelem].RealArray(tag_x)[lo.loc_elem_dof_id];
                    };
                case 1: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x){
                        return (*(p.edges))[p.local_edge_index[lo.nelem]].RealArray(tag_x)[lo.loc_elem_dof_id];
                    };
                case 2: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x){
                        return (*(p.faces))[p.local_face_index[lo.nelem]].RealArray(tag_x)[lo.loc_elem_dof_id];
                    };
                case 3: return []( const FemExprDescr::LocalOrder& lo, ElementalAssembler &p, const INMOST::Tag& tag_x) -> INMOST::Storage::real{
                        return p.cell->RealArray(tag_x)[lo.loc_elem_dof_id];
                    };
            }
            return nullptr;
        };
        for (int ti = 0; ti < 4; ++ti) {
            auto choose = chooser(ti);
            auto t = static_cast<FemExprDescr::GeomType >(ti);
            for (int v = 0; v < p.vars->NumTestVars(); ++v) {
                auto initVal = p.vars->begin(v);
                auto& var = p.vars->descr->base_funcs[v].odf;
                auto ndof = var->NumDofOnTet(t);
                for (int ldof = 0; ldof < ndof; ++ldof){
                    auto lo = var->LocalOrderOnTet(t, ldof);
                    initVal[lo.gid] = choose(lo, p, tag_vec[v]);
                }
            }
        }
    };
    return initValueSetter;
}

void Assembler::SaveSolution(const INMOST::Sparse::Vector& from, INMOST::Tag& to) const{
    ElementType mask = m_enum->GetGeomMask();
#ifndef NDEBUG
    for (int i = 0; i < 4; ++i) {
        assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_enum->NumDof[i]) && "Tag has not enough size");
    }
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = m_enum->beginByGeom(); it != m_enum->endByGeom(); ++it){
        auto res = *it;
        res.GetGeomInd().elem.RealArray(to)[res.GetGeomInd().GetElemDofInd()] = from[res.GetVecInd().id];
    }
    m_mesh->ExchangeData(to, mask);
}

void Assembler::SaveSolution(const INMOST::Tag& from, INMOST::Sparse::Vector& to) const{
    to.SetInterval(m_enum->BegInd, m_enum->EndInd);
    for (auto it = m_enum->beginByGeom(); it != m_enum->endByGeom(); ++it){
        auto res = *it;
        to[res.GetVecInd().id] = res.GetGeomInd().elem.RealArray(from)[res.GetGeomInd().GetElemDofInd()];
    }
}

void Assembler::SaveSolution(const std::vector<INMOST::Tag>& from, INMOST::Sparse::Vector& to) const{
    to.SetInterval(m_enum->BegInd, m_enum->EndInd);
    for (auto it = m_enum->beginByGeom(); it != m_enum->endByGeom(); ++it){
        auto res = *it;
        to[res.GetVecInd().id] = res.GetGeomInd().elem.RealArray(from[res.GetGeomInd().var_id])[res.GetGeomInd().var_elem_dof_id];
    }
}

void Assembler::SaveSolution(const INMOST::Sparse::Vector& from, std::vector<INMOST::Tag> to) const{
    ElementType mask = m_enum->GetGeomMask();
#ifndef NDEBUG
    assert(to.size() == m_info.base_funcs.size() && "Wrong number of tags");
    for (unsigned v = 0; v < to.size(); ++v) {
        INMOST::ElementType mask = 0;
        for (int i = 0; i < 4; ++i) {
            mask |= (m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i)) > 0) ? (1 << i) : 0;
            assert(to[v].GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i))) &&
                   "Tag has not enough size");
        }
        assert(to[v].isDefinedMask(mask) && "Tag is not defined on the geom mask");
    }
#endif
    for (auto it = m_enum->beginByGeom(); it != m_enum->endByGeom(); ++it){
        auto res = *it;
        res.GetGeomInd().elem.RealArray(to[res.GetGeomInd().var_id])[res.GetGeomInd().var_elem_dof_id] = from[res.GetVecInd().id];
    }
    m_mesh->ExchangeData(to, mask);
}

void Assembler::SaveSolution(const INMOST::Tag& from, INMOST::Tag& to) const{
    ElementType mask = m_enum->GetGeomMask();
#ifndef NDEBUG
    for (int i = 0; i < 4; ++i) {
        assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_enum->NumDof[i]) && "Tag has not enough size");
    }
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = m_mesh->BeginElement(mask); it != m_mesh->EndElement(); ++it){
        for (int k = 0, k_sz = m_enum->NumDof[ElementNum(it->GetElementType())]; k < k_sz; ++k)
        it->RealArray(to)[k] = it->RealArray(from)[k];
    }
}

void Assembler::SaveSolution(const INMOST::Tag& from, std::vector<INMOST::Tag> to) const{
    ElementType mask = m_enum->GetGeomMask();
#ifndef NDEBUG
    assert(to.size() == m_info.base_funcs.size() && "Wrong number of tags");
    for (unsigned v = 0; v < to.size(); ++v) {
        INMOST::ElementType mask = 0;
        for (int i = 0; i < 4; ++i) {
            mask |= (m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i)) > 0) ? (1 << i) : 0;
            assert(to[v].GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i))) &&
                   "Tag has not enough size");
        }
        assert(to[v].isDefinedMask(mask) && "Tag is not defined on the geom mask");
    }
#endif
    for (auto it = m_mesh->BeginElement(mask); it != m_mesh->EndElement(); ++it){
        auto ti = ElementNum(it->GetElementType());
        for (int v = 0; v < static_cast<int>(m_enum->unite_var->m_spaces.size()); ++v)
            for (int k = 0,
                     k_shft = m_enum->unite_var->m_spaceNumDof[ti][v],
                     k_sz = m_enum->unite_var->m_spaceNumDof[ti][v+1] - m_enum->unite_var->m_spaceNumDof[ti][v]; 
                        k < k_sz; ++k)
                it->RealArray(to[v])[k] = it->RealArray(from)[k_shft + k];
    }
}

void Assembler::SaveSolution(const std::vector<INMOST::Tag>& from, INMOST::Tag& to) const{
    ElementType mask = m_enum->GetGeomMask();
#ifndef NDEBUG
    assert(from.size() == m_info.base_funcs.size() && "Wrong number of tags");
    for (int i = 0; i < 4; ++i) {
        assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_enum->NumDof[i]) && "Tag has not enough size");
    }
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = m_mesh->BeginElement(mask); it != m_mesh->EndElement(); ++it){
        auto ti = ElementNum(it->GetElementType());
        for (unsigned v = 0; v < m_enum->unite_var->m_spaces.size(); ++v)
            for (int k = 0,
                         k_shft = m_enum->unite_var->m_spaceNumDof[ti][v],
                         k_sz = m_enum->unite_var->m_spaceNumDof[ti][v+1] - m_enum->unite_var->m_spaceNumDof[ti][v]; k < k_sz; ++k)
                it->RealArray(to)[k_shft + k] = it->RealArray(from[v])[k];
    }
}

void Assembler::SaveVar(const INMOST::Sparse::Vector& from, int v, INMOST::Tag& to) const {
    INMOST::ElementType mask = 0;
    for (int i = 0; i < 4; ++i) {
        mask |= (m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i)) > 0) ? (1 << i) : 0;
    }
#ifndef NDEBUG
    {
        for (int i = 0; i < 4; ++i) {
            assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i))) &&
                   "Tag has not enough size");
        }
        assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
    }
#endif
    for (auto it = m_enum->beginByGeom(mask, v, 0, ProblemGlobEnumeration::SliceIteratorData::SAME_VAR); it != m_enum->endByGeom(); ++it){
        auto res = *it;
        res.GetGeomInd().elem.RealArray(to)[res.GetGeomInd().var_elem_dof_id] = from[res.GetVecInd().id];
    }
    m_mesh->ExchangeData(to, mask);
}

void Assembler::SaveVar(const INMOST::Tag& from, int v, INMOST::Tag& to) const {
    INMOST::ElementType mask = 0;
    for (int i = 0; i < 4; ++i) {
        mask |= (m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i)) > 0) ? (1 << i) : 0;
    }
#ifndef NDEBUG
    {
        for (int i = 0; i < 4; ++i) {
            assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i))) &&
                   "Tag has not enough size");
        }
        assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
    }
#endif
    for (auto it = m_mesh->BeginElement(mask); it != m_mesh->EndElement(); ++it){
        auto ti = ElementNum(it->GetElementType());
        for (int k = 0, k_shft = m_enum->unite_var->m_spaceNumDof[ti][v],
                     k_sz = m_enum->unite_var->m_spaceNumDof[ti][v+1] - m_enum->unite_var->m_spaceNumDof[ti][v]; k < k_sz; ++k)
            it->RealArray(to)[k] = it->RealArray(from)[k_shft + k];
    }
}

void Assembler::SaveVar(int v, const INMOST::Tag& from, INMOST::Tag& to) const {
    INMOST::ElementType mask = 0;
    for (int i = 0; i < 4; ++i) {
        mask |= (m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i)) > 0) ? (1 << i) : 0;
    }
#ifndef NDEBUG
    for (int i = 0; i < 4; ++i) {
        assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_enum->NumDof[i]) && "Tag has not enough size");
    }
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = m_mesh->BeginElement(mask); it != m_mesh->EndElement(); ++it){
        auto ti = ElementNum(it->GetElementType());
        for (int k = 0, k_shft = m_enum->unite_var->m_spaceNumDof[ti][v],
                     k_sz = m_enum->unite_var->m_spaceNumDof[ti][v+1] - m_enum->unite_var->m_spaceNumDof[ti][v]; k < k_sz; ++k)
            it->RealArray(to)[k_shft + k] = it->RealArray(from)[k];
    }
}

void Assembler::CopyVar(int v, const INMOST::Tag& from, INMOST::Tag& to) const {
    INMOST::ElementType mask = 0;
    for (int i = 0; i < 4; ++i) {
        mask |= (m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i)) > 0) ? (1 << i) : 0;
    }
#ifndef NDEBUG
    for (int i = 0; i < 4; ++i) {
            assert(to.GetSize() >= static_cast<INMOST_DATA_ENUM_TYPE>(m_info.base_funcs[v].odf->NumDof(static_cast<FemExprDescr::GeomType>(i))) &&
                   "Tag has not enough size");
        }
    assert(to.isDefinedMask(mask) && "Tag is not defined on the geom mask");
#endif
    for (auto it = m_mesh->BeginElement(mask); it != m_mesh->EndElement(); ++it){
        auto ti = ElementNum(it->GetElementType());
        int k_sz = m_enum->unite_var->m_spaceNumDof[ti][v+1] - m_enum->unite_var->m_spaceNumDof[ti][v];
        for (int k = 0; k < k_sz; ++k)
            it->RealArray(to)[k] = it->RealArray(from)[k];
    }
}

bool Assembler::find_local_edge_index(const ElementArray<Node>& nodes, const ElementArray<Edge>& edges, int local_edge_index[6]){
    std::array<std::array<INMOST::Storage::integer, 3>, 6> nds_map, edgs_map;
    std::array<INMOST::Storage::integer, 4> inds_loc = {nodes[0].DataLocalID(), nodes[1].DataLocalID(), nodes[2].DataLocalID(), nodes[3].DataLocalID()};
    for (int i = 0, cnt = 0; i < 3; i++)
        for (int j = i + 1; j < 4; j++) {
            auto id1 = inds_loc[i], id2 = inds_loc[j];
            if (id2 < id1) std::swap(id1, id2);
            nds_map[cnt] = std::array<INMOST::Storage::integer, 3>{id1, id2, cnt};
            ++cnt;
        }
    for (int k = 0; k < 6; ++k){
        auto nds = edges[k].getNodes();
        auto id1 = nds[0].DataLocalID(), id2 = nds[1].DataLocalID();
        if (id2 < id1) std::swap(id1, id2);
        edgs_map[k] = std::array<INMOST::Storage::integer, 3>{id1, id2, k};
    }
    static auto comp = [](const std::array<INMOST::Storage::integer, 3>& a, const std::array<INMOST::Storage::integer, 3>& b){
        return std::lexicographical_compare(a.data(), a.data() + 2, b.data(), b.data() + 2);
    };
    std::sort(nds_map.begin(), nds_map.end(), comp);
    std::sort(edgs_map.begin(), edgs_map.end(), comp);
    for (int k = 0; k < 6; ++k){
        auto &dat1 = nds_map[k], &dat2 = edgs_map[k];
        bool same = (dat1[0] == dat2[0]) && (dat1[1] == dat2[1]);
        if (!same) return false;
        local_edge_index[dat1[2]] = dat2[2];
    }

    return true;
}

bool Assembler::find_local_face_index(const ElementArray<Node>& nodes, const ElementArray<Face>& faces, int local_face_index[4]){
    std::array<std::array<INMOST::Storage::integer, 4>, 4> nds_map, fcs_map;
    std::array<INMOST::Storage::integer, 4> inds_loc = {nodes[0].DataLocalID(), nodes[1].DataLocalID(), nodes[2].DataLocalID(), nodes[3].DataLocalID()};
    static auto three_sort = [](INMOST::Storage::integer& i1, INMOST::Storage::integer& i2, INMOST::Storage::integer& i3){
        if (i1 > i2) {
            if (i2 > i3) {
                std::swap(i1, i3);
            } else {
                if (i1 > i3) {
                    auto tmp = i1;
                    i1 = i2;
                    i2 = i3;
                    i3 = tmp;
                } else {
                    std::swap(i1, i2);
                }
            }
        } else {
            if (i2 < i3) return ;
            if (i1 > i3){
                auto tmp = i1;
                i1 = i3;
                i3 = i2;
                i2 = tmp;
            } else {
                std::swap(i2, i3);
            }
        }
    };
    for (int i = 0; i < 4; i++) {
        auto id1 = inds_loc[i], id2 = inds_loc[(i+1)%4], id3 = inds_loc[(i+2)%4];
        three_sort(id1, id2, id3);
        assert(id1 <= id2 && id2 <= id3 && "sort error");
        nds_map[i] = std::array<INMOST::Storage::integer, 4>{id1, id2, id3, i};
    }
    for (int i = 0; i < 4; ++i){
        auto nds = faces[i].getNodes();
        auto id1 = nds[0].DataLocalID(), id2 = nds[1].DataLocalID(), id3 = nds[2].DataLocalID();
        three_sort(id1, id2, id3);
        assert(id1 <= id2 && id2 <= id3 && "sort error");
        fcs_map[i] = std::array<INMOST::Storage::integer, 4>{id1, id2, id3, i};
    }
    static auto comp = [](const std::array<INMOST::Storage::integer, 4>& a, const std::array<INMOST::Storage::integer, 4>& b){
        return std::lexicographical_compare(a.data(), a.data() + 3, b.data(), b.data() + 3);
    };
    std::sort(nds_map.begin(), nds_map.end(), comp);
    std::sort(fcs_map.begin(), fcs_map.end(), comp);
    for (int k = 0; k < 4; ++k){
        auto &dat1 = nds_map[k], &dat2 = fcs_map[k];
        bool same = (dat1[0] == dat2[0]) && (dat1[1] == dat2[1]) && (dat1[2] == dat2[2]);
        if (!same) return false;
        local_face_index[dat1[3]] = dat2[3];
    }

    return true;
}

void Assembler::extend_memory_for_fem_func(ElemMatEval *func) {
    if (!func) return;
    size_t sz_iw, sz_w, sz_args, sz_res;
    func->working_sizes(sz_args, sz_res, sz_w, sz_iw);
    if (m_w.m_w.size() < sz_w) m_w.m_w.resize(sz_w);
    if (m_w.m_iw.size() < sz_iw) m_w.m_iw.resize(sz_iw);
    if (m_w.m_args.size() < sz_args) m_w.m_args.resize(sz_args);
    if (m_w.m_res.size() < sz_res) m_w.m_res.resize(sz_res);
}

void Assembler::PrepareProblem() {
    if (!m_mesh) throw std::runtime_error("Mesh was not specified");
    if (m_info.base_funcs.empty()) throw std::runtime_error("Description of fem expression is empty, try SetProbDescr(...)");
    if (m_enum == nullptr) {
        static unsigned int unique_num = 0;
        m_enum = DefaultEnumeratorFactory(*m_mesh, m_info).build("_GenEnum" + std::to_string(unique_num));
        unique_num++;
    }
    if (initial_value_setter == nullptr) initial_value_setter = makeInitValueSetter();

    long nRows = 4*m_enum->NumDof[0] + 6*m_enum->NumDof[1] + 4*m_enum->NumDof[2] + m_enum->NumDof[3];
    m_w.m_A.resize(nRows*nRows), m_w.m_F.resize(nRows);
    m_w.m_Ab.resize(nRows * nRows, true);
    m_w.m_indexesR.resize(nRows), m_w.m_indexesC.resize(nRows);
    m_helper.descr = &m_info;
    m_helper.initValues.resize(m_info.NumDofOnTet());
    m_helper.base_MemOffsets.resize(m_info.NumVars() + 1);
    m_helper.test_MemOffsets.resize(m_info.test_funcs.size() + 1);
    m_helper.base_MemOffsets[0] = m_helper.test_MemOffsets[0] = 0;
    for (int i = 1; i < m_info.NumVars()+1; ++i){
        m_helper.base_MemOffsets[i] = m_helper.base_MemOffsets[i-1] + m_info.base_funcs[i-1].odf->NumDofOnTet();
    }
    for (int i = 1; i < static_cast<int>(m_info.test_funcs.size()+1); ++i){
        m_helper.test_MemOffsets[i] = m_helper.test_MemOffsets[i-1] + m_info.test_funcs[i-1].odf->NumDofOnTet();
    }
    m_helper.same_template = (m_info.base_funcs.size() == m_info.test_funcs.size());
    for (int i = 0; i < static_cast<int>(m_info.base_funcs.size()) && m_helper.same_template; ++i)
        m_helper.same_template &= (m_info.base_funcs[i].odf == m_info.test_funcs[i].odf);

    extend_memory_for_fem_func(mat_func.get());
    extend_memory_for_fem_func(rhs_func.get());
    extend_memory_for_fem_func(mat_rhs_func.get());
    if (mat_func && !mat_rhs_func){
        if (mat_func->out_nnz(0) != mat_func->out_size1(0) * mat_func->out_size2(0)) {
            m_w.colindA.resize(mat_func->out_size2(0) + 1), m_w.rowA.resize(mat_func->out_nnz(0));
            mat_func->out_csc(0, m_w.colindA.data(), m_w.rowA.data());
        }
    }
    if (rhs_func && !mat_rhs_func){
        if (rhs_func->out_nnz(0) != rhs_func->out_size1(0) * rhs_func->out_size2(0)) {
            m_w.colindF.resize(rhs_func->out_size2(0) + 1), m_w.rowF.resize(rhs_func->out_nnz(0));
            rhs_func->out_csc(0, m_w.colindF.data(), m_w.rowF.data());
        }
    }
    if (mat_rhs_func){
        if (mat_rhs_func->out_nnz(0) != mat_rhs_func->out_size1(0) * mat_rhs_func->out_size2(0)) {
            m_w.colindA.resize(mat_rhs_func->out_size2(0) + 1), m_w.rowA.resize(mat_rhs_func->out_nnz(0));
            mat_rhs_func->out_csc(0, m_w.colindA.data(), m_w.rowA.data());
            m_w.colindF.resize(mat_rhs_func->out_size2(1) + 1), m_w.rowF.resize(mat_rhs_func->out_nnz(1));
            mat_rhs_func->out_csc(1, m_w.colindF.data(), m_w.rowF.data());
        }
    }
    if (!rhs_func && !mat_func && !mat_rhs_func) throw std::runtime_error("Fem expression evaluator is not set");
    orderR.resize(nRows);
    orderC.resize(nRows);
    for (int v = 0, offset = 0; v < m_info.NumVars(); ++v){
        auto nDOF = m_info.base_funcs[v].odf->NumDofOnTet();
        for (int i = 0; i < nDOF; ++i){
            auto ord = m_info.base_funcs[v].odf->LocalOrderOnTet(i);
            orderC[offset + i].var_id = v;
            orderC[offset + i].etype = (1 << ord.etype);
            orderC[offset + i].nelem = ord.nelem;
            orderC[offset + i].loc_elem_dof_id = ord.loc_elem_dof_id;
        }
        offset += nDOF;
    }
    if (!m_helper.same_template){
        for (int v = 0, offset = 0; v < static_cast<int>(m_info.test_funcs.size()); ++v){
            auto nDOF = m_info.test_funcs[v].odf->NumDofOnTet();
            for (int i = 0; i < nDOF; ++i){
                auto ord = m_info.test_funcs[v].odf->LocalOrderOnTet(i);
                orderR[offset + i].var_id = v;
                orderR[offset + i].etype = (1 << ord.etype);
                orderR[offset + i].nelem = ord.nelem;
                orderR[offset + i].loc_elem_dof_id = ord.loc_elem_dof_id;
            }
            offset += nDOF;
        }
    } else {
        std::copy(orderC.begin(), orderC.end(), orderR.begin());
    }
}

bool Assembler::fill_assemble_templates(int nRows,
                                         ElementArray<Node>& nodes, ElementArray<Edge>& edges, ElementArray<Face>& faces, Mesh::iteratorCell it,
                                         std::vector<int>& indexesC, std::vector<int>& indexesR,
                                         int* local_edge_index, int* local_face_index){
    bool has_active = false;
    if (m_helper.same_template) {
        for (int i = 0; i < nRows; ++i) {
            auto ord = orderR[i];
            ProblemGlobEnumeration::GeomIndex ind;
            ind.var_id = ord.var_id;
            ind.var_elem_dof_id = ord.loc_elem_dof_id;
            switch (ord.etype) {
                case INMOST::NODE: {
                    ind.elem = nodes[ord.nelem].getAsElement();
                    break;
                }
                case INMOST::EDGE: {
                    ind.elem = edges[local_edge_index[ord.nelem]].getAsElement();
                    break;
                }
                case INMOST::FACE: {
                    ind.elem = faces[local_face_index[ord.nelem]].getAsElement();
                    break;
                }
                case INMOST::CELL: {
                    ind.elem = it->getAsElement();
                    break;
                }
            }
            bool is_active = (ind.elem.GetStatus() != INMOST::Element::Ghost);
            indexesC[i] = m_enum->OrderC(ind).id;
            indexesR[i] = is_active ? indexesC[i] : -1;
            if (is_active) has_active = true;
        }
    } else {
        for (int i = 0; i < nRows; ++i) {
            auto ord = orderR[i];
            ProblemGlobEnumeration::GeomIndex ind;
            ind.var_id = ord.var_id;
            ind.var_elem_dof_id = ord.loc_elem_dof_id;
            switch (ord.etype) {
                case INMOST::NODE: {
                    ind.elem = nodes[ord.nelem].getAsElement();
                    break;
                }
                case INMOST::EDGE: {
                    ind.elem = edges[local_edge_index[ord.nelem]].getAsElement();
                    break;
                }
                case INMOST::FACE: {
                    ind.elem = faces[local_face_index[ord.nelem]].getAsElement();
                    break;
                }
                case INMOST::CELL: {
                    ind.elem = it->getAsElement();
                    break;
                }
            }
            if (ind.elem.GetStatus() != INMOST::Element::Ghost) {
                indexesR[i] = m_enum->OrderR(ind).id;
                has_active = true;
            } else
                indexesR[i] = -1;
        }
        if (!has_active) return has_active;
        for (int i = 0; i < nRows; ++i) {
            auto ord = orderC[i];
            ProblemGlobEnumeration::GeomIndex ind;
            ind.var_id = ord.var_id;
            ind.var_elem_dof_id = ord.loc_elem_dof_id;
            switch (ord.etype) {
                case INMOST::NODE: {
                    ind.elem = nodes[ord.nelem].getAsElement();
                    break;
                }
                case INMOST::EDGE: {
                    ind.elem = edges[local_edge_index[ord.nelem]].getAsElement();
                    break;
                }
                case INMOST::FACE: {
                    ind.elem = faces[local_face_index[ord.nelem]].getAsElement();
                    break;
                }
                case INMOST::CELL: {
                    ind.elem = it->getAsElement();
                    break;
                }
            }
            indexesC[i] = m_enum->OrderC(ind).id;
        }
    }
    return has_active;
}

void Assembler::generate_mat_rhs_func() {
    struct MatRhsWrap: public ElemMatEval{
    private:
        std::shared_ptr<ElemMatEval> m_mat_func;
        std::shared_ptr<ElemMatEval> m_rhs_func;
    public:
        MatRhsWrap(std::shared_ptr<ElemMatEval> mat_func, std::shared_ptr<ElemMatEval> rhs_func):
                m_mat_func{std::move(mat_func)}, m_rhs_func{std::move(rhs_func)}{}
        void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
            Real* rhs_p = res[1];
            m_mat_func->operator()(args, res, w, iw, user_data);
            res[1] = rhs_p;
            m_rhs_func->operator()(args, res+1, w, iw, user_data);
        }
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override{
            m_mat_func->working_sizes(sz_args, sz_res, sz_w, sz_iw);
            size_t sz_args1 = 0, sz_res1 = 0, sz_w1 = 0, sz_iw1 = 0;
            m_rhs_func->working_sizes(sz_args1, sz_res1, sz_w1, sz_iw1);
            if (sz_args1 > sz_args) sz_args = sz_args1;
            if (sz_res1+1 > sz_res) sz_res = sz_res1+1;
            if (sz_w1 > sz_w) sz_w = sz_w1;
            if (sz_iw1 > sz_iw) sz_iw = sz_iw1;
        }
        size_t n_in() const override{ return std::max(m_mat_func->n_in(), m_rhs_func->n_in()); };
        size_t n_out() const override{ return 1 + m_rhs_func->n_out(); };
        Int in_nnz(Int arg_id) const override{ return m_rhs_func->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const override{ return m_rhs_func->in_size1(arg_id); };
        Int in_size2(Int arg_id) const override{ return m_rhs_func->in_size2(arg_id); };
        void in_csc(Int arg_id, Int* colind, Int* row) const override { m_rhs_func->in_csc(arg_id, colind, row); }
        Int out_nnz(Int res_id) const override{ return (res_id == 0) ? m_mat_func->out_nnz(0) : m_rhs_func->out_nnz(res_id-1); };
        Int out_size1(Int res_id) const override{ return (res_id == 0) ? m_mat_func->out_size1(0) : m_rhs_func->out_size1(res_id-1); };
        Int out_size2(Int res_id) const override{ return (res_id == 0) ? m_mat_func->out_size2(0) : m_rhs_func->out_size2(res_id-1);  };
        void out_csc(Int res_id, Int* colind, Int* row) const override{
            return (res_id == 0) ? m_mat_func->out_csc(0, colind, row) : m_rhs_func->out_csc(res_id-1, colind, row);
        }
    };
    SetMatRHSFunc(std::make_shared<MatRhsWrap>(mat_func, rhs_func));
    extend_memory_for_fem_func(mat_rhs_func.get());
}

void Assembler::generate_mat_func() {
    struct MatWrap: public ElemMatEval{
    private:
        std::shared_ptr<ElemMatEval> func;
    public:
        MatWrap(std::shared_ptr<ElemMatEval> mat_rhs_func):
                func{std::move(mat_rhs_func)}{}
        void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
            func->operator()(args, res, w, iw, user_data);
        }
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override{
            func->working_sizes(sz_args, sz_res, sz_w, sz_iw);
        }
        size_t n_in() const override{ return func->n_in(); };
        size_t n_out() const override{ return 1; };
        Int in_nnz(Int arg_id) const override{ return func->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const override{ return func->in_size1(arg_id); };
        Int in_size2(Int arg_id) const override{ return func->in_size2(arg_id); };
        void in_csc(Int arg_id, Int* colind, Int* row) const override { func->in_csc(arg_id, colind, row); }
        Int out_nnz(Int res_id) const override{ return func->out_nnz(res_id); };
        Int out_size1(Int res_id) const override{ return func->out_size1(res_id); };
        Int out_size2(Int res_id) const override{ return func->out_size2(res_id);  };
        void out_csc(Int res_id, Int* colind, Int* row) const override{ func->out_csc(res_id, colind, row); }
    };
    SetMatFunc(std::make_shared<MatWrap>(mat_rhs_func));
}

void Assembler::generate_rhs_func() {
    struct RhsWrap: public ElemMatEval{
    private:
        std::shared_ptr<ElemMatEval> func;
    public:
        RhsWrap(std::shared_ptr<ElemMatEval> mat_rhs_func):
                func{std::move(mat_rhs_func)}{}
        void operator()(const Real** args, Real** res, Real* w, Int* iw, void* user_data) override{
            func->operator()(args, res, w, iw, user_data);
            res[0] = res[1];
        }
        void working_sizes(size_t& sz_args, size_t& sz_res, size_t& sz_w, size_t& sz_iw) const override{
            func->working_sizes(sz_args, sz_res, sz_w, sz_iw);
        }
        size_t n_in() const override{ return func->n_in(); };
        size_t n_out() const override{ return 1; };
        Int in_nnz(Int arg_id) const override{ return func->in_nnz(arg_id); }
        Int in_size1(Int arg_id) const override{ return func->in_size1(arg_id); };
        Int in_size2(Int arg_id) const override{ return func->in_size2(arg_id); };
        void in_csc(Int arg_id, Int* colind, Int* row) const override { func->in_csc(arg_id, colind, row); }
        Int out_nnz(Int res_id) const override{ return func->out_nnz(res_id+1); };
        Int out_size1(Int res_id) const override{ return func->out_size1(res_id+1); };
        Int out_size2(Int res_id) const override{ return func->out_size2(res_id+1);  };
        void out_csc(Int res_id, Int* colind, Int* row) const override{ func->out_csc(res_id+1, colind, row); }
    };
    SetRHSFunc(std::make_shared<RhsWrap>(mat_rhs_func));
}

static inline void print_assemble_matrix_indeces_incompatible(std::ostream& out, Mesh* m,
                                                              std::vector<int> &indexesC, long* NumElem, long* MinElem, long* MaxElem,
                                                              int MatrSize, int BegInd, int EndInd, int nRows, int wrong_index) {
    out<<"Proc rank "<<m->GetProcessorRank()<<std::endl;
    out<<"wrong index R "<<indexesC[wrong_index]<<" "<<0<<" "<<MatrSize<<std::endl;
    out<<"Num elems"<<std::endl;
    for(int i1=0;i1<4;i1++)
        out<<NumElem[i1]<<" ";
    out<<std::endl;
    out<<"Min elems"<<std::endl;
    for(int i1=0;i1<4;i1++)
        out<<MinElem[i1]<<" ";
    out<<"Max elems"<<std::endl;
    for(int i1=0;i1<4;i1++)
        out<<MaxElem[i1]<<" ";
    out<<std::endl;
    out<<"Begin index "<< BegInd<<" End index "<<EndInd<< std::endl;
    out<<"indexes"<<std::endl;
    for(int i1=0;i1<nRows;i1++)
        out<<indexesC[i1]<<" ";
    out<<std::endl;
}

static inline void print_rhs_nan(std::ostream& out, std::vector<double>& F, Mesh::iteratorCell it, int nRows, int nan_ind) {
    out<<"not a number in rhs"<<"\n";
    out<<"\tF["<<nan_ind <<"] = " << F[nan_ind] <<"\n";
    for(int i1=0;i1<nRows;i1++) { out << "\t\t" << F[i1] << "\n"; }
    out<<"\tnum tet " << it->DataLocalID() << ":\n";
    auto nodes = it->getNodes();
    reorderNodesOnTetrahedron(nodes);
    for (int nP = 0; nP < 4; ++nP) {
        auto p = nodes[nP];
        out << "\t\tP" << nP << ": (" << p.Coords()[0] <<", " << p.Coords()[1] << ", " << p.Coords()[2] << ")\n";
    }
    out << std::endl;
}

static inline void print_matrix_nan(std::ostream& out, std::vector<double>& A, Mesh::iteratorCell it, int nRows, int nan_i, int nan_j) {
    out<<"not a number in matrix"<<"\n";
    out<<"num tet " << it->DataLocalID()<< " A["<<nan_i<<"]["<<nan_j<<"] = " << A[nan_j*nRows +nan_i]<<"\n";
    for(int i1=0;i1<nRows;i1++) {
        for (int j1 = 0; j1 < nRows; j1++) {
            out << A[j1 * nRows + i1] << " ";
        }
        out<<"\n";
    }

    auto nodes = it->getNodes();
    reorderNodesOnTetrahedron(nodes);
    for (int nP = 0; nP < 4; ++nP) {
        auto p = nodes[nP];
        out << "P" << nP << ": ("
            << p.Coords()[0] <<", " << p.Coords()[1] << ", " << p.Coords()[2] << ")\n";

    }
    out << std::endl;
}

///This function assembles matrix and rhs of FE problem previously setted by PrepareProblemGen(...) function
///input:
/// @param Label_Tag - tag for setting BC.
/// @param DDATA - pointer to double array of additional data passed to problem handler, may be NULL.
/// @param DDATA_size - available size of DDATA.
/// @param DATA - pointer to integer array of additional data passed to problem handler, may be NULL.
/// @param IDATA_size - available size of IDATA.
/// @param drp_val - if during computation of local matrix some of their elements will be less
///                than drp_val in absolute value then they will be set to zero and ignored in assembling global matrix.
///in-out:
/// @param matrix <- matrix + assembled_matrix, where assembled_matrix - matrix assembled from the problem.
/// @param rhs <- rhs + assembled_rhs, where assembled_rhs - rhs assembled from the problem.
/// @return  0 if assembling successful else some error_code (number < 0)
///             ATTENTION: In case of unsuccessful completion of this function matrix and rhs has wrong elements!!!:
///         -1 if some local matrix or rhs had NaN.
///         -2 if some element is not tetrahedral or has broken component connectivity cell - faces - edges - nodes
int Assembler::Assemble (Sparse::Matrix &matrix, Sparse::Vector &rhs, void* user_data, double drp_val){
    reset_timers();
    int nRows = 4 * m_enum->NumDof[0] + 6 * m_enum->NumDof[1] + 4 * m_enum->NumDof[2] + m_enum->NumDof[3];
    std::vector<double>& A = m_w.m_A, &F = m_w.m_F;
    std::vector<int> &indexesC = m_w.m_indexesR, &indexesR = m_w.m_indexesC;

    int local_edge_index[6];
    int local_face_index[4];
    if (!mat_rhs_func && mat_func && rhs_func) generate_mat_rhs_func();
    assert(mat_rhs_func && "mat_rhs_func is not specified");
    ElemMatEval::Memory fmem{m_w.m_iw.data(), m_w.m_w.data(), const_cast<const ElemMatEval::Real **>(m_w.m_args.data()), m_w.m_res.data(), user_data};
    ElementalAssembler dat(mat_rhs_func.get(), ElementalAssembler::MAT_RHS,
                           ElementalAssembler::SparsedDat{m_w.m_A.data(), m_w.colindA.data(), m_w.rowA.data(),
                                                          mat_rhs_func->out_size1(0),
                                                          mat_rhs_func->out_size2(0),
                                                          mat_rhs_func->out_nnz(0)},
                           ElementalAssembler::SparsedDat{m_w.m_F.data(), m_w.colindF.data(), m_w.rowF.data(),
                                                          mat_rhs_func->out_size1(1),
                                                          mat_rhs_func->out_size2(1),
                                                          mat_rhs_func->out_nnz(1)},
                           fmem, &m_w.m_Ab, &m_helper, m_mesh,
                           nullptr, nullptr, nullptr,m_mesh->BeginCell(),
                           indexesC.data(), indexesR.data(), local_edge_index, local_face_index
#ifndef NO_ASSEMBLER_TIMERS
                           , getTimeMessures()
#endif
    );

    rhs.SetInterval(getBegInd(), getEndInd());
    matrix.SetInterval(getBegInd(),getEndInd());
    for(int i=getBegInd();i<getEndInd();i++) {
        if (matrix[i].get_safe(i) == 0.0)
            matrix[i][i] = 0.0;
    }
    ElementArray<Node> nodes; ElementArray<Edge> edges; ElementArray<Face> faces;
    dat.nodes = &nodes, dat.edges = &edges, dat.faces = &faces;
    bool flag_control_edge = true, flag_control_face = true;
#ifndef NO_ASSEMBLER_TIMERS
    m_time_init_assemble_dat += m_timer.elapsed_and_reset();
#endif

    for (Mesh::iteratorCell it = m_mesh->BeginCell(); it != m_mesh->EndCell(); ++it) {
        nodes = it->getNodes();
        reorderNodesOnTetrahedron(nodes);
        edges = it->getEdges();
        faces = it->getFaces();
        dat.cell = it;
        flag_control_edge = find_local_edge_index(nodes, edges, local_edge_index);
        assert(flag_control_edge && "didnt find edge ");
        flag_control_face = find_local_face_index(nodes, faces, local_face_index);
        assert(flag_control_face && "didnt find face ");
        if (!flag_control_edge || !flag_control_face) return -2;
#ifndef NO_ASSEMBLER_TIMERS
        m_time_init_assemble_dat += m_timer.elapsed_and_reset();
#endif

        bool has_active = fill_assemble_templates(nRows, nodes, edges, faces, it,
                                     indexesC, indexesR, local_edge_index, local_face_index);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_fill_map_template += m_timer.elapsed_and_reset();
#endif
        if (!has_active) continue;

        initial_value_setter(dat);
        dat.update(nodes);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_init_val_setter += m_timer.elapsed_and_reset();
#endif
        m_prob_handler(dat);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_proc_user_handler += m_timer.elapsed_and_reset();
#endif

        for(int i=0;i<nRows;i++){
            if(indexesR[i] != -1){//cols
                if((indexesR[i]<getBegInd()) ||(indexesR[i]>=getEndInd())){
                    std::cout<<"wrong index C "<<indexesR[i]<<" "<<getBegInd()<<" "<<getEndInd()<<" "<<i<<std::endl;
                    abort();
                }
                for(int j=0;j<nRows;j++){ //rows
                    {

                        if((indexesC[i]<0) ||(indexesC[i]>m_enum->MatrSize)){
                            print_assemble_matrix_indeces_incompatible(std::cout, m_mesh,
                                                                       indexesC, m_enum->NumElem, m_enum->MinElem,
                                                                       m_enum->MaxElem, m_enum->MatrSize, m_enum->BegInd, m_enum->EndInd, nRows, i);
                            abort();
                        }

                        if(m_w.m_Ab[j*nRows +i] && fabs(A[j*nRows +i]) > drp_val) {
                            matrix[indexesR[i]][indexesC[j]] += A[j * nRows + i];
                        }

                        if(m_w.m_Ab[j*nRows +i] && !std::isfinite(A[j*nRows +i])){
//                            print_matrix_nan(std::cout, A, it, nRows, i, j);
                            return -1;
                        }
                    }
                }
                rhs[indexesR[i]] +=F[i];
                if (!std::isfinite(F[i])) {
//                    print_rhs_nan(std::cout, F, it, nRows, i);
                    return -1;
                }
            }

        }
#ifndef NO_ASSEMBLER_TIMERS
        m_time_set_elemental_res += m_timer.elapsed_and_reset();
#endif
    }
#ifndef NO_ASSEMBLER_TIMERS
    m_timer_total = m_timer_ttl.elapsed();
#endif

    return 0;
}

int Assembler::AssembleMatrix (Sparse::Matrix &matrix, void* user_data, double drp_val){
    reset_timers();
    int nRows = 4 * m_enum->NumDof[0] + 6 * m_enum->NumDof[1] + 4 * m_enum->NumDof[2] + m_enum->NumDof[3];
    std::vector<double>& A = m_w.m_A;
    std::vector<int> &indexesC = m_w.m_indexesR, &indexesR = m_w.m_indexesC;

    int local_edge_index[6];
    int local_face_index[4];
    if (!mat_func && mat_rhs_func) generate_mat_func();
    assert(mat_func && "mat_func is not specified");
    ElemMatEval::Memory fmem{m_w.m_iw.data(), m_w.m_w.data(), const_cast<const ElemMatEval::Real **>(m_w.m_args.data()), m_w.m_res.data(), user_data};
    ElementalAssembler dat(mat_func.get(), ElementalAssembler::MAT,
                           ElementalAssembler::SparsedDat{m_w.m_A.data(), m_w.colindA.data(), m_w.rowA.data(),
                                                          mat_func->out_size1(0),
                                                          mat_func->out_size2(0),
                                                          mat_func->out_nnz(0)},
                           ElementalAssembler::SparsedDat{nullptr, nullptr, nullptr, 0, 0, 0},
                           fmem, &m_w.m_Ab, &m_helper, m_mesh,
                           nullptr, nullptr, nullptr,m_mesh->BeginCell(),
                           indexesC.data(), indexesR.data(), local_edge_index, local_face_index
#ifndef NO_ASSEMBLER_TIMERS
                           , getTimeMessures()
#endif
    );

    matrix.SetInterval(getBegInd(),getEndInd());
    for(int i=getBegInd();i<getEndInd();i++) {
        if (matrix[i].get_safe(i) == 0.0)
            matrix[i][i] = 0.0;
    }
    ElementArray<Node> nodes; ElementArray<Edge> edges; ElementArray<Face> faces;
    dat.nodes = &nodes, dat.edges = &edges, dat.faces = &faces;
    bool flag_control_edge = true, flag_control_face = true;
#ifndef NO_ASSEMBLER_TIMERS
    m_time_init_assemble_dat += m_timer.elapsed_and_reset();
#endif

    for (Mesh::iteratorCell it = m_mesh->BeginCell(); it != m_mesh->EndCell(); ++it) {
        nodes = it->getNodes();
        reorderNodesOnTetrahedron(nodes);
        edges = it->getEdges();
        faces = it->getFaces();
        dat.cell = it;
        flag_control_edge = find_local_edge_index(nodes, edges, local_edge_index);
        assert(flag_control_edge && "didnt find edge ");
        flag_control_face = find_local_face_index(nodes, faces, local_face_index);
        assert(flag_control_face && "didnt find face ");
        if (!flag_control_edge || !flag_control_face) return -2;
#ifndef NO_ASSEMBLER_TIMERS
        m_time_init_assemble_dat += m_timer.elapsed_and_reset();
#endif

        bool has_active = fill_assemble_templates(nRows, nodes, edges, faces, it,
                                indexesC, indexesR, local_edge_index, local_face_index);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_fill_map_template += m_timer.elapsed_and_reset();
#endif
        if (!has_active) continue;

        initial_value_setter(dat);
        dat.update(nodes);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_init_val_setter += m_timer.elapsed_and_reset();
#endif
        m_prob_handler(dat);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_proc_user_handler += m_timer.elapsed_and_reset();
#endif

        for(int i=0;i<nRows;i++){
            if(indexesR[i] != -1){//cols
                if((indexesR[i]<getBegInd()) ||(indexesR[i]>=getEndInd())){
                    std::cout<<"wrong index C "<<indexesR[i]<<" "<<getBegInd()<<" "<<getEndInd()<<" "<<i<<std::endl;
                    abort();
                }
                for(int j=0;j<nRows;j++){ //rows
                    {

                        if((indexesC[i]<0) ||(indexesC[i]>m_enum->MatrSize)){
                            print_assemble_matrix_indeces_incompatible(std::cout, m_mesh,
                                                                       indexesC, m_enum->NumElem, m_enum->MinElem,
                                                                       m_enum->MaxElem, m_enum->MatrSize, m_enum->BegInd, m_enum->EndInd, nRows, i);
                            abort();
                        }

                        if(m_w.m_Ab[j*nRows +i] && fabs(A[j*nRows +i]) > drp_val) {
                            matrix[indexesR[i]][indexesC[j]] += A[j * nRows + i];
                        }

                        if(m_w.m_Ab[j*nRows +i] && !std::isfinite(A[j*nRows +i])){
//                            print_matrix_nan(std::cout, A, it, nRows, i, j);
                            return -1;
                        }
                    }
                }
            }

        }
#ifndef NO_ASSEMBLER_TIMERS
        m_time_set_elemental_res += m_timer.elapsed_and_reset();
#endif
    }
#ifndef NO_ASSEMBLER_TIMERS
    m_timer_total = m_timer_ttl.elapsed();
#endif

    return 0;
}

int Assembler::AssembleRHS (Sparse::Vector &rhs, void* user_data, double drp_val){
    (void) drp_val;
    reset_timers();
    int nRows = 4 * m_enum->NumDof[0] + 6 * m_enum->NumDof[1] + 4 * m_enum->NumDof[2] + m_enum->NumDof[3];
    std::vector<double> &F = m_w.m_F;
    std::vector<int> &indexesC = m_w.m_indexesR, &indexesR = m_w.m_indexesC;

    int local_edge_index[6];
    int local_face_index[4];
    if (!rhs_func && mat_rhs_func) {
        std::cerr << "WARNING: rhs_func is not specified, so it will be generated from mat_rhs_func func" << std::endl;
        generate_rhs_func();
    }
    assert(rhs_func && "rhs_func is not specified");
    ElemMatEval::Memory fmem{m_w.m_iw.data(), m_w.m_w.data(), const_cast<const ElemMatEval::Real **>(m_w.m_args.data()), m_w.m_res.data(), user_data};
    ElementalAssembler dat(rhs_func.get(), ElementalAssembler::RHS,
                           ElementalAssembler::SparsedDat{nullptr, nullptr, nullptr, 0, 0, 0},
                           ElementalAssembler::SparsedDat{m_w.m_F.data(), m_w.colindF.data(), m_w.rowF.data(),
                                                          rhs_func->out_size1(0),
                                                          rhs_func->out_size2(0),
                                                          rhs_func->out_nnz(0)},
                           fmem, &m_w.m_Ab, &m_helper, m_mesh,
                           nullptr, nullptr, nullptr,m_mesh->BeginCell(),
                           indexesC.data(), indexesR.data(), local_edge_index, local_face_index
#ifndef NO_ASSEMBLER_TIMERS
                           , getTimeMessures()
#endif
    );

    rhs.SetInterval(getBegInd(), getEndInd());
    ElementArray<Node> nodes; ElementArray<Edge> edges; ElementArray<Face> faces;
    dat.nodes = &nodes, dat.edges = &edges, dat.faces = &faces;
    bool flag_control_edge = true, flag_control_face = true;
#ifndef NO_ASSEMBLER_TIMERS
    m_time_init_assemble_dat += m_timer.elapsed_and_reset();
#endif

    for (Mesh::iteratorCell it = m_mesh->BeginCell(); it != m_mesh->EndCell(); ++it) {
        nodes = it->getNodes();
        reorderNodesOnTetrahedron(nodes);
        edges = it->getEdges();
        faces = it->getFaces();
        dat.cell = it;
        flag_control_edge = find_local_edge_index(nodes, edges, local_edge_index);
        assert(flag_control_edge && "didnt find edge ");
        flag_control_face = find_local_face_index(nodes, faces, local_face_index);
        assert(flag_control_face && "didnt find face ");
        if (!flag_control_edge || !flag_control_face) return -2;
#ifndef NO_ASSEMBLER_TIMERS
        m_time_init_assemble_dat += m_timer.elapsed_and_reset();
#endif

        bool has_active = fill_assemble_templates(nRows, nodes, edges, faces, it,
                                indexesC, indexesR, local_edge_index, local_face_index);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_fill_map_template += m_timer.elapsed_and_reset();
#endif
        if (!has_active) continue;

        initial_value_setter(dat);
        dat.update(nodes);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_init_val_setter += m_timer.elapsed_and_reset();
#endif
        m_prob_handler(dat);
#ifndef NO_ASSEMBLER_TIMERS
        m_time_proc_user_handler += m_timer.elapsed_and_reset();
#endif

        for(int i=0;i<nRows;i++){
            if(indexesR[i] != -1){//cols
                if((indexesR[i]<getBegInd()) ||(indexesR[i]>=getEndInd())){
                    std::cout<<"wrong index C "<<indexesR[i]<<" "<<getBegInd()<<" "<<getEndInd()<<" "<<i<<std::endl;
                    abort();
                }
                rhs[indexesR[i]] +=F[i];
                if (!std::isfinite(F[i])) {
//                    print_rhs_nan(std::cout, F, it, nRows, i);
                    return -1;
                }
            }

        }
#ifndef NO_ASSEMBLER_TIMERS
        m_time_set_elemental_res += m_timer.elapsed_and_reset();
#endif
    }
#ifndef NO_ASSEMBLER_TIMERS
    m_timer_total = m_timer_ttl.elapsed();
#endif

    return 0;
}

void Assembler::Clear() {
    m_w.Clear();
    m_helper.Clear();
    orderC.clear();
    orderR.clear();
    mat_func = nullptr;
    rhs_func = nullptr;
    mat_rhs_func = nullptr;
    m_info.Clear();
    m_prob_handler = nullptr;
    initial_value_setter = nullptr;
    m_enum.reset();
    m_mesh = nullptr;
}

#ifndef NO_ASSEMBLER_TIMERS
#define TAKE_TIMER(X) X
#else
#define TAKE_TIMER(X) -1
#endif

double Assembler::GetTimeInitAssembleData() { return TAKE_TIMER(m_time_init_assemble_dat); }
double Assembler::GetTimeFillMapTemplate() { return TAKE_TIMER(m_time_fill_map_template); }
double Assembler::GetTimeInitValSet() { return TAKE_TIMER(m_time_init_val_setter); }
double Assembler::GetTimeInitUserHandler() { return TAKE_TIMER(m_time_init_user_handler); }
double Assembler::GetTimeEvalLocFunc() { return TAKE_TIMER(m_time_comp_func); }
double Assembler::GetTimePostProcUserHandler() { return TAKE_TIMER(m_time_proc_user_handler); }
double Assembler::GetTimeFillGlobalStructs() { return TAKE_TIMER(m_time_set_elemental_res); }
double Assembler::GetTimeTotal() { return TAKE_TIMER(m_timer_total); }

#undef TAKE_TIMER
