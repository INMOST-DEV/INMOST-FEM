#include "elemental_assembler.h"
namespace Ani{
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(const INMOST::Tag& from, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp){
        Ani::GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(from, p.vars->descr->trial_funcs, *(p.cell), *(p.faces), *(p.edges), *(p.nodes), p.node_permutation, out, component, ncomp);
    }  
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(const INMOST::Tag* var_tags, const std::size_t nvar_tags, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp){
        Ani::GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags, nvar_tags, p.vars->descr->trial_funcs, *(p.cell), *(p.faces), *(p.edges), *(p.nodes), p.node_permutation, out, component, ncomp);
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(const std::vector<INMOST::Tag>& var_tags, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp){
        GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags.data(), var_tags.size(), p, out, component, ncomp);
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(std::initializer_list<INMOST::Tag> var_tags, const ElementalAssembler& p, RandomIt out, const int* component/*[ncomp]*/, unsigned int ncomp){
        GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags.begin(), var_tags.size(), p, out, component, ncomp);
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(const INMOST::Tag& from, const ElementalAssembler& p, RandomIt out, std::initializer_list<int> components) {
        GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(from, p, out, components.begin(), components.size());
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(std::vector<INMOST::Tag>& var_tags, const ElementalAssembler& p, RandomIt out, std::initializer_list<int> components){
        GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags, p, out, components.begin(), components.size());
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(std::initializer_list<INMOST::Tag> var_tags, const ElementalAssembler& p, RandomIt out, std::initializer_list<int> components){
        GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags, p, out, components.begin(), components.size());
    }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(const INMOST::Tag& from, const ElementalAssembler& p, RandomIt out, unsigned int physVar){ GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(from, p, out, {int(physVar)}); }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(std::vector<INMOST::Tag>& var_tags, const ElementalAssembler& p, RandomIt out, unsigned int physVar) { GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags, p, out, {int(physVar)}); }
    template<bool OnlyIfDataAvailable, class RandomIt>
    void ElementalAssembler::GatherDataOnElement(std::initializer_list<INMOST::Tag> var_tags, const ElementalAssembler& p, RandomIt out, unsigned int physVar) { GatherDataOnElement<OnlyIfDataAvailable, RandomIt>(var_tags, p, out, {int(physVar)}); }
}