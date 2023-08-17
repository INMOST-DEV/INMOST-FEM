//
// Created by Liogky Alexey on 01.03.2022.
//

#ifndef CARNUM_FEM_H
#define CARNUM_FEM_H

#include "anifem++/fem/operations/operations.h"
#include "anifem++/fem/spaces/spaces.h"
#include "assembler.h"

namespace Ani{

template<typename FEMTYPE> //[[deprecated("Use DofT::DofMap(Dof<FemType>::Map()) instead")]]
DofT::DofMap GenerateHelper(){ return DofT::DofMap( Dof<FEMTYPE>::Map() ); }

//[[deprecated("Use fs.m_order instead")]] 
inline DofT::DofMap GenerateHelper(const BaseFemSpace& fs) { return fs.m_order; }

}

#endif //CARNUM_FEM_H
