//
// Created by Liogky Alexey on 10.01.2024.
//

#ifndef ANIFEM_CUCHY_STRAIN_INVARIANTS_INL
#define ANIFEM_CUCHY_STRAIN_INVARIANTS_INL

#include "cauchy_strain_invariants.h"

namespace Ani{ namespace Mech{

template<typename FT>
inline SymMtx3D<FT> F_to_E(const Mtx3D<FT>& F){ //E = (F^T*F - I)/2
    SymMtx3D<FT> E;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = i+1; j < 3; ++j)
            E(i,j) = (F(0,i)*F(0,j) + F(1,i)*F(1,j) + F(2,i)*F(2,j)) / 2;
    for (unsigned i = 0; i < 3; ++i)
        E(i,i) = (F(0,i)*F(0,i) + F(1,i)*F(1,i) + F(2,i)*F(2,i) - 1) / 2;
    return E;           
}
template<typename FT>
inline SymMtx3D<FT> grU_to_E(const Mtx3D<FT>& grU){ //E = (F^T*F - I)/2
    SymMtx3D<FT> E;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = i; j < 3; ++j)
            E(i, j) = (grU(i, j) + grU(j, i) + grU(0,i)*grU(0,j) + grU(1,i)*grU(1,j) + grU(2,i)*grU(2,j))/2;
    return E;      
}
template<typename FT>
inline Sym4Tensor3D<FT> dS_to_dP(const Mtx3D<FT>& grU, const SymMtx3D<FT>& S, const BiSym4Tensor3D<FT>& dS){
    // //D_ijkl = I_ik S_lj + F_in F_km dS_njml
    Mtx3D<FT> F = Mtx3D<FT>::Identity() + grU;
    Sym4Tensor3D<FT> D;
    for (unsigned j = 0; j < 3; ++j)
    for (unsigned l = j; l < 3; ++l){
        auto v = S(l, j);
        for (unsigned i = 0; i < 3; ++i){
            unsigned k = i;
            D(i, j, k, l) = v;
        }
        // unsigned id = l + 3*(k + 3 * (j + 3*i));
        // D_out[id] = S(l, j);
    }
    for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = 0; j < 3; ++j){
        unsigned k = i, l = j;
        D(i, j, k, l) += F(i, 0)*F(i, 0)*dS(0, j, 0, j) + F(i, 1)*F(i, 1)*dS(1, j, 1, j) + F(i, 2)*F(i, 2)*dS(2, j, 2, j)
                    +2*(F(i, 0)*F(i, 1)*dS(0, j, 1, j)+F(i, 0)*F(i, 2)*dS(0, j, 2, j) + F(i, 1)*F(i, 2)*dS(1, j, 2, j));
        // unsigned id = l + 3*(k + 3 * (j + 3*i));
        // D_out[id] += F(i, 0)*F(i, 0)*dS(0, j, 0, j) + F(i, 1)*F(i, 1)*dS(1, j, 1, j) + F(i, 2)*F(i, 2)*dS(2, j, 2, j)
        //             +2*(F(i, 0)*F(i, 1)*dS(0, j, 1, j)+F(i, 0)*F(i, 2)*dS(0, j, 2, j) + F(i, 1)*F(i, 2)*dS(1, j, 2, j));
    }
    for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = 0; j < 3; ++j)
    for (unsigned kl = i+3*j+1; kl < 9; ++kl){
        unsigned k = kl%3, l = kl/3;
        FT s = FT(0);
        for (unsigned n = 0; n < 3; ++n)
        for (unsigned m = 0; m < 3; ++m)
            s += dS(n, j, m, l) * F(i, n) * F(k, m);
        D(i, j, k, l) += s; 
        // unsigned id = l + 3*(k + 3 * (j + 3*i));   
        // D_out[id] += s;
        // unsigned id1 = j + 3*(i + 3 * (l + 3*k));
        // D_out[id1] += s;
    }

    return D;
}

template<typename FT>
inline BiSym4Tensor3D<FT> C_I2_ddE() { 
    BiSym4Tensor3D<FT> r; //c * (I_ij I_kl - (I_ik I_jl + I_il I_jk) / 2)
    const FT c = FT(4); // derivation factor

    r(0, 0, 1, 1) = c; r(0, 1, 0, 1) = -c/2;
    r(0, 0, 2, 2) = c; r(0, 2, 0, 2) = -c/2;
    r(1, 1, 2, 2) = c; r(1, 2, 1, 2) = -c/2;
    
    return r;    
}
template<typename FT>
inline BiSym4Tensor3D<FT> C_I3_ddE(const SymMtx3D<FT>& E) {
    FT trE = E.Trace();
    const FT c = FT(4); // derivation factor
    BiSym4Tensor3D<FT> r;
    
    r(0, 0, 1, 1) = ( (1 + 2*trE) - 2 * (E(1, 1) + E(0, 0)))*c;
    r(0, 0, 2, 2) = ( (1 + 2*trE) - 2 * (E(2, 2) + E(0, 0)))*c;
    r(1, 1, 2, 2) = ( (1 + 2*trE) - 2 * (E(2, 2) + E(1, 1)))*c;
    r(0, 1, 0, 1) = (-(1 + 2*trE) / 2 + (E(1, 1) + E(0, 0)))*c;
    r(0, 2, 0, 2) = (-(1 + 2*trE) / 2 + (E(2, 2) + E(0, 0)))*c;
    r(1, 2, 1, 2) = (-(1 + 2*trE) / 2 + (E(2, 2) + E(1, 1)))*c;
    r(0, 1, 1, 2) = (                   E(0, 2)            )*c;
    r(0, 1, 0, 2) = (                   E(1, 2)            )*c;
    r(0, 2, 1, 2) = (                   E(0, 1)            )*c;
    r(0, 0, 1, 2) = (              -2 * E(1, 2)            )*c;
    r(0, 2, 1, 1) = (              -2 * E(0, 2)            )*c;
    r(0, 1, 2, 2) = (              -2 * E(0, 1)            )*c;
    
    return r;
}
template<typename FT>
inline BiSym4Tensor3D<FT> C_J_ddE(const SymMtx3D<FT>& E) {
    FT Jv = C_J(E);
    SymMtx3D<FT> t = (SymMtx3D<FT>::Identity()+2*E).Adj();
    return C_I3_ddE(E)/(2*Jv) - BiSym4Tensor3D<FT>::TensorSquare(t) / (Jv*Jv*Jv);
}
template<typename FT>
inline SymMtx3D<FT> C_I8fs_dE(std::array<FT, 3> f, std::array<FT, 3> s) { 
    SymMtx3D<FT> r;
    for (int i = 0; i < 3; ++i)
        r(i, i) = 2*f[i]*s[i];
    for (int i = 0; i < 3; ++i)
    for (int j = i+1; j < 3; ++j)
        r(i, j) = f[i]*s[j] + f[j]*s[i];
    return r;    
}

}}

#endif //ANIFEM_CUCHY_STRAIN_INVARIANTS_H