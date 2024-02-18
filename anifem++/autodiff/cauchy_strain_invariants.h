//
// Created by Liogky Alexey on 10.01.2024.
//

#include "physical_tensors.h"
#include <cmath>

#ifndef ANIFEM_CUCHY_STRAIN_INVARIANTS_H
#define ANIFEM_CUCHY_STRAIN_INVARIANTS_H

namespace Ani{
    template<typename FT = double> using Mtx3D = PhysMtx<3, FT>;
    template<typename FT = double> using SymMtx3D = SymMtx<3, FT>;
    template<typename FT = double> using Sym4Tensor3D = SymTensor4Rank<3, FT>;
    template<typename FT = double> using BiSym4Tensor3D = BiSymTensor4Rank<3, FT>;

    namespace Mech{
        /// Construct E(i, j) from F(i, j) =  delta_ij + nabla_j u_i 
        template<typename FT = double>
        inline SymMtx3D<FT> F_to_E(const Mtx3D<FT>& F);

        /// Construct E(i, j) from grU(i, j) = nabla_j u_i
        template<typename FT = double>
        inline SymMtx3D<FT> grU_to_E(const Mtx3D<FT>& grU);

        /// Construct P_{ij} = F_{ik} * S_{kj} 
        template<typename FT = double>
        inline Mtx3D<FT> S_to_P(const Mtx3D<FT>& grU, const SymMtx3D<FT>& S) { return S + grU * S;}
        
        /// Construct dP_{ijkl} = d^2 (W) / (dF_{ij} dF_{kl}) = I_ik S_lj + F_in F_km dS_njml
        template<typename FT = double>
        inline Sym4Tensor3D<FT> dS_to_dP(const Mtx3D<FT>& grU, const SymMtx3D<FT>& S, const BiSym4Tensor3D<FT>& dS);

        /// @return I1(C) = tr(C)
        template<typename FT = double>
        inline FT C_I1(const SymMtx3D<FT>& E) { return 3 + 2*E.Trace(); }
        /// @return S_1 = d (I1(C)) / d E_{ij} 
        template<typename FT = double>
        inline SymMtx3D<FT> C_I1_dE() { return SymMtx3D<FT>::Identity(FT(2)); }
        /// @return dS_1 = d^2 (I1(C)) / (d E_{ij} d E_{kl}) 
        template<typename FT = double>
        inline SymMtx3D<FT> C_I1_dE(const SymMtx3D<FT>& E) { (void) E; return C_I1_dE<FT>(); }
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I1_ddE() { return BiSym4Tensor3D<FT>(); }
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I1_ddE(const SymMtx3D<FT>& E) { (void) E; return C_I1_ddE(); }

        /// @return I2(C) = (tr(C)^2 - tr(C^2)) / 2
        template<typename FT = double>
        inline FT C_I2(const SymMtx3D<FT>& E) { FT trE = E.Trace(); return 3 + 4 * trE + 2 * trE*trE - 2*E.SquareFrobNorm(); }
        /// @return S_2 = d (I2(C)) / d E_{ij} 
        template<typename FT = double>
        inline SymMtx3D<FT> C_I2_dE(const SymMtx3D<FT>& E) { return 4*(SymMtx3D<FT>::Identity(FT(1) + E.Trace()) - E); }
        /// @return dS_2 = d^2 (I2(C)) / (d E_{ij} d E_{kl}) 
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I2_ddE();
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I2_ddE(const SymMtx3D<FT>& E) { (void) E; return C_I2_ddE(); }

        /// @return I3(C) = det(C)
        template<typename FT = double>
        inline FT C_I3(const SymMtx3D<FT>& E) { return (SymMtx3D<FT>::Identity()+2*E).Det(); }
        /// @return S_3 = d (I3(C)) / d E_{ij}
        template<typename FT = double>
        inline SymMtx3D<FT> C_I3_dE(const SymMtx3D<FT>& E) { return 2*(SymMtx3D<FT>::Identity()+2*E).Adj(); }
        /// @return dS_3 = d^2 (I3(C)) / (d E_{ij} d E_{kl}) 
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I3_ddE(const SymMtx3D<FT>& E);

        /// @return J(C) = sqrt(det(C)) = det(F)
        template<typename FT = double>
        inline FT C_J(const SymMtx3D<FT>& E) { return sqrt(C_I3(E)); }
        /// @return S_J = d (J(C)) / d E_{ij}
        template<typename FT = double>
        inline SymMtx3D<FT> C_J_dE(const SymMtx3D<FT>& E) { return (SymMtx3D<FT>::Identity()+2*E).Adj() / C_J(E); }
        /// @return dS_J = d^2 (J(C)) / (d E_{ij} d E_{kl}) 
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_J_ddE(const SymMtx3D<FT>& E);

        /// @return I4fs(C) = f^T*C*s
        template<typename FT = double>
        inline FT C_I4fs(const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s) { return (SymMtx3D<FT>::Identity()+2*E).Dot(PhysArr<3, FT>(f), PhysArr<3, FT>(s)); }
        /// @return S_4fs = d (I4fs(C)) / d E_{ij} 
        template<typename FT = double>
        inline SymMtx3D<FT> C_I4fs_dE(std::array<FT, 3> f, std::array<FT, 3> s);
        template<typename FT = double>
        inline SymMtx3D<FT> C_I4fs_dE(const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s) { (void) E; return C_I4fs_dE(f, s); }
        /// @return dS_4fs = d^2 (I4fs(C)) / (d E_{ij} d E_{kl}) 
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I4fs_ddE() { return BiSym4Tensor3D<FT>(); }
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I4fs_ddE(std::array<FT, 3> f, std::array<FT, 3> s) { (void) f, (void) s; return C_I4fs_ddE(); }
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I4fs_ddE(const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s) { (void) E, (void) f, (void) s; return C_I4fs_ddE(); }
    
        /// @return I5fs(C) = f^T*C^2*s
        template<typename FT = double>
        inline FT C_I5fs(const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s);
        /// @return S_5fs(C) = d (I5fs(C)) / d E_{ij}
        template<typename FT = double>
        inline SymMtx3D<FT> C_I5fs_dE(const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s);
        /// @return dS_5fs = d^2 (I5fs(C)) / (d E_{ij} d E_{kl}) 
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I5fs_ddE(std::array<FT, 3> f, std::array<FT, 3> s);
        template<typename FT = double>
        inline BiSym4Tensor3D<FT> C_I5fs_ddE(const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s){ (void) E; return C_I5fs_ddE(f, s); }
    }

}

#include "cauchy_strain_invariants.inl"

#endif //ANIFEM_CUCHY_STRAIN_INVARIANTS_H