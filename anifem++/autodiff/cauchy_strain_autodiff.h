//
// Created by Liogky Alexey on 10.01.2024.
//

#include <cmath>

#include "cauchy_strain_invariants.h"
#include "autodiff.h"

#ifndef ANIFEM_CUCHY_STRAIN_AUTODIFF_H
#define ANIFEM_CUCHY_STRAIN_AUTODIFF_H

namespace Ani{
namespace Mech{

    /// Autodifferentiable variable for I1(C) = tr(C)
    template<typename FT = double>
    struct I1: public ADExpr{
        using State = ADState<true, true, false>;
        using Storage = PhysSymTensorStorageSet<3, FT>;
        using VT = typename Storage::ValueType;
        using GT = typename Storage::GradientType;
        using HT = typename Storage::HessianType;

        FT m_v = FT(0);
        unsigned char m_dif = -1;

        void Init(int numdif, const SymMtx3D<FT>& E){
            m_v = C_I1<FT>(E);
            m_dif = numdif;
        }

        I1() = default;
        I1(int numdif, const SymMtx3D<FT>& E){ Init(numdif, E); }

        FT operator()() const { return m_v; }
        SymMtx3D<FT> D() const { return C_I1_dE<FT>(); }
        BiSym4Tensor3D<FT> DD() const { return C_I1_ddE<FT>(); }
    };

    /// Autodifferentiable variable for I4fs(C) = f^T*C*s
    template<typename FT = double>
    struct I4fs: public ADExpr{
        using State = ADState<true, true, false>;
        using Storage = PhysSymTensorStorageSet<3, FT>;
        using VT = typename Storage::ValueType;
        using GT = typename Storage::GradientType;
        using HT = typename Storage::HessianType;

        SymMtx3D<FT> m_d;
        FT m_v = FT(0);
        unsigned char m_dif = -1;

        void Init(int numdif, const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s){
            m_v = C_I4fs(E, f, s);
            if (numdif >= 1) m_d = C_I4fs_dE(f, s);
            m_dif = numdif;
        }

        I4fs() = default;
        I4fs(int numdif, const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s){ Init(numdif, E, f, s); }

        FT operator()() const { return m_v; }
        SymMtx3D<FT> D() const { return m_d; }
        BiSym4Tensor3D<FT> DD() const { return C_I4fs_ddE<FT>(); }
    };

    /// Autodifferentiable variable for I2(C) = (tr(C)^2 - tr(C^2)) / 2
    template<typename FT = double>
    struct I2: public ADExpr{
        using State = ADState<true, true, true>;
        using Storage = PhysSymTensorStorageSet<3, FT>;
        using VT = typename Storage::ValueType;
        using GT = typename Storage::GradientType;
        using HT = typename Storage::HessianType;

        BiSym4Tensor3D<FT> m_dd;
        SymMtx3D<FT> m_d;
        FT m_v = FT(0);
        unsigned char m_dif = -1;

        void Init(int numdif, const SymMtx3D<FT>& E){
            m_v = C_I2(E);
            if (numdif >= 1) m_d = C_I2_dE(E);
            m_dif = numdif;
        }

        I2() = default;
        I2(int numdif, const SymMtx3D<FT>& E){ Init(numdif, E); }

        FT operator()() const { return m_v; }
        SymMtx3D<FT> D() const { return m_d; }
        BiSym4Tensor3D<FT> DD() const { return C_I2_ddE<FT>(); }
    };

    /// Autodifferentiable variable for I5fs(C) = f^T*C^2*s
    template<typename FT = double>
    struct I5fs: public ADExpr{
        using State = ADState<true, true, true>;
        using Storage = PhysSymTensorStorageSet<3, FT>;
        using VT = typename Storage::ValueType;
        using GT = typename Storage::GradientType;
        using HT = typename Storage::HessianType;

        BiSym4Tensor3D<FT> m_dd;
        SymMtx3D<FT> m_d;
        FT m_v = FT(0);
        unsigned char m_dif = -1;

        void Init(int numdif, const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s){
            m_v = C_I5fs(E, f, s);
            if (numdif >= 1) m_d = C_I5fs_dE(E, f, s);
            if (numdif >= 2) m_dd = C_I5fs_ddE(f, s);
            m_dif = numdif;
        }

        I5fs() = default;
        I5fs(int numdif, const SymMtx3D<FT>& E, std::array<FT, 3> f, std::array<FT, 3> s){ Init(numdif, E, f, s); }

        FT operator()() const { return m_v; }
        SymMtx3D<FT> D() const { return m_d; }
        BiSym4Tensor3D<FT> DD() const { return m_dd; }
    };

    /// Autodifferentiable variable for I3(C) = det(C)
    template<typename FT = double>
    struct I3: public ADExpr{
        using State = ADState<true, true, true>;
        using Storage = PhysSymTensorStorageSet<3, FT>;
        using VT = typename Storage::ValueType;
        using GT = typename Storage::GradientType;
        using HT = typename Storage::HessianType;

        BiSym4Tensor3D<FT> m_dd;
        SymMtx3D<FT> m_d;
        FT m_v = 0;
        unsigned char m_dif = -1;

        void Init(int numdif, const SymMtx3D<FT>& E){
            m_v = C_I3(E);
            if (numdif >= 1) m_d = C_I3_dE(E);
            if (numdif >= 2) m_dd = C_I3_ddE(E);
            m_dif = numdif;
        }

        I3() = default;
        I3(int numdif, const SymMtx3D<FT>& E){ Init(numdif, E); }

        FT operator()() const { return m_v; }
        SymMtx3D<FT> D() const { return m_d; }
        BiSym4Tensor3D<FT> DD() const { return m_dd; }
    };

    /// Autodifferentiable variable for J(C) = sqrt(det(C))
    template<typename FT = double>
    struct J: public ADExpr{
        using State = ADState<true, true, true>;
        using Storage = PhysSymTensorStorageSet<3, FT>;
        using VT = typename Storage::ValueType;
        using GT = typename Storage::GradientType;
        using HT = typename Storage::HessianType;

        BiSym4Tensor3D<FT> m_dd;
        SymMtx3D<FT> m_d;
        FT m_v = 0;
        unsigned char m_dif = -1;

        void Init(int numdif, const SymMtx3D<FT>& E){
            m_v = C_J(E);
            if (numdif >= 1) m_d = C_J_dE(E);
            if (numdif >= 2) m_dd = C_J_ddE(E);
            m_dif = numdif;
        }

        J() = default;
        J(int numdif, const SymMtx3D<FT>& E){ Init(numdif, E); }

        FT operator()() const { return m_v; }
        SymMtx3D<FT> D() const { return m_d; }
        BiSym4Tensor3D<FT> DD() const { return m_dd; }
    };

}
}

#endif  //AUTODIFF