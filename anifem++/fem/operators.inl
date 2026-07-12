namespace Ani{

    namespace FemComDetails {
        template<int DIM, typename FEMTYPE>
        struct FemVecTImpl<true, DIM, FEMTYPE> {
            using Dim = std::integral_constant<int, DIM>;
            using Base = FEMTYPE;
        };

        template<class T>
        struct CheckFemVar{};

        template<int OP>
        struct CheckFemVar<FemFix<OP>>: public std::integral_constant<bool, true> {};

        template<int DIM, int OP>
        struct CheckFemVar<FemVec<DIM, OP>>: public std::integral_constant<bool, true> {};

        template<int DIM, typename FEMTYPE>
        struct CheckFemVar<FemVecTImpl<true, DIM, FEMTYPE>>: public std::integral_constant<bool, true> {};

        template<class... Types>
        struct CheckFemVar<FemComImpl<true, Types...>>: public std::integral_constant<bool, true> {};

        template<std::size_t i>
        struct CheckFemVars<i> {
        };

        template<std::size_t i, class U, class... Types>
        struct CheckFemVars<i, U, Types...> : public CheckFemVars<i + 1, Types...> {
            using Val = std::integral_constant<bool, CheckFemVar<U>::value>;
        };

        template<bool isFem, class... Types>
        struct FemComImpl {
            static_assert(!isFem, "Some of types is not type of fem variable");
        };

        template<class... Types>
        struct FemComImpl<true, Types ...> : public std::tuple<Types ...> {
            using Base = std::tuple<Types ...>;
        };

        template<int SUM, int OP>
        struct DimCounter<SUM, OP>: public std::integral_constant<int, SUM>{};

        template<int SUM, int OP, class T, class... Types>
        struct DimCounter<SUM, OP, T, Types...>: public std::integral_constant<int, Operator<OP, T>::Dim::value + DimCounter<SUM, OP, Types...>::value>{};

        template<int SUM, int OP>
        struct OrderCounter<SUM, OP>: public std::integral_constant<int, SUM>{};

        template<int SUM, int OP, class T, class... Types>
        struct OrderCounter<SUM, OP, T, Types...>: public std::integral_constant<int, std::max(Operator<OP, T>::Order::value, DimCounter<SUM, OP, Types...>::value)>{};
        
        template<int SUM, int OP>
        struct NfaCounter<SUM, OP>: public std::integral_constant<int, SUM>{};

        template<int SUM, int OP, class T, class... Types>
        struct NfaCounter<SUM, OP, T, Types...>: public std::integral_constant<int, Operator<OP, T>::Nfa::value + NfaCounter<SUM, OP, Types...>::value>{};

        template<int SUM, int OP>
        struct NpartsCounter<SUM, OP>: public std::integral_constant<int, SUM>{};

        template<int SUM, int OP, class T, class... Types>
        struct NpartsCounter<SUM, OP, T, Types...>: public std::integral_constant<int,
                decltype(convToBendMx(Operator<OP, T>::template apply<double, int>(std::declval<AniMemory<double, int>& >(), std::declval<ArrayView<double>& >()), std::declval<int>(), std::declval<int>()))::Nparts::value
                + NpartsCounter<SUM, OP, Types...>::value>{};

        template<int SUM, int OP>
        struct UnionOrderCounter<SUM, OP>: public std::integral_constant<int, SUM>{};

        template<int SUM, int OP, class T, class... Types>
        struct UnionOrderCounter<SUM, OP, T, Types...>: public std::integral_constant<int,
                std::max(Operator<OP, T>::Order::value, UnionOrderCounter<SUM, OP, Types...>::value)>{};

        template<int OP, int RefDim, class... Types>
        struct AllUnionDimEqualHelper {
            static const bool value = true;
        };

        template<int OP, int RefDim, class Head, class... Tail>
        struct AllUnionDimEqualHelper<OP, RefDim, Head, Tail...> {
            static const bool value = (Operator<OP, Head>::Dim::value == RefDim)
                && AllUnionDimEqualHelper<OP, RefDim, Tail...>::value;
        };

        template<int OP, class... Types>
        struct AllUnionDimEqual {
            static const int RefDim = Operator<OP, std::tuple_element_t<0, std::tuple<Types...>>>::Dim::value;
            static const bool value = AllUnionDimEqualHelper<OP, RefDim, Types...>::value;
        };

        template<int MAX, class... Types>
        struct MaxIdenNfaCounter;

        template<int MAX>
        struct MaxIdenNfaCounter<MAX>: public std::integral_constant<int, MAX>{};

        template<int MAX, class T, class... Types>
        struct MaxIdenNfaCounter<MAX, T, Types...>: public std::integral_constant<int,
                MaxIdenNfaCounter<std::max(MAX, Operator<IDEN, T>::Nfa::value), Types...>::value>{};

        template<std::size_t I, class... Types>
        struct SubsetIdenNfaOffset {
            using Type = std::tuple_element_t<I, std::tuple<Types...>>;
            static const int Nfa = Operator<IDEN, Type>::Nfa::value;
            static const int Offset = I == 0 ? 0 : SubsetIdenNfaOffset<I - 1, Types...>::Offset + SubsetIdenNfaOffset<I - 1, Types...>::Nfa;
        };

        template<typename ScalarType, int DIM, int NFA>
        struct FillPhiFromDenseIdenApply {
            static void fill(const DenseMatrix<ScalarType>& res, ScalarType* phi) {
                for (int d = 0; d < DIM; ++d)
                    for (int i = 0; i < NFA; ++i)
                        phi[d + DIM * i] = res(d, i);
            }
        };

        template<int NPART, typename ScalarType, typename IndexType, int DIM, int NFA>
        struct FillPhiFromBandIdenApply {
            static void fill(const BandDenseMatrix<NPART, ScalarType, IndexType>& res, ScalarType* phi) {
                for (int p = 0; p < NPART; ++p)
                    for (int d = res.stRow[p]; d < res.stRow[p + 1]; ++d)
                        for (int i = res.stCol[p]; i < res.stCol[p + 1]; ++i)
                            phi[d + DIM * i] = res.data[p](d - res.stRow[p], i - res.stCol[p]);
            }
        };

        template<typename ApplyResult, typename ScalarType, int DIM, int NFA, bool IsDense>
        struct FillPhiFromIdenApplyDispatch;

        template<typename ApplyResult, typename ScalarType, int DIM, int NFA>
        struct FillPhiFromIdenApplyDispatch<ApplyResult, ScalarType, DIM, NFA, true> {
            static void fill(const ApplyResult& res, ScalarType* phi) {
                FillPhiFromDenseIdenApply<ScalarType, DIM, NFA>::fill(res, phi);
            }
        };

        template<typename ApplyResult, typename ScalarType, int DIM, int NFA>
        struct FillPhiFromIdenApplyDispatch<ApplyResult, ScalarType, DIM, NFA, false> {
            static void fill(const ApplyResult& res, ScalarType* phi) {
                FillPhiFromBandIdenApply<ApplyResult::Nparts::value, ScalarType, int, DIM, NFA>::fill(res, phi);
            }
        };

        template<typename FEMTYPE, typename ScalarType, typename IndexType>
        inline void evalUnionBasisFunctions(AniMemory<ScalarType, IndexType>& mem, const std::array<double, 3>& x, ScalarType* phi) {
            mem.XYG.Init(const_cast<double*>(x.data()), 3);
            mem.XYL[0] = ScalarType(1) - (x[0] + x[1] + x[2]);
            for (int k = 0; k < 3; ++k)
                mem.XYL[1 + k] = x[k];
            const int dim = Operator<IDEN, FEMTYPE>::Dim::value;
            const int nf = Operator<IDEN, FEMTYPE>::Nfa::value;
            std::fill(phi, phi + dim * nf, ScalarType(0));
            ArrayView<ScalarType> U(mem.U.data, mem.U.size);
            auto res = Operator<IDEN, FEMTYPE>::template apply<ScalarType, IndexType>(mem, U);
            FillPhiFromIdenApplyDispatch<decltype(res), ScalarType, dim, nf,
                std::is_same<decltype(res), DenseMatrix<ScalarType>>::value>::fill(res, phi);
            mem.XYG.Init(nullptr, 3);
        }

        template<std::size_t SI, std::size_t SJ, class... Types>
        struct OrthBFillOffDiag;

        template<std::size_t SI, std::size_t SJ, class... Types>
        struct OrthBFillOffDiagSkip {
            template<typename ScalarType, typename IndexType, uint FUSION>
            static void run(DenseMatrix<ScalarType>& B, int nfshi, int nfshj,
                            AniMemory<ScalarType, IndexType>& mem, const Tetra<const double>& XYZ,
                            std::array<ArrayView<ScalarType>, FUSION>& udofs, uint max_quad_order) {
                using TypeSi = std::tuple_element_t<SI, std::tuple<Types...>>;
                const int nfi = Operator<IDEN, TypeSi>::Nfa::value;
                OrthBFillOffDiag<SI, SJ + 1, Types...>::template run<ScalarType, IndexType, FUSION>(
                    B, nfshi, nfshj + nfi, mem, XYZ, udofs, max_quad_order);
            }
        };

        template<std::size_t SI, std::size_t SJ, class... Types>
        struct OrthBFillOffDiagFill {
            template<typename ScalarType, typename IndexType, uint FUSION>
            static void run(DenseMatrix<ScalarType>& B, int nfshi, int nfshj,
                            AniMemory<ScalarType, IndexType>& mem, const Tetra<const double>& XYZ,
                            std::array<ArrayView<ScalarType>, FUSION>& udofs, uint max_quad_order) {
                using TypeSi = std::tuple_element_t<SI, std::tuple<Types...>>;
                using TypeSj = std::tuple_element_t<SJ, std::tuple<Types...>>;
                const int nfi = Operator<IDEN, TypeSi>::Nfa::value;
                const int nfj = Operator<IDEN, TypeSj>::Nfa::value;
                for (uint k = 0; k < static_cast<uint>(nfi); ++k) {
                    Dof<TypeSi>::template interpolate<FUSION>(
                        XYZ,
                        [&mem](const std::array<double, 3>& X, ScalarType* res, uint dim, void* /*user_data*/)->int {
                            (void) dim;
                            evalUnionBasisFunctions<TypeSj, ScalarType, IndexType>(mem, X, res);
                            return 0;
                        },
                        udofs, static_cast<int>(k), nullptr, max_quad_order);
                    for (uint l = 0; l < static_cast<uint>(nfj); ++l)
                        B(nfshi + k, nfshj + l) = udofs[l][k];
                }
                OrthBFillOffDiag<SI, SJ + 1, Types...>::template run<ScalarType, IndexType, FUSION>(
                    B, nfshi, nfshj + nfj, mem, XYZ, udofs, max_quad_order);
            }
        };

        template<std::size_t SI, std::size_t SJ, class... Types>
        struct OrthBFillOffDiag {
            template<typename ScalarType, typename IndexType, uint FUSION>
            static void run(DenseMatrix<ScalarType>& B, int nfshi, int nfshj,
                            AniMemory<ScalarType, IndexType>& mem, const Tetra<const double>& XYZ,
                            std::array<ArrayView<ScalarType>, FUSION>& udofs, uint max_quad_order) {
                OrthBFillOffDiagDispatch<SI, SJ, (SJ < sizeof...(Types)), (SJ == SI), Types...>::template run<ScalarType, IndexType, FUSION>(
                    B, nfshi, nfshj, mem, XYZ, udofs, max_quad_order);
            }
        private:
            template<std::size_t SI_, std::size_t SJ_, bool Continue, bool Skip, class... Types_>
            struct OrthBFillOffDiagDispatch {
                template<typename ScalarType, typename IndexType, uint FUSION>
                static void run(DenseMatrix<ScalarType>&, int, int, AniMemory<ScalarType, IndexType>&,
                                const Tetra<const double>&, std::array<ArrayView<ScalarType>, FUSION>&, uint) {}
            };
            template<std::size_t SI_, std::size_t SJ_, class... Types_>
            struct OrthBFillOffDiagDispatch<SI_, SJ_, true, true, Types_...> {
                template<typename ScalarType, typename IndexType, uint FUSION>
                static void run(DenseMatrix<ScalarType>& B, int nfshi, int nfshj,
                                AniMemory<ScalarType, IndexType>& mem, const Tetra<const double>& XYZ,
                                std::array<ArrayView<ScalarType>, FUSION>& udofs, uint max_quad_order) {
                    OrthBFillOffDiagSkip<SI_, SJ_, Types_...>::template run<ScalarType, IndexType, FUSION>(
                        B, nfshi, nfshj, mem, XYZ, udofs, max_quad_order);
                }
            };
            template<std::size_t SI_, std::size_t SJ_, class... Types_>
            struct OrthBFillOffDiagDispatch<SI_, SJ_, true, false, Types_...> {
                template<typename ScalarType, typename IndexType, uint FUSION>
                static void run(DenseMatrix<ScalarType>& B, int nfshi, int nfshj,
                                AniMemory<ScalarType, IndexType>& mem, const Tetra<const double>& XYZ,
                                std::array<ArrayView<ScalarType>, FUSION>& udofs, uint max_quad_order) {
                    OrthBFillOffDiagFill<SI_, SJ_, Types_...>::template run<ScalarType, IndexType, FUSION>(
                        B, nfshi, nfshj, mem, XYZ, udofs, max_quad_order);
                }
            };
        };

        template<std::size_t SI, class... Types>
        struct OrthBFill {
            template<typename ScalarType, typename IndexType, uint FUSION>
            static int run(DenseMatrix<ScalarType>& B, int nfshi,
                           AniMemory<ScalarType, IndexType>& mem, const Tetra<const double>& XYZ,
                           std::array<ArrayView<ScalarType>, FUSION>& udofs, uint max_quad_order) {
                using TypeSi = std::tuple_element_t<SI, std::tuple<Types...>>;
                const int nfi = Operator<IDEN, TypeSi>::Nfa::value;
                for (int k = 0; k < nfi; ++k)
                    B(k + nfshi, k + nfshi) = ScalarType(1);
                OrthBFillOffDiag<SI, 0, Types...>::template run<ScalarType, IndexType, FUSION>(B, nfshi, 0, mem, XYZ, udofs, max_quad_order);
                return OrthBFill<SI + 1, Types...>::template run<ScalarType, IndexType, FUSION>(B, nfshi + nfi, mem, XYZ, udofs, max_quad_order);
            }
        };

        template<class... Types>
        struct OrthBFill<sizeof...(Types), Types...> {
            template<typename ScalarType, typename IndexType, uint FUSION>
            static int run(DenseMatrix<ScalarType>&, int nfshi, AniMemory<ScalarType, IndexType>&,
                           const Tetra<const double>&, std::array<ArrayView<ScalarType>, FUSION>&, uint) {
                return nfshi;
            }
        };

        template<std::size_t I, class... Types>
        struct UnionIdenMemReq {
            template<typename ScalarType, typename IndexType>
            static void Impl(std::size_t &reqUsz, std::size_t &reqExtraR) {
                using T = std::tuple_element_t<I, std::tuple<Types...>>;
                std::size_t Usz = 0, extraR = 0, extraI = 0;
                Operator<IDEN, T>::template memoryRequirements<ScalarType, IndexType>(1, 1, Usz, extraR, extraI);
                (void) extraI;
                if (reqUsz < Usz) reqUsz = Usz;
                if (reqExtraR < extraR) reqExtraR = extraR;
                UnionIdenMemReq<I + 1, Types...>::template Impl<ScalarType, IndexType>(reqUsz, reqExtraR);
            }
        };

        template<class... Types>
        struct UnionIdenMemReq<sizeof...(Types), Types...> {
            template<typename ScalarType, typename IndexType>
            static void Impl(std::size_t&, std::size_t&) {}
        };

        template<class T>
        struct UnionOrthCoefsStorageSingle {
            static const int NF = Operator<IDEN, T>::Nfa::value;

            static const std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)>& get() {
                static const std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)> eye = []() {
                    std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)> a{};
                    for (int i = 0; i < NF; ++i)
                        a[static_cast<std::size_t>(i) + static_cast<std::size_t>(NF) * static_cast<std::size_t>(i)] = 1.0;
                    return a;
                }();
                return eye;
            }
            static DenseMatrix<const double> asMatrix() {
                return DenseMatrix<const double>(get().data(), NF, NF);
            }
        };

        template<class T1, class T2, class... Rest>
        struct UnionOrthCoefsStorage {
            static const int NF = NfaCounter<0, IDEN, T1, T2, Rest...>::value;
            static const int MaxNfa = MaxIdenNfaCounter<0, T1, T2, Rest...>::value;

            static const std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)>& get() {
                static const std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)> coefs = compute();
                return coefs;
            }

            static DenseMatrix<const double> asMatrix() {
                return DenseMatrix<const double>(get().data(), NF, NF);
            }

        private:
            static std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)> compute(uint max_quad_order = 5) {
                (void) max_quad_order;
                std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)> orth{};
                std::array<double, static_cast<std::size_t>(NF) * static_cast<std::size_t>(NF)> Bd{};
                DenseMatrix<double> B(Bd.data(), NF, NF);
                B.SetZero();

                std::size_t reqUsz = 0, reqExtraR = 0;
                UnionIdenMemReq<0, T1, T2, Rest...>::template Impl<double, int>(reqUsz, reqExtraR);

                std::vector<double> dmem(reqUsz + reqExtraR);
                double XYZa[3 * 4]{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
                Tetra<const double> XYZ(XYZa + 0, XYZa + 3, XYZa + 6, XYZa + 9);
                AniMemory<double, int> mem;
                std::array<double, 4> xyl{1, 0, 0, 0};
                mem.XYP.Init(XYZa, 3 * 4);
                mem.PSI.Init(XYZa + 3, 3 * 3);
                mem.XYG.Init(nullptr, 3);
                mem.DET.Init(nullptr, 0);
                mem.XYL.Init(xyl.data(), 4);
                mem.WG.Init(nullptr, 0);
                mem.U.Init(dmem.data(), reqUsz);
                mem.extraR.Init(dmem.data() + reqUsz, reqExtraR);
                mem.q = 1;
                mem.f = 1;

                std::array<ArrayView<double>, static_cast<std::size_t>(MaxNfa)> udofs;
                for (uint i = 0; i < static_cast<uint>(MaxNfa); ++i)
                    udofs[i] = ArrayView<double>(orth.data() + NF * i, NF);

                OrthBFill<0, T1, T2, Rest...>::template run<double, int, static_cast<uint>(MaxNfa)>(B, 0, mem, XYZ, udofs, max_quad_order);

                DenseMatrix<double> A(orth.data(), NF, NF);
                {
                    std::vector<double> wmem(2 * NF * NF);
                    std::vector<int> imem(2 * NF);
                    fullPivLU_inverse(Bd.data(), A.data, NF, wmem.data(), imem.data());
                }

                {
                    double max_err = 0.0, max_g = 0.0;
                    for (int i = 0; i < NF; ++i)
                    for (int k = 0; k < NF; ++k){
                        double s = 0.0;
                        for (int j = 0; j < NF; ++j)
                            s += A(i, j) * B(j, k);
                        double target = (i == k) ? 1.0 : 0.0;
                        max_err = std::max(max_err, std::abs(s - target));
                        max_g = std::max(max_g, std::abs(B(i, k)));
                    }
                    const double tol = 1e-8;
                    if (max_err > tol * (1.0 + max_g))
                        throw std::runtime_error("FemUnionT: input bases or interpolating functionals are linearly dependent (Gram matrix is singular)");
                }

                for (int i = 0; i < NF; ++i) {
                    auto p = std::max_element(A.data + i * NF, A.data + (i + 1) * NF,
                        [](double a, double b) { return std::abs(a) < std::abs(b); });
                    double m = std::abs(*p);
                    std::transform(A.data + i * NF, A.data + (i + 1) * NF, A.data + i * NF,
                        [m](double a) { return (std::abs(a) > m * 1e-7) ? a : 0.0; });
                }
                return orth;
            }
        };

        template<std::size_t I, int OP, class... Types>
        struct UnionMemReq {
            template<typename ScalarType, typename IndexType>
            static void Impl(int f, int q, std::size_t &extraR, std::size_t &extraI) {
                using T = std::tuple_element_t<I, std::tuple<Types...>>;
                std::size_t Usz = 0, eR = 0, eI = 0;
                Operator<OP, T>::template memoryRequirements<ScalarType, IndexType>(f, q, Usz, eR, eI);
                if (extraR < eR + Usz) extraR = eR + Usz;
                if (extraI < eI) extraI = eI;
                UnionMemReq<I + 1, OP, Types...>::template Impl<ScalarType, IndexType>(f, q, extraR, extraI);
            }
        };

        template<int OP, class... Types>
        struct UnionMemReq<sizeof...(Types), OP, Types...> {
            template<typename ScalarType, typename IndexType>
            static void Impl(int, int, std::size_t&, std::size_t&) {}
        };

        template<std::size_t I, int OP, int GDIM, int NF, class... Types>
        struct UnionApplySubset {
            template<typename ScalarType, typename IndexType>
            static int Impl(AniMemory<ScalarType, IndexType>& mem, DenseMatrix<ScalarType>& lU, int nfa_shift) {
                using T = std::tuple_element_t<I, std::tuple<Types...>>;
                const int LNFA = Operator<IDEN, T>::Nfa::value;
                std::size_t Usz = 0, eR = 0, eI = 0;
                Operator<OP, T>::template memoryRequirements<ScalarType, IndexType>(mem.f, mem.q, Usz, eR, eI);
                assert(mem.extraR.size >= Usz && "Wrong size of allocated memory");
                mem.extraR.size -= Usz;
                ArrayView<ScalarType> Vm(mem.extraR.data + mem.extraR.size, Usz);
                auto Vx = convToBendMx(Operator<OP, T>::template apply<ScalarType, IndexType>(mem, Vm), Operator<OP, T>::Dim::value, Operator<OP, T>::Nfa::value);
                const int ANpart = decltype(Vx)::Nparts::value;
                for (int p = 0; p < ANpart; ++p) {
                    int nCol = Vx.stCol[p + 1] - Vx.stCol[p];
                    int nRow = Vx.stRow[p + 1] - Vx.stRow[p];
                    for (std::size_t r = 0; r < mem.f; ++r)
                        for (int i = 0; i < nCol; ++i)
                            for (std::size_t n = 0; n < mem.q; ++n)
                                for (int k = 0; k < nRow; ++k)
                                    lU(k + Vx.stRow[p] + GDIM * static_cast<int>(n), nfa_shift + i + Vx.stCol[p] + NF * static_cast<int>(r))
                                        = Vx.data[p](k + nRow * static_cast<int>(n), i + nCol * static_cast<int>(r));
                }
                mem.extraR.size += Usz;
                return UnionApplySubset<I + 1, OP, GDIM, NF, Types...>::template Impl<ScalarType, IndexType>(mem, lU, nfa_shift + LNFA);
            }
        };

        template<int OP, int GDIM, int NF, class... Types>
        struct UnionApplySubset<sizeof...(Types), OP, GDIM, NF, Types...> {
            template<typename ScalarType, typename IndexType>
            static int Impl(AniMemory<ScalarType, IndexType>&, DenseMatrix<ScalarType>&, int nfa_shift) {
                return nfa_shift;
            }
        };

        template<int OP, class T>
        struct UnionOperatorApply<OP, T> {
            template<typename ScalarType, typename IndexType>
            static DenseMatrix<ScalarType> Impl(AniMemory<ScalarType, IndexType>& mem, ArrayView<ScalarType>& U) {
                return Operator<OP, T>::template apply<ScalarType, IndexType>(mem, U);
            }
        };

        template<int OP, class T1, class T2, class... Rest>
        struct UnionOperatorApply<OP, T1, T2, Rest...> {
            template<typename ScalarType, typename IndexType>
            static DenseMatrix<ScalarType> Impl(AniMemory<ScalarType, IndexType>& mem, ArrayView<ScalarType>& udat) {
                const int NF = NfaCounter<0, OP, T1, T2, Rest...>::value;
                const int GDIM = Operator<OP, T1>::Dim::value;

                DenseMatrix<ScalarType> lU(udat.data, mem.q * GDIM, mem.f * NF, udat.size);
                lU.SetZero();

                UnionApplySubset<0, OP, GDIM, NF, T1, T2, Rest...>::template Impl<ScalarType, IndexType>(mem, lU, 0);

                return lU;
            }
        };

        template<class... Types>
        struct CheckFemVar<FemUnionTImpl<true, Types...>>: public std::integral_constant<bool, true> {};

        template<bool isFem, class... Types>
        struct FemUnionTImpl {
            static_assert(!isFem, "Some of types is not type of fem variable");
        };

        template<class T>
        struct FemUnionTImpl<true, T> : public std::tuple<T> {
            using Base = std::tuple<T>;

            static DenseMatrix<const double> GetOrthBasisShiftMatrix() {
                return UnionOrthCoefsStorageSingle<T>::asMatrix();
            }

            static const std::array<double, static_cast<std::size_t>(Operator<IDEN, T>::Nfa::value)
                * static_cast<std::size_t>(Operator<IDEN, T>::Nfa::value)>& orthCoefs() {
                return UnionOrthCoefsStorageSingle<T>::get();
            }
        };

        template<class T1, class T2, class... Rest>
        struct FemUnionTImpl<true, T1, T2, Rest...> : public std::tuple<T1, T2, Rest...> {
            using Base = std::tuple<T1, T2, Rest...>;

            /// Mixing matrix W = G^{-1} for union interpolating functionals (bases kept as-is).
            static DenseMatrix<const double> GetOrthBasisShiftMatrix() {
                return UnionOrthCoefsStorage<T1, T2, Rest...>::asMatrix();
            }

            static const std::array<double, static_cast<std::size_t>(NfaCounter<0, IDEN, T1, T2, Rest...>::value)
                * static_cast<std::size_t>(NfaCounter<0, IDEN, T1, T2, Rest...>::value)>& orthCoefs() {
                return UnionOrthCoefsStorage<T1, T2, Rest...>::get();
            }
        };
    }

};