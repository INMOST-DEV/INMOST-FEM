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
                decltype(convToBendMx(Operator<OP, T>::template apply<double, int>(std::declval<AniMemory<double, int>& >(), std::declval<ArrayView<double>& >())))::Nparts::value
                + NfaCounter<SUM, OP, Types...>::value>{};
    }

};