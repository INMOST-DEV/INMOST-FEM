Данная инструкция состоит из двух частей: в [первой части](#описание-anifem) приводится обзор основных элементов и возможностей AniFem++, а [вторая часть](#примеры-использования) представлена в виде набора постепенно усложняющихся примеров, которые призваны научить использовать данную библиотеку. Если вы были ранее знакомы с библиотекой [Ani3D/AniFem](https://sourceforge.net/projects/ani3d/), то можете начинать знакомство сразу с написания [кода](#первое-знакомство).

- [Описание возможностей AniFem++](#описание-возможностей-anifem)
  - [Квадратурные формулы](#квадратурные-формулы)
  - [Конечные элементы](#конечные-элементы)
  - [Элементная матрица](#элементная-матрица)
    - [Формы тензора $\mathbf{D}$](#формы-тензора)
    - [Интерфейсы `fem3d[tet|face|edge|node]`](#интерфейсы-fem3dtetfaceedgenode)
  - [Вычисление КЭ функций в точке](#вычисление-кэ-функций-в-точке)
  - [Наложение условий Дирихле](#наложение-условий-дирихле)  
- [Примеры использования](#примеры-использования)
  - [Первое знакомство](#первое-знакомство)

# Описание AniFem++
Основными элементами библиотеки AniFem++ являются: 
- множество квадратурных формул
- классы с описанием используемых КЭ пространств
- функции для вычисления элементных матриц
- функции вычисления КЭ функции в точке
- функции для наложения граничных условий Дирихле

Последовательно рассмотрим каждый из этих элементов

## Квадратурные формулы
Для численной аппроксимации возникающих при КЭ дискретизации интегралов реализованы квадратурные формулы вида $\{w^i, \mathbf{\lambda}^i\}_{i=1}^{N}$:
$$\int_{S_d} f(x) dx \approx |S_d| \sum_{i = 1}^{N} w^i f\left(\sum_{k = 1}^{d+1} \lambda^i_k \mathbf{p}_k\right), $$
где $\mathbf{\lambda}$ - барицентрические координаты точки внутри $d$-мерного симплекса $S_d$ с опорными точками $\mathbf{p}_k$. Формулы реализованы вплоть до 20-го порядка для 1D, 2D и 3D симплексов (для отрезков, треугольников, тетраэдров), имеют узлы только внутри симлексов ($\lambda^i_k > 0$) и только положительные веса ($w^i > 0$).

Функции для получения квадратур, см. [quadrature_formulas.h](../../anifem%2B%2B/fem/quadrature_formulas.h):
```C++
TetraQuadFormula    tetrahedron_quadrature_formulas(int order);
TriangleQuadFormula triangle_quadrature_formulas(   int order);
SegmentQuadFormula  segment_quadrature_formulas(    int order);
```
Ввиду реализации формул целиком через статическую память, вызов этих функций является дешёвой операцией.

## Конечные элементы
В рамках AniFem++ конечные элементы имеют 2 варианта реализации: compile-time (через шаблоны) и runtime (через наследование). 

Исходно предоставляется следующий набор КЭ пространств (хотя можно определять и собственные пространства):
|Базовые пространства | compile-time представление | runtime представление |
|--------------|:--------------:|:---------:|
|piecewise constant, $P_0$| `FemFix<FEM_P0>`| `P0Space` |
|continuous piecewise linear, $P_1$| `FemFix<FEM_P1>`| `P1Space` |
|continuous piecewise quadratic, $P_2$| `FemFix<FEM_P2>`| `P2Space` |
|continuous piecewise cubic, $P_3$| `FemFix<FEM_P3>`| `P3Space` |
|the lowest order Nedelec (edge) finite element| `FemFix<FEM_ND0>` | `ND0Space`|
|the lowest order Raviart-Thomas (face) finite element| `FemFix<FEM_RT0>` | `RT0Space`|
|the Crouzeix-Raviart finite element| `FemFix<FEM_CR1>` | `CR1Space` | 
|Bubble subspace, 4-order cell-centred basis function, incomplete space| `FemFix<FEM_B4>` | `BubbleSpace` |

Важно отметить, что `BubbleSpace` не является завершённым КЭ пространством а лишь содержит единственную базисную функцию $\phi = \lambda_1 \lambda_2 \lambda_3 \lambda_4$ и предоставляется для обогащения других пространств. 

|Операции над пространствами| compile-time пример | runtime пример |
|--------------|:--------------:|:---------:|
|определение пространства| `using P1Fem = FemFix<FEM_P1>; using P2Fem = FemFix<FEM_P2>`| `FemSpace P1{P1Space()}, P2{P2Space()}, B4{BubbleSpace()};` |
|декартово произведение, $\times$| `FemCom<P1Fem, P2Fem, P1Fem>` | `P1*P2*P1` |
|возведение в натуральную степень, ^k| `FemVecT<3, P1Fem>` | `(P1^3)` |
|обогащение пространства, +| недоступно | `P1 + B4` |

Все пространства предоставляют функторы для вычисления действия следующих линейных дифференциальных операторов (если они математически определены) на базисные функции:
 |Имя оператора|   Описание   |
 |--------------|--------------|
 |`IDEN`| тождественный оператор, $\mathrm{IDEN}(v^h) = v^h$|
 |`GRAD`| оператор градиента, $\mathrm{GRAD}(v^h) = \nabla v^h$|
 |`DIV` | оператор дивергенции, $\mathrm{DIV}(v^h) = \mathrm{div}\ v^h$|
 |`CURL`| оператор ротора, $\mathrm{CURL}(v^h) = \mathrm{rot}\ v^h$|
 |`DUDX`| частная производная в x-направлении, $\mathrm{DUDX}(v^h) = \partial v^h / \partial x$|
 |`DUDY`| частная производная в y-направлении, $\mathrm{DUDY}(v^h) = \partial v^h / \partial y$|
 |`DUDZ`| частная производная в z-направлении, $\mathrm{DUDZ}(v^h) = \partial v^h / \partial z$|

*Также в runtime режиме у класса FemSpace есть метод*
 ```C++
 void evalBasisFunctions(const Expr& lmb, const Expr& grad_lmb, std::vector<Expr>& phi);
 ```
 *который можно использовать для символьного вычисления действия любых линейных дифференциальных операторов на элементные базисные функции.*

Конечные элементы в AniFem++ представлены следующим набором:
- функторы для вычисления действия дифференциальных операторов на множество элементных базисных функций КЭ пространства, $Op(\phi)$
- отображение элементных степеней свободы на элементы тетраэдра $DofMap$
- функции интерполяции гладких функций на отдельные степени свободы, $I_i(f)$ или степени свободы, лежащие на определённой части тетраэдра, $I_g(f)$

|Операция пространства `UFem` | compile-time пример | runtime пример |
|--------------|:--------------:|:---------:|
| получить функтор $Op(\phi)$, в примерах $Op = \mathrm{GRAD}$ | `Operator<GRAD, UFem>` | `UFem.getOP(GRAD);` |
|получить DofMap| `Dof<UFem>::Map()` | `UFem.dofMap()` |
|вычислить $I_i(f)$| `Dof<UFem>::interpolate(tetra, f, dof_vals, idof_on_tet)`| `UFem.interpolateOnDOF(tetra, f, dof_vals, idof_on_tet, wmem)`|
|вычислить $I_g(f)$| `interpolateByDOFs<UFem>(tetra, f, dof_vals, tet_part)`| `UFem.interpolateByDOFs(tetra, f, dof_vals, tet_part, wmem)`|

## Элементная матрица
Для вычисления элементных матриц используются функции `fem3d[tet|face|edge|node]` (здесь внутри [] перечислены возможные суффиксы), которые вычисляют элементную матрицу для билинейной формы
$$\int_{S_d} (\mathbf{D}\ Op_A(u^h)) \cdot Op_B(v^h) dx,$$
где $S_d$ - весь тетраэдр для суффикса `tet`, грань тетраэдра для -  `face`, ребро тетраэдра для - `edge` и конкретный узел для - `node`, $\mathbf{D}$ это заданный тензор, $Op_A$ и $Op_B$ линейные дифференциальные операторы и $u^h$, $v^h$ - КЭ функции.

Для осмысленности выражения выше произведения матриц должны иметь смысл. Например, если $Op_A(u^h) = GRAD((P_1)^3)$, т.е. имеет размерность $9 \times 12$, то $\mathbf{D}$ должно иметь $9$ строк. В общем случае если $Op_A(u^h)$ имеет физическую размерность $d_A$ и включает $nf_A$  элементных степеней свободы, т.е. имеет полную размерность  $d_A \times nf_A$, а $Op_B(v^h)$ имеет полную размерность $d_B \times nf_B$, то $\mathbf{D}$ должно иметь размерность $d_B \times d_A$, а элементная матрица будет иметь размерность $nf_B \times nf_A$.

### Формы тензора $\mathbf{D}$

В качестве тензора $\mathbf{D}$ может выступать функтор (функция или любая структура для которой переопределён `operator()(...)` соотвествующей сигнатуры)
с одной из двух вариантов сигнатуры:
- поточечная сигнатура `OnePointTensor`
```C++
TensorType Dcoef(const std::array<Scalar, 3> &X, Scalar *D, TensorDims Ddims, void *user_data, int iTet);
/*
 * X         - Cartesian coordinates of a 3D point where tensor Dcoef should be evaluated
 * Ddims     - dimensions of tensor:  Ddims.first x Ddims.second
 * D         - storage for row-major matrix with dimensions Ddims.first x Ddims.second
 * user_data - user given data
 * iTet      - number of tetrahedron from range from 0 to f-1, where f is number of tetrahedrons processed per one call (usually f = 1 and iTet = 0)
 */
```
- сплавляющая сигнатура `FusiveTensor`:
```C++
TensorType Dcoef(ArrayView<Scalar> X, ArrayView<Scalar> D, TensorDims Ddims, void *user_data, const AniMemory<Scalar, IndexType>& mem);
/*
 * mem       - special structure that stores all the data that the local matrix assembler works with, e.g. mem.XYL is baricentric coordinates of used quadrature, mem.q is number of points from quadrature, mem.f is number of tetrahedrons processed per one call, other see in anifem++/fem/fem_memory.h
 * X         - quadrature 3D points: X_r = (p_{0,r}^T, ..., p_{q-1,r}^T)^T,  X = [X_0, ..., X_{f-1}] where tensor Dcoef should be evaluated
 * Ddims     - physical dimensions of tensor at point:  Ddims.first x Ddims.second
 * D         - contigous storage of D tensor evaluations, where D + q*r*Ddims.first*Ddims.second points on start of memory for storing value of the tensor evaluated at point p_{q,r} as row-major matrix with dimensions Ddims.first x Ddims.second
 * user_data - user given data
 */
```
Отметим, что в большинстве приложений достаточно использования поточечной сигнатуры, однако сплавляющая сигнатура потенциально может приводить к бОльшей векторизации кода, а кроме того через неё можно организовать расчёты для тензоров, исходно задаваемых своими значениями в узлах квадратур (такое может случится, например, если значение тензора определяется в результате решения ОДУ, которые решаются в узлах квадратур).  

Поддерживаемые типы тензоров `TensorType`:
- `TENSOR_GENERAL` - произвольная $m \times n$ матрица
- `TENSOR_SYMMETRIC` - симметричная матрица
- `TENSOR_SCALAR` - тензор вида $\alpha \mathbb{I}$, где $\mathbb{I}$ - единичная матрица
- `TENSOR_NULL` - единичная матрица $\mathbb{I}$

Для максимального использования априорной информации об используемых тензорах на этапе компиляции, а также чтобы выбрать правильную ожидаемую его форму, каждый тензор при использовании снабжается отдельной структурой с информацией о нём. 

Для тензоров поточечной сигнатуры используется следующая структура для информации
```C++
template<int TensorTypeSparse = PerPoint, bool isConstant = false, long idim = -1, long jdim = -1>
struct DfuncTraits;
/*
 * if idim > 0 && jdim > 0 then D tensor should be idim x jdim matrix 
 * isConstant shows whether tensor is constant
 * TensorTypeSparse shows whether the type of tensor depends on the point of its calculation and how this dependence is expressed
 */
```
|тип параметра `TensorTypeSparse`| Описание|
|--------------|:--------------:|
|значение типа `TensorType`| означает, что тензор всегда возвращает это значение|
|`PerSelection`| тензор имеет одинаковый тип в пределах одного вызова `fem3d[...]` |
|`PerTetra`| тензор имеет одинаковый тип в пределах одного тетраэдра|
|`PerPoint`| тип тензора может меняться от точки к точке|

Для тензоров сплавляющей сигнатуры структура для информации намного проще:
```C++
template<long idim = -1, long jdim = -1>
struct DfuncTraitsFusive;
// if idim > 0 && jdim > 0 then D tensor should be idim x jdim matrix
```

Приведём примеры некоторых типов априорной информации:
- `DfuncTraits<PerPoint, false>` - произвольный тензор, чей тип определяется в runtime на основе его возвращаемого значения в каждой точке (обычный тензор, как он был в [Ani3D/AniFem](https://sourceforge.net/projects/ani3d/))
- `DfuncTraits<TENSOR_NULL>` - compile-time единичный тензор (в runtime функция вычисления этого тензора вызываться не будет)
- `DfuncTraits<TENSOR_SCALAR, true>` - compile-time скалярный тензор, при вычислениях будет вызван только в одной точке, а в остальных будет использовано вычисленной в первой точке значение
- `DfuncTraits<TENSOR_GENERAL>` - общий тензор, который в каждой точке должен возвращать `TENSOR_GENERAL`
- `DfuncTraitsFusive<>` - означает, что ожидается тензор сплавляющей сигнатуры


### Интерфейсы `fem3d[tet|face|edge|node]`
В зависимости от задачи может требоваться разная степень гибкости и изменчивости ассемблируемых локальных матриц, так а) иногда достаточно использовать только $P_1$ (или каких-то конкретных других) конечных элементов и структура тензоров известна заранее, однако также бывают случаи, когда по каким-то причинам б) необходимо иметь возможность легко менять КЭ пространства для дискретизации (без перекомпиляции проекта), а также о структуре тензора априори известно мало. Чтобы предложить максимальную эффективность для сценариев типа а) и при этом иметь возможность реализовать гибкий интерфейс, как в сценарии типа б), предложено несколько перегрузок для функций `fem3d[...]`. 

Для примера продемонстрируем предлагаемые перегрузки для `fem3dtet`, т.к. для других функций они аналогичны и отличаются лишь наличием одного дополнительного аргумента (см. заголовки из  anifem++/fem/operations/)
- **Шаблонный интерфейс с собственной статической памятью**
```C++
/*
 * The function to compute elemental matrix for the volume integral
 * over tetrahedron T: \f$\int_T [(D \dot OpA(u)) \dot OpB(v)] dx \f$
 * Memory-less version of fem3Dtet
 * @tparam OpA is left fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
 * @tparam OpB is right fem operator
 * @tparam FuncTraits is traits will be associated with tensor function Dfnc
 * @tparam FUSION is maximal available number of tetrahedron in tetra container XYZ
 * @tparam TetrasT is any Tetras structure defined in geometry.h 
 * @param XYZ is storage for tetrahedron nodes
 * @param Dfnc is tensor function
 * @param[in,out] A is matrix to store resulting elemental matrix
 * @param order is order of used gauss quadrature formula
 * @param user_data is user defined data block will be propagate to Dfnc
 */ 
template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, 
         int FUSION = 1, typename RT = double, typename IT = int, 
         typename Functor, typename TetrasT>
void fem3Dtet(const TetrasT& XYZ, const Functor& Dfnc, DenseMatrix<RT>& A, 
              int order = 5, void *user_data = nullptr);
```
Размер выделяемой статической памяти в этой функции не зависит от используемого порядка квадратур и типа КЭ пространства и потому выделяемой памяти может в общем случае не хватить (что вызовет срабатывание assert в Debug режим). Однако для большинства приложений (для квадратур не выше 6-го порядка и КЭ пространств имеющих не более 30 степеней свободы) статической памяти будет достаточно и этот вариант функции можно использовать.

Однако для тех случаев, когда возможна нехватка статической памяти предусмотрен интерфейс с внешней памятью
- **Шаблонный интерфейс с внешней памятью**
```C++
///@param plainMemory is external memory for work
template<typename OpA, typename OpB, typename FuncTraits = DfuncTraits<>, 
         typename RT = double, typename IT = int, 
         typename Functor, typename TetrasT>
void fem3Dtet(const TetrasT& XYZ, const Functor& Dfnc, DenseMatrix<RT>& A,
              PlainMemory<RT, IT> plainMemory,
              int order = 5, void *user_data = nullptr);
```
Здесь добавился один дополнительный параметр содержащий внешнюю память.
Сама оценка необходимой памяти может быть получена с помощью вызова
```C++
///Give memory requirement for external memory-dependent specialization of fem3Dtet
///@param order is order of quadrature formula will be used
///@param fusion is fusion parameter will be used
///@return PlainMemory where were set dSize and iSize fields to required amount of memory
template<typename OpA, typename OpB, typename RT = double, typename IT = int>
PlainMemory<RT, IT> 
  fem3Dtet_memory_requirements(int order, int fusion = 1);
```
При этом установить память необходимого размера в структуре `PlainMemory<...>` пользователь должен самостоятельно.
- **Гибкий динамический интерфейс**
```C++
///@param applyOpU,applyOpV are evaluators of Op(Fem_Basis) 
template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TetrasT>
void fem3Dtet(
  const TetrasT& XYZ, 
  const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, 
  const Functor& Dfnc, DenseMatrix<>& A,
  PlainMemoryX<> mem, int order = 5, void *user_data = nullptr);

template<typename FuncTraits = DfuncTraits<>>
PlainMemoryX<> fem3Dtet_memory_requirements(
  const ApplyOpBase& applyOpU, const ApplyOpBase& applyOpV, 
  int order, int fusion = 1)
```
Здесь в отличие от предыдущих интерфейсов действие КЭ пространств является фактическим параметром функций, а не шаблонным, и представляется посредством наследования от абстрактного класса `ApplyOpBase` и варианта использования статического пула памяти не предусмотрено.

В функциях `fem3d[face|edge|node]` в отличие от `fem3dtet` есть дополнительный аргумент, который обозначает локальный номер грани, ребра или узла, соответственно, на тетраэдре. Используется следующее **соглашение локальной нумерации**:
- узлы имеют номера от 0 до 3 и соответствуют полям XYZ.XYi в контейнере тетраэдров (i от 0 до 3)
- рёбра имеют номера от 0 до 6, обозначая рёбра между вершинами 01, 02, 03, 12, 13, 23 соответственно
- грани имеют номера от 0 до 4, обозначая рёбра между вершинами 012, 123, 230, 301 соответственно

Для вычисления элементные матрицы для **линейных форм правых частей** можно использовать следующую хитрость:
$$f(Op(v^h)) = \int_{S_d} (\mathbf{D}_{rhs}\ \mathrm{IDEN}(\mathrm{FEM\_P0})) \cdot Op(v^h) dx,$$
где $\mathbf{D}_{rhs}$ представляет функцию правой части $f$, что выражается в коде как
```C++
//template variant
using TP0 = Operator<IDEN, FemFix<FEM_P0>>;
using TV = Operator<Op, VFem>; 
fem3Dtet<TP0, TV>(XYZ, Drhs, A, order, user_data);

//runtime variant
ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
auto op_u = UFem.getOP(Op);
fem3Dtet(XYZ, iden_p0, op_u, Drhs, A, mem, order, user_data);
```

## Вычисление КЭ функций в точке
При работе с нелинейными задачами очень важно иметь возможность оценить значение дискретной КЭ функции в точках, т.к. эта функция может входить в состав тензоров, участвующих в формировании элементных матриц. С этой целью реализован набор функций, которые, как и в случае с функциями ассмблирования элементных матриц, отличаются интерфейсом работы с КЭ пространствами:

- **Шаблонный интерфейс с собственной статической памятью**
```C++
/**
 * @brief Evaluate \f$ Op(u)[x] \f$
 * @param[in] dofs is vector degrees of freedom of the FE variable, e.g. for Op = Operator<GRAD, FemFix<FEM_P1>> dofs is d.o.f's vector of FEM_P1 element type 
 * @param[in,out] opU: = [vec(opU[p_1^T])^T, ..., vec(opU[p_q^T])^T]^T and vec(opU[p_i^T]) is vectorized matrix computed at point p_i in the tetrahedron
 * @param[in] X = [p_1^T, ..., p_q^T] are 3D points for function evaluation
 * @param[in] XYZ is storage for tetrahedron nodes on which the value is evaluated
 * @tparam MAXPNTNUM is maximal number of points to be evaluated at the same time
 * @tparam Op is fem operator (e.g. \code Operator<GRAD, FemFix<FEM_P1>> \endcode)
 * @return result of evaluating Op(u) at X point set
 */
template<typename Op, int MAXPNTNUM = 24, typename RT = double, typename IT = int, typename TRT>
void fem3DapplyX(
    const Tetra<TRT>& XYZ, const ArrayView<const RT> X,
    const ArrayView<RT>& dofs, ArrayView<RT>& opU);
```
- **Шаблонный интерфейс с внешней памятью**
```C++
template<typename Op, typename RT = double, typename IT = int, typename TRT>
void fem3DapplyX(
    const Tetra<TRT>& XYZ, const ArrayView<const RT> X,
    const ArrayView<RT>& dofs, ArrayView<RT>& opU,
    PlainMemory<RT, IT> plainMemory);

template<typename Op, typename RT = double, typename IT = int>
PlainMemory<RT, IT> fem3DapplyX_memory_requirements(int pnt_per_tetra, int fusion = 1);    
```
- **Гибкий динамический интерфейс**
```C++
///@brief Evaluate \f$ Op(u)[x] \f$
template<typename TRT>
void fem3DapplyX(
  const Tetra<TRT>& XYZ, const ArrayView<const double> X, 
  const DenseMatrix<>& dofs, const ApplyOpBase& applyOpU,
  DenseMatrix<>& opU, PlainMemoryX<> mem);

PlainMemoryX<> fem3DapplyX_memory_requirements(const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1)  
```

Помимо вычисления КЭ функции в точке $Op(u^h)[x]$ в динамическом интерфейсе доступно **вычисление КЭ выражений** 
$$\{D \cdot Op(u^h)\}[x]$$ 
с помощью функций:
```C++
///@brief Evaluate \f$ D[x] * Op(u)[x] \f$
///@param Ddim1 is number of rows in D tensor
///@param Dfnc is function of the tensor
///@param user_data is user-supplied data to be postponed to the tensor Dfnc
template<typename FuncTraits = DfuncTraits<>, typename Functor, typename TRT>
void fem3DapplyX(
  const Tetra<TRT>& XYZ, const ArrayView<const double> X, 
  const DenseMatrix<>& dofs, const ApplyOpBase& applyOpU, 
  uint Ddim1, const Functor& Dfnc, 
  DenseMatrix<>& opU, PlainMemoryX<> mem, void *user_data = nullptr); 

template<typename FuncTraits = DfuncTraits<>>
PlainMemoryX<> fem3DapplyX_memory_requirements(uint Ddim1, const ApplyOpBase& applyOpU, uint pnt_per_tetra = 1, uint fusion = 1);  
```

## Наложение условий Дирихле

TODO: написать
# Примеры использования

TODO: написать
## Первое знакомство

TODO: написать