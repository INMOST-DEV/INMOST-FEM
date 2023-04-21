Данная инструкция состоит из двух частей: в [первой части](#описание-anifem) приводится обзор основных элементов и возможностей AniFem++, а [вторая часть](#примеры-использования) представлена в виде набора постепенно усложняющихся примеров, которые призваны научить использовать данную библиотеку. Если вы были ранее знакомы с библиотекой [Ani3D/AniFem](https://sourceforge.net/projects/ani3d/), то можете начинать знакомство сразу с написания [кода](#первое-знакомство).

- [Описание возможностей AniFem++](#описание-возможностей-anifem)
  - [Квадратурные формулы](#квадратурные-формулы)
  - [Конечные элементы](#конечные-элементы)
  - [Элементная матрица](#элементная-матрица)
    - [Формы тензора $\mathbf{D}$](#формы-тензора)
    - [Интерфейсы `fem3d[tet|face|edge|node]`](#интерфейсы-fem3dtetfaceedgenode)
  - [Вычисление КЭ функций в точке](#вычисление-кэ-функций-в-точке)
  - [Наложение условий Дирихле](#наложение-условий-дирихле)
    - [Фиксация значения по номеру степени свободы](#фиксация-значения-по-номеру-степени-свободы)
    - [Условие Дирихле на части тетраэдра](#условие-дирихле-на-части-тетраэдра)
    - [Векторное условие Дирихле](#векторное-условие-дирихле)  
- [Примеры использования](#примеры-использования)
  - [Первое знакомство](#первое-знакомство)
    - [Немного теории](#немного-теории)
    - [Простейшая реализация](#простейшая-реализация)
    - [Общая реализация](#общая-реализация)

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

$$\int_{S_d} f(x) dx \approx |S_d| \sum_{i = 1}^{N} w^i f\left(\sum_{k = 1}^{d+1} \lambda^i_k \mathbf{p}_k\right),$$

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

Для вычисления элементных матриц для **линейных форм правых частей** можно использовать следующую хитрость:
$$f(v^h) = \int_{S_d} (\mathbf{D}_{rhs}\ \mathrm{IDEN}(\mathrm{FEM\_P0})) \cdot \mathrm{IDEN}(v^h) dx,$$
где $\mathbf{D}_{rhs}$ представляет функцию правой части $f$, что выражается в коде как
```C++
//template variant
using TP0 = Operator<IDEN, FemFix<FEM_P0>>;
using TV = Operator<IDEN, VFem>; 
fem3Dtet<TP0, TV>(XYZ, Drhs, A, order, user_data);

//runtime variant
ApplyOpFromTemplate<IDEN, FemFix<FEM_P0>> iden_p0;
auto iden_u = UFem.getOP(IDEN);
fem3Dtet(XYZ, iden_p0, iden_u, Drhs, A, mem, order, user_data);
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
Наложение условий типа Дирихле производится посредством модификации элементных матриц и правых частей. 
### Фиксация значения по номеру степени свободы
Пусть в результате дискретизации условие Дирихле на тетраэдре приняло вид:
$$u^h_k = d_k,$$
где $u^h_k$ - $k$-ая степень свободы для дискретной переменной $u^h$, а $d_k$ - скаляр

Предусмотрено 2 сценария для наложения условий данного типа:
- при решении линейной задачи при одновременной сборке матрицы и правой части

В этом случае модификация матрицы и правой части должна производиться согласовано и для этого может ипользоваться функция
```C++
///Set dirichlet value on k-th tetrahedron's dof. Supposed trial and test FEM spaces are same
///@param A is elemental matrix
///@param F is elemental right hand side
///@param k is index of d.o.f. on element (tetrahedron)
///@param bc is dirichlet value to be set
template<typename RT>
void applyDir(DenseMatrix<RT>& A, DenseMatrix<RT>& F, int k, RT bc);
```
- при сборке невязки и якобиана для решения нелинейных систем

В этом случае модификация матрицы и правой части независимы и выполняются разными функциями
```C++
///Apply k-th degree of freedom Dirichlet condition on matrix
///@param A is elemental matrix
///@param k is index of d.o.f. on element (tetrahedron) 
template<typename RT>
void applyDirMatrix(DenseMatrix<RT>& A, int k);
    
///Set zero on k-th position in F vector
///@see applyDir
template<typename RT>
void applyDirResidual(DenseMatrix<RT>& F, int k);
```
Безусловно, наложение условия Дирихле на конкретную степень свободы задачи не всегда удобна, т.к. нумерация степеней свободы зависит от используемых при дискретизации типов элементов ($P_1$, $P_2$ и т.д.)
### Условие Дирихле на части тетраэдра
Часто возникает ситуация, когда условие Дирихле на тетраэдре можно записать как
$$\mathbf{u}^h|_g = \mathbf{d}^h,$$
где $g$ - часть геометрических элементов тетраэдра (какие-то его узлы, рёбра и т.д.), $\mathbf{u}^h|_g$ - сужение дискретной переменной на часть тетраэдра $g$ (или вектор степеней свободы, которые лежат на $g$), $\mathbf{d}(\mathbf{x})$ - гладкая функция, задаваемая исходным условием Дирихле, а $\mathbf{d}^h$ это её интерполяция на степени свободы на тетраэдре.

В этом случае процесс наложения этого граниченого условия представляется в виде трёх этапов:

1. Спецификация частей тетраэдра, на которых условия будут наложены

Для этого используется класс `DofT::TetGeomSparsity` (он реализован эффективно, использовать его "дёшево"). Приведём пример:
```C++
// see local tetrahedron enumeration agreement
DofT::TetGeomSparsity tet_part;
// add node with index 1
tet_part.setNode(1);
// add face with index 2 and interior parts, 
// i.e. nodes 2,3,0 and edges 1,2,5
tet_part.setFace(2, true);
// add face with index 3 (without interior parts)
tet_part.setFace(3);
```

2. Интерполяция граничного условия на необходимые степени свободы

В случае, если гран. условие задано гладкой функцией, надо вычислить её интерполяцию на степени свободы, лежащие на выбранных на первом этапе частях тетраэдра. Для этого используются функции интерполяции $I_g(f)$, которые рассматриваются в разделе [Конечные элементы](#конечные-элементы). 

3. Непосредственно модификация элементных матриц и правых частей

Здесь также предусмотрено 2 сценария наложения
- при решении линейной задачи при одновременной сборке матрицы и правой части
```C++
/// @brief Applies the Dirichlet condition var[i-th dof] = dofs[i] where geometrical locus of i-th d.o.f. belongs to sp 
/// @param dofs is array of d.o.f.s that should be considered as value of dirichlet d.o.f.s in indexes defined by sp
/// @param trial_map is dof map of trial FEM space
/// @param test_map is dof map of test FEM space
/// @param A is elemental matrix
/// @param F is elemental right hand side
template<typename RT>
void applyDirByDofs(const BaseDofMap& trial_map, const BaseDofMap& test_map, DenseMatrix<RT>& A, DenseMatrix<RT>& F, const TetGeomSparsity& sp, const ArrayView<const RT>& dofs);

/// @note If trial_map == test_map
template<typename RT>
void applyDirByDofs(const BaseDofMap& trial_map, DenseMatrix<RT>& A, DenseMatrix<RT>& F, const TetGeomSparsity& sp, const ArrayView<const RT>& dofs);

/// @brief Applies the constant Dirichlet condition var = bc * I, where dim(u) = dim(I) and I is vector of units (1, 1, ..., 1)
template<typename RT>
void applyConstantDirByDofs(const BaseDofMap& trial_map, const BaseDofMap& test_map, DenseMatrix<RT>& A, DenseMatrix<RT>& F, const TetGeomSparsity& sp, RT bc);

/// @note If trial_map == test_map
template<typename RT>
void applyConstantDirByDofs(const BaseDofMap& trial_map, DenseMatrix<RT>& A, DenseMatrix<RT>& F, const TetGeomSparsity& sp, RT bc);
```

- при сборке невязки и якобиана для решения нелинейных систем
```C++
template<typename RT>
void applyDirMatrix(const BaseDofMap& trial_map, const BaseDofMap& test_map, DenseMatrix<RT>& A, const TetGeomSparsity& sp);

/// @note If trial_map == test_map
template<typename RT>
void applyDirMatrix(const BaseDofMap& trial_map, DenseMatrix<RT>& A, const TetGeomSparsity& sp);
    
template<typename RT>
void applyDirResidual(const BaseDofMap& test_map, DenseMatrix<RT>& F, const TetGeomSparsity& sp);
```

### Векторное условие Дирихле 
При работе с векторными переменными иногда возникает ситуация, когда на тетраэдре необходимо наложить условие вида:
$$A(\mathbf{x}) \cdot \mathbf{u}|_g = \mathbf{d},$$
где
- $g$ - часть геометрических элементов тетраэдра (какие-то его узлы, рёбра и т.д.)
- $A: g \rightarrow \mathcal{M}^{m \times n},\ m < n$ - матричная функция, определённая на $g$, а $\mathcal{M}^{m \times n}$ - пространство матриц полного ранга размера ${m \times n}$
- $\mathbf{u}$ - векторная физическая переменная задачи, $\mathrm{dim}\ \mathbf{u} = n$ 
- $\mathbf{d}$ - гладкая вектор-функция, $\mathrm{dim}\ \mathbf{d} = m$.

В рамках AniFem++ наложение такого рода условий возможно при условии, что для дискретизации каждой компоненты физической переменной $\mathbf{u}$ используется одинаковое КЭ пространство $S$, например, если $\mathbf{u}^h$ дискретизовано как $(S)^n$. В этом случае условие Дирихле на дискретном уровне представляется как:
$$I_i^S(A) \cdot \mathbf{u}^h_{S,i} = I_i^S(\mathbf{d})$$
где 
- $i$ - это номер степени свободы, и принимает все значения, для которых степень свободы КЭ пространсва $S$ лежит на $g$
- $I_i^S$ - оператор интерполяции на $i$-ую степень свободы КЭ пространсва $S$, который на векторы и матрицы действет поэлементно
- $\mathbf{u}^h_{S,i} = ((u_1)^h_i, (u_2)^h_i, ..., (u_n)^h_i)^T$ - вектор, где каждая компонента представлена $i$-ой степенью свободы КЭ пространсва $S$ (которой эта степень свободы и дискретизована).

TODO: дореализовать интерфейс для их накладывания

# Примеры использования

Здесь представлено несколько тестовых задач разной степени сложности и демонстрирующие отдельные элементы возможностей библиотеки.
## Первое знакомство

Начнём знакомство с AniFem++ с рассмотрения классического уравнения диффузии в кубической области $\Omega = [0, 1]^3$:
$$
\begin{aligned} 
-\mathrm{div}\ (\mathbb{D}(\mathbf{x})\ \mathrm{grad}\ u) &= f(\mathbf{x})\ &in\ \ &\Omega,\\
u &= u_0(\mathbf{x})\ &on\ \ &\partial\Omega,
\end{aligned}
$$
где $\mathbb{D}(\mathbf{x}) = (1 + x^2) \mathbb{I}$, $f(\mathbf{x}) = 1$, $u_0(\mathbf{x}) = 0$

### Немного теории
Слабая постановка имеет вид: *найти функцию $u \in \mathring H^1_u = \{s \in H^1(\Omega): s|_{\partial \Omega} = 0 \}$, которая при любых соответствующих пространству тестовых функциях $\phi$ удовлетворяют следующим уравнениям:*
$$\int_{\Omega}(\mathbb{D}(\mathbf{x})\ \mathrm{grad}\ u) \cdot \mathrm{grad}\ \phi\ d\mathbf{x} = \int_{\Omega} f(\mathbf{x})\cdot \phi\ d\mathbf{x}.$$

После дискретизации области $\Omega^h$ и выбора соответствующих дискретных КЭ пространств $u^h = \sum_{j} u_j \phi_j$ имеем систему уравнений: 
$$\sum_{j} \sum_{T \in \Omega^h} \sum_{\mathbf{x}_q \in T} w_q [(\mathbb{D}\ \mathrm{grad}\ \phi_j) \cdot \mathrm{grad}\ \phi_i]_{\mathbf{x}_q} u_j = \sum_{T \in \Omega^h} \sum_{\mathbf{x}_q \in T} w_q [f \phi_i]_{\mathbf{x}_q},$$
где $\sum_{T \in \Omega^h}$ - сумма по всем элементам $T$ области $\Omega^h$, $\sum_{\mathbf{x}_q \in T} w_q$ - сумма по точкам квадратурных формул с соответствующими весами. После переобозначений последнее выражение принимает вид:
$$\mathcal{A}_{D} \cdot U^h = F^h$$

### Простейшая реализация
Для начала продемонстрируем как написать код для какой-то конкретной КЭ дискретизации, например, для $P_1$, при этом не предполгая его простую переносимость для использования других пространств и явно используя карту степеней свободы этого пространства. Смотрите [ex1.cpp](../../examples/tutorials/ex1.cpp)

Этапы написания кода:
1.  Инициализируем INMOST, подготовливаем сетку, метки границы и параметры задачи - 30-71 строки [ex1.cpp](../../examples/tutorials/ex1.cpp)
2. Задаём шаблонное КЭ пространство $P_1$ и число его элементных степеней свободы - 44-47 строки [ex1.cpp](../../examples/tutorials/ex1.cpp)
```C++
using UFem = FemFix<FEM_P1>;
constexpr auto UNF = Operator<IDEN, UFem>::Nfa::value;
 ```
3. Создаём тэг для сохраненения значений в степенях свободы - 73-74 строки [ex1.cpp](../../examples/tutorials/ex1.cpp)

Элементы типа $P_1$ имеют степени свободы только на узлах и на каждый узел приходится только одна степень свободы
```C++
Tag u = m->CreateTag("u", DATA_REAL, NODE, NONE, 1);
```
4. Определяем структуру для хранения данных, которые мы хотим передавать в функцию сбора элементных матриц - 76-80 строки [ex1.cpp](../../examples/tutorials/ex1.cpp)

Здесь мы хотим передавать только метки узлов, т.к. по ним мы будем выявлять узлы, лежащие на границе
```C++
struct ProbLocData{
    std::array<int, 4> nlbl = {0};
};
```
5. Определяем функцию для ассемблирования элементной матрицы и правой части - 82-106 строки [ex1.cpp](../../examples/tutorials/ex1.cpp)

В нашем случае мы хотим ассемблировать матрицы выражений $$\int_{S^3}(\mathbb{D}(\mathbf{x})\ \mathrm{GRAD}(P_1)) \cdot \mathrm{GRAD}(P_1)\ d\mathbf{x}\ \textit{  и  }\ \int_{S^3} (f(\mathbf{x})\ \mathrm{IDEN}(P_0)) \cdot \mathrm{IDEN}(P_1)\ d\mathbf{x}$$
```C++
std::function<void(const double**, double*, double*, void*)> local_assembler =
      [&D_tensor, &F_tensor, &U_0](const double** XY/*[4]*/, double* Adat, double* Fdat, void* user_data) -> void{
  DenseMatrix<> A(Adat, UNF, UNF), F(Fdat, UNF, 1);
  Tetra<const double> XYZ(XY[0], XY[1], XY[2], XY[3]);

  // elemental stiffness matrix <grad(P1), D grad(P1)>
  fem3Dtet<Operator<GRAD, UFem>, Operator<GRAD, UFem>, DfuncTraits<TENSOR_SCALAR, false>>(XYZ, D_tensor, A, 2);

  // elemental right hand side vector <F, P1>
  fem3Dtet<Operator<IDEN, FemFix<FEM_P0>>, Operator<IDEN, UFem>, DfuncTraits<TENSOR_SCALAR, true>>(XYZ, F_tensor, F, 2);

  // read node labels from user_data
  auto& dat = *static_cast<ProbLocData*>(user_data);
  // impose dirichlet condition
  for (int i_node = 0; i_node < 4; ++i_node)
    if (dat.nlbl[i_node] > 0) {
        double bc = 0;
        std::array<double, 3> x{XY[i_node][0], XY[i_node][1], XY[i_node][2]};
        U_0(x, &bc, 1, nullptr);
        applyDir(A, F, i_node, bc); // set Dirichlet BC
    }
};
```
6. Определяем функцию, которая собирает все данные, которые нужны для сборки локальных матриц - строки 107-117 [ex1.cpp](../../examples/tutorials/ex1.cpp).

Эта функция всегда должна сохранять координаты узлов элемента, а также она собирает определяемые пользователем данные, т.е. в нашем случае метки узлов
```C++
auto local_data_gatherer = [&BndLabel](ElementalAssembler& p) -> void{
    //save nodes cooridinates
    double *nn_p = p.get_nodes();
    const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
    //specify user data
    ProbLocData data;
    std::fill(data.nlbl.begin(), data.nlbl.end(), 0);
    for (unsigned i = 0; i < data.nlbl.size(); ++i) 
        data.nlbl[i] = (*p.nodes)[i].Integer(BndLabel);

    //run elemental matrix assembler
    p.compute(args, &data);
    //here you can modify final elemental matrices if required
};
```
7. Инициализируем глобальный ассемблер - строки 119-131 [ex1.cpp](../../examples/tutorials/ex1.cpp)
- вызываем конструктор
```C++
Assembler discr(m);
```
- задаём определённые ранее функции для сборки элементных матриц и элементных данных
```C++
discr.SetMatRHSFunc(GenerateElemMatRhs(local_assembler, UNF, UNF));
discr.SetDataGatherer(local_data_gatherer);
```
- задаём тип используемой локальной нумерации степеней свободы
```C++
auto Var0Helper = GenerateHelper<UFem>();
FemExprDescr fed;
fed.PushVar(Var0Helper, "u");
fed.PushTestFunc(Var0Helper, "phi_u");
discr.SetProbDescr(std::move(fed));
```
- на основе заданных параметров подготавливаем некоторые внутренние данные ассемблера
```C++
discr.PrepareProblem();
```
8. Ассемблируем линейную систему, решаем её и сохраняем результат - строки 138-161 [ex1.cpp](../../examples/tutorials/ex1.cpp)
```C++
Sparse::Matrix A("A");
Sparse::Vector x("x"), b("b");
discr.Assemble(A, b);

Solver solver(INNER_ILU2);
solver.SetMatrix(A);
solver.Solve(b, x);

//copy result to the tag and save solution
discr.SaveVar(x, 0, u);
m->Save("out.pvtu");
```

### Общая реализация
Теперь продемонстрируем, как изменить предыдущий пример, чтобы он оставался валидным для любых скалярных КЭ пространств, см. пример [ex2.cpp](../../examples/tutorials/ex2.cpp)

TODO: привести список изменений


