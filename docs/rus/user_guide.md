Данная инструкция состоит из двух частей: в [первой части](#описание-anifem) приводится обзор основных элементов и возможностей AniFem++, а [вторая часть](#примеры-использования) представлена в виде набора постепенно усложняющихся примеров, которые призваны научить использовать данную библиотеку. Если вы были ранее знакомы с библиотекой [Ani3D/AniFem](https://sourceforge.net/projects/ani3d/), то можете начинать знакомство сразу с написания [кода](#первое-знакомство).

- [Описание возможностей AniFem++](#описание-возможностей-anifem)
  - [Квадратурные формулы](#квадратурные-формулы)
  - [Конечные элементы](#конечные-элементы)
  - [Элементная матрица](#элементная-матрица)
    - [Формы тензора D](#формы-тензора-d)
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
    - [Работа с памятью](#работа-с-памятью)
    - [КЭ пространства времени исполнения](#кэ-пространства-времени-исполнения)
    - [Потокобезопасная динамическая память](#потокобезопасная-динамическая-память)
  - [Стационарное уравнение реакции-диффузии](#стационарное-уравнение-реакции-диффузии)
  - [Стационарное уравнение Стокса](#стационарное-уравнение-стокса)
  - [Нестационарное уравнение конвекции-диффузии](#нестационарное-уравнение-конвекции-диффузии)

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

### Формы тензора D

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
```math
\begin{aligned} 
-\mathrm{div}\ (\mathbb{D}(\mathbf{x})\ \mathrm{grad}\ u) &= f(\mathbf{x})\ &in\ \ &\Omega,\\
u &= u_0(\mathbf{x})\ &on\ \ &\partial\Omega,
\end{aligned}
```
где $\mathbb{D}(\mathbf{x}) = (1 + x^2) \mathbb{I}$, $f(\mathbf{x}) = 1$, $u_0(\mathbf{x}) = 0$

### Немного теории
Слабая постановка имеет вид: *найти функцию* $u \in \mathring H^1_u = \{s \in H^1(\Omega): s|_{\partial \Omega} = 0 \}$ *, которая при любых соответствующих пространству тестовых функциях* $\phi$ *удовлетворяют следующим уравнениям:*

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

1. Необходимо сделать метки не только на граничных узлах, но также рёбрах и гранях - строки 51-62 [ex2.cpp](../../examples/tutorials/ex2.cpp)

В общем случае степени свободы могут лежать не толька на узлах, а значит и условия типа Дирихле тоже могут накладываться не только на узловые степени свободы

2. Также усложняется процедура для создания тега для хранения степеней свободы - строки 78-95 [ex2.cpp](../../examples/tutorials/ex2.cpp)

Приходится создавать тэг для данных перменной длины, т.к. INMOST не поддерживает создание тэга, имеющего разные фиксированные длины на разных геометрических элементах.

3. В пользовательскую структуру данных теперь надо добавить метки рёбер и граней - строки 97-103 [ex2.cpp](../../examples/tutorials/ex2.cpp)
```C++
struct ProbLocData{
    std::array<int, 4> nlbl = {0};
    std::array<int, 6> elbl = {0};
    std::array<int, 4> flbl = {0};
};
```
4. В функции ассемблирования элементных матриц для накладывания условия Дирихле теперь следует использовать интерефейс, который не зависит от структуры степеней свободы отдельного КЭ пространства - строки 121-136 [ex2.cpp](../../examples/tutorials/ex2.cpp)
```C++
auto& dat = *static_cast<ProbLocData*>(user_data);
// choose boundary parts of the tetrahedron 
DofT::TetGeomSparsity sp;
for (int i = 0; i < 4; ++i) if (dat.nlbl[i] > 0)
    sp.setNode(i);
for (int i = 0; i < 6; ++i) if (dat.elbl[i] > 0)
    sp.setEdge(i);  
for (int i = 0; i < 4; ++i) if (dat.flbl[i] > 0)
    sp.setFace(i);

//set dirichlet condition
if (!sp.empty()){
    std::array<double, UNF> dof_values;
    interpolateByDOFs<UFem>(XYZ, U_0, ArrayView<>(dof_values.data(), UNF), sp);
    applyDirByDofs(Dof<UFem>::Map(), A, F, sp, ArrayView<const double>(dof_values.data(), UNF));
}
```
5. И произвести соответствующие дополнения в функции сбора элементных данных - строки 138-161 [ex2.cpp](../../examples/tutorials/ex2.cpp)
```C++
auto local_data_gatherer = [&BndLabel](ElementalAssembler& p) -> void{
  double *nn_p = p.get_nodes();
  const double *args[] = {nn_p, nn_p + 3, nn_p + 6, nn_p + 9};
  ProbLocData data;
  auto geom_mask = Dof<UFem>::Map().GetGeomMask();
  // set labels if required
  if (geom_mask & DofT::NODE)
    for (unsigned i = 0; i < data.nlbl.size(); ++i) 
      data.nlbl[i] = (*p.nodes)[i].Integer(BndLabel);
  
  if (geom_mask & DofT::EDGE)
    for (unsigned i = 0; i < data.elbl.size(); ++i) 
      data.elbl[i] = (*p.edges)[i].Integer(BndLabel);
  
  if (geom_mask & DofT::FACE)
    for (unsigned i = 0; i < data.flbl.size(); ++i) 
      data.flbl[i] = (*p.faces)[i].Integer(BndLabel);
  
  p.compute(args, &data);
};
```

Наконец, всё готово! 

**Теперь вы можете поменять `using UFem = FemFix<FEM_P1>` на `using UFem = FemFix<FEM_P2>` или `using UFem = FemFix<FEM_P3>`, запустить перекомпиляцию программы и она будет корректно работать с новым пространством.**

### Работа с памятью 
При работе с КЭ пространствами, задаваемыми через шаблоны (т.е. известными на этапе времени компиляции), вообще говоря можно на этапе компиляции оценить количество памяти, необходимое для работы функций сборки элементных матриц. Однако иногда требуемое количество памяти может не поместиться в стек, т.е произойдёт его переполнение, и программа скорее всего упадёт. Поэтому мы ограничили выделяемую статическую память так, чтобы её хватало для работы с КЭ пространствами имеющими не более 30 степеней свободы (достаточно для $(P_2)^3$) и использующими квадратурные формулы с не более чем 24 точками (формулы 6-го порядка). Если в ваших приложениях могут использоваться более сложные квадратурные формули и более богатые КЭ пространства, то рекомендуется использовать варианты функций сборки матриц, которые работают на внешней памяти. 

Для примера использования шаблонного интерфейса с внешней памятью смотрите [ex3.cpp](../../examples/tutorials/ex3.cpp).

### КЭ пространства времени исполнения
Может случится так, что вы не знаете априори с каким КЭ пространством решение вашей задачи будет наиболее эффективно. В этом случае можно использовать интерфейс КЭ пространств, основанный на наследовании, который позволяет определять тип КЭ пространств на этапе исполнения. 

Для примера использования динамического интерфейса в приложении к рассматриваемой задаче смотрите [ex4.cpp](../../examples/tutorials/ex4.cpp).

### Потокобезопасная динамическая память
В некоторых приложениях оказывается, что заранее определить необходимый для сборки матриц объём памяти довольно проблематично. В этом случае можно воспользоваться блочно-линейным аллокатором памяти `DynMem`, который устроен так, что уже после сборки нескольких локальных матриц память фактически перестанет перевыделяться, а вместо этого будет переиспользоваться одни и тот же кусок памяти.

Для примера использования данного аллокатора в рассматриваемой задаче смотрите [ex5.cpp](../../examples/tutorials/ex5.cpp).

## Стационарное уравнение реакции-диффузии

```math
 \begin{aligned}
   -\mathrm{div}\ \mathbb{K}\ \mathrm{grad}\ u\ + A u      &= F  \ \ in\  \Omega  \\
                                                    u      &= u_0\   on\  \Gamma_D\\
    (\mathbb{K}\ \mathrm{grad}\ u) \cdot \mathbf{n}        &= g_0\   on\  \Gamma_N\\
    (\mathbb{K}\ \mathrm{grad}\ u) \cdot \mathbf{n}        &= 0\ \   on\  \Gamma_{N_0}\\
    (\mathbb{K}\ \mathrm{grad}\ u) \cdot \mathbf{n}\ + S u &= g_1\   on\  \Gamma_R\\
 \end{aligned},
```
где $\Omega = [0,1]^3$, $\Gamma_D = (\{0\} \cup \{1\}) \times [0,1]^2$, $\Gamma_N = [0,1]^2 \times \{0\}$, $\Gamma_R = [0,1]^2 \times \{1\}$, $\Gamma_{N_0} = \partial \Omega \backslash (\Gamma_D \cup \Gamma_R \cup \Gamma_N)$,

```math
\mathbb{K} = \begin{pmatrix} 1 & -1 & 0 \\ -1 & 1 & 0 \\ 0 & 0 & 1  \end{pmatrix},
```

- $A = 1$ 
- $S = 1$ 
- $F = (x+y+z)^2 - 2$ 
- $u_0(\mathbf{x}) = e^{z} + (x + y + z)^2$
- $g_0(\mathbf{x}) = -e^{z} - 2(x+y+z)$ 
- $g_1 = 2e^{z} + (x + y + z)(x+y+z+2)$

с аналитическим решением $u_{a} = e^{z} + (x+y+z)^2$.

Слабая поставновка имеет вид:
$$\int_{\Omega}(\mathbb{K}\ \mathrm{grad}\ u) \cdot \mathrm{grad}\ \phi\ d^3\mathbf{x} + \int_{\Omega} (A\ u) \cdot \phi\ d^3\mathbf{x} + \int_{\Gamma_R} (S\ u)\cdot \phi\ d^2\mathbf{x} = \int_{\Omega} f(\mathbf{x})\cdot \phi\ d^3\mathbf{x} + \int_{\Gamma_N} g_0 \cdot \phi\ d^2\mathbf{x} + \int_{\Gamma_R} g_1 \cdot \phi\ d^2\mathbf{x}$$

В файле [react_diff1.cpp](../../examples/tutorials/react_diff1.cpp) представлено решение этой задачи с использованием шаблонного интерфейса, а в [react_diff2.cpp](../../examples/tutorials/react_diff2.cpp) - с использованием runtime интрефейса.

## Стационарное уравнение Стокса
Данный пример демонстрирует работу с векторными КЭ пространствами и подход к КЭ дискретизации задач, содержащих несколько физических переменных.
```math
 \begin{aligned}
  -(\mathrm{div}\ \nu\ \mathrm{grad})\ &\mathbf{u}\ + &\mathrm{grad}\ p &= 0\   in\  \Omega  \\
    \mathrm{div}\               &\mathbf{u}    &                 &= 0\   in\  \Omega\\
                                &\mathbf{u}    &                 &= \mathbf{u}_0\ on\  \Gamma_1\\
                                &\mathbf{u}    &                 &= 0\   on\  \Gamma_2\\
            -(\mathbf{n}, \nu \nabla)\ & \mathbf{u}\ + &               p\ \mathbf{n} &= 0\   on\  \Gamma_3\\
  \end{aligned},
```
где $\Omega = [0,1]^3 \setminus ([0, 0.5]^2\times[0,1])$, $\Gamma_1 = \{0\}\times [0.5, 1]\times [0, 1]$, $\Gamma_3 = \{1\}\times [0,1]^2$, $\Gamma_2 = \partial \Omega \setminus (\Gamma_1 \cup \Gamma_3)$, $\nu = 1$, $\mathbf{u}_0 = (64(y-0.5)(1-y)z(1-z), 0, 0)$

Слабая постановка имеет вид ($\mathbf{u} \leftrightarrow \mathbf{\phi}$, $p \leftrightarrow q$):
$$J((\mathbf{u}, p), (\mathbf{\phi}, q)) = \int_{\Omega} (\nu \nabla_i u_j, \nabla_i \phi_j) - (p, \nabla_j \phi_j) - (\nabla_i u_i, q)\ d^3\mathbf{x} = 0$$
$$J((\mathbf{u}, p), (\mathbf{\phi}, q)) = \int_{\Omega} (\mathrm{grad}\ \mathbf{u})^T :  \mathrm{grad}\ \mathbf{\phi} - p\ \mathrm{div} \mathbf{\phi} - \mathrm{div}\ \mathbf{u}\ q\ d^3\mathbf{x} = 0$$


После дискретизации $\mathbf{u}^h = \sum\limits_{k = 1}^{K} u_k \mathbf{\phi}_k$, $p = \sum\limits_{l = 1}^{L} p_l q_l$, соответственно выражение конечно-элементной невязки принимает вид:

$$J^h_{k} = \int_{\Omega} \sum_m u_m [(\nu\ \mathrm{grad}\ \mathbf{\phi}_m)^T : \mathrm{grad}\ \mathbf{\phi}_k] - \sum_n p_n [q_n\ \mathrm{div}\ \mathbf{\phi}_k]$$

$$J^h_{K+l} = - \sum_m u_m [\mathrm{div}\ \mathbf{\phi}_m\ q_l]$$

Будем, для определённости, дискретизировать скорость $\mathbf{u}$ элементами типа $(P_2)^3$, а давление $p$ - элементами типа $P_1$. Тогда учитывая, что КЭ матрица (якобиан) определяется как $H = \frac{\partial \mathbf{J}^h}{\partial (u_1, \dots, u_K, p_1, \dots, p_L)}$, мы готовы записать вид элементной КЭ матрицы:
```math
\begin{aligned}
H &= \begin{pmatrix} A_{\nu} & -B\\ -B^T & 0\end{pmatrix}\\
A_{\nu} &= \int_{S^3} \nu\ GRAD((P_2)^3) \cdot GRAD((P_2)^3)\ d^3 \mathbf{x}\\ 
B &= \int_{S^3} IDEN(P_1) \cdot DIV((P_2)^3)\ d^3 \mathbf{x}
\end{aligned}
```

В файле [stokes.cpp](../../examples/tutorials/stokes.cpp) представлено решение этой задачи с использованием runtime интерфейса.

## Нестационарное уравнение конвекции-диффузии

Данный пример демонстрирует как отделить ассемблирование правой части и матрицы.

```math
\begin{aligned}
\frac{\partial u}{\partial t} -\mathrm{div}\ \mathbb{D}\ \mathrm{grad}\ u + \mathbf{v} \cdot \mathrm{grad}\ u &= f,\ in\ \Omega\\
u &= g,\ on\ \Gamma_1\\
u &= 0,\ on\ \Gamma_2\\
u|_{t = 0} &= 0,\ in\ \Omega
\end{aligned},
```
где 
- $\Omega = [0,1]^3$, $\Gamma_1 = \{0\}\times [0.25,0.75]^2$, $\Gamma_2 = \partial \Omega \setminus \Gamma_1$, $\Theta = [0, T]$, $T = 0.5$
- $\mathbb{D} = \mathrm{diag}(10^{-4}, 10^{-4}, 10^{-4})$
- $\mathbf{v} = (1, 0, 0)$
- $g = 1$, $f = 0$

Слабая постановка имеет вид:
$$\int_\Omega \left(\frac{\partial u}{\partial t}\ \phi + (\mathbb{D}\ \mathrm{grad}\ u) \cdot \mathrm{grad}\ \phi + (\mathbf{v} \cdot \mathrm{grad}\ u) \phi \right)\ d^3\mathbf{x} = \int_\Omega f \phi\ d^3\mathbf{x}$$


Далее будем использовать обозначение $(f, g) = \int_{\Omega} f g\ d^3\mathbf{x}$. Для дискретизации будем использовать $P_1$ конечные элементы, тогда после дискретизации имеем $u^h = \sum_{k = 1} u_k \phi$.

Для повышения устойчивости расчётов будем использовать SUPG (streamline upwind Petrov-Galerkin) стабилизирующий член. Он заключается в поэлементном добавлении в уравнение скалярного произведения невязки исходного уравнения и $\mathbf{v} \cdot \mathrm{grad}\ \phi_k$ c коэффициентами $\delta_c$, определяемыми для каждой ячейки $с$

```math
\begin{equation*}
\begin{split}
\left(\frac{\partial u^h}{\partial t},\ \phi_k\right)& + (\mathbb{D}\ \mathrm{grad}\ u^h,\ \mathrm{grad}\ \phi_k) + (\mathbf{v} \cdot \mathrm{grad}\ u^h,\ \phi_k) + \\
&+\sum_{c \in \Omega^h}\delta_c\left(\frac{\partial u^h}{\partial t} -\mathrm{div}\ \mathbb{D}\ \mathrm{grad}\ u^h + \mathbf{v} \cdot \mathrm{grad}\ u^h - f,\ \mathbf{v} \cdot \mathrm{grad}\ \phi_k\right)_c = (f, \phi_k)
\end{split}
\end{equation*}
```

где $\delta_c = \begin{cases}0.01, Pe_c \ge 1\\ 0,\ \textit{otherwise}\end{cases}$, $Pe_c = \frac{\mathrm{diam}(c) ||\mathbf{v}||}{\mathcal{D}_v}$ - сеточное число Пекле, $\mathcal{D}_v = \frac{(\mathbb{D} \mathbf{v}, \mathbf{v})}{(\mathbf{v}, \mathbf{v})}$

Заметим, что для $P_1$ дискретизации выражение $\mathrm{div}\ \mathbb{D}\ \mathrm{grad}\ u^h$ внутри стабилизирующей поправки обращается в ноль, поэтому далее это слагаемое не учитывается.

Для аппроксимации уравнения по времени воспользуемся схемой [BDF2](https://en.wikipedia.org/wiki/Backward_differentiation_formula):

$$\frac{\partial y}{\partial t} = f(t) \Rightarrow \frac{1.5 y^{n+1} - 2y^n + 0.5y^{n-1}}{\Delta t} = f^{n+1}$$

Вводём обозначения для некоторых элементных матриц и правых частей:
- $\mathcal{M}_{ij} = (\phi_j, \phi_i) + \sum_c \delta_c (\phi_j, v_k \nabla_k \phi_i)$
- $\mathcal{A}_{ij} = ( \mathbb{D}_{lk} \nabla_k \phi_j, \nabla_l \phi_i ) + ( v_k \nabla_k \phi_j, \phi_i ) + \sum_c \delta_c ( v_k \nabla_k \phi_j, v_l \nabla_l \phi_i )$
- $\mathcal{F}_i = (f, \phi_i) + \sum_c \delta_c (f, v_k \nabla_k \phi_i)$
- $U^{n}$ - вектор степеней свободы задачи

Тогда дискретная система уравнений принимает вид:
$$\left(\mathcal{M} + \frac{2}{3}\Delta t \mathcal{A}\right)\ U^{n+1} = \mathcal{M} \left(\frac{4}{3} U^{n} - \frac{1}{3}U^{n-1}\right) + \frac{2}{3}\Delta t \mathcal{F}$$

В файле [unsteady_conv_dif.cpp](../../examples/tutorials/unsteady_conv_dif.cpp) представлено решение этой задачи с использованием шаблонного интерфейса для $P_1$ элементов.

