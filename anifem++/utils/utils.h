#include <utility>
#include "inmost.h"
#include "anifem++/fem/fem_space.h"
#include "anifem++/fem/operations/operations.h"
#include "mesh_utils.h"
#include "anifem++/inmost_interface/assembler.h"

#ifndef ANIFEM_UTILS_UTILS_H
#define ANIFEM_UTILS_UTILS_H

/// Initialize inmost modules
void InmostInit(int* argc, char** argv[], const std::string& solver_db, int& pRank, int& pCount);
/// Finalize inmost modules
void InmostFinalize();

/// @brief Create unite 1D space from name
/// @param name is one of "P0", "P1", "P2", "P3", "MINI" (=P1+B4), "MINI1" (=P1+B4), "MINI2" (=P2+B4), "MINI3" (=P3+B4), "CR1", "MINI_CR" (=CR1+B4), "MINI_CR1" (=CR1+B4)
/// @return corresponding space
Ani::FemSpace choose_space_from_name(const std::string& name);

/// @brief Print status of last linear problem solution
void print_linear_solver_status(INMOST::Solver& s, const std::string& prob_name = "problem", bool exit_on_fail = false);

/// @brief Create a tag which is enough to store all mesh d.o.f.'s based on cell-local dof map
/// @param use_fixed_size if true then will be created fixed-size tag
/// @return created tag
INMOST::Tag createFemVarTag(INMOST::Mesh* m, const Ani::DofT::BaseDofMap& dofmap, const std::string& tag_name = "var", bool use_fixed_size = true);

/// @brief Compute action op on cell-local dof's corresponding to the variable component relative discr and problem tag problem_tag
/// @return result in out array
/// @warning out must have size at least op.Dim()
template<typename Traits>
void eval_op_var_at_point(double* out, INMOST::Cell c, std::array<double, 3> x_point, const Ani::ApplyOpBase& op, Ani::AssemblerT<Traits>& discr, INMOST::Tag problem_tag, const int* component = nullptr, unsigned int ncomp = 0);
/// @return action op on the variable component in array
/// @warning expected N >= op.Dim()
template<std::size_t N, typename Traits>
std::array<double, N> eval_op_var_at_point(INMOST::Cell c, std::array<double, 3> x_point, const Ani::ApplyOpBase& op, Ani::AssemblerT<Traits>& discr, INMOST::Tag problem_tag, const int* component = nullptr, unsigned int ncomp = 0);
template<std::size_t N, typename Traits>
std::array<double, N> eval_op_var_at_point(INMOST::Cell c, std::array<double, 3> x_point, const Ani::ApplyOpBase& op, Ani::AssemblerT<Traits>& discr, INMOST::Tag problem_tag, std::initializer_list<int> components = {});

/// @brief Compute approximate intergal of function over mesh domain.
/// @return \sum_{cell \in mesh} |cell| * \sum_{q \in quad_formula} w_q * f(cell, X_q)
template<int N = 1, typename FUNC>
std::array<double, N> integrate_vector_func(INMOST::Mesh* m, const FUNC& f, uint order = 5);

/// @brief Compute approximate intergal of scalar field over mesh domain.
template<typename FUNC>
double integrate_scalar_func(INMOST::Mesh* m, const FUNC& f, uint order = 5);

/// @brief Compute l2-projection with quadrature order 'order' of vector-field func defined on mesh on FEM space fem with vector of d.o.f.'s in tag res_tag.
/// @details Solve the problem \int_Omega f * v dX = \int_Omega f^h * v dX, f^h(X) = sum_i f_i * v_i(X)
void make_l2_project(const std::function<void(INMOST::Cell c, std::array<double, 3> X, double* res)>& func, INMOST::Mesh* m, INMOST::Tag res_tag, Ani::FemSpace fem, int order = 5);
/// @tparam FEMTYPE is type of FEM space of variable, e.g. FemVec<3, FEM_P2>
template<typename FEMTYPE>
void make_l2_project(const std::function<void(INMOST::Cell c, std::array<double, 3> X, double* res)>& func, INMOST::Mesh* m, INMOST::Tag res_tag, int order = 5);

/// @brief Create or load if exists fem-storage tag
/// @return pair of fem-storage tag and bool indicating on true if tag was loaded
std::pair<INMOST::Tag, bool> create_or_load_problem_tag(INMOST::Mesh *m, const Ani::DofT::BaseDofMap &dofmap, std::string var_name, bool load_if_exists = true);

#if defined(USE_MPI)
#define BARRIER MPI_Barrier(MPI_COMM_WORLD);
#else
#define BARRIER
#endif

#include "utils.inl"

#endif //ANIFEM_UTILS_UTILS_H