#ifndef ANIFEM_UTILS_ADAPT_MESH_H
#define ANIFEM_UTILS_ADAPT_MESH_H

#include "inmost.h"
#include "anifem++/fem/fem_space.h"

/**
 * Parameters for metric construction from a scalar FEM field.
 *
 * Typical input is an a-posteriori edge-bubble / energy-complement error estimator,
 * or a P2+ solution when distrib_heuristics is false.
 *
 * Output metric on nodes is stored as a symmetric 3x3 tensor in Voigt order
 * XX YY ZZ XY YZ XZ.
 */
struct MetricsConstructTraits {
    bool distrib_heuristics = true;      ///< split edge weights via (P.3.10); else use raw DOFs
    int  vertex_projection_strategy = 0; ///< 0: max-det; 1: geometric mean via log/exp
    int  max_quad_order = 5;             ///< upper bound on tet/face quadrature order

    /// Spectral bounds applied to |H| before forming M = (det|H|)^{-1/5}|H|
    double M_lambda_min = 1e-12;
    double M_lambda_max = 1e12;
    double M_lambda_max_rel = 1e2;       ///< max allowed lambda_max / lambda_min

    /// |H| is degenerate if lambda_min~0 or lambda_max/lambda_min > H_lambda_max_rel
    double H_lambda_max_rel = 1e7;
    double alpha_boost = 1.01;           ///< one-shot boost of max |d_k| when |H| is degenerate
    double isotropic_fallback = 1.0;     ///< M = c I when the whole mesh has zero H
    bool   verbosity = false;            ///< print mesh-wide statistics (rank 0)
};

/**
 * Build a continuous node metric from scalar FEM dofs in @p var.
 *
 * @param fem     scalar FemSpace of @p var (dim == 1); local DOF count is not fixed
 * @param var     FEM coefficient tag
 * @param metrics valid INMOST tag (passed by reference so CreateTag fallback updates the caller);
 *                if not defined on NODE (or fixed size < 6), a 6-component NODE tag with the same name
 *                is created via CreateTag; if size is ENUMUNDEF, RealArray on each node is resized to 6
 * @param traits  construction / clamping / verbosity options
 *
 * Cell and node loops are parallelized with OpenMP when WITH_OPENMP is enabled.
 */
void construct_metrics(Ani::FemSpace fem, INMOST::Tag var, INMOST::Tag& metrics,
                       MetricsConstructTraits traits = {});

#endif // ANIFEM_UTILS_ADAPT_MESH_H
