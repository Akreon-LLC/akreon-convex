/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include "linalg/vector/algebra_traits.hpp"
#include "linalg/vector/algebra_util.hpp"

namespace config {
/**
 * (dakre) note increase constexpr steps / depth for # steps to increase here
 * for constexpr usage
 */
#define PD_IPM_STEPS 32

/**
 * @brief FactorConfig
 */
template <linalg::RealLike T>
struct FactorConfig {
  size_t steps{64};    ///< total number of iter
  T tol{1e-8};         ///< tolerance of of solution
  T floor_eps{1e-10};  ///< machine precision (lower bounds)
};

/**
 * @brief RegularizationConfig
 */
template <linalg::RealLike T>
struct RegularizationConfig {
  T global_eps{1e-12};  ///< tikhonov reg applied to full schur kkt
  T eq_eps{1e-8};       ///< tikhonov reg applied to only weak equality dir
};

/**
 * @brief PrimalDualIPMConfig
 */
template <linalg::RealLike T>
struct PrimalDualIPMConfig {
  size_t steps{PD_IPM_STEPS};     ///< total number of iter
  size_t backtrack_steps{64};     ///< total number of backtracking steps
  T primal_feas_tol{1e-5};        ///< tolerance for primal feasbility
  T dual_feas_tol{1e-5};          ///< tolerance for dual feasbility
  T duality_gap_tol{1e-5};        ///< tolerance for complementarity
  T Delta{2 * linalg::ln2_v<T>};  ///< relative neighborhood around F(\Beta)
  FactorConfig<T> schur{.tol = T(1e-12),
                        .floor_eps = T(1e-14)};  ///< Schur Factor Config
  RegularizationConfig<T> reg;                   ///< Regularization config
};

/**
 * @brief SolverResult
 */
enum struct SolverResult : uint8_t {
  INFEASIBLE_OR_UNBOUNDED = 0,
  OPTIMAL,
  PRIMAL_FEASIBLE,
  DUAL_FEASIBLE,
  PRIMAL_DUAL_FEASIBLE,
  PRIMAL_OPTIMAL,
  DUAL_OPTIMAL,
  INFEASIBLE_OPTIMAL
};

}  // namespace config

namespace linalg {

/**
 * @brief NumericalEvent
 */
enum struct NumericalEvent : uint8_t {
  NONE = 0,

  // detection / observation
  RANK_DEFICIENCY,
  INDEFINITE_FACTORIZATION,
  NEAR_SINGULAR_TRIANGULAR_SOLVE,
  NUMERICAL_SAFEGUARD,
  DB_MAX_ITERS_REACHED,

  // failure
  FACTORIZATION_FAILED,
  EVD_FAILED
};

/**
 * @brief is_fatal_event
 *
 * @param event (NumericalEvent)
 *
 * @return T/F
 */
inline constexpr bool is_fatal_event(const NumericalEvent& event) {
  return event == NumericalEvent::FACTORIZATION_FAILED ||
         event == NumericalEvent::EVD_FAILED;
}

}  // namespace linalg
