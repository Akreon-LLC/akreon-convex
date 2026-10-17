/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/linalg/numerical_config.hpp"
#include "akreon/linalg/vector/algebra_util.hpp"

namespace config {
/**
 * (dakre) note increase constexpr steps / depth for # steps to increase here
 * for constexpr usage
 */
#ifndef PD_IPM_STEPS
#define PD_IPM_STEPS 32
#endif

/**
 * @brief RegularizationConfig
 */
template <linalg::Real T>
struct RegularizationConfig {
  T global_eps{1e-12}; ///< tikhonov reg applied to full schur kkt
  T eq_eps{1e-8};      ///< tikhonov reg applied to only weak equality dir
};

/**
 * @brief PrimalDualIPMConfig
 */
template <linalg::Real T>
struct PrimalDualIPMConfig {
  std::size_t steps{PD_IPM_STEPS}; ///< total number of iter
  std::size_t backtrack_steps{64}; ///< total number of backtracking steps
  T primal_feas_tol{1e-5};         ///< tolerance for primal feasbility
  T dual_feas_tol{1e-5};           ///< tolerance for dual feasbility
  T duality_gap_tol{1e-5};         ///< tolerance for complementarity
  T Delta{2 * linalg::ln2_v<T>};   ///< relative neighborhood around F(Beta)
  FactorConfig<T> schur{.tol = T(1e-12),
                        .floor_eps = T(1e-14)}; ///< Schur Factor Config
  RegularizationConfig<T> reg;                  ///< Regularization config
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

} // namespace config
