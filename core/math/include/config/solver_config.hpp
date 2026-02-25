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

namespace config {
/**
 * (dakre) note increase constexpr steps / depth for # steps to increase here
 * for constexpr usage
 */

/**
 * @brief FactorConfig
 */
template <linalg::FieldLike T>
struct FactorConfig {
  size_t steps{64};  ///< total number of iter
  T tol{1e-8};       ///< tolerance of of solution
};

/**
 * @brief PrimalDualIPMConfig
 */
template <linalg::FieldLike T>
struct PrimalDualIPMConfig {
  size_t steps{32};         ///< total number of iter
  T primal_feas_tol{1e-5};  ///< tolerance for primal feasbility
  T dual_feas_tol{1e-5};    ///< tolerance for dual feasbility
  T duality_gap_tol{1e-4};  ///< tolerance for complementarity
};

/**
 * @brief SolverResult
 */
enum struct SolverResult : uint8_t {
  INVALID = 0,
  OPTIMAL,
  PRIMAL_FEASIBLE,
  DUAL_FEASIBLE,
  PRIMAL_DUAL_FEASIBLE,
  INFEASIBLE_OR_UNBOUNDED
};

}  // namespace config
