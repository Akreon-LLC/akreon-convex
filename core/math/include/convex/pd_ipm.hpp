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

#include "convex_problem.hpp"
#include "linalg/operator/kkt_builder.hpp"

namespace convex {

/**
 * @brief Primal Dual Interior Point Method Solver
 */
struct PrimalDualIPM {
  /**
   * @brief solve
   *
   * @param prob ConvexProblem
   *
   * @return result
   */
  template <BuilderLike Builder>
  constexpr auto solve(
      const ConvexProblem<Builder, StageType::PROBLEM_SET>& prob) const {
    if (!prob.validate()) return false;
    return true;
  }
};

}  // namespace convex
