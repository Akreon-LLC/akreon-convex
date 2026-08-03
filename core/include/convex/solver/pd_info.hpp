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

#include "convex/solver_traits.hpp"

namespace convex {

/**
 * @brief PrimalDualIterate
 */
template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar>
struct PrimalDualIterate {
  Var z{};     ///< decision var in Rn + aux lifts
  ConeVar x{}; ///< primal cone var in K
  ConeVar y{}; ///< dual cone var in K^*
  NuVar nu{};  ///< lagrangian multiplier on equality constraints
};

} // namespace convex
