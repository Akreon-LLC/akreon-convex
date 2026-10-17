/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
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
