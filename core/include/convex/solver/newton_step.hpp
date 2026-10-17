/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace convex {

/**
 * @brief Newton step bundle for the primal-dual interior-point iterate.
 */
template <linalg::VectorContainer Var, linalg::VectorContainer ConeVar,
          linalg::VectorContainer NuVar>
struct NewtonStep {
  Var pz{};     ///< decision-variable step
  NuVar pnu{};  ///< equality-multiplier step
  ConeVar px{}; ///< primal-cone step
  ConeVar py{}; ///< dual-cone step
};

} // namespace convex
