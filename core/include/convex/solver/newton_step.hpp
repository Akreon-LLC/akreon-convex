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
