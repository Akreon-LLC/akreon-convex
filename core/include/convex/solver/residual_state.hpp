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
 * @brief Residual state bundle for a PD-IPM iterate.
 */
template <linalg::VectorLike Var, linalg::VectorLike ConeVar,
          linalg::VectorLike NuVar>
struct ResidualState {
  using T = linalg::vec_type_t<Var>;

  Var rd{};       ///< dual residual
  ConeVar rp{};   ///< primal residual
  NuVar req{};    ///< equality residual
  T rd_norm{};    ///< ||rd||_2
  T rp_norm{};    ///< ||rp||_2
  T req_norm{};   ///< ||req||_2
};

}  // namespace convex
