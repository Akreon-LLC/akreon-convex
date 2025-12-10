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

#include "cone.hpp"

namespace conic {

/// @brief exp cone class
template <linalg::VectorLike Vec>
struct LogDetCone : public Cone<Vec, LogDetCone<Vec>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Vec, LogDetCone<Vec>>;

  /**
   * @brief compute barrier
   *
   * @param slack vector (primal space)
   *
   * @return barrier for cone
   */
  constexpr auto barrier_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute dual barrier
   *
   * @param dual vector (dual space)
   *
   * @return dual barrier for cone
   */
  constexpr auto dual_barrier_impl(const Vec& /*y*/) const { return false; }

  /**
   * @brief compute gradient
   *
   * @param slack vector (primal space)
   *
   * @return gradient for cone
   */
  constexpr auto gradient_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute gradient
   *
   * @param dual vector (dual space)
   *
   * @return gradient for dual cone
   */
  constexpr auto dual_gradient_impl(const Vec& /*y*/) const { return false; }

  /**
   * @brief compute hessian
   *
   * @param slack vector (primal space)
   *
   * @return hessian for cone
   */
  constexpr auto hessian_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute hessian
   *
   * @param dual vector (dual space)
   *
   * @return hessian for dual cone
   */
  constexpr auto dual_hessian_impl(const Vec& /*y*/) const { return false; }
};

}  // namespace conic
