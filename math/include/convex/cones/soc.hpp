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

/// @brief soc cone class
template <linalg::VectorLike Vec>
struct SecondOrderCone : public Cone<Vec, SecondOrderCone<Vec>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Vec, SecondOrderCone<Vec>>;

  /**
   * @brief compute barrier
   *
   * @return barrier for cone
   */
  constexpr auto barrier_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute dual barrier
   *
   * @return dual barrier for cone
   */
  constexpr auto dual_barrier_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute gradient
   *
   * @return gradient for cone
   */
  constexpr auto gradient_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute gradient
   *
   * @return gradient for dual cone
   */
  constexpr auto dual_gradient_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute hessian
   *
   * @return hessian for cone
   */
  constexpr auto hessian_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute hessian
   *
   * @return hessian for dual cone
   */
  constexpr auto dual_hessian_impl(const Vec& /*s*/) const { return false; }
};

}  // namespace conic
