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

#include "convex/cone_traits.hpp"

namespace conic {

/// @brief cone base class
template <linalg::MatrixLike Mat>
struct NonnegOrthantCone : public Cone<Mat, NonnegOrthantCone<Mat>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Mat, NonnegOrthantCone<Mat>>;
  using typename ConeBase::Bias;

  /**
   * @brief compute barrier
   *
   * @return barrier for cone
   */
  constexpr auto barrier_impl(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }

  /**
   * @brief compute dual barrier
   *
   * @return dual barrier for cone
   */
  constexpr auto dual_barrier_impl(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }

  /**
   * @brief compute gradient
   *
   * @return gradient for cone
   */
  constexpr auto gradient_impl(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }

  /**
   * @brief compute hessian
   *
   * @return hessian for cone
   */
  constexpr auto hessian_impl(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }
};

}  // namespace conic
