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
struct PowerCone : public Cone<Mat, PowerCone<Mat>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Mat, PowerCone<Mat>>;
  using typename ConeBase::Bias;

  /**
   * @brief compute barrier
   *
   * @return barrier for cone
   */
  constexpr auto barrier(const Mat& /*F*/, const Bias& /*g*/) const {
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
