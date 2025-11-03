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

#include "cone_view_base.hpp"

namespace convex {

/**
 * @brief Dual Cone View is another owning view onto the underlying cone K
 * but in the dual sense.
 *
 * K^* = { y | x^T y >= 0, x \in K }
 */
template <ConvexConeLike Cone>
struct DualConeView : ConeViewBase<DualConeView<Cone>> {
  /// @helpers from base class
  using ConeBase = ConeViewBase<DualConeView<Cone>>;
  using typename ConeBase::Point;
  using typename ConeBase::T;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit DualConeView(const Cone& cone_in) : ConeBase(cone_in) {}

  /**
   * @brief get dual view of cone
   *
   * K^** = K
   *
   * @return cone view
   */
  constexpr auto dual_impl(const Cone& cone) const {
    return ConeView<Cone>(cone);
  }
};

}  // namespace convex
