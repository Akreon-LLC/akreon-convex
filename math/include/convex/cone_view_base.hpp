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

#include "cone_traits.hpp"

namespace convex {

template <typename Derived>
struct ConeViewBase {
  /// @brief alias to extract underlying scalar type of convex set
  using T = cone_view_type_t<Derived>;

  /// @brief alias for point
  using Point = cone_view_point_t<Derived>;

  /// @brief alias for space underlying cone
  using Space = cone_view_space_t<Derived>;

  /// @brief alias for cone in view
  using Cone = cone_view_cone_t<Derived>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief CTOR
   *
   * @param ConvexSet argument
   */
  constexpr explicit ConeViewBase(const Cone& cone_in) : cone(cone_in) {}

  /**
   * @brief is proper check
   *
   * 1. K is convex (satisfied)
   * 2. K is closed (limit (x, t) \in K included)
   * 3. K is solid
   * 4. K is pointed x \in K -x \notin K except for 0 (K \intersect -K = 0)
   *
   * @return { False, True, Unknown }
   */
  constexpr ConeCheck is_proper() const {
    cone_traits<Cone> ct;
    const auto closed = ct.is_closed(cone);
    if (closed == ConeCheck::False) return ConeCheck::False;
    const auto solid = ct.is_solid(cone);
    if (solid == ConeCheck::False) return ConeCheck::False;
    const auto pointed = ct.is_pointed(cone);
    if (pointed == ConeCheck::False) return ConeCheck::False;

    if (closed == ConeCheck::Unknown || solid == ConeCheck::Unknown ||
        pointed == ConeCheck::Unknown)
      return ConeCheck::Unknown;

    return ConeCheck::True;
  }

  /**
   * @brief get dual view of cone
   *
   * @return dual cone view
   */
  constexpr auto dual_view() const { return self().dual_view_impl(cone); }

  /**
   * @brief self func to return dervied object instance
   *
   * return self
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }

  /** @brief member vars for cone */
  Cone cone;  //< Cone representation of underlying convex set
};

}  // namespace convex
