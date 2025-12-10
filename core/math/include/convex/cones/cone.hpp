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
template <linalg::VectorLike Vec, typename Derived>
struct Cone {
  /// @brief alias for mat type
  using T = linalg::vec_type_t<Vec>;

  /// @brief helpers to get N sizes
  static constexpr size_t N = linalg::vec_size_v<Vec>;

  /**
   * @brief compute barrier
   *
   * @param slack vector (primal space)
   *
   * @return barrier for cone
   */
  constexpr auto barrier(const Vec& s) const { return self().barrier_impl(s); }

  /**
   * @brief compute barrier for dual
   *
   * @param dual vector (dual space)
   *
   * @return dual barrier
   */
  constexpr auto dual_barrier(const Vec& y) const {
    return self().dual_barrier_impl(y);
  }

  /**
   * @brief compute gradient
   *
   * @param slack vector (primal space)
   *
   * @return gradient for cone
   */
  constexpr auto gradient(const Vec& s) const {
    return self().gradient_impl(s);
  }

  /**
   * @brief compute gradient for dual
   *
   * @param dual vector (dual space)
   *
   * @return dual gradient
   */
  constexpr auto dual_gradient(const Vec& y) const {
    return self().dual_gradient_impl(y);
  }

  /**
   * @brief compute hessian
   *
   * @param slack vector (primal space)
   *
   * @return hessian for cone
   */
  constexpr auto hessian(const Vec& s) const { return self().hessian_impl(s); }

  /**
   * @brief compute dual hessian
   *
   * @param dual vector (dual space)
   *
   * @return hessian for dual cone
   */
  constexpr auto dual_hessian(const Vec& y) const {
    return self().hessian_impl(y);
  }

  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};

}  // namespace conic
