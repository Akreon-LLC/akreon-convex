/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/algebra_util.hpp"
#include "convex_traits.hpp"

namespace convex {

template <linalg::VectorSpace Space, typename Derived>
class ConvexSet {
 public:
  /// @brief alias to extract underlying scalar type of convex set
  using T = linalg::space_type_t<Space>;

  /// @brief alias for point
  using Point = linalg::space_point_t<Space>;

  /// @brief helper to extract dimension of convex set
  static constexpr size_t Dim = linalg::space_traits<Space>::Dim;

  /**
   * @brief is_valid
   *
   * Check to see if the underlying convex set is valid
   *
   * @return T/F
   */
  constexpr auto is_valid() const { return true; }

  /**
   * @brief contains to check if the point lies in the convex set
   *
   * @param point (n-dim vector)
   * @param optional tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const Point& x, T tol = 1e-8) const {
    (void)x;
    (void)tol;
    return true;
  }

  /**
   * @brief convert convex set to conic form for solver
   *
   * @return conic form
   */
  constexpr auto to_conic_form() const { return true; }

 private:
  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }
};

}  // namespace convex
