/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/algebra_util.hpp"
#include "convex_traits.hpp"

namespace convex {

/**
 * @brief Base convex set class CRTP
 *
 * Templated by a vector space and derived class
 */
template <linalg::VectorSpace Space, typename Derived>
class ConvexSet {
 public:
  /// @brief alias to extract underlying scalar type of convex set
  using T = linalg::space_type_t<Space>;

  /// @brief alias for point
  using Point = linalg::space_point_t<Space>;

  /// @briaf alias for scalar type of undering set type
  using ScalarT = linalg::scalar_type_t<T>;

  /// @brief helper to extract dimension of convex set
  static constexpr size_t Dim = linalg::space_traits<Space>::Dim;

  /**
   * @brief is_well_formed
   *
   * Check to see if the underlying convex set is valid
   *
   * @return T/F
   */
  constexpr auto is_well_formed() const { return self().is_well_formed_impl(); }

  /**
   * @brief contains to check if the point lies in the convex set
   *
   * @param point (n-dim vector)
   * @param optional tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const Point& x, T tol = 1e-8) const {
    return self().contains_impl(x, tol);
  }

  /**
   * @brief transform convex set by affine map
   *
   * @param affine map
   *
   * @return transformed set
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform(const AffineMap<Mat>& map) const {
    return self().transform_impl(map);
  }

  /**
   * @brief equality comparison overload
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool operator==(const ConvexSet<Space, Derived>& other) const {
    return self().is_equal_impl(other);
  }

  /**
   * @brief inequality comparison overload
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool operator!=(const ConvexSet<Space, Derived>& other) const {
    return !self().is_equal_impl(other);
  }

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
