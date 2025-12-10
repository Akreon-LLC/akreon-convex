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

#include <optional>

#include "affine/affine_map.hpp"
#include "algebra/algebra_util.hpp"
#include "convex/cone_traits.hpp"
#include "convex/convex_set_traits.hpp"
#include "linalg/symmetric_matrix.hpp"

namespace convex {

/**
 * @brief Base convex set class CRTP
 *
 * Restricting domain to R^n for the time being
 *
 * Templated by a vector space and derived class
 */
template <linalg::RnSpace Space, typename Derived>
class ConvexSet {
 public:
  /// @brief alias to extract underlying scalar type of convex set
  using T = linalg::space_type_t<Space>;

  /// @brief alias for point
  using Point = linalg::space_point_t<Space>;

  /// @brief helper to extract dimension of convex set
  static constexpr size_t Dim = linalg::space_dim_v<Space>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

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
   * This is applied to convex sets like ellipsoid, polyhedron, simplex, etc...
   * its image is another convex set
   *
   * @param affine map
   *
   * @return transformed set
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform(const affine::AffineMap<Mat>& map) const {
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
   * @brief canonicalize
   *
   * P^{-1}(C) -> K 
   *
   * @return cone block
   */
  constexpr auto canonicalize() const { return self().canonicalize_impl(); }

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
