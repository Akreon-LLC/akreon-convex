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
#include "convex/cone_view.hpp"
#include "convex/convex_set_traits.hpp"
#include "linalg/symmetric_matrix.hpp"

namespace convex {

/**
 * @brief Base convex set class CRTP
 *
 *
 * IMPORTANT following Boyd's restriction to R^n space, perhaps
 * adapter or something to encode complex into reals for convex could be in
 * scope later on
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

  /// @brief alias for view tag
  static constexpr ViewTag Tag = set_view_tag_v<Derived>;

  /// @brief alias for coord map mat type
  using CoordMat = linalg::Matrix<T, N + 1, N + 1>;

  /**
   * @brief CTOR for convex set
   *
   * @param affine map to cone coordinates
   */
  constexpr explicit ConvexSet() : coord_map(affine::make_map_id<CoordMat>()) {}

  /**
   * @brief CTOR for convex set
   *
   * @param affine map to cone coordinates
   */
  constexpr explicit ConvexSet(const affine::AffineMap<CoordMat>& map_in)
      : coord_map(map_in) {}

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
  constexpr bool contains(const Point& x, T tol = 1e-8) const
    requires(Tag == ViewTag::ConvexSet)
  {
    return self().contains_impl(x, tol);
  }

  /**
   * @brief contains to check if the point lies in the convex cone
   *
   * When lifted to convex cone the pair (x,t) is checked to see if its
   * contained within the set
   *
   * @param (point, t) (n-dim vector) x R
   * @param optional tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const std::tuple<Point, T>& x, T tol = 1e-8) const
    requires(Tag != ViewTag::ConvexSet)
  {
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
    requires(Tag == ViewTag::ConvexSet)
  constexpr auto transform(const affine::AffineMap<Mat>& map) const {
    return self().transform_impl(map);
  }

  /**
   * @brief get cone view (apply inverse perspective func to convex set)
   *
   * P^{-1}(C) -> K where P^{-1} : R^m -> R^{m + 1}
   *
   * @return cone view
   */
  constexpr auto cone_view() const { return self().cone_view_impl(); }

  /**
   * @brief project (x,t) onto cone coordinates
   *
   * @param x_in (x,t) \in R^n x R to cone coords via coord map
   *
   * @return projected point
   */
  constexpr auto project(const std::tuple<Point, T>& x_in) const {
    if constexpr (std::same_as<Point, linalg::Vector<T, N>>) {
      auto x = std::get<0>(x_in);
      const auto t = std::get<1>(x_in);

      linalg::Vector<T, N + 1> x_cat;
      for (size_t i = 0; i < N; ++i) x_cat[i] = x(i);
      x_cat[N] = t;

      const auto x_proj = coord_map(x_cat);

      x = std::get<0>(linalg::splice<N>(x_proj));
      const auto s = x_proj(N);

      return std::tuple(x, s);
    } else {
      const auto U = coord_map.linear_part();
      const auto t = std::get<1>(x_in);
      return std::tuple(std::get<0>(x_in), t * U(N)(N));

      // TODO - get similarity transform working
#if 0
      auto x = std::get<0>(x_in);
      const auto t = std::get<1>(x_in);

      CoordMat A;
      A = linalg::set_block<0, N, 0, N>(A, x);
      A[N][N] = t;

      // (dakre) perhaps enforce congruence map for faster execution
      const auto UTAU = linalg::inverse(U) * (A * U);
      const auto x_prime = linalg::get_block<0, N, 0, N>(UTAU);
      const auto s = UTAU(N)(N);

      return std::tuple(x_prime, s);
#endif
    }
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

  /** @brief member vars */
  affine::AffineMap<CoordMat> coord_map;  //< map (x, t) \in R^n x R -> cone
};

}  // namespace convex
