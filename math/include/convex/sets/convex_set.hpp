/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <optional>

#include "../../algebra/algebra_util.hpp"
#include "../cone_view.hpp"
#include "../convex_set_traits.hpp"

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

  /// @brief helper to extract dimension of convex set
  static constexpr size_t Dim = linalg::space_dim_v<Space>;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<linalg::space_point_t<Space>>;

  /// @brief alias for point
  using Point = linalg::space_point_t<Space>;

  /// @brief alias for view tag
  static constexpr ViewTag Tag = set_view_tag_v<Derived>;

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
  constexpr auto transform(const AffineMap<Mat>& map) const {
    return self().transform_impl(map);
  }

  /**
   * @brief projective_map x \in dom(C) by linear fractional function
   *
   * Q(x) -> [z; t] \in K (K = cone in N+1 dim space formed from inverse
   * perspective function)
   * [z; t] \in P^{-1}(C) \equiv K
   *
   * AKA perspective function composed with an affine func: x_proj = g(f(x))
   *
   * @param projective map
   * @param point in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
    requires(Tag == ViewTag::ConvexSet)
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  projective_map(const ProjectiveMap<MatA, MatC>& map, const Point& x) const {
    return self().projective_map_impl(map, x);
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
