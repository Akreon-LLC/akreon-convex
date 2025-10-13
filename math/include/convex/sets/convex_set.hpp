/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <optional>

#include "../../algebra/algebra_util.hpp"
#include "../convex_traits.hpp"

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
template <linalg::VectorSpace Space, typename Derived>
  requires(!linalg::is_complex_v<linalg::space_type_t<Space>>)
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
  constexpr auto transform(const AffineMap<Mat>& map) const {
    // (0) if this is a cone an affine transform remains a cone (assuming
    // congruence transform)
    if constexpr (ConeLike<Derived>)
      return self();
    else
      return self().transform_impl(map);
  }

  /**
   * @brief project x \in dom(C) by linear fractional function
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
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  project(const ProjectiveMap<MatA, MatC>& map, const Point& x) const {
    // (0) Affine map of a cone will break cone property so no-op here for the
    // time being. Cones are homogenous sets anchored at the origin and don't
    // "move" with affine maps like other convex sets
    if constexpr (ConeLike<Derived>) {
      return std::nullopt;
    } else {
      // (1) if this is a convex set (not cone) then projection of point lies in
      // K if x is in C
      if (!contains(x)) return std::nullopt;
      return map.apply(x);
    }
  }

  /**
   * @brief to cone (apply inverse perspective func to convex set (non-cone))
   *
   * P^{-1}(C) -> K where P^{-1} : R^m -> R^{m + 1}
   *
   * @param t (slice defining cone)
   *
   * @return cone
   */
  constexpr auto to_cone(T t = 1.0) const {
    if constexpr (ConeLike<Derived>) {
      return std::nullopt;
    } else {
      return self().to_cone_impl(t);
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
};

}  // namespace convex
