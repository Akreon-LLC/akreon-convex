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

#include "config/assert.hpp"

#include "affine/affine_map.hpp"
#include "convex/cone_traits.hpp"
#include "convex/convex_traits.hpp"
#include "linalg/vector/algebra_util.hpp"

namespace convex {

/**
 * @brief Base convex set class CRTP
 *
 * Restricting domain to R^n for the time being
 *
 * Templated by a vector space and derived class
 */
template <typename Meta, typename Derived, typename Child>
struct ConvexSet : Atom<Meta, ConvexSet<Meta, Derived, Child>, Child> {
  /// @brief alias to extract type traits from Domain convex set
  using BaseT = Atom<Meta, ConvexSet<Meta, Derived, Child>, Child>;
  using typename BaseT::PointT;
  using typename BaseT::T;

  static_assert(is_var_type_v<Child>,
                "Convex set assumes variable in its domain");

  /**
   * @brief CTOR
   *
   * @param children
   */
  constexpr explicit ConvexSet(const Child& child_in)
      : BaseT(std::make_tuple(child_in)) {}

  /**
   * @brief is_well_formed
   *
   * Check to see if the underlying convex set is valid
   *
   * @return T/F
   */
  constexpr void validate_domain_impl() const {
    AK_ASSERT(self().is_well_formed_impl() &&
           "ConvexSet::validate_domain_impl expects set it to be good");
  }

  /**
   * @brief contains to check if the point lies in the convex set
   *
   * @param point (n-dim vector)
   * @param optional tolerance
   *
   * @return T/F
   */
  constexpr bool contains(const PointT& x, T tol = 1e-8) const {
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
  constexpr bool operator==(
      const ConvexSet<Meta, Derived, Child>& other) const {
    return self().is_equal_impl(other);
  }

  /**
   * @brief inequality comparison overload
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool operator!=(
      const ConvexSet<Meta, Derived, Child>& other) const {
    return !self().is_equal_impl(other);
  }

  /**
   * @brief canonicalize
   *
   * P^{-1}(C) -> K
   *
   * @return cone block
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    return self().template canonicalize_set<Type>(builder);
  }

  /**
   * @brief self function which returns a const ref to the derived object
   *
   * @return const reference to the derived object
   */
  constexpr auto self() const { return static_cast<const Derived&>(*this); }
};

}  // namespace convex
