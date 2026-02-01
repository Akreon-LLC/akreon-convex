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

#include "convex_set.hpp"

namespace convex {

/// @brief alias for domain on hyperplane
template <linalg::RnSpace Space>
using SimplexDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on hyperplane
template <linalg::RnSpace Space>
using SimplexCodom = SimplexDom<Space>;

/// @brief alias for meta on hyperplane
template <linalg::RnSpace Space>
using SimplexMeta = ConvexAtomMeta<SimplexDom<Space>, SimplexCodom<Space>,
                                   CurvatureType::CONVEX>;

/**
 * @brief Simplex definition
 *
 * S = { x | x >= 0, \sum x_i = 1 }
 *
 * Useful in probabilities
 */
template <typename Meta, typename Child>
struct Simplex : ConvexSet<Meta, Simplex<Meta, Child>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, Simplex<Meta, Child>, Child>;
  using ConvexBase::M;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Simplex(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief contains function to see if x is in the simplex
   *
   * @param point x
   * @param tolerance (scalar)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& tol) const {
    // (dakre) projected map would just be id
    T sum(0);
    for (size_t i = 0; i < N; ++i) {
      if (x(i) < T(0)) return false;
      sum += x(i);
    }
    return linalg::constexpr_abs(sum - 1) <= tol;
  }

  /**
   * @brief is_well_formed impl check
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const { return true; }

  /**
   * @brief transform simplex to polyhedron
   *
   * S = { x | x >= 0, sum_i x_i = 1 }
   * f(x) = U x + t
   * S' =  { y = f(x) | x \in S }
   * y = U x + t \to x = U^{-1} (y - t)
   *
   * Affine Inequality
   * A x <= b... A x <= 0... A U^{-1} y <= U^{-1} t
   * A = [ -U^{-1} ]
   * b = -U^{-1} t
   *
   * Linear Eqaulity
   * C x = d ... C = I and d = 1 in simplex
   * C U^{-1} y - U^{-1} t = d
   * C U^{-1} y = d + U^{-1} t
   *
   * C = [ 1^T U^{-1} ]
   * d = 1 + 1^T U^{-1} t
   *
   * @param AffineMap
   *
   * @return polyhedron
   */
  template <linalg::MatrixLike Mat>
    requires(std::same_as<T, linalg::mat_type_t<Mat>>)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto& U = map.linear_part();
    const auto& t = map.translation();
    const auto U_inv = linalg::inverse(U);

    const auto A = T(-1) * U_inv;
    const auto b = T(-1) * U_inv * t;

    using Dom = linalg::Matrix<T, 1, N>;
    using Codom = linalg::Matrix<T, 1, N>;

    const auto poly = Polyhedron<PolyhedronMeta<Dom, Codom>, Child>(
        A, b, std::get<0>(this->children));

    linalg::vec_traits<linalg::Vector<T, N>> vec;
    const auto ones = vec.fill(T(1));
    const auto one_T_U_inv = as_row_vector(ones) * U_inv;
    const auto C = as_vector(one_T_U_inv);
    const auto d = T(1) + (one_T_U_inv * t)(0);

    const auto hyperplane = Hyperplane<HyperplaneMeta<Dom>, Child>(
        C, d, std::get<0>(this->children));

    return Intersection<IntersectionMeta<Dom>, decltype(poly),
                        decltype(hyperplane)>(poly, hyperplane);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /*other*/) const {
    return true;
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K
   * C = { x | x_i >= 0 \forall i, sum(x_i) <= 1 }
   * K = { (x,t) | x_i >= 0 \forall i, 1^T x <= t, t >= 0 }
   *
   * 0 <= x... nonneg cone
   * 0 <= t - 1^T x... nonneg cone
   *
   * [ I    0 ] [ x; t ]
   * [ -1^T 1 ]
   */
  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_set(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::SET>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    static constexpr size_t ChildCodom =
        linalg::mat_cols_v<decltype(child_coord.linear_part())>;
    static_assert(ChildCodom == N,
                  "Convex set domain does not equal the child coordinate "
                  "linear map's codomain in size");

    linalg::Matrix<T, N + 1, N + 1> F;
    for (size_t i = 0; i < N; ++i) F(i)(i) = T(1);
    for (size_t i = 0; i < N; ++i) F(N)(i) = T(-1);
    F(N)(N) = T(1);
    linalg::Vector<T, N + 1> g;
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto rplus1 = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
    const auto rplus2 = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto cones = std::make_tuple(rplus1, rplus2);

    const auto new_builder =
        ConeBlockBuilder<builder_dom_t<decltype(builder)>, decltype(map),
                         decltype(child_cb)>(map, child_cb);
    return new_builder.template make<Type>(this->idx, this->lift_map, cones);
  }
};

}  // namespace convex
