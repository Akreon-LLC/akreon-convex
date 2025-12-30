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

/// @brief alias for domain on halfspace
template <linalg::RnSpace Space>
using HalfspaceDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on halfspace
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HalfspaceCodom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for meta on halfspace
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using HalfspaceMeta = ConvexAtomMeta<HalfspaceDom<Dom>, HalfspaceCodom<Codom>,
                                     CurvatureType::CONVEX>;

/**
 * @brief Halfspace defined on a valid vector space and one of the following
 * type traits:
 *
 * {x | a^T x <= b}... closed lower halfspace
 * {x | a^T x < b}... open lower halfspace
 * {x | a^T x >= b}... closed upper halfspace
 * {x | a^T x > b}... open upper halfspace
 */
template <typename Meta, typename Child, HalfspaceType Type>
struct Halfspace : ConvexSet<Meta, Halfspace<Meta, Child, Type>, Child> {
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Meta, Halfspace<Meta, Child, Type>, Child>;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::N;

  /**
   * @brief CTOR for halfspace set
   *
   * @param Point (i.e., n-dim vector of type T)
   * @param b bias scalar
   * @param child
   */
  constexpr explicit Halfspace(const PointT& a_in, const T& b_in,
                               const Child& child_in)
      : ConvexBase(child_in), a(a_in), b(b_in) {}

  /**
   * @brief contains function to see if x is in the halfspace a^T x <= b
   *
   * Standard set: { x | a^T x <= b }... { x | a^T x - b <= 0 }
   *
   * @param point x of type T with n elements
   * @param tolerance of type T (ignored)
   *
   * @return T/F
   */
  constexpr bool contains_impl(const PointT& x, const T& /* tol */) const {
    switch (Type) {
      case HalfspaceType::UH:
        return linalg::dot(a, x) >= b;
      case HalfspaceType::LH:
      default:
        return linalg::dot(a, x) <= b;
    }
  }

  /**
   * @brief is_well_formed checks to see if underlying set is a valid convex
   * set
   *
   * @return T/F
   */
  constexpr bool is_well_formed() const {
    if (linalg::all_zero(a)) {
      if constexpr (Type == HalfspaceType::LH) {
        if (b < T(0)) return false;
      } else {
        if (b > T(0)) return false;
      }
    }
    return true;
  }

  /**
   * @brief transform halfspace by affine map
   *
   * H = { x | a^T x <= b }
   * f(H) = { y = f(x) | x \in H}... image of halfspace under f mapping
   * where affine map is: y = U x + t -> x = U^{-1} (y - t)
   * so, a^T U^{-1} (y - t) <= b \to a^T U^{-1} y <= b + a^T U^{-1} t
   * we arrive at: f(H) = { y | a^T U^{-1} y <= b + a^T U^{-1} t }
   *
   * @param affine map
   *
   * @return transformed halfspace
   */
  template <linalg::SquareMatrixLike Mat>
    requires(std::same_as<T, linalg::mat_type_t<Mat>> &&
             linalg::mat_cols_v<Mat> == N)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& map) const {
    const auto aT_U =
        transpose(as_col_vector(a)) * linalg::inverse(map.linear_part());
    return Halfspace<Meta, Child, Type>(as_vector(aT_U),
                                        b + (aT_U * map.translation())(0));
  }

  /**
   * @brief is_equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& other) const {
    return other.self().a == a && other.self().b == b;
  }

  /**
   * @brief canonicalize impl
   *
   * Apply P^{-1}(C) -> K and map K to cone block
   *
   * H = { x | a^T x <= b }
   * P^{-1}(H) = { (x,t) | a^T x <= b t, t >= 0 }
   *
   * [ a -b ] [ x; t ] = 0
   * [ 0  1 ]
   *
   * [a x - b t; t], where ax - bt <= 0, t >= 0
   * 0 <= bt - ax... nonneg cone
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    static_assert(Type != HalfspaceType::UH && Type != HalfspaceType::LH,
                  "cannot cannonicalize an open set -> non-proper cone");

    linalg::Matrix<T, 2, N + 1> F;
    F = linalg::set_block<0, 1, 0, N>(F, T(-1) * linalg::as_row_vector(a));
    F[0][N] = b;
    F[1][N] = T(1);
    linalg::Vector<T, 2> g;
    const auto map = affine::AffineMap<decltype(F)>(F, g);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    return conic::ConeBlock(map, rplus, rplus);
  }

 private:
  /** @brief member variables in the halfspace */
  PointT a;  //< n-dim vector of type T
  T b;       //< bias scalar of type T
};

}  // namespace convex
