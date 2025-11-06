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

/// @brief Exp Functor
template <linalg::RnSpace Domain>
  requires(linalg::space_dim_v<Domain> ==
           linalg::vec_size_v<linalg::space_point_t<Domain>>)
struct Exp {
  using T = linalg::space_type_t<Domain>;

  /**
   * @brief compute exp
   *
   * f(x,y) = y e^{x / y} s.t. y > 0
   *
   * @param scalar x
   * @param scalar y
   *
   * @return optional scalar \in R
   */
  constexpr std::optional<T> operator()(const T& x, const T& y) const {
    if (y <= T(0)) return std::nullopt;
    return y * linalg::constexpr_exp(x / y);
  }
};

/**
 * @brief exponential cone
 *
 * Standard Form
 * { (x, y, z) | y e^{x/y} <= z, y > 0 } \cup { (x, 0, z) | x <= 0, z >= 0 }
 *
 * Epigraph
 * { (x, y, z) | y e^{x/y} <= z, y > 0, z >= 0 }
 *
 * Hypograph with implicit slack on z
 * { (x, y, z) | y e^{x/y} <= -z, y > 0, z <= 0 }
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::space_dim_v<Space> ==
               linalg::vec_size_v<linalg::space_point_t<Space>> &&
           (linalg::space_dim_v<Space> == 1 || linalg::space_dim_v<Space> == 2))
class ExpCone : public ConvexSet<Space, ExpCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, ExpCone<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief exp cone \subset R^2
   *
   * @param Point \in R^2
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& tol) const {
    return contains_impl(std::tuple(x, T(1)), tol);
  }

  /**
   * @brief exp cone \subset R^2
   *
   * @param Point \in R^2 x R
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& /*tol*/) const {
    const auto x = std::get<0>(x_in);
    const auto t = std::get<1>(x_in);
    T y(1);
    if constexpr (Tag == ViewTag::EpigraphCone) {
      if (t < T(0)) return false;
    } else if constexpr (Tag == ViewTag::HypographCone) {
      if (t > T(0)) return false;
    } else {
      if (x(1) <= T(0)) {
        if (x(0) <= T(0) && t >= T(0))
          return true;
        else
          return false;
      }
      y = x(1);
    }
    const auto res = Exp<Space>()(x(0), y);
    if (!res.has_value()) return false;
    return res.value() <= t;
  }

  /**
   * @brief is well formed check
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief is equal impl
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(ExpCone<Space, ViewTag::LiftedCone>());
    else
      return ConeView(ExpCone<Space, Tag>());
  }
};

}  // namespace convex
