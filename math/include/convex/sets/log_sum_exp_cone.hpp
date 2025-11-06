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
#include "exp_cone.hpp"

namespace convex {

/**
 * @brief Log Sum Exp cone
 *
 * C = { (x, t) | \sum_i e^{x_i - t} <= 1 }
 * C = k_1 x k_2 ... is the epigraph for log sum exp and is a cartesian product
 * of multiple exp cones
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::LiftedCone>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class LogSumExpCone : public ConvexSet<Space, LogSumExpCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, LogSumExpCone<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /**
   * @brief is_well_formed impl
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief check if point is in the set
   *
   * @param point in vector space R^n
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /*tol*/) const {
    return contains_impl(std::make_tuple(x, T(1)));
  }

  /**
   * @brief contains with (x,t) in cone (standard as epigraph)
   *
   * (1) Epigraph lift: C' = { (x,t) \in R^n x R | \sum_i e^{x_i - t} <= 1 }
   * {(x_i - t, 1, s_i) | e^{x_i - t} <= s_i, \sum_i s_i <= 1, \forall s_i >= 0}
   * (2) Hypograph lift: { (x,t) \in R^n x R | \sum_i e^{x_i + t} <= 1 } {
   * {(x_i + t, 1, s_i) | e^{x_i + t} <= s_i, \sum_i s_i <= 1, \forall s_i >= 0}
   *
   * @param (x,t) \in R^n x R_++ or R^n x R depending on lift
   * @param scalar tol
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& lifted_x,
                               const T& /*tol*/) const {
    const auto x = std::get<0>(lifted_x);
    const auto t = std::get<1>(lifted_x);

    T sum(0);
    if constexpr (Tag == ViewTag::HypographCone) {
      std::for_each(x.begin(), x.end(), [&sum, &t](const auto& pt) {
        sum += Exp<Space>()(pt + t, T(1)).value();
      });
    } else {
      std::for_each(x.begin(), x.end(), [&sum, &t](const auto& pt) {
        sum += Exp<Space>()(pt - t, T(1)).value();
      });
    }
    return sum <= T(1);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
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
      return ConeView(LogSumExpCone<Space, ViewTag::LiftedCone>());
    else
      return ConeView(LogSumExpCone<Space, Tag>());
  }

  /**
   * @brief project impl
   *
   * @param projection map
   * @para point in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  projectve_map_impl(const ProjectiveMap<MatA, MatC>& /* map */,
                     const Point& /* x */) const {
    return std::nullopt;
  }
};
}  // namespace convex
