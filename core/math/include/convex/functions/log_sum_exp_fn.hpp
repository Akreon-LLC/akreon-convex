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

#include "convex_fn.hpp"

namespace convex {

/**
 * @brief log sum exp = smooth max function
 *
 * lse(x) = log(\sum_i^n(e^{x_i})) which is a smooth mapping from R^n to R
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class LogSumExpFn : public ConvexFn<Domain, LogSumExpFn<Domain, Tag>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, LogSumExpFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief alias for affine map (codomain 1-dim)
  using Mat = linalg::Matrix<T, 1, N>;

  /**
   * @brief eval impl
   *
   * @param point x \in dom f
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    T sum(0);
    std::for_each(x.begin(), x.end(),
                  [&](const auto& pt) { sum += linalg::constexpr_exp(pt); });

    const auto res = linalg::constexpr_ln(sum);
    if constexpr (Tag == FnTag::Convex)
      return res;
    else {
      if (!res.has_value()) return std::nullopt;
      return T(-1) * res.value();
    }
  }

  /**
   * @brief negate func
   *
   * @return negate f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Convex)
      return LogSumExpFn<Domain, FnTag::Concave>();
    else
      return LogSumExpFn<Domain, FnTag::Convex>();
  }

  /**
   * @brief canonicalize impl
   *
   * (1) f(x) = log(sum_i(exp(x_i)))
   * K = { (x, t) | log(sum_i(exp(x_i))) <= t }
   * K = { (x, t) | sum_i(exp(x_i)) <= exp(t) }
   * K = { (x, t) | sum_i(exp(x_i)) / exp(t) <= 1 }
   * K = { (x, t) | sum_i(exp(x_i - t)) <= 1 }
   * K = { (x, t, s) | exp(x_i - t) <= s_i, 1^T s <= 1 }
   *
   * Exp cone : { (u, v, w) | v exp(u/v) <= w, v > 0 } \cup { (u, 0, w) | u <=
   * 0, w >= 0 }
   *
   * let u = x_i - t, v = 1, w = s_i... n exp cones
   * 0 <= 1 - 1^T s ... 1 nonneg orthant cone
   *
   * [ A -1    0 ] [ x; t; s ] + [ b; 1; 0 | 1 ] -> (x_i - t, 1, s_i) + rplus
   * [ 0  0    0 ]
   * [ 0  0    I ]
   * -------------
   * [ 0  0 -1^T ]
   *
   * (2) -f(x) = -log(sum_i(exp(x_i)))
   * K = { (x, t) | sum_i(exp(x_i + t)) <= 1 }
   * K = { (x, t, s) | exp(x_i + t) <= s_i, 1^T s <= 1 }
   *
   * let u = x_i + t, v = 1, w = s_i... n exp cones
   * 0 <= 1 - 1^T s ... 1 nonneg orthant cone
   *
   * [ A  1    0 ] [ x; t; s ] + [ b; 1; 0 | 1 ] -> (x_i + t, 1, s_i) + rplus
   * [ 0  0    0 ]
   * [ 0  0    I ]
   * -------------
   * [ 0  0 -1^T ]
   *
   * @param cone map builder
   * @param coord map
   * @param cone block
   *
   * @return cone block
   */
  template <linalg::RnSpace DecVarDom, affine::AffineMapLike CoordMap,
            affine::AffineMapLike ConeMap, conic::ConeLike... Cs>
  constexpr auto canonicalize_impl(
      const ConeMapBuilder<DecVarDom>& cmap_builder, const CoordMap& coord_map,
      const conic::ConeBlock<ConeMap, Cs...>& cone_block) const {
    using AffineMatT = affine::affine_mat_t<CoordMap>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == N,
                  "input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");
    linalg::Matrix<T, 2 * N + 2, 2 * Dom + 1> F;
    linalg::Vector<T, 2 * N + 2> g;
    F = linalg::set_block<0, N, 0, Dom>(F, coord_map.linear_part());
    const auto I = linalg::make_identity<T, Dom>();

    if constexpr (Tag == FnTag::Convex) {
      for (size_t i = 0; i < N; ++i) F[i][N] = T(-1);
    } else {
      for (size_t i = 0; i < N; ++i) F[i][N] = T(1);
    }
    F = linalg::set_block<N + 1, 2 * N + 1, N + 1, 2 * N + 1>(F, I);
    for (size_t i = 0; i < N; ++i) F[2 * N + 1][N + 1 + i] = T(-1);
    for (size_t i = 0; i < N; ++i) g[i] = coord_map.translation()(i);
    g[N] = T(1);
    g[2 * N + 1] = T(1);

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto map_res = cmap_builder(coord_map_up, cone_block.map, -2);
    const auto emitted_map = std::get<0>(map_res);
    const auto map = std::get<1>(map_res);

    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 2 * N + 1>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    return std::tuple(
        emitted_map,
        conic::ConeBlock(map,
                         std::tuple_cat(cone_block.cones,
                                        std::make_tuple(exp_cone, rplus))));
  }
};

}  // namespace convex
