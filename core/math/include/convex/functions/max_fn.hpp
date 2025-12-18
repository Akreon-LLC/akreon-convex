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
 * @brief max convex function CRTP
 *
 * F(x) = max { x_1, x_2, ... x_n }, x \in R^n
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Convex>
class MaxFn : public ConvexFn<Domain, MaxFn<Domain, Tag>> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, MaxFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param Point \in domain
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    T mx(0);
    std::for_each(x.begin(), x.end(), [&mx](const auto& pt) {
      if (mx < pt) mx = pt;
    });
    if constexpr (Tag == FnTag::Convex)
      return mx;
    else
      return T(-1) * mx;
  }

  /**
   * @brief negate impl
   *
   * if concave -> convex
   * else convex -> concave
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return MaxFn<Domain, FnTag::Convex>();
    else
      return MaxFn<Domain, FnTag::Concave>();
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = max{x_1, ... x_n}
   * K = { (x,t) | max(x) <= t }
   *
   * x_i <= t forall i convex constraints
   * 0 <= t - x_i... in nonneg orthant cone
   *
   * [ -A 1 ] [ x ; t ] + [ -b ; 0 ] -> (-(A x + b), t)
   *
   * (2) f(x) = -max{x_1, ..., x_n}
   * K = { (x,t) | -max(x) >= t }
   *
   * -x_i >= t for all i convex constraints
   *  0 <= -t - x_i... in nonneg orthant cone
   *
   * [ -A -1 ] [ x; t ] + [ -b ; 0 ] -> (-(A x + b), -t)
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
    linalg::Matrix<T, N, Dom + 1> F;
    F = linalg::set_block<0, N, 0, Dom>(F, T(-1) * coord_map.linear_part());

    linalg::Vector<T, N> g;
    for (size_t i = 0; i < N; ++i) g[i] = T(-1) * coord_map.translation()(i);

    if constexpr (Tag == FnTag::Convex) {
      for (size_t i = 0; i < N; ++i) F[i][N] = T(1);
    } else {
      for (size_t i = 0; i < N; ++i) F[i][N] = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
    const auto emitted_map = std::get<0>(map_res);
    const auto map = std::get<1>(map_res);

    std::array<conic::NonnegOrthantCone<linalg::Vector<T, 1>>, N> arr;
    const auto rpluses = linalg::array_to_tuple(arr);
    return std::tuple(
        emitted_map,
        conic::ConeBlock(map, std::tuple_cat(cone_block.cones, rpluses)));
  }
};

}  // namespace convex
