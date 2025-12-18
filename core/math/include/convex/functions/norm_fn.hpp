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
 * @brief norm convex function wrapper
 */
template <linalg::RnSpace Domain, linalg::NormLike Norm,
          FnTag Tag = FnTag::Convex>
  requires(linalg::space_dim_v<Domain> == linalg::norm_size_v<Norm>)
class NormFn : public ConvexFn<Domain, NormFn<Domain, Norm, Tag>> {
 public:
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Domain, NormFn<Domain, Norm, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief eval impl
   *
   * @param point x \in domain
   *
   * @return y (scalar \in R)
   */
  constexpr std::optional<T> eval_impl(const Point& x) const {
    if constexpr (Tag == FnTag::Convex)
      return Norm()(x);
    else
      return T(-1) * Norm()(x);
  }

  /**
   * @brief negate impl
   *
   * if concave tag -> convex
   * else convex tag -> concave
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return NormFn<Domain, Norm, FnTag::Convex>();
    else
      return NormFn<Domain, Norm, FnTag::Concave>();
  }

  /**
   * @brief canonicalize_impl
   *
   * L2 Norm Function
   * (1) f(x) = ||x||_2
   * K = { (x, t) | ||x||_2 <= t }
   * K_soc = { (u, v) | ||u||_2 <= v, v >= 0 }
   *
   * let u = x, v = t
   * [ A 0 ] [ x; t ] + [ b; 0 ] -> (A x + b, t)
   * [ 0 1 ]
   *
   * (2) -f(x) = -||x||_2
   * K = { (x, t) | -||x||_2 >= t }... { (x, t) | ||x||_2 <= -t }
   *
   * [ A  0 ] [ x; t ] + [ b; 0 ] -> (A x + b, -t)
   * [ 0 -1 ]
   *
   * Linf Norm Function
   * (1) f(x) = ||x||_inf
   * K = { (x, t) | ||x||_inf <= t }
   *
   * -t <= x_i <= t... \forall i
   * 0 <= x_i + t... nonneg orthant cone(s)
   * 0 <= t - x_i... nonneg orthant cone(s)
   *
   * [  A  1 ] [ x; t ] + [ b; -b ]
   * [ -A  1 ]
   *
   * (2) -f(x) = -||x||_inf
   * K = { (x, t) | -||x||_inf >= t }... { (x, t) | ||x||_inf <= -t }
   *
   * [  A  -1 ] [ x; t ] + [ b; -b ]
   * [ -A  -1 ]
   *
   * L1 Norm Function
   * (1) f(x) = ||x||_1
   * K = { (x, t) | ||x||_1 <= t }
   *
   * let u_i >= |x_i|... convex constraint
   * sum(u_i) <= t... convex constraint
   *
   * -u_i <= y_i <= u_i
   * 0 <= y_i + u_i... nonneg orthant cone
   * 0 <= u_i - y_i... nonneg orthant cone
   * 0 <= t - 1^T u... nonneg orthant cone
   *
   * [ A  I   0 ] [ x; u; t ] + [ b; -b; 0 ]
   * [ -A I   0 ]
   * [ 0 -1^T 1 ]
   *
   * (2) -f(x) = -||x||_1
   * K = { (x, t) | -||x||_1 >= t }... { (x, t) | ||x||_1 <= -t }
   *
   * [ A  I    0 ] [ x; u; t ] + [ b; -b; 0 ]
   * [ -A I    0 ]
   * [ 0 -1^T -1 ]
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
    using Vec = linalg::Vector<T, N>;

    using AffineMatT = affine::affine_mat_t<CoordMap>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == N,
                  "input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");

    if constexpr (std::same_as<Norm, linalg::L2<Vec>>) {
      linalg::Matrix<T, N + 1, Dom + 1> F;
      F = linalg::set_block<0, N, 0, Dom>(F, coord_map.linear_part());

      linalg::Vector<T, N + 1> g;
      for (size_t i = 0; i < N; ++i) g[i] = coord_map.translation()(i);

      if constexpr (Tag == FnTag::Concave) {
        F[N][N] = T(-1);
      } else {
        F[N][N] = T(1);
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
      const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
      const auto emitted_map = std::get<0>(map_res);
      const auto map = std::get<1>(map_res);

      const auto soc = conic::SecondOrderCone<linalg::Vector<T, N + 1>>();
      return std::tuple(
          emitted_map,
          conic::ConeBlock(
              map, std::tuple_cat(cone_block.cones, std::make_tuple(soc))));
    } else if constexpr (std::same_as<Norm, linalg::Linf<Vec>>) {
      linalg::Matrix<T, 2 * N, Dom + 1> F;
      F = linalg::set_block<0, N, 0, N>(F, coord_map.linear_part());
      F = linalg::set_block<N, 2 * N, 0, N>(F, T(-1) * coord_map.linear_part());

      linalg::Vector<T, 2 * N> g;
      for (size_t i = 0; i < N; ++i) g[i] = coord_map.translation()(i);
      for (size_t i = 0; i < N; ++i)
        g[i + N] = T(-1) * coord_map.translation()(i);

      if constexpr (Tag == FnTag::Convex) {
        for (size_t i = 0; i < 2 * N; ++i) F[i][N] = T(1);
      } else {
        for (size_t i = 0; i < 2 * N; ++i) F[i][N] = T(-1);
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
      const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
      const auto emitted_map = std::get<0>(map_res);
      const auto map = std::get<1>(map_res);

      const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
      return std::tuple(
          emitted_map,
          conic::ConeBlock(map, std::tuple_cat(cone_block.cones,
                                               std::make_tuple(rplus, rplus))));
    } else {
      linalg::Matrix<T, 2 * N + 1, 2 * Dom + 1> F;
      const auto I = linalg::make_identity<T, N>();
      F = linalg::set_block<0, N, 0, N>(F, coord_map.linear_part());
      F = linalg::set_block<N, 2 * N, 0, N>(F, T(-1) * coord_map.linear_part());
      F = linalg::set_block<0, N, N, 2 * N>(F, I);
      F = linalg::set_block<N, 2 * N, N, 2 * N>(F, I);
      for (size_t i = 0; i < N; ++i) F[2 * N][N + i] = T(-1);

      linalg::Vector<T, 2 * N + 1> g;
      for (size_t i = 0; i < N; ++i) g[i] = coord_map.translation()(i);
      for (size_t i = 0; i < N; ++i)
        g[i + N] = T(-1) * coord_map.translation()(i);

      if constexpr (Tag == FnTag::Convex) {
        F[2 * N][2 * N] = T(1);
      } else {
        F[2 * N][2 * N] = T(-1);
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
      const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
      const auto emitted_map = std::get<0>(map_res);
      const auto map = std::get<1>(map_res);

      const auto rplus1 = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
      const auto rplus2 = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
      return std::tuple(
          emitted_map,
          conic::ConeBlock(
              map, std::tuple_cat(cone_block.cones,
                                  std::make_tuple(rplus1, rplus1, rplus2))));
    }
  }
};

}  // namespace convex
