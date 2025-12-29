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

/// @brief alias for domain on abs function
template <linalg::RnSpace Space>
using LogSumExpDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogSumExpCodom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for concave codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogSumExpConcaveCodom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using LogSumExpMeta = ConvexAtomMeta<LogSumExpDom<Dom>, LogSumExpCodom<Codom>,
                                     CurvatureType::CONVEX>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using LogSumExpConcaveMeta =
    ConvexAtomMeta<LogSumExpDom<Dom>, LogSumExpConcaveCodom<Codom>,
                   CurvatureType::CONCAVE>;

/**
 * @brief log sum exp = smooth max function
 *
 * lse(x) = log(\sum_i^n(e^{x_i})) which is a smooth mapping from R^n to R
 */
template <typename Meta, typename Child>
struct LogSumExp : ConvexFn<Meta, LogSumExp<Meta, Child>, Child> {
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = ConvexFn<Meta, LogSumExp<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::LinearMatT;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief alias for affine map (codomain 1-dim)
  using Mat = linalg::Matrix<T, 1, N>;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit LogSumExp(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit LogSumExp(const Child& child_in,
                               const affine::AffineMap<LinearMatT>& map_in)
      : ConvexBase(child_in, map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity_impl() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX)
      return MonotonicityType::NONDEC;
    else
      return MonotonicityType::NONINC;
  }

  /**
   * @brief eval impl
   *
   * @param point x \in dom f
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const PointT& x) const {
    T sum(0);
    std::for_each(x.begin(), x.end(),
                  [&](const auto& pt) { sum += linalg::constexpr_exp(pt); });

    const auto res = linalg::constexpr_ln(sum);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX)
      return res;
    else {
      if (!res.has_value()) return std::nullopt;
      return T(-1) * res.value();
    }
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    using Dom = dom_space_t<meta_dom_t<Meta>>;
    using Codom = dom_space_t<meta_codom_t<Meta>>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      return LogSumExp<LogSumExpMeta<Dom, Codom>, Child>(
          std::get<0>(this->children), this->map);
    } else {
      return LogSumExp<LogSumExpConcaveMeta<Dom, Codom>, Child>(
          std::get<0>(this->children), this->map);
    }
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

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX) {
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
