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

/// @brief alias for domain on log function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogDom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogCodom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for convex codomain on log function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogConvexCodom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Space>
using LogMeta =
    ConvexAtomMeta<LogDom<Space>, LogCodom<Space>, CurvatureType::CONCAVE>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Space>
using LogConvexMeta =
    ConvexAtomMeta<LogDom<Space>, LogConvexCodom<Space>, CurvatureType::CONVEX>;

/**
 * @brief log function
 *
 * f(x) = log(x), x \in R_++
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Log : ConvexFn<Meta, Log<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Meta, Log<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::LinearMatT;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Log(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Log(const Child& child_in,
                         const affine::AffineMap<LinearMatT>& map_in)
      : ConvexBase(child_in, map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity_impl() const {
    return MonotonicityType::NONDEC;
  }

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return log(x) \in R
   */
  constexpr std::optional<T> eval_impl(const PointT& x) const {
    if (x(0) <= T(0)) return std::nullopt;
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONCAVE)
      return linalg::constexpr_ln(x(0));
    else
      return T(-1) * linalg::constexpr_ln(x(0)).value();
  }

  /**
   * @brief negate impl
   *
   * @return negated log
   */
  constexpr auto negate_impl() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    using Space = dom_space_t<meta_dom_t<Meta>>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      return Log<LogConvexMeta<Space>, Child>(std::get<0>(this->children),
                                              this->map);
    } else {
      return Log<LogMeta<Space>, Child>(std::get<0>(this->children), this->map);
    }
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = log(x), x in R_++
   * K = { (x,t) | log(x) >= t, x in R_++ } -> { (x, t) | x >= exp(t) }
   * K_exp = { (u,v,w) | v exp(u/v) <= w, v > 0 }
   *
   * let u = t, v = 1, w = x
   * [ 0   1 ] [ x; t ] + [ 0; 1; b ] -> (t, 1, a^T x + b)
   * [ 0   0 ]
   * [ a^T 0 ]
   *
   * (2) f(x) = -log(x), x in R_++
   * K = { (x,t) | -log(x) <= t } -> { (x,t) | log(x) >= -t }
   * -> { (x,t) | x >= exp(-t) }
   *
   * let u = -t, v = 1, w = x
   * [ 0   -1 ] [ x; t ] + [ 0; 1; b ] -> (-t, 1, a^T x + b)
   * [ 0    0 ]
   * [ a^T  0 ]
   *
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
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");
    linalg::Matrix<T, 3, Dom + 1> F;
    linalg::Vector<T, 3> g;
    F = linalg::set_block<2, Codom + 2, 0, Dom>(F, coord_map.linear_part());
    g[1] = T(1);
    g[2] = coord_map.translation()(0);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      F[0][1] = T(1);
    } else {
      F[0][1] = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
    const auto emitted_map = std::get<0>(map_res);
    const auto map = std::get<1>(map_res);

    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 3>>();
    return std::tuple(
        emitted_map,
        conic::ConeBlock(
            map, std::tuple_cat(cone_block.cones, std::make_tuple(exp_cone))));
  }
};

}  // namespace convex
