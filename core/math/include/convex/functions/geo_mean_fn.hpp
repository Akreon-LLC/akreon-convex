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

/// @brief alias for domain on geo mean function
template <linalg::RnSpace Space>
using GeoMeanDom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for codomain on geo mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using GeoMeanCodom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for convex codomain on geo mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using GeoMeanConvexCodom = ConvexDomain<Space, SignType::NEG>;

/// @brief alias for meta for geo mean function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using GeoMeanMeta = ConvexAtomMeta<GeoMeanDom<Dom>, GeoMeanCodom<Codom>,
                                   CurvatureType::CONCAVE>;

/// @brief alias for meta for geo mean function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using GeoMeanConvexMeta =
    ConvexAtomMeta<GeoMeanDom<Dom>, GeoMeanConvexCodom<Codom>,
                   CurvatureType::CONVEX>;

/**
 * @brief geometric mean f(x) = (\prod_{i=1}^n x_i)^{1/n} is concave on dom f =
 * R^n_++
 */
template <typename Meta, typename Child>
struct GeoMean : ConvexFn<Meta, GeoMean<Meta, Child>, Child> {
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = ConvexFn<Meta, GeoMean<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::LinearMatT;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit GeoMean(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit GeoMean(const Child& child_in,
                             const affine::AffineMap<LinearMatT>& map_in)
      : ConvexBase(child_in, map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity_impl() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONCAVE)
      return MonotonicityType::NONDEC;
    else
      return MonotonicityType::NONINC;
  }

  /**
   * @brief evaluation impl
   *
   * @param x \in dom f
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const PointT& x) const {
    T prod(1);
    std::for_each(x.begin(), x.end(), [&](const auto& pt) { prod *= pt; });
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX)
      return T(-1) * linalg::constexpr_nrt<T>(prod, N);
    else
      return linalg::constexpr_nrt<T>(prod, N);
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
      return GeoMean<GeoMeanConvexMeta<Dom, Codom>, Child>(
          std::get<0>(this->children), this->map);
    } else {
      return GeoMean<GeoMeanMeta<Dom, Codom>, Child>(
          std::get<0>(this->children), this->map);
    }
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = prod_i^n(x_i)^{1/n}
   * K_geo = { (x, t) | prod_i^n(x_i)^{1/n} >= t }... x in R_+
   * power cone = { (x,t) \in R_+ x R | prod(x_i)^alpha_i >= |t|, sum(a_i) = 1,
   * a_i >= 0 }
   *
   * [ A 0 ] [ x; t ] + [ b; 0 ] -> [ A x + b; t ]
   * [ 0 1 ]
   *
   * (2) f(x) = -prod_i^n(x_i)^{1/n}
   * K_geo = { (x, t) | prod_i^n(x_i)^{1/n} >= -t }... x in R_+
   *
   * [ A  0 ] [ x; t ] + [ b; 0 ] -> [ A x + b; -t ]
   * [ 0 -1 ]
   *
   * @param cone map builder
   * @param coord_map
   * @param cone_block
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
    linalg::Matrix<T, N + 1, Dom + 1> F;
    F = linalg::set_block<0, N, 0, Dom>(F, coord_map.linear_part());
    linalg::Vector<T, N + 1> g;
    for (size_t i = 0; i < N; ++i) g[i] = coord_map.translation()(i);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      F[N][N] = T(-1);
    } else {
      F[N][N] = T(1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
    const auto emitted_map = std::get<0>(map_res);
    const auto map = std::get<1>(map_res);

    const auto power_cone = conic::PowerCone<linalg::Vector<T, N + 1>>();
    return std::tuple(
        emitted_map,
        conic::ConeBlock(map, std::tuple_cat(cone_block.cones,
                                             std::make_tuple(power_cone))));
  }
};

}  // namespace convex
