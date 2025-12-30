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

/// @brief alias for domain on inv pos function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using InvPosDom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for codomain on inv pos function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using InvPosCodom = ConvexDomain<Space, SignType::POS>;

/// @brief alias for concave codomain on inv pos function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using InvPosConcaveCodom = ConvexDomain<Space, SignType::NEG>;

/// @brief alias for meta for inv pos function
template <linalg::RnSpace Space>
using InvPosMeta =
    ConvexAtomMeta<InvPosDom<Space>, InvPosCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for inv pos function
template <linalg::RnSpace Space>
using InvPosConcaveMeta =
    ConvexAtomMeta<InvPosDom<Space>, InvPosConcaveCodom<Space>,
                   CurvatureType::CONCAVE>;

/**
 * @brief inv pos function
 *
 * f(x) = 1/x, x in R_++
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct InvPos : ConvexFn<Meta, InvPos<Meta, Child>, Child> {
 public:
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Meta, InvPos<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::LinearMatT;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit InvPos(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit InvPos(const Child& child_in,
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
      return MonotonicityType::NONINC;
    else
      return MonotonicityType::NONDEC;
  }

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return res \in R_++ or R_-- (depending on convex or concave)
   */
  constexpr std::optional<T> eval_impl(const PointT& x) const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if (x(0) <= T(0)) return std::nullopt;
    if constexpr (Tag == CurvatureType::CONVEX) {
      return T(1) / x(0);
    } else {
      return T(-1) / x(0);
    }
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    using Space = dom_space_t<meta_dom_t<Meta>>;
    if constexpr (Tag == CurvatureType::CONCAVE) {
      return InvPos<InvPosMeta<Space>, Child>(std::get<0>(this->children),
                                              this->map);
    } else {
      return InvPos<InvPosConcaveMeta<Space>, Child>(
          std::get<0>(this->children), this->map);
    }
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = 1/x
   * K = { (x, t) | 1/x <= t, x in R_++ implies t >= 0 }
   * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v > 0, w >= 0 }
   *
   * let u = sqrt(2), v = x, w = t
   * [ 0   0 ] [ x; t ] + [ sqrt(2); b; 0] -> (sqrt(2), a^T x + b, t)
   * [ a^T 0 ]
   * [ 0   1 ]
   *
   * (2) f(x) = -1/x
   * K = { (x, t) | 1/x <= -t, x in R_++ t in R_- }
   *
   * let u = sqrt(2), v = x, w = -t
   * [ 0    0 ] [ x; t ] + [ sqrt(2); b; 0 ]
   * [ a^T  0 ]
   * [ 0   -1 ]
   *
   * @param cone block
   *
   * @return cone block
   */
#if 0
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
    F = linalg::set_block<1, 2, 0, Dom>(F, coord_map.linear_part());
    linalg::Vector<T, 3> g;
    g[0] = linalg::constexpr_sqrt(T(2));
    g[1] = coord_map.translation()(0);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      F[2][1] = T(1);
    } else {
      F[2][1] = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto map_res = cmap_builder(coord_map_up, cone_block.map, -1);
    const auto emitted_map = std::get<0>(map_res);
    const auto map = std::get<1>(map_res);

    const auto rsoc = conic::RotatedSecondOrderCone<linalg::Vector<T, 3>>();
    return std::tuple(
        emitted_map,
        conic::ConeBlock(
            map, std::tuple_cat(cone_block.cones, std::make_tuple(rsoc))));
  }
#endif
};

}  // namespace convex
