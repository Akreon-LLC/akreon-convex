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
  requires(linalg::space_dim_v<Space> == 1)
using SquareDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SquareCodom = ConvexDomain<Space, SignType::NONNEG>;

/// @brief alias for concave codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SquareConcaveCodom = ConvexDomain<Space, SignType::NONPOS>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Space>
using SquareMeta =
    ConvexAtomMeta<SquareDom<Space>, SquareCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Space>
using SquareConcaveMeta =
    ConvexAtomMeta<SquareDom<Space>, SquareConcaveCodom<Space>,
                   CurvatureType::CONCAVE>;

/**
 * @brief square function
 *
 * f(x) = x*x, x \in R
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Square : ConvexFn<Meta, Square<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Meta, Square<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::LinearMatT;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Square(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Square(const Child& child_in,
                            const affine::AffineMap<LinearMatT>& map_in)
      : ConvexBase(child_in, map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity_impl() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    static constexpr SignType sign =
        dom_sign_v<meta_codom_t<atom_meta_t<Child>>>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      if constexpr (sign == SignType::NONNEG) {
        return MonotonicityType::NONDEC;
      } else if constexpr (sign == SignType::NONPOS) {
        return MonotonicityType::NONINC;
      } else {
        return MonotonicityType::UNKNOWN;
      }
    } else {
      if constexpr (sign == SignType::NONNEG) {
        return MonotonicityType::NONINC;
      } else if constexpr (sign == SignType::NONPOS) {
        return MonotonicityType::NONDEC;
      } else {
        return MonotonicityType::UNKNOWN;
      }
    }
  }

  /**
   * @brief eval impl
   *
   * @param x \in R
   *
   * @return x^2 \in R
   */
  constexpr std::optional<T> eval_impl(const PointT& x) const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX)
      return x(0) * x(0);
    else
      return T(-1) * x(0) * x(0);
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
      return Square<SquareMeta<Space>, Child>(std::get<0>(this->children),
                                              this->map);
    } else {
      return Square<SquareConcaveMeta<Space>, Child>(
          std::get<0>(this->children), this->map);
    }
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = x^2
   * K = { (x,t) | x^2 <= t }
   * K_rsoc = { (u, v, w) | ||u||_2^2 <= 2 v w, v > 0, w >= 0 }
   *
   * let u = x, v = 1/2, w = t
   * [ a^T 0 ] [ x; t ] + [ b; 1/2; 0 ] -> (a^T x + b, 1/2, t)
   * [ 0   0 ]
   * [ 0   1 ]
   *
   * (2) f(x) = -x^2
   * K = { (x,t) | -x^2 >= t }... { (x,t) | x^2 <= -t }
   *
   * let u = x, v = 1/2, w = -t
   * [ a^T  0 ] [ x; t ] + [ b; 1/2; 0 ] -> (a^T x, + b 1/2, -t)
   * [ 0    0 ]
   * [ 0   -1 ]
   *
   * @param cone map builder
   * @param coord map
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
    F = linalg::set_block<0, 1, 0, Dom>(F, coord_map.linear_part());
    linalg::Vector<T, 3> g;
    g[0] = coord_map.translation()(0);
    g[1] = T(0.5);

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
