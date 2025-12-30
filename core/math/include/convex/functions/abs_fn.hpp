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
using AbsDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using AbsCodom = ConvexDomain<Space, SignType::NONNEG>;

/// @brief alias for concave codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using AbsConcaveCodom = ConvexDomain<Space, SignType::NONPOS>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Space>
using AbsMeta =
    ConvexAtomMeta<AbsDom<Space>, AbsCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Space>
using AbsConcaveMeta = ConvexAtomMeta<AbsDom<Space>, AbsConcaveCodom<Space>,
                                      CurvatureType::CONCAVE>;

/**
 * @brief abs function
 *
 * f(x) = |x|, x \in R
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Abs : ConvexFn<Meta, Abs<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = ConvexFn<Meta, Abs<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::LinearMatT;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Abs(const Child& child_in) : ConvexBase(child_in) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Abs(const Child& child_in,
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
   * @return abs(x) \in R
   */
  constexpr std::optional<T> eval_impl(const PointT& x) const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX)
      return linalg::constexpr_abs(x(0));
    else
      return T(-1) * linalg::constexpr_abs(x(0));
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
      return Abs<AbsMeta<Space>, Child>(std::get<0>(this->children), this->map);
    } else {
      return Abs<AbsConcaveMeta<Space>, Child>(std::get<0>(this->children),
                                               this->map);
    }
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) abs(x) -> epi -> cones
   * f(x) = |x|
   * K_epi = { (x,t) | |x| <= t }
   *
   * let u >= |x|... first convex constraint
   * 0 <= t - u... second convex constraint
   * u >= 0... third convex constraint
   * t >= 0... fourth convex constraint
   *
   * -u <= x <= u
   * 0 <= u + x... nonneg orthant cone
   * 0 <= u - x... nonneg orthant cone
   * 0 <= t - u... nonneg orthant cone
   * u >= 0... nonneg orthant cone
   * t >= 0... nonneg orthant cone
   *
   * [  a^T  1 0 ] [ x; u; t ] + [ b; -b; 0; 0; 0 ]
   * [ -a^T  1 0 ]
   * [  0   -1 1 ]
   * [  0    1 0 ]
   * [  0    0 1 ]
   *
   * (2) -abs(x) -> hypo -> cones
   * f(x) = - |x|
   * K_hypo = { (x,t) | -|x| >= t, t <= 0}
   *
   * let u >= |x|... first convex constraint
   * let t <= -u... second convex constraint
   * u >= 0... third convex constraint
   * t <= 0... fourth convex constraint
   *
   * 0 <= u + x... nonneg orthant cone
   * 0 <= u - x... nonneg orthant cone
   * 0 <= -u - t... nonneg orthant cone
   * u >= 0... nonneg orthant cone
   * -t >= 0... nonneg orthant cone
   *
   * [  a^T  1  0 ] [ x; u; t ] + [ b; -b; 0; 0; 0 ]
   * [ -a^T  1  0 ]
   * [  0   -1 -1 ]
   * [  0    1  0 ]
   * [  0    0 -1 ]
   *
   * @return cone block
   */
  template <linalg::RnSpace DecVarDom>
  constexpr auto canonicalize_fn(
      const ConeMapBuilder<DecVarDom>& builder) const {
    const auto child_canon = std::get<0>(this->children).canonicalize();
    const auto child_sel = std::get<0>(child_canon);
    const auto child_cb = std::get<1>(child_canon);

    using AffineMatT = affine::affine_mat_t<decltype(child_sel)>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");
    linalg::Matrix<T, 5, Dom + 2> F;
    F = linalg::set_block<0, 1, 0, Dom>(F, child_sel.linear_part());
    F = linalg::set_block<1, 2, 0, Dom>(F, T(-1) * child_sel.linear_part());
    F[0][1] = T(1);
    F[1][1] = T(1);
    F[2][1] = T(-1);
    F[3][1] = T(1);

    linalg::Vector<T, 5> g;
    g[0] = child_sel.translation()(0);
    g[1] = T(-1) * child_sel.translation()(0);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      F[2][2] = T(1);
      F[4][2] = T(1);
    } else {
      F[2][2] = T(-1);
      F[4][2] = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto map_res = builder(coord_map_up, child_cb.map, -1);
    const auto emitted_map = std::get<0>(map_res);
    const auto map = std::get<1>(map_res);

    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    return std::tuple(
        emitted_map,
        conic::ConeBlock(
            map,
            std::tuple_cat(child_cb.cones, std::make_tuple(rplus, rplus, rplus,
                                                           rplus, rplus))));
  }
};

}  // namespace convex
