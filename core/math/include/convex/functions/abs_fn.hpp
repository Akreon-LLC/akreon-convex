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

#include "convex/convex_traits.hpp"
#include "convex/convex_util.hpp"

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
struct Abs : Atom<Meta, Abs<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, Abs<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief alias for lifted var map
  using LiftedVarMatT = linalg::Matrix<T, 1, 1>;
  using LiftedVarBiasT = linalg::Vector<T, 1>;

  /// @brief alias to extract space type
  using Space = dom_space_t<meta_dom_t<Meta>>;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Abs(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMatT>()) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Abs(const Child& child_in,
                         const affine::AffineMap<LiftedVarMatT>& map_in)
      : ConvexBase(child_in), lift_map(map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
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
   * @brief eval via operator overload
   *
   * @param point x
   *
   * @return result
   */
  constexpr std::optional<T> operator()(const PointT& x) const {
    return eval(x);
  }

  /**
   * @brief eval impl
   *
   * @param x \in R
   *
   * @return abs(x) \in R
   */
  constexpr std::optional<T> eval(const PointT& x) const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX)
      return alpha * linalg::constexpr_abs(x(0)) + beta;
    else
      return T(-1) * alpha * linalg::constexpr_abs(x(0)) + beta;
  }

  /**
   * @brief mul
   *
   * @return abs fn (updated map)
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMatT> new_map(alpha * lift_map.linear_part(),
                                               lift_map.translation());
      static constexpr CurvatureType Tag = meta_curv_v<Meta>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Abs<AbsMeta<Space>, Child>(std::get<0>(this->children), new_map);
      } else {
        return Abs<AbsConcaveMeta<Space>, Child>(std::get<0>(this->children),
                                                 new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMatT> new_map(Scalar * lift_map.linear_part(),
                                               lift_map.translation());
      return Abs<Meta, Child>(std::get<0>(this->children, new_map));
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return abs fn (updated map)
   */
  constexpr auto add(const T& beta) const {
    LiftedVarBiasT bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMatT> new_map(lift_map.linear_part(), bias);
    return Abs<Meta, Child>(std::get<0>(this->children, new_map));
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
  template <OrigType Type, BuilderLike Builder>
  constexpr auto canonicalize(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<OrigType::FN>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    using AffineMatT = affine::affine_mat_t<decltype(child_coord)>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");
    linalg::Matrix<T, 5, Dom + 2> F;
    F = linalg::set_block<0, 1, 0, Dom>(F, child_coord.linear_part());
    F = linalg::set_block<1, 2, 0, Dom>(F, T(-1) * child_coord.linear_part());
    F[0][Dom] = T(1);
    F[1][Dom] = T(1);
    F[2][Dom] = T(-1);
    F[3][Dom] = T(1);

    linalg::Vector<T, 5> g;
    g[0] = child_coord.translation()(0);
    g[1] = T(-1) * child_coord.translation()(0);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      F[2][Dom + 1] = T(1);
      F[4][Dom + 1] = T(1);
    } else {
      F[2][Dom + 1] = T(-1);
      F[4][Dom + 1] = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto new_builder =
        ConeBlockBuilder<builder_dom_t<decltype(builder)>,
                         decltype(coord_map_up), decltype(child_cb)>(
            coord_map_up, child_cb);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto cones = std::make_tuple(rplus, rplus, rplus, rplus, rplus);

    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMatT> lift_map;
  const int idx{-1};
};

}  // namespace convex
