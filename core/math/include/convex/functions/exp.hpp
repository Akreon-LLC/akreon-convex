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

/// @brief alias for domain on exp function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using ExpDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on exp function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using ExpCodom = ConvexDomain<Space, SignType::NONNEG>;

/// @brief alias for concave codomain on exp function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using ExpConcaveCodom = ConvexDomain<Space, SignType::NONPOS>;

/// @brief alias for meta for exp function
template <linalg::RnSpace Space>
using ExpMeta =
    ConvexAtomMeta<ExpDom<Space>, ExpCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for exp function
template <linalg::RnSpace Space>
using ExpConcaveMeta = ConvexAtomMeta<AbsDom<Space>, ExpConcaveCodom<Space>,
                                      CurvatureType::CONCAVE>;

/**
 * @brief exp function
 *
 * f(x) = e^x, x \in R
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Exp : Atom<Meta, Exp<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, Exp<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief alias to extract space type
  using Space = dom_space_t<meta_dom_t<Meta>>;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Exp(const Child& child_in)
      : ConvexBase(child_in),
        lift_map(affine::make_map_id<LiftedVarMat<T>>()) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Exp(const Child& child_in,
                         const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), lift_map(map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX)
      return MonotonicityType::NONDEC;
    else
      return MonotonicityType::NONINC;
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
   * @return exp(x) \in R
   */
  constexpr std::optional<T> eval(const PointT& x) const {
    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX)
      return alpha * linalg::constexpr_exp(x(0)) + beta;
    else
      return T(-1) * alpha * linalg::constexpr_exp(x(0)) + beta;
  }

  /**
   * @brief mul
   *
   * @return exp fn (updated map)
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      static constexpr CurvatureType Tag = meta_curv_v<Meta>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Exp<ExpMeta<Space>, Child>(std::get<0>(this->children), new_map);
      } else {
        return Exp<ExpConcaveMeta<Space>, Child>(std::get<0>(this->children),
                                                 new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Exp<Meta, Child>(std::get<0>(this->children, new_map));
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return exp fn (updated map)
   */
  constexpr auto add(const T& beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return Exp<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = exp(x)
   * K = { (x,t) | exp(x) <= t }
   * K_exp = { (u,v,w) | v exp(u/v) <= w, v > 0 }
   *
   * let u = x, v = 1, w = t
   *
   * [ a^T 0 ] [ x; t ] + [ b; 1; 0 ] -> (a^T x + b, 1, t)
   * [ 0   0 ]
   * [ 0   1 ]
   *
   * (2) f(x) = -exp(x)
   * K_exp = { (x, t) | -exp(x) >= t }... { (x, t) | exp(x) <= -t, t <= 0 }
   *
   * [ a^T  0 ] [ x; t ] + [ b; 1; 0 ] -> (a^T x + b, 1, -t)
   * [ 0    0 ]
   * [ 0   -1 ]
   *
   * @param cone block builder
   *
   * @return cone block builder
   */
  template <AtomType Type, BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::FN>(builder);
    const auto child_coord = child_builder.coord_map;
    const auto child_cb = child_builder.cb;

    using AffineMatT = affine::affine_mat_t<decltype(child_coord)>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");
    linalg::Matrix<T, 3, Dom + 1> F;
    F = linalg::set_block<0, 1, 0, Dom>(F, child_coord.linear_part());
    linalg::Vector<T, 3> g;
    g[0] = child_coord.translation()(0);
    g[1] = T(1);

    static constexpr CurvatureType Tag = meta_curv_v<Meta>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      F[2][Dom] = T(1);
    } else {
      F[2][Dom] = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto new_builder =
        ConeBlockBuilder<builder_dom_t<decltype(builder)>,
                         decltype(coord_map_up), decltype(child_cb)>(
            coord_map_up, child_cb);
    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 3>>();
    const auto cones = std::make_tuple(exp_cone);

    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

}  // namespace convex
