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

#include "convex/cone_block_builder.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on log function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogDom = ConvexDomain<Space, VarType::POS>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for convex codomain on log function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogConvexCodom = ConvexDomain<Space, VarType::FREE>;

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
struct Log : Atom<Meta, Log<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, Log<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief alias to extract space type
  using Space = dom_space_t<meta_dom_t<Meta>>;

  /// @brief curvature tag
  static constexpr CurvatureType Tag = meta_curv_v<Meta>;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Log(const Child& child_in)
      : ConvexBase(child_in),
        lift_map(affine::make_map_id<LiftedVarMat<T>>()) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Log(const Child& child_in,
                         const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), lift_map(map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
    if constexpr (Tag == CurvatureType::CONCAVE)
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
  constexpr T operator()(const PointT& x) const { return eval(x); }

  /**
   * @brief eval impl
   *
   * @param x \in R_++
   *
   * @return log(x) \in R
   */
  constexpr T eval(const PointT& x) const {
    AK_ASSERT(x(0) > T(0) && "Invalid domain for log(x)");

    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONCAVE)
      return alpha * linalg::constexpr_ln(x(0)).value() + beta;
    else
      return T(-1) * alpha * linalg::constexpr_ln(x(0)).value() + beta;
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
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Log<LogConvexMeta<Space>, Child>(std::get<0>(this->children),
                                                new_map);
      } else {
        return Log<LogMeta<Space>, Child>(std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Log<Meta, Child>(std::get<0>(this->children), new_map);
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
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return Log<Meta, Child>(std::get<0>(this->children), new_map);
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
  template <AtomType Type, conic::BuilderLike Builder>
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
    linalg::Vector<T, 3> g;
    F = linalg::set_block<2, 3, 0, Dom>(F, child_coord.linear_part());
    g(1) = T(1);
    g(2) = child_coord.translation()(0);

    if constexpr (Tag == CurvatureType::CONCAVE) {
      F(0)(Dom) = T(1);
    } else {
      F(0)(Dom) = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
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
