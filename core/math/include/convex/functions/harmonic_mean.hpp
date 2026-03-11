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
#include "convex/cones/soc.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on harmonic mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 2)
using HarmonicMeanDom = ConvexDomain<Space, VarType::POS>;

/// @brief alias for codomain on harmonic mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HarmonicMeanCodom = ConvexDomain<Space, VarType::POS>;

/// @brief alias for convex codomain on harmonic mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HarmonicMeanConvexCodom = ConvexDomain<Space, VarType::NEG>;

/// @brief alias for meta for harmonic mean function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using HarmonicMeanMeta =
    ConvexAtomMeta<HarmonicMeanDom<Dom>, HarmonicMeanCodom<Codom>,
                   CurvatureType::CONCAVE>;

/// @brief alias for meta for harmonic mean function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using HarmonicMeanConvexMeta =
    ConvexAtomMeta<HarmonicMeanDom<Dom>, HarmonicMeanConvexCodom<Codom>,
                   CurvatureType::CONVEX>;

/**
 * @brief harmonic mean f(x) = 2xy / (x + y) is convex where dom f = R^2_++
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 2)
struct HarmonicMean : Atom<Meta, HarmonicMean<Meta, Child>, Child> {
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = Atom<Meta, HarmonicMean<Meta, Child>, Child>;
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
  constexpr explicit HarmonicMean(const Child& child_in)
      : ConvexBase(child_in),
        lift_map(affine::make_map_id<LiftedVarMat<T>>()) {}

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit HarmonicMean(
      const Child& child_in, const affine::AffineMap<LiftedVarMat<T>>& map_in)
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
   * @param x \in dom f
   *
   * @return y \in R_++ or R_--
   */
  constexpr T eval(const PointT& x) const {
    AK_ASSERT(x(0) >= T(0) && x(1) >= T(1) &&
              "Invalid domain for harmonic mean");

    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONCAVE)
      return alpha * (T(2) * x(0) * x(1)) / (x(0) + x(1)) + beta;
    else
      return alpha * (T(-2) * x(0) * x(1)) / (x(0) + x(1)) + beta;
  }

  /**
   * @brief mul
   *
   * @return harmonic mean fn
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      using Dom = dom_space_t<meta_dom_t<Meta>>;
      using Codom = dom_space_t<meta_codom_t<Meta>>;
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      if constexpr (Tag == CurvatureType::CONVEX)
        return HarmonicMean<HarmonicMeanMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
      else
        return HarmonicMean<HarmonicMeanConvexMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation);
      return HarmonicMean<Meta, Child>(std::get<0>(this->children), new_map);
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return harmonic mean fn
   */
  constexpr auto add(const T& beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return HarmonicMean<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalization impl
   *
   * (1) f(x) = 2 x y / (x + y)
   * C = { (x, y, t) | 2 x y / (x + y) >= t, x > 0, y > 0 }
   * 2xy >= t(x + y)
   *
   * RSOC let u = t / sqrt(2), v = x - t/2, w = y - t/2
   * t^2 / 2 <= 2(x - t/2)(y - t/2)
   * t^2 <= 4[xy - xt/2 - yt/2 + t^2/4]
   * 0 <= 4xy - 2t(x + y) + t^2... 2xy >= t(x + t)
   *
   * [ 0   0   1/sqrt(2) ] [ x; y; t ] + [ 0; b; b ] -> (u, v, w)
   * [ a0  0  -1/2       ]
   * [ 0   a1 -1/2       ]
   *
   * (2) -f(x) = -2 x y / (x + y)
   * RSOC let u = -t/sqrt(2), v = x + t/2, w = y + t/2
   *
   * [ 0   0 -1/sqrt(2) ] [ x; y; t ] + [ 0; b; b ] -> (u, v, w)
   * [ a0  0  1/2       ]
   * [ 0   a1 1/2       ]
   *
   * @param builder
   *
   * @return new builder
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
    static_assert(Codom == N,
                  "input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");

    linalg::Matrix<T, N + 1, Dom + 1> F;
    F(1)(0) = child_coord.linear_part()(0)(0);
    F(2)(1) = child_coord.linear_part()(1)(1);
    linalg::Vector<T, N + 1> g;
    g(1) = child_coord.translation()(0);
    g(2) = child_coord.translation()(1);

    if constexpr (Tag == CurvatureType::CONCAVE) {
      F(0)(Dom) = T(1) / linalg::constexpr_sqrt_safe(T(2));
      F(1)(Dom) = T(-0.5);
      F(2)(Dom) = T(-0.5);
    } else {
      F(0)(Dom) = T(-1) / linalg::constexpr_sqrt_safe(T(2));
      F(1)(Dom) = T(0.5);
      F(2)(Dom) = T(0.5);
    }

    const auto rsoc_coords = affine::AffineMap<decltype(F)>(F, g);
    const auto coord_map_up =
        conic::Rotated2SOC<decltype(rsoc_coords)>()(rsoc_coords);
    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                decltype(coord_map_up), decltype(child_cb)>(
            coord_map_up, child_cb);
    const auto rsoc = conic::SecondOrderCone<linalg::Vector<T, N + 1>>();
    const auto cones = std::make_tuple(rsoc);

    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

}  // namespace convex
