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

#include <algorithm>

#include "convex/canonical_form_builder.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on geo mean function
template <linalg::RnSpace Space>
using GeoMeanDom = ConvexDomain<Space, VarType::POS>;

/// @brief alias for codomain on geo mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using GeoMeanCodom = ConvexDomain<Space, VarType::POS>;

/// @brief alias for convex codomain on geo mean function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using GeoMeanConvexCodom = ConvexDomain<Space, VarType::NEG>;

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
 * @brief geometric mean f(x) = (prod_{i=1}^n x_i)^{1/n} is concave on dom f =
 * R^n_++
 */
template <typename Meta, typename Child>
struct GeoMean : Atom<Meta, GeoMean<Meta, Child>, Child> {
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = Atom<Meta, GeoMean<Meta, Child>, Child>;
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
  constexpr explicit GeoMean(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit GeoMean(const Child& child_in,
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
    else return MonotonicityType::NONINC;
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
   * @brief evaluation impl
   *
   * @param x in dom f
   *
   * @return y in R
   */
  constexpr T eval(const PointT& x) const {
    T prod(1);
    std::for_each(x.begin(), x.end(), [&](const auto& pt) { prod *= pt; });
    const auto alpha = lift_map.linear_part()(0, 0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX)
      return T(-1) * alpha * linalg::constexpr_nrt_safe<T>(prod, N) + beta;
    else return alpha * linalg::constexpr_nrt_safe<T>(prod, N) + beta;
  }

  /**
   * @brief mul
   *
   * @return geo mean fn (updated map)
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      using Dom = dom_space_t<meta_dom_t<Meta>>;
      using Codom = dom_space_t<meta_codom_t<Meta>>;
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return GeoMean<GeoMeanConvexMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
      } else {
        return GeoMean<GeoMeanMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return GeoMean<Meta, Child>(std::get<0>(this->children), new_map);
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return geo mean fn (updated map)
   */
  constexpr auto add(T beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return GeoMean<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = prod_i^n(x_i)^{1/n}
   * K_geo = { (x, t) | prod_i^n(x_i)^{1/n} >= t }... x in R_+
   * power cone = { (x,t) in R_+ x R | prod(x_i)^alpha_i >= |t|, sum(a_i) = 1,
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
   *
   * @return cone block builder
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::FN>(builder);
    const auto child_coord = child_builder.coord_map;

    using AffineMatT = affine::affine_mat_t<decltype(child_coord)>;
    static constexpr std::size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr std::size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == N,
                  "input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");
    linalg::Matrix<T, N + 1, Dom + 1> F;
    F = linalg::set_block<0, N, 0, Dom>(F, child_coord.linear_part());
    linalg::Vector<T, N + 1> g;
    for (std::size_t i = 0; i < N; ++i) g(i) = child_coord.translation()(i);

    if constexpr (Tag == CurvatureType::CONVEX) {
      F(N, Dom) = T(-1);
    } else {
      F(N, Dom) = T(1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const linalg::vector_traits<linalg::Vector<T, N>> vec_t;
    const auto power_cone =
        conic::PowerCone<linalg::Vector<T, N + 1>>(vec_t.fill(T(1 / N)));
    const auto cones = std::make_tuple(power_cone);
    const auto emission =
        conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

    return child_builder.template realize<Type>(idx, lift_map, emission);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

template <typename Child>
GeoMean(const Child&) -> GeoMean<
    GeoMeanMeta<
        dom_space_t<meta_codom_t<atom_meta_t<Child>>>,
        linalg::Matrix<dom_type_t<meta_codom_t<atom_meta_t<Child>>>, 1, 1>>,
    atom_decay_t<Child>>;

} // namespace convex
