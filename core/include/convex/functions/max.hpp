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

/// @brief alias for domain on abs function
template <linalg::RnSpace Space>
using MaxDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using MaxCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for concave codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using MaxConcaveCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using MaxMeta =
    ConvexAtomMeta<MaxDom<Dom>, MaxCodom<Codom>, CurvatureType::CONVEX>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using MaxConcaveMeta =
    ConvexAtomMeta<MaxDom<Dom>, MaxConcaveCodom<Codom>, CurvatureType::CONCAVE>;

/**
 * @brief max convex function CRTP
 *
 * F(x) = max { x_1, x_2, ... x_n }, x in R^n
 */
template <typename Meta, typename Child>
struct Max : Atom<Meta, Max<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, Max<Meta, Child>, Child>;
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
  constexpr explicit Max(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Max(const Child& child_in,
                         const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), lift_map(map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
    if constexpr (Tag == CurvatureType::CONVEX) return MonotonicityType::NONDEC;
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
   * @brief eval impl
   *
   * @param Point in domain
   *
   * @return y in R
   */
  constexpr T eval(const PointT& x) const {
    T mx(0);
    std::for_each(x.begin(), x.end(), [&mx](const auto& pt) {
      if (mx < pt) mx = pt;
    });

    const auto alpha = lift_map.linear_part()(0, 0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX) return alpha * mx + beta;
    else return T(-1) * alpha * mx + beta;
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
      using Dom = dom_space_t<meta_dom_t<Meta>>;
      using Codom = dom_space_t<meta_codom_t<Meta>>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Max<MaxMeta<Dom, Codom>, Child>(std::get<0>(this->children),
                                               new_map);
      } else {
        return Max<MaxConcaveMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Max<Meta, Child>(std::get<0>(this->children), new_map);
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
    return Max<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize_impl
   *
   * (1) f(x) = max{x_1, ... x_n}
   * K = { (x,t) | max(x) <= t }
   *
   * x_i <= t forall i convex constraints
   * 0 <= t - x_i... in nonneg orthant cone
   *
   * [ -A 1 ] [ x ; t ] + [ -b ; 0 ] -> (-(A x + b), t)
   *
   * (2) f(x) = -max{x_1, ..., x_n}
   * K = { (x,t) | -max(x) >= t }
   *
   * -x_i >= t for all i convex constraints
   *  0 <= -t - x_i... in nonneg orthant cone
   *
   * [ -A -1 ] [ x; t ] + [ -b ; 0 ] -> (-(A x + b), -t)
   * @return cone block
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
                  "Input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");
    static_assert(Dom == N,
                  "Expecting child affine map to have square linear part");
    linalg::Matrix<T, N, Dom + 1> F;
    F = linalg::set_block<0, N, 0, Dom>(F, T(-1) * child_coord.linear_part());

    linalg::Vector<T, N> g;
    for (std::size_t i = 0; i < N; ++i)
      g(i) = T(-1) * child_coord.translation()(i);

    if constexpr (Tag == CurvatureType::CONVEX) {
      for (std::size_t i = 0; i < N; ++i) F(i, N) = T(1);
    } else {
      for (std::size_t i = 0; i < N; ++i) F(i, N) = T(-1);
    }

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
    const auto cones = std::make_tuple(rplus);
    const auto emission =
        conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

    return child_builder.template realize<Type>(idx, lift_map, emission);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

template <typename Child>
Max(const Child&) -> Max<
    MaxMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>,
            linalg::Matrix<dom_type_t<meta_codom_t<atom_meta_t<Child>>>, 1, 1>>,
    atom_decay_t<Child>>;

} // namespace convex
