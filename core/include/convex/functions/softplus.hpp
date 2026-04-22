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

#include "convex/canonical_form_builder.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on softplus function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SoftplusDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on softplus function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SoftplusCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for concave codomain on softplus function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SoftplusConcaveCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for meta for softplus function
template <linalg::RnSpace Space>
using SoftplusMeta = ConvexAtomMeta<SoftplusDom<Space>, SoftplusCodom<Space>,
                                    CurvatureType::CONVEX>;

/// @brief alias for meta for softplus function
template <linalg::RnSpace Space>
using SoftplusConcaveMeta =
    ConvexAtomMeta<SoftplusDom<Space>, SoftplusConcaveCodom<Space>,
                   CurvatureType::CONCAVE>;

/**
 * @brief softplus function
 *
 * f(x) = log(1 + x)
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Softplus : Atom<Meta, Softplus<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, Softplus<Meta, Child>, Child>;
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
  constexpr explicit Softplus(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Softplus(const Child& child_in,
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
   * @param x \in R
   *
   * @return log(1 + exp(x))
   */
  constexpr T eval(const PointT& x) const {
    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX)
      return alpha *
                 linalg::constexpr_ln(1 + linalg::constexpr_exp(x(0))).value() +
             beta;
    else
      return T(-1) * alpha *
                 linalg::constexpr_ln(1 + linalg::constexpr_exp(x(0))).value() +
             beta;
  }

  /**
   * @brief mul
   *
   * @return softplus fn (updated map)
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Softplus<SoftplusMeta<Space>, Child>(std::get<0>(this->children),
                                                    new_map);
      } else {
        return Softplus<SoftplusConcaveMeta<Space>, Child>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Softplus<Meta, Child>(std::get<0>(this->children), new_map);
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return softplus fn (updated map)
   */
  constexpr auto add(const T& beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return Softplus<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief softplus canonicalization
   *
   * (1) f(x) = log(1 + exp(x))
   * K = { (x,t) | 1 + exp(x) <= exp(t) }... { (x,t) | exp(-t) + exp(x-t) <= 1 }
   * K_exp1 = (-t, 1, r)
   * K_exp2 = (x-t, 1, s)
   * K_nonneg = 0 <= 1 - s - r
   *
   * [ 0   -1  0  0 ] [ x; t; r; s ] + [ 0; 1; 0; b; 1; 0 | 1 ]
   * [ 0    0  0  0 ]
   * [ 0    0  1  0 ]
   * [ a^T -1  0  0 ]
   * [ 0    0  0  0 ]
   * [ 0    0  0  1 ]
   * --------------
   * [ 0    0 -1 -1 ]
   *
   * (2) -f(x) = -log(1 + exp(x))
   * K = { (x,t) | exp(0) + exp(x) >= exp(-t) }
   * K_exp1 = (t, 1, r)
   * K_exp2 = (x+t, 1, s)
   * K_nonneg = 0 <= 1 - r - s
   *
   * [ 0   1  0  0 ] [ x; t; r; s ] + [ 0; 1; 0; b; 1; 0 | 1 ]
   * [ 0   0  0  0 ]
   * [ 0   0  1  0 ]
   * [ a^T 1  0  0 ]
   * [ 0   0  0  0 ]
   * [ 0   0  0  1 ]
   * --------------
   * [ 0   0 -1 -1 ]
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
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");

    linalg::Matrix<T, 7, Dom + 3> F;
    linalg::Vector<T, 7> g;

    if constexpr (Tag == CurvatureType::CONVEX) {
      F(0)(Dom) = T(-1);
      F(3)(Dom) = T(-1);
    } else {
      F(0)(Dom) = T(1);
      F(3)(Dom) = T(1);
    }

    F(2)(Dom + 1) = T(1);
    F = linalg::set_block<3, 4, 0, Dom>(F, child_coord.linear_part());
    F(5)(3) = T(1);
    F(6)(2) = T(-1);
    F(6)(3) = T(-1);

    g(1) = T(1);
    g(3) = child_coord.translation()(0);
    g(4) = T(1);
    g(6) = T(1);

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto new_builder =
        conic::CanonicalFormBuilder<conic::builder_dom_t<decltype(builder)>,
                                    decltype(coord_map_up), decltype(child_cb),
                                    decltype(child_builder.eq_map)>(
            coord_map_up, child_cb, child_builder.eq_map);
    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 3>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto cones = std::make_tuple(exp_cone, exp_cone, rplus);

    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-3};
};

template <typename Child>
Softplus(const Child&)
    -> Softplus<SoftplusMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>>,
                atom_decay_t<Child>>;

} // namespace convex
