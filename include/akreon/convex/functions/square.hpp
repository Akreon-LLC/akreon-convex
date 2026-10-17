/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "akreon/convex/canonical_form_builder.hpp"
#include "akreon/convex/cones/soc.hpp"
#include "akreon/convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on square function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SquareDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on square function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SquareCodom = ConvexDomain<Space, VarType::NONNEG>;

/// @brief alias for concave codomain on square function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using SquareConcaveCodom = ConvexDomain<Space, VarType::NONPOS>;

/// @brief alias for meta for square function
template <linalg::RnSpace Space>
using SquareMeta =
    ConvexAtomMeta<SquareDom<Space>, SquareCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for square function
template <linalg::RnSpace Space>
using SquareConcaveMeta =
    ConvexAtomMeta<SquareDom<Space>, SquareConcaveCodom<Space>,
                   CurvatureType::CONCAVE>;

/**
 * @brief square function
 *
 * f(x) = x*x, x in R_+
 */
template <typename Meta, typename Child>
  requires(dom_size_v<meta_dom_t<Meta>> == 1)
struct Square : Atom<Meta, Square<Meta, Child>, Child> {
  /// @brief bring into scope convex fn alias / helpers
  using ConvexBase = Atom<Meta, Square<Meta, Child>, Child>;
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
  constexpr explicit Square(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Square(const Child& child_in,
                            const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), lift_map(map_in) {}

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
    static constexpr VarType sign =
        dom_vartype_v<meta_codom_t<atom_meta_t<Child>>>;
    if constexpr (Tag == CurvatureType::CONVEX) {
      if constexpr (sign == VarType::NONNEG) {
        return MonotonicityType::NONDEC;
      } else if constexpr (sign == VarType::NONPOS) {
        return MonotonicityType::NONINC;
      } else {
        return MonotonicityType::UNKNOWN;
      }
    } else {
      if constexpr (sign == VarType::NONNEG) {
        return MonotonicityType::NONINC;
      } else if constexpr (sign == VarType::NONPOS) {
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
  constexpr T operator()(const PointT& x) const { return eval(x); }

  /**
   * @brief eval impl
   *
   * @param x in R
   *
   * @return x^2 in R
   */
  constexpr T eval(const PointT& x) const {
    const auto alpha = lift_map.linear_part()(0, 0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX)
      return alpha * x(0) * x(0) + beta;
    else return T(-1) * alpha * x(0) * x(0) + beta;
  }

  /**
   * @brief mul
   *
   * @return square fn (updated map)
   */
  template <T Scalar>
  constexpr auto mul() const {
    if constexpr (Scalar < T(0)) {
      const auto alpha = T(-1) * Scalar;
      affine::AffineMap<LiftedVarMat<T>> new_map(alpha * lift_map.linear_part(),
                                                 lift_map.translation());
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Square<SquareMeta<Space>, Child>(std::get<0>(this->children),
                                                new_map);
      } else {
        return Square<SquareConcaveMeta<Space>, Child>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Square<Meta, Child>(std::get<0>(this->children), new_map);
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return square fn (updated map)
   */
  constexpr auto add(T beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return Square<Meta, Child>(std::get<0>(this->children), new_map);
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
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    const auto child_builder =
        std::get<0>(this->children)
            .template canonicalize<AtomType::FN>(builder);
    const auto child_coord = child_builder.coord_map;

    using AffineMatT = affine::affine_mat_t<decltype(child_coord)>;
    static constexpr std::size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr std::size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == 1,
                  "input affine map from cone block has invalid codom dim for "
                  "scalar fn domain (i.e., != 1");
    linalg::Matrix<T, 3, Dom + 1> F{};
    F = linalg::set_block<0, 1, 0, Dom>(F, child_coord.linear_part());
    linalg::Vector<T, 3> g{};
    g(0) = child_coord.translation()(0);
    g(1) = T(0.5);

    if constexpr (Tag == CurvatureType::CONVEX) {
      F(2, Dom) = T(1);
    } else {
      F(2, Dom) = T(-1);
    }

    const auto rsoc_coords = affine::AffineMap<decltype(F)>(F, g);
    const auto coord_map_up =
        conic::Rotated2SOC<decltype(rsoc_coords)>()(rsoc_coords);
    const auto rsoc = conic::SecondOrderCone<linalg::Vector<T, 3>>();
    const auto cones = std::make_tuple(rsoc);
    const auto emission =
        conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

    return child_builder.template realize<Type>(idx, lift_map, emission);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

template <typename Child>
Square(const Child&)
    -> Square<SquareMeta<dom_space_t<meta_codom_t<atom_meta_t<Child>>>>,
              atom_decay_t<Child>>;

} // namespace convex
