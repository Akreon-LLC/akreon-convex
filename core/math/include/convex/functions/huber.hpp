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

/// @brief alias for domain on huber function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HuberDom = ConvexDomain<Space, SignType::FREE>;

/// @brief alias for codomain on huber function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HuberCodom = ConvexDomain<Space, SignType::NONNEG>;

/// @brief alias for concave codomain on huber function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HuberConcaveCodom = ConvexDomain<Space, SignType::NONPOS>;

/// @brief alias for meta for huber function
template <linalg::RnSpace Space>
using HuberMeta =
    ConvexAtomMeta<HuberDom<Space>, HuberCodom<Space>, CurvatureType::CONVEX>;

/// @brief alias for meta for huber function
template <linalg::RnSpace Space>
using HuberConcaveMeta =
    ConvexAtomMeta<HuberDom<Space>, HuberConcaveCodom<Space>,
                   CurvatureType::CONCAVE>;

/**
 * @brief huber loss convex atom
 */
template <typename Meta, typename Child>
struct Huber : Atom<Meta, Huber<Meta, Child>, Child> {
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = Atom<Meta, Huber<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief curvature tag
  static constexpr CurvatureType Tag = meta_curv_v<Meta>;

  /**
   * @brief CTOR
   *
   * @param child
   */
  constexpr explicit Huber(const T& delta_in, const Child& child_in)
      : ConvexBase(child_in),
        delta(delta_in),
        lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
    assert(delta_in <= T(0) && "Huber expects delta <= 0");
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Huber(const T& delta_in, const Child& child_in,
                           const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), delta(delta_in), lift_map(map_in) {
    assert(delta_in <= T(0) && "Huber expects delta <= 0");
  }

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
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
   * @param point x \in domain
   *
   * @return y (scalar \in R)
   */
  constexpr std::optional<T> eval(const PointT& x) const {
    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if (linalg::constexpr_abs(x(0)) <= delta) {
      if constexpr (Tag == CurvatureType::CONVEX)
        return alpha * T(0.5) * x(0) * x(0) + beta;
      else
        return T(-0.5) * alpha * x(0) * x(0) + beta;
    } else {
      if constexpr (Tag == CurvatureType::CONVEX)
        return delta * (linalg::constexpr_abs(x(0)) - T(0.5) * delta);
      else
        return T(-1) * delta * (linalg::constexpr_abs(x(0)) - T(0.5) * delta);
    }
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
      using Space = dom_space_t<meta_dom_t<Meta>>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Huber<HuberMeta<Space>, Child>(
            delta, std::get<0>(this->children), new_map);
      } else {
        return Huber<HuberConcaveMeta<Space>, Child>(
            delta, std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Huber<Meta, Child>(delta, std::get<0>(this->children), new_map);
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
    return Huber<Meta, Child>(delta, std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize impl
   *
   * (1) f(x) = 0.5 x^2 if abs(x) <= delta else delta * (abs(x) - 0.5 delta)
   * Piecewise function implictly casted as max(a,b) \equiv epi(max(fn)) \equiv
   * intersections of epigraph of the functions in the piecewise function
   *
   * C = { (x,t) | phi(x) <= t }... lift conditional abs(x) <= delta to:
   * (a) -x <= s... 0 <= s + x \in R+
   * (b) x <= s... 0 <= s - x \in R+
   *
   * huber = max(0.5 s^2, delta * s - 0.5 * delta^2)
   * (i) { (s,t) | 0.5 s^2 <= t }... { (s, 1, t) | s^2 <= t } \in RSOC
   * (ii) { (s,t) | delta * s - 0.5 * delta^2 <= t } \in R+
   * 0 <= t - delta * s + 0.5 delta^2
   *
   * [ a  1     0 ] [ x; s; t ] + [ b; -b; 0; 1; 0; delta^2 ]
   * [ -a 1     0 ]
   * [ 0  1     0 ]
   * [ 0  0     0 ]
   * [ 0  0     1 ]
   * [ 0 -delta 1 ]
   *
   * (2) -f(x) = -0.5 x^2 if abs(x) <= delta else -delta * (abs(x) - 0.5 delta)
   * same (a) & (b)
   *
   * (i) { (s,t) | 0.5 s^2 <= -t }... (s, 1, -t) \in RSOC
   * (ii) { (s,t) | delta * s - 0.5 delta^2 <= -t } \in R+
   *
   * [ a  1      0 ] [ x; s; t ] + [ b; -b; 0; 1; 0; delta^2 ]
   * [ -a 1      0 ]
   * [ 0  1      0 ]
   * [ 0  0      0 ]
   * [ 0  0     -1 ]
   * [ 0 -delta -1 ]
   *
   * @param builder
   *
   * @return new builder
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

    linalg::Matrix<T, 6, Dom + 2> F;
    linalg::Vector<T, 6> g;
    F = linalg::set_block<0, 1, 0, Dom>(F, child_coord.linear_part());
    F(0)(Dom) = T(1);
    F = linalg::set_block<1, 2, 0, Dom>(F, T(-1) * child_coord.linear_part());
    F(1)(Dom) = T(1);
    F(2)(Dom) = T(1);
    if constexpr (Tag == CurvatureType::CONVEX) {
      F(4)(Dom + 1) = T(1);
      F(5)(Dom + 1) = T(1);
    } else {
      F(4)(Dom + 1) = T(-1);
      F(5)(Dom + 1) = T(-1);
    }
    F(5)(Dom) = T(-1) * delta;

    g(0) = child_coord.translation()(0);
    g(1) = T(-1) * child_coord.translation()(0);
    g(3) = T(1);
    g(5) = delta * delta;

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto new_builder =
        ConeBlockBuilder<builder_dom_t<decltype(builder)>,
                         decltype(coord_map_up), decltype(child_cb)>(
            coord_map_up, child_cb);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto rsoc = conic::RotatedSecondOrderCone<linalg::Vector<T, 3>>();
    const auto cones = std::make_tuple(rplus, rplus, rsoc, rplus);

    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  T delta;
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

}  // namespace convex
