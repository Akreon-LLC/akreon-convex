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

#include "config/assert.hpp"
#include "convex/cone_block_builder.hpp"
#include "convex/cones/soc.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on huber function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HuberDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on huber function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HuberCodom = ConvexDomain<Space, VarType::NONNEG>;

/// @brief alias for concave codomain on huber function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using HuberConcaveCodom = ConvexDomain<Space, VarType::NONPOS>;

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
    AK_ASSERT(delta_in >= T(0) && "Huber expects delta >= 0");
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
    AK_ASSERT(delta_in >= T(0) && "Huber expects delta >= 0");
  }

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
   * @param point x \in domain
   *
   * @return y (scalar \in R)
   */
  constexpr T eval(const PointT& x) const {
    const auto alpha = lift_map.linear_part()(0)(0);
    const auto beta = lift_map.translation()(0);
    if (linalg::constexpr_abs(x(0)) <= delta) {
      if constexpr (Tag == CurvatureType::CONVEX)
        return alpha * T(0.5) * x(0) * x(0) + beta;
      else
        return T(-0.5) * alpha * x(0) * x(0) + beta;
    } else {
      if constexpr (Tag == CurvatureType::CONVEX)
        return alpha * delta * (linalg::constexpr_abs(x(0)) - T(0.5) * delta) +
               beta;
      else
        return T(-1) * alpha * delta *
                   (linalg::constexpr_abs(x(0)) - T(0.5) * delta) +
               beta;
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
   * L_\delta(a) = \piecewise{
   *    0.5 a^2, for |a| <= \delta,
   *    \delta(|a| - 0.5 \delta) otherwise
   * }
   *
   * Variational Form
   * h(x, \delta) = min_{u, v, \gamma, q, t} t
   * s.t. x = u + v, \gamma >= |v|, q >= u^2, t >= q + \delta \gamma
   *
   * Convex
   * [  a -1 -1  0      0  0 ] \hat{x} + [  b ]
   * [ -a  1  1  0      0  0 ]          [ -b ]
   * [  0  1  0  1      0  0 ]          [  0 ]
   * [  0 -1  0  1      0  0 ]          [  0 ]
   * [  0  0  0  1      0  0 ]          [  0 ]
   * [  0  0  0 -\delta -1 1 ]          [  0 ]
   * [  0  0  1  0      0  0 ]          [  0 ]
   * [  0  0  0  0      0  0 ]          [ 1/2 ]
   * [  0  0  0  0      1  0 ]          [  0 ]
   *
   * Concave
   * row 5 becomes [ 0 0 0 -\delta -1 -1 ] so t <= -(q + \delta \gamma)
   *
   * \hat{x} = [ x; v; u; \gamma; q; t ]
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

    linalg::Matrix<T, 9, Dom + 5> F{};
    linalg::Vector<T, 9> g{};
    const auto a = child_coord.linear_part();
    const auto b = child_coord.translation();
    F = linalg::set_block<0, 1, 0, Dom>(F, a);
    F = linalg::set_block<1, 2, 0, Dom>(F, T(-1) * a);
    F(0)(Dom) = T(-1), F(0)(Dom + 1) = T(-1);
    F(1)(Dom) = T(1), F(1)(Dom + 1) = T(1);
    F(2)(Dom) = T(1), F(2)(Dom + 2) = T(1);
    F(3)(Dom) = T(-1), F(3)(Dom + 2) = T(1);
    F(4)(Dom + 2) = T(1);
    F(5)(Dom + 2) = T(-1) * delta;
    F(5)(Dom + 3) = T(-1);
    g(0) = b(0);
    g(1) = T(-1) * b(0);

    linalg::Matrix<T, 3, Dom + 5> F_rsoc{};
    F_rsoc(0)(Dom + 1) = T(1);
    F_rsoc(2)(Dom + 3) = T(1);

    if constexpr (Tag == CurvatureType::CONVEX) {
      F(5)(Dom + 4) = T(1);
    } else {
      F(5)(Dom + 4) = T(-1);
    }

    linalg::Vector<T, 3> g_rsoc{T(0), T(0.5), T(0)};

    const auto rsoc_coords =
        affine::AffineMap<decltype(F_rsoc)>(F_rsoc, g_rsoc);
    const auto soc_coords =
        conic::Rotated2SOC<decltype(rsoc_coords)>()(rsoc_coords);
    F = linalg::set_block<6, 9, 0, Dom + 5>(F, soc_coords.linear_part());
    const auto g_hat = soc_coords.translation();
    g(6) = g_hat(0);
    g(7) = g_hat(1);
    g(8) = g_hat(2);

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto new_builder =
        conic::ConeBlockBuilder<conic::builder_dom_t<decltype(builder)>,
                                decltype(coord_map_up), decltype(child_cb)>(
            coord_map_up, child_cb);
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 6>>();
    const auto rsoc = conic::SecondOrderCone<linalg::Vector<T, 3>>();
    const auto cones = std::make_tuple(rplus, rsoc);

    return new_builder.template make<Type>(idx, lift_map, cones);
  }

  /** @brief member vars */
  T delta;
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

}  // namespace convex
