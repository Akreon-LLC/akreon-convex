/*
 * Copyright © 2025-2026 Akreon LLC
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>

#include "akreon/convex/canonical_form_builder.hpp"
#include "akreon/convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on abs function
template <linalg::RnSpace Space>
using LogSumExpDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogSumExpCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for concave codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using LogSumExpConcaveCodom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using LogSumExpMeta = ConvexAtomMeta<LogSumExpDom<Dom>, LogSumExpCodom<Codom>,
                                     CurvatureType::CONVEX>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using LogSumExpConcaveMeta =
    ConvexAtomMeta<LogSumExpDom<Dom>, LogSumExpConcaveCodom<Codom>,
                   CurvatureType::CONCAVE>;

/**
 * @brief log sum exp = smooth max function
 *
 * lse(x) = log(sum_i^n(e^{x_i})) which is a smooth mapping from R^n to R
 */
template <typename Meta, typename Child>
struct LogSumExp : Atom<Meta, LogSumExp<Meta, Child>, Child> {
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = Atom<Meta, LogSumExp<Meta, Child>, Child>;
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
  constexpr explicit LogSumExp(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit LogSumExp(const Child& child_in,
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
   * @param point x in dom f
   *
   * @return y in R
   */
  constexpr T eval(const PointT& x) const {
    T sum(0);
    std::for_each(x.begin(), x.end(),
                  [&](const auto& pt) { sum += linalg::constexpr_exp(pt); });

    const auto res = linalg::constexpr_ln_safe(sum);

    if constexpr (Tag == CurvatureType::CONVEX) return res;
    else { return T(-1) * res; }
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
        return LogSumExp<LogSumExpMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
      } else {
        return LogSumExp<LogSumExpConcaveMeta<Dom, Codom>, Child>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return LogSumExp<Meta, Child>(std::get<0>(this->children), new_map);
    }
  }

  /**
   * @brief add
   *
   * @param scalar
   *
   * @return abs fn (updated map)
   */
  constexpr auto add(T beta) const {
    LiftedVarBias<T> bias{beta + lift_map.translation()(0)};
    affine::AffineMap<LiftedVarMat<T>> new_map(lift_map.linear_part(), bias);
    return LogSumExp<Meta, Child>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize impl
   *
   * (1) f(x) = log(sum_i(exp(x_i)))
   * K = { (x, t) | log(sum_i(exp(x_i))) <= t }
   * K = { (x, t) | sum_i(exp(x_i)) <= exp(t) }
   * K = { (x, t) | sum_i(exp(x_i)) / exp(t) <= 1 }
   * K = { (x, t) | sum_i(exp(x_i - t)) <= 1 }
   * K = { (x, t, s) | exp(x_i - t) <= s_i, 1^T s <= 1 }
   *
   * Exp cone : { (u, v, w) | v exp(u/v) <= w, v > 0 } union { (u, 0, w) | u <=
   * 0, w >= 0 }
   *
   * let u = x_i - t, v = 1, w = s_i... n exp cones
   * 0 <= 1 - 1^T s ... 1 nonneg orthant cone
   *
   * [ A -1    0 ] [ x; t; s ] + [ b; 1; 0 | 1 ] -> (x_i - t, 1, s_i) + rplus
   * [ 0  0    0 ]
   * [ 0  0    I ]
   * -------------
   * [ 0  0 -1^T ]
   *
   * (2) -f(x) = -log(sum_i(exp(x_i)))
   * K = { (x, t) | sum_i(exp(x_i + t)) <= 1 }
   * K = { (x, t, s) | exp(x_i + t) <= s_i, 1^T s <= 1 }
   *
   * let u = x_i + t, v = 1, w = s_i... n exp cones
   * 0 <= 1 - 1^T s ... 1 nonneg orthant cone
   *
   * [ A  1    0 ] [ x; t; s ] + [ b; 1; 0 | 1 ] -> (x_i + t, 1, s_i) + rplus
   * [ 0  0    0 ]
   * [ 0  0    I ]
   * -------------
   * [ 0  0 -1^T ]
   *
   * @param cone block builder
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
                  "Input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");
    static_assert(Dom == N,
                  "Expecting child affine map to have square linear part");
    linalg::Matrix<T, 3 * N + 1, 2 * Dom + 1> F;
    linalg::Vector<T, 3 * N + 1> g;
    F = linalg::set_block<0, N, 0, Dom>(F, child_coord.linear_part());
    const auto I = linalg::make_identity<T, Dom>();

    if constexpr (Tag == CurvatureType::CONVEX) {
      for (std::size_t i = 0; i < N; ++i) F(i, N) = T(-1);
    } else {
      for (std::size_t i = 0; i < N; ++i) F(i, N) = T(1);
    }
    F = linalg::set_block<2 * N, 3 * N, N + 1, 2 * N + 1>(F, I);
    for (std::size_t i = 0; i < N; ++i) F(3 * N, N + 1 + i) = T(-1);
    for (std::size_t i = 0; i < N; ++i) g(i) = child_coord.translation()(i);
    for (std::size_t i = N; i < 2 * N; ++i) g(i) = T(1);
    g(3 * N) = T(1);

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 3 * N>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto cones = std::make_tuple(exp_cone, rplus);
    const auto emission =
        conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

    return child_builder.template realize<Type>(idx, lift_map, emission);
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-2};
};

template <typename Child>
LogSumExp(const Child&) -> LogSumExp<
    LogSumExpMeta<
        dom_space_t<meta_codom_t<atom_meta_t<Child>>>,
        linalg::Matrix<dom_type_t<meta_codom_t<atom_meta_t<Child>>>, 1, 1>>,
    atom_decay_t<Child>>;

} // namespace convex
