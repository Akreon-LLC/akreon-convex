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

/// @brief alias for domain on abs function
template <linalg::RnSpace Space>
using NormDom = ConvexDomain<Space, VarType::FREE>;

/// @brief alias for codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using NormCodom = ConvexDomain<Space, VarType::NONNEG>;

/// @brief alias for concave codomain on abs function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using NormConcaveCodom = ConvexDomain<Space, VarType::NONPOS>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using NormMeta =
    ConvexAtomMeta<NormDom<Dom>, NormCodom<Codom>, CurvatureType::CONVEX>;

/// @brief alias for meta for abs function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using NormConcaveMeta = ConvexAtomMeta<NormDom<Dom>, NormConcaveCodom<Codom>,
                                       CurvatureType::CONCAVE>;

/**
 * @brief norm convex function wrapper
 */
template <typename Meta, typename Child, linalg::NormFunction NormFunc>
  requires(dom_size_v<meta_dom_t<Meta>> == linalg::norm_dim_v<NormFunc>)
struct Norm : Atom<Meta, Norm<Meta, Child, NormFunc>, Child> {
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = Atom<Meta, Norm<Meta, Child, NormFunc>, Child>;
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
  constexpr explicit Norm(const Child& child_in)
      : ConvexBase(child_in), lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
  }

  /**
   * @brief CTOR
   *
   * @param child
   * @param map
   */
  constexpr explicit Norm(const Child& child_in,
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
   * @param point x in domain
   *
   * @return y (scalar in R)
   */
  constexpr T eval(const PointT& x) const {
    const auto alpha = lift_map.linear_part()(0, 0);
    const auto beta = lift_map.translation()(0);
    if constexpr (Tag == CurvatureType::CONVEX)
      return alpha * NormFunc()(x) + beta;
    else return T(-1) * alpha * NormFunc()(x) + beta;
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

      using Dom = dom_space_t<meta_dom_t<Meta>>;
      using Codom = dom_space_t<meta_codom_t<Meta>>;
      if constexpr (Tag == CurvatureType::CONCAVE) {
        return Norm<NormMeta<Dom, Codom>, Child, NormFunc>(
            std::get<0>(this->children), new_map);
      } else {
        return Norm<NormConcaveMeta<Dom, Codom>, Child, NormFunc>(
            std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return Norm<Meta, Child, NormFunc>(std::get<0>(this->children), new_map);
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
    return Norm<Meta, Child, NormFunc>(std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize_impl
   *
   * L2 Norm Function
   * (1) f(x) = ||x||_2
   * K = { (x, t) | ||x||_2 <= t }
   * K_soc = { (u, v) | ||u||_2 <= v, v >= 0 }
   *
   * let u = x, v = t
   * [ A 0 ] [ x; t ] + [ b; 0 ] -> (A x + b, t)
   * [ 0 1 ]
   *
   * (2) -f(x) = -||x||_2
   * K = { (x, t) | -||x||_2 >= t }... { (x, t) | ||x||_2 <= -t }
   *
   * [ A  0 ] [ x; t ] + [ b; 0 ] -> (A x + b, -t)
   * [ 0 -1 ]
   *
   * Linf Norm Function
   * (1) f(x) = ||x||_inf
   * K = { (x, t) | ||x||_inf <= t }
   *
   * -t <= x_i <= t... for all i
   * 0 <= x_i + t... nonneg orthant cone(s)
   * 0 <= t - x_i... nonneg orthant cone(s)
   *
   * [  A  1 ] [ x; t ] + [ b; -b ]
   * [ -A  1 ]
   *
   * (2) -f(x) = -||x||_inf
   * K = { (x, t) | -||x||_inf >= t }... { (x, t) | ||x||_inf <= -t }
   *
   * [  A  -1 ] [ x; t ] + [ b; -b ]
   * [ -A  -1 ]
   *
   * L1 Norm Function
   * (1) f(x) = ||x||_1
   * K = { (x, t) | ||x||_1 <= t }
   *
   * let u_i >= |x_i|... convex constraint
   * sum(u_i) <= t... convex constraint
   *
   * -u_i <= y_i <= u_i
   * 0 <= y_i + u_i... nonneg orthant cone
   * 0 <= u_i - y_i... nonneg orthant cone
   * 0 <= t - 1^T u... nonneg orthant cone
   *
   * [ A  I   0 ] [ x; u; t ] + [ b; -b; 0 ]
   * [ -A I   0 ]
   * [ 0 -1^T 1 ]
   *
   * (2) -f(x) = -||x||_1
   * K = { (x, t) | -||x||_1 >= t }... { (x, t) | ||x||_1 <= -t }
   *
   * [ A  I    0 ] [ x; u; t ] + [ b; -b; 0 ]
   * [ -A I    0 ]
   * [ 0 -1^T -1 ]
   * [ 0  I    0 ]
   *
   * @param cone map builder
   * @param coord map
   * @param cone block
   *
   * @return cone block
   */
  template <AtomType Type, conic::BuilderLike Builder>
  constexpr auto canonicalize_impl(const Builder& builder) const {
    using Vec = linalg::Vector<T, N>;
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

    if constexpr (std::same_as<NormFunc, linalg::L2<Vec>>) {
      linalg::Matrix<T, N + 1, Dom + 1> F;
      F = linalg::set_block<0, N, 0, Dom>(F, child_coord.linear_part());

      linalg::Vector<T, N + 1> g;
      for (std::size_t i = 0; i < N; ++i) g(i) = child_coord.translation()(i);

      if constexpr (Tag == CurvatureType::CONCAVE) {
        F(N, N) = T(-1);
      } else {
        F(N, N) = T(1);
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);

      const auto soc = conic::SecondOrderCone<linalg::Vector<T, N + 1>>();
      const auto cones = std::make_tuple(soc);
      const auto emission =
          conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

      return child_builder.template realize<Type>(idx, lift_map, emission);
    } else if constexpr (std::same_as<NormFunc, linalg::Linf<Vec>>) {
      linalg::Matrix<T, 2 * N, Dom + 1> F;
      F = linalg::set_block<0, N, 0, N>(F, child_coord.linear_part());
      F = linalg::set_block<N, 2 * N, 0, N>(F,
                                            T(-1) * child_coord.linear_part());

      linalg::Vector<T, 2 * N> g;
      for (std::size_t i = 0; i < N; ++i) g(i) = child_coord.translation()(i);
      for (std::size_t i = 0; i < N; ++i)
        g(i + N) = T(-1) * child_coord.translation()(i);

      if constexpr (Tag == CurvatureType::CONVEX) {
        for (std::size_t i = 0; i < 2 * N; ++i) F(i, N) = T(1);
      } else {
        for (std::size_t i = 0; i < 2 * N; ++i) F(i, N) = T(-1);
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);

      const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
      const auto cones = std::make_tuple(rplus, rplus);
      const auto emission =
          conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

      return child_builder.template realize<Type>(idx, lift_map, emission);
    } else {
      linalg::Matrix<T, 3 * N + 1, 2 * Dom + 1> F;
      const auto I = linalg::make_identity<T, N>();
      F = linalg::set_block<0, N, 0, N>(F, child_coord.linear_part());
      F = linalg::set_block<N, 2 * N, 0, N>(F,
                                            T(-1) * child_coord.linear_part());
      F = linalg::set_block<0, N, N, 2 * N>(F, I);
      F = linalg::set_block<N, 2 * N, N, 2 * N>(F, I);
      for (std::size_t i = 0; i < N; ++i) F(2 * N, N + i) = T(-1);
      F = linalg::set_block<2 * N + 1, 3 * N + 1, N, 2 * N>(F, I);

      linalg::Vector<T, 3 * N + 1> g;
      for (std::size_t i = 0; i < N; ++i) g(i) = child_coord.translation()(i);
      for (std::size_t i = 0; i < N; ++i)
        g(i + N) = T(-1) * child_coord.translation()(i);

      if constexpr (Tag == CurvatureType::CONVEX) {
        F(2 * N, 2 * N) = T(1);
      } else {
        F(2 * N, 2 * N) = T(-1);
      }

      const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);

      const auto rplus1 = conic::NonnegOrthantCone<linalg::Vector<T, N>>();
      const auto rplus2 = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
      const auto cones = std::make_tuple(rplus1, rplus1, rplus2, rplus1);
      const auto emission =
          conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

      return child_builder.template realize<Type>(idx, lift_map, emission);
    }
  }

  /** @brief member vars */
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

} // namespace convex
