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
#include "convex/canonical_form_builder.hpp"
#include "convex/convex_traits.hpp"

namespace convex {

/// @brief alias for domain on kl div function (q in R_++)
template <linalg::RnSpace Space>
using KLDivDom = ConvexDomain<Space, VarType::POS>;

/// @brief alias for codomain on kl div function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using KLDivCodom = ConvexDomain<Space, VarType::NONNEG>;

/// @brief alias for concave codomain on kl div function
template <linalg::RnSpace Space>
  requires(linalg::space_dim_v<Space> == 1)
using KLDivConcaveCodom = ConvexDomain<Space, VarType::NEG>;

/// @brief alias for meta for kl div function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using KLDivMeta =
    ConvexAtomMeta<KLDivDom<Dom>, KLDivCodom<Codom>, CurvatureType::CONVEX>;

/// @brief alias for meta for kl div function
template <linalg::RnSpace Dom, linalg::RnSpace Codom>
using KLDivConcaveMeta = ConvexAtomMeta<KLDivDom<Dom>, KLDivConcaveCodom<Codom>,
                                        CurvatureType::CONCAVE>;

/**
 * @brief kl divergence D_kl(p || q) = sum(p_i log(p_i / q_i))
 */
template <typename Meta, typename Child>
struct KLDivergence : Atom<Meta, KLDivergence<Meta, Child>, Child> {
  /// @brief bring into scope the convex fn alias / helpers
  using ConvexBase = Atom<Meta, KLDivergence<Meta, Child>, Child>;
  using ConvexBase::N;
  using typename ConvexBase::PointT;
  using typename ConvexBase::T;

  /// @brief curvature tag
  static constexpr CurvatureType Tag = meta_curv_v<Meta>;

  /**
   * @brief CTOR
   *
   * @param p fixed
   * @param child q
   */
  constexpr explicit KLDivergence(const PointT& p_in, const Child& child_in)
      : ConvexBase(child_in), p(p_in),
        lift_map(affine::make_map_id<LiftedVarMat<T>>()) {
    for (const auto& x : p) {
      AK_ASSERT(x < T(0) &&
                "KLDivergence expects each fixed p entry to satisfy x < 0");
      (void)x;
    }
  }

  /**
   * @brief CTOR
   *
   * @param p fixed
   * @param child
   * @param map
   */
  constexpr explicit KLDivergence(
      const PointT& p_in, const Child& child_in,
      const affine::AffineMap<LiftedVarMat<T>>& map_in)
      : ConvexBase(child_in), p(p_in), lift_map(map_in) {
    for (const auto& x : p) {
      AK_ASSERT(x < T(0) &&
                "KLDivergence expects each fixed p entry to satisfy x < 0");
      (void)x;
    }
  }

  /**
   * @brief mono
   *
   * @return monotonicity
   */
  constexpr MonotonicityType monotonicity() const {
    return MonotonicityType::UNKNOWN;
  }

  /**
   * @brief eval via operator overload
   *
   * @param point p
   * @param point q
   *
   * @return result
   */
  constexpr T operator()(const PointT& q) const { return eval(q); }

  /**
   * @brief eval impl
   *
   * @param point p in dom f
   * @param point q in dom f
   *
   * @return y in R
   */
  constexpr T eval(const PointT& q) const {
    T res(0);
    for (size_t i = 0; i < N; ++i) {
      AK_ASSERT(p(i) >= T(0) && q(i) > T(0) && "Invalid domain for kl div");
      res += p(i) * linalg::constexpr_ln(p(i) / q(i)).value();
    }

    if constexpr (Tag == CurvatureType::CONVEX) return res;
    else return T(-1) * res;
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
        return KLDivergence<KLDivMeta<Dom, Codom>, Child>(
            p, std::get<0>(this->children), new_map);
      } else {
        return KLDivergence<KLDivConcaveMeta<Dom, Codom>, Child>(
            p, std::get<0>(this->children), new_map);
      }
    } else {
      affine::AffineMap<LiftedVarMat<T>> new_map(
          Scalar * lift_map.linear_part(), lift_map.translation());
      return KLDivergence<Meta, Child>(p, std::get<0>(this->children), new_map);
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
    return KLDivergence<Meta, Child>(p, std::get<0>(this->children), new_map);
  }

  /**
   * @brief canonicalize impl
   *
   * (1) f(x) = sum(p_i log (p_i / q_i))
   * C = { (p,q,t) | p_i log(p_i / q_i) <= s_i, sum(s_i) <= t, q_i > 0 }
   * exp(-s_i / p_i) <= q_i / p_i
   * p_i exp(-s_i/p_i) <= q_i
   *
   * Exp cone: let x = -s_i, y = p_i, z = q_i
   * Nonneg cone: 0 <= t - 1^T s
   *
   * [ 0 0 -I   0 ] [ p; q; s; t ] + [ 0; 0; b; 0]
   * [ I 0  0   0 ]
   * [ 0 A  0   0 ]
   *  ------------
   * [ 0 0 -1^T 1 ]
   *
   * (2) -f(x) = -sum(p_i log (p_i / q_i))
   *
   * Exp cone: let x = s_i, y = p_i, z = q_i
   * Nonneg cone: 0 <= t - 1^T s
   *
   * [ 0 0  I   0 ] [ p; q; s; t ] + [ 0; 0; b; 0]
   * [ I 0  0   0 ]
   * [ 0 A  0   0 ]
   *  ------------
   * [ 0 0 -1^T 1 ]
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

    using AffineMatT = affine::affine_mat_t<decltype(child_coord)>;
    static constexpr size_t Dom = linalg::mat_cols_v<AffineMatT>;
    static constexpr size_t Codom = linalg::mat_rows_v<AffineMatT>;
    static_assert(Codom == N,
                  "Input affine map from cone block has invalid dom which is a "
                  "mismatch between domain in the convex function");
    static_assert(Dom == N,
                  "Expecting child affine map to have square linear part");
    linalg::Matrix<T, 3 * N + 1, 3 * N + 1> F;
    linalg::Vector<T, 3 * N + 1> g;
    for (size_t i = 0; i < N; ++i) g(2 * N + i) = child_coord.translation()(i);

    const auto I = linalg::make_identity<T, N>();
    if constexpr (Tag == CurvatureType::CONVEX)
      F = linalg::set_block<0, N, 2 * N, 3 * N>(F, T(-1) * I);
    else F = linalg::set_block<0, N, 2 * N, 3 * N>(F, I);

    F = linalg::set_block<N, 2 * N, 0, N>(F, I);
    F = linalg::set_block<2 * N, 3 * N, N, 2 * N>(F, child_coord.linear_part());
    for (size_t i = 0; i < N; ++i) F(3 * N)(2 * N + i) = T(-1);
    F(3 * N)(3 * N) = T(1);

    const auto coord_map_up = affine::AffineMap<decltype(F)>(F, g);
    const auto exp_cone = conic::ExpCone<linalg::Vector<T, 3 * N>>();
    const auto rplus = conic::NonnegOrthantCone<linalg::Vector<T, 1>>();
    const auto cones = std::make_tuple(exp_cone, rplus);
    const auto emission =
        conic::LocalEmission(coord_map_up, cones, child_builder.eq_map);

    return child_builder.template realize<Type>(idx, lift_map, emission);
  }

  /** @brief member vars */
  PointT p;
  affine::AffineMap<LiftedVarMat<T>> lift_map;
  const int idx{-1};
};

template <typename Point, typename Child>
KLDivergence(const Point&, const Child&) -> KLDivergence<
    KLDivMeta<
        dom_space_t<meta_codom_t<atom_meta_t<Child>>>,
        linalg::Matrix<dom_type_t<meta_codom_t<atom_meta_t<Child>>>, 1, 1>>,
    atom_decay_t<Child>>;

} // namespace convex
