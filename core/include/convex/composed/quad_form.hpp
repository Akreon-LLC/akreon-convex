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
#include "convex/functions/norm.hpp"
#include "convex/functions/square.hpp"
#include "convex/sum.hpp"
#include "linalg/operator/cholesky.hpp"
#include "weighted_sum.hpp"

namespace convex {

/**
 * @brief ScalarQuadForm
 *
 * 0.5 p x^2 + q x + r
 */
template <linalg::FieldLike T>
struct ScalarQuadForm {
  /// @brief aliases from type info
  using Space = linalg::Matrix<T, 1, 1>;
  using Singleton = Space;

  /**
   * @brief operator overload
   *
   * @param p (scalar)
   * @param q (scalar)
   * @param r (scalar)
   * @param x (child)
   *
   * @return sum object embedding QP
   */
  template <AtomLike Atom>
  constexpr auto operator()(const T& p, const T& q, const T& r, const Atom& x) {
    static_assert(
        std::same_as<dom_space_t<meta_codom_t<atom_meta_t<Atom>>>, Space>,
        "Scalar quad form requires scalar codomain");
    static_assert(is_fn_type_v<Atom> || is_index_type_v<Atom>,
                  "Scalar quad form must operate on scalar affine atoms");

    const T alpha = linalg::constexpr_sqrt_safe(T(0.5) * p);
    const auto scaled_x = x.mul(alpha);
    const auto quadratic =
        Square<SquareMeta<Singleton>, decltype(scaled_x)>(scaled_x);
    const auto linear = x.mul(q);

    using SumDom = linalg::Matrix<T, 1, 2>;
    const auto sum =
        Sum<SumMeta<SumDom, Singleton>, decltype(quadratic), decltype(linear)>(
            quadratic, linear);
    return sum.add(r);
  }

  /**
   * @brief operator overload (r = 0)
   *
   * @param p (scalar)
   * @param q (scalar)
   * @param x (child)
   *
   * @return sum object embedding qp
   */
  template <AtomLike Atom>
  constexpr auto operator()(const T& p, const T& q, const Atom& x) {
    return this->operator()(p, q, T(0), x);
  }
};

/**
 * @brief QuadForm
 *
 * 0.5 x^T P x + q^T x + r
 *
 * Wraps ||L^T x||_2^2 + q^T x + r when 0.5 P = L L^T.
 */
template <linalg::RnSpace Space>
struct QuadForm {
  /// @brief alias and helpers
  using T = linalg::space_type_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;

  /// @brief matrix type for P
  using Mat = linalg::Matrix<T, N, N>;

  /// @brief vector type for q
  using Vec = linalg::Vector<T, N>;

  /// @brief singleton
  using Singleton = linalg::Matrix<T, 1, 1>;

  /**
   * @brief operator overload
   *
   * x^T P x = x^T L L^T x
   *
   * @param P \in S^n_+
   * @param q \in R^n
   * @param r \in R
   * @param x \in R^n (var)
   *
   * @return 0.5 x^T P x + q^T x + r
   */
  template <AtomLike Var>
  constexpr auto operator()(const Mat& P, const Vec& q, const T& r,
                            const Var& x) {
    static_assert(N > 1,
                  "Use ScalarQuadForm<T> for scalar quadratic expressions");
    static_assert(is_var_type_v<Var>, "Quad form must operate on a var type");
    static_assert(
        std::same_as<dom_space_t<meta_codom_t<atom_meta_t<Var>>>, Space>,
        "Invalid quad operator mismatch on vector types");

    const auto chol = linalg::Cholesky<Mat>(T(0.5) * P);
    AK_ASSERT(chol.is_valid() && chol.is_psd() && "Chol(P) is not PSD");

    const auto LT_x = x.mul(linalg::transpose(chol.G()));
    const auto l2 =
        Norm<NormMeta<Space, Singleton>, Var, linalg::L2<Vec>>(LT_x);
    const auto quadratic = Square<SquareMeta<Singleton>, decltype(l2)>(l2);
    const auto weighted_sum = WeightedSum<Space>()(q, x);

    using SumDom = linalg::Matrix<T, 1, 2>;
    const auto sum = Sum<SumMeta<SumDom, Singleton>, decltype(quadratic),
                         decltype(weighted_sum)>(quadratic, weighted_sum);
    return sum.add(r);
  }

  /**
   * @brief operator overload without constant offset
   *
   * @param P \in S^n_+
   * @param q \in R^n
   * @param x \in R^n (var)
   *
   * @return 0.5 x^T P x + q^T x
   */
  template <AtomLike Var>
  constexpr auto operator()(const Mat& P, const Vec& q, const Var& x) {
    return this->operator()(P, q, T(0), x);
  }
};

}  // namespace convex
