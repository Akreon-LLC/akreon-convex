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
 * @brief QuadForm
 *
 * 0.5 x^T P x + q^T x
 *
 * Wraps ||L^T x||_2^2 + q^T x
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
   * @param x \in R^n (var)
   *
   * @return 0.5 * ||L^T x||_2^2 + q^T x
   */
  template <AtomLike Var>
  constexpr auto operator()(const Mat& P, const Vec& q, const Var& x) {
    static_assert(is_var_type_v<Var>, "Quad form must operate on a var type");
    static_assert(
        std::same_as<dom_space_t<meta_codom_t<atom_meta_t<Var>>>, Space>,
        "Invalid quad operator mismatch on vector types");

    const auto chol = linalg::Cholesky<Mat>()(P);
    AK_ASSERT(chol.is_valid() && chol.is_psd() && "Chol(P) is not SPD");

    const auto LT_x = x.mul(linalg::transpose(chol.G()));

    const auto l2 =
        Norm<NormMeta<Space, Singleton>, Var, linalg::L2<Vec>>(LT_x);
    const auto sq = Square<SquareMeta<Singleton>, decltype(l2)>(l2);
    const auto half_sq = sq.template mul<T(0.5)>();

    const auto weighted_sum = WeightedSum<Space>()(q, x);

    using SumDom = linalg::Matrix<T, 1, 2>;
    const auto sum = Sum<SumMeta<SumDom, Singleton>, decltype(half_sq),
                         decltype(weighted_sum)>(half_sq, weighted_sum);

    return sum;
  }
};

}  // namespace convex
