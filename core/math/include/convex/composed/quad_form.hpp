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

#include "convex/cone_block_builder.hpp"
#include "convex/convex_traits.hpp"
#include "convex/functions/norm.hpp"
#include "convex/functions/square.hpp"
#include "convex/sum.hpp"

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

    (void)P, (void)q, (void)x;

    /// TODO - return square(L2(L^T x)) + WeightedSum<Space>()(q, x)
    return T(0);
  }
};

}  // namespace convex
