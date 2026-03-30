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

#include "convex/sum.hpp"

namespace convex {

/**
 * @brief weighted sum (AKA Dot)
 *
 * f(x) = q^T x
 */
template <linalg::RnSpace Space>
struct WeightedSum {
  /// @brief type
  using T = linalg::space_type_t<Space>;
  static constexpr size_t N = linalg::space_dim_v<Space>;

  /// @brief vec type for q
  using Vec = linalg::Vector<T, N>;

  /**
   * @brief operator overload
   *
   * @param q \in R^n
   * @param x \in R^n (var)
   *
   * @return q^T x
   */
  template <AtomLike Var>
  constexpr auto operator()(const Vec& q, const Var& x) {
    static_assert(is_var_type_v<Var>,
                  "Weighted sum must operate on a var type");
    static_assert(
        std::same_as<dom_space_t<meta_codom_t<atom_meta_t<Var>>>, Space>,
        "Invalid q^T x operator mismatch on vector types");

    using Singleton = linalg::Matrix<T, 1, 1>;

    auto make_sum = [&]<typename... Cs>(const std::tuple<Cs...>& vars) {
      return std::make_from_tuple<Sum<SumMeta<Space, Singleton>, Cs...>>(vars);
    };

    const auto W = linalg::make_diag(q);
    const auto weighted_x = x.mul(W);
    const auto vars = weighted_x.split();

    return make_sum(vars);
  }
};

}  // namespace convex
