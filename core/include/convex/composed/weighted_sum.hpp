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

#include "linalg/linalg_util.hpp"
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
   * @param q in R^n
   * @param x in R^n (var)
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

    const auto W = linalg::make_diag(q);
    const auto weighted_x = x.mul(W);
    const auto vars = weighted_x.split();
    return std::apply([](const auto&... cs) { return Sum(cs...); }, vars);
  }
};

/**
 * @brief make weighted sum with domain deduced from the variable atom
 *
 * f(x) = q^T x
 *
 * @param q in R^n
 * @param x decision variable atom
 *
 * @return q^T x
 */
template <AtomLike Var>
constexpr auto make_weighted_sum(
    const linalg::Vector<dom_type_t<meta_codom_t<atom_meta_t<Var>>>,
                         dom_size_v<meta_codom_t<atom_meta_t<Var>>>>& q,
    const Var& x) {
  using Space = dom_space_t<meta_codom_t<atom_meta_t<Var>>>;
  return WeightedSum<Space>()(q, x);
}

} // namespace convex
