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

#include <tuple>

#include "cone_traits.hpp"

namespace convex {

template <ConvexAtomLike... Cs>
struct DAG {
  static constexpr size_t N = sizeof...(Cs);
  using BoolMatT = linalg::Matrix<bool, N, N>;

  constexpr explicit DAG(const BoolMatT& mat, Cs... cs)
      : adj(mat), atoms(std::forward<Cs>(cs)...) {}

  constexpr explicit DAG(const BoolMatT& mat, std::tuple<Cs...> cs)
      : adj(mat), atoms(std::move(cs)) {}

  BoolMatT adj;
  std::tuple<Cs...> atoms;
};

template <ConvexAtomLike... Cs>
constexpr bool has_cycle(const DAG<Cs...>& dag) {
  static constexpr size_t N = sizeof...(Cs);
  auto dfs = [](this auto&& self, const auto& adj, auto& vis,
                  size_t idx) -> bool {
    if (vis(idx)) return true;

    vis[idx] = true;

    for (size_t i = 0; i < N; ++i) {
      if (adj(idx)(i)) {
        if (!vis(i)) return self(adj, vis, i);
      }
    }

    return false;
  };

  for (size_t i = 0; i < N; ++i) {
    linalg::Vector<bool, N> vis;
    if (dfs(dag.adj, vis, i)) return true;
  }
  return false;
}

}  // namespace convex
