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
  // (dakre) unvisited = 0, visiting = 1, visited = 2
  std::array<uint8_t, N> state{};
  const auto& adj = dag.adj;

  auto dfs = [&](this auto&& self, size_t idx) -> bool {
    state[idx] = 1;
    for (size_t i = 0; i < N; ++i) {
      if (!adj(idx)(i)) continue;
      if (state[i] == 1) return true;  // (dakre) back edge detected
      if (state[i] == 0 && self(i)) return true;
    }
    state[idx] = 2;
    return false;
  };

  for (size_t i = 0; i < N; ++i)
    if (state[i] == 0 && dfs(i)) return true;
  return false;
}

}  // namespace convex
