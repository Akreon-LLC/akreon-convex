/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/field.hpp"
#include "topological_space.hpp"

namespace spaces {

template <typename T, size_t N, typename D>
constexpr bool distance_metric_is_valid_on_space(
    const topology::Topology<T, N>& tau, const D& d) {
  const auto universe = compute_universe(tau);

  for (const auto& x : universe) {
    for (const auto& y : universe) {
      typename T::value_type dxy = d(x, y);
      typename T::value_type dyx = d(y, x);

      if (dxy < 0) return false;
      if (x == y && dxy != 0) return false;
      if (x != y && dxy == 0) return false;
      if (algebra::constexpr_abs(dxy - dyx) > 1e-9) return false;
      for (const auto& z : universe)
        if (d(x, z) > d(x, y) + d(y, z) + 1e-9) return false;
    }
  }
  return true;
}

template <typename T, size_t N, typename D>
concept MetricSpace =
    TopologicalSpace<T, N> &&
    requires(const topology::Topology<T, N>& space, const D& d) {
      { distance_metric_is_valid_on_space(space, d) } -> std::same_as<bool>;
    };

}  // namespace spaces
