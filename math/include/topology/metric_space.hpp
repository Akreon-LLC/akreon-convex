/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/field.hpp"
#include "topological_space.hpp"

namespace topology {

template <typename T, size_t N, typename DistanceFn>
struct MetricSpace : TopologicalSpace<T, N, MetricSpace<T, N, DistanceFn>> {
  template <typename... Sets>
  constexpr MetricSpace(Sets... sets)
      : TopologicalSpace<T, N, MetricSpace<T, N, DistanceFn>>(
            std::forward<Sets>(sets)...) {}

  constexpr bool is_valid() const {
    const auto top_space = static_cast<
        const TopologicalSpace<T, N, MetricSpace<T, N, DistanceFn>>*>(this);
    if (!top_space->is_valid()) return false;

    const auto base = static_cast<const Topology<
        T, N, TopologicalSpace<T, N, MetricSpace<T, N, DistanceFn>>>*>(this);
    const auto universe = base->compute_universe();
    for (const auto& x : universe) {
      for (const auto& y : universe) {
        typename T::value_type dxy = dist_fn(x, y);
        typename T::value_type dyx = dist_fn(y, x);

        if (dxy < 0) return false;
        if (x == y && dxy != 0) return false;
        if (x != y && dxy == 0) return false;
        if (linalg::constexpr_abs(dxy - dyx) > 1e-9) return false;
        for (const auto& z : universe)
          if (dist_fn(x, z) > dist_fn(x, y) + dist_fn(y, z) + 1e-9)
            return false;
      }
    }
    return true;
  }

  DistanceFn dist_fn;
};

}  // namespace topology
