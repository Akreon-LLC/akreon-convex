/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../algebra/field.hpp"
#include "topology.hpp"

namespace topology {

template <typename T, typename D>
constexpr bool distance_metric_is_valid_on_space(const SetTopology<T>& tau,
                                                 const D& d) {
  T universe;
  for (const auto& U : tau.open_sets) universe.insert(U.begin(), U.end());

  for (const auto& x : universe) {
    for (const auto& y : universe) {
      typename T::value_type dxy = d(x, y);
      typename T::value_type dyx = d(y, x);

      if (dxy < 0) return false;
      if (x == y && dxy != 0) return false;
      if (x != y && dxy == 0) return false;
      if (std::abs(dxy - dyx) > 1e-9) return false;
      for (const auto& z : universe)
        if (d(x, z) > d(x, y) + d(y, z) + 1e-9) return false;
    }
  }
  return true;
}

template <typename Derived, typename T, typename D>
concept MetricSpace =
    TopologicalSpace<Derived, T> && requires(const Derived& space, const D& d) {
      { distance_metric_is_valid_on_space(space, d) } -> std::same_as<bool>;
    };

}  // namespace topology
