/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "topology.hpp"

namespace topology {

template <typename T, typename D>
  requires requires(const T& x, const T& y, const D& d) { d(x, y); }
struct MetricSpace {
  Topology<T> tau;
  D d;

  constexpr bool distance_metric_is_valid() const {
    Set<T> universe;
    for (const auto& U : tau.open_sets) universe.insert(U.begin(), U.end());

    for (const auto& x : universe) {
      for (const auto& y : universe) {
        T dxy = d(x, y);
        T dyx = d(y, x);

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
};

template <typename T, typename D>
concept ValidMetricSpace = requires(const MetricSpace<T, D>& metric_space) {
  { metric_space.distance_metric_is_valid() } -> std::same_as<bool>;
};

}  // namespace topology
