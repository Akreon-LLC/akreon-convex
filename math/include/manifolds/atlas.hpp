/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <set>
#include <vector>

#include "chart.hpp"

namespace atlas {

template <typename T, typename Rn, typename MapToR, typename MapFromR>
  requires chart::ValidChart<T, Rn, MapToR, MapFromR>
struct Atlas {
  using ChartT = chart::Chart<T, Rn, MapToR, MapFromR>;
  std::vector<ChartT> charts;

  bool covers(const topology::Topology<T>& M) const;
};

template <typename T, typename Rn, typename MapToR, typename MapFromR>
concept ValidAtlas = requires(const Atlas<T, Rn, MapToR, MapFromR>& atlas,
                              const topology::Topology<T>& M) {
  requires topology::TopologicalManifold<T>;  // M must satisfy the manifold concept
  { atlas.covers(M) } -> std::same_as<bool>;
};

}  // namespace atlas
