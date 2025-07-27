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

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
  requires chart::ValidChart<T, Rn, MapToRn, MapFromRn>
struct Atlas {
  using ChartT = chart::Chart<T, Rn, MapToRn, MapFromRn>;
  std::vector<ChartT> charts;

  bool covers(const topology::Topology<T>& M) const;

  bool is_compatible() const;
};

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
concept ValidAtlas = requires(const Atlas<T, Rn, MapToRn, MapFromRn>& atlas,
                              const topology::Topology<T>& M) {
  requires topology::TopologicalManifold<T>;  // M must satisfy the manifold
                                              // concept
  { atlas.covers(M) } -> std::same_as<bool>;
};

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
concept SmoothAtlas = ValidAtlas<T, Rn, MapToRn, MapFromRn> &&
                      requires(const Atlas<T, Rn, MapToRn, MapFromRn>& atlas) {
                        { atlas.is_compatible() } -> std::same_as<bool>;
                      };

}  // namespace atlas
