/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <functional>
#include <set>
#include <vector>

#include "topology.hpp"

namespace chart {

using Real = double;
using RealN = std::vector<Real>;
using Phi = std::function<RealN(const topology::Set<Real>&)>;
using InvPhi = std::function<topology::Set<Real>(const RealN&)>;

template <typename T, typename Rn, typename MapToR, typename MapFromR>
struct Chart {
  topology::Set<T> U;  // domain open subset U of the manifold M
  MapToR phi;          // map T -> R^n
  MapFromR inv_phi;    // inverse map R^n -> T
};

template <typename T, typename Rn, typename MapToR, typename MapFromR>
bool is_homeomorphism(const Chart<T, Rn, MapToR, MapFromR>& chart);

template <typename T, typename Rn, typename MapToR, typename MapFromR>
concept ValidChart = requires(const Chart<T, Rn, MapToR, MapFromR>& chart,
                              const topology::Topology<T>& M) {
  requires topology::TopologicalManifold<T>;  // M must satisfy the manifold
                                              // concept
  // domain must be an open subset of M
  { M.contains(chart.U) } -> std::same_as<bool>;
  // phi must be a valid bijection with a continuous inverse
  { is_homeomorphism(chart) } -> std::same_as<bool>;
};

}  // namespace chart
