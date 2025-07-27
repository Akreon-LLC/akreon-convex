/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <Eigen/Dense>
#include <concepts>
#include <functional>
#include <set>
#include <vector>

#include "../algebra/vector_space.hpp"
#include "../topology/topology.hpp"

namespace chart {

using Real = double;

using RealN = vector_space::VectorSpace<Real, Eigen::Dynamic>;

template <typename T>
using Phi = std::function<RealN(const T&)>;

template <typename T>
using InvPhi = std::function<T(const RealN&)>;

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
struct Chart {
  topology::Set<T> U;  // domain open subset U of the manifold M
  MapToRn phi;         // map T -> R^n
  MapFromRn inv_phi;   // inverse map R^n -> T
};

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
bool is_homeomorphism(const Chart<T, Rn, MapToRn, MapFromRn>& chart);

template <typename T, typename Rn, typename MapToRn, typename MapFromRn>
concept ValidChart = requires(const Chart<T, Rn, MapToRn, MapFromRn>& chart,
                              const topology::Topology<T>& M) {
  requires topology::TopologicalManifold<T>;  // M must satisfy the manifold
                                              // concept
  // domain must be an open subset of M
  { M.contains(chart.U) } -> std::same_as<bool>;
  // phi must be a valid bijection with a continuous inverse
  { is_homeomorphism(chart) } -> std::same_as<bool>;
};

}  // namespace chart
