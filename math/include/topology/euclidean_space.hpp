/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <cmath>

#include "../algebra/vector_space.hpp"
#include "metric_space.hpp"
#include "topology.hpp"

namespace topology {

template <typename T>
constexpr bool all_finite(const T& X) {
  for (const auto& x : X)
    if (!std::isfinite(x)) return false;
  return true;
}

template <typename T, size_t N>
concept LocallyEuclidean =
    algebra::FieldLike<T> && MetricSpace<T, N, decltype(algebra::L2<T>{})>;

template <typename T, size_t N>
concept TopologicalManifold =
    TopologicalSpace<T, N> && LocallyEuclidean<T, N> &&
    requires(const Topology<T, N>& tau) {
      { is_hausdorff(tau) } -> std::same_as<bool>;
      { is_second_countable(tau) } -> std::same_as<bool>;
    };

template <typename T, size_t N>
concept EuclideanSpace =
    topology::TopologicalManifold<T, N> && requires(const T& X) {
      { all_finite(X) } -> std::same_as<bool>;
    };

template <typename T, size_t N>
  requires EuclideanSpace<T, N>
using EuclideanTopology = Topology<T, N>;

}  // namespace topology
