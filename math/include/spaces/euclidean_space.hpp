/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <cmath>

#include "../algebra/vector_space.hpp"
#include "../topology/metric_space.hpp"

namespace spaces {

template <typename T>
constexpr bool all_finite(const T& X) {
  for (const auto& x : X)
    if (!std::isfinite(x)) return false;
  return true;
}

template <typename T, size_t N>
concept LocallyEuclidean =
    algebra::InnerProduct<T, N> &&
    topology::MetricSpace<T, N, decltype(algebra::L2<T, N>)>;

template <typename T, size_t N>
concept TopologicalManifold =
    topology::TopologicalSpace<T, N> && LocallyEuclidean<T, N> &&
    requires(const topology::Topology<T, N>& tau) {
      { is_hausdorff(tau) } -> std::same_as<bool>;
      { is_second_countable(tau) } -> std::same_as<bool>;
    };

template <typename T, size_t N>
concept RnSpace = TopologicalManifold<T, N> && requires(const T& X) {
  { all_finite(X) } -> std::same_as<bool>;
};

template <typename T, size_t N>
  requires RnSpace<T, N>
using EuclideanSpace = topology::Topology<T, N>;

}  // namespace spaces
