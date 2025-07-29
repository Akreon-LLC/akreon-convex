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

template <typename T>
struct RnSpaceTopology : Topology<RnSpaceTopology<T>, T> {
 private:
  friend Topology<RnSpaceTopology<T>, T>;

  constexpr bool contains_impl(const T& /*X*/) const { return true; }

  constexpr T interior_impl(const T& X) const { return X; }

  constexpr T closure_impl(const T& X) const { return X; }

  constexpr T exterior_impl(const T& X) const { return X; }

  constexpr T boundary_impl(const T& X) const { return X; }

  constexpr bool is_closed_under_finite_intersection_impl() const {
    return true;
  }

  constexpr bool is_hausdoff_impl() const { return true; }

  constexpr bool is_second_countable_impl() const { return true; }
};

template <typename Derived, typename T>
concept LocallyEuclidean = algebra::FieldLike<T> &&
                           MetricSpace<Derived, T, decltype(algebra::L2<T>{})>;

template <typename Derived, typename T>
concept TopologicalManifold =
    TopologicalSpace<Derived, T> && LocallyEuclidean<Derived, T> &&
    requires(const Topology<Derived, T>& tau) {
      { tau.is_hausdorff() } -> std::same_as<bool>;
      { tau.is_second_countable() } -> std::same_as<bool>;
    };

template <typename Derived, typename T>
concept RnSpace =
    topology::TopologicalManifold<Derived, T> && requires(const T& X) {
      { all_finite(X) } -> std::same_as<bool>;
    };

}  // namespace topology
