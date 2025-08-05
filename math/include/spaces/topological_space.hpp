/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "topology/topology.hpp"

namespace spaces {

template <typename T, size_t N>
concept TopologicalSpace = requires(topology::Topology<T, N> tau, T X) {
  { contains(tau, T{}) } -> std::same_as<bool>;  // null set is open
  { contains(tau, X) } -> std::same_as<bool>;    // X is open
  { is_closed_under_union(tau) } -> std::same_as<bool>;
  { is_closed_under_finite_intersection(tau) } -> std::same_as<bool>;
};

}  // namespace spaces
