/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "topology/topology.hpp"

namespace topology {

template <typename T, size_t N>
struct TopologicalSpace : Topology<T, N, TopologicalSpace<T, N>> {
  template <typename... Sets>
  constexpr explicit TopologicalSpace(Sets... sets)
      : Topology<T, N, TopologicalSpace<T, N>>(std::forward<Sets>(sets)...) {}

  constexpr bool is_valid() const {
    return this->contains(T{}) && this->s_size > 0 &&
           this->contains(this->open_sets(0)) &&
           this->is_closed_under_union() &&
           this->is_closed_under_finite_intersection();
  }
};

}  // namespace topology
