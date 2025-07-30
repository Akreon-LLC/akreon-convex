/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <functional>
#include <limits>
#include <set>
#include <vector>

#include "open_set.hpp"

namespace topology {

template <typename Derived>
struct TopologyTraits;

// Forward declarations
template <typename T, size_t N>
struct SetTopology;

template <typename T, size_t N>
struct EuclideanTopology;

// Specialize the trait
template <typename T, size_t N>
struct TopologyTraits<SetTopology<T, N>> {
  static constexpr size_t s_size = N;
};

// Collection of open sets = topology
template <typename Derived, typename T>
struct Topology {
  using value_type = T;

  OpenSet<T, TopologyTraits<Derived>::s_size> open_sets;

  constexpr bool contains(const T& set) const {
    static_assert(std::is_invocable_r_v<bool, decltype(&Derived::contains_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->contains_impl(set);
  }

  constexpr T interior(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::interior_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->interior_impl(set);
  }

  constexpr T closure(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::closure_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->closure_impl(set);
  }

  constexpr T exterior(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::exterior_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->exterior_impl(set);
  }

  constexpr T boundary(const T& set) const {
    static_assert(std::is_invocable_r_v<T, decltype(&Derived::boundary_impl),
                                        const Derived*, const T&>);
    return static_cast<const Derived*>(this)->boundary_impl(set);
  }

  constexpr bool is_closed_under_union() const {
    static_assert(
        std::is_invocable_r_v<bool,
                              decltype(&Derived::is_closed_under_union_impl),
                              const Derived*>);
    return static_cast<const Derived*>(this)->is_closed_under_union_impl();
  }

  constexpr bool is_closed_under_finite_intersection() const {
    static_assert(
        std::is_invocable_r_v<
            bool, decltype(&Derived::is_closed_under_finite_intersection_impl),
            const Derived*>);
    return static_cast<const Derived*>(this)
        ->is_closed_under_finite_intersection_impl();
  }

  constexpr bool is_hausdorff() const {
    static_assert(
        std::is_invocable_r_v<bool, decltype(&Derived::is_hausdorff_impl),
                              const Derived*>);
    return static_cast<const Derived*>(this)->is_hausdorff_impl();
  }

  constexpr bool is_second_countable() const {
    static_assert(
        std::is_invocable_r_v<bool,
                              decltype(&Derived::is_second_countable_impl),
                              const Derived*>);
    return static_cast<const Derived*>(this)->is_second_countable_impl();
  }
};

template <typename Derived, typename T>
concept TopologicalSpace = requires(Topology<Derived, T> tau, T X) {
  { tau.contains(T{}) } -> std::same_as<bool>;  // null set is open
  { tau.contains(X) } -> std::same_as<bool>;    // X is open
  { tau.is_closed_under_union() } -> std::same_as<bool>;
  { tau.is_closed_under_finite_intersection() } -> std::same_as<bool>;
};

}  // namespace topology
