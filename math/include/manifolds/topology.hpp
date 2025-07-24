/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <concepts>
#include <set>
#include <vector>

namespace topology {

// Define default metric space
using Real = double;
using RealN = std::vector<Real>;

// Base set X
template <typename T>
using Set = std::set<T>;

// Collection of open sets = topology
template <typename T>
struct Topology {
  using value_type = T;
  std::set<Set<T>> open_sets;

  bool contains(const Set<T>& s) const;

  Set<T> interior(const Set<T>& X);

  Set<T> closure(const Set<T>& X);

  Set<T> exterior(const Set<T>& X);

  Set<T> boundary(const Set<T>& X);
};

template <typename T>
concept EuclideanType = std::floating_point<T> || requires(T x) {
  typename T::value_type;
  requires std::floating_point<typename T::value_type>;
};

template <typename T>
bool is_closed_under_union(const Topology<T>& tau);

template <typename T>
bool is_closed_under_finite_intersection(const Topology<T>& tau);

template <typename T>
bool is_hausdorff(const Topology<T>& tau);

template <typename T>
bool is_second_countable(const Topology<T>& tau);

template <typename T>
bool is_locally_euclidean(const Topology<T>& tau)
  requires EuclideanType<T>;

template <typename T>
concept TopologicalSpace = requires(Topology<T> tau, Set<T> X) {
  { tau.contains(Set<T>{}) } -> std::same_as<bool>;  // null set is open
  { tau.contains(X) } -> std::same_as<bool>;         // X is open
  { is_closed_under_union(tau) } -> std::same_as<bool>;
  { is_closed_under_finite_intersection(tau) } -> std::same_as<bool>;
};

template <typename T>
concept TopologicalManifold =
    TopologicalSpace<T> && requires(const Topology<T>& tau) {
      { is_locally_euclidean(tau) } -> std::same_as<bool>;
      { is_hausdorff(tau) } -> std::same_as<bool>;
      { is_second_countable(tau) } -> std::same_as<bool>;
    };

}  // namespace topology
