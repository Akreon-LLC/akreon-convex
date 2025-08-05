/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <cmath>

#include "inner_product_space.hpp"
#include "metric_space.hpp"

namespace spaces {

template <typename T>
constexpr bool all_finite(const T& X) {
  for (const auto& x : X)
    if (!std::isfinite(x)) return false;
  return true;
}

template <typename T, size_t N>
concept LocallyEuclidean = MetricSpace<T, N, decltype(algebra::L2<T, N>)>;

template <typename T, size_t N>
concept TopologicalManifold =
    LocallyEuclidean<T, N> && requires(const topology::Topology<T, N>& tau) {
      { is_hausdorff(tau) } -> std::same_as<bool>;
      { is_second_countable(tau) } -> std::same_as<bool>;
    };

template <typename T, size_t N>
concept RnSpace = TopologicalManifold<T, N> && requires(const T& X) {
  { all_finite(X) } -> std::same_as<bool>;
};

// (dakre) TODO - This is prime for refactor / update to incorporate MapToRn and
// MapFromRn chart functions
template <typename Derived>
  requires(Derived::value_type::s_size > 0)
struct EuclideanSpace {
  using T = std::conditional_t<algebra::is_complex_v<Derived>,
                               std::complex<algebra::scalar_type_t<Derived>>,
                               algebra::scalar_type_t<Derived>>;
  static const size_t N = Derived::s_size;
  static const size_t M = Derived::value_type::s_size;

  constexpr auto as_topological_space()
    requires RnSpace<T, N>
  {
    const auto derived = static_cast<Derived&>(*this);
    if constexpr (std::is_same_v<
                      decltype(derived),
                      topology::Topology<topology::OpenSet<T, M>, N>>)
      return derived;
    return to_topological_space(derived);
  }

  constexpr auto as_vector_space()
    requires(InnerProductSpace<T, N>)
  {
    const auto derived = static_cast<Derived&>(*this);
    if constexpr (std::is_same_v<decltype(derived),
                                 algebra::Vector<algebra::Vector<T, M>, N>>)
      return derived;
    return to_vector_space(derived);
  }

 private:
  // (dakre) TODO - this would be MapToRn chart func
  constexpr algebra::Vector<T, N> to_vector_space(
      const topology::Topology<T, N>& tau) {
    using InnerT = algebra::Vector<T, M>;
    algebra::Vector<InnerT, N> X;
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < M; ++j)
        X.coordinates[i].coordinates[j] = tau(i)(j);
    return X;
  }

  // (dakre) TODO - this would be MapFromRn chart func
  constexpr topology::Topology<T, N> to_topological_space(
      const algebra::Vector<T, N>& X) {
    const auto num_rows = algebra::number_of_rows_v<decltype(X)>;
    using InnerT = topology::OpenSet<T, M>;
    topology::Topology<InnerT, N> tau;
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j < M; ++j) tau.open_sets(i).insert(X(i)(j));

    return tau;
  }
};

}  // namespace spaces
