/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <Eigen/Core>

#include "../topology/topology.hpp"

namespace vector_space {

template <typename Scalar, int N>
using VectorSpace = Eigen::Matrix<Scalar, N, 1>;

template <typename T>
struct L2 {
  constexpr auto operator()(const T& x, const T& y) const {
    if constexpr (std::is_arithmetic_v<T>)
      return std::abs(x - y);
    else
      return (x - y).norm();
  }
};

template <typename T>
constexpr topology::MetricSpace<T, L2<T>> to_euclidean_space(const T& v) {
  topology::MetricSpace<T, L2<T>> metric_space;
  metric_space.tau.open_sets.insert(v.begin(), v.end());
  return metric_space;
}

template <typename T>
concept RnSpace = requires(T v) {
  typename T::Scalar;
  typename Eigen::NumTraits<typename T::Scalar>::Real;
  { v + v } -> std::convertible_to<T>;
  { v * typename T::Scalar{1} } -> std::convertible_to<T>;
  { to_euclidean_space(v).distance_metric_is_valid() } -> std::same_as<bool>;
};

template <typename T>
concept LocallyEuclidean = std::floating_point<T> && requires(const T& x, const T& y) {
  { L2<T>{}(x, y) } -> std::convertible_to<double>;
};

}  // namespace vector_space

namespace topology {
template <typename T>
concept TopologicalManifold =
    TopologicalSpace<T> && vector_space::LocallyEuclidean<T> &&
    requires(const Topology<T>& tau) {
      { is_hausdorff(tau) } -> std::same_as<bool>;
      { is_second_countable(tau) } -> std::same_as<bool>;
    };
}  // namespace topology
