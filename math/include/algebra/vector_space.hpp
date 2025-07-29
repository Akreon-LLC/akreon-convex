/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <Eigen/Core>

#include "../algebra/field.hpp"
#include "../topology/metric_space.hpp"
#include "../topology/topology.hpp"

namespace algebra {

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

#if 0
template <typename T>
constexpr topology::MetricSpace<T, L2<T>> to_euclidean_space(const T& v) {
  topology::MetricSpace<T, L2<T>> metric_space;
  metric_space.tau.open_sets.insert(v.begin(), v.end());
  return metric_space;
}
#endif

template <typename T>
concept LocallyEuclidean = FieldLike<T> && requires(const T& x, const T& y) {
  { L2<T>{}(x, y) } -> std::convertible_to<double>;
};

}  // namespace algebra

namespace topology {

template <typename Derived, typename T>
concept TopologicalManifold =
    TopologicalSpace<Derived, T> && algebra::LocallyEuclidean<T> &&
    requires(const Topology<Derived, T>& tau) {
      { tau.is_hausdorff() } -> std::same_as<bool>;
      { tau.is_second_countable() } -> std::same_as<bool>;
    };

}  // namespace topology

namespace algebra {

template <typename Derived, typename T>
concept RnSpace = topology::TopologicalManifold<Derived, T> && requires(T v) {
  typename T::Scalar;
  typename Eigen::NumTraits<typename T::Scalar>::Real;
  { v + v } -> std::convertible_to<T>;
  { v * typename T::Scalar{1} } -> std::convertible_to<T>;
  //{ to_euclidean_space(v).distance_metric_is_valid() } -> std::same_as<bool>;
};

}  // namespace algebra
