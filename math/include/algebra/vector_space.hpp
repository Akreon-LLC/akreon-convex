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
    return (x - y).norm();
  }
};

template <typename T>
constexpr topology::MetricSpace<T, L2<T>> to_metric_space(const T& v) {
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
  { to_metric_space(v).distance_metric_is_valid() } -> std::same_as<bool>;
};

}  // namespace vector_space
