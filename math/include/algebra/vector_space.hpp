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
concept RnSpace = requires(T v) {
  typename T::Scalar;
  typename Eigen::NumTraits<typename T::Scalar>::Real;
  { v + v } -> std::convertible_to<T>;
  { v * typename T::Scalar{1} } -> std::convertible_to<T>;
  { v.norm() } -> std::convertible_to<typename T::Scalar>;
};

}  // namespace vector_space
