/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#include <Eigen/Core>

#include "../topology/topology.hpp"

namespace vector_space {

template <typename Scalar = double, int N = Eigen::Dynamic>
using VectorSpace = Eigen::Matrix<Scalar, N, 1>;

template <typename T>
concept RnSpace = topology::TopologicalSpace<T> && requires(T v, T w) {
  typename Eigen::NumTraits<typename T::Scalar>::Real;
  { v + w } -> std::same_as<T>;
  {
    v.norm()
  } -> std::convertible_to<typename Eigen::NumTraits<typename T::Scalar>::Real>;
};

}  // namespace vector_space
