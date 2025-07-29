/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <Eigen/Core>
#include <cmath>

#include "../algebra/field.hpp"

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

}  // namespace algebra
