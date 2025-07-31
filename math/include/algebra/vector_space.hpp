/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <cmath>

#include "../topology/topology.hpp"
#include "field.hpp"

namespace algebra {

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
template <typename T, size_t N>
  requires topology::TopologicalSpace<T, N>
concept InnerProduct = FieldLike<T> && requires(const scalar_type_t<T>& alpha,
                                                const scalar_type_t<T>& beta,
                                                const T& X, const T& Y) {
  { alpha * X } -> std::same_as<T>;
  { alpha * (X + Y) == alpha * X + alpha * Y };
  { (alpha * beta) * X == alpha * X + beta * X };
  { (alpha * beta) * X == alpha * (beta * X) };
  { is_ones_like(X / X) };
};
#endif

}  // namespace algebra
