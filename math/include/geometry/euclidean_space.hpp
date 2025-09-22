/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../linalg/inner_product_space.hpp"

namespace geometry {

template <typename T>
  requires(linalg::FieldLike<T>)
using Point2d = linalg::Vector<T, 2>;

template <typename T>
  requires(linalg::FieldLike<T>)
using Point3d = linalg::Vector<T, 3>;

template <typename T, size_t N, size_t M>
  requires(linalg::InnerProductLike<linalg::Vector<T, N>>)
struct EuclideanSpace
    : topology::MetricSpace<linalg::Vector<T, N>, M, linalg::L2<T, N>> {
  using value_type = T;

  template <typename... Sets>
  constexpr explicit EuclideanSpace(Sets... sets)
      : topology::MetricSpace<linalg::Vector<T, N>, M, linalg::L2<T, N>>(
            std::forward<Sets>(sets)...) {}

  // (dakre) TODO need to refactor topology and the like, but for now stub this
  // out... the representation of open sets should step from some n-dim points
  constexpr bool is_valid() const { return true; }
};

}  // namespace geometry
