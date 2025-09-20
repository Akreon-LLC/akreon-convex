/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "affine_set.hpp"

namespace convex {

template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class AffineMap {
 public:
  constexpr explicit AffineMap(const linalg::Matrix<T, N, M>& U_in,
                               const linalg::Vector<T, N>& c_in)
      : U(U_in), c(c_in) {}

  constexpr auto compose(const AffineMap<T, M, N>& other) const {
    return AffineMap(U * other.U, U * other.c + c);
  }

  constexpr auto inverse() const {
    const auto U_inv = linalg::inverse(U);
    return AffineMap(U_inv, -1 * U_inv * c);
  }

  constexpr auto linear_part() const { return U; }

  constexpr auto translation() const { return c; }

 private:
  linalg::Matrix<T, N, M> U;
  linalg::Vector<T, N> c;
};

}  // namespace convex
