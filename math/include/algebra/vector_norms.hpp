/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "vector_ops.hpp"

namespace algebra {

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto L2(const Vector<T, N>& X) {
  scalar_type_t<T> result = 0;
  if constexpr (!is_complex_v<T>)
    result = dot(X, X);
  else {
    const auto tmp = dot(conjugate(X), X);
    assert(is_real(tmp));
    result = tmp.real();
  }
  return constexpr_sqrt(result);
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto L1(const Vector<T, N>& X) {
  T sum = 0;
  for (const auto& x : X) sum += constexpr_abs(x);
  return sum;
}

template <typename T, size_t N>
  requires(FieldLike<T>)
constexpr auto max_norm(const Vector<T, N>& X) {
  T mx = 0;
  for (const auto& x : X) mx = std::max(mx, constexpr_abs(x));
  return mx;
}

}  // namespace algebra
