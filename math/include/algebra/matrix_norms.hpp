/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix_ops.hpp"

namespace algebra {

template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && Rows > 0 && FieldLike<T>)
constexpr auto frobenius_norm(const Matrix<T, Rows, Cols>& A) {
  scalar_type_t<T> result = 0;
  if constexpr (!is_complex_v<T>) {
    result = trace(transpose(A) * A);
  } else {
    const auto tmp = trace(hermitian(A) * A);
    assert(is_real(tmp));
    result = tmp.real();
  }
  return constexpr_sqrt(result);
}

}  // namespace algebra
