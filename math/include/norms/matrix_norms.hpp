/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../linalg/matrix_ops.hpp"

namespace linalg {

/**
 * @brief compute the frobenius norm of a matrix
 *
 * sqrt(tr(A^T A)) if A \in R else sqrt(tr(A^* A)) where A \in C
 *
 * @param matrix A Rows x Cols elements of type T (must be square)
 *
 * @return scalar
 */
template <typename T, size_t Rows, size_t Cols>
  requires(Rows == Cols && Rows > 0 && FieldLike<T>)
constexpr auto frobenius_norm(const Matrix<T, Rows, Cols>& A) {
  scalar_type_t<T> result = 0;
  if constexpr (!is_complex_v<T>) {
    result = trace(transpose(A) * A);
  } else {
    const auto tmp = trace(conjugate_transpose(A) * A);
    assert(is_real(tmp));
    result = tmp.real();
  }
  return constexpr_sqrt(result);
}

}  // namespace linalg
