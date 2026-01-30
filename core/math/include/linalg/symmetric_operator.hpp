/*
 * Copyright © 2025 Akreon LLC. All rights reserved.
 *
 * Proprietary and Confidential.
 * This source code is the property of Akreon LLC and constitutes a trade secret
 * protected by U.S. and international law.
 *
 * Unauthorized copying, modification, distribution, or disclosure of this file,
 * via any medium, is strictly prohibited without express written permission
 * from Akreon LLC.
 */

#pragma once

#include "matrix_traits.hpp"

namespace linalg {

/**
 * @brief Symmetric linear operator CRTP base
 */

template <SquareMatrixLike Mat, typename Derived>
struct SymmetricOperator {
  /// @brief alias helpers for types
  using T = mat_type_t<Mat>;
  using ScalarT = scalar_type_t<T>;
  using ComplexT = std::complex<ScalarT>;

  /// @brief helpers to get N and DIM from mat
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = (N * (N + 1)) / 2;

  /**
   * @brief CTOR
   *
   * @param Mat A
   */
  constexpr explicit SymmetricOperator(const Mat& A, ScalarT tol = 1e-8) {
    mutable_self().factor(A, tol);
  }

  /**
   * @brief overload for accessor (provide row col pair and get value)
   *
   * @param size_t i = row index
   * @param size_t j = col index
   *
   * @return L element
   */
  constexpr auto operator()(size_t i, size_t j) const { return L(index(i, j)); }

  /**
   * @brief index from (row, col)
   *
   * @param size_t i = row index
   * @param size_t j = col index
   *
   * @return index
   */
  constexpr size_t index(size_t i, size_t j) const {
    assert(j < N && i < N);
    if (i >= j) return i * (i + 1) / 2 + j;
    return j * (j + 1) / 2 + i;
  }

  /**
   * @brief is psd check
   *
   * @return T/F
   */
  constexpr bool is_psd() const {
    return type == SymmetricMatrixType::PD || type == SymmetricMatrixType::PSD;
  }

  /**
   * @brief is nsd check
   *
   * @return T/F
   */
  constexpr bool is_nsd() const {
    return type == SymmetricMatrixType::ND || type == SymmetricMatrixType::NSD;
  }

  /**
   * @brief self
   *
   * @return derived object
   */
  constexpr Derived& mutable_self() { return static_cast<Derived&>(*this); }

  /**
   * @brief self
   *
   * @return self
   */
  constexpr const Derived& self() const { return static_cast<Derived&>(*this); }

  /** @brief member variables */
  SymmetricMatrixType type;  //< SymmetricMatrixType value
  Vector<T, DIM> L;  //< Data stored from input square matrix as flattened
                     // n(n+1)/2 dim vector
};

}  // namespace linalg
