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

#include "linear_operator.hpp"
#include "operator_traits.hpp"

namespace linalg {

/**
 * @brief Symmetric linear operator CRTP base
 */
template <SquareMatrixLike Mat, typename Derived>
struct SymmetricOperator
    : LinearOperator<Mat, SymmetricOperator<Mat, Derived>> {
  /// @brief alias to extract base types
  using OperatorBase = LinearOperator<Mat, SymmetricOperator<Mat, Derived>>;
  using typename OperatorBase::ComplexT;
  using typename OperatorBase::ScalarT;
  using typename OperatorBase::T;

  /// @brief extract N from base
  using OperatorBase::N;

  /// @brief helpers to get DIM from mat
  static constexpr size_t DIM = (N * (N + 1)) / 2;

  /**
   * @brief CTOR
   *
   * @param Mat A
   */
  constexpr explicit SymmetricOperator(const Mat& A)
      : svec(get_lower_triangle_with_diag(A)) {}

  /**
   * @brief get element at (i,j) index
   *
   * @param row index
   * @param col index
   *
   * @return element
   */
  constexpr auto operator()(const size_t& i, const size_t& j) const {
    return svec(index(i, j));
  }

  /**
   * @brief index from (row, col)
   *
   * @param size_t i = row index
   * @param size_t j = col index
   *
   * @return index
   */
  constexpr size_t index(const size_t& i, const size_t& j) const {
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
    return type == SymmetricOperatorType::PD ||
           type == SymmetricOperatorType::PSD;
  }

  /**
   * @brief is nsd check
   *
   * @return T/F
   */
  constexpr bool is_nsd() const {
    return type == SymmetricOperatorType::ND ||
           type == SymmetricOperatorType::NSD;
  }

  /**
   * @brief is indefinite
   *
   * @return T/F
   */
  constexpr bool is_ind() const { return type == SymmetricOperatorType::IND; }

  /**
   * @brief is valid
   *
   * @return T/F
   */
  constexpr bool is_valid() const { return type != SymmetricOperatorType::INV; }

  /**
   * @brief compute definiteness
   *
   * @param vec (diagonal of elements)
   *
   * @return void
   */
  template <typename DiagT>
  constexpr void compute_definiteness(const Vector<DiagT, N>& diag,
                                      const ScalarT& tol) {
    size_t npos = 0, nneg = 0, nzero = 0;

    auto classify_scalar = [&](const auto& d_real) {
      if (d_real > tol)
        ++npos;
      else if (d_real < -tol)
        ++nneg;
      else
        ++nzero;
    };

    if constexpr (is_complex_v<DiagT>) {
      for (const auto& d : diag) classify_scalar(d.real());
    } else {
      for (const auto& d : diag) classify_scalar(d);
    }

    if (nneg == 0 && nzero == 0)
      type = SymmetricOperatorType::PD;
    else if (nneg == 0 && nzero > 0)
      type = SymmetricOperatorType::PSD;
    else if (npos == 0 && nzero == 0)
      type = SymmetricOperatorType::ND;
    else if (npos == 0 && nzero > 0)
      type = SymmetricOperatorType::NSD;
    else
      type = SymmetricOperatorType::IND;
  }

  /**
   * @brief self
   *
   * @return self
   */
  constexpr const Derived& self() const { return static_cast<Derived&>(*this); }

  /** @brief member variables */
  SymmetricOperatorType type;  //< SymmetricOperatorType value
  Vector<T, DIM> svec;  //< Data stored from input square matrix as flattened
                        // n(n+1)/2 dim vector
};

}  // namespace linalg
