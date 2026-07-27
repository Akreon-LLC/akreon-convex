/* Copyright © 2025 Akreon LLC. All rights reserved.
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

#include <complex>
#include <cstddef>
#include <utility>

#include "matrix.hpp"

namespace linalg {

template <typename T, std::size_t Dim>
using ColVector = Matrix<T, Dim, 1>;

template <typename T, std::size_t Dim>
using RowVector = Matrix<T, 1, Dim>;

template <class Mat>
struct mat_traits;

template <typename T, std::size_t Rows, std::size_t Cols>
struct mat_traits<Matrix<T, Rows, Cols>> {
  using value_type = T;
  static constexpr std::size_t ROWS = Rows;
  static constexpr std::size_t COLS = Cols;

  template <typename... Ts>
  static constexpr auto operator()(Ts&&... xs) {
    static_assert(sizeof...(xs) == Rows * Cols,
                  "mat_traits::operator() requires Rows*Cols coordinates");
    Matrix<T, Rows, Cols> matrix;
    std::size_t i = 0;
    auto set_element = [&](const T& x) {
      const auto col = i % Cols;
      const auto row = i / Cols;
      matrix(row, col) = x;
      ++i;
    };
    ((void)set_element(std::forward<Ts>(xs)), ...);
    return matrix;
  }

  static constexpr auto fill(const T& x) {
    Matrix<T, Rows, Cols> matrix;
    for (std::size_t i = 0; i < Rows; ++i)
      for (std::size_t j = 0; j < Cols; ++j) matrix(i, j) = x;
    return matrix;
  }
};

template <class Mat>
using matrix_scalar_t = typename mat_traits<remove_cvref_t<Mat>>::value_type;

template <class Mat>
inline constexpr std::size_t mat_rows_v = mat_traits<remove_cvref_t<Mat>>::ROWS;

template <class Mat>
inline constexpr std::size_t mat_cols_v = mat_traits<remove_cvref_t<Mat>>::COLS;

template <std::size_t Dim>
using Vectord = Vector<double, Dim>;

template <std::size_t Dim>
using Vectorf = Vector<float, Dim>;

template <std::size_t Dim>
using Vectorcd = Vector<std::complex<double>, Dim>;

template <std::size_t Dim>
using Vectorcf = Vector<std::complex<float>, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixd = Matrix<double, Rows, Cols>;

template <std::size_t Dim>
using ColVectord = ColVector<double, Dim>;

template <std::size_t Dim>
using RowVectord = RowVector<double, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixf = Matrix<float, Rows, Cols>;

template <std::size_t Dim>
using ColVectorf = ColVector<float, Dim>;

template <std::size_t Dim>
using RowVectorf = RowVector<float, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixcd = Matrix<std::complex<double>, Rows, Cols>;

template <std::size_t Dim>
using ColVectorcd = ColVector<std::complex<double>, Dim>;

template <std::size_t Dim>
using RowVectorcd = RowVector<std::complex<double>, Dim>;

template <std::size_t Rows, std::size_t Cols>
using Matrixcf = Matrix<std::complex<float>, Rows, Cols>;

template <std::size_t Dim>
using ColVectorcf = ColVector<std::complex<float>, Dim>;

template <std::size_t Dim>
using RowVectorcf = RowVector<std::complex<float>, Dim>;

} // namespace linalg
