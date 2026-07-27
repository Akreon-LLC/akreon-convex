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

#include "config/assert.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>

#include "linalg/scalar/scalar_concepts.hpp"
#include "linalg/vector/vector.hpp"

namespace linalg {

/** A constexpr, fixed-size, row-major matrix container. */
template <Field T, std::size_t Rows, std::size_t Cols>
struct Matrix {
  constexpr Matrix() = default;

  template <typename... RowVectors>
    requires(sizeof...(RowVectors) == Rows && Rows > 0 &&
             (std::same_as<remove_cvref_t<RowVectors>, Vector<T, Cols>> && ...))
  constexpr explicit Matrix(RowVectors&&... rows) {
    std::size_t row = 0;
    const auto copy_row = [this, &row](const Vector<T, Cols>& values) {
      for (std::size_t col = 0; col < Cols; ++col)
        (*this)(row, col) = values(col);
      ++row;
    };
    (copy_row(rows), ...);
  }

  constexpr const T& operator()(std::size_t row, std::size_t col) const {
    AK_ASSERT(row < Rows && col < Cols &&
              "Matrix const access index out of bounds");
    return elements[row * Cols + col];
  }

  constexpr T& operator()(std::size_t row, std::size_t col) {
    AK_ASSERT(row < Rows && col < Cols &&
              "Matrix mutable access index out of bounds");
    return elements[row * Cols + col];
  }

  constexpr auto row(std::size_t index) const {
    AK_ASSERT(index < Rows && "Matrix row index out of bounds");
    Vector<T, Cols> result;
    for (std::size_t col = 0; col < Cols; ++col)
      result(col) = (*this)(index, col);
    return result;
  }

  constexpr auto begin() { return elements.begin(); }
  constexpr auto begin() const { return elements.begin(); }
  constexpr auto end() { return elements.end(); }
  constexpr auto end() const { return elements.end(); }
  constexpr auto view() { return std::span<T, Rows * Cols>{elements}; }
  constexpr auto view() const {
    return std::span<const T, Rows * Cols>{elements};
  }
  constexpr auto size() const { return Rows * Cols; }

  constexpr bool operator==(const Matrix&) const = default;

private:
  std::array<T, Rows * Cols> elements{};
};

template <Field T, std::size_t Cols, typename... Rows>
Matrix(Vector<T, Cols>, Rows...) -> Matrix<T, 1 + sizeof...(Rows), Cols>;

template <Field T, std::size_t Dim>
constexpr auto as_row_matrix(const Vector<T, Dim>& vector) {
  return Matrix<T, 1, Dim>{vector};
}

template <Field T, std::size_t Dim>
constexpr auto as_col_matrix(const Vector<T, Dim>& vector) {
  Matrix<T, Dim, 1> result;
  for (std::size_t i = 0; i < Dim; ++i) result(i, 0) = vector(i);
  return result;
}

template <Field T, std::size_t Rows, std::size_t Cols>
  requires(Rows == 1 || Cols == 1)
constexpr auto as_vector(const Matrix<T, Rows, Cols>& matrix) {
  constexpr auto Dim = Rows == 1 ? Cols : Rows;
  Vector<T, Dim> result;
  for (std::size_t i = 0; i < Dim; ++i)
    result(i) = Rows == 1 ? matrix(0, i) : matrix(i, 0);
  return result;
}

} // namespace linalg
