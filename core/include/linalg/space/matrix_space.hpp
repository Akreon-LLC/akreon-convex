/* Copyright © 2025 Akreon LLC. All rights reserved. */

#pragma once

#include "coordinate_map.hpp"

namespace linalg {

template <Field T, std::size_t Rows, std::size_t Cols>
struct MatrixSpace {};

template <Field T, std::size_t Rows, std::size_t Cols>
struct space_traits<MatrixSpace<T, Rows, Cols>> {
  using scalar_type = T;
  using element_type = Matrix<T, Rows, Cols>;
  using coordinate_type = Vector<T, Rows * Cols>;
  static constexpr std::size_t dim = Rows * Cols;
  static constexpr std::size_t DIM = dim;
};

template <Field T, std::size_t Rows, std::size_t Cols>
struct coordinate_map<MatrixSpace<T, Rows, Cols>> {
  static constexpr auto to_coordinates(const Matrix<T, Rows, Cols>& element) {
    Vector<T, Rows * Cols> coordinate;
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        coordinate(row * Cols + col) = element(row, col);
    return coordinate;
  }

  static constexpr auto from_coordinates(
      const Vector<T, Rows * Cols>& coordinate) {
    Matrix<T, Rows, Cols> element;
    for (std::size_t row = 0; row < Rows; ++row)
      for (std::size_t col = 0; col < Cols; ++col)
        element(row, col) = coordinate(row * Cols + col);
    return element;
  }
};

} // namespace linalg
