/* Copyright © 2025 Akreon LLC. All rights reserved. */

#pragma once

#include <concepts>

#include "space_traits.hpp"

namespace linalg {

template <class Space>
struct coordinate_map;

template <Field T, std::size_t Rows, std::size_t Cols>
  requires(Rows == 1 || Cols == 1)
struct coordinate_map<Matrix<T, Rows, Cols>> {
  static constexpr auto to_coordinates(const Vector<T, Rows * Cols>& element) {
    return element;
  }

  static constexpr auto from_coordinates(
      const Vector<T, Rows * Cols>& coordinate) {
    return coordinate;
  }
};

template <Field T, std::size_t Rows, std::size_t Cols>
  requires(Rows > 1 && Cols > 1)
struct coordinate_map<Matrix<T, Rows, Cols>> {
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

template <class Space>
using coordinate_map_t = coordinate_map<remove_cvref_t<Space>>;

template <class Space>
constexpr auto coordinates(const space_element_t<Space>& element) {
  return coordinate_map_t<Space>::to_coordinates(element);
}

template <class Space>
constexpr auto from_coordinates(const space_coordinate_t<Space>& coordinate) {
  return coordinate_map_t<Space>::from_coordinates(coordinate);
}

} // namespace linalg
