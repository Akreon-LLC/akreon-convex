/* Copyright © 2025 Akreon LLC. All rights reserved. */

#pragma once

#include "coordinate_map.hpp"
#include "linalg/scalar/scalar_concepts.hpp"

namespace linalg {

template <Field T, std::size_t Dim>
struct CoordinateSpace {};

template <Field T, std::size_t Dim>
struct space_traits<CoordinateSpace<T, Dim>> {
  using scalar_type = T;
  using element_type = Vector<T, Dim>;
  using coordinate_type = Vector<T, Dim>;
  static constexpr std::size_t dim = Dim;
  static constexpr std::size_t DIM = dim;
};

template <Field T, std::size_t Dim>
struct coordinate_map<CoordinateSpace<T, Dim>> {
  static constexpr auto to_coordinates(const Vector<T, Dim>& element) {
    return element;
  }

  static constexpr auto from_coordinates(const Vector<T, Dim>& coordinate) {
    return coordinate;
  }
};

} // namespace linalg
