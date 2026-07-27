/* Copyright © 2025 Akreon LLC. All rights reserved. */

#pragma once

#include "coordinate_map.hpp"
#include "linalg/operator/symmetric_matrix.hpp"

namespace linalg {

template <Real T, std::size_t Dim>
struct SymmetricMatrixSpace {};

template <Real T, std::size_t Dim>
struct space_traits<SymmetricMatrixSpace<T, Dim>> {
  using scalar_type = T;
  using element_type = SymmetricMatrix<Matrix<T, Dim, Dim>>;
  using coordinate_type = Vector<T, Dim*(Dim + 1) / 2>;
  static constexpr std::size_t dim = Dim * (Dim + 1) / 2;
  static constexpr std::size_t DIM = dim;
};

template <Real T, std::size_t Dim>
struct coordinate_map<SymmetricMatrixSpace<T, Dim>> {
  using Space = SymmetricMatrixSpace<T, Dim>;

  static constexpr auto to_coordinates(const space_element_t<Space>& element) {
    return element.svec;
  }

  static constexpr auto from_coordinates(
      const space_coordinate_t<Space>& coordinate) {
    return space_element_t<Space>{make_smat<Dim>(coordinate)};
  }
};

template <class AmbientSpace>
  requires CoordinatizedSpace<AmbientSpace>
struct PsdCone {
  using ambient_space = AmbientSpace;
  using element_type = space_element_t<AmbientSpace>;

  static constexpr bool contains(const element_type& element) {
    return element.is_psd();
  }
};

} // namespace linalg
