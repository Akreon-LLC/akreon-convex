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

#include "linalg/matrix/matrix_traits.hpp"

namespace linalg {

template <class Space>
struct space_traits;

template <typename T>
struct space_traits<Matrix<T, 1, 1>> {
  using value_type = T;
  using Point = Vector<T, 1>;
  using scalar_type = T;
  using element_type = Point;
  using coordinate_type = Point;
  static constexpr std::size_t DIM = 1;
};

template <typename T, std::size_t Dim>
struct space_traits<RowVector<T, Dim>> {
  using value_type = T;
  using Point = Vector<T, Dim>;
  using scalar_type = T;
  using element_type = Point;
  using coordinate_type = Point;
  static constexpr std::size_t DIM = Dim;
};

template <typename T, std::size_t Dim>
struct space_traits<ColVector<T, Dim>> {
  using value_type = T;
  using Point = Vector<T, Dim>;
  using scalar_type = T;
  using element_type = Point;
  using coordinate_type = Point;
  static constexpr std::size_t DIM = Dim;
};

template <typename T, std::size_t Rows, std::size_t Cols>
  requires(Rows > 1 && Cols > 1)
struct space_traits<Matrix<T, Rows, Cols>> {
  using value_type = T;
  using Point = Vector<T, Cols>;
  using scalar_type = T;
  using element_type = Matrix<T, Rows, Cols>;
  using coordinate_type = Vector<T, Rows * Cols>;
  static constexpr std::size_t DIM = Rows * Cols;
};

template <class Space>
using space_type_t = typename space_traits<remove_cvref_t<Space>>::value_type;

template <class Space>
using space_point_t = typename space_traits<remove_cvref_t<Space>>::Point;

template <class Space>
inline constexpr std::size_t space_dim_v =
    space_traits<remove_cvref_t<Space>>::DIM;

template <class Space>
using space_scalar_t =
    typename space_traits<remove_cvref_t<Space>>::scalar_type;

template <class Space>
using space_element_t =
    typename space_traits<remove_cvref_t<Space>>::element_type;

template <class Space>
using space_coordinate_t =
    typename space_traits<remove_cvref_t<Space>>::coordinate_type;

} // namespace linalg
