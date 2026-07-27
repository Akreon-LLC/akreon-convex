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

#include <concepts>

#include "linalg/scalar/scalar_concepts.hpp"
#include "linalg/vector/vector_concepts.hpp"
#include "matrix_traits.hpp"

namespace linalg {

template <class Mat>
concept MatrixContainer = requires {
  typename matrix_scalar_t<Mat>;
  { mat_rows_v<Mat> } -> std::convertible_to<std::size_t>;
  { mat_cols_v<Mat> } -> std::convertible_to<std::size_t>;
} && Field<matrix_scalar_t<Mat>>;

template <class Mat>
concept SquareMatrix =
    MatrixContainer<Mat> && mat_rows_v<Mat> == mat_cols_v<Mat>;

template <class Mat>
concept RealMatrix = MatrixContainer<Mat> && Real<matrix_scalar_t<Mat>>;

template <class Mat>
concept ComplexMatrix = MatrixContainer<Mat> && Complex<matrix_scalar_t<Mat>>;

template <class Mat, class Vec>
concept MatVecCompatible =
    MatrixContainer<Mat> && FieldVector<Vec> &&
    std::same_as<matrix_scalar_t<Mat>, vector_element_t<Vec>> &&
    mat_cols_v<Mat> == vector_dim_v<Vec>;

template <class MatA, class MatB>
concept MatMulCompatible =
    MatrixContainer<MatA> && MatrixContainer<MatB> &&
    std::same_as<matrix_scalar_t<MatA>, matrix_scalar_t<MatB>> &&
    mat_cols_v<MatA> == mat_rows_v<MatB>;

} // namespace linalg
