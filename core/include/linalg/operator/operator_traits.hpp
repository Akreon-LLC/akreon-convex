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

#include "linalg/constants.hpp"
#include "linalg/space/space_concepts.hpp"
#include "linalg/space/coordinate_map.hpp"

namespace linalg {

/// @brief simple enum to define symmetric matrix types
/// INV = Invalid
/// IND = Indefinite
/// PSD = Positive Semidefinite
/// PD = Positive Definite
/// NSD = Negative Semidefinite
/// ND = Negative Definite
enum struct SymmetricOperatorType { INV = 0, IND, PSD, PD, NSD, ND };

/// @brief differential type
enum struct DifferentialType { PUSHFORWARD, PULLBACK };

/// @brief forward declaration for linear operator
template <SquareMatrix Mat, typename Derived>
struct Endomorphism;

/// @brief forward declaration for symmetric operator
template <SquareMatrix Mat, typename Derived>
struct SymmetricOperator;

/// @brief forward declaration for a matrix
template <SquareMatrix Mat>
struct SymmetricMatrix;

/// @brief forward declaration of hessenberg matrix
template <SquareMatrix Mat>
struct Hessenberg;

/// @brief forward declaration of hessian
template <SquareMatrix Mat, DifferentialType Type>
struct Hessian;

/// @brief forward declaration of LDL^T operator
template <SquareMatrix Mat>
struct LDL;

/// @brief forward declaration of P(x) / P(x^{-1})
template <SquareMatrix Mat, typename Circ>
struct QuadraticRep;

/// @brief forward declaration of cholesky
template <SquareMatrix Mat>
struct Cholesky;

/// @brief type traits definition for linear operator (subspace of Rn)
template <SquareMatrix Mat, typename Derived>
struct space_traits<Endomorphism<Mat, Derived>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * N;
};

/// @brief type traits definition for symmetric operator (subspace of Rn)
template <SquareMatrix Mat, typename Derived>
struct space_traits<SymmetricOperator<Mat, Derived>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * (N + 1) / 2;
};

/// @brief type traits definition for symmetric matrices (subspace of R^n)
template <SquareMatrix Mat>
struct space_traits<SymmetricMatrix<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;
  using scalar_type = value_type;
  using element_type = Point;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * (N + 1) / 2;
  using coordinate_type = Vector<value_type, DIM>;
};

template <SquareMatrix Mat>
struct coordinate_map<SymmetricMatrix<Mat>> {
  using Space = SymmetricMatrix<Mat>;
  using T = space_scalar_t<Space>;
  static constexpr std::size_t N = mat_rows_v<Mat>;

  static constexpr auto to_coordinates(const Mat& element) {
    space_coordinate_t<Space> coordinate;
    for (std::size_t row = 0, index = 0; row < N; ++row) {
      for (std::size_t col = 0; col <= row; ++col, ++index) {
        coordinate(index) = element(row, col);
        if (row != col) coordinate(index) *= sqrt2_v<T>;
      }
    }
    return coordinate;
  }

  static constexpr auto from_coordinates(
      const space_coordinate_t<Space>& coordinate) {
    Mat element;
    for (std::size_t row = 0, index = 0; row < N; ++row) {
      for (std::size_t col = 0; col <= row; ++col, ++index) {
        auto value = coordinate(index);
        if (row != col) value /= sqrt2_v<T>;
        element(row, col) = element(col, row) = value;
      }
    }
    return element;
  }
};

/// @brief type traits definition for hessenberg matrices (subspace of R^n)
template <SquareMatrix Mat>
struct space_traits<Hessenberg<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * N;
};

/// @brief type traits definition for hessian matrices (subspace of R^n)
template <SquareMatrix Mat, DifferentialType Type>
struct space_traits<Hessian<Mat, Type>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * N;
};

/// @brief type traits definition for Cholesky (subspace of R^n)
template <SquareMatrix Mat>
struct space_traits<LDL<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * (N + 1) / 2;
};

/// @brief type traits definition for P(x) (subspace of J)
template <SquareMatrix Mat, typename Circ>
struct space_traits<QuadraticRep<Mat, Circ>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * N;
};

/// @brief type traits definition for LDL (subspace of R^n)
template <SquareMatrix Mat>
struct space_traits<Cholesky<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  using Point = Mat;

  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t DIM = N * (N + 1) / 2;
};

/// @brief mat type traits definition for linear operator
template <SquareMatrix Mat, typename Derived>
struct mat_traits<Endomorphism<Mat, Derived>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for symmetric operator
template <SquareMatrix Mat, typename Derived>
struct mat_traits<SymmetricOperator<Mat, Derived>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for symmetric matrices
template <SquareMatrix Mat>
struct mat_traits<SymmetricMatrix<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for hessenberg matrices
template <SquareMatrix Mat>
struct mat_traits<Hessenberg<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for hessian matrices
template <SquareMatrix Mat, DifferentialType Type>
struct mat_traits<Hessian<Mat, Type>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for LDL
template <SquareMatrix Mat>
struct mat_traits<LDL<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for P(x)
template <SquareMatrix Mat, typename Circ>
struct mat_traits<QuadraticRep<Mat, Circ>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief mat type traits definition for Cholesky
template <SquareMatrix Mat>
struct mat_traits<Cholesky<Mat>> {
  using value_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t ROWS = mat_rows_v<Mat>;
  static constexpr std::size_t COLS = mat_cols_v<Mat>;
  using MatT = Mat;
};

/// @brief vec type traits for linear operator
template <SquareMatrix Mat, typename Derived>
struct vector_traits<Endomorphism<Mat, Derived>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for symmetric operator
template <SquareMatrix Mat, typename Derived>
struct vector_traits<SymmetricOperator<Mat, Derived>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for symmetric matrices
template <SquareMatrix Mat>
struct vector_traits<SymmetricMatrix<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for hessenberg matrices
template <SquareMatrix Mat>
struct vector_traits<Hessenberg<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for hessian matrices
template <SquareMatrix Mat, DifferentialType Type>
struct vector_traits<Hessian<Mat, Type>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for LDL matrices
template <SquareMatrix Mat>
struct vector_traits<LDL<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for quadratic rep matrices
template <SquareMatrix Mat, typename Circ>
struct vector_traits<QuadraticRep<Mat, Circ>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * N;
  using Vec = Vector<element_type, N>;
};

/// @brief vec type traits for cholesky matrices
template <SquareMatrix Mat>
struct vector_traits<Cholesky<Mat>> {
  using element_type = matrix_scalar_t<Mat>;
  static constexpr std::size_t N = mat_rows_v<Mat>;
  static constexpr std::size_t dim = N * (N + 1) / 2;
  using Vec = Vector<element_type, N>;
};

/// @brief symmetric matrix space type trait
template <typename Space>
struct is_symmetric_matrix_space : std::false_type {};

/// @brief symmetric matrix space type trait
template <SquareMatrix Mat>
struct is_symmetric_matrix_space<SymmetricMatrix<Mat>> : std::true_type {};

/// @brief symmetric matrix space type trait var
template <typename Space>
inline constexpr bool is_symmetric_matrix_space_v =
    is_symmetric_matrix_space<remove_cvref_t<Space>>::value;

/// @brief underlying type of symmetric operator
template <typename Operator>
using operator_type_t =
    typename mat_traits<remove_cvref_t<Operator>>::value_type;

/// @brief operator vec type (dom == codom)
template <typename Operator>
using operator_vec_t = typename vector_traits<remove_cvref_t<Operator>>::Vec;

/// @brief operator mat type
template <typename Operator>
using operator_mat_t = typename mat_traits<remove_cvref_t<Operator>>::MatT;

/// @brief helper to extract rows from symmetric operator
template <typename Operator>
inline constexpr std::size_t operator_rows_v =
    mat_traits<remove_cvref_t<Operator>>::ROWS;

/// @brief helper to extract cols from symmetric operator
template <typename Operator>
inline constexpr std::size_t operator_cols_v =
    mat_traits<remove_cvref_t<Operator>>::COLS;

/// @brief helper to extract dim from symmetric operator
template <typename Operator>
inline constexpr std::size_t operator_dim_v =
    space_traits<remove_cvref_t<Operator>>::DIM;

/// @brief Linear Operator Concept
template <typename Operator>
concept LinearOperatorLike =
    MatrixContainer<Operator> ||
    requires(const Operator& op, const operator_type_t<Operator>& X) {
      { op.apply(X) } -> std::convertible_to<operator_type_t<Operator>>;
    } || requires(const Operator& op, const operator_vec_t<Operator>& x) {
      { op.apply(x) } -> std::convertible_to<operator_vec_t<Operator>>;
    };

/// @brief Endomorphism concept
template <typename Operator>
concept EndomorphismLike =
    LinearOperatorLike<Operator> || requires(const Operator& op) {
      { op.type } -> std::convertible_to<SymmetricOperatorType>;
      {
        op.L
      } -> std::convertible_to<
          Vector<operator_type_t<Operator>, operator_dim_v<Operator>>>;
    };
} // namespace linalg
