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

#include "linalg/vector/algebra_traits.hpp"

namespace linalg {

/// @brief forward declaration KKTBuilder
template <linalg::SquareMatrixLike HessianMat, linalg::MatrixLike EqualityMat>
  requires(std::same_as<mat_type_t<HessianMat>, mat_type_t<EqualityMat>>)
struct KKTBuilder;

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
template <SquareMatrixLike Mat, typename Derived>
struct Endomorphism;

/// @brief forward declaration for symmetric operator
template <SquareMatrixLike Mat, typename Derived>
struct SymmetricOperator;

/// @brief forward declaration for a matrix
template <SquareMatrixLike Mat>
struct SymmetricMatrix;

/// @brief forward declaration of hessenberg matrix
template <SquareMatrixLike Mat>
struct Hessenberg;

/// @brief forward declaration of hessian
template <SquareMatrixLike Mat, DifferentialType Type>
struct Hessian;

/// @brief forward declaration of LDL^T operator
template <SquareMatrixLike Mat>
struct LDL;

/// @brief type traits definition for linear operator (subspace of Rn)
template <SquareMatrixLike Mat, typename Derived>
struct space_traits<Endomorphism<Mat, Derived>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * N;
};

/// @brief type traits definition for symmetric operator (subspace of Rn)
template <SquareMatrixLike Mat, typename Derived>
struct space_traits<SymmetricOperator<Mat, Derived>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * (N + 1) / 2;
};

/// @brief type traits definition for symmetric matrices (subspace of R^n)
template <SquareMatrixLike Mat>
struct space_traits<SymmetricMatrix<Mat>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * (N + 1) / 2;
};

/// @brief type traits definition for hessenberg matrices (subspace of R^n)
template <SquareMatrixLike Mat>
struct space_traits<Hessenberg<Mat>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * N;
};

/// @brief type traits definition for hessian matrices (subspace of R^n)
template <SquareMatrixLike Mat, DifferentialType Type>
struct space_traits<Hessian<Mat, Type>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * N;
};

/// @brief type traits definition for LDL (subspace of R^n)
template <SquareMatrixLike Mat>
struct space_traits<LDL<Mat>> {
  using value_type = mat_type_t<Mat>;
  using Point = Mat;

  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t DIM = N * (N + 1) / 2;
};

/// @brief mat type traits definition for linear operator
template <SquareMatrixLike Mat, typename Derived>
struct mat_traits<Endomorphism<Mat, Derived>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief mat type traits definition for symmetric operator
template <SquareMatrixLike Mat, typename Derived>
struct mat_traits<SymmetricOperator<Mat, Derived>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief mat type traits definition for symmetric matrices
template <SquareMatrixLike Mat>
struct mat_traits<SymmetricMatrix<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief mat type traits definition for hessenberg matrices
template <SquareMatrixLike Mat>
struct mat_traits<Hessenberg<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief mat type traits definition for hessian matrices
template <SquareMatrixLike Mat, DifferentialType Type>
struct mat_traits<Hessian<Mat, Type>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief mat type traits definition for LDL
template <SquareMatrixLike Mat>
struct mat_traits<LDL<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t ROWS = mat_rows_v<Mat>;
  static constexpr size_t COLS = mat_cols_v<Mat>;
};

/// @brief vec type traits for symmetric operator
template <SquareMatrixLike Mat, typename Derived>
struct vec_traits<Endomorphism<Mat, Derived>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * N;
};

/// @brief vec type traits for symmetric operator
template <SquareMatrixLike Mat, typename Derived>
struct vec_traits<SymmetricOperator<Mat, Derived>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * (N + 1) / 2;
};

/// @brief vec type traits for symmetric matrices
template <SquareMatrixLike Mat>
struct vec_traits<SymmetricMatrix<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * (N + 1) / 2;
};

/// @brief vec type traits for hessenberg matrices
template <SquareMatrixLike Mat>
struct vec_traits<Hessenberg<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * N;
};

/// @brief vec type traits for hessian matrices
template <SquareMatrixLike Mat, DifferentialType Type>
struct vec_traits<Hessian<Mat, Type>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * N;
};

/// @brief vec type traits for LDL matrices
template <SquareMatrixLike Mat>
struct vec_traits<LDL<Mat>> {
  using value_type = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;
  static constexpr size_t SIZE = N * (N + 1) / 2;
};

/// @brief underlying type of symmetric operator
template <typename Operator>
using operator_type_t = typename mat_traits<Operator>::value_type;

/// @brief helper to extract rows from symmetric operator
template <typename Operator>
inline constexpr size_t operator_rows_v = mat_traits<Operator>::ROWS;

/// @brief helper to extract cols from symmetric operator
template <typename Operator>
inline constexpr size_t operator_cols_v = mat_traits<Operator>::COLS;

/// @brief helper to extract dim from symmetric operator
template <typename Operator>
inline constexpr size_t operator_dim_v = space_traits<Operator>::DIM;

/// @brief OperatorLike concept
template <typename Operator>
concept EndomorphismLike =
    MatrixLike<Operator> || requires(const Operator& op) {
      { op.type } -> std::convertible_to<SymmetricOperatorType>;
      {
        op.L
      } -> std::convertible_to<
          Vector<operator_type_t<Operator>, operator_dim_v<Operator>>>;
    };

}  // namespace linalg
