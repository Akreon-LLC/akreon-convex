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

#include <algorithm>

#include "config/assert.hpp"
#include "linalg/linalg_util.hpp"
#include "symmetric_operator.hpp"

namespace linalg {

/**
 * @brief LDL^T factorization of a square / symmetric matrix A
 */
template <SquareMatrix Mat>
struct LDL : SymmetricOperator<Mat, LDL<Mat>> {
  /// @brief alias for symmetric matrix
  using SymmetricBase = SymmetricOperator<Mat, LDL<Mat>>;
  using typename SymmetricBase::ComplexT;
  using typename SymmetricBase::ScalarT;
  using typename SymmetricBase::T;

  /// @brief helpers for symmetric matrix
  using SymmetricBase::DIM;
  using SymmetricBase::N;

  static_assert(N > 1, "LDL^T requires at least a 2x2 matrix");

  /**
   * @brief CTOR
   *
   * @param A (matrix)
   */
  constexpr explicit LDL(const Mat& A) : SymmetricBase(A) { factor(A); }

  /**
   * @brief CTOR
   *
   * @param A (matrix)
   * @param config (FactorConfig)
   */
  constexpr explicit LDL(const Mat& A, const config::FactorConfig<T>& config_in)
      : SymmetricBase(A), config(config_in) {
    factor(A);
  }

  /**
   * @brief D
   *
   * @return D diag mat
   */
  constexpr auto D() const { return make_diag(D_vec); }

  /**
   * @brief L
   *
   * @return L mat
   */
  constexpr auto L() const { return make_unit_lower_triangle<Mat>(L_vec); }

  /**
   * @brief solve (forward solve)
   *
   * A x = b, A = LDL^T
   * L ( D ( L^T x) ) = b
   * L^T x = z
   * D z = y
   * L y = b
   *
   * @param vector b
   *
   * @return x
   */
  constexpr auto solve(const Vector<T, N>& b) const {
    const auto L_mat = L();
    const auto y = forward_solve(L_mat, b);
    const auto [z, event] = diag_solve(y);
    if (is_fatal_event(event)) return std::tuple(z, event);

    return std::tuple(back_solve(transpose(L_mat), z),
                      linalg::NumericalEvent::NONE);
  }

  /**
   * @brief det
   *
   * det(A) = prod_i^n(d_{ii})
   *
   * @return det(A)
   */
  constexpr T det() const {
    T res{1};
    std::for_each(D_vec.begin(), D_vec.end(), [&](const auto& d) { res *= d; });
    return res;
  }

  /**
   * @brief apply
   *
   * @param Y (mat)
   *
   * @return A^{-1} Y
   */
  template <linalg::MatrixContainer MatIn>
    requires(std::same_as<linalg::matrix_scalar_t<MatIn>, T>)
  constexpr auto apply(const MatIn& Y) const {
    constexpr std::size_t YRows = linalg::mat_rows_v<MatIn>;
    constexpr std::size_t YCols = linalg::mat_cols_v<MatIn>;
    static_assert(YRows == N, "Codom of Y must match expected dom of A");

    linalg::Matrix<T, N, YCols> A_inv_Y{};
    for (std::size_t i = 0; i < YCols; ++i) {
      const auto y_col = get_col(Y, i);
      const auto [res, event] = solve(y_col);
      if (linalg::is_fatal_event(event)) return std::tuple(Y, event);
      for (std::size_t j = 0; j < YRows; ++j) A_inv_Y(j, i) = res(j);
    }
    return std::tuple(A_inv_Y, linalg::NumericalEvent::NONE);
  }

private:
  /**
   * @brief LDL^T of a square matrix
   *
   * @param A (matrix)
   *
   * @return void
   */
  constexpr void factor(Mat A) {
    const auto& tol = config.tol;
    auto ldl_t = [&]<std::size_t j = 0>(this auto&& self) -> bool {
      /// (dakre) base case for recursion
      if constexpr (j >= SymmetricBase::N) return true;
      else {
        /// (dakre) handle j = 0 iteration separately (don't have prior intertia
        /// build up, only need to scale first col by first diag elem)
        constexpr std::size_t next = j + 1;
        if constexpr (j == 0) {
          if (constexpr_abs(A(0, 0)) < tol) return false;

          const auto A_col = get_block<next, SymmetricBase::N, 0, next>(A);
          const auto col_up = (T(1) / A(0, 0)) * A_col;

          A = set_block<next, SymmetricBase::N, 0, next>(A, col_up);
        } else {
          /// (dakre) nominal LDL^T algo denoted in Golub & Van Load p. 158
          Vector<T, j> v{};
          for (std::size_t k = 0; k < j; ++k) v(k) = A(j, k) * A(k, k);

          const auto A_row = get_block<j, next, 0, j>(A);
          A(j, j) = A(j, j) - (A_row * v)(0);

          if (constexpr_abs(A(j, j)) < tol) return false;

          if constexpr (next < SymmetricBase::N) {
            const auto A_col = get_block<next, SymmetricBase::N, j, next>(A);
            const auto A_blk = get_block<next, SymmetricBase::N, 0, j>(A);

            const auto col_up =
                T(1) / A(j, j) * (A_col - as_col_vector(A_blk * v));

            A = set_block<next, SymmetricBase::N, j, next>(A, col_up);
          }
        }
        return self.template operator()<next>();
      }
    };

    if (!ldl_t()) {
      this->type = SymmetricOperatorType::INV;
      return;
    }

    L_vec = get_lower_triangle(A);
    D_vec = get_diag(A);

    this->compute_definiteness(D_vec, config.tol);
  }

  /**
   * @brief forward solve
   *
   * L y = b
   *
   * @param matrix L
   * @param vector b
   *
   * @return y
   */
  constexpr auto forward_solve(const Mat& L_mat, const Vector<T, N>& b) const {
    Vector<T, N> y{};
    for (std::size_t i = 0; i < N; ++i) {
      T sum(0);
      for (std::size_t j = 0; j < i; ++j) sum += L_mat(i, j) * y(j);
      y(i) = b(i) - sum;
    }
    return y;
  }

  /**
   * @brief diag solve
   *
   * @param y (vec)
   *
   * @return (z, event)
   */
  constexpr auto diag_solve(const Vector<T, N>& y) const {
    Vector<T, N> z{};

    AK_ASSERT(this->is_valid() &&
              "LDL::diag_solve requires a valid factorization (non-singular)");
    if (!this->is_valid())
      return std::tuple(z, linalg::NumericalEvent::FACTORIZATION_FAILED);

    for (std::size_t i = 0; i < N; ++i) z(i) = y(i) / D_vec(i);

    return std::tuple(z, linalg::NumericalEvent::NONE);
  }

  /**
   * @brief back solve
   *
   * @param matrix L
   * @param vector z
   *
   * @return vector x
   */
  constexpr auto back_solve(const Mat& L_mat, const Vector<T, N>& z) const {
    Vector<T, N> x{};
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
      T sum(0);
      for (int j = i + 1; j < static_cast<int>(N); ++j)
        sum += L_mat(i, j) * x(j);
      x(i) = z(i) - sum;
    }
    return x;
  }

  /** @brief member variables */
  config::FactorConfig<T> config; ///< factor config
  Vector<T, DIM - N> L_vec;       ///< lower left triangle of factorization
  Vector<T, N> D_vec;             ///< diag of factorization
};

} // namespace linalg
