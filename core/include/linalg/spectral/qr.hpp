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
#include "config/solver_config.hpp"
#include "linalg/linalg_util.hpp"
#include "linalg/vector/householder.hpp"

namespace linalg {
/**
 * @brief QR decomposition class definition
 *
 * Orthonormalizes a given Matrix into an orthogonal basis Q and upper right
 * triangular matrix R. Deprecated gram schmidt for householder for better
 * numerical properties.
 *
 * Q is now compact in householder where A = QR => Q = A R^{-1} in the
 * representation of this class
 *
 * For short and fat matrices just transpose A -> QR -> transpose result back
 * into A = L Q result
 */
template <MatrixLike Mat>
struct QR {
  // @brief alias for underlying type
  using T = mat_type_t<Mat>;
  using ScalarT = scalar_type_t<T>;

  // @brief get rows and cols from matrix type
  static constexpr size_t Rows = mat_rows_v<Mat>;
  static constexpr size_t Cols = mat_cols_v<Mat>;

  static_assert(Rows >= Cols && Rows > 0,
                "QR Factorization requires [square | overdetermined] system");

  /**
   * @brief CTOR
   *
   * @param A (matrix of type T with Rows x Cols elements)
   */
  constexpr QR(const Mat& A) { solve_householder(A); }

  /**
   * @brief CTOR
   *
   * @param A (matrix)
   * @param config (FactorConfig)
   */
  constexpr QR(const Mat& A, const config::FactorConfig<ScalarT>& config_in)
      : config(config_in) {
    solve_householder(A);
  }

  /**
   * @brief get Q from compact householder Q
   *
   * To acquire Q we simply apply reflections in the reverse order with Q = I to
   * start. Then A = QR.
   *
   * @return orthonormal Q
   */
  constexpr auto Q() const { return get_unitary_matrix(); }

  /**
   * @brief basic get function for R matrix
   *
   * @return R
   */
  constexpr auto R() const { return get_upper_right_triangular_matrix(H); }

  /**
   * @brief get rank of matrix
   *
   * @return rank
   */
  constexpr auto rank(ScalarT tol = 1e-6) const {
    const auto LT = R();
    const size_t mn = std::min(Rows, Cols);
    size_t rank = 0;
    for (size_t i = 0; i < mn; ++i)
      if (L2<Vector<T, 1>>()(Vector<T, 1>{LT(i)(i)}) > tol) rank += 1;
    return rank;
  }

  /**
   * @brief least squares solver A x = b using QR decomposition
   *
   * LSQ method: A x = b => Q R x = b => R x = Q^T b
   *
   * @param vector b representing the outputs or observables of a linear system
   *
   * @return (x, event)
   */
  constexpr auto solve(const Vector<T, Rows>& b) const {
    Vector<T, Rows> c;

    const auto U = Q();
    const auto LT = R();

    if constexpr (ComplexLike<T>) c = conjugate_transpose(U) * b;
    else c = transpose(U) * b;

    return back_solve(LT, c);
  }

private:
  /**
   * @brief construct householder matrix from A (solve QR where Q is compact
   * form)
   *
   * Premultiplication matrix-vector product update with rank-1 update
   * PA = (I - beta v v^T) A = A - (beta v)*(v^T A)
   *
   * Postmultiplication
   * AP = A*(I - beta v v^T) = A - (A v)(beta v)^T
   *
   * Householder essentially annihilates subdiagonal column entries based on
   * each successive iterations reflections. Look at premultiplication case:
   *
   * P0 A annhiliates first col and creates basis vector in spanning first dir.
   * P1 P0 A annihilates second col and creates basis spanning 2 dim now... so
   * on and so forth till R = Pk Pk-1 ... P0 A where R is our right upper
   * triangular matrix
   *
   * @param matrix A of type T with Rows x Cols elements
   *
   * @return (void)
   */
  constexpr void solve_householder(Mat A) {
    H = A;
    constexpr_for<0, Cols>([&]<size_t j>(std::integral_constant<size_t, j>) {
      const auto& Q_jm = get_n_rows<j, Rows>(H);
      const auto& Q_jm_j = get_col(Q_jm, j);

      const auto& q = HouseholderVector(Q_jm_j);
      beta(j) = q.beta;
      const auto& v = as_col_vector(q.v);

      auto Q_jm_jn = get_n_cols<j, Cols>(Q_jm);
      if constexpr (ComplexLike<T>)
        Q_jm_jn = Q_jm_jn - (beta(j) * v) * (conjugate_transpose(v) * Q_jm_jn);
      else Q_jm_jn = Q_jm_jn - (beta(j) * v) * (transpose(v) * Q_jm_jn);

      for (size_t k = j, a = 0; k < Rows; ++k, ++a)
        for (size_t l = j, b = 0; l < Cols; ++l, ++b) H(k)(l) = Q_jm_jn(a)(b);

      for (size_t k = j + 1, a = 1; k < Rows; ++k, ++a) H(k)(j) = v(a)(0);
    });
  }

  /**
   * @brief get unitary matrix U from factored householder matrix
   *
   * To acquire U we simply apply reflections in the reverse order starting with
   * U = I => U^* A U = H
   *
   * @return unitary matrix U
   */
  constexpr auto get_unitary_matrix() const {
    auto U = make_identity<T, Rows>();
    constexpr_for<0, Cols>([&]<size_t jj>(std::integral_constant<size_t, jj>) {
      const size_t j = Cols - jj - 1;
      Vector<T, Rows - j> v;
      v(0) = T(1);
      for (size_t i = j + 1, a = 1; i < Rows; ++i, ++a) v(a) = H(i)(j);

      const auto& v_col = as_col_vector(v);

      auto Q_block = get_n_rows<j, Rows>(U);
      if constexpr (ComplexLike<T>)
        Q_block = Q_block -
                  (beta(j) * v_col) * (conjugate_transpose(v_col) * Q_block);
      else Q_block = Q_block - (beta(j) * v_col) * (transpose(v_col) * Q_block);

      for (size_t r = j, a = 0; r < Rows; ++r, ++a)
        for (size_t c = 0; c < Rows; ++c) U(r)(c) = Q_block(a)(c);
    });
    return U;
  }

  /**
   * @brief back solve linear systems of equations
   *
   * R x = rhs
   *
   * @param upper right triangular R matrix
   * @param rhs vector to be solved
   *
   * @return (x, event)
   */
  constexpr auto back_solve(const Matrix<T, Rows, Cols>& R,
                            const Vector<T, Rows>& rhs) const {
    Vector<T, Cols> x;
    constexpr auto mn = std::min(Rows, Cols);
    linalg::NumericalEvent event{linalg::NumericalEvent::NONE};
    for (int i = static_cast<int>(mn) - 1; i >= 0; --i) {
      T sum = 0;
      for (size_t j = i + 1; j < Cols; ++j) sum += (R(i)(j) * x(j));

      AK_ASSERT(R(i)(i) != T(0) &&
                "QR back_solve requires non-zero diagonal in R");
      T d = R(i)(i);
      ScalarT mag = constexpr_abs(d);
      if (mag <= config.floor_eps) {
        d = (mag > ScalarT(0)) ? d * config.floor_eps / mag
                               : T(config.floor_eps);
        event = linalg::NumericalEvent::NEAR_SINGULAR_TRIANGULAR_SOLVE;
      }
      x(i) = (rhs(i) - sum) / d;
    }
    return std::tuple{x, event};
  }

  /** @brief QR decomposition for matrix input A */
  Mat H;                ///< orthonormal basis Q (slighty different than GS
                        ///< as its fully compact and can use straight up)
  Vector<T, Rows> beta; ///< beta scalars
  config::FactorConfig<ScalarT> config; ///< Factor config
};

} // namespace linalg
