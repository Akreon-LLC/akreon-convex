/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "householder.hpp"
#include "matrix_ops.hpp"

namespace algebra {
/**
 * @brief Hessenberg Matrix
 *
 * H is an upper right triangular matrix formed by successive Householder
 * reductions creating the similarity transform: H = U_0^T A U_0.
 */
template <typename T, size_t N>
class HessenbergMatrix {
 public:
  /**
   * @brief CTOR
   *
   * @param Matrix A of type T with N x N elements
   */
  constexpr HessenbergMatrix(const Matrix<T, N, N>& A) { make_hessenberg(A); }

  /**
   * @brief operator override to acquire H from HessenbergMatrix object
   *
   * @return Matrix of type T with N x N elements (upper right triangular)
   */
  constexpr auto operator()() const {
    Matrix<T, N, N> res;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        if (i >= 2 && j <= (i - 2)) continue;
        res[i][j] = H(i)(j);
      }
    }
    return res;
  }

  /**
   * @brief simple getter for unitar matrix
   *
   * @return U of type T N x N
   */
  constexpr auto get_U() const { return get_unitary_matrix(); }

  /**
   * @brief get beta vector
   *
   * @return beta vector of type T with N elements
   */
  constexpr auto get_beta() const { return beta; }

 private:
  /**
   * @brief Householder reduction to hessenberg form
   *
   * A \in R^nxn \to H = U_0^T A U_0 where H is upper Hessenberg and U_0 is
   * unitary
   *
   * @param matrix A type T with N x N elements
   *
   * @return H (hessenberg matrix)
   */
  constexpr void make_hessenberg(const Matrix<T, N, N>& A) {
    H = A;
    constexpr_for<0, N - 2>([&]<size_t k>(std::integral_constant<size_t, k>) {
      const auto& H_block = get_n_rows<k + 1, N>(H);
      const auto& H_col = get_col(H_block, k);
      const auto h = HouseholderVector(H_col);

      const auto& v = as_col_vector(h.get_v());
      beta[k] = h.get_beta();

      if (beta[k] == scalar_type_t<T>(0)) return;

      // (P1...Pk-1)^T A (P1...Pk-1)
      auto H_block_kn = get_n_cols<k, N>(H_block);
      if constexpr (is_complex_v<T>)
        H_block_kn =
            H_block_kn - (beta[k] * v * conjugate_transpose(v)) * H_block_kn;
      else
        H_block_kn = H_block_kn - (beta[k] * v * transpose(v)) * H_block_kn;

      for (size_t i = k + 1, a = 0; i < N; ++i, ++a)
        for (size_t j = k, b = 0; j < N; ++j, ++b) H[i][j] = H_block_kn(a)(b);

      // (P1...Pk)^T A (P1...Pk) -- handling case when Pk is of order N - k
      auto H_zn_kn = get_block<0, N, k+1, N>(H);
      if constexpr (is_complex_v<T>)
        H_zn_kn = H_zn_kn - H_zn_kn * (beta(k) * v * conjugate_transpose(v));
      else
        H_zn_kn = H_zn_kn - H_zn_kn * (beta(k) * v * transpose(v));

      for (size_t i = 0, a = 0; i < N; ++i, ++a)
        for (size_t j = k + 1, b = 0; j < N; ++j, ++b) H[i][j] = H_zn_kn(a)(b);

      for (size_t j = k + 2, a = 1; j < N; ++j, ++a) H[j][k] = v(a)(0);
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
    auto U = make_identity<T, N>();
    constexpr_for<0, N - 2>([&]<size_t k>(std::integral_constant<size_t, k>) {
      const size_t j = N - k - 1;
      Vector<T, j> v;
      v[0] = T(1);
      for (size_t i = k + 2, a = 1; i < N; ++i, ++a) v[a] = H(i)(k);

      const auto& v_col = as_col_vector(v);

      auto U_block = get_n_cols<k + 1, N>(U);
      if constexpr (is_complex_v<T>)
        U_block =
            U_block - (beta(k) * v_col * conjugate_transpose(v_col)) * U_block;
      else
        U_block = U_block - (beta(k) * v_col * transpose(v_col)) * U_block;

      for (size_t r = 0; r < N; ++r)
        for (size_t c = k + 1, a = 0; c < N; ++c, ++a) U[r][c] = U_block(r)(a);
    });
    return U;
  }

  /** @brief Vector member variables */
  Matrix<T, N, N> H; ///< Hessenberg matrix
  Vector<T, N> beta; ///< beta vector from householder reflections
};

}  // namespace algebra
