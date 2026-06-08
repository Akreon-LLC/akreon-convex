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

#include "endomorphism.hpp"
#include "linalg/vector/householder.hpp"

namespace linalg {
/**
 * @brief Hessenberg Matrix
 *
 * H is an upper right triangular matrix formed by successive Householder
 * reductions creating the similarity transform: H = U_0^T A U_0.
 */
template <SquareMatrixLike Mat>
struct Hessenberg : Endomorphism<Mat, Hessenberg<Mat>> {
  /// @brief alias from base class
  using OperatorBase = Endomorphism<Mat, Hessenberg<Mat>>;
  using typename OperatorBase::T;

  /// @brief helper from base to get num rows
  using OperatorBase::N;

  /**
   * @brief CTOR
   *
   * Householder reduction to hessenberg form
   *
   * A in R^nxn to H = U_0^T A U_0 where H is upper Hessenberg and U_0 is
   * unitary
   *
   * @param Matrix A of type T with N x N elements
   */
  constexpr explicit Hessenberg(const Mat& A) {
    H_pack = A;
    constexpr_for<0, N - 2>([&]<size_t k>(std::integral_constant<size_t, k>) {
      const auto& H_block = get_n_rows<k + 1, N>(H_pack);
      const auto& H_col = get_col(H_block, k);
      const auto h = HouseholderVector(H_col);

      const auto& v = as_col_vector(h.v);
      beta(k) = h.beta;

      if (beta(k) == scalar_type_t<T>(0)) return;

      // (P1...Pk-1)^T A (P1...Pk-1)
      auto H_block_kn = get_n_cols<k, N>(H_block);
      if constexpr (ComplexLike<T>)
        H_block_kn =
            H_block_kn - (beta(k) * v * conjugate_transpose(v)) * H_block_kn;
      else H_block_kn = H_block_kn - (beta(k) * v * transpose(v)) * H_block_kn;

      for (size_t i = k + 1, a = 0; i < N; ++i, ++a)
        for (size_t j = k, b = 0; j < N; ++j, ++b)
          H_pack(i)(j) = H_block_kn(a)(b);

      // (P1...Pk)^T A (P1...Pk) -- handling case when Pk is of order N - k
      auto H_zn_kn = get_block<0, N, k + 1, N>(H_pack);
      if constexpr (ComplexLike<T>)
        H_zn_kn = H_zn_kn - H_zn_kn * (beta(k) * v * conjugate_transpose(v));
      else H_zn_kn = H_zn_kn - H_zn_kn * (beta(k) * v * transpose(v));

      for (size_t i = 0, a = 0; i < N; ++i, ++a)
        for (size_t j = k + 1, b = 0; j < N; ++j, ++b)
          H_pack(i)(j) = H_zn_kn(a)(b);

      for (size_t j = k + 2, a = 1; j < N; ++j, ++a) H_pack(j)(k) = v(a)(0);
    });
  }

  /**
   * @brief operator override to acquire H from Hessenberg object
   *
   * @return Matrix of type T with N x N elements (upper right triangular)
   */
  constexpr auto H() const {
    Matrix<T, N, N> res{};
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        if (i >= 2 && j <= (i - 2)) continue;
        res(i)(j) = H_pack(i)(j);
      }
    }
    return res;
  }

  /**
   * @brief simple getter for unitary matrix
   *
   * Get unitary matrix U from factored householder matrix
   *
   * To acquire U we simply apply reflections in the reverse order starting with
   * U = I => U^* A U = H
   *
   * @return U of type T N x N
   */
  constexpr auto U() const {
    auto U = make_identity<T, N>();
    constexpr_for<0, N - 2>([&]<size_t k>(std::integral_constant<size_t, k>) {
      const size_t j = N - k - 1;
      Vector<T, j> v;
      v(0) = T(1);
      for (size_t i = k + 2, a = 1; i < N; ++i, ++a) v(a) = H_pack(i)(k);

      const auto& v_col = as_col_vector(v);

      auto U_block = get_n_cols<k + 1, N>(U);
      if constexpr (ComplexLike<T>)
        U_block = U_block -
                  beta(k) * (U_block * (v_col * conjugate_transpose(v_col)));
      else U_block = U_block - beta(k) * (U_block * (v_col * transpose(v_col)));

      for (size_t r = 0; r < N; ++r)
        for (size_t c = k + 1, a = 0; c < N; ++c, ++a) U(r)(c) = U_block(r)(a);
    });
    return U;
  }

  /**
   * @brief indexing into upper right H
   *
   * @param row index
   * @param col index
   *
   * @return elem
   */
  constexpr auto operator()(const size_t& i, const size_t& j) const {
    if (i >= 2 && j <= (i - 2)) return T(0);
    return H_pack(i)(j);
  }

  /** @brief Vector member variables */
  Vector<T, N> beta; ///< beta vector from householder reflections
private:
  Matrix<T, N, N>
      H_pack; ///< Hessenberg matrix (packed w/ householder below diag)
};

} // namespace linalg
