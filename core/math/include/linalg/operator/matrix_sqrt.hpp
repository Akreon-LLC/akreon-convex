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

#include "linalg/linalg_util.hpp"
#include "linalg/spectral/eig_decomp.hpp"

namespace linalg {

/**
 * @brief Matrix sqrt (X^{1/2})
 */
template <SquareMatrixLike MatIn>
struct MatrixSqrt {
  /// @brief alias & types
  using Mat = linalg::remove_cvref_t<MatIn>;
  using T = mat_type_t<Mat>;
  static constexpr size_t N = mat_rows_v<Mat>;

  static_assert(RealLike<T>, "Sqrt assumes reals domain");
  static_assert(N >= 1, "Sqrt requires matrix dim of at least 1");

  /**
   * @brief operator
   *
   * @param A (matrix)
   *
   * @return A^{1/2}
   */
  constexpr auto operator()(const Mat& A) const { return apply(A); }

  /**
   * @brief apply
   *
   * @param A (matrix)
   *
   * @return A^{1/2}
   */
  constexpr auto apply(const Mat& A) const {
    const auto eig_decomp = EigenDecomposition(A);
    AK_ASSERT(eig_decomp.converged && "Eigen decomposition failed");
    const auto eigs = eig_decomp.lambda;
    Vector<std::complex<T>, N> sqrt_eigs{};
    for (size_t i = 0; i < N; ++i)
      sqrt_eigs(i) =
          std::complex<T>(constexpr_sqrt_safe<T>(eigs(i).real()));

    const auto V = eig_decomp.V;
    const auto VTV = V * (make_diag(sqrt_eigs) * conjugate_transpose(V));

    Mat A_sqrt{};
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        AK_ASSERT(is_real(VTV(i)(j)) &&
                  "V^TV result has imaginary component which is incompatible "
                  "with domain of R");
        A_sqrt(i)(j) = VTV(i)(j).real();
      }
    }

    return A_sqrt;
  }
};

}  // namespace linalg
