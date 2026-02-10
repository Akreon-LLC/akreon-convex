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

#include <assert.h>

#include "hessian.hpp"
#include "linalg/vector/algebra_traits.hpp"

namespace linalg {

/**
 * @brief \tilde{A} = [ -F I ]
 *
 * where F z + g = x, x \in ambient and z \in dom(F)
 * \tilde{A} [ z; x ] -> x - F z
 *
 * @param F Equality Matrix
 *
 * @return \tilde{A}
 */
template <MatrixLike EqualityMat>
constexpr auto make_augmented_equality(const EqualityMat& F) {
  using T = mat_type_t<EqualityMat>;
  static constexpr size_t ERows = mat_rows_v<EqualityMat>;
  static constexpr size_t ECols = mat_cols_v<EqualityMat>;
  Matrix<T, ERows, ERows + ECols> A{};
  A = set_block<0, ERows, 0, ECols>(A, T(-1) * F);
  const auto I = make_identity<T, ERows>();
  A = set_block<0, ERows, ECols, ERows + ECols>(A, I);
  return A;
}

/**
 * @brief \tilde{H} = [ 0 0; 0 F''(w) ]
 *
 * where \tilde{H} [ z; x ] -> F''(w) x as nominal
 *
 * @param H Hessian
 *
 * @return \tilde{H}
 */
template <SquareMatrixLike HessianMat>
constexpr auto make_augmented_hessian(const HessianMat& H) {
  using T = mat_type_t<HessianMat>;
  static constexpr size_t N = mat_rows_v<HessianMat>;
  Matrix<T, 2*N, 2*N> H_aug{};
  H_aug = set_block<N, 2*N, N, 2*N>(H_aug, H);
  return H_aug;
}

/**
 * @brief KKT Builder
 *
 * Establish conditions for optimality of general convex (differentiable)
 * problem:
 * \nabla f_0(x^*) + \sum_i^m \lambda_i \nabla f_i(x^*) + \sum_i^p \nu_i \nabla
 * h_i(x^*) = 0
 *
 * (i)   f_i(x^*) <= 0
 * (ii)  h_i(x^*) = 0
 * (iii) \lambda_i^* >= 0
 * (iv)  \lambda_i^* f_i(x^*) = 0
 * (v)   \nabla f_0(x^*) + \sum_i^m \lambda_i^* \nabla f_i(x^*) + \sum_i^p \nu_i
 * \nabla h_i(x^*) = 0
 *
 * KKT System
 * [ H A^T ] [ x; nu ] = [ -q; b ]
 * [ A 0   ]
 */
template <SquareMatrixLike HessianMat, MatrixLike EqualityMat>
  requires(std::same_as<mat_type_t<HessianMat>, mat_type_t<EqualityMat>>)
struct KKTBuilder {
  /// @brief alias for type
  using T = mat_type_t<HessianMat>;

  /// @brief helpers for rows & sizes of vectors / matrices
  static constexpr size_t HRows = mat_rows_v<HessianMat>;
  static constexpr size_t ERows = mat_rows_v<EqualityMat>;
  static constexpr size_t ECols = mat_cols_v<EqualityMat>;
  static constexpr size_t N = HRows + ERows;

  /// @brief alias for mat and schur matrices
  using Mat = linalg::Matrix<T, N, N>;
  using Schur = linalg::Matrix<T, ERows, ERows>;

  static_assert(
      HRows == ECols,
      "EqualityMat number of cols must equal number of rows in hessian");

  /**
   * @brief make kkt mat
   *
   * @param HessianMat H
   * @param EqualityMat A
   *
   * @return KKT mat
   */
  constexpr auto make(const HessianMat& H, const EqualityMat& A) const {
    Mat kkt;
    kkt = set_block<0, HRows, 0, HRows>(kkt, H);
    kkt = set_block<0, ECols, HRows, N>(kkt, transpose(A));
    kkt = set_block<HRows, N, 0, ECols>(kkt, A);
    return kkt;
  }

  /**
   * @brief make schur S = A H^{-1} A^T \in S^{m x m}
   *
   * @param HessianMat H^{-1} \in S^n
   * @param EqualityMat A \in R^{mxn}
   *
   * @return S
   */
  constexpr auto make_schur(
      const Hessian<HessianMat, DifferentialType::PULLBACK>& H_inv,
      const EqualityMat& A) const {
    const auto S = A * (H_inv.apply(transpose(A)));
    return regularize(S);
  }

  /**
   * @brief solve schur
   *
   * Affine-scaling
   * [ H A^T ] [ \Delta_x; \Delta_y ] = [ s ; 0 ]
   * [ A 0   ]
   *
   * Centering
   * [ H A^T ] [ \Delta_x; \Delta_y ] = [ s + mu(x,s) F'(x) ; 0 ]
   * [ A 0   ]
   *
   * @param H_inv Hessian Operator
   * @param A Equality Matrix
   * @param rhs Vector (complementary residual + dual residual)
   *
   * return newton step (delta)
   */
  constexpr auto solve_schur(
      const Hessian<HessianMat, DifferentialType::PULLBACK>& H_inv,
      const EqualityMat& A, const Vector<T, N>& rhs) const {
    const auto split = splice<HRows>(rhs);
    const auto rc = std::get<0>(split);
    const auto rd = std::get<1>(split);

    const auto S = make_schur(H_inv, A);
    const auto ldl = LDL(S);
    const auto qy = ldl.solve(A * H_inv.apply(rc) - rd);
    const auto px = H_inv.apply(rc - transpose(A) * qy);

    Vector<T, N> delta{};
    for (size_t i = 0; i < HRows; ++i) delta(i) = px(i);

    /// (dakre) TODO - determine where py = -qy should happen... it might make
    /// more sense downstream as LDL^T of full KKT != Schur w/ backsolves
    /// currently as the flip doesnt happen naturally in LDLT of full KKT
    for (size_t i = HRows, j = 0; i < N; ++i, ++j) delta(i) = qy(j);
    return delta;
  }

  /**
   * @brief tikhonov regularization full KKT
   *
   * @param KKT mat
   * @param delta applied to H
   * @param delta lower block -- adding damping to Ax + c nu = 0
   *
   * @return updated KKT
   */
  constexpr auto regularize(Mat kkt, const T& delta, const T& c) const {
    for (size_t i = 0; i < N / 2; ++i) kkt(i)(i) += delta;
    for (size_t i = N / 2; i < N; ++i) kkt(i)(i) += c;
    return kkt;
  }

  /**
   * @brief tikhonov regularization
   *
   * TODO - IPM aware (use w-norm instead of max)
   *
   * @param Schur mat
   * @param eps scalar
   *
   * @return regularized schur
   */
  constexpr auto regularize(Schur S, T eps = 1e-10) const {
    auto d = get_diag(S);
    const auto mx = max(d);
    d = (T(1) / mx) * d;
    size_t cnt{0};
    std::for_each(d.begin(), d.end(),
                  [&](const auto& elem) { cnt += constexpr_abs(elem) < eps; });
    if (cnt > 0)
      for (size_t i = 0; i < ERows; ++i) S(i)(i) += cnt * eps;
    return S;
  }

  /**
   * @brief H block
   *
   * @param KKT mat
   *
   * @return H from kkt
   */
  constexpr auto H_block(const Mat& kkt) const {
    return get_block<0, HRows, 0, HRows>(kkt);
  }

  /**
   * @brief A block
   *
   * @param KKT mat
   *
   * @return A from kkt
   */
  constexpr auto A_block(const Mat& kkt) const {
    return get_block<HRows, N, 0, ECols>(kkt);
  }
};

}  // namespace linalg
