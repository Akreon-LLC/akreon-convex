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
 * @brief KKT Builder (Augmented)
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
 *
 * H = [ 0 0   ] \in R^{m+n x m+n}
 *     [ 0 H_x ]
 * A = [ -F I ] \in R^{m x m + n}
 * A^T = [ -F ] \in R^{m + n x m}
 *       [ I  ]
 *
 * KKT \in R^{ 2m + n x 2m + x }
 */
template <SquareMatrixLike HessianMat, MatrixLike EqualityMat>
  requires(std::same_as<mat_type_t<HessianMat>, mat_type_t<EqualityMat>>)
struct AugmentedKKT {
  /// @brief alias for type
  using T = mat_type_t<HessianMat>;

  /// @brief helpers for rows & sizes of vectors / matrices
  static constexpr size_t M = mat_rows_v<EqualityMat>;
  static constexpr size_t N = mat_cols_v<EqualityMat>;
  static constexpr size_t KKT_ROWS = 2 * M + N;

  /// @brief alias for mat and schur matrices
  using KKT = linalg::Matrix<T, KKT_ROWS, KKT_ROWS>;
  using Schur = linalg::Matrix<T, N, N>;

  static_assert(mat_rows_v<HessianMat> == M, "A rows != H rows");

  /**
   * @brief make kkt mat
   *
   * @param HessianMat H
   * @param EqualityMat A
   *
   * @return KKT mat
   */
  constexpr auto make(const HessianMat& H, const EqualityMat& F) const {
    KKT kkt;
    const auto I = make_identity<T, M>();
    kkt = set_block<N, M + N, N, M + N>(kkt, H);
    kkt = set_block<0, N, M + N, KKT_ROWS>(kkt, T(-1) * transpose(F));
    kkt = set_block<N, M + N, M + N, KKT_ROWS>(kkt, I);
    kkt = set_block<M + N, KKT_ROWS, 0, N>(kkt, T(-1) * F);
    kkt = set_block<M + N, KKT_ROWS, N, M + N>(kkt, I);
    return regularize(kkt);
  }

  /**
   * @brief make augmented schur
   *
   * \hat{S} = F^T H_x F
   *
   * @param H_x (hessian for cone var in lifted dom-- [ z; x ] \in R^n x E
   * @param F (cone block slice for affine equality)
   *
   * @return \hat{S}
   */
  constexpr auto make_schur(
      const Hessian<HessianMat, DifferentialType::PUSHFORWARD>& H_x,
      const EqualityMat& F) const {
    const auto S = transpose(F) * (H_x.apply(F));
    return regularize(S);
  }

  /**
   * @brief solve augmented schur
   *
   * \tilde{KKT} = [ 0  0   -F^T ] [ pz; px; qy ] = [ rp; rc; rd ]
   *               [ 0  H_x I    ]
   *               [ -F I   0    ]
   * \tilde{S} = F^T H_x F... augmented schur block
   *
   * @param H_x Hessian Operator
   * @param F cone block slice
   * @param rhs Vector (complementary residual + dual residual)
   *
   * Newton step
   * \Delta = [ pz px qy ]... TODO flip qy = -py
   *
   * @return \Delta
   */
  constexpr auto solve_schur(
      const Hessian<HessianMat, DifferentialType::PUSHFORWARD>& H_x,
      const EqualityMat& F, const Vector<T, N + 2 * M>& rhs) const {
    const auto split_one = splice<N>(rhs);
    const auto rp = std::get<0>(split_one);
    const auto split_two = splice<M>(std::get<1>(split_one));
    const auto rc = std::get<0>(split_two);
    const auto rd = std::get<1>(split_two);

    const auto S = make_schur(H_x, F);
    const auto ldl = LDL(S);
    const auto FT = transpose(F);
    const auto pz = ldl.solve(FT * rc - FT * (H_x.apply(rd)) + rp);
    const auto qy = rc - H_x.apply(F * pz + rd);
    const auto px = F * pz + rd;

    Vector<T, N + 2 * M> delta{};
    for (size_t i = 0; i < N; ++i) delta(i) = pz(i);
    for (size_t i = 0, j = N; i < M; ++i, ++j) delta(j) = px(i);
    for (size_t i = 0, j = N + M; i < M; ++i, ++j) delta(j) = qy(i);
    return delta;
  }

  /**
   * @brief H block
   *
   * @param KKT mat
   *
   * @return H from kkt
   */
  constexpr auto H_block(const KKT& kkt) const {
    return get_block<N, M + N, N, M + N>(kkt);
  }

  /**
   * @brief A block
   *
   * @param KKT mat
   *
   * @return A from kkt
   */
  constexpr auto A_block(const KKT& kkt) const {
    return get_block<M + N, KKT_ROWS, 0, M + N>(kkt);
  }

 private:
  /**
   * @brief tikhonov regularization
   *
   * TODO - IPM aware (use w-norm instead of max)
   *
   * @param Schur or KKT mat
   *
   * @return regularized
   */
  template <MatrixLike MatIn>
  constexpr auto regularize(MatIn S) const {
    auto d = get_diag(S);
    const auto mx = max(d);
    d = (T(1) / mx) * d;
    size_t cnt{0};
    std::for_each(d.begin(), d.end(), [&](const auto& elem) {
      cnt += constexpr_abs(elem) < SolverTraits<T>::tol;
    });
    static constexpr size_t MatRows = mat_rows_v<MatIn>;
    if (cnt > 0)
      for (size_t i = 0; i < MatRows; ++i) {
        if (i >= KKT_ROWS)
          S(i)(i) -= cnt * SolverTraits<T>::tol;
        else
          S(i)(i) += cnt * SolverTraits<T>::tol;
      }
    return S;
  }
};

}  // namespace linalg
