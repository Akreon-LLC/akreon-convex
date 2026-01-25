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

#include "cone.hpp"
#include "linalg/symmetric_matrix.hpp"

namespace conic {

/// @brief Log Det Functor
template <linalg::RnSpace Domain>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
struct LogDet {
  using T = linalg::space_type_t<Domain>;
  using Point = linalg::space_point_t<Domain>;

  /**
   * @brief compute the log det of a symmetric matrix
   *
   * @param X \in S^n_++
   *
   * @return optional scalar \in R
   */
  constexpr std::optional<T> operator()(const Point& X) const {
    linalg::SymmetricMatrix S(X);
    if (S.type != linalg::SymmetricMatrixType::PD) return std::nullopt;
    if (!S.det.has_value()) return std::nullopt;
    if (!linalg::is_real(S.det.value())) return std::nullopt;
    return linalg::constexpr_ln(S.det.value().real());
  }
};

/** @brief psd cone class
 *
 * X \in S^n_+, tr(X) >= 0
 */
template <linalg::VectorLike Slack, size_t N>
struct PositiveSemidefiniteCone
    : public Cone<Slack, PositiveSemidefiniteCone<Slack, N>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Slack, PositiveSemidefiniteCone<Slack, N>>;

  /// @brief helpers to get T and M
  using ConeBase::M;
  using typename ConeBase::T;

  static_assert(M == N * (N + 1) / 2,
                "Encoded dim for PSD Cone does not align with slack dim");

  /**
   * @brief default interior point
   *
   * @return vector
   */
  constexpr Slack x0_impl() const {
    const auto I = linalg::make_identity<T, N>();
    return linalg::lower_triangle_to_vec(I);
  }

  /**
   * @brief in interior of K
   *
   * @param slack vector
   *
   * @return bool
   */
  constexpr bool in_K_impl(const Slack& x) const {
    const auto X = linalg::make_smat<N, Slack>(x);
    const auto SymMat = linalg::SymmetricMatrix(X);
    return SymMat.is_psd();
  }

  /**
   * @brief fraction to boundary dK
   *
   * (dakre) TODO - review further
   *
   * sigma_x = \frac{1}{sup{alpha | X - alpha P in K}}
   * let M = L^{-1} P L^{-T} ... cholesky factorization
   * X - alpha P \equiv L (I - alpha M) L^{T}
   * I - alpha M >= 0... PSD \equiv alpha <= 1 / eig_max(M)
   *
   * alpha_max = 1 / eig_max(M) if eig_max(M) > 0 else inf
   *
   * However this is another decomposition cost so we can step
   * by unit length and then backtrack with the IPM algo
   *
   * @param x \in K
   * @param p \in E (delta dir)
   *
   * @return \alpha_max
   */
  constexpr std::optional<T> alpha_max_impl(
      const Slack& /*x*/, const linalg::Vector<T, M>& /*p*/) const {
    return T(1);
  }
};

}  // namespace conic
