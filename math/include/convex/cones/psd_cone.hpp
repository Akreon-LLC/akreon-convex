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

/// @brief psd cone class
template <linalg::VectorLike Vec>
struct PositiveSemidefiniteCone
    : public Cone<Vec, PositiveSemidefiniteCone<Vec>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Vec, PositiveSemidefiniteCone<Vec>>;

  /**
   * @brief compute barrier
   *
   * @return barrier for cone
   */
  constexpr auto barrier_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute dual barrier
   *
   * @return dual barrier for cone
   */
  constexpr auto dual_barrier_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute gradient
   *
   * @return gradient for cone
   */
  constexpr auto gradient_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute gradient
   *
   * @return gradient for dual cone
   */
  constexpr auto dual_gradient_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute hessian
   *
   * @return hessian for cone
   */
  constexpr auto hessian_impl(const Vec& /*s*/) const { return false; }

  /**
   * @brief compute hessian
   *
   * @return hessian for dual cone
   */
  constexpr auto dual_hessian_impl(const Vec& /*s*/) const { return false; }
};

}  // namespace conic
