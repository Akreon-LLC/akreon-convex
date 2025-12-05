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

#include "convex/cone_traits.hpp"
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

/// @brief cone base class
template <linalg::MatrixLike Mat>
struct PositiveSemidefiniteCone
    : public Cone<Mat, PositiveSemidefiniteCone<Mat>> {
  /// @brief bring into scope alias from cone base
  using ConeBase = Cone<Mat, PositiveSemidefiniteCone<Mat>>;
  using typename ConeBase::Bias;

  /**
   * @brief compute barrier
   *
   * @return barrier for cone
   */
  constexpr auto barrier(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }

  /**
   * @brief compute gradient
   *
   * @return gradient for cone
   */
  constexpr auto gradient_impl(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }

  /**
   * @brief compute hessian
   *
   * @return hessian for cone
   */
  constexpr auto hessian_impl(const Mat& /*F*/, const Bias& /*g*/) const {
    return false;
  }
};

}  // namespace conic
