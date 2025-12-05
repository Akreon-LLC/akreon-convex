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

#include "convex_fn.hpp"
#include "linalg/symmetric_matrix.hpp"

namespace convex {

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

/**
 * @brief log determinant is concave on dom f = S^n_++
 *
 * det(A) = measures volume of matrix (i.e., or linear mapping)
 * log det(A) = log volume (entropy-like) which penalizes small \lambda and less
 * so for well conditioned ones
 *
 * Example 1 usage is MLE of Gaussian: Let X be cov mat of gaussian, so min log
 * det(X) = min vol enclosing ellipsoid
 *
 * Example 2 usage is barrier function in interior point methods for SDP & conic
 * optimization (blows up near boundary)
 */
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Concave>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
class LogDetFn : public ConvexFn<Domain, LogDetFn<Domain, Tag>> {
 public:
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = ConvexFn<Domain, LogDetFn<Domain, Tag>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /**
   * @brief evaluation impl
   *
   * det(A - \lambda I), \lambda are the eigs of A
   * \equiv (\lambda_1 - \lambda)(\lambda_2 - \lambda)... let \lambda = 0 we get
   * det(A) = \lambda_1 * \lambda_2 * ...
   *
   * @param x \in dom f (S^n_++)
   *
   * @return y \in R
   */
  constexpr std::optional<T> eval_impl(const Point& X) const {
    const auto res = LogDet<Domain>()(X);
    if (!res.has_value()) return std::nullopt;

    if constexpr (Tag == FnTag::Convex)
      return T(-1) * res.value();
    else
      return res.value();
  }

  /**
   * @brief negate impl
   *
   * @return negated f
   */
  constexpr auto negate_impl() const {
    if constexpr (Tag == FnTag::Concave)
      return LogDetFn<Domain, FnTag::Convex>();
    else
      return LogDetFn<Domain, FnTag::Concave>();
  }

  /**
   * @brief canonicalize_impl
   *
   * abs(x) -> epi or hypo -> homogenized cone
   *
   * @return cone block
   */
  constexpr auto canonicalize_impl() const {
    if constexpr (Tag == FnTag::Convex) {
    }
  }
};

}  // namespace convex
