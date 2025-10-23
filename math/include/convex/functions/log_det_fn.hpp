/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "../../linalg/symmetric_matrix.hpp"
#include "concave_fn.hpp"

namespace convex {

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
template <linalg::RnSpace Domain>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
class LogDetFn : public ConcaveFn<Domain, LogDetFn<Domain>> {
 public:
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase = ConcaveFn<Domain, LogDetFn<Domain>>;
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
    linalg::SymmetricMatrix S(X);
    if (S.type != linalg::SymmetricMatrixType::PD) return std::nullopt;
    if (!S.det.has_value()) return std::nullopt;
    if (!linalg::is_real(S.det.value())) return std::nullopt;
    return linalg::constexpr_ln(S.det.value().real());
  }

  /**
   * @brief hypograph impl
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const {
    // (dakre) TODO - impl
    return false;
  }
};

}  // namespace convex
