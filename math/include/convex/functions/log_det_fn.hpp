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
    if (!linalg::is_real(S.det)) return std::nullopt;
    return S.det.value().real();
  }

  /**
   * @brief epigraph impl
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const {
    // (dakre) TODO - impl
    return false;
  }
};

}  // namespace convex
