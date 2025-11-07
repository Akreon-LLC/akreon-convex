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

#include "concave_fn.hpp"
#include "convex/sets/log_det_cone.hpp"
#include "linalg/symmetric_matrix.hpp"

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
template <linalg::RnSpace Domain, FnTag Tag = FnTag::Concave>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Domain>>)
class LogDetFn
    : public std::conditional_t<Tag == FnTag::Concave,
                                ConcaveFn<Domain, LogDetFn<Domain, Tag>>,
                                ConvexFn<Domain, LogDetFn<Domain, Tag>>> {
 public:
  /// @brief bring into scope the concave fn alias / helpers
  using ConvexBase =
      std::conditional_t<Tag == FnTag::Concave,
                         ConcaveFn<Domain, LogDetFn<Domain, Tag>>,
                         ConvexFn<Domain, LogDetFn<Domain, Tag>>>;
  using ConvexBase::Dim;
  using ConvexBase::N;
  using typename ConvexBase::CoordMat;
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
   * @brief hypograph impl
   *
   * F(X) = log det X
   * graph(F(X)) = { (x, log(det(X))) | x \in dom f }
   * hypo(F(X)) = { (x, t) | log(det(X)) >= t, x \in dom f }
   * \equiv { (x, t) | \prod_i^n \lambda_i >= exp(t) }
   *
   * @return hypograph
   */
  constexpr auto hypograph_impl() const
    requires(Tag == FnTag::Concave)
  {
    return LogDetCone<Domain, ViewTag::LiftedCone>().cone_view();
  }

  /**
   * @brief epigraph impl
   *
   * C = { (x, t) | \prod_i^n \lambda_i >= -exp(t) }
   *
   * @return epigraph
   */
  constexpr auto epigraph_impl() const
    requires(Tag == FnTag::Convex)
  {
    return LogDetCone<Domain, ViewTag::LiftedCone>(
               default_log_det_map<CoordMat>(T(-1)))
        .cone_view();
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
};

}  // namespace convex
