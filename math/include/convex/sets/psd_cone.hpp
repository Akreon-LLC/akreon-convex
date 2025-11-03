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

#include "convex_set.hpp"

namespace convex {

/**
 * @brief PSD cone definition valid on vector space
 *
 * S^n = { X | X = X^T }... set of symmetric matrices (hermitian for complex
 * case) S^n_++ { X \in S^n | X > 0 }... set of positive definite matrices S^n_+
 * { X \in S^n | X >= 0 }... set of psd matrices
 */
template <linalg::RnSpace Space, ViewTag Tag = ViewTag::ConvexSet>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>>)
class PsdCone : public ConvexSet<Space, PsdCone<Space, Tag>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, PsdCone<Space, Tag>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /**
   * @brief is_well_formed implementation stubbed to true for this set type
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief check if candidate matrix X \in S^n_++
   *
   * @param point in Vector space which is S^n
   * @param scalar tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& X, const T& /* tol */) const {
    linalg::SymmetricMatrix S(X);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
  }

  /**
   * @brief check if candidate matrix X \in S^n_++
   *
   * @param (x, t) \in R^n x R
   * @param scalar tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const std::tuple<Point, T>& x_in,
                               const T& tol) const {
    const auto X = std::get<0>(x_in);
    return contains_impl(X, tol);
  }

  /**
   * @brief equality comparison
   *
   * @param convex set
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }

  /**
   * @brief transform impl
   *
   * @return self
   */
  template <linalg::MatrixLike Mat>
    requires(Tag == ViewTag::ConvexSet)
  constexpr auto transform_impl(const affine::AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const {
    if constexpr (Tag == ViewTag::ConvexSet)
      return ConeView(PsdCone<Space, ViewTag::LiftedCone>());
    else
      return ConeView(PsdCone<Space, Tag>());
  }

  /**
   * @brief project impl
   *
   * @param project map
   * @param point in set
   *
   * @return projected set
   */
  template <linalg::MatrixLike MatA, linalg::MatrixLike MatC>
  constexpr std::optional<linalg::Vector<T, linalg::mat_cols_v<MatA> + 1>>
  projective_map_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
                      const Point& /*x*/) const {
    return std::nullopt;
  }
};

}  // namespace convex
