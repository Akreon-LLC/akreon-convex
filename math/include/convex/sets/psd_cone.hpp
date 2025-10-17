/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
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
template <linalg::VectorSpace Space, ViewTag Tag = ViewTag::ConvexSet>
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
  constexpr auto contains_impl(const Point& X, const T& /* tol */) const {
    linalg::SymmetricMatrix S(X);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
  }

  /**
   * @brief contains dual impl (S^n_+ is self-dual)
   *
   * { Y | X^T Y >= 0, X \in K}
   *
   * @param point Y
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_dual_impl(const Point& Y, const T& tol) const
    requires(Tag == ViewTag::ConvexCone)
  {
    return contains_impl(Y, tol);
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
  constexpr auto transform_impl(const AffineMap<Mat>& /* map */) const {
    return *this;
  }

  /**
   * @brief cone view impl
   *
   * @return cone
   */
  constexpr auto cone_view_impl() const
    requires(Tag == ViewTag::ConvexSet)
  {
    return ConeView(PsdCone<Space, ViewTag::ConvexCone>());
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
  project_impl(const ProjectiveMap<MatA, MatC>& /*map*/,
               const Point& /*x*/) const {
    return std::nullopt;
  }
};

}  // namespace convex
