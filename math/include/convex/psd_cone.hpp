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
template <linalg::VectorSpace Space>
  requires(linalg::SquareMatrixLike<linalg::space_point_t<Space>>)
class PsdCone : public ConvexSet<Space, PsdCone<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, PsdCone<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief bring into scope the const data members
  using ConvexBase::Dim;

  /// @brief alias for scalar type of underlying set type
  using ScalarT = linalg::scalar_type_t<T>;

 protected:
  /// @brief allow convex base to invoke impl
  friend ConvexBase;

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
  constexpr auto contains_impl(const Point& X, const ScalarT& /* tol */) const {
    linalg::SymmetricMatrix S(X);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
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
};

}  // namespace convex
