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
  requires(!linalg::is_complex_v<linalg::mat_type_t<Space>>)
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

  /// @brief helper to get n elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

  /**
   * @brief is_valid implementation stubbed to true for this set type
   *
   * @return T/F
   */
  constexpr bool is_valid_impl() const { return true; }

  /**
   * @brief check if candidate matrix X \in S^n_++
   *
   * @param matrix X of type T N x N
   *
   * @return T/F
   */
  constexpr auto contains_impl(const Point& X, const ScalarT& tol) const {
    (void)tol;
    linalg::SymmetricMatrix S(X);
    return S.type == linalg::SymmetricMatrixType::PD ||
           S.type == linalg::SymmetricMatrixType::PSD;
  }

  /**
   * @brief to conic form helper
   *
   * @return conic form
   */
  constexpr auto to_conic_form_impl() const {
    // (dakre) TODO implement
    return true;
  }
};

}  // namespace convex
