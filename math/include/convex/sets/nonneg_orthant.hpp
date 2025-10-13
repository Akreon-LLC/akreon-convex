/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "convex_set.hpp"

namespace convex {

/**
 * @brief Nonnegative orthant R^n_+
 */
template <linalg::VectorSpace Space>
  requires(linalg::space_dim_v<Space> ==
           linalg::vec_size_v<linalg::space_point_t<Space>>)
class NonnegOrthant : public ConvexSet<Space, NonnegOrthant<Space>> {
 public:
  /// @brief bring into scope the type traits from the base class
  using ConvexBase = ConvexSet<Space, NonnegOrthant<Space>>;
  using typename ConvexBase::Point;
  using typename ConvexBase::T;

  /// @brief dimension
  using ConvexBase::Dim;

  /// @brief helper to get N elements in point
  static constexpr size_t N = linalg::vec_size_v<Point>;

 protected:
  /// @brief allow for base class to invoke impl funcs
  friend ConvexBase;

  /**
   * @brief contains function to see if point x lies in the R^n_+ set
   *
   * @param point x of type T (n-dim)
   * @param tolerance
   *
   * @return T/F
   */
  constexpr bool contains_impl(const Point& x, const T& /* tol */) const {
    for (size_t i = 0; i < N; ++i)
      if (x(i) < T(0)) return false;
    return true;
  }

  /**
   * @brief is well formed check
   *
   * @return T/F
   */
  constexpr bool is_well_formed_impl() const { return true; }

  /**
   * @brief is equal implementation
   *
   * @return T/F
   */
  constexpr bool is_equal_impl(const ConvexBase& /* other */) const {
    return true;
  }
};

}  // namespace convex
