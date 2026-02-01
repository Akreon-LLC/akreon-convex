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

#include "operator_traits.hpp"

namespace linalg {

/**
 * @brief Linear operator CRTP base
 */
template <SquareMatrixLike Mat, typename Derived>
struct LinearOperator {
  /// @brief alias helpers for types
  using T = mat_type_t<Mat>;
  using ScalarT = scalar_type_t<T>;
  using ComplexT = std::complex<ScalarT>;

  /// @brief helper to extract N
  static constexpr size_t N = mat_rows_v<Mat>;

  /**
   * @brief self
   *
   * @return object
   */
  constexpr Derived& self() { return static_cast<Derived&>(*this); }

  /**
   * @brief self
   *
   * @return const object
   */
  constexpr const Derived& self() const {
    return static_cast<const Derived&>(*this);
  }

  /**
   * @brief get element at (i,j) index
   *
   * @param row index
   * @param col index
   *
   * @return element
   */
  constexpr const T& operator()(const size_t& i, const size_t& j) const {
    return self()(i, j);
  }

  /**
   * @brief get element at (i,j) index
   *
   * @param row index
   * @param col index
   *
   * @return element (mutable)
   */
  constexpr T& operator()(const size_t& i, const size_t& j) {
    return self()(i, j);
  }
};

}  // namespace linalg
