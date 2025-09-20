/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "affine_set.hpp"

namespace convex {

/**
 * @brief AffineMap class definition
 *
 * Valid mapping which transforms one affine set to another.
 *
 * f(x) = U x + c
 *
 * Applies linear transform via U and then translation by vector c
 */
template <typename T, size_t N, size_t M>
  requires(linalg::FieldLike<T>)
class AffineMap {
 public:
  /**
   * @brief CTOR
   *
   * @param matrix of type T N x M elements
   * @param vector of type T N elements
   */
  constexpr explicit AffineMap(const linalg::Matrix<T, N, M>& U_in,
                               const linalg::Vector<T, N>& c_in)
      : U(U_in), c(c_in) {}

  /**
   * @brief compose function
   *
   * Compose a sequence of affine maps as such:
   * f \circ g = f(g(x))
   *
   * @param valid reference to an affine map object
   *
   * @return new affine map object with the new applied mapping
   */
  constexpr auto compose(const AffineMap<T, M, N>& other) const {
    return AffineMap(U * other.U, U * other.c + c);
  }

  /**
   * @brief inverse mapping
   *
   * Apply the inverse of the linear transformation matrix U and back out the
   * translation by the inverse.
   *
   * f(x) = U x + c
   * f^{-1}(x) = U^{-1} x - U^{-1} c
   * f^{-1}(f(x)) = U^{-1}(U x + c) - U^{-1} c == x
   *
   * @return new affine map object that is the inverse of the forward mapping
   */
  constexpr auto inverse() const {
    const auto U_inv = linalg::inverse(U);
    return AffineMap(U_inv, -1 * U_inv * c);
  }

  /**
   * @brief linear part
   *
   * @return U
   */
  constexpr auto linear_part() const { return U; }

  /**
   * @brief translation
   *
   * @return c
   */
  constexpr auto translation() const { return c; }

 private:
  /** @brief member variables for affine map */
  linalg::Matrix<T, N, M> U;  //<  valid linear transformation mapping
  linalg::Vector<T, N> c;  //< c bias vector that translates the linear part to
                           // a valid affine subspace
};

}  // namespace convex
