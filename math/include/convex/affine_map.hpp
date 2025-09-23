/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "affine_set.hpp"

namespace convex {

/// @brief general affine map thats 2D
template <typename T>
  requires(linalg::FieldLike<T>)
using AffineMap2d = AffineMap<T, 2, 2>;

/// @brief general affine map thats 3D
template <typename T>
  requires(linalg::FieldLike<T>)
using AffineMap3d = AffineMap<T, 3, 3>;

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
   * @brief CTOR
   *
   * @param matrix of type T N x (M + 1) elements
   */
  constexpr explicit AffineMap(const linalg::Matrix<T, N + 1, M + 1>& H)
      : U(get_block<0, N, 0, M>(H)),
        c(as_vector(get_block<0, N, M, M + 1>(H))) {}

  /**
   * @brief compose function
   *
   * Compose a sequence of affine maps as such:
   * f \circ g = f(g(x))
   *
   * g(x) = Ax + b
   * f(g(x)) = A(Ax + b)x + b = A A' x + A b' x + b
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
