/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include "matrix.hpp"
#include "vector_ops.hpp"

namespace linalg {
/**
 * @brief Householder Vector
 *
 * Householder vector constructed based on algorithm implementation in Matrix
 * Algorithms by Golub & Van Loan
 */
template <typename T, size_t N>
  requires(FieldLike<T> && N > 0)
struct HouseholderVector {
  /**
   * @brief CTOR to compute householder vector
   *
   * Householder reflector: P = I - beta v v^T is \perp and Px = ||x||_2 e_1
   * beta = 2 / v^T v
   *
   * @param vector x to be transformed into householder
   */
  constexpr HouseholderVector(const Vector<T, N>& x)
      : beta(scalar_type_t<T>(0)) {
    make_householder(x, std::integral_constant<size_t, N>{});
  }

  /**
   * @brief simple getter for beta scalar
   *
   * @return beta scalar
   */
  constexpr auto get_beta() const { return beta; }

  /**
   * @brief simple getter for v vector
   *
   * @return v vector of type T with N elements
   */
  constexpr auto get_v() const { return v; }

 private:
  /**
   * @brief from a vector X create a householder vector
   *
   * The whole idea of householder vectors is to generate orthogonal vectors
   * such that Q^T Q = I or Q^* Q = I \in C from the householder reflectors
   * described above
   *
   * @param vector x with type T and N elements
   *
   * @return (void)
   */
  constexpr void make_householder(const Vector<T, N>& x, auto) {
    if constexpr (is_complex_v<T>) {
      const T norm_x = L2<Vector<T, N>>()(x);
      if (norm_x == scalar_type_t<T>(0)) {
        // (dakre) no reflector required
        return;
      }

      T alpha = T(1);
      if (x(0) != T(0)) alpha = x(0) / L2<Vector<T, 1>>()(Vector<T, 1>{x(0)});

      v = x;
      v[0] += alpha * norm_x;
      // (dakre) normalize v like reals otherwise hosueholder mat wont
      // orthogonalize correctly!!
      v = (scalar_type_t<T>(1) / L2<Vector<T, 1>>()(Vector<T, 1>{v(0)})) * v;
      const auto vv = dot(conjugate(v), v);
      assert(is_real(vv));
      beta = scalar_type_t<T>(2) / vv.real();
    } else {
      const auto x_shifted = get_elements<1, N>(x);
      scalar_type_t<T> sigma;
      sigma = dot(x_shifted, x_shifted);

      v = x;
      v[0] = 1;
      if (sigma == 0 && x(0) >= T(0))
        beta = T(0);
      else if (sigma == 0 && x(0) < T(0))
        beta = T(-2);
      else {
        const auto mu = constexpr_sqrt(x(0) * x(0) + sigma);
        if (x(0) <= T(0))
          v[0] = x(0) - mu;
        else
          v[0] = -sigma / (x(0) + mu);
        beta = 2 * (v(0) * v(0)) / (sigma + v(0) * v(0));
        v = (T(1) / v(0)) * v;
      }
    }
  }

  /**
   * @brief handles edge case where N = 1 return degenerate householder vector
   * beta = 0 & v = [1 ... 0]
   *
   * @param vector type T with N elements (null)
   * @param integral constant of 1 of type size_t
   *
   * @return (void)
   */
  constexpr void make_householder(const Vector<T, N>&,
                                  std::integral_constant<std::size_t, 1>) {
    beta = scalar_type_t<T>(0);
    v[0] = T(1);
  }

  /** @brief Vector member variables */
  Vector<T, N> v;         ///< householder vector
  scalar_type_t<T> beta;  ///< scalar to form P
};

}  // namespace linalg
