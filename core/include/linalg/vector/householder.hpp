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

#include "config/assert.hpp"
#include "linalg/linalg_ops.hpp"
#include "linalg/linalg_util.hpp"
#include "linalg/norms/vector_norms.hpp"

namespace linalg {
/**
 * @brief Compact representation of a real or complex Householder reflector.
 *
 * The stored `v` and real coefficient `beta` represent
 * `P = I - beta * v * v^T` over a real field and
 * `P = I - beta * v * v^*` over a complex field. Construction follows the
 * Golub and Van Loan algorithm and selects a numerically stable leading
 * coordinate. Applying the reflector maps the input toward a multiple of the
 * first coordinate vector.
 *
 * For `N == 1`, the representation is the identity reflector (`v = [1]`,
 * `beta = 0`). A zero complex input similarly requires no reflector.
 *
 * @tparam T Real or complex field scalar.
 * @tparam N Positive compile-time vector dimension.
 */
template <typename T, std::size_t N>
  requires(Field<T> && N > 0)
struct HouseholderVector {
  /**
   * @brief Compute the compact reflector representation for `x`.
   * @param x Vector to reduce toward the first coordinate axis.
   */
  constexpr HouseholderVector(const Vector<T, N>& x) : beta(real_type_t<T>(0)) {
    make_householder(x, std::integral_constant<std::size_t, N>{});
  }

  Vector<T, N> v;      ///< Reflector direction vector.
  real_type_t<T> beta; ///< Real coefficient in `I - beta * v * v^*`.

private:
  /** @brief Compute the general `N > 1` reflector representation. */
  constexpr void make_householder(const Vector<T, N>& x, auto) {
    if constexpr (Complex<T>) {
      const T norm_x = L2<Vector<T, N>>()(x);
      if (norm_x == real_type_t<T>(0)) {
        // (dakre) no reflector required
        return;
      }

      T alpha = T(1);
      if (x(0) != T(0)) alpha = x(0) / L2<Vector<T, 1>>()(Vector<T, 1>{x(0)});

      v = x;
      v(0) += alpha * norm_x;
      // (dakre) normalize v like reals otherwise hosueholder mat wont
      // orthogonalize correctly!!
      v = (real_type_t<T>(1) / L2<Vector<T, 1>>()(Vector<T, 1>{v(0)})) * v;
      const auto vv = inner_product(v, v);
      AK_ASSERT(is_real(vv) &&
                "Householder complex branch expects real inner_product(v, v)");
      beta = real_type_t<T>(2) / vv.real();
    } else {
      const auto x_shifted = get_elements<1, N>(x);
      real_type_t<T> sigma;
      sigma = inner_product(x_shifted, x_shifted);

      v = x;
      v(0) = 1;
      if (sigma == 0 && x(0) >= T(0)) beta = T(0);
      else if (sigma == 0 && x(0) < T(0)) beta = T(-2);
      else {
        const auto mu = constexpr_sqrt_safe(x(0) * x(0) + sigma);
        if (x(0) <= T(0)) v(0) = x(0) - mu;
        else v(0) = -sigma / (x(0) + mu);
        beta = 2 * (v(0) * v(0)) / (sigma + v(0) * v(0));
        v = (T(1) / v(0)) * v;
      }
    }
  }

  /** @brief Construct the identity-reflector representation for `N == 1`. */
  constexpr void make_householder(const Vector<T, N>&,
                                  std::integral_constant<std::size_t, 1>) {
    beta = real_type_t<T>(0);
    v(0) = T(1);
  }
};

} // namespace linalg
