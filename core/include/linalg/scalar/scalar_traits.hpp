/* Copyright © 2025 Akreon LLC. All rights reserved.
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

#include <complex>
#include <type_traits>

namespace linalg {

template <typename T>
using remove_cvref_t = std::remove_cvref_t<T>;

template <typename T>
struct scalar_traits;

template <std::floating_point T>
struct scalar_traits<T> {
  using scalar_type = T;
  using real_type = T;

  static constexpr bool is_real = true;
  static constexpr bool is_complex = false;
};

template <std::floating_point T>
struct scalar_traits<std::complex<T>> {
  using scalar_type = std::complex<T>;
  using real_type = T;

  static constexpr bool is_real = false;
  static constexpr bool is_complex = true;
};

template <typename T>
using scalar_traits_t = scalar_traits<remove_cvref_t<T>>;

template <typename T>
using scalar_type_t = typename scalar_traits_t<T>::scalar_type;

template <typename T>
using real_type_t = typename scalar_traits_t<T>::real_type;

template <typename T>
inline constexpr bool is_real_v = scalar_traits_t<T>::is_real;

template <typename T>
inline constexpr bool is_complex_v = scalar_traits_t<T>::is_complex;

} // namespace linalg
