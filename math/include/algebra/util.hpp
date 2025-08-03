/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

namespace algebra {

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())),
                                  decltype(std::end(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

template <typename T>
constexpr T constexpr_abs(const T& x) {
  return x < 0 ? -1 * x : x;
}

template <typename T>
constexpr T constexpr_sqrt(const T& x, T guess = 1.0) {
  // (dakre) newton step 0.5 * (guess + x / guess)
  return (guess * guess - x > 1e-10 || x - guess * guess > 1e-10)
             ? constexpr_sqrt(x, 0.5 * (guess + x / guess))
             : guess;
}

}  // namespace algebra
