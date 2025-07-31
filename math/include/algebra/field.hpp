/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <type_traits>

// Forward declarations
namespace topology {

template <typename T, size_t N>
struct Topology;

template <typename T, size_t N>
struct OpenSet;

}  // namespace topology

namespace algebra {

template <typename T>
struct scalar_type {
  using value_type = T;
};

template <typename T, size_t N>
struct scalar_type<std::array<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T>
struct scalar_type<std::complex<T>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T, size_t N>
struct scalar_type<topology::OpenSet<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T, size_t N>
struct scalar_type<topology::Topology<T, N>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T>
using scalar_type_t = typename scalar_type<T>::value_type;

template <typename T>
concept FloatingPoint = std::floating_point<scalar_type_t<T>>;

template <typename T>
concept GroupLike = requires(T a, T b, T c) {
  { a + b } -> std::same_as<T>;
  { (a + b) + c == a + (a + c) };
  { -a } -> std::same_as<T>;
  { a + b == b + a };
};

template <typename T>
concept RingLike = GroupLike<T> && requires(T a, T b, T c) {
  { a * b } -> std::same_as<T>;
  { (a * b) * c == a * (b * c) };
  { a * (b + c) == a * b + a * c };
  { a * b == b * a };
};

template <typename T, size_t N>
constexpr bool is_ones_like(const topology::OpenSet<T, N>& set) {
  return std::all_of(std::begin(set), std::end(set),
                     [](const auto& x) { return x == 1; });
}

template <typename T>
concept FieldLike = FloatingPoint<T> || (RingLike<T> && requires(T a, T b) {
                      { is_ones_like(a / a) };
                      { a + T{0} } -> std::same_as<T>;
                      { a * T{1} } -> std::same_as<T>;
                    });

}  // namespace algebra
