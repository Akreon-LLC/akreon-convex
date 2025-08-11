/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <assert.h>

#include <ranges>

#include "field.hpp"
#include "linear_algebra_util.hpp"

namespace algebra {

template <typename T, size_t N>
struct Vector {
  using value_type = T;

  static_assert(FieldLike<T>);

  static constexpr size_t s_size = N;

  template <typename... Coords>
  constexpr Vector(Coords... coords)
    requires(sizeof...(coords) == N)
      : coordinates{coords...} {}

  constexpr Vector() : coordinates{} {}

  constexpr auto begin() const { return coordinates.begin(); }

  constexpr auto end() const { return coordinates.end(); }

  constexpr const T& operator()(const size_t& idx) const {
    assert(idx < N);
    return coordinates[idx];
  }

  constexpr T& operator[](const size_t& idx) {
    assert(idx < N);
    return coordinates[idx];
  }

  constexpr bool operator==(const Vector& other) const {
    return is_equal(other);
  }

  constexpr bool operator!=(const Vector& other) const {
    return !is_equal(other);
  }

  constexpr auto view() { return std::span<T, N>{coordinates}; }

  constexpr auto size() const { return N; }

 private:
  constexpr bool is_equal(const Vector& other) const {
    for (size_t i = 0; i < s_size; ++i)
      if (other.coordinates[i] != coordinates[i]) return false;
    return true;
  }

  std::array<T, N> coordinates;
};

}  // namespace algebra
