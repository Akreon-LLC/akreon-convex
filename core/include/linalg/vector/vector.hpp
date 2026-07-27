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

#include <array>
#include <ranges>
#include <span>
#include <type_traits>

#include "algebra_util.hpp"

namespace linalg {

/**
 * @brief Vector structure definition
 *
 * Vector is constexpr friendly static compile time container holding N elements
 * of type T
 */
template <typename T, std::size_t N>
struct Vector {
  /**
   * @brief CTOR with initializer list taking in variadic argument list of
   * coordinates
   *
   * @param variadic coordinates to be initialized via initialization list
   */
  template <typename... Coords>
  constexpr Vector(Coords... coords)
    requires(sizeof...(coords) == N)
      : coordinates{coords...} {}

  /**
   * @brief default CTOR
   */
  constexpr Vector() : coordinates{} {}

  /**
   * @brief standard iterable begin function
   *
   * @return iterator to coordinates
   */
  constexpr auto begin() const { return coordinates.begin(); }

  /**
   * @brief standard iterable end function
   *
   * @return iterator to coordinates (end)
   */
  constexpr auto end() const { return coordinates.end(); }

  /**
   * @brief operator overload parenthesis (const / immutable)
   *
   * @param index to acquire coordinates
   *
   * @return coordinates at index
   */
  constexpr const T& operator()(std::size_t idx) const {
    AK_ASSERT(idx < N && "Vector const access index out of bounds");
    return coordinates[idx];
  }

  /**
   * @brief operator overload parenthesis (const / immutable)
   *
   * @param index to acquire coordinates
   *
   * @return coordinates at index
   */
  constexpr T& operator()(std::size_t idx) {
    AK_ASSERT(idx < N && "Vector mutable access index out of bounds");
    return coordinates[idx];
  }

  /**
   * @brief equality operator overload to check if two vectors are equivalent
   *
   * @param another vector to compare with this
   *
   * @return T/F
   */
  constexpr bool operator==(const Vector& other) const {
    return is_equal(other);
  }

  /**
   * @brief equality operator overload to check if two vectors are not
   * equivalent
   *
   * @param another vector to compare with this
   *
   * @return T/F
   */
  constexpr bool operator!=(const Vector& other) const {
    return !is_equal(other);
  }

  /**
   * @brief windowing operation to peak inside the vector coordinates
   *
   * @return non-owning view to coordinates
   */
  constexpr auto view() { return std::span<T, N>{coordinates}; }

  /**
   * @brief size of vector
   *
   * @return N
   */
  constexpr auto size() const { return N; }

private:
  /**
   * @brief check if given vector's coordinates are equivalent to this
   * coordinates
   *
   * @return T/F
   */
  constexpr bool is_equal(const Vector& other) const {
    for (std::size_t i = 0; i < N; ++i)
      if (other.coordinates[i] != coordinates[i]) return false;
    return true;
  }

  /** @brief Vector member variables */
  std::array<T, N> coordinates; ///< coordinates of vector
};

template <typename T, typename... Ts>
Vector(T, Ts...) -> Vector<std::common_type_t<T, Ts...>, 1 + sizeof...(Ts)>;

} // namespace linalg
