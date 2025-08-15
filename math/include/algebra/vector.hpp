/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <assert.h>

#include <ranges>
#include <span>

#include "field.hpp"
#include "linear_algebra_util.hpp"

namespace algebra {

/**
 * @brief Vector structure definition
 *
 * Vector is constexpr friendly static compile time container holding N elements
 * of type T
 */
template <typename T, size_t N>
struct Vector {
  ///< value_type = T
  using value_type = T;

  ///@brief type T must satisfy axioms of a proper field
  static_assert(FieldLike<T>);

  ///@brief s_size holds the size of the elements N within the scope of the
  /// vector
  static constexpr size_t s_size = N;

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
  constexpr const T& operator()(const size_t& idx) const {
    assert(idx < N);
    return coordinates[idx];
  }

  /**
   * @brief operator overload brackets (mutable)
   *
   * @param index to acquire coordinates
   *
   * @return coordinates at index
   */
  constexpr T& operator[](const size_t& idx) {
    assert(idx < N);
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
    for (size_t i = 0; i < N; ++i)
      if (other.coordinates[i] != coordinates[i]) return false;
    return true;
  }

  /** @brief Vector member variables */
  std::array<T, N> coordinates;  ///< coordinates of vector
};

/**
 * @brief get n elements based on start to end indices
 *
 * @param vector of type T with N elements
 *
 * @return vector of type T with End - Start elements
 */
template <size_t Start, size_t End, typename T, size_t N>
  requires(FieldLike<T> && End - Start < N && Start < N)
constexpr auto get_elements(const Vector<T, N>& x) {
  Vector<T, End - Start> result;
  for (size_t i = Start, idx = 0; i < End; ++i, ++idx) result[idx] = x(i);
  return result;
}

/**
 * @brief Householder Vector
 *
 * Householder vector constructed based on algorithm implementation in Matrix
 * Algorithms by Golub & Van Loan
 */
template <typename T, size_t N>
  requires(FieldLike<T> && !is_iterable_v<T>)
struct HouseholderVector {
  /**
   * @brief CTOR to compute householder vector
   *
   * P = I - beta v v^T is \perp and Px = ||x||_2 e_1
   *
   * @param vector x to be transformed into householder
   */
  constexpr HouseholderVector(const Vector<T, N>& x) {
    const auto x_shifted = get_elements<1, N>(x);
    scalar_type_t<T> sigma;
    if constexpr (!is_complex_v<T>)
      sigma = dot(x_shifted, x_shifted);
    else {
      const auto res = (conjugate_transpose(x_shifted) * x_shifted);
      assert(is_real(res));
      sigma = res.real();
    }

    v = x;
    v[0] = 1;
    if (sigma == 0 && x(0) >= 0)
      beta = 0;
    else if (sigma == 0 && x(0) < 0)
      beta = -2;
    else {
      const auto mu = constexpr_sqrt(x(0) * x(0) + sigma);
      if (x(0) <= 0)
        v[0] = x(0) - mu;
      else
        v[0] = -sigma / (x(0) + mu);
      beta = 2 * (v(0) * v(0)) / (sigma + v(0) * v(0));
      v = (1 / v(0)) * v;
    }
  }

  constexpr auto get_beta() const { return beta; }

  constexpr auto get_v() const { return v; }

 private:
  /** @brief Vector member variables */
  Vector<T, N> v;         ///< householder vector
  scalar_type_t<T> beta;  ///< scalar to form P
};

}  // namespace algebra
