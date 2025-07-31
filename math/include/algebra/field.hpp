/*
 * Copyright (c) 2025 Akreon.
 * All rights reserved.
 */

#pragma once

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "../topology/topology.hpp"

namespace algebra {

template <typename T>
struct scalar_type {
  using value_type = T;
};

template <typename T>
struct scalar_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T, int Rows, int Cols>
struct scalar_type<Eigen::Matrix<T, Rows, Cols>> {
  using value_type = typename scalar_type<T>::value_type;
};

template <typename T>
struct scalar_type<std::vector<T>> {
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
concept FieldLike = FloatingPoint<T> || requires(T a, T b) {
  { a + b } -> std::same_as<T>;
  { a * b } -> std::same_as<T>;
  { -a } -> std::same_as<T>;
  { a / b } -> std::floating_point;
  { a + T{0} } -> std::same_as<T>;
  { a * T{1} } -> std::same_as<T>;
};

}  // namespace algebra
